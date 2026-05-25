from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _compute_grid(patches_per_image: int):
    side = int(np.sqrt(max(1, patches_per_image)))
    if side * side == patches_per_image:
        return side, side
    grid_h = 16
    grid_w = int(np.ceil(max(1, patches_per_image) / grid_h))
    return grid_h, grid_w


def _tensor_or_array_to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _overlay_pruned_patches(frame_rgb, local_pruned, patches_per_image, alpha):
    from PIL import Image

    out = frame_rgb.copy()
    h, w = out.shape[0], out.shape[1]
    grid_h, grid_w = _compute_grid(patches_per_image)
    strength = float(np.clip(alpha / 255.0, 0.0, 0.90))
    dark_value = 0.03

    for local_idx in local_pruned:
        row = int(local_idx) // grid_w
        col = int(local_idx) % grid_w
        if row < 0 or row >= grid_h:
            continue
        x0 = int(round(col * w / grid_w))
        y0 = int(round(row * h / grid_h))
        x1 = int(round((col + 1) * w / grid_w))
        y1 = int(round((row + 1) * h / grid_h))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = out[y0:y1, x0:x1, :]
        patch_f = patch.astype(np.float32) / 255.0
        dark = np.ones_like(patch_f) * dark_value
        masked = patch_f * (1.0 - strength) + dark * strength
        out[y0:y1, x0:x1, :] = np.clip(masked * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")


def _to_rgb_uint8(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def overlay_padi_scores_on_frame(frame, padi_out: Optional[Any], step_idx, position: str = "top_left", panel_scale: float = 0.82):
    from PIL import Image, ImageDraw, ImageFont

    rgb = _to_rgb_uint8(frame)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", max(12, int(15 * panel_scale)))
        font_text = ImageFont.truetype("DejaVuSans.ttf", max(10, int(12 * panel_scale)))
    except Exception:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    if padi_out is None:
        geometry_risk = 0.0
        precise_active = False
        transit_score = 0.0
        startup_guard_active = False
        local_interaction_candidate = False
        phase = "NO SIGNAL"
    else:
        geometry_risk = float(getattr(padi_out, "geometry_risk", 0.0))
        precise_active = bool(getattr(padi_out, "precise_active", False))
        transit_score = float(getattr(padi_out, "transit_score", 0.0))
        dbg = getattr(padi_out, "debug", {}) or {}
        startup_guard_active = bool(dbg.get("startup_guard_active", False))
        local_interaction_candidate = bool(dbg.get("local_interaction_candidate", False))

        if startup_guard_active and not precise_active:
            phase = "STARTUP"
        elif precise_active:
            phase = "PRECISE"
        elif transit_score >= 0.6 and geometry_risk < 0.3:
            phase = "TRANSIT"
        elif local_interaction_candidate:
            phase = "LOCAL"
        else:
            phase = "IDLE"

    geometry_risk = max(0.0, min(1.0, geometry_risk))
    transit_score = max(0.0, min(1.0, transit_score))

    w, h = img.size
    panel_w = int(210 * panel_scale)
    panel_h = int(120 * panel_scale)
    margin = int(8 * panel_scale)

    if position == "top_right":
        x0, y0 = w - panel_w - margin, margin
    elif position == "bottom_left":
        x0, y0 = margin, h - panel_h - margin
    elif position == "bottom_right":
        x0, y0 = w - panel_w - margin, h - panel_h - margin
    else:
        x0, y0 = margin, margin
    x1, y1 = x0 + panel_w, y0 + panel_h

    draw.rounded_rectangle([x0, y0, x1, y1], radius=int(8 * panel_scale), fill=(0, 0, 0, 155), outline=(230, 230, 230, 220), width=1)

    tx = x0 + int(10 * panel_scale)
    ty = y0 + int(8 * panel_scale)
    draw.text((tx, ty), f"PADI | {phase}", fill=(255, 225, 150, 255), font=font_title)

    ty += int(24 * panel_scale)
    g_label = "G*" if startup_guard_active else "G"
    draw.text((tx, ty), f"{g_label} {geometry_risk:.3f}", fill=(255, 140, 140, 255), font=font_text)

    bar_x0 = tx + int(62 * panel_scale)
    bar_x1 = x1 - int(10 * panel_scale)
    bar_h = int(8 * panel_scale)
    bar_y0 = ty + int(3 * panel_scale)
    bar_y1 = bar_y0 + bar_h
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(45, 45, 45, 220), outline=(120, 120, 120, 220))
    fill_x = int(bar_x0 + (bar_x1 - bar_x0) * geometry_risk)
    draw.rectangle([bar_x0, bar_y0, fill_x, bar_y1], fill=(230, 80, 80, 240))

    ty += int(20 * panel_scale)
    draw.text((tx, ty), f"P {'ON' if precise_active else 'OFF'}", fill=(140, 220, 255, 255), font=font_text)

    ty += int(20 * panel_scale)
    draw.text((tx, ty), f"T {transit_score:.3f}", fill=(140, 255, 150, 255), font=font_text)
    bar_y0 = ty + int(3 * panel_scale)
    bar_y1 = bar_y0 + bar_h
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(45, 45, 45, 220), outline=(120, 120, 120, 220))
    fill_x = int(bar_x0 + (bar_x1 - bar_x0) * transit_score)
    draw.rectangle([bar_x0, bar_y0, fill_x, bar_y1], fill=(60, 210, 110, 240))

    ty += int(20 * panel_scale)
    draw.text((tx, ty), f"step {step_idx}", fill=(225, 225, 225, 255), font=font_text)

    return np.asarray(img).astype(np.uint8)


def overlay_fastv_pruning_on_frame(
    frame,
    pruning_info,
    image_token_start_index: int = 1,
    image_token_length: int = 256,
    alpha: int = 170,
    show_label: bool = True,
    label: str = "Global",
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    rgb = _to_rgb_uint8(frame)
    if pruning_info is None:
        return rgb

    pruned_indices = pruning_info.get("pruned_indices", [])
    try:
        if pruned_indices is None:
            pruned = np.asarray([], dtype=np.int64)
        else:
            pruned = np.array(
                pruned_indices.tolist() if hasattr(pruned_indices, "tolist") else list(pruned_indices),
                dtype=np.int64,
            )
    except Exception:
        pruned = np.asarray([], dtype=np.int64)

    skipped = bool(pruning_info.get("skipped", False))
    has_mask = pruning_info is not None and (not skipped) and pruned.size > 0
    length = max(1, int(image_token_length))
    if has_mask:
        start = int(image_token_start_index)
        end = start + length
        local_pruned = pruned[(pruned >= start) & (pruned < end)] - start
        img = _overlay_pruned_patches(
            rgb,
            local_pruned,
            patches_per_image=length,
            alpha=alpha,
        )
    else:
        img = Image.fromarray(rgb, mode="RGB")

    if show_label:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(14, int(img.height * 0.055)))
        except Exception:
            font = ImageFont.load_default()
        text = label
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        x = max(6, (img.width - text_w) // 2)
        y = max(6, img.height - max(22, int(img.height * 0.10)))
        draw.text((x, y), text, fill=(245, 245, 245), font=font)

    return np.asarray(img, dtype=np.uint8)


def _self_check():
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    out = overlay_padi_scores_on_frame(frame, None, step_idx=0)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
    pruning_info = {
        "pruned_indices": np.array([1, 2, 3, 10, 20, 255], dtype=np.int64),
        "pruning_layer": 3,
        "skipped": False,
    }
    out2 = overlay_fastv_pruning_on_frame(
        frame,
        pruning_info,
        image_token_start_index=1,
        image_token_length=256,
    )
    assert isinstance(out2, np.ndarray)
    assert out2.dtype == np.uint8
    assert out2.ndim == 3 and out2.shape[2] == 3
    pruning_info_empty = {
        "pruned_indices": np.array([], dtype=np.int64),
        "skipped": False,
    }
    out3 = overlay_fastv_pruning_on_frame(
        frame,
        pruning_info_empty,
        image_token_start_index=1,
        image_token_length=256,
        alpha=170,
        show_label=True,
        label="Global",
    )
    assert isinstance(out3, np.ndarray)
    assert out3.dtype == np.uint8
    assert out3.ndim == 3 and out3.shape[2] == 3
