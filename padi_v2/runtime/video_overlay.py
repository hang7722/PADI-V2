from __future__ import annotations

from typing import Any, Optional

import numpy as np


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


def _self_check():
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    out = overlay_padi_scores_on_frame(frame, None, step_idx=0)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.ndim == 3 and out.shape[2] == 3
