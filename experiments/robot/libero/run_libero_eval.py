"""
run_libero_eval.py
Runs a model in a LIBERO simulation environment.
"""

import os
import sys
import time
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import imageio
import draccus
import numpy as np
import tqdm
import wandb
import torch

script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.parent
libero_path = project_root / "LIBERO"
if str(libero_path) not in sys.path:
    sys.path.insert(0, str(libero_path))

from libero.libero import benchmark

# Append current directory so that interpreter can find experiments.robot
sys.path.insert(0, str(project_root))
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # Use VLA-Pruner for faster inference
    # FastV Token Pruning Configuration
    use_fastv: bool = True              #enable fastvforward for token pruning
    fastv_k: int = 3                    # Layer to start pruning
    fastv_r: float = 0.50               # Pruning ratio (tokens to remove)
    fastv_image_token_start_index: int = 1  # Image token start index
    fastv_image_token_length: int = 256     # Image token length
    sparsevlm: bool = False         # enable sparsevlm
    #VLA-Pruner Settings
    use_temporal: bool = True      # Whether to use temporal attention guidance
    temporal_w: int = 3           # Temporal window size
    temporal_gamma: float = 0.80  # Temporal weight decay factor
    #use test-to-vison attention or prefill attention
    use_text_vision_selection: bool = False
    use_prefil_attention: bool = False
    measure_latency: bool = False
    latency_warmup_queries: int = 3
    latency_log_per_query: bool = False
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "checkpoints/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under
    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on

def _cuda_sync_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_latency_mode(cfg) -> str:
    modes = []
    if getattr(cfg, "use_fastv", False):
        modes.append(f"fastv_r{cfg.fastv_r}_k{cfg.fastv_k}")
    if not modes:
        return "baseline"
    return "+".join(modes)


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    # Set random seed
    set_seed_everywhere(cfg.seed)
    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    # Load model
    model = get_model(cfg)
    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )
    log_file.write("=" * 80 + "\n")
    log_file.write("CONFIGURATION SUMMARY:\n")
    log_file.write("=" * 80 + "\n")
    for field_name, field_value in cfg.__dict__.items():
        log_file.write(f"  {field_name}: {field_value}\n")
    log_file.write("=" * 80 + "\n")
    log_file.write("\n")
    latency_cfg_msg = (
        f"[PADI-V2 Config][Latency] enabled={cfg.measure_latency} "
        f"warmup_queries={cfg.latency_warmup_queries} "
        f"log_per_query={cfg.latency_log_per_query} mode={_get_latency_mode(cfg)}"
    )
    print(latency_cfg_msg)
    log_file.write(latency_cfg_msg + "\n")
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
        # Start episodes
        task_episodes, task_successes = 0, 0
        task_latency_records = []
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            # Reset environment
            env.reset()
            model.reset_av_history()
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            # Setup
            t = 0
            replay_images = []
            replay_images_heatmap = []
            prev_img = None
            last_caches = None
            latency_records = []
            policy_query_idx = 0
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue
                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)
                    # Save previous image
                    if prev_img is None:
                        prev_img = img
                    else:
                        prev_img = replay_images[-1]
                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "prev_image": prev_img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    # Query model to get action
                    if cfg.measure_latency:
                        _cuda_sync_if_available()
                        policy_t0 = time.perf_counter()
                    action, last_caches, result_image = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        last_caches=last_caches,
                    )
                    if cfg.measure_latency:
                        _cuda_sync_if_available()
                        policy_t1 = time.perf_counter()
                        policy_query_latency_ms = float((policy_t1 - policy_t0) * 1000.0)
                    else:
                        policy_query_latency_ms = None
                    if cfg.measure_latency:
                        latest_fastv_pruning_info = getattr(model, "last_pruning_info", None)
                        llm_latency_ms = getattr(model, "last_llm_latency_ms", None)
                        llm_forward_count = getattr(model, "last_llm_forward_count", None)
                        record = {
                            "query_idx": policy_query_idx,
                            "step": t,
                            "policy_query_latency_ms": policy_query_latency_ms,
                            "llm_forward_latency_ms": llm_latency_ms,
                            "llm_forward_count": llm_forward_count,
                            "use_fastv": bool(getattr(cfg, "use_fastv", False)),
                            "fastv_r": getattr(model, "fastv_r", getattr(cfg, "fastv_r", None))
                            if getattr(cfg, "use_fastv", False)
                            else None,
                            "fastv_k": getattr(cfg, "fastv_k", None) if getattr(cfg, "use_fastv", False) else None,
                        }
                        if isinstance(latest_fastv_pruning_info, dict):
                            for key in [
                                "original_seq_length",
                                "kept_seq_length",
                                "pruned_count",
                                "num_keep_per_image",
                                "skipped",
                                "skip_reason",
                                "pruned_indices",
                            ]:
                                if key in latest_fastv_pruning_info:
                                    record[key] = latest_fastv_pruning_info[key]
                        latency_records.append(record)
                        if cfg.latency_log_per_query:
                            kept = record.get("kept_seq_length", "NA")
                            pruned = record.get("pruned_count", "NA")
                            query_msg = (
                                f"[PADI-V2 Latency-Query] step={t} query={policy_query_idx} "
                                f"policy_ms={policy_query_latency_ms:.3f} llm_ms={llm_latency_ms} "
                                f"llm_forward_count={llm_forward_count} "
                                f"use_fastv={cfg.use_fastv} kept={kept} pruned={pruned}"
                            )
                            print(query_msg)
                            log_file.write(query_msg + "\n")
                        policy_query_idx += 1
                    replay_images_heatmap.append(result_image)
                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)
                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)
                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
            task_episodes += 1
            total_episodes += 1
            if cfg.measure_latency:
                task_latency_records.extend(latency_records)

            save_rollout_video(
                        replay_images_heatmap, total_episodes, success=done, task_description=task_description, log_file=log_file
                    )
            # Save a replay video of the episode
            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()
        # Log final results
        if cfg.measure_latency:
            warmup = max(int(cfg.latency_warmup_queries), 0)
            valid_records = [r for r in task_latency_records if int(r.get("query_idx", 0)) >= warmup]
            warmup_fallback = False
            if len(valid_records) == 0:
                valid_records = task_latency_records
                warmup_fallback = True

            policy_vals = [r["policy_query_latency_ms"] for r in valid_records if r.get("policy_query_latency_ms") is not None]
            llm_vals = [r["llm_forward_latency_ms"] for r in valid_records if r.get("llm_forward_latency_ms") is not None]
            llm_forward_count_vals = [int(r["llm_forward_count"]) for r in valid_records if r.get("llm_forward_count") is not None]

            def _stats(values):
                if len(values) == 0:
                    return float("nan"), float("nan"), float("nan")
                arr = np.asarray(values, dtype=np.float64)
                return float(np.mean(arr)), float(np.percentile(arr, 50)), float(np.percentile(arr, 90))

            policy_mean, policy_p50, policy_p90 = _stats(policy_vals)
            llm_mean, llm_p50, llm_p90 = _stats(llm_vals)
            if len(llm_forward_count_vals) > 0:
                llm_forward_count_mean, llm_forward_count_p50, llm_forward_count_p90 = _stats(llm_forward_count_vals)
                llm_forward_count_mean = f"{llm_forward_count_mean:.3f}"
                llm_forward_count_p50 = f"{llm_forward_count_p50:.3f}"
                llm_forward_count_p90 = f"{llm_forward_count_p90:.3f}"
            else:
                llm_forward_count_mean = llm_forward_count_p50 = llm_forward_count_p90 = "NA"

            token_orig_vals, token_kept_vals, token_pruned_vals, token_keep_ratio_vals = [], [], [], []
            for r in valid_records:
                original = r.get("original_seq_length")
                kept = r.get("kept_seq_length")
                try:
                    original_f = float(original)
                    kept_f = float(kept)
                except (TypeError, ValueError):
                    continue
                if original_f <= 0:
                    continue
                pruned = r.get("pruned_count")
                if pruned is None and r.get("pruned_indices") is not None:
                    pruned_indices = r.get("pruned_indices")
                    pruned = len(pruned_indices.tolist()) if hasattr(pruned_indices, "tolist") else len(pruned_indices)
                if pruned is None:
                    pruned = original_f - kept_f
                token_orig_vals.append(original_f)
                token_kept_vals.append(kept_f)
                token_pruned_vals.append(float(pruned))
                token_keep_ratio_vals.append(kept_f / original_f)

            if len(token_orig_vals) > 0:
                token_orig = f"{float(np.mean(token_orig_vals)):.3f}"
                token_kept = f"{float(np.mean(token_kept_vals)):.3f}"
                token_pruned = f"{float(np.mean(token_pruned_vals)):.3f}"
                token_keep_ratio = f"{float(np.mean(token_keep_ratio_vals)):.6f}"
            else:
                token_orig = token_kept = token_pruned = token_keep_ratio = "NA"

            latency_summary_msg = (
                f"[PADI-V2 Latency] scope=task task_id={task_id} task=\"{task_description}\" "
                f"mode={_get_latency_mode(cfg)} episodes={task_episodes} queries={len(task_latency_records)} "
                f"measured={len(valid_records)} warmup={warmup} warmup_fallback={warmup_fallback} "
                f"policy_ms_mean={policy_mean:.3f} policy_ms_p50={policy_p50:.3f} policy_ms_p90={policy_p90:.3f} "
                f"llm_ms_mean={llm_mean:.3f} llm_ms_p50={llm_p50:.3f} llm_ms_p90={llm_p90:.3f} "
                f"llm_forward_count_mean={llm_forward_count_mean} llm_forward_count_p50={llm_forward_count_p50} "
                f"llm_forward_count_p90={llm_forward_count_p90} "
                f"token_orig={token_orig} token_kept={token_kept} token_pruned={token_pruned} token_keep_ratio={token_keep_ratio}"
            )
            print(latency_summary_msg)
            log_file.write(latency_summary_msg + "\n")
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
    # Save local log file
    log_file.close()
    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

if __name__ == "__main__":
    eval_libero()
