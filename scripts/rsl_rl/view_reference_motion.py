# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Preview reference motion clips in Isaac Lab / Isaac Sim (no policy, no training).
#
# Usage (same isaaclab.bat / Kit as training):
#   isaaclab.bat -p scripts/rsl_rl/view_reference_motion.py --robot h1 --reference_motion_path <path/to/amass_all.pkl>
#
# Headless MP4 (same Kit path as teacher play/training — no live viewport):
#   isaaclab.bat -p scripts/rsl_rl/view_reference_motion.py --record_video --clip_index 0 --robot h1
#
# List clip indices without launching sim (plain python only — do not use isaaclab.bat):
#   python scripts/rsl_rl/view_reference_motion.py --list_clips_only --reference_motion_path ... --max_list 50
#
# GUI is default so you can orbit the viewport; pass --headless for no window (not useful for viewing).
#
# Import note: run only via isaaclab.bat / Isaac Sim python so SimulationApp starts before NeuralWBCEnv imports.
#
# If Kit exits with "dependency solver failure" / omni.physx.stageupdate "can't be satisfied" / ModuleNotFoundError
# for omni.kit.usd: your conda Python version must match Isaac Sim's supported ABI (Isaac Sim 4.x kits are typically
# cp310 — use a Python 3.10 env, or the exact interpreter Isaac Lab's install guide pairs with your Sim build).

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HOVER_ROOT = Path(__file__).resolve().parents[2]


def _list_clips(pkl_path: str, max_list: int | None) -> None:
    import joblib

    data = joblib.load(pkl_path)
    keys = sorted(data.keys())
    n = len(keys)
    cap = n if max_list is None else min(max_list, n)
    for i in range(cap):
        clip = data[keys[i]]
        frames = len(clip["root_trans_offset"])
        print(f"{i:5d}  frames={frames:5d}  {keys[i]}")
    if cap < n:
        print(f"... showing {cap} of {n} clips")
    else:
        print(f"total {n} clips")


def _apply_reference_to_robot(env, ref, env_ids):
    """Snap robot root + joints to reference; return position-control actions (scaled)."""
    core = env.unwrapped if hasattr(env, "unwrapped") else env
    asset = core._robot
    joint_pos = ref.joint_pos[env_ids]
    joint_vel = ref.joint_vel[env_ids]
    asset.write_joint_state_to_sim(joint_pos, joint_vel, core._joint_ids, env_ids=env_ids)

    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, :3] = ref.root_pos[env_ids]
    root_states[:, 2] += 0.04
    root_states[:, 3:7] = ref.root_rot[env_ids]
    root_states[:, 7:10] = ref.root_lin_vel[env_ids]
    root_states[:, 10:13] = ref.root_ang_vel[env_ids]
    asset.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)

    default = asset.data.default_joint_pos[:, core._joint_ids]
    return (ref.joint_pos - default) / core.cfg.action_scale


def _run_list_only() -> None:
    parser = argparse.ArgumentParser(description="List motion clips in a PKL (no Isaac).")
    parser.add_argument("--list_clips_only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reference_motion_path", type=str, default=None)
    parser.add_argument("--max_list", type=int, default=80)
    args = parser.parse_args()
    default_pkl = HOVER_ROOT / "neural_wbc" / "data" / "data" / "motions" / "amass_all.pkl"
    motion_path = os.path.abspath(args.reference_motion_path or str(default_pkl))
    if not os.path.isfile(motion_path):
        print(f"[ERROR] Motion file not found: {motion_path}", file=sys.stderr)
        sys.exit(1)
    _list_clips(motion_path, args.max_list)


def main() -> None:
    if "--list_clips_only" in sys.argv:
        _run_list_only()
        return

    import joblib
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Preview reference motion in Isaac Lab without training or a policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot", type=str, choices=["h1"], default="h1")
    parser.add_argument("--reference_motion_path", type=str, default=None, help="Joblib PKL (dict of clips).")
    parser.add_argument("--clip_index", type=int, default=0, help="Which motion in sorted PKL key order (0-based).")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep ~policy_dt per step so playback speed matches training control rate.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=1,
        help="Repeat the clip this many times (rewind to t=0 via env.reset).",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Write MP4(s) headless (enable_cameras + RecordVideo), same profile as play.py without --gui.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Output folder for MP4s when using --record_video (default: logs/reference_motion_preview).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="With --record_video, try live viewport as well (needs working GUI Kit). Default record is headless only.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    default_pkl = HOVER_ROOT / "neural_wbc" / "data" / "data" / "motions" / "amass_all.pkl"
    motion_path = args_cli.reference_motion_path or str(default_pkl)
    motion_path = os.path.abspath(motion_path)
    if not os.path.isfile(motion_path):
        print(f"[ERROR] Motion file not found: {motion_path}", file=sys.stderr)
        sys.exit(1)

    keys_sorted = sorted(joblib.load(motion_path).keys())
    n_motions = len(keys_sorted)
    if args_cli.clip_index < 0 or args_cli.clip_index >= n_motions:
        print(f"[ERROR] clip_index out of range [0, {n_motions - 1}]. Use --list_clips_only.", file=sys.stderr)
        sys.exit(1)

    if args_cli.record_video:
        args_cli.enable_cameras = True
        if not args_cli.gui:
            args_cli.headless = True
    elif not getattr(args_cli, "headless", False):
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Omniverse requires SimulationApp before importing envs that pull in isaaclab.assets / omni.* (same order as play.py).
    import gymnasium as gym
    import torch
    from neural_wbc.core.modes import NeuralWBCModes
    from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
    from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_h1 import NeuralWBCEnvCfgH1

    env_cfg = NeuralWBCEnvCfgH1(mode=NeuralWBCModes.TEST)

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 4.0
    env_cfg.terrain.env_spacing = env_cfg.scene.env_spacing
    env_cfg.reference_motion_manager.motion_path = motion_path
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, 600.0)

    render_mode = "rgb_array" if args_cli.record_video else None
    env = NeuralWBCEnv(cfg=env_cfg, render_mode=render_mode)
    mgr = env._ref_motion_mgr

    print(f"[INFO] Clip {args_cli.clip_index} / {n_motions}: {keys_sorted[args_cli.clip_index]}")
    mgr.load_motions(random_sample=False, start_idx=args_cli.clip_index)
    policy_dt = env.cfg.decimation * env.cfg.dt
    n_steps = int(mgr.get_motion_num_steps().reshape(-1)[0].item())
    total_steps = n_steps * max(1, args_cli.loops)

    if args_cli.record_video:
        video_root = os.path.abspath(
            args_cli.video_dir or str(HOVER_ROOT / "logs" / "reference_motion_preview")
        )
        os.makedirs(video_root, exist_ok=True)
        safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in keys_sorted[args_cli.clip_index])[:80]
        video_folder = os.path.join(video_root, f"clip_{args_cli.clip_index:05d}_{safe_key}")
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=total_steps,
            disable_logger=True,
        )
        mgr.load_motions(random_sample=False, start_idx=args_cli.clip_index)
        env.reset()
        print(f"[INFO] Recording reference playback to: {video_folder} ({total_steps} control steps).")
    else:
        env.reset()

    core = env.unwrapped
    env_ids = torch.arange(core.num_envs, device=core.device, dtype=torch.long)

    print(
        f"[INFO] Playing ~{n_steps} control steps per loop (~{n_steps * policy_dt:.2f}s), "
        f"policy_dt={policy_dt:.4f}s. Close the viewport or Ctrl+C to stop."
    )

    try:
        for loop_i in range(max(1, args_cli.loops)):
            if loop_i > 0:
                mgr.load_motions(random_sample=False, start_idx=args_cli.clip_index)
                env.reset()

            for _step in range(n_steps):
                if not simulation_app.is_running():
                    break
                ref = mgr.get_state_from_motion_lib_cache(
                    episode_length_buf=core.episode_length_buf,
                    offset=core._start_positions_on_terrain,
                    terrain_heights=core.get_terrain_heights(),
                )
                actions = _apply_reference_to_robot(env, ref, env_ids)
                core._ref_motion_visualizer.visualize(ref, core._mask)
                env.step(actions)
                if args_cli.realtime:
                    time.sleep(policy_dt)

            if not simulation_app.is_running():
                break
    except KeyboardInterrupt:
        print("[INFO] Interrupted.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
