"""Render K1 mjlab policy rollout to MP4.

Bypasses mjlab's OffscreenRenderer and captures frames directly via
mujoco.Renderer so the camera can be explicitly positioned.

Usage:
    python3 scripts/render_k1_rollout.py
    python3 scripts/render_k1_rollout.py --clip 36          # longest clip (19s walk)
    python3 scripts/render_k1_rollout.py --start_frame 4142 # same, by frame index
    python3 scripts/render_k1_rollout.py --list_clips       # print clip table and exit
    python3 scripts/render_k1_rollout.py --steps 300 --checkpoint logs/.../model_22000.pt
"""

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

os.environ["MUJOCO_GL"] = "osmesa"
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("WANDB_MODE", "disabled")

MOTION_FILE = str(
    Path(__file__).parent.parent
    / "neural_wbc/data/data/mujoco/motions/k1_cmu_simple.npz"
)

WIDTH, HEIGHT = 640, 480


def detect_clip_boundaries(motion_file: str) -> list[int]:
    """Return list of start frame indices for each clip (ends at next start / total)."""
    data = np.load(motion_file)
    root_pos = data["body_pos_w"][:, 0, :2]  # XY only
    jumps = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)
    boundaries = list(np.where(jumps > 1.0)[0] + 1)
    return [0] + boundaries  # each entry is the start frame of that clip


def print_clip_table(motion_file: str):
    starts = detect_clip_boundaries(motion_file)
    data = np.load(motion_file)
    total = data["body_pos_w"].shape[0]
    ends = starts[1:] + [total]
    print(f"{'Clip':>5}  {'Start':>6}  {'End':>6}  {'Dur(s)':>7}")
    print("-" * 32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        print(f"{i:>5}  {s:>6}  {e:>6}  {(e-s)*0.02:>7.1f}")


def find_latest_checkpoint(log_root: str = "logs/rsl_rl/k1_tracking") -> str:
    runs = sorted(Path(log_root).glob("*/model_*.pt"))
    if not runs:
        raise FileNotFoundError(f"No checkpoints found under {log_root}")
    def iteration(p):
        try: return int(p.stem.split("_")[1])
        except: return 0
    best = max(runs, key=iteration)
    print(f"[render] Using checkpoint: {best}")
    return str(best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--steps", type=int, default=300,
                        help="Number of steps to render (300 = 6s at 50 Hz)")
    parser.add_argument("--clip", type=int, default=None,
                        help="Clip index to start from (see --list_clips)")
    parser.add_argument("--start_frame", type=int, default=None,
                        help="Exact frame index to start from (overrides --clip)")
    parser.add_argument("--list_clips", action="store_true",
                        help="Print clip table and exit")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.list_clips:
        print_clip_table(MOTION_FILE)
        return

    clip_starts = detect_clip_boundaries(MOTION_FILE)

    if args.start_frame is not None:
        start_frame = args.start_frame
    elif args.clip is not None:
        if args.clip < 0 or args.clip >= len(clip_starts):
            raise ValueError(f"--clip must be 0–{len(clip_starts)-1}, got {args.clip}")
        start_frame = clip_starts[args.clip]
        print(f"[render] Clip {args.clip} → start frame {start_frame}")
    else:
        start_frame = None  # random RSI (default training behaviour)

    checkpoint = args.checkpoint or find_latest_checkpoint()

    clip_tag = f"_clip{args.clip}" if args.clip is not None else (
               f"_frame{start_frame}" if start_frame is not None else "")
    video_folder = args.out or str(Path(checkpoint).parent / "videos" / "play")
    Path(video_folder).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(video_folder) / f"k1_rollout{clip_tag}.mp4")

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.tracking.mdp.commands import MotionCommand
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    from mjlab_k1.env_cfg import k1_flat_tracking_env_cfg
    from mjlab_k1.rl_cfg import k1_tracking_ppo_runner_cfg

    TASK_ID = f"Mjlab-Tracking-Flat-K1-Render{clip_tag}"
    play_env_cfg = k1_flat_tracking_env_cfg(motion_file=MOTION_FILE, play=True)
    play_env_cfg.scene.num_envs = 1
    rl_cfg = k1_tracking_ppo_runner_cfg()

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=play_env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=rl_cfg,
        runner_cls=MotionTrackingOnPolicyRunner,
    )

    print("[render] Creating env ...")
    env = ManagerBasedRlEnv(cfg=play_env_cfg, device=args.device)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    runner = MotionTrackingOnPolicyRunner(
        env_wrapped, asdict(rl_cfg), device=args.device
    )
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    import mujoco
    mj_model = env.sim.mj_model
    mj_data = mujoco.MjData(mj_model)

    for candidate in ("robot/Trunk", "Trunk"):
        trunk_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if trunk_id >= 0:
            break
    else:
        trunk_id = 2
    print(f"[render] Tracking body id={trunk_id} ({mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, trunk_id)})")

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = trunk_id
    cam.distance = 3.5
    cam.elevation = -15.0
    cam.azimuth = 90.0

    opt = mujoco.MjvOption()
    opt.geomgroup[3] = True  # K1 geoms are all in group 3 (collision-only model)

    renderer = mujoco.Renderer(mj_model, height=HEIGHT, width=WIDTH)

    frames = []
    obs, _ = env_wrapped.reset()

    # Jump to the requested frame after reset
    if start_frame is not None:
        motion_cmd = cast(
            MotionCommand,
            env.command_manager.get_term("motion"),
        )
        all_envs = torch.arange(env.num_envs, device=args.device)
        motion_cmd.reset_to_frame(all_envs, start_frame)
        # Re-compute obs after teleporting to the new frame
        obs = env_wrapped.get_observations()
        print(f"[render] Reset to frame {start_frame}")

    steps_label = args.steps
    print(f"[render] Rolling out {steps_label} steps ...")
    with torch.no_grad():
        for step in range(args.steps):
            action = policy(obs)
            obs, _, _, _ = env_wrapped.step(action)

            mj_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            mj_data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam, scene_option=opt)
            frames.append(renderer.render().copy())

            if step % 50 == 0:
                print(f"  step {step}/{args.steps}")

    renderer.close()
    env_wrapped.close()

    import mediapy as media
    media.write_video(out_path, frames, fps=50)
    print(f"[render] Video saved → {out_path}")

    gif_path = out_path.replace(".mp4", ".gif")
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", out_path,
        "-vf", "fps=25,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop", "0", gif_path,
    ], check=True, capture_output=True)
    print(f"[render] GIF saved   → {gif_path}")


if __name__ == "__main__":
    main()
