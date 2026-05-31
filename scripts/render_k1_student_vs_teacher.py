"""Render teacher vs student K1 policy side-by-side on the same motion clip.

Layout per frame:
  [ Teacher (640x480) | Student (640x480) ]
  [   Top-down root trajectory overlay (1280x240)   ]

Usage:
    python3 scripts/render_k1_student_vs_teacher.py
    python3 scripts/render_k1_student_vs_teacher.py --clip 0 --steps 300
    python3 scripts/render_k1_student_vs_teacher.py --list_clips
    python3 scripts/render_k1_student_vs_teacher.py \\
        --teacher_checkpoint logs/.../model_29999.pt \\
        --student_checkpoint logs/.../student_all.../final_model.pt \\
        --mask_mode full
"""

import argparse
import io
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
LOG_ROOT = "logs/rsl_rl/k1_tracking"
STUDENT_DIR_GLOB = "*/student_all_k1_student_all"

WIDTH, HEIGHT = 640, 480
TRAJ_H = 240   # height of trajectory panel
TRAJ_W = WIDTH * 2  # 1280


def detect_clip_boundaries(motion_file: str) -> list[int]:
    data = np.load(motion_file)
    root_pos = data["body_pos_w"][:, 0, :2]
    jumps = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)
    return [0] + list(np.where(jumps > 1.0)[0] + 1)


def get_reference_xy(motion_file: str, start_frame: int, steps: int) -> np.ndarray:
    """Extract reference root XY positions for a clip window."""
    data = np.load(motion_file)
    total = data["body_pos_w"].shape[0]
    end = min(start_frame + steps, total)
    xy = data["body_pos_w"][start_frame:end, 0, :2].copy()
    # Remove env_origin offset — reference starts wherever it starts, normalise to origin.
    xy -= xy[0]
    return xy


def print_clip_table(motion_file: str):
    starts = detect_clip_boundaries(motion_file)
    data = np.load(motion_file)
    total = data["body_pos_w"].shape[0]
    ends = starts[1:] + [total]
    print(f"{'Clip':>5}  {'Start':>6}  {'End':>6}  {'Dur(s)':>7}")
    print("-" * 32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        print(f"{i:>5}  {s:>6}  {e:>6}  {(e-s)*0.02:>7.1f}")


def find_latest_checkpoint(log_root: str, pattern: str = "*/model_*.pt") -> str:
    runs = sorted(Path(log_root).glob(pattern))
    if not runs:
        raise FileNotFoundError(f"No checkpoints found under {log_root}/{pattern}")
    def _iter(p):
        try: return int(p.stem.split("_")[1])
        except: return 0
    return str(max(runs, key=_iter))


def find_student_checkpoint(log_root: str) -> str:
    candidates = sorted(Path(log_root).glob(STUDENT_DIR_GLOB + "/final_model.pt"))
    if not candidates:
        candidates = sorted(Path(log_root).glob(STUDENT_DIR_GLOB + "/model_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No student checkpoint found under {log_root}")
        def _iter(p):
            try: return int(p.stem.split("_")[1])
            except: return 0
        return str(max(candidates, key=_iter))
    return str(candidates[-1])


def make_renderer(mj_model, trunk_id):
    import mujoco
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(mj_model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = trunk_id
    cam.distance = 3.5
    cam.elevation = -15.0
    cam.azimuth = 90.0
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = True
    renderer = mujoco.Renderer(mj_model, height=HEIGHT, width=WIDTH)
    return renderer, cam, opt


def rollout(env, env_wrapped, policy_fn, motion_cmd, mj_model, trunk_id,
            start_frame, steps, device, label):
    import mujoco
    renderer, cam, opt = make_renderer(mj_model, trunk_id)
    mj_data = mujoco.MjData(mj_model)

    obs, _ = env_wrapped.reset()
    if start_frame is not None:
        motion_cmd.reset_to_frame(torch.arange(env.num_envs, device=device), start_frame)
        obs = env_wrapped.get_observations()
        print(f"[render] {label}: reset to frame {start_frame}")

    frames, root_xy = [], []
    print(f"[render] {label}: rolling out {steps} steps ...")
    with torch.no_grad():
        for step in range(steps):
            action = policy_fn(obs)
            obs, rew, dones, _ = env_wrapped.step(action)

            mj_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            mj_data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam, scene_option=opt)
            frames.append(renderer.render().copy())
            root_xy.append(mj_data.qpos[:2].copy())

            if step % 50 == 0:
                print(f"  {label} step {step}/{steps}")

    renderer.close()
    return frames, np.array(root_xy)


def rollout_student(env, env_wrapped, student_fn, motion_cmd, robot,
                    mj_model, trunk_id, start_frame, steps, device, mask_mode):
    import mujoco
    from neural_wbc.core.mask import create_mask
    from mjlab_k1.k1_student_obs import (
        K1_MASK_ELEMENT_NAMES, K1_DISTILL_MASK_MODES, compute_k1_student_obs,
    )

    renderer, cam, opt = make_renderer(mj_model, trunk_id)
    mj_data = mujoco.MjData(mj_model)

    env_wrapped.reset()
    if start_frame is not None:
        motion_cmd.reset_to_frame(torch.arange(env.num_envs, device=device), start_frame)
        env_wrapped.get_observations()
        print(f"[render] student: reset to frame {start_frame}")

    modes = {mask_mode: K1_DISTILL_MASK_MODES[mask_mode]}
    mask = create_mask(1, K1_MASK_ELEMENT_NAMES, modes,
                       enable_sparsity_randomization=False, device=device).float()
    last_actions = torch.zeros(1, 22, device=device)

    frames, root_xy = [], []
    print(f"[render] student ({mask_mode}): rolling out {steps} steps ...")
    with torch.no_grad():
        for step in range(steps):
            obs_dict = compute_k1_student_obs(motion_cmd, robot, last_actions, mask, device)
            student_obs = torch.cat(list(obs_dict.values()), dim=-1)
            action = student_fn(student_obs)
            last_actions = action.clone()

            _, rew, dones, _ = env_wrapped.step(action)

            mj_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            mj_data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam, scene_option=opt)
            frames.append(renderer.render().copy())
            root_xy.append(mj_data.qpos[:2].copy())

            if step % 50 == 0:
                print(f"  student step {step}/{steps}")

    renderer.close()
    return frames, np.array(root_xy)


def make_trajectory_panel(
    t_xy: np.ndarray,   # [N, 2] teacher positions so far
    s_xy: np.ndarray,   # [N, 2] student positions so far
    ref_xy: np.ndarray, # [M, 2] reference motion XY (normalised)
    xlim: tuple, ylim: tuple,
    width: int, height: int,
) -> np.ndarray:
    """Render a top-down trajectory plot to an RGB numpy array [H, W, 3]."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    # Reference motion
    if len(ref_xy) > 1:
        ax.plot(ref_xy[:, 0], ref_xy[:, 1],
                color="#888888", lw=1.2, ls="--", alpha=0.6, label="Reference")

    # Teacher trail + current dot
    if len(t_xy) > 1:
        ax.plot(t_xy[:, 0], t_xy[:, 1], color="#4c9be8", lw=1.5, alpha=0.8)
    if len(t_xy):
        ax.plot(t_xy[-1, 0], t_xy[-1, 1], "o", color="#4c9be8", ms=7, label="Teacher")

    # Student trail + current dot
    if len(s_xy) > 1:
        ax.plot(s_xy[:, 0], s_xy[:, 1], color="#f5a623", lw=1.5, alpha=0.8)
    if len(s_xy):
        ax.plot(s_xy[-1, 0], s_xy[-1, 1], "o", color="#f5a623", ms=7, label="Student")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="#cccccc", fontsize=8)
    ax.set_ylabel("Y (m)", color="#cccccc", fontsize=8)
    ax.tick_params(colors="#cccccc", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.set_title("Root trajectory (top-down)", color="#cccccc", fontsize=9, pad=4)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.3,
              labelcolor="#cccccc", facecolor="#333333")
    ax.grid(True, color="#333333", lw=0.5)

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    # Resize to exact target size
    from PIL import Image
    img = Image.fromarray(buf).resize((width, height), Image.LANCZOS)
    return np.array(img)


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    try:
        import cv2
        out = frame.copy()
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return out
    except ImportError:
        return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--student_checkpoint", type=str, default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--clip", type=int, default=None)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--list_clips", action="store_true")
    parser.add_argument("--mask_mode", type=str, default="full",
                        choices=["locomotion", "upper_body", "full"])
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
        start_frame = clip_starts[args.clip]
        print(f"[render] Clip {args.clip} → start frame {start_frame}")
    else:
        start_frame = clip_starts[0]
        print(f"[render] Defaulting to clip 0 (frame {start_frame})")

    teacher_ckpt = args.teacher_checkpoint or find_latest_checkpoint(LOG_ROOT)
    student_ckpt = args.student_checkpoint or find_student_checkpoint(LOG_ROOT)
    print(f"[render] Teacher: {teacher_ckpt}")
    print(f"[render] Student: {student_ckpt}")

    # Reference trajectory (normalised to start at origin)
    ref_xy = get_reference_xy(MOTION_FILE, start_frame, args.steps)

    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.tracking.mdp.commands import MotionCommand
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
    from neural_wbc.student_policy.policy import StudentPolicy

    from mjlab_k1.env_cfg import k1_flat_tracking_env_cfg
    from mjlab_k1.rl_cfg import k1_tracking_ppo_runner_cfg

    rl_cfg = k1_tracking_ppo_runner_cfg()
    clip_tag = f"_clip{args.clip}" if args.clip is not None else f"_frame{start_frame}"

    # ── Teacher ──────────────────────────────────────────────────────────────
    TASK_T = f"Mjlab-K1-RenderTeacher{clip_tag}"
    cfg_t = k1_flat_tracking_env_cfg(motion_file=MOTION_FILE, play=True)
    cfg_t.scene.num_envs = 1
    register_mjlab_task(TASK_T, cfg_t, cfg_t, rl_cfg, MotionTrackingOnPolicyRunner)

    print("[render] Creating teacher env ...")
    env_t = ManagerBasedRlEnv(cfg=cfg_t, device=args.device)
    wrap_t = RslRlVecEnvWrapper(env_t, clip_actions=rl_cfg.clip_actions)

    runner = MotionTrackingOnPolicyRunner(wrap_t, asdict(rl_cfg), device=args.device)
    runner.load(teacher_ckpt, load_cfg={"actor": True}, strict=True, map_location=args.device)
    raw_teacher = runner.get_inference_policy(device=args.device)

    def teacher_policy(obs_td):
        return raw_teacher({"actor": obs_td["actor"]})

    mj_model = env_t.sim.mj_model
    trunk_id = next(
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("robot/Trunk", "Trunk")
        if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0
    )
    motion_cmd_t = cast(MotionCommand, env_t.command_manager.get_term("motion"))

    teacher_frames, teacher_xy = rollout(
        env_t, wrap_t, teacher_policy, motion_cmd_t,
        mj_model, trunk_id, start_frame, args.steps, args.device, "teacher"
    )
    wrap_t.close()

    # ── Student ───────────────────────────────────────────────────────────────
    TASK_S = f"Mjlab-K1-RenderStudent{clip_tag}"
    cfg_s = k1_flat_tracking_env_cfg(motion_file=MOTION_FILE, play=True)
    cfg_s.scene.num_envs = 1
    register_mjlab_task(TASK_S, cfg_s, cfg_s, rl_cfg, MotionTrackingOnPolicyRunner)

    print("[render] Creating student env ...")
    env_s = ManagerBasedRlEnv(cfg=cfg_s, device=args.device)
    wrap_s = RslRlVecEnvWrapper(env_s, clip_actions=rl_cfg.clip_actions)

    student = StudentPolicy(222, 22, [512, 256, 128], "elu", 0.001).to(args.device)
    ckpt = torch.load(student_ckpt, map_location=args.device)
    student.load_state_dict(ckpt["model_state_dict"])
    student.eval()

    mj_model_s = env_s.sim.mj_model
    motion_cmd_s = cast(MotionCommand, env_s.command_manager.get_term("motion"))
    robot_s = env_s.scene.entities["robot"]

    student_frames, student_xy = rollout_student(
        env_s, wrap_s, student.act_inference, motion_cmd_s, robot_s,
        mj_model_s, trunk_id, start_frame, args.steps, args.device, args.mask_mode
    )
    wrap_s.close()

    # ── Trajectory axis limits (fixed for whole video) ─────────────────────
    # Normalise both trajectories to teacher's start position so they're comparable.
    t0 = teacher_xy[0].copy()
    teacher_xy -= t0
    student_xy -= student_xy[0]  # student starts at its own reset position

    all_xy = np.concatenate([teacher_xy, student_xy, ref_xy], axis=0)
    pad = 0.5
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    # Square up the axes
    xr, yr = xlim[1] - xlim[0], ylim[1] - ylim[0]
    if xr > yr:
        mid = (ylim[0] + ylim[1]) / 2
        ylim = (mid - xr / 2, mid + xr / 2)
    else:
        mid = (xlim[0] + xlim[1]) / 2
        xlim = (mid - yr / 2, mid + yr / 2)

    # ── Composite frames ───────────────────────────────────────────────────
    print("[render] Compositing frames ...")
    combined = []
    for i, (tf, sf) in enumerate(zip(teacher_frames, student_frames)):
        top = np.concatenate([
            add_label(tf, "Teacher"),
            add_label(sf, f"Student ({args.mask_mode})"),
        ], axis=1)   # [480, 1280, 3]

        panel = make_trajectory_panel(
            teacher_xy[:i+1], student_xy[:i+1], ref_xy,
            xlim, ylim, TRAJ_W, TRAJ_H,
        )                            # [240, 1280, 3]

        combined.append(np.concatenate([top, panel], axis=0))  # [720, 1280, 3]

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = args.out or str(Path(student_ckpt).parent / "videos")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"{clip_tag}_{args.mask_mode}"
    mp4_path = str(Path(out_dir) / f"k1_vs{tag}.mp4")
    gif_path = mp4_path.replace(".mp4", ".gif")

    import mediapy as media
    media.write_video(mp4_path, combined, fps=50)
    print(f"[render] Video saved → {mp4_path}")

    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path,
        "-vf", "fps=20,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop", "0", gif_path,
    ], check=True, capture_output=True)
    print(f"[render] GIF saved   → {gif_path}")


if __name__ == "__main__":
    main()
