"""Render teacher vs student side-by-side for K1.

Layout per frame:
  [ Teacher (640x480) | Student (640x480) ]
  [        Trajectory panel (1280x240)     ]

Usage:
    python3 scripts/render_k1_comparison.py \\
        --teacher_checkpoint logs/mjlab/.../model_49999.pt \\
        --student_checkpoint logs/mjlab/.../student_all_k1_student_v6/final_model.pt \\
        --motion_file neural_wbc/data/data/mujoco/motions/k1_teleop_no_jump.npz \\
        --clip 1 --steps 300 --out comparison_clip1.mp4
"""

import argparse
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

HALF_W, HEIGHT = 640, 480
TRAJ_H, TRAJ_W = 240, HALF_W * 2

_K1_SCENE_XML = str(
    Path(__file__).parent.parent
    / "neural_wbc/data/data/mujoco/models/scene_k1_vis.xml"
)


def detect_clip_boundaries(motion_file: str) -> list[int]:
    data = np.load(motion_file)
    root_pos = data["body_pos_w"][:, 0, :2]
    jumps = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)
    return [0] + list(np.where(jumps > 1.0)[0] + 1)


def get_reference_xy(motion_file: str, start_frame: int, steps: int) -> np.ndarray:
    data = np.load(motion_file)
    total = data["body_pos_w"].shape[0]
    end = min(start_frame + steps, total)
    xy = data["body_pos_w"][start_frame:end, 0, :2].copy()
    xy -= xy[0]
    return xy


def make_half_renderer(mj_model, trunk_id):
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
    renderer = mujoco.Renderer(mj_model, height=HEIGHT, width=HALF_W)
    return renderer, cam, opt


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    try:
        import cv2
        out = frame.copy()
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return out
    except ImportError:
        return frame


def make_trajectory_panel(t_xy, s_xy, ref_xy, xlim, ylim, width, height) -> np.ndarray:
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    if len(ref_xy) > 1:
        ax.plot(ref_xy[:, 0], ref_xy[:, 1], color="#888888", lw=1.2, ls="--",
                alpha=0.6, label="Reference")
    if len(t_xy) > 1:
        ax.plot(t_xy[:, 0], t_xy[:, 1], color="#4c9be8", lw=1.5, alpha=0.8)
    if len(t_xy):
        ax.plot(t_xy[-1, 0], t_xy[-1, 1], "o", color="#4c9be8", ms=7, label="Teacher")
    if len(s_xy) > 1:
        ax.plot(s_xy[:, 0], s_xy[:, 1], color="#e8784c", lw=1.5, alpha=0.8)
    if len(s_xy):
        ax.plot(s_xy[-1, 0], s_xy[-1, 1], "s", color="#e8784c", ms=7, label="Student")

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
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

    from PIL import Image
    return np.array(Image.fromarray(buf).resize((width, height), Image.LANCZOS))


def rollout_teacher(env, wrap, runner, args, start_frame, vis_model, vis_data,
                    trunk_id, steps, device):
    import mujoco
    from mjlab.tasks.tracking.mdp.commands import MotionCommand

    motion_cmd = cast(MotionCommand, env.command_manager.get_term("motion"))
    raw_policy = runner.get_inference_policy(device=device)

    frames, root_xy = [], []
    renderer, cam, opt = make_half_renderer(vis_model, trunk_id)
    print("[teacher] Rolling out ...")
    with torch.no_grad():
        obs, _ = wrap.reset()
        motion_cmd.reset_to_frame(torch.arange(env.num_envs, device=device), start_frame)
        obs = wrap.get_observations()
        for step in range(steps):
            action = raw_policy({"actor": obs["actor"]})
            obs, _, _, _ = wrap.step(action)
            vis_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            vis_data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(vis_model, vis_data)
            renderer.update_scene(vis_data, camera=cam, scene_option=opt)
            frames.append(renderer.render().copy())
            root_xy.append(vis_data.qpos[:2].copy())
            if step % 50 == 0:
                print(f"  teacher step {step}/{steps}")
    renderer.close()
    return frames, np.array(root_xy)


def rollout_student(env, wrap, student_policy, args, start_frame, vis_model, vis_data,
                    trunk_id, steps, mask_mode, device):
    import mujoco
    from mjlab.tasks.tracking.mdp.commands import MotionCommand
    from mjlab_k1.k1_student_obs import compute_k1_student_obs, K1_MASK_ELEMENT_NAMES
    from neural_wbc.core.mask import create_mask

    motion_cmd = cast(MotionCommand, env.command_manager.get_term("motion"))
    robot = env.scene.entities["robot"]

    # Build a fixed mask for the chosen mode.
    from mjlab_k1.k1_student_obs import K1_DISTILL_MASK_MODES
    if mask_mode == "full":
        mask_modes = {"full": K1_DISTILL_MASK_MODES["full"]}
    else:
        mask_modes = {mask_mode: K1_DISTILL_MASK_MODES[mask_mode]}
    mask = create_mask(1, K1_MASK_ELEMENT_NAMES, mask_modes, False, device).float()  # [1, 52]

    last_actions = torch.zeros(1, 22, device=device)
    frames, root_xy = [], []
    renderer, cam, opt = make_half_renderer(vis_model, trunk_id)
    print(f"[student] Rolling out (mask_mode={mask_mode}) ...")
    with torch.no_grad():
        wrap.reset()
        motion_cmd.reset_to_frame(torch.arange(env.num_envs, device=device), start_frame)
        for step in range(steps):
            obs_dict = compute_k1_student_obs(motion_cmd, robot, last_actions, mask, device)
            student_obs = torch.cat(list(obs_dict.values()), dim=-1)  # [1, 222]
            action = student_policy(student_obs)  # act_inference → [1, 22]
            last_actions = action.clone()
            wrap.step(action)
            vis_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            vis_data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(vis_model, vis_data)
            renderer.update_scene(vis_data, camera=cam, scene_option=opt)
            frames.append(renderer.render().copy())
            root_xy.append(vis_data.qpos[:2].copy())
            if step % 50 == 0:
                print(f"  student step {step}/{steps}")
    renderer.close()
    return frames, np.array(root_xy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--student_checkpoint", type=str, required=True)
    parser.add_argument("--motion_file", type=str, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--clip", type=int, default=None)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mask_mode", type=str, default="full",
                        choices=["locomotion", "upper_body", "full"],
                        help="Student mask mode to render")
    args = parser.parse_args()

    clip_starts = detect_clip_boundaries(args.motion_file)
    if args.start_frame is not None:
        start_frame = args.start_frame
    elif args.clip is not None:
        start_frame = clip_starts[args.clip]
        print(f"[render] Clip {args.clip} → start frame {start_frame}")
    else:
        start_frame = clip_starts[0]
        print(f"[render] Defaulting to clip 0 (frame {start_frame})")

    ref_xy = get_reference_xy(args.motion_file, start_frame, args.steps)

    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    from mjlab_k1.env_cfg import k1_flat_tracking_env_cfg
    from mjlab_k1.rl_cfg import k1_tracking_ppo_runner_cfg

    from neural_wbc.student_policy.policy import StudentPolicy

    clip_tag = f"clip{args.clip}" if args.clip is not None else f"frame{start_frame}"
    TASK_ID = f"Mjlab-K1-Comparison-{clip_tag}"
    rl_cfg = k1_tracking_ppo_runner_cfg()
    cfg = k1_flat_tracking_env_cfg(motion_file=args.motion_file, play=True)
    cfg.scene.num_envs = 1
    register_mjlab_task(TASK_ID, cfg, cfg, rl_cfg, MotionTrackingOnPolicyRunner)

    print("[render] Creating env ...")
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    wrap = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    # Load teacher runner.
    runner = MotionTrackingOnPolicyRunner(wrap, asdict(rl_cfg), device=args.device)
    runner.load(args.teacher_checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=args.device)

    # Load student policy.
    print(f"[render] Loading student from {args.student_checkpoint} ...")
    student = StudentPolicy(222, 22, [512, 256, 128], "elu", 0.001)
    student.load(args.student_checkpoint, device=args.device)
    student.eval()

    # Shared vis model / data.
    vis_model = mujoco.MjModel.from_xml_path(_K1_SCENE_XML)
    vis_data = mujoco.MjData(vis_model)
    trunk_id = next(
        mujoco.mj_name2id(vis_model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("robot/Trunk", "Trunk", "Trunk_link")
        if mujoco.mj_name2id(vis_model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0
    )

    teacher_frames, teacher_xy_raw = rollout_teacher(
        env, wrap, runner, args, start_frame, vis_model, vis_data, trunk_id,
        args.steps, args.device,
    )
    student_frames, student_xy_raw = rollout_student(
        env, wrap, student.act_inference, args, start_frame, vis_model, vis_data,
        trunk_id, args.steps, args.mask_mode, args.device,
    )
    wrap.close()

    teacher_xy = teacher_xy_raw - teacher_xy_raw[0]
    student_xy = student_xy_raw - student_xy_raw[0]

    all_xy = np.concatenate([teacher_xy, student_xy, ref_xy], axis=0)
    pad = 0.5
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
    xr, yr = xlim[1] - xlim[0], ylim[1] - ylim[0]
    if xr > yr:
        mid = (ylim[0] + ylim[1]) / 2
        ylim = (mid - xr / 2, mid + xr / 2)
    else:
        mid = (xlim[0] + xlim[1]) / 2
        xlim = (mid - yr / 2, mid + yr / 2)

    print("[render] Compositing frames ...")
    combined = []
    for i, (tf, sf) in enumerate(zip(teacher_frames, student_frames)):
        top = np.concatenate([
            add_label(tf, "Teacher"),
            add_label(sf, f"Student ({args.mask_mode})"),
        ], axis=1)  # [480, 1280, 3]
        panel = make_trajectory_panel(
            teacher_xy[:i+1], student_xy[:i+1], ref_xy,
            xlim, ylim, TRAJ_W, TRAJ_H,
        )
        combined.append(np.concatenate([top, panel], axis=0))  # [720, 1280, 3]

    if args.out and args.out.endswith(".mp4"):
        mp4_path = args.out
        Path(mp4_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = args.out or str(Path(args.teacher_checkpoint).parent / "videos")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        mp4_path = str(Path(out_dir) / f"k1_comparison_{clip_tag}_{args.mask_mode}.mp4")

    gif_path = mp4_path.replace(".mp4", ".gif")

    import mediapy as media
    media.write_video(mp4_path, combined, fps=50)
    print(f"[render] Video → {mp4_path}")

    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path,
        "-vf", "fps=20,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        "-loop", "0", gif_path,
    ], check=True, capture_output=True)
    print(f"[render] GIF  → {gif_path}")


if __name__ == "__main__":
    main()
