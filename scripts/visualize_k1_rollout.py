"""
Visualize K1 mjlab policy rollout: joint positions + root orientation.

Loads the latest checkpoint, runs one episode in play mode (env=1),
collects reference vs actual joint positions and root orientation, then
plots them as a multi-panel figure saved to PNG and written to TensorBoard.

Usage:
    python3 scripts/visualize_k1_rollout.py
    python3 scripts/visualize_k1_rollout.py --checkpoint logs/rsl_rl/k1_tracking/2026-05-22_20-16-53/model_1500.pt
    python3 scripts/visualize_k1_rollout.py --steps 300
"""

import argparse
import os
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("WANDB_MODE", "disabled")

MOTION_FILE = str(
    Path(__file__).parent.parent
    / "neural_wbc/data/data/mujoco/motions/k1_cmu_simple.npz"
)
JOINT_NAMES = [
    "Head_yaw", "Head_pitch",
    "L_Shl_Pitch", "L_Shl_Roll", "L_Elb_Pitch", "L_Elb_Yaw",
    "R_Shl_Pitch", "R_Shl_Roll", "R_Elb_Pitch", "R_Elb_Yaw",
    "L_Hip_Pitch", "L_Hip_Roll", "L_Hip_Yaw", "L_Knee",
    "L_Ank_Pitch", "L_Ank_Roll",
    "R_Hip_Pitch", "R_Hip_Roll", "R_Hip_Yaw", "R_Knee",
    "R_Ank_Pitch", "R_Ank_Roll",
]


def quat_wxyz_to_euler(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion → roll/pitch/yaw (radians)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin( np.clip(2*(w*y - z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.stack([roll, pitch, yaw], axis=-1)


def find_latest_checkpoint(log_root: str = "logs/rsl_rl/k1_tracking") -> str:
    runs = sorted(Path(log_root).glob("*/model_*.pt"))
    if not runs:
        raise FileNotFoundError(f"No checkpoints found under {log_root}")
    # Pick highest iteration across all runs
    def iteration(p):
        try: return int(p.stem.split("_")[1])
        except: return 0
    best = max(runs, key=iteration)
    print(f"[viz] Using checkpoint: {best}")
    return str(best)


def collect_rollout(checkpoint: str, n_steps: int, device: str):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.tracking.mdp.commands import MotionCommand
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
    from dataclasses import asdict

    from mjlab_k1.env_cfg import k1_flat_tracking_env_cfg
    from mjlab_k1.rl_cfg import k1_tracking_ppo_runner_cfg

    TASK_ID = "Mjlab-Tracking-Flat-K1-Viz"
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

    env = ManagerBasedRlEnv(cfg=play_env_cfg, device=device)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    runner = MotionTrackingOnPolicyRunner(env_wrapped, asdict(rl_cfg), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    ref_joint_pos_list  = []
    act_joint_pos_list  = []
    ref_root_quat_list  = []
    act_root_quat_list  = []
    ref_root_pos_list   = []
    act_root_pos_list   = []

    obs, _ = env_wrapped.reset()

    with torch.no_grad():
        for _ in range(n_steps):
            action = policy(obs)
            obs, _, _, _ = env_wrapped.step(action)

            motion_cmd = cast(
                MotionCommand,
                env.command_manager.get_term("motion"),
            )

            ref_joint_pos_list.append(motion_cmd.joint_pos[0].cpu().numpy())
            act_joint_pos_list.append(motion_cmd.robot_joint_pos[0].cpu().numpy())
            ref_root_quat_list.append(motion_cmd.anchor_quat_w[0].cpu().numpy())
            act_root_quat_list.append(motion_cmd.robot_anchor_quat_w[0].cpu().numpy())
            ref_root_pos_list.append(motion_cmd.anchor_pos_w[0].cpu().numpy())
            act_root_pos_list.append(motion_cmd.robot_anchor_pos_w[0].cpu().numpy())

    env_wrapped.close()

    def stack(lst): return np.stack(lst, axis=0)
    return {
        "ref_joint_pos":  stack(ref_joint_pos_list),   # [T, 22]
        "act_joint_pos":  stack(act_joint_pos_list),
        "ref_root_quat":  stack(ref_root_quat_list),   # [T, 4]  wxyz
        "act_root_quat":  stack(act_root_quat_list),
        "ref_root_pos":   stack(ref_root_pos_list),    # [T, 3]
        "act_root_pos":   stack(act_root_pos_list),
    }


def make_figure(data: dict, dt: float, out_path: str):
    T = data["ref_joint_pos"].shape[0]
    t = np.arange(T) * dt

    ref_euler = quat_wxyz_to_euler(data["ref_root_quat"])  # [T, 3]
    act_euler = quat_wxyz_to_euler(data["act_root_quat"])

    n_joints = 22
    # Layout: top block = root (pos 3 + ori 3), bottom block = joints (22)
    fig = plt.figure(figsize=(28, 32))
    fig.patch.set_facecolor("#1a1a2e")

    # ── title ─────────────────────────────────────────────────────────────────
    ckpt_iters = out_path.split("model_")[-1].replace(".png", "") if "model_" in out_path else "?"
    fig.suptitle(
        f"K1 mjlab rollout — checkpoint iter {ckpt_iters}   ({T} steps @ {1/dt:.0f} Hz)",
        color="white", fontsize=16, y=0.99,
    )

    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35,
                              height_ratios=[1, 3.2])

    # ── root block ────────────────────────────────────────────────────────────
    root_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0],
                                               hspace=0.5, wspace=0.35)
    root_titles = ["Pos X (m)", "Pos Y (m)", "Pos Z (m)",
                   "Roll (rad)", "Pitch (rad)", "Yaw (rad)"]
    root_ref = np.concatenate([data["ref_root_pos"], ref_euler], axis=1)
    root_act = np.concatenate([data["act_root_pos"], act_euler], axis=1)

    for i, title in enumerate(root_titles):
        ax = fig.add_subplot(root_gs[i // 3, i % 3])
        _style_ax(ax, title)
        ax.plot(t, root_ref[:, i], color="#4fc3f7", lw=1.2, label="reference", alpha=0.85)
        ax.plot(t, root_act[:, i], color="#ef9a9a", lw=1.0, label="actual",    alpha=0.85)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7,
                      facecolor="#2a2a3e", labelcolor="white", framealpha=0.7)

    # ── joint block ───────────────────────────────────────────────────────────
    cols = 4
    rows = (n_joints + cols - 1) // cols   # 6 rows × 4 cols = 24 cells for 22 joints
    joint_gs = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=outer[1],
                                                hspace=0.55, wspace=0.35)
    for j in range(n_joints):
        ax = fig.add_subplot(joint_gs[j // cols, j % cols])
        _style_ax(ax, JOINT_NAMES[j])
        ax.plot(t, data["ref_joint_pos"][:, j], color="#4fc3f7", lw=1.2, alpha=0.85)
        ax.plot(t, data["act_joint_pos"][:, j], color="#ef9a9a", lw=1.0, alpha=0.85)
        ax.axhline(0, color="#555577", lw=0.5, ls=":")

    # hide unused cells
    for j in range(n_joints, rows * cols):
        fig.add_subplot(joint_gs[j // cols, j % cols]).set_visible(False)

    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved → {out_path}")


def _style_ax(ax, title: str):
    ax.set_facecolor("#12122a")
    ax.tick_params(colors="#aaaacc", labelsize=6.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.set_title(title, color="#ccccee", fontsize=8, pad=3)
    ax.set_xlabel("time (s)", color="#888899", fontsize=6)
    ax.grid(color="#222244", lw=0.4)


def write_to_tensorboard(out_path: str, tb_logdir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
        from PIL import Image
        img = np.array(Image.open(out_path)).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)[:3]  # CHW, drop alpha
        writer = SummaryWriter(log_dir=tb_logdir)
        writer.add_image("rollout/joint_pos_and_root_ori", img_tensor, global_step=0)
        writer.close()
        print(f"[viz] Image written to TensorBoard at {tb_logdir}")
    except Exception as e:
        print(f"[viz] TensorBoard write skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model_*.pt; defaults to latest in logs/rsl_rl/k1_tracking")
    parser.add_argument("--steps", type=int, default=250,
                        help="Number of simulation steps to roll out")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path; defaults to logs/rsl_rl/k1_tracking/<run>/rollout_viz.png")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    checkpoint = args.checkpoint or find_latest_checkpoint()
    out_path = args.out or str(Path(checkpoint).parent / "rollout_viz.png")
    tb_logdir = str(Path(checkpoint).parent)

    print(f"[viz] Rolling out {args.steps} steps on {args.device} ...")
    data = collect_rollout(checkpoint, args.steps, args.device)

    dt = 0.02  # 50 Hz (mjlab: timestep=0.005, decimation=4)
    make_figure(data, dt, out_path)
    write_to_tensorboard(out_path, tb_logdir)


if __name__ == "__main__":
    main()
