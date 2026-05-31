"""Launch K1 motion tracking training with mjlab.

Usage:
    python3 scripts/train_k1_mjlab.py
    python3 scripts/train_k1_mjlab.py --num_envs 4096
    python3 scripts/train_k1_mjlab.py --num_envs 4096 --motion_file /path/to/motion.npz
"""

import argparse
import os
import sys
from pathlib import Path

# Allow importing from scripts/mjlab_k1/
sys.path.insert(0, str(Path(__file__).parent))

MOTION_FILE_DEFAULT = (
    Path(__file__).parent.parent
    / "neural_wbc/data/data/mujoco/motions/k1_cmu_simple.npz"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument(
        "--motion_file", type=str, default=str(MOTION_FILE_DEFAULT),
        help="Path to the motion npz file",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_iterations", type=int, default=30_000)
    parser.add_argument("--experiment_name", type=str, default="k1_tracking")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in load_run")
    parser.add_argument("--load_run", type=str, default=".*", help="Regex to match run dir to resume from")
    parser.add_argument("--load_checkpoint", type=str, default="model_.*.pt", help="Regex to match checkpoint file")
    args = parser.parse_args()

    motion_file = str(Path(args.motion_file).resolve())
    if not Path(motion_file).exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"  # after CUDA_VISIBLE_DEVICES remapping
    os.environ["WANDB_MODE"] = "disabled"  # no W&B account needed

    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
    from mjlab.scripts.train import launch_training, TrainConfig

    from mjlab_k1.env_cfg import k1_flat_tracking_env_cfg
    from mjlab_k1.rl_cfg import k1_tracking_ppo_runner_cfg

    TASK_ID = "Mjlab-Tracking-Flat-K1"

    env_cfg = k1_flat_tracking_env_cfg(motion_file=motion_file)
    play_env_cfg = k1_flat_tracking_env_cfg(motion_file=motion_file, play=True)
    rl_cfg = k1_tracking_ppo_runner_cfg()

    env_cfg.scene.num_envs = args.num_envs
    play_env_cfg.scene.num_envs = 1
    rl_cfg.max_iterations = args.max_iterations
    rl_cfg.experiment_name = args.experiment_name
    rl_cfg.resume = args.resume
    rl_cfg.load_run = args.load_run
    rl_cfg.load_checkpoint = args.load_checkpoint

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=rl_cfg,
        runner_cls=MotionTrackingOnPolicyRunner,
    )

    train_cfg = TrainConfig(
        env=env_cfg,
        agent=rl_cfg,
        video=args.video,
        gpu_ids=[0],
    )

    launch_training(task_id=TASK_ID, args=train_cfg)


if __name__ == "__main__":
    main()
