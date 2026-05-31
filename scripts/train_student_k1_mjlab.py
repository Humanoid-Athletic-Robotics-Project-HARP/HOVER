"""K1 student policy distillation with mjlab.

Loads a trained teacher PPO checkpoint and distils it into a student policy that
only observes proprioceptive + masked reference-motion state (no privileged state,
no velocity sensor state estimation).

Usage:
    python3 scripts/train_student_k1_mjlab.py --teacher_checkpoint logs/rsl_rl/k1_tracking/.../model_22000.pt
    python3 scripts/train_student_k1_mjlab.py \\
        --teacher_checkpoint logs/.../model_22000.pt \\
        --mask_mode locomotion \\
        --num_envs 1024 \\
        --max_iterations 50000
"""

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("WANDB_MODE", "disabled")

MOTION_FILE_DEFAULT = str(
    Path(__file__).parent.parent
    / "neural_wbc/data/data/mujoco/motions/k1_cmu_simple.npz"
)

VALID_MASK_MODES = ["locomotion", "upper_body", "full", "all"]


def find_latest_checkpoint(log_root: str = "logs/rsl_rl/k1_tracking") -> str:
    runs = sorted(Path(log_root).glob("*/model_*.pt"))
    if not runs:
        raise FileNotFoundError(f"No checkpoints found under {log_root}")
    def _iter(p):
        try: return int(p.stem.split("_")[1])
        except: return 0
    best = max(runs, key=_iter)
    print(f"[distill] Using teacher checkpoint: {best}")
    return str(best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Path to teacher PPO checkpoint (.pt). Auto-detects latest if omitted.")
    parser.add_argument("--motion_file", type=str, default=MOTION_FILE_DEFAULT)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=50_000)
    parser.add_argument("--save_iteration", type=int, default=500)
    parser.add_argument("--mask_mode", type=str, default="all",
                        choices=VALID_MASK_MODES,
                        help="Which mask mode(s) to train with. 'all' uses all three modes.")
    parser.add_argument("--experiment_name", type=str, default="k1_student")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sparsity_randomization", action="store_true",
                        help="Enable random mask sparsity during training.")
    args = parser.parse_args()

    teacher_ckpt = args.teacher_checkpoint or find_latest_checkpoint()

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    from mjlab_k1.env_cfg import k1_flat_tracking_env_cfg
    from mjlab_k1.rl_cfg import k1_tracking_ppo_runner_cfg
    from mjlab_k1.k1_student_obs import K1_DISTILL_MASK_MODES
    from mjlab_k1.k1_teacher_policy import K1TeacherPolicy
    from mjlab_k1.student_env_wrapper import K1StudentEnvWrapper

    from neural_wbc.student_policy.student_policy_trainer import StudentPolicyTrainer
    from neural_wbc.student_policy.student_policy_trainer_cfg import StudentPolicyTrainerCfg

    motion_file = str(Path(args.motion_file).resolve())
    if not Path(motion_file).exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    # --- Build env ---
    TASK_ID = "Mjlab-Tracking-Flat-K1-Distill"
    env_cfg = k1_flat_tracking_env_cfg(motion_file=motion_file)
    play_env_cfg = k1_flat_tracking_env_cfg(motion_file=motion_file, play=True)
    rl_cfg = k1_tracking_ppo_runner_cfg()

    env_cfg.scene.num_envs = args.num_envs
    play_env_cfg.scene.num_envs = 1
    # Disable obs noise for distillation (clean teacher labels).
    env_cfg.observations["actor"].enable_corruption = False

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=rl_cfg,
        runner_cls=MotionTrackingOnPolicyRunner,
    )

    print(f"[distill] Creating env with {args.num_envs} envs ...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    # --- Load teacher ---
    print(f"[distill] Loading teacher from {teacher_ckpt} ...")
    runner = MotionTrackingOnPolicyRunner(env_wrapped, asdict(rl_cfg), device=args.device)
    runner.load(teacher_ckpt, load_cfg={"actor": True}, strict=True, map_location=args.device)
    inference_fn = runner.get_inference_policy(device=args.device)
    teacher = K1TeacherPolicy(inference_fn, teacher_ckpt)

    # --- Mask modes ---
    if args.mask_mode == "all":
        mask_modes = K1_DISTILL_MASK_MODES
    else:
        mask_modes = {args.mask_mode: K1_DISTILL_MASK_MODES[args.mask_mode]}

    # --- Wrap env ---
    wrapped_env = K1StudentEnvWrapper(
        env, env_wrapped,
        mask_modes=mask_modes,
        enable_sparsity_randomization=args.sparsity_randomization,
    )

    # --- Output path ---
    teacher_dir = Path(teacher_ckpt).parent
    ckpt_iter = Path(teacher_ckpt).stem.split("_")[-1]
    out_dir = teacher_dir / f"student_{args.mask_mode}_{args.experiment_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[distill] Saving student checkpoints to: {out_dir}")

    # --- Student obs/action dims ---
    from mjlab_k1.k1_student_obs import NUM_STUDENT_OBS, NUM_TEACHER_OBS
    NUM_ACTIONS = 22

    cfg = StudentPolicyTrainerCfg(
        teacher_policy=teacher,
        student_policy_path=str(out_dir),
        num_policy_obs=NUM_TEACHER_OBS,
        num_student_obs=NUM_STUDENT_OBS,
        num_actions=NUM_ACTIONS,
        max_iteration=args.max_iterations,
        save_iteration=args.save_iteration,
        num_steps_per_env=24,   # ~0.5s at 50Hz per env
        student_rollout_iteration=5000,
        learning_rate=5e-4,
        num_mini_batches=4,
        num_learning_epochs=2,
    )

    trainer = StudentPolicyTrainer(wrapped_env, cfg)
    print(f"[distill] Starting distillation: {args.max_iterations} iterations ...")
    trainer.run()
    print(f"[distill] Done. Final model saved to {out_dir}/final_model.pt")


if __name__ == "__main__":
    main()
