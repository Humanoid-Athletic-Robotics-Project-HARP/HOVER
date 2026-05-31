#!/bin/bash
# Warmstarts v4 from v1's final checkpoint immediately.
set -e
cd /workspace/testing-grounds/projects/hover

V1_RUN_DIR="logs/mjlab/k1_tracking_teleop_no_jump/2026-05-25_05-46-36"
V1_CKPT="model_29999.pt"
V4_EXP="k1_tracking_teleop_no_jump_v4"
V1_RUN_NAME=$(basename "$V1_RUN_DIR")

mkdir -p "logs/rsl_rl/$V4_EXP"
ln -sfn "$(pwd)/$V1_RUN_DIR" "logs/rsl_rl/$V4_EXP/$V1_RUN_NAME"
echo "[watchdog_v4] Resume symlink: logs/rsl_rl/$V4_EXP/$V1_RUN_NAME"
echo "[watchdog_v4] Warmstarting from v1: $V1_CKPT"
echo "[watchdog_v4] Key changes: uniform sampling, root_pos weight 2.0, foot rewards, leg smoothness"

python3 scripts/train_k1_mjlab.py \
    --motion_file neural_wbc/data/data/mujoco/motions/k1_teleop_no_jump.npz \
    --num_envs 4096 \
    --max_iterations 50000 \
    --experiment_name "$V4_EXP" \
    --resume \
    --load_run "$V1_RUN_NAME" \
    --load_checkpoint "$V1_CKPT" \
    > /tmp/k1_train_teleop_no_jump_v4.log 2>&1

echo "[watchdog_v4] v4 training complete."
