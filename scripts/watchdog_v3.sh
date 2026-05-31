#!/bin/bash
# Warmstarts v3 from v1's final checkpoint immediately.
set -e
cd /workspace/testing-grounds/projects/hover

V1_RUN_DIR="logs/mjlab/k1_tracking_teleop_no_jump/2026-05-25_05-46-36"
V1_CKPT="model_29999.pt"
V3_EXP="k1_tracking_teleop_no_jump_v3"
V1_RUN_NAME=$(basename "$V1_RUN_DIR")

# Pre-create v3 experiment dir and symlink v1's run inside it so RSL-RL's
# get_checkpoint_path can resolve --load_run across experiment boundaries.
mkdir -p "logs/rsl_rl/$V3_EXP"
ln -sfn "$(pwd)/$V1_RUN_DIR" "logs/rsl_rl/$V3_EXP/$V1_RUN_NAME"
echo "[watchdog_v3] Resume symlink: logs/rsl_rl/$V3_EXP/$V1_RUN_NAME -> $V1_RUN_DIR"
echo "[watchdog_v3] Warmstarting from v1: $V1_CKPT"

echo "[watchdog_v3] Starting v3 training (50k iters, warmstart from v1) ..."
python3 scripts/train_k1_mjlab.py \
    --motion_file neural_wbc/data/data/mujoco/motions/k1_teleop_no_jump.npz \
    --num_envs 4096 \
    --max_iterations 50000 \
    --experiment_name "$V3_EXP" \
    --resume \
    --load_run "$V1_RUN_NAME" \
    --load_checkpoint "$V1_CKPT" \
    > /tmp/k1_train_teleop_no_jump_v3.log 2>&1

echo "[watchdog_v3] v3 training complete."
