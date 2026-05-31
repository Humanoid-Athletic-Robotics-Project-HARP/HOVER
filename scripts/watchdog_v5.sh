#!/bin/bash
# Warmstarts v5 from v4's best checkpoint (model_45000.pt — last good iter before collapse).
set -e
cd /workspace/testing-grounds/projects/hover

V1_RUN_DIR="logs/mjlab/k1_tracking_teleop_no_jump/2026-05-25_05-46-36"
V1_CKPT="model_29999.pt"
V5_EXP="k1_tracking_teleop_no_jump_v5"
V1_RUN_NAME=$(basename "$V1_RUN_DIR")

mkdir -p "logs/rsl_rl/$V5_EXP"
ln -sfn "$(pwd)/$V1_RUN_DIR" "logs/rsl_rl/$V5_EXP/$V1_RUN_NAME"
echo "[watchdog_v5] Resume symlink: logs/rsl_rl/$V5_EXP/$V1_RUN_NAME"
echo "[watchdog_v5] Warmstarting from v1 (clean stable base): $V1_CKPT"
echo "[watchdog_v5] Key changes:"
echo "  - HOVER upper/lower body pos split (upper sigma=0.05 w=3.0, lower sigma=0.5 w=1.5)"
echo "  - VR keypoints: Head_2 + hands (sigma=0.05, w=4.0)"
echo "  - Action smoothness: upper=-0.625, leg=-3.0 (HOVER exact)"
echo "  - Foot contact penalties: stumble, slippage, air_time"
echo "  - desired_kl=0.008 (vs 0.01 in v4) to prevent PPO collapse"

python3 scripts/train_k1_mjlab.py \
    --motion_file neural_wbc/data/data/mujoco/motions/k1_teleop_no_jump.npz \
    --num_envs 4096 \
    --max_iterations 50000 \
    --experiment_name "$V5_EXP" \
    --resume \
    --load_run "$V1_RUN_NAME" \
    --load_checkpoint "$V1_CKPT" \
    > /tmp/k1_train_teleop_no_jump_v5.log 2>&1

echo "[watchdog_v5] v5 training complete."
