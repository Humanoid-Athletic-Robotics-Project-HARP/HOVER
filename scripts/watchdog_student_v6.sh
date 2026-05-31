#!/bin/bash
# Distils v6 teacher (model_49999.pt) into a student policy.
set -e
cd /workspace/testing-grounds/projects/hover

V6_CKPT="logs/mjlab/k1_tracking_teleop_no_jump_v6/2026-05-29_03-28-29/model_49999.pt"

echo "[watchdog_student_v6] Distilling from: $V6_CKPT"
echo "[watchdog_student_v6] mask_mode=all (locomotion + upper_body + full)"
echo "[watchdog_student_v6] 50k iterations, 1024 envs"

python3 scripts/train_student_k1_mjlab.py \
    --teacher_checkpoint "$V6_CKPT" \
    --motion_file neural_wbc/data/data/mujoco/motions/k1_teleop_no_jump.npz \
    --num_envs 1024 \
    --max_iterations 50000 \
    --mask_mode all \
    --experiment_name k1_student_v6 \
    > /tmp/k1_student_v6.log 2>&1

echo "[watchdog_student_v6] Distillation complete."
