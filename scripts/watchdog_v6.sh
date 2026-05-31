#!/bin/bash
# Cold-start v6: HOVER upper/lower split with cold-start-safe weights.
set -e
cd /workspace/testing-grounds/projects/hover

V6_EXP="k1_tracking_teleop_no_jump_v6"
echo "[watchdog_v6] Cold-starting v6 from scratch"
echo "[watchdog_v6] Key changes vs v5:"
echo "  - Cold-start (no warmstart)"
echo "  - upper_body_pos: std=0.1 w=2.0 (was 0.05/3.0)"
echo "  - motion_keypoints: std=0.1 w=2.0 (was 0.05/4.0)"
echo "  - leg_action_rate_l2: -0.5 (was -3.0)"
echo "  - foot contact penalties removed (noise before gait emerges)"

python3 scripts/train_k1_mjlab.py \
    --motion_file neural_wbc/data/data/mujoco/motions/k1_teleop_no_jump.npz \
    --num_envs 4096 \
    --max_iterations 50000 \
    --experiment_name "$V6_EXP" \
    > /tmp/k1_train_teleop_no_jump_v6.log 2>&1

echo "[watchdog_v6] v6 training complete."
