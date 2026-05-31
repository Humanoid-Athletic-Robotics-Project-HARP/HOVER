#!/bin/bash
# Wait for both retargeting jobs to finish, merge pkls, convert to npz, start training.
set -e
cd /workspace/testing-grounds/projects/hover

ALL_PKL="data/k1/cmu_all.pkl"
COMBINED_PKL="neural_wbc/data/data/motions/k1_all.pkl"
COMBINED_NPZ="neural_wbc/data/data/mujoco/motions/k1_all.npz"

echo "[pipeline] Waiting for retargeting to finish (data/k1/cmu_all.pkl)..."
while [ ! -f "$ALL_PKL" ]; do
    sleep 60
done
# give a moment for file flush
sleep 10

echo "[pipeline] Copying $ALL_PKL → $COMBINED_PKL"
cp "$ALL_PKL" "$COMBINED_PKL"
python3 -c "
import joblib
d = joblib.load('$COMBINED_PKL')
print(f'  Total clips: {len(d)}')
"

echo "[pipeline] Converting pkl → npz ..."
python3 scripts/convert_k1_motion.py \
    --input  "$COMBINED_PKL" \
    --output "$COMBINED_NPZ"

echo "[pipeline] Starting training on all motions ..."
MUJOCO_GL=egl WANDB_MODE=disabled \
python3 scripts/train_k1_mjlab.py \
    --motion_file "$COMBINED_NPZ" \
    --num_envs 4096 \
    --max_iterations 30000 \
    --experiment_name k1_tracking_all \
    > /tmp/k1_train_all.log 2>&1 &
echo "[pipeline] Training PID: $!"
echo "[pipeline] Log: /tmp/k1_train_all.log"
