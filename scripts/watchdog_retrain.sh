#!/bin/bash
# Watchdog: waits for current training PID to finish, applies tweaks, starts v2.
set -e
cd /workspace/testing-grounds/projects/hover

TRAIN_PID=${1:-387474}

echo "[watchdog] Monitoring PID $TRAIN_PID ..."
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
done
echo "[watchdog] Training finished. Applying tweaks ..."

python3 - << 'PYEOF'
import re

# --- env_cfg.py ---
with open('scripts/mjlab_k1/env_cfg.py') as f:
    cfg = f.read()

# Z threshold 0.4 -> 0.5
cfg = cfg.replace('"threshold": 0.4', '"threshold": 0.5')

# Root pos reward weight 1.0 -> 1.5
cfg = cfg.replace(
    'weight=1.0,\n        params={"command_name": "motion", "std": 0.6}',
    'weight=1.5,\n        params={"command_name": "motion", "std": 0.6}',
)

with open('scripts/mjlab_k1/env_cfg.py', 'w') as f:
    f.write(cfg)
print("[watchdog] env_cfg.py: z_threshold 0.4→0.5, root_pos weight 1.0→1.5")

# --- rl_cfg.py ---
with open('scripts/mjlab_k1/rl_cfg.py') as f:
    rl = f.read()

rl = rl.replace('entropy_coef=0.005', 'entropy_coef=0.01')
rl = rl.replace('num_steps_per_env=24', 'num_steps_per_env=48')

with open('scripts/mjlab_k1/rl_cfg.py', 'w') as f:
    f.write(rl)
print("[watchdog] rl_cfg.py: entropy_coef 0.005→0.01, num_steps_per_env 24→48")
PYEOF

echo "[watchdog] Tweaks applied. Starting v2 training (50k iters) ..."
python3 scripts/train_k1_mjlab.py \
    --motion_file neural_wbc/data/data/mujoco/motions/k1_teleop_no_jump.npz \
    --num_envs 4096 \
    --max_iterations 50000 \
    --experiment_name k1_tracking_teleop_no_jump_v2 \
    > /tmp/k1_train_teleop_no_jump_v2.log 2>&1

echo "[watchdog] v2 training complete."
