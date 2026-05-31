#!/bin/bash
# End-to-end K1 training pipeline: retarget → train.
# Usage: bash train_k1.sh [--robot k1|k1_aggressive] [--num-envs N] [--max-iterations N] [--skip-retarget] [--resume] [--log FILE]
#
# Steps:
#   1. Kill any running teacher training
#   2. Retarget all clips in cmu_simple.yaml (unless --skip-retarget)
#   3. Launch teacher policy training (nohup, survives SSH disconnect)

set -e
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

K1_USD_PATH="${K1_USD_PATH:-$SCRIPT_DIR/third_party/booster_assets/robots/K1/K1_22dof.usd}"
MOTIONS_YAML="$SCRIPT_DIR/cmu_simple.yaml"
MOTIONS_DIR="$SCRIPT_DIR/neural_wbc/data/data/motions"
MOTIONS_PKL="$MOTIONS_DIR/cmu_simple.pkl"

NUM_ENVS=4096
MAX_ITER=50000
RESUME=0
ROBOT="k1"
TRAIN_LOG="/tmp/k1_train.log"
SKIP_RETARGET=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --num-envs)       NUM_ENVS="$2";  shift 2 ;;
        --max-iterations) MAX_ITER="$2";  shift 2 ;;
        --robot)          ROBOT="$2";     shift 2 ;;
        --log)            TRAIN_LOG="$2"; shift 2 ;;
        --skip-retarget)  SKIP_RETARGET=1; shift ;;
        --resume)         RESUME=1;       shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── 1. Kill existing training ────────────────────────────────────────────────
echo "=== [1/3] Stopping any existing training processes ==="
pkill -f "train_teacher_policy.py" || true
sleep 3

# ── 2. Retarget motions ──────────────────────────────────────────────────────
if [ "$SKIP_RETARGET" -eq 0 ]; then
    echo "=== [2/3] Retargeting $MOTIONS_YAML → $MOTIONS_PKL ==="
    mkdir -p "$MOTIONS_DIR"
    bash "$SCRIPT_DIR/retarget_k1.sh" \
        --motions-file "$MOTIONS_YAML" \
        --save-dir     "$MOTIONS_DIR"

    if [ ! -f "$MOTIONS_PKL" ]; then
        echo "ERROR: retargeting did not produce $MOTIONS_PKL"
        exit 1
    fi
    echo "Motion pkl ready: $(python3 -c "import joblib; d=joblib.load('$MOTIONS_PKL'); print(len(d), 'clips')")"
else
    echo "=== [2/3] Skipping retargeting (--skip-retarget) ==="
fi

# ── 3. Launch training ───────────────────────────────────────────────────────
echo "=== [3/3] Launching training: robot=$ROBOT envs=$NUM_ENVS log=$TRAIN_LOG ==="

RESUME_FLAG=""
if [ "$RESUME" -eq 1 ]; then
    RESUME_FLAG="--teacher_policy.resume"
fi

K1_USD_PATH="$K1_USD_PATH" \
nohup python3 scripts/rsl_rl/train_teacher_policy.py \
    --robot "$ROBOT" \
    --num_envs "$NUM_ENVS" \
    --headless \
    --reference_motion_path "$MOTIONS_PKL" \
    --teacher_policy.max_iterations "$MAX_ITER" \
    $RESUME_FLAG \
    > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
echo "Monitor:  tail -f $TRAIN_LOG"
echo "TensorBoard: tensorboard --logdir $SCRIPT_DIR/logs/teacher --port 6006"
