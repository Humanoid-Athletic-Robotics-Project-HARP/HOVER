#!/bin/bash
# Render teacher vs student comparison for K1 v6.
set -e
cd /workspace/testing-grounds/projects/hover

TEACHER="logs/mjlab/k1_tracking_teleop_no_jump_v6/2026-05-29_03-28-29/model_49999.pt"
STUDENT="logs/mjlab/k1_tracking_teleop_no_jump_v6/2026-05-29_03-28-29/student_all_k1_student_v6/final_model.pt"
MOTION="neural_wbc/data/data/mujoco/motions/k1_teleop_no_jump.npz"
OUT_DIR="logs/mjlab/comparison_v6/videos"

mkdir -p "$OUT_DIR"

for CLIP in 1 5 20; do
    for MASK in full locomotion upper_body; do
        OUT="$OUT_DIR/clip_${CLIP}_${MASK}.mp4"
        echo "--- clip $CLIP mask $MASK ---"
        python3 scripts/render_k1_comparison.py \
            --teacher_checkpoint "$TEACHER" \
            --student_checkpoint "$STUDENT" \
            --motion_file "$MOTION" \
            --clip "$CLIP" \
            --steps 300 \
            --mask_mode "$MASK" \
            --out "$OUT"
        echo "--- done: $OUT ---"
    done
done

echo "[watchdog_render_comparison] All done."
