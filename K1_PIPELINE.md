# K1 Pipeline: Retargeting → Training → Evaluation → Rendering

## Environment variable (required for all Isaac Lab steps)

```bash
export K1_USD_PATH=/workspace/testing-grounds/projects/hover/third_party/booster_assets/robots/K1/K1_22dof.usd
```

---

## 0. Prerequisites (one-time setup)

Compute K1-specific SMPL body shape and limb scale factors:

```bash
cd /workspace/testing-grounds/projects/hover
python3 scripts/data_process/grad_fit_k1_shape.py
# output: third_party/human2humanoid/data/k1/shape_optimized_v1.pkl
```

---

## 1. Retargeting (AMASS motion → K1 .pkl)

Single clip:

```bash
python3 scripts/data_process/grad_fit_k1.py \
  --amass_root third_party/human2humanoid/data/AMASS/AMASS_Complete \
  --amass_file CMU/CMU/01/01_01_poses.npz \
  --save_path neural_wbc/data/data/motions/my_motion_k1.pkl
```

Batch via YAML file listing clips:

```bash
python3 scripts/data_process/grad_fit_k1.py \
  --amass_root third_party/human2humanoid/data/AMASS/AMASS_Complete \
  --motions_file path/to/motions.yaml \
  --save_path neural_wbc/data/data/motions/my_batch_k1.pkl
```

---

## 2. Inspect retargeted motion (optional)

```bash
python3 scripts/rsl_rl/view_reference_motion.py \
  --reference_motion_path neural_wbc/data/data/motions/my_motion_k1.pkl \
  --clip_index 0
```

---

## 3. Train teacher policy

```bash
K1_USD_PATH=$K1_USD_PATH \
python3 scripts/rsl_rl/train_teacher_policy.py \
  --robot k1 \
  --num_envs 4096 \
  --headless \
  --reference_motion_path neural_wbc/data/data/motions/my_motion_k1.pkl
# checkpoints → logs/teacher/<timestamp>/model_*.pt
# tensorboard  → logs/teacher/<timestamp>/events.out.tfevents.*
```

Monitor training:

```bash
tensorboard --logdir logs/teacher/
```

---

## 4. Play / visualize teacher policy

Headless, no video (GPU physics works, fastest):

```bash
K1_USD_PATH=$K1_USD_PATH \
python3 scripts/rsl_rl/play.py \
  --robot k1 \
  --num_envs 4 \
  --headless \
  --no_video \
  --teacher_policy.resume_path logs/teacher/<timestamp> \
  --teacher_policy.checkpoint model_5000.pt
```

Deterministic replay from t=0 (TEST mode, not random phase):

```bash
K1_USD_PATH=$K1_USD_PATH \
python3 scripts/rsl_rl/play.py \
  --robot k1 \
  --num_envs 4 \
  --headless \
  --no_video \
  --no_randomize \
  --teacher_policy.resume_path logs/teacher/<timestamp> \
  --teacher_policy.checkpoint model_5000.pt
```

---

## 5. Evaluate teacher policy (metrics)

```bash
K1_USD_PATH=$K1_USD_PATH \
python3 scripts/rsl_rl/eval.py \
  --robot k1 \
  --num_envs 4 \
  --headless \
  --no_randomize \
  --metrics_path logs/teacher/<timestamp>/eval_metrics.json \
  --teacher_policy.resume_path logs/teacher/<timestamp> \
  --teacher_policy.checkpoint model_5000.pt
```

---

## 6. Train student policy (distillation)

```bash
K1_USD_PATH=$K1_USD_PATH \
python3 scripts/rsl_rl/train_student_policy.py \
  --robot k1 \
  --num_envs 4096 \
  --headless \
  --teacher_policy.resume_path logs/teacher/<timestamp> \
  --teacher_policy.checkpoint model_5000.pt
# checkpoints → logs/student/<timestamp>/
```

---

## 7. Export student policy to TorchScript

```bash
python3 scripts/export_student_torchscript.py \
  --student_path logs/student/<timestamp> \
  --student_checkpoint model_50000.pt \
  --output k1_hover_student.pt
```

---

## 8. Render video

> **Note:** Requires Vulkan. In headless docker environments this falls back to CPU physics and may be unusable. Use `--no_video` (step 4) for evaluation instead.

```bash
K1_USD_PATH=$K1_USD_PATH \
python3 scripts/rsl_rl/play.py \
  --robot k1 \
  --num_envs 4 \
  --headless \
  --video \
  --video_length 500 \
  --teacher_policy.resume_path logs/teacher/<timestamp> \
  --teacher_policy.checkpoint model_5000.pt
# videos → logs/teacher/<timestamp>/videos/play/
```

---

## Known gaps (steps needing k1 wired up)

| Script | Status |
|---|---|
| `train_teacher_policy.py` | K1 fully supported |
| `play.py` | K1 fully supported |
| `eval.py` | needs `k1` added to `--robot` choices |
| `train_student_policy.py` | needs `k1` added to `--robot` choices and player wired |
| `export_student_torchscript.py` | needs `k1` added to `--robot` choices |
| `view_reference_motion.py` | H1 only (robot arg not used for motion viewing) |
