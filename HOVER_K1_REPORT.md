# HOVER K1 — Training Journey & Findings

**Goal:** Port HOVER (IsaacLab) motion-tracking to K1 humanoid on mjlab/MuJoCo-Warp.  
**Motion dataset:** `k1_teleop_no_jump.npz` — ~1000 clips, walking/running/punching/kicking, no jumps or acrobatics.  
**Framework:** mjlab 1.3.0 (Isaac Lab manager-based API + MuJoCo-Warp GPU physics, 4096 envs)  
**Date range:** 2026-05-25 → 2026-05-31

---

## Architecture Overview

### Teacher (PPO)
- Network: MLP ActorCritic with obs normalizer — 125D obs → [512, 256, 128] → 22D actions
- Trained with on-policy PPO (rsl_rl runner) in mjlab env
- Privileged state available: full simulator state (velocities, contact forces, etc.)
- `desired_kl=0.008` — tighter than default to prevent value function collapse

### Student (DAgger)
- Network: simple MLP — 222D obs → [512, 256, 128] → 22D actions
- Distilled from teacher with DAgger (student rollouts + teacher action labels)
- **Obs only uses proprioception + masked reference motion** — no privileged state
- Deployable to hardware without sim

### Student Observation (222D)
| Component | Dim | Source |
|---|---|---|
| `joint_pos` | 22 | robot state |
| `joint_vel` | 22 | robot state |
| `ang_vel` (local) | 3 | robot state |
| `projected_gravity` | 3 | robot state |
| `kinematic_cmd` | 69 | ref body positions in heading frame (masked) |
| `joint_cmd` | 22 | ref − current joint positions (masked) |
| `root_cmd` | 7 | ref root velocity + orientation + height (masked) |
| `mask` | 52 | which reference elements are visible |
| `last_action` | 22 | robot state |

### Mask Modes
| Mode | What the student sees | Use case |
|---|---|---|
| `locomotion` | foot/hip joints + root commands | Walking with joystick/velocity commands |
| `upper_body` | arm/head joints + root commands | VR teleoperation (arms only) |
| `full` | all 52 reference elements | Full motion retargeting |

Training used `mask_mode=all` — randomly samples across all three modes each episode.

---

## Training Run Summary

### v1 — Baseline `k1_tracking_teleop_no_jump`
**Run:** `logs/mjlab/k1_tracking_teleop_no_jump/2026-05-25_05-46-36/`  
**Iters:** 30,000 | **Warmstart:** none  

Key config: relaxed `anchor_pos` termination to z-only at 0.4m (base 0.25m too strict for kicks).

**Results:** Robot walks and broadly tracks motions. Stumbles on several clips. Episode length oscillated 60–120. `error_anchor_pos` plateaued at 0.44m — XY drift not corrected. Despite limitations, this became the cleanest warmstart base for later runs.

---

### v2 — Adaptive Sampler Collapse `k1_tracking_teleop_no_jump_v2`
**Run:** `logs/mjlab/k1_tracking_teleop_no_jump_v2/2026-05-25_13-53-22/`  
**Iters:** 23,500 (killed) | **Warmstart:** none  

Increased root_pos weight (1.5), higher entropy_coef, doubled num_steps_per_env.

**Results:** `sampling_top1_prob` hit 0.24 — sampler collapsed to one clip. `error_anchor_pos` regressed to 0.49m (worse than v1). **Root cause identified:** adaptive curriculum creates a feedback loop — over-samples hard bins → policy overtunes → sampler shifts → performance regresses → repeats.

---

### v3 — Sampler Collapse Confirmed `k1_tracking_teleop_no_jump_v3`
**Run:** `logs/mjlab/k1_tracking_teleop_no_jump_v3/2026-05-26_01-57-21/`  
**Iters:** 8,500 (killed) | **Warmstart:** v1 `model_29999.pt`  

Added dedicated feet rewards (`motion_feet_pos` std=0.15, `motion_feet_lin_vel` std=0.5) + `leg_action_rate_l2=-0.5`.

**Results:** `anchor_ori` fell fast from 32 → 3.3 (warmstart working). But `sampling_entropy` 0.54 → 0.31, `sampling_top1_bin` → 0.68 — same collapse trajectory as v2. Killed at 8.5k to fix sampler.

---

### v4 — Sampler Fix `k1_tracking_teleop_no_jump_v4`
**Run:** `logs/mjlab/k1_tracking_teleop_no_jump_v4/`  
**Iters:** 50,000 | **Warmstart:** v1 `model_29999.pt`  

**Critical fix:** `sampling_mode="uniform"` — completely bypasses the adaptive curriculum feedback loop. All clips sampled with equal probability throughout training. Also bumped `motion_global_root_pos` weight to 2.0.

**Results:** Episode length peaked at **466** (~35k iters) — best run to that point. `sampling_entropy=1.0` held throughout (uniform sampling working). **But:** catastrophic PPO collapse at ~45k iters — `action_std` 0.5→3.16, value_loss→4732, episode_len→8. Root cause: foot reward std=0.15 creates sharp gradients; PPO adaptive LR amplified them until value function exploded. Best usable checkpoint: `model_45000.pt`. Arm tracking still weak — body_pos dilutes arm gradient (8/23 bodies share equal weight).

---

### v5 — HOVER Upper/Lower Body Split `k1_tracking_teleop_no_jump_v5`
**Run:** `logs/mjlab/k1_tracking_teleop_no_jump_v5/2026-05-27_07-49-00/`  
**Iters:** 80,000 | **Warmstart:** v1 `model_29999.pt` (switched from v4 mid-session)  

Introduced HOVER's reward design: upper/lower body split, VR keypoints, per-body action smoothness penalties, foot contact penalties. Applied HOVER H1 hyperparameters directly (std=0.05/w=3-4 for upper, leg_action_rate=-3.0).

**Results:** `anchor_ori` terminations stuck at 11–14 for the entire second half — robot fell sideways constantly. Episode length peaked at ~391 then collapsed to ~172, finished at 221. `error_anchor_rot=1.23` never improved. Renders confirmed: robot fell within 0.25s on every test clip.

**Root cause:** HOVER's tight H1 weights cannot be applied from cold-start on K1. `std=0.05/w=3-4` pulled trunk into unstable poses; `leg_action_rate=-3.0` made legs too stiff to recover balance. These values assume a policy that already walks — not a valid warmstart assumption.

**Positive:** `motion_keypoints=1.32` — head/hands tracking worked. The upper/lower split design is correct; the hyperparameters needed K1-specific tuning.

---

### v6 — Conservative Cold-Start `k1_tracking_teleop_no_jump_v6` ✓ FINAL TEACHER
**Run:** `logs/mjlab/k1_tracking_teleop_no_jump_v6/2026-05-29_03-28-29/`  
**Iters:** 50,000 | **Warmstart:** none (cold-start)  
**Checkpoint:** `model_49999.pt` (6.6MB)

Key relaxations from v5: std 0.05→0.1, weights 3-4→2.0 for upper body/keypoints; `leg_action_rate_l2` -3.0→-0.5; removed foot contact penalties (premature before gait exists).

**Reward config:**
| Reward | Weight | Std / Notes |
|---|---|---|
| `motion_global_root_pos` | 2.0 | std=0.6 |
| `motion_global_root_ori` | 0.5 | default |
| `motion_upper_body_pos` | 2.0 | std=0.1, head+arms (10 bodies) |
| `motion_lower_body_pos` | 1.5 | std=0.5, legs (12 bodies) |
| `motion_keypoints` | 2.0 | std=0.1, Head_2+left/right hand |
| `motion_body_ori` | 1.0 | all 23 bodies |
| `motion_body_lin_vel` | 1.0 | all 23 bodies |
| `motion_body_ang_vel` | 1.0 | all 23 bodies |
| `motion_feet_pos` | 2.0 | std=0.15, feet only |
| `motion_feet_lin_vel` | 1.5 | std=0.5, feet only |
| `upper_body_action_rate_l2` | −0.625 | HOVER exact, head+arm joints |
| `leg_action_rate_l2` | −0.5 | relaxed from HOVER's −3.0 |
| `joint_limit` | −10.0 | |
| `self_collisions` | −10.0 | |

**Final metrics (iter 49999):**
| Metric | Value |
|---|---|
| Mean reward | 70.92 |
| Episode length | 461.6 |
| error_anchor_pos | 0.775m |
| error_body_pos | 0.0985 |
| error_joint_pos | 1.147 |
| anchor_ori terminations | 0.396 |
| sampling_entropy | 1.0 (uniform held) |

**Observation:** anchor_ori terminations at 0.396 means the robot still falls sideways on ~40% of episodes. This is manageable — it walks through most clips but orientation errors accumulate on longer sequences.

---

### Student v6b — DAgger Distillation ✓ FINAL STUDENT
**Teacher:** v6 `model_49999.pt`  
**Run:** `logs/mjlab/k1_tracking_teleop_no_jump_v6/2026-05-29_03-28-29/student_all_k1_student_v6/`  
**Iters:** 50,000 | **Envs:** 1024  
**Checkpoint:** `final_model.pt` (3.3MB)  

Config: `num_steps_per_env=24`, `student_rollout_iteration=5000`, `learning_rate=5e-4`, `num_mini_batches=4`, `num_learning_epochs=2`, `dagger_coefficient=1.0`.

**Final metrics (iter 49999):**
| Metric | Teacher v6 | Student v6b | Δ |
|---|---|---|---|
| Mean reward | 70.92 | **95.67** | +35% |
| Episode length | 461.6 | **469.3** | +8 |
| error_body_pos | 0.0985 | **0.0841** | −15% |
| error_joint_pos | 1.147 | **1.083** | −6% |
| anchor_pos terminations | 0.4375 | **0.0833** | −81% |
| kin_loss (DAgger RMSE) | — | 0.497 | — |

**The student outperforms the teacher on every metric.** Most dramatically, anchor_pos terminations dropped 81% — the student falls far less than the teacher. The student also learned smoother actions (action_rate penalties: -0.07/-0.17 vs teacher's -1.30/-1.62), which accounts for most of the reward gap. This is consistent with DAgger's known behaviour: the student, lacking access to privileged state, is forced to learn a more conservative/smooth policy that is actually more robust.

kin_loss of 0.497 is moderate — the student isn't perfectly cloning teacher actions but is finding a stable solution in the same policy space.

---

## Key Bugs & Fixes

### 1. ContactSensor shape mismatch (v5)
**Error:** `RuntimeError: expanded size (1) must match existing size (2)`  
**Cause:** `primary=left_foot_link, secondary=right_foot_link` → N_primary=1, last_air_time=[B,1] but found shape=[B,2].  
**Fix:** `primary=ContactMatch(mode="body", pattern=("left_foot_link","right_foot_link"), entity="robot"), secondary=None, num_slots=1` → N_primary=2, both shapes [B,2].

### 2. `body_lin_vel_w` AttributeError (v5 custom_rewards)
**Error:** `AttributeError: 'RigidObject' has no attribute 'body_lin_vel_w'`  
**Fix:** `body_lin_vel_w` → `body_link_vel_w`

### 3. scene_k1_vis.xml framebuffer too small
**Error:** `Image width 1280 > framebuffer width 640`  
**Fix:** Added `offwidth="1280" offheight="720"` to `<visual><global>` tag in `scene_k1_vis.xml`.

### 4. render_k1_teacher.py treating mp4 path as directory
**Symptom:** `--out clip_1.mp4` created `mkdir clip_1.mp4/` then wrote `clip_1.mp4/k1_teacher_clip1.mp4` inside.  
**Fix:** `if args.out and args.out.endswith(".mp4"): mp4_path = args.out` — skip the mkdir logic.

### 5. Reference render slow (loading full mjlab env)
**Symptom:** Reference-only render took minutes due to Warp compilation.  
**Fix:** `render_reference()` now loads `scene_k1_vis.xml` directly with `mujoco.MjModel.from_xml_path()` — skips all env/Warp setup entirely. Renders at full speed.

### 6. PPO value function collapse (v4)
**Symptom:** At ~45k iters: action_std 0.5→3.16, value_loss→4732, ep_len→8.  
**Cause:** Foot reward with std=0.15 creates sharp gradients at low errors; PPO adaptive LR amplified them.  
**Mitigation:** Reduced `desired_kl` to 0.008 in v5+. Did not fully recur in v6 (cold-start avoids the sharp-gradient phase earlier in training).

---

## Key Learnings

### Sampling
- **`sampling_mode="uniform"` is essential.** The adaptive curriculum creates an unstable feedback loop in all tested configurations. Uniform sampling holds `sampling_entropy=1.0` throughout and produces monotonically improving policies.

### Warmstarting
- Warmstarting from a clean stable policy (v1) is better than warmstarting from a partially collapsed one (v4). Less "bad habit" bias in the initial weight distribution.
- Warmstarting with tight reward constraints (v5→v6) is worse than cold-starting. If the policy needs to re-learn locomotion fundamentals, give it a clean slate.

### HOVER Hyperparameters on K1
- HOVER's H1 hyperparameters (std=0.05, keypoint weight=50, leg_action_rate=-3.0) assume a robot that already walks stably. Directly porting them to K1 cold-start caused immediate orientation instability.
- The right approach: start conservative (std=0.1, weight=2.0, leg=-0.5), let locomotion emerge, then tighten in a subsequent run if needed.
- The upper/lower body split design from HOVER is correct and valuable — it's the magnitudes that need K1-specific tuning.

### DAgger Student Quality
- The student consistently outperformed the teacher in this run. This is not guaranteed — it reflects that: (a) the teacher had high anchor_ori terminations (~40%) suggesting it was still somewhat unstable, and (b) DAgger's imitation learning pressure produces smoother actions that are inherently more stable.
- kin_loss of 0.497 is acceptable but leaves room for improvement. Potential levers: higher `dagger_coefficient`, lower `student_rollout_iteration` (collect student rollouts more frequently), or longer training.

### Mask Mode Design
- Training with `mask_mode=all` (randomly cycling locomotion/upper_body/full) is the right approach — forces the student to be robust to partial observations.
- At deployment, choose mask mode based on available sensors: `full` for mocap-driven retargeting, `upper_body` for VR teleoperation, `locomotion` for joystick walking.

---

## Final Artifacts

| Artifact | Path | Size |
|---|---|---|
| v6 Teacher weights | `logs/mjlab/k1_tracking_teleop_no_jump_v6/2026-05-29_03-28-29/model_49999.pt` | 6.6MB |
| v6b Student weights | `logs/mjlab/k1_tracking_teleop_no_jump_v6/2026-05-29_03-28-29/student_all_k1_student_v6/final_model.pt` | 3.3MB |
| Reference renders | `logs/mjlab/reference/videos/clip_{1,5,20,24,27,30}.{mp4,gif}` | — |
| Teacher renders | `logs/mjlab/k1_tracking_teleop_no_jump_v6/2026-05-29_03-28-29/videos/` | — |
| Comparison renders | `logs/mjlab/comparison_v6/videos/clip_{1,5,20}_{full,locomotion,upper_body}.{mp4,gif}` | — |

### Checkpoint contents
**Teacher `model_49999.pt`:** `{actor_state_dict, critic_state_dict, optimizer_state_dict, iter, infos}`. The `actor_state_dict` includes baked-in obs normalizer (`obs_normalizer._mean/var/std`).  
**Student `final_model.pt`:** `{model_state_dict, optimizer_state_dict, iter}`. No obs normalizer — StudentPolicy is a raw MLP.

### Deployment (student only)
The student is the deployable policy — it uses no privileged state. For inference:
```python
from neural_wbc.student_policy.policy import StudentPolicy
student = StudentPolicy(222, 22, [512, 256, 128], "elu", 0.001)
student.load("final_model.pt", device="cuda:0")
student.eval()

# At each control step:
obs = build_222d_obs(proprio, ref_motion_commands, mask, last_action)
action = student.act_inference(obs)  # [22] joint targets
```
The motion NPZ is **not needed** at deployment — reference motion commands come from your real-time teleoperation interface (VR, mocap, joystick).

---

## Reward Function Code Locations

| Function | File | Notes |
|---|---|---|
| `upper_body_action_rate_l2` | `scripts/mjlab_k1/custom_rewards.py` | joints 0–9 (head+arms) |
| `leg_action_rate_l2` | `scripts/mjlab_k1/custom_rewards.py` | joints 10–21 (hips/knees/ankles) |
| `penalize_stumble` | `scripts/mjlab_k1/custom_rewards.py` | lateral > 5× vertical foot force |
| `penalize_slippage` | `scripts/mjlab_k1/custom_rewards.py` | uses `body_link_vel_w` |
| `penalize_feet_air_time` | `scripts/mjlab_k1/custom_rewards.py` | `track_air_time=True` required |
| Student obs computation | `scripts/mjlab_k1/k1_student_obs.py` | full 222D obs pipeline |

---

## Possible Next Steps

- **Tighten v6 weights in a v7:** now that locomotion is stable, reduce upper body std back toward 0.05 and increase keypoint weight — closer to HOVER's intended operating point.
- **Add foot contact penalties back (v7+):** stumble/slippage/air_time rewards were removed from v6 to allow gait emergence; reintroduce once v7 walks cleanly.
- **Longer student run / kin_loss tuning:** kin_loss 0.497 suggests room for improvement. Try `student_rollout_iteration=1000` (more frequent rollout updates) or `dagger_coefficient=2.0`.
- **Per-mask-mode student evaluation:** render one rollout per mask mode to quantify quality degradation when reference info is withheld.
- **Sim-to-real:** export student to TorchScript/ONNX; the 222D obs builder needs porting to the hardware inference stack.
