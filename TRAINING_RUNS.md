# K1 Teacher Training Runs

Motion file: `neural_wbc/data/data/mujoco/motions/k1_teleop_no_jump.npz`
Dataset: `cmu_teleop_no_jump.yaml` — 1000 clips, walking/running/kicking/punching, no jumps/acrobatics

---

## v1 — `k1_tracking_teleop_no_jump`

**Run:** `logs/mjlab/k1_tracking_teleop_no_jump/2026-05-25_05-46-36/`
**Iterations:** 30,000
**Final checkpoint:** `model_29999.pt`

### Config changes from base
| Setting | Value | Reason |
|---|---|---|
| `anchor_pos` termination | `bad_anchor_pos_z_only`, threshold=0.4m | Base 0.25m too strict for kicks/walking slopes |
| `motion_global_root_pos` weight | 1.0 (↑ from 0.5) | Drive XY drift correction |
| `motion_global_root_pos` std | 0.6 (↑ from 0.3) | Prevent gradient vanishing at ~0.5m XY error |
| `ee_body_pos` termination | removed | Foot Z exceeds 0.25m during walking |

### Results
- Robot walks and broadly tracks motions
- **Stumbles on clips 1, 5, 20; doesn't complete full trajectory**
- Diagnosis: no z-termination during original v0 training caused the robot to tolerate stumbles rather than prevent them; z_only at 0.4m partially helps but stumbling persists
- Episode length oscillated 60–120 (diverse dataset, clip difficulty variation)
- `error_anchor_pos` plateau ~0.44m (XY drift not fully corrected)

---

## v2 — `k1_tracking_teleop_no_jump_v2`

**Run:** `logs/mjlab/k1_tracking_teleop_no_jump_v2/2026-05-25_13-53-22/`
**Iterations:** ~23,500 (killed early)
**Warmstart:** from scratch

### Config changes from v1
| Setting | Value | Reason |
|---|---|---|
| `anchor_pos` threshold | 0.5m (↑ from 0.4m) | Reduce premature resets on dynamic clips |
| `motion_global_root_pos` weight | 1.5 (↑ from 1.0) | Harder push on XY correction |
| `entropy_coef` | 0.01 (↑ from 0.005) | Maintain diverse clip exploration |
| `num_steps_per_env` | 48 (↑ from 24) | More temporal context per PPO update |

### Results
- Stumbles less than v1 visually
- **Killed at 23.5k**: `sampling_top1_prob` hit 0.24 (sampler collapsed to one clip), `error_anchor_pos` 0.49m — worse than v1's 0.44m
- Root cause: adaptive sampler curriculum with `adaptive_uniform_ratio=0.1` not enough to prevent entropy collapse

---

## v3 — `k1_tracking_teleop_no_jump_v3`

**Run:** `logs/mjlab/k1_tracking_teleop_no_jump_v3/2026-05-26_01-57-21/`
**Iterations:** ~8,500 (killed early)
**Warmstart:** v1 `model_29999.pt`

### Config changes from v2
| Setting | Value | Reason |
|---|---|---|
| `motion_feet_pos` reward | weight=2.0, std=0.15, feet only | Feet diluted to 2/23 in body_pos average |
| `motion_feet_lin_vel` reward | weight=1.5, std=0.5, feet only | Match foot swing velocity |
| `leg_action_rate_l2` penalty | −0.5, legs only | Smooth legs; leave arms free for punches |

### Results
- `anchor_ori` dropped to 3.3 (vs 32 at start) — warmstart stability landed fast
- `sampling_entropy` dropped 0.54 → 0.31 in 4h, `sampling_top1_bin` hitting 0.68 — same collapse trajectory as v2
- **Root cause identified**: oscillating episode length across ALL runs is caused by the adaptive sampler's feedback loop — it oversamples hard bins → policy overtunes → sampler shifts → regresses → repeats
- Killed at 8.5k iters to fix the sampler before wasting more compute

---

## v4 — `k1_tracking_teleop_no_jump_v4`

**Run:** `logs/mjlab/k1_tracking_teleop_no_jump_v4/`
**Iterations:** 50,000
**Warmstart:** v1 `model_29999.pt`
**Status:** training

### Config changes from v3
| Setting | Value | Reason |
|---|---|---|
| `sampling_mode` | `"uniform"` (was `"adaptive"`) | Breaks the curriculum feedback loop — the root cause of oscillating episode length across all prior runs |
| `motion_global_root_pos` weight | 2.0 (↑ from 1.5) | Fix XY drift; foot rewards (total 3.5) were outweighing root tracking |

### All active rewards
| Reward | Weight | Notes |
|---|---|---|
| `motion_global_root_pos` | 2.0 | std=0.6 |
| `motion_global_root_ori` | 0.5 | default |
| `motion_body_pos` | 1.0 | all 23 bodies |
| `motion_body_ori` | 1.0 | all 23 bodies |
| `motion_body_lin_vel` | 1.0 | all 23 bodies |
| `motion_body_ang_vel` | 1.0 | all 23 bodies |
| `motion_feet_pos` | 2.0 | feet only, std=0.15 |
| `motion_feet_lin_vel` | 1.5 | feet only, std=0.5 |
| `action_rate_l2` | −0.1 | all joints |
| `leg_action_rate_l2` | −0.5 | legs only |
| `joint_limit` | −10.0 | |
| `self_collisions` | −10.0 | |

### Watchdog
`tmux attach -t watchdog_v4` to monitor.
Log: `/tmp/k1_train_teleop_no_jump_v4.log`

### Results
- Episode length peaked at 466 (~35k iters) — best run so far
- **Catastrophic collapse at ~45k**: action_std 0.5→3.16, value_loss→4732, episode_len→8
- Root cause: foot reward std=0.15 creates sharp gradients; PPO adaptive LR amplified them until value function exploded
- Best checkpoint: `model_45000.pt` (last stable iter before collapse)
- Arms still not tracking punch clips — body_pos dilutes arm gradient (8/23 bodies share equal weight)

---

## v5 — `k1_tracking_teleop_no_jump_v5`

**Run:** `logs/mjlab/k1_tracking_teleop_no_jump_v5/2026-05-27_07-49-00/`
**Iterations:** 80,000
**Warmstart:** v1 `model_29999.pt` (switched from v4 — cleaner base, no collapse history)
**Status:** complete

### Config changes from v4
| Setting | Value | Reason |
|---|---|---|
| `motion_body_pos` | removed | Replaced by upper/lower split (HOVER design) |
| `motion_upper_body_pos` | weight=3.0, std=0.05, upper 10 bodies | Arms get dedicated high-weight term; tight sigma drives punch fidelity |
| `motion_lower_body_pos` | weight=1.5, std=0.5, lower 12 bodies | Matches HOVER's lower body sigma exactly |
| `motion_keypoints` | weight=4.0, std=0.05, Head_2+hands | VR keypoints — highest priority targets (HOVER: weight=50, sigma=0.03) |
| `action_rate_l2` | removed | Replaced by per-half split |
| `upper_body_action_rate_l2` | −0.625, head+arm joints | HOVER exact |
| `leg_action_rate_l2` | −3.0 (↑ from −0.5) | HOVER exact — tight leg smoothness |
| `penalize_stumble` | −5.0, foot contact sensor | Lateral > 5x vertical foot force |
| `penalize_slippage` | −2.0, foot contact sensor | Foot velocity during ground contact |
| `penalize_feet_air_time` | +1.0, min_air_time=0.25s | Reward correct foot lift duration |
| `desired_kl` | 0.008 (↓ from 0.01) | Tighter KL cap to prevent repeat of v4 PPO collapse |
| foot contact sensor | `track_air_time=True` | Required for air_time reward |

### All active rewards
| Reward | Weight | Notes |
|---|---|---|
| `motion_global_root_pos` | 2.0 | std=0.6 |
| `motion_global_root_ori` | 0.5 | default |
| `motion_upper_body_pos` | 3.0 | std=0.05, head+arms (10 bodies) |
| `motion_lower_body_pos` | 1.5 | std=0.5, legs (12 bodies) |
| `motion_keypoints` | 4.0 | std=0.05, Head_2+hands |
| `motion_body_ori` | 1.0 | all 23 bodies |
| `motion_body_lin_vel` | 1.0 | all 23 bodies |
| `motion_body_ang_vel` | 1.0 | all 23 bodies |
| `motion_feet_pos` | 2.0 | std=0.15, feet only |
| `motion_feet_lin_vel` | 1.5 | std=0.5, feet only |
| `upper_body_action_rate_l2` | −0.625 | head+arm joints |
| `leg_action_rate_l2` | −3.0 | leg joints |
| `penalize_stumble` | −5.0 | foot contact sensor |
| `penalize_slippage` | −2.0 | foot contact sensor |
| `penalize_feet_air_time` | +1.0 | min_air_time=0.25s |
| `joint_limit` | −10.0 | |
| `self_collisions` | −10.0 | |

### Results
- `anchor_ori` terminations stuck at 11–14 for the entire second half — robot fell sideways constantly
- Episode length peaked at ~391 early then collapsed to ~172, finished at 221
- `error_anchor_rot=1.23` never improved — rotational instability was the bottleneck
- `motion_keypoints=1.32` — head/hands tracking actually worked; arm tracking improved over v4
- **Root cause**: tight upper body constraints (std=0.05, w=3/4) pulled trunk into unstable poses; `leg_action_rate_l2=-3.0` made legs too stiff to recover — both were HOVER H1 values not tuned for K1 cold-start
- Renders showed robot falling within 0.25s on every test clip

---

## v6 — `k1_tracking_teleop_no_jump_v6`

**Run:** `logs/mjlab/k1_tracking_teleop_no_jump_v6/`
**Iterations:** 50,000
**Warmstart:** none (cold-start)
**Status:** training

### Config changes from v5
| Setting | Value | Reason |
|---|---|---|
| Warmstart | none | v5 proved warmstarting a destabilised policy makes things worse |
| `motion_upper_body_pos` std/weight | 0.1 / 2.0 (was 0.05 / 3.0) | Relaxed so locomotion can emerge before arm tracking tightens |
| `motion_keypoints` std/weight | 0.1 / 2.0 (was 0.05 / 4.0) | Prevents trunk being pulled into unstable poses early |
| `leg_action_rate_l2` | −0.5 (was −3.0) | v5 lesson: −3.0 made legs too stiff to balance from cold-start |
| foot contact penalties | removed | Noise before gait exists; add back once locomotion is stable |

### All active rewards
| Reward | Weight | Notes |
|---|---|---|
| `motion_global_root_pos` | 2.0 | std=0.6 |
| `motion_global_root_ori` | 0.5 | default |
| `motion_upper_body_pos` | 2.0 | std=0.1, head+arms (10 bodies) |
| `motion_lower_body_pos` | 1.5 | std=0.5, legs (12 bodies) |
| `motion_keypoints` | 2.0 | std=0.1, Head_2+hands |
| `motion_body_ori` | 1.0 | all 23 bodies |
| `motion_body_lin_vel` | 1.0 | all 23 bodies |
| `motion_body_ang_vel` | 1.0 | all 23 bodies |
| `motion_feet_pos` | 2.0 | std=0.15, feet only |
| `motion_feet_lin_vel` | 1.5 | std=0.5, feet only |
| `upper_body_action_rate_l2` | −0.625 | head+arm joints |
| `leg_action_rate_l2` | −0.5 | leg joints |
| `joint_limit` | −10.0 | |
| `self_collisions` | −10.0 | |

### Watchdog
`tmux attach -t watchdog_v6` to monitor.
Log: `/tmp/k1_train_teleop_no_jump_v6.log`
