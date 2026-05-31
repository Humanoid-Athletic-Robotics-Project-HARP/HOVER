"""Simulate K1 zero-pose drop using MuJoCo Newton solver.

Drops the robot from standing height with all joints at zero.
Policy choices:
  zero   - zero torques (passive drop)
  random - uniform random torques within each actuator's ctrlrange each step

Uses MuJoCo's Newton constraint solver (already configured in k1.xml).

Usage:
    MUJOCO_GL=egl python3 scripts/simulate_zero_pose_mujoco_newton.py \
        --policy zero --max_steps 1000 --video /tmp/k1_newton_drop.mp4
    MUJOCO_GL=egl python3 scripts/simulate_zero_pose_mujoco_newton.py \
        --policy random --max_steps 1000 --video /tmp/k1_newton_random.mp4
"""
import argparse
import os
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str,
    default="neural_wbc/data/data/mujoco/models/scene_k1.xml")
parser.add_argument("--policy", type=str, default="zero", choices=["zero", "random"],
    help="zero: no torques; random: uniform random torques within ctrlrange each step")
parser.add_argument("--torque_scale", type=float, default=1.0,
    help="Scale random torques beyond ctrlrange (>1 disables clamping). Try 5 or 10.")
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--out", type=str, default="/tmp/k1_newton_drop.npz")
parser.add_argument("--video", type=str, default="/tmp/k1_newton_drop.mp4")
parser.add_argument("--init_z", type=float, default=0.61,
    help="Initial root z (meters). 0.61 matches RSI height in IsaacLab.")
args = parser.parse_args()

import mujoco

scene_path = os.path.abspath(args.scene)
model = mujoco.MjModel.from_xml_path(scene_path)
data  = mujoco.MjData(model)

solver_names = {0: "PGS", 1: "CG", 2: "Newton"}
print(f"[INFO] Loaded: {scene_path}")
print(f"[INFO] Solver: {solver_names.get(model.opt.solver, model.opt.solver)}")
print(f"[INFO] nq={model.nq}  nv={model.nv}  nu={model.nu}  dt={model.opt.timestep}")

# ── Set initial state: zero joints, standing height ───────────────────────────
mujoco.mj_resetData(model, data)
data.qpos[:3] = [0.0, 0.0, args.init_z]  # root xyz
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]   # root quat wxyz (identity)
data.qpos[7:] = 0.0                       # all joints = 0
data.qvel[:] = 0.0
data.ctrl[:] = 0.0                        # zero control = zero torques
mujoco.mj_forward(model, data)

print(f"[INFO] Policy: {args.policy}")
print(f"[INFO] Initial root_z={data.qpos[2]:.4f}  ncon={data.ncon}")

# Pre-compute ctrlrange for random policy
ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

# If torque_scale > 1, disable MuJoCo's built-in ctrl clamping so we can exceed limits
if args.torque_scale != 1.0:
    model.actuator_ctrllimited[:] = 0
    print(f"[INFO] torque_scale={args.torque_scale}x  (ctrl clamping disabled)")

# ── Renderer ──────────────────────────────────────────────────────────────────
renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance  = 3.5
cam.azimuth   = 90.0
cam.elevation = -15.0
cam.lookat[0] = 0.0
cam.lookat[1] = 0.0
cam.lookat[2] = 0.4

# k1.xml uses group=3 for collision geoms (group 3 is hidden by default).
# Enable it so the robot body is visible.
opt = mujoco.MjvOption()
opt.geomgroup[3] = 1

# ── Simulation loop ───────────────────────────────────────────────────────────
# timestep in k1.xml is 0.001 s; render every 5 steps → 200 Hz render / 0.005 s
RENDER_EVERY = 5
frames = []
joint_positions  = []
root_positions   = []
root_quaternions = []

for step in range(args.max_steps):
    if args.policy == "random":
        data.ctrl[:] = np.random.uniform(ctrl_lo, ctrl_hi) * args.torque_scale
    else:
        data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)

    joint_positions.append(data.qpos[7:].copy())
    root_positions.append(data.qpos[:3].copy())
    root_quaternions.append(data.qpos[3:7].copy())

    if step % RENDER_EVERY == 0:
        renderer.update_scene(data, camera=cam, scene_option=opt)
        frames.append(renderer.render().copy())

    if step % 50 == 0:
        print(f"  step {step:4d}  root_z={data.qpos[2]:.4f}  ncon={data.ncon}")

renderer.close()

joint_positions  = np.array(joint_positions)
root_positions   = np.array(root_positions)
root_quaternions = np.array(root_quaternions)

np.savez(args.out,
         joint_pos=joint_positions,
         root_pos=root_positions,
         root_quat=root_quaternions)

print(f"\nSaved {len(joint_positions)} physics steps to {args.out}")
print(f"  root_z range: {root_positions[:,2].min():.4f} -> {root_positions[:,2].max():.4f}")
print(f"  final root_z: {root_positions[-1,2]:.4f}")

# ── Save video ────────────────────────────────────────────────────────────────
print(f"Saving {len(frames)} frames to {args.video} ...")
try:
    import imageio
    fps = int(1.0 / (model.opt.timestep * RENDER_EVERY))
    writer = imageio.get_writer(args.video, fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"Saved video: {args.video}")
except Exception as e:
    out_gif = args.video.replace(".mp4", ".gif")
    import imageio
    fps = int(1.0 / (model.opt.timestep * RENDER_EVERY))
    imageio.mimsave(out_gif, frames, fps=fps)
    print(f"Saved gif: {out_gif}  (mp4 failed: {e})")
