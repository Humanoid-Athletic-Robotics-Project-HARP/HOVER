"""Simulate K1 from an initial state under gravity with full collision physics.

Steps the physics sim directly (bypasses episode termination). Can start from
zero pose or from frame 0 of a recorded episode (--init_from_episode).

Usage:
    K1_USD_PATH=.../K1_22dof.usd python3 scripts/rsl_rl/simulate_zero_pose.py \
        --out /tmp/k1_zero_pose_sim.npz --max_steps 200
    K1_USD_PATH=.../K1_22dof.usd python3 scripts/rsl_rl/simulate_zero_pose.py \
        --init_from_episode /tmp/k1_ep_0.npz --out /tmp/k1_rsi_drop.npz
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="/tmp/k1_zero_pose_sim.npz")
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--init_from_episode", type=str, default=None,
                    help="If set, use frame 0 of this .npz as initial state instead of zero pose")
parser.add_argument("--robot", type=str, default="k1_aggressive",
                    choices=["k1", "k1_aggressive"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, sys, torch, numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import get_customized_rsl_rl
get_customized_rsl_rl()

from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_k1 import NeuralWBCEnvCfgK1, NeuralWBCEnvCfgK1Aggressive
from neural_wbc.core.modes import NeuralWBCModes

if args_cli.robot == "k1_aggressive":
    env_cfg = NeuralWBCEnvCfgK1Aggressive(mode=NeuralWBCModes.TEST)
else:
    env_cfg = NeuralWBCEnvCfgK1(mode=NeuralWBCModes.TEST)

env_cfg.scene.num_envs = 1
env_cfg.scene.env_spacing = 20
env_cfg.terrain.env_spacing = 20

# ── Physics tuning: make PhysX behave closer to MuJoCo ────────────────────
# bounce_threshold_velocity=0.2 has known weird behaviour (NVIDIA forums);
# lower it to near-zero so slow contacts never bounce.
env_cfg.sim.physx.bounce_threshold_velocity = 0.01

# TGS solver: apply external forces every position iteration for accurate
# velocity updates (especially important for contact-rich locomotion).
env_cfg.sim.physx.enable_external_forces_every_iteration = True

# Tighten friction anchor merging distance (default 0.04 → 0.01).
env_cfg.sim.physx.friction_offset_threshold = 0.01

# More solver iterations per articulation: 8→16 position, 1→4 velocity.
env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = 16
env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = 4

env = NeuralWBCEnv(cfg=env_cfg, render_mode=None)
robot = env.scene["robot"]
device = env.device

# One reset to fully initialize the scene
env.reset()

# ── Set initial pose ───────────────────────────────────────────────────────
root_pose = torch.zeros(1, 7, device=device)
root_vel  = torch.zeros(1, 6, device=device)
joint_pos = torch.zeros(1, robot.num_joints, device=device)
joint_vel = torch.zeros(1, robot.num_joints, device=device)

if args_cli.init_from_episode:
    ep = np.load(args_cli.init_from_episode)
    root_pose[0, :3] = torch.tensor(ep["root_pos"][0], device=device)
    root_pose[0, 3:] = torch.tensor(ep["root_quat"][0], device=device)  # wxyz
    joint_pos[0, :]  = torch.tensor(ep["joint_pos"][0], device=device)
    print(f"[INFO] Init from episode frame 0: root_z={ep['root_pos'][0,2]:.4f}")
else:
    root_pose[0, 2] = 0.61   # z — RSI standing height
    root_pose[0, 3] = 1.0    # quat w (wxyz identity)
    print("[INFO] Init: zero pose at z=0.95")

robot.write_root_pose_to_sim(root_pose)
robot.write_root_velocity_to_sim(root_vel)
robot.write_joint_state_to_sim(joint_pos, joint_vel)
env.sim.step(render=False)
robot.update(dt=env_cfg.sim.dt)

print(f"[INFO] root_z after apply={robot.data.root_pos_w[0,2].item():.4f}")

# ── Step physics directly — no termination resets ─────────────────────────
# Set joint position targets to zero so the PD controller holds zero
# (zero effort but still valid targets). This avoids wild actuator forces.
zero_pos_targets = torch.zeros(1, robot.num_joints, device=device)

joint_positions  = []
root_positions   = []
root_quaternions = []

for step in range(args_cli.max_steps):
    # Write zero position targets to actuators each step
    robot.set_joint_position_target(zero_pos_targets)
    robot.write_data_to_sim()

    # Step the physics — this runs collision detection and contacts
    env.sim.step(render=False)
    robot.update(dt=env_cfg.sim.dt)

    joint_positions.append(robot.data.joint_pos[0].cpu().numpy())
    root_positions.append(robot.data.root_pos_w[0].cpu().numpy())
    root_quaternions.append(robot.data.root_quat_w[0].cpu().numpy())

    rz = robot.data.root_pos_w[0, 2].item()
    if step % 20 == 0:
        print(f"  step {step:4d}  root_z={rz:.4f}")

joint_positions  = np.array(joint_positions)
root_positions   = np.array(root_positions)
root_quaternions = np.array(root_quaternions)

np.savez(args_cli.out,
         joint_pos=joint_positions,
         root_pos=root_positions,
         root_quat=root_quaternions)

print(f"\nSaved {len(joint_positions)} frames to {args_cli.out}")
print(f"  root_z range: {root_positions[:,2].min():.4f} -> {root_positions[:,2].max():.4f}")
print(f"  final root_z: {root_positions[-1,2]:.4f}")

env.close()
simulation_app.close()
