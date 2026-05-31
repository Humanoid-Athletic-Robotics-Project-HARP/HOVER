"""Quick check: print robot root pose and joint positions after RSI reset."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="k1_aggressive")
parser.add_argument("--reference_motion_path", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_k1 import NeuralWBCEnvCfgK1, NeuralWBCEnvCfgK1Aggressive

if args_cli.robot == "k1_aggressive":
    env_cfg = NeuralWBCEnvCfgK1Aggressive()
else:
    env_cfg = NeuralWBCEnvCfgK1()

env_cfg.scene.num_envs = 4
env_cfg.scene.env_spacing = 20
env_cfg.terrain.env_spacing = 20
if args_cli.reference_motion_path:
    env_cfg.reference_motion_manager.motion_path = args_cli.reference_motion_path

env = NeuralWBCEnv(cfg=env_cfg, render_mode=None)
env.reset()

robot = env.scene["robot"]
root_pos = robot.data.root_pos_w        # [N, 3]
root_quat = robot.data.root_quat_w      # [N, 4]  (w, x, y, z)
joint_pos = robot.data.joint_pos        # [N, 22]

print("\n=== Root positions (x, y, z) ===")
print(root_pos.cpu().numpy())
print("\n=== Root quaternions (w, x, y, z) ===")
print(root_quat.cpu().numpy())
print("\n=== Joint positions (first env) ===")
print(joint_pos[0].cpu().numpy())

# z < 0.3 → robot is on the ground / sideways
print("\n=== Height check ===")
for i in range(root_pos.shape[0]):
    z = root_pos[i, 2].item()
    w = root_quat[i, 0].item()
    print(f"  env {i}: z={z:.3f}m, quat_w={w:.3f} {'UPRIGHT OK' if z > 0.3 and w > 0.7 else '<<< PROBLEM'}")

env.close()
simulation_app.close()
