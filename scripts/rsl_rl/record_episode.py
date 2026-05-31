"""Record N episodes of K1 policy inference and save joint/root states to .npz."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="k1_aggressive", choices=["k1", "k1_aggressive"])
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_XXXX.pt checkpoint")
parser.add_argument("--reference_motion_path", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--out", type=str, default="/tmp/k1_episode.npz")
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from vecenv_wrapper import RslRlNeuralWBCVecEnvWrapper
from utils import get_customized_rsl_rl, get_ppo_runner_and_checkpoint_path
get_customized_rsl_rl()

from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_k1 import NeuralWBCEnvCfgK1, NeuralWBCEnvCfgK1Aggressive
from neural_wbc.core.modes import NeuralWBCModes

if args_cli.robot == "k1_aggressive":
    env_cfg = NeuralWBCEnvCfgK1Aggressive(mode=NeuralWBCModes.TEST)
else:
    env_cfg = NeuralWBCEnvCfgK1(mode=NeuralWBCModes.TEST)

env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.scene.env_spacing = 20
env_cfg.terrain.env_spacing = 20
if args_cli.reference_motion_path:
    env_cfg.reference_motion_manager.motion_path = args_cli.reference_motion_path

env = NeuralWBCEnv(cfg=env_cfg, render_mode=None)
wrapped_env = RslRlNeuralWBCVecEnvWrapper(env)

# Load policy — read config from checkpoint directory then construct runner
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from teacher_policy_cfg import TeacherPolicyCfg

checkpoint_path = args_cli.checkpoint
resume_path = os.path.dirname(checkpoint_path)
checkpoint_file = os.path.basename(checkpoint_path)

teacher_policy_cfg = TeacherPolicyCfg()
teacher_policy_cfg.runner.resume_path = resume_path
teacher_policy_cfg.runner.checkpoint = checkpoint_file
teacher_policy_cfg.overwrite_policy_cfg_from_file(os.path.join(resume_path, "config.json"))

runner = OnPolicyRunner(wrapped_env, teacher_policy_cfg.to_dict(), log_dir=None, device=wrapped_env.device)
runner.load(checkpoint_path)
policy = runner.get_inference_policy(device=wrapped_env.device)
print(f"[INFO] Loaded checkpoint: {checkpoint_path}")

# Record
robot = env.scene["robot"]
out_base = args_cli.out  # e.g. /tmp/k1_episode.npz
out_stem = out_base.replace(".npz", "")

obs = wrapped_env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]

for ep in range(args_cli.num_episodes):
    joint_positions = []
    root_positions = []
    root_quaternions = []

    # Reset env for episodes after the first
    if ep > 0:
        obs, _ = wrapped_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

    for step in range(args_cli.max_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, _, rewards, dones, _ = wrapped_env.step(actions)

        joint_positions.append(robot.data.joint_pos[0].cpu().numpy())
        root_positions.append(robot.data.root_pos_w[0].cpu().numpy())
        root_quaternions.append(robot.data.root_quat_w[0].cpu().numpy())

        if dones[0]:
            print(f"[ep {ep}] Episode ended at step {step + 1}")
            break

    joint_positions = np.array(joint_positions)
    root_positions = np.array(root_positions)
    root_quaternions = np.array(root_quaternions)

    out_path = f"{out_stem}_{ep}.npz" if args_cli.num_episodes > 1 else out_base
    np.savez(out_path,
             joint_pos=joint_positions,
             root_pos=root_positions,
             root_quat=root_quaternions)

    print(f"Saved {len(joint_positions)} frames to {out_path}")
    print(f"  joint_pos: {joint_positions.shape}")

env.close()
simulation_app.close()
