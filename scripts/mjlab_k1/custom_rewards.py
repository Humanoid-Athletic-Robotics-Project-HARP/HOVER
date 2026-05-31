"""Custom reward functions for K1 tracking.

Reward design follows HOVER (NVlabs/HOVER) adapted for Unitree K1:
  - Upper/lower body position tracked with separate sigmas (HOVER: 0.03 upper, 0.5 lower)
  - Action smoothness split by body half (HOVER: -3.0 lower, -0.625 upper)
  - Foot contact quality penalties (stumble, slippage, air time)

K1 actuator ordering in K1_ARTICULATION:
  [0-1]  HEAD       : AAHead_yaw, Head_pitch
  [2-9]  ARMS       : ALeft_Shoulder_Pitch, Left_Shoulder_Roll, Left_Elbow_Pitch,
                      Left_Elbow_Yaw, ARight_Shoulder_Pitch, Right_Shoulder_Roll,
                      Right_Elbow_Pitch, Right_Elbow_Yaw
  [10-11] HIP_PITCH : Left_Hip_Pitch, Right_Hip_Pitch
  [12-15] HIP_ROLL_YAW: Left_Hip_Roll, Left_Hip_Yaw, Right_Hip_Roll, Right_Hip_Yaw
  [16-17] KNEES     : Left_Knee_Pitch, Right_Knee_Pitch
  [18-21] ANKLES    : Left_Ankle_Pitch, Left_Ankle_Roll, Right_Ankle_Pitch, Right_Ankle_Roll
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_UPPER_INDICES = list(range(0, 10))   # head + arms
_LEG_INDICES   = list(range(10, 22))  # hips, knees, ankles


def upper_body_action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize action rate for head+arm joints only.

    Mirrors HOVER's penalize_upper_body_action_changes (-0.625).
    Arms need to move for punches/kicks, so this is deliberately light.
    """
    action = env.action_manager.action[:, _UPPER_INDICES]
    prev   = env.action_manager.prev_action[:, _UPPER_INDICES]
    return torch.sum(torch.square(action - prev), dim=1)


def leg_action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize action rate for leg joints only (hips, knees, ankles).

    Mirrors HOVER's penalize_lower_body_action_changes (-3.0).
    Tight smoothness on legs is essential for stable walking.
    """
    action = env.action_manager.action[:, _LEG_INDICES]
    prev   = env.action_manager.prev_action[:, _LEG_INDICES]
    return torch.sum(torch.square(action - prev), dim=1)


def penalize_stumble(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Penalize foot sliding contact (lateral force > 5x vertical).

    Mirrors HOVER's penalize_stumble. Fires when foot skids sideways on
    landing — a direct indicator of poor foot placement.
    K1 foot geoms: left_foot_1_collision, right_foot_1_collision.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    forces = sensor.data.force  # [B, N, 3]  (x, y, z)
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    lateral = torch.norm(forces[..., :2], dim=-1)   # [B, N]
    vertical = torch.abs(forces[..., 2])             # [B, N]
    return torch.any(lateral > 5.0 * vertical, dim=-1).float()


def penalize_slippage(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    foot_body_names: tuple[str, ...],
) -> torch.Tensor:
    """Penalize foot velocity while in ground contact (slipping).

    Mirrors HOVER's penalize_slippage.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    forces = sensor.data.force  # [B, N, 3]
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    foot_ids = [robot.find_bodies(n)[0][0] for n in foot_body_names]
    foot_vel = env.scene["robot"].data.body_link_vel_w[:, foot_ids, :]  # [B, 2, 3]
    foot_speed = torch.norm(foot_vel, dim=-1)          # [B, 2]
    in_contact = (torch.norm(forces, dim=-1) > 1.0)    # [B, N]
    return torch.sum(foot_speed * in_contact.float(), dim=-1)


def penalize_feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    min_air_time: float = 0.25,
) -> torch.Tensor:
    """Reward correct foot lift duration; ignore when root is nearly stationary.

    Mirrors HOVER's penalize_feet_air_time (positive weight = reward).
    """
    from mjlab.tasks.tracking.mdp.commands import MotionCommand
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.last_air_time is None:
        return torch.zeros(env.num_envs, device=env.device)

    command = env.command_manager.get_term(command_name)
    assert isinstance(command, MotionCommand)
    ref_root_vel_xy = command.body_lin_vel_w[:, 0, :2]  # [B, 2]
    moving = (torch.norm(ref_root_vel_xy, dim=-1) > 0.1)  # [B]

    first_contact = sensor.compute_first_contact(env.step_dt)  # [B, N]
    last_air = sensor.data.last_air_time                        # [B, N]
    reward = torch.sum((last_air - min_air_time) * first_contact.float(), dim=-1)
    return reward * moving.float()
