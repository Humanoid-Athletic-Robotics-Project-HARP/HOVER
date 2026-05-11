# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from typing import Any, Literal

from hw_wrappers.booster_k1_sdk_wrapper import BoosterK1SDKWrapper
from mujoco_wrapper.mujoco_simulator import WBCMujoco

from neural_wbc.core.robot_wrapper import Robot, register_robot


@register_robot
class BoosterK1(Robot):
    """Real Booster K1 robot, complemented with MuJoCo for forward kinematics."""

    def __init__(
        self,
        cfg: Any,
        num_instances: int = 1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__(cfg, num_instances=num_instances, device=device)
        self.cfg = cfg
        self.device = device

        self._k1_sdk = BoosterK1SDKWrapper(cfg=cfg)
        self.send_command = self._resolve_command_fn(robot_actuation_type=cfg.robot_actuation_type)

        self._kinematic_model = WBCMujoco(
            model_path=cfg.model_xml_path,
            sim_dt=cfg.dt,
            enable_viewer=cfg.enable_viewer,
            num_instances=num_instances,
            device=device,
        )

        self._kinematic_model.model.opt.gravity = np.array([0, 0, self.cfg.gravity_value])

        self._joint_names = self._kinematic_model.joint_names
        self._body_names = self._kinematic_model.body_names

        self._free_joint_offset = 1 if self._kinematic_model.has_free_joint else 0

        self.joint_pos_offset = self._kinematic_model.joint_pos_offset
        self.joint_vel_offset = self._kinematic_model.joint_vel_offset

    def _resolve_command_fn(self, robot_actuation_type: Literal["Pos", "Torque"] = "Pos"):
        if robot_actuation_type == "Pos":
            return self._send_position_command
        elif robot_actuation_type == "Torque":
            return self._send_torque_command
        else:
            raise ValueError(f"Unrecognized robot actuation type {robot_actuation_type}")

    def update(self, obs_dict: dict[str, torch.Tensor]) -> None:
        if "root_pos" in obs_dict:
            self._root_position = obs_dict["root_pos"]
        if "root_orientation" in obs_dict:
            self._root_rotation = obs_dict["root_orientation"]

        self._root_lin_vel = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self._root_ang_vel = torch.zeros(1, 3, dtype=torch.float32, device=self.device)

        self._joint_positions = (
            torch.from_numpy(self._k1_sdk.joint_positions).unsqueeze(0).to(dtype=torch.float32, device=self.device)
        )
        self._joint_velocities = (
            torch.from_numpy(self._k1_sdk.joint_velocities).unsqueeze(0).to(dtype=torch.float32, device=self.device)
        )

        qpos = torch.hstack((self._root_position, self._root_rotation, self._joint_positions))
        qvel = torch.hstack((self._root_ang_vel, self._root_lin_vel, self._joint_velocities))

        self._kinematic_model.reset(qpos, qvel)

        self._joint_positions = self._kinematic_model.joint_positions
        self._joint_velocities = self._kinematic_model.joint_velocities
        self._body_positions = self._kinematic_model.body_positions
        self._body_rotations = self._kinematic_model.body_rotations
        self._body_lin_vels, self._body_ang_vels = self._kinematic_model.body_velocities

    def reset(self, **kwargs) -> None:
        qpos = kwargs.get("qpos")
        qvel = kwargs.get("qvel")
        self._kinematic_model.reset(qpos=qpos, qvel=qvel)

        joint_positions = qpos[..., self._kinematic_model.joint_pos_offset:]
        self._k1_sdk.reset(joint_positions.numpy())
        self.update({})

    def _send_position_command(self, positions: np.ndarray | None = None) -> None:
        if positions is not None:
            self._k1_sdk.publish_joint_position_cmd(positions.flatten())

    def _send_torque_command(self, torques: np.ndarray | None = None) -> None:
        raise NotImplementedError("Torque mode is not supported for K1 real hardware; use position mode.")

    def step(self, actions: np.ndarray | None = None, nsteps: int = 1) -> None:
        self._kinematic_model.forward()
        self._kinematic_model.update_viewer()
        self.send_command(actions)

    def get_body_ids(self, body_names: list[str] | None = None) -> dict[str, int]:
        return self._kinematic_model.get_body_ids(body_names, self._free_joint_offset)

    def get_joint_ids(self, joint_names: list[str] | None = None) -> dict[str, int]:
        return self._kinematic_model.get_joint_ids(joint_names, self._free_joint_offset)

    def get_body_pose(self, body_name: str = "Trunk") -> tuple[torch.Tensor, torch.Tensor]:
        return self._kinematic_model.get_body_pose(body_name)

    def get_base_projected_gravity(self, base_name: str = "Trunk") -> torch.Tensor:
        orientation_quat_wxyz = self._k1_sdk.trunk_orientation  # wxyz
        rot_mat_np = R.from_quat(orientation_quat_wxyz, scalar_first=True).as_matrix()
        rot_mat = torch.tensor(rot_mat_np, device=self.device, dtype=torch.float32)
        world_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        return (rot_mat.T @ world_gravity).unsqueeze(0)

    def get_base_angular_velocity(self, base_name: str = "Trunk") -> torch.Tensor:
        return torch.tensor(
            self._k1_sdk.trunk_angular_velocity, device=self.device, dtype=torch.float32
        ).unsqueeze(0)

    def get_terrain_heights(self) -> torch.Tensor:
        return self._kinematic_model.get_terrain_heights()

    def visualize(self, **payload) -> None:
        if "ref_motion_state" in payload:
            self._kinematic_model.visualize_ref_state(payload["ref_motion_state"])
