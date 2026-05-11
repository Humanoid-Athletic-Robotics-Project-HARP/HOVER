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

import numpy as np
import time
from threading import RLock
from typing import Any

from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowCmdType,
    MotorCmd,
)
from scipy.spatial.transform import Rotation as R

# K1 has 22 joints in SERIAL mode (virtual joint space: ankle_pitch/roll instead of crank).
# Joint order matches JointIndexK1 enum in b1_api_const.hpp:
#   0: HeadYaw, 1: HeadPitch
#   2-5: Left arm (Shoulder Pitch/Roll, Elbow Pitch/Yaw)
#   6-9: Right arm (Shoulder Pitch/Roll, Elbow Pitch/Yaw)
#   10-13: Left leg (Hip Pitch/Roll/Yaw, Knee)
#   14-15: Left ankle (Pitch, Roll) — virtual in SERIAL mode
#   16-19: Right leg (Hip Pitch/Roll/Yaw, Knee)
#   20-21: Right ankle (Pitch, Roll) — virtual in SERIAL mode
K1_JOINT_CNT = 22


class BoosterK1SDKWrapper:
    """Low-level interface for the Booster K1 robot using SERIAL mode commands."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

        self._lock = RLock()
        self._joint_positions = np.zeros(K1_JOINT_CNT, dtype=np.float64)
        self._joint_velocities = np.zeros(K1_JOINT_CNT, dtype=np.float64)
        # IMU: orientation as wxyz quaternion (converted from RPY), gyro in robot frame
        self._orientation_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._angular_velocity = np.zeros(3, dtype=np.float64)

        self._state_received = False
        self._cmd_received = False

        # Pre-allocate command message
        self._low_cmd = LowCmd()
        self._low_cmd.cmd_type = LowCmdType.SERIAL
        self._low_cmd.motor_cmd = [MotorCmd() for _ in range(K1_JOINT_CNT)]
        self._init_cmd()

        ChannelFactory.Instance().Init(0, self.cfg.network_interface)

        self._publisher = B1LowCmdPublisher()
        self._publisher.InitChannel()

        self._subscriber = B1LowStateSubscriber(self._state_handler)
        self._subscriber.InitChannel()

    def _init_cmd(self):
        for i in range(K1_JOINT_CNT):
            motor_name = self.cfg.motor_id_to_name.get(i, "")
            if motor_name in self.cfg.weak_motors:
                kp = self.cfg.kp_low
                kd = self.cfg.kd_low
            else:
                kp = self.cfg.kp_high
                kd = self.cfg.kd_high
            self._low_cmd.motor_cmd[i].q = 0.0
            self._low_cmd.motor_cmd[i].dq = 0.0
            self._low_cmd.motor_cmd[i].tau = 0.0
            self._low_cmd.motor_cmd[i].kp = kp
            self._low_cmd.motor_cmd[i].kd = kd
            self._low_cmd.motor_cmd[i].weight = 1.0

    def _state_handler(self, msg):
        with self._lock:
            # IMU: convert RPY (radians) to wxyz quaternion
            rpy = msg.imu_state.rpy
            rot = R.from_euler("xyz", [rpy[0], rpy[1], rpy[2]])
            q_xyzw = rot.as_quat()
            self._orientation_quat = np.array(
                [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
            )

            gyro = msg.imu_state.gyro
            self._angular_velocity = np.array([gyro[0], gyro[1], gyro[2]], dtype=np.float64)

            # Joint state from SERIAL virtual joint space
            for i in range(K1_JOINT_CNT):
                self._joint_positions[i] = msg.motor_state_serial[i].q
                self._joint_velocities[i] = msg.motor_state_serial[i].dq

            self._state_received = True

    def publish_joint_position_cmd(self, positions: np.ndarray) -> None:
        with self._lock:
            for i in range(K1_JOINT_CNT):
                self._low_cmd.motor_cmd[i].q = float(positions[i])
                self._low_cmd.motor_cmd[i].dq = 0.0
                self._low_cmd.motor_cmd[i].tau = 0.0
            self._publisher.Write(self._low_cmd)
            self._cmd_received = True

    def reset(self, desired_joint_positions: np.ndarray | None = None) -> None:
        desired = np.zeros(K1_JOINT_CNT) if desired_joint_positions is None else desired_joint_positions.flatten()
        print("Resetting K1 to given pose.")
        t = 0.0
        duration = self.cfg.reset_duration
        step_dt = self.cfg.reset_step_dt
        while t < duration:
            t += step_dt
            ratio = t / duration
            print(f"\rResetting: {int(duration - t)}s remaining...", end="", flush=True)
            with self._lock:
                current = self._joint_positions.copy()
            target = current + (desired - current) * ratio
            self.publish_joint_position_cmd(target)
            time.sleep(step_dt)
        print("\nReset complete.")

    @property
    def joint_positions(self) -> np.ndarray:
        with self._lock:
            return self._joint_positions.copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        with self._lock:
            return self._joint_velocities.copy()

    @property
    def trunk_orientation(self) -> np.ndarray:
        """Trunk orientation as wxyz quaternion in world frame."""
        with self._lock:
            return self._orientation_quat.copy()

    @property
    def trunk_angular_velocity(self) -> np.ndarray:
        """Trunk angular velocity in robot body frame (rad/s)."""
        with self._lock:
            return self._angular_velocity.copy()
