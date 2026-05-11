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

import torch
from dataclasses import dataclass
from typing import Literal

from inference_env.neural_wbc_env_cfg import NeuralWBCEnvCfg

from neural_wbc.core.mask import calculate_mask_length
from neural_wbc.data import get_data_path


@dataclass
class NeuralWBCEnvCfgRealK1(NeuralWBCEnvCfg):
    decimation = 1
    dt = 0.02  # 50 Hz
    cmd_publish_dt = 0.005  # 200 Hz
    max_episode_length_s = 3600
    action_scale = 0.25
    ctrl_delay_step_range = [0, 0]
    default_rfi_lim = 0
    robot = "booster_k1"
    base_name = "Trunk"

    # K1 has physical hand and head links — no virtual extension needed
    extend_body_parent_names = []
    extend_body_names = []
    extend_body_pos = torch.zeros(0, 3)

    tracked_body_names = [
        "left_hand_link",
        "right_hand_link",
        "Head_2",
    ]

    # Distillation parameters:
    single_history_dim = 72  # 22*3 + 6
    observation_history_length = 25
    num_bodies = 23
    num_joints = 22
    mask_length = calculate_mask_length(
        num_bodies=num_bodies,
        num_joints=num_joints,
    )

    control_type: Literal["Pos", "Torque", "None"] = "None"
    robot_actuation_type: Literal["Pos", "Torque"] = "Pos"

    # Hardware parameters
    network_interface = "eth0"
    reset_duration = 10.0  # seconds
    reset_step_dt = 0.01   # seconds
    gravity_value = -9.8   # m/s^2

    # In SERIAL mode, motor index == JointIndexK1 enum value (identity mapping).
    # Order: HeadYaw(0), HeadPitch(1), L_ShPitch(2), L_ShRoll(3), L_ElPitch(4), L_ElYaw(5),
    #        R_ShPitch(6), R_ShRoll(7), R_ElPitch(8), R_ElYaw(9),
    #        L_HipPitch(10), L_HipRoll(11), L_HipYaw(12), L_Knee(13),
    #        L_AnklePitch(14), L_AnkleRoll(15),
    #        R_HipPitch(16), R_HipRoll(17), R_HipYaw(18), R_Knee(19),
    #        R_AnklePitch(20), R_AnkleRoll(21)
    JointSeq2MotorID = list(range(22))
    MotorID2JointSeq = list(range(22))

    motor_id_to_name = {
        0:  "AAHead_yaw",
        1:  "Head_pitch",
        2:  "ALeft_Shoulder_Pitch",
        3:  "Left_Shoulder_Roll",
        4:  "Left_Elbow_Pitch",
        5:  "Left_Elbow_Yaw",
        6:  "ARight_Shoulder_Pitch",
        7:  "Right_Shoulder_Roll",
        8:  "Right_Elbow_Pitch",
        9:  "Right_Elbow_Yaw",
        10: "Left_Hip_Pitch",
        11: "Left_Hip_Roll",
        12: "Left_Hip_Yaw",
        13: "Left_Knee_Pitch",
        14: "Left_Ankle_Pitch",
        15: "Left_Ankle_Roll",
        16: "Right_Hip_Pitch",
        17: "Right_Hip_Roll",
        18: "Right_Hip_Yaw",
        19: "Right_Knee_Pitch",
        20: "Right_Ankle_Pitch",
        21: "Right_Ankle_Roll",
    }

    # Joints with lower kp/kd
    weak_motors = {
        "AAHead_yaw",
        "Head_pitch",
        "ALeft_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "ARight_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    }

    kp_low = 40.0
    kp_high = 200.0
    kd_low = 1.0
    kd_high = 5.0

    stiffness = {
        "AAHead_yaw": 10.0,
        "Head_pitch": 10.0,
        "ALeft_Shoulder_Pitch": 40.0,
        "Left_Shoulder_Roll": 40.0,
        "Left_Elbow_Pitch": 40.0,
        "Left_Elbow_Yaw": 40.0,
        "ARight_Shoulder_Pitch": 40.0,
        "Right_Shoulder_Roll": 40.0,
        "Right_Elbow_Pitch": 40.0,
        "Right_Elbow_Yaw": 40.0,
        "Left_Hip_Pitch": 150.0,
        "Left_Hip_Roll": 150.0,
        "Left_Hip_Yaw": 100.0,
        "Left_Knee_Pitch": 200.0,
        "Left_Ankle_Pitch": 20.0,
        "Left_Ankle_Roll": 20.0,
        "Right_Hip_Pitch": 150.0,
        "Right_Hip_Roll": 150.0,
        "Right_Hip_Yaw": 100.0,
        "Right_Knee_Pitch": 200.0,
        "Right_Ankle_Pitch": 20.0,
        "Right_Ankle_Roll": 20.0,
    }

    damping = {
        "AAHead_yaw": 2.0,
        "Head_pitch": 2.0,
        "ALeft_Shoulder_Pitch": 5.0,
        "Left_Shoulder_Roll": 5.0,
        "Left_Elbow_Pitch": 5.0,
        "Left_Elbow_Yaw": 5.0,
        "ARight_Shoulder_Pitch": 5.0,
        "Right_Shoulder_Roll": 5.0,
        "Right_Elbow_Pitch": 5.0,
        "Right_Elbow_Yaw": 5.0,
        "Left_Hip_Pitch": 5.0,
        "Left_Hip_Roll": 5.0,
        "Left_Hip_Yaw": 5.0,
        "Left_Knee_Pitch": 5.0,
        "Left_Ankle_Pitch": 4.0,
        "Left_Ankle_Roll": 4.0,
        "Right_Hip_Pitch": 5.0,
        "Right_Hip_Roll": 5.0,
        "Right_Hip_Yaw": 5.0,
        "Right_Knee_Pitch": 5.0,
        "Right_Ankle_Pitch": 4.0,
        "Right_Ankle_Roll": 4.0,
    }

    effort_limit = {
        "AAHead_yaw": 6.0,
        "Head_pitch": 6.0,
        "ALeft_Shoulder_Pitch": 14.0,
        "Left_Shoulder_Roll": 14.0,
        "Left_Elbow_Pitch": 14.0,
        "Left_Elbow_Yaw": 14.0,
        "ARight_Shoulder_Pitch": 14.0,
        "Right_Shoulder_Roll": 14.0,
        "Right_Elbow_Pitch": 14.0,
        "Right_Elbow_Yaw": 14.0,
        "Left_Hip_Pitch": 30.0,
        "Left_Hip_Roll": 35.0,
        "Left_Hip_Yaw": 20.0,
        "Left_Knee_Pitch": 40.0,
        "Left_Ankle_Pitch": 20.0,
        "Left_Ankle_Roll": 20.0,
        "Right_Hip_Pitch": 30.0,
        "Right_Hip_Roll": 35.0,
        "Right_Hip_Yaw": 20.0,
        "Right_Knee_Pitch": 40.0,
        "Right_Ankle_Pitch": 20.0,
        "Right_Ankle_Roll": 20.0,
    }

    position_limit = {
        "AAHead_yaw": [-1.0, 1.0],
        "Head_pitch": [-0.349, 0.855],
        "ALeft_Shoulder_Pitch": [-3.316, 1.22],
        "Left_Shoulder_Roll": [-1.74, 1.57],
        "Left_Elbow_Pitch": [-2.27, 2.27],
        "Left_Elbow_Yaw": [-2.44, 0.0],
        "ARight_Shoulder_Pitch": [-3.316, 1.22],
        "Right_Shoulder_Roll": [-1.57, 1.74],
        "Right_Elbow_Pitch": [-2.27, 2.27],
        "Right_Elbow_Yaw": [0.0, 2.44],
        "Left_Hip_Pitch": [-3.0, 2.21],
        "Left_Hip_Roll": [-0.4, 1.57],
        "Left_Hip_Yaw": [-1.0, 1.0],
        "Left_Knee_Pitch": [0.0, 2.23],
        "Left_Ankle_Pitch": [-0.87, 0.345],
        "Left_Ankle_Roll": [-0.345, 0.345],
        "Right_Hip_Pitch": [-3.0, 2.21],
        "Right_Hip_Roll": [-1.57, 0.4],
        "Right_Hip_Yaw": [-1.0, 1.0],
        "Right_Knee_Pitch": [0.0, 2.23],
        "Right_Ankle_Pitch": [-0.87, 0.345],
        "Right_Ankle_Roll": [-0.345, 0.345],
    }

    velocity_limit = {
        "AAHead_yaw": 10.0,
        "Head_pitch": 10.0,
        "ALeft_Shoulder_Pitch": 10.0,
        "Left_Shoulder_Roll": 10.0,
        "Left_Elbow_Pitch": 10.0,
        "Left_Elbow_Yaw": 10.0,
        "ARight_Shoulder_Pitch": 10.0,
        "Right_Shoulder_Roll": 10.0,
        "Right_Elbow_Pitch": 10.0,
        "Right_Elbow_Yaw": 10.0,
        "Left_Hip_Pitch": 20.0,
        "Left_Hip_Roll": 20.0,
        "Left_Hip_Yaw": 20.0,
        "Left_Knee_Pitch": 20.0,
        "Left_Ankle_Pitch": 10.0,
        "Left_Ankle_Roll": 10.0,
        "Right_Hip_Pitch": 20.0,
        "Right_Hip_Roll": 20.0,
        "Right_Hip_Yaw": 20.0,
        "Right_Knee_Pitch": 20.0,
        "Right_Ankle_Pitch": 10.0,
        "Right_Ankle_Roll": 10.0,
    }

    robot_init_state = {
        "base_pos": [0.0, 0.0, 0.56],
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "joint_pos": {
            "AAHead_yaw": 0.0,
            "Head_pitch": 0.0,
            "ALeft_Shoulder_Pitch": 0.0,
            "Left_Shoulder_Roll": 0.0,
            "Left_Elbow_Pitch": 0.0,
            "Left_Elbow_Yaw": 0.0,
            "ARight_Shoulder_Pitch": 0.0,
            "Right_Shoulder_Roll": 0.0,
            "Right_Elbow_Pitch": 0.0,
            "Right_Elbow_Yaw": 0.0,
            "Left_Hip_Pitch": -0.28,
            "Left_Hip_Roll": 0.0,
            "Left_Hip_Yaw": 0.0,
            "Left_Knee_Pitch": 0.56,
            "Left_Ankle_Pitch": -0.28,
            "Left_Ankle_Roll": 0.0,
            "Right_Hip_Pitch": -0.28,
            "Right_Hip_Roll": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Right_Knee_Pitch": 0.56,
            "Right_Ankle_Pitch": -0.28,
            "Right_Ankle_Roll": 0.0,
        },
        "joint_vel": {},
    }

    lower_body_joint_ids = list(range(10, 22))  # hips, knees, ankles
    upper_body_joint_ids = list(range(0, 10))   # head, shoulders, elbows

    def __post_init__(self):
        self.reference_motion_cfg.motion_path = get_data_path("motions/k1_fight_001.pkl")
        self.reference_motion_cfg.skeleton_path = get_data_path("motion_lib/k1.xml")
        self.reference_motion_cfg.fk_frame_rotation = [0.5, 0.5, 0.5, 0.5]
        self.reference_motion_cfg.extend_hand = False
        self.reference_motion_cfg.extend_head = False
