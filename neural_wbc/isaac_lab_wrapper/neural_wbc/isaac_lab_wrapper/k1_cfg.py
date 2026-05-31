# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

_DEFAULT_K1_USD = ""
K1_USD_PATH = os.environ.get("K1_USD_PATH", _DEFAULT_K1_USD)

# Default standing pose.
# At all-zero angles the arms extend laterally (T-pose).
# Left_Shoulder_Roll ~ -1.4 rad folds the left arm down; Right_Shoulder_Roll ~ +1.4 for right.
_DEFAULT_Q = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.0,
    "Left_Hip_Pitch": -0.25,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.5,
    "Left_Ankle_Pitch": -0.25,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.25,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.5,
    "Right_Ankle_Pitch": -0.25,
    "Right_Ankle_Roll": 0.0,
}

K1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=K1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.62),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=_DEFAULT_Q,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
                "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
                "Left_Knee_Pitch", "Right_Knee_Pitch",
            ],
            effort_limit={
                ".*_Hip_Pitch": 30.0,
                ".*_Hip_Roll": 35.0,
                ".*_Hip_Yaw": 20.0,
                ".*_Knee_Pitch": 40.0,
            },
            velocity_limit={
                ".*_Hip_Pitch": 7.1,
                ".*_Hip_Roll": 12.9,
                ".*_Hip_Yaw": 18.1,
                ".*_Knee_Pitch": 12.5,
            },
            stiffness=0,
            damping=0,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=["Left_Ankle_Pitch", "Left_Ankle_Roll", "Right_Ankle_Pitch", "Right_Ankle_Roll"],
            effort_limit=20.0,
            velocity_limit=18.1,
            stiffness=0,
            damping=0,
        ),
        "head": IdealPDActuatorCfg(
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit=6.0,
            velocity_limit=18.0,
            stiffness=0,
            damping=0,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
                "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
            ],
            effort_limit=14.0,
            velocity_limit=18.0,
            stiffness=0,
            damping=0,
        ),
    },
)
