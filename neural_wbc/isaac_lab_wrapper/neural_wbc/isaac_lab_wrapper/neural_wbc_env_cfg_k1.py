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

import torch

from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.data import get_data_path

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from .events import NeuralWBCPlayEventCfg, NeuralWBCTrainEventCfg
from .k1_cfg import K1_CFG
from .neural_wbc_env_cfg import NeuralWBCEnvCfg
from .terrain import HARD_ROUGH_TERRAINS_CFG, flat_terrain

# K1 body/joint arm mapping notes (critical — read before editing):
#
#   Arm chain: Trunk → Left_Arm_1 (shoulder, 77 mm out) → Left_Arm_2 (shoulder roll, 68 mm)
#              → Left_Arm_3 (44 mm, near shoulder area; has Left_Elbow_Pitch joint)
#              → left_hand_link (121.5 mm; has Left_Elbow_Yaw joint; IS the functional elbow body)
#              → hand tip = left_hand_link + [0, 0.228, 0] in local frame
#
#   left_hand_link is used as the tracked "elbow" body because it sits at the true elbow.
#   Left_Arm_3 is NOT the elbow — it is only 44 mm from Left_Arm_2 (shoulder roll), near shoulder.
#   left_hand_tip (extend body) is the virtual hand-tip 228 mm along local Y from left_hand_link.

DISTILL_MASK_MODES_ALL = {
    "upper_body": {
        "upper_body": [
            ".*Shoulder.*",
            ".*Elbow.*",
            ".*Head.*",
        ],
        "lower_body": ["root.*"],
    },
    "lower_body": {
        "lower_body": [
            ".*Hip.*",
            ".*Knee.*",
            ".*Ankle.*",
            "root.*",
        ],
    },
}


@configclass
class NeuralWBCEnvCfgK1(NeuralWBCEnvCfg):
    # General parameters.
    # K1: 22 joints, 2 feet.
    #
    # K1's pkl pose_aa has 23 physical bodies only (no virtual joints baked in),
    # unlike H1's which has 22 = 20 physical + 2 virtual hands. MotionLibH1's FK
    # cannot extend K1 correctly (hardcodes H1 parent indices), so
    # motion_lib_extend_hand/head are False. The sim-side extends (left_hand_tip,
    # right_hand_tip, head_tip) are still computed via extend_body_parent_names.
    #
    # observation_space breakdown:
    #   robot_state = (N_sim-1)*3 + N_sim*6 + N_sim*3 + N_sim*3   N_sim=26  → 387
    #   imitation   = N_trk*(3+6+3+3+3+6)                         N_trk=23  → 552
    #   last_action = 22
    #   total                                                                 = 961
    #
    # state_space = observation_space + privileged:
    #   base_com_bias(3) + ground_friction(2) + body_mass_scale(7) +
    #   kp_scale(22) + kd_scale(22) + rfi_lim_scale(22) +
    #   contact_forces(2*3) + recovery_counters(1)                          =  85
    #   total                                                                = 1046
    action_space = 22
    observation_space = 961
    state_space = 1046

    # Disable motion-lib FK extending: K1 pose_aa has 23 physical joints only.
    # The sim-side extend bodies are still computed from extend_body_parent_names.
    motion_lib_extend_hand = False
    motion_lib_extend_head = False

    # Distillation parameters.
    single_history_dim = 72  # 22*3 + 6
    observation_history_length = 25

    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = {"upper_body": DISTILL_MASK_MODES_ALL["upper_body"]}

    # Actuators: pulled from k1_cfg and overridden here for IdealPD (teacher policy).
    actuators = K1_CFG.actuators

    robot = K1_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)

    # All 23 physical K1 bodies in MJCF tree order.
    body_names = [
        "Trunk",
        "Head_1",
        "Head_2",
        "Left_Arm_1",
        "Left_Arm_2",
        "Left_Arm_3",
        "left_hand_link",   # functional elbow (Left_Elbow_Yaw joint is here)
        "Right_Arm_1",
        "Right_Arm_2",
        "Right_Arm_3",
        "right_hand_link",  # functional elbow (Right_Elbow_Yaw joint is here)
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Shank",
        "Left_Ankle_Cross",
        "left_foot_link",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Shank",
        "Right_Ankle_Cross",
        "right_foot_link",
    ]

    # 22 joints in MJCF tree order.
    joint_names = [
        "AAHead_yaw",
        "Head_pitch",
        "ALeft_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",     # joint at Left_Arm_3 (drives left_hand_link)
        "Left_Elbow_Yaw",       # joint at left_hand_link (true elbow yaw)
        "ARight_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",    # joint at Right_Arm_3 (drives right_hand_link)
        "Right_Elbow_Yaw",      # joint at right_hand_link (true elbow yaw)
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    ]

    # Joint index ranges (indices into joint_names above).
    upper_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]    # head + arms
    lower_body_joint_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # hips + knees + ankles

    base_name = "Trunk"
    root_id = body_names.index(base_name)  # 0

    feet_name = ".*_foot_link"

    # Native USD joint indices for the ankle roll joints (one per foot).
    # Used to index joint_friction in privileged obs instead of body indices,
    # because K1's right_foot_link body index (22) equals the joint count and
    # would be OOB for joint_friction[:, feet_joint_ids].
    feet_joint_ids = [20, 21]  # Left_Ankle_Roll=20, Right_Ankle_Roll=21 in native USD ordering

    # Virtual bodies extended from the robot skeleton for tracking.
    # - left_hand_link  is the functional elbow: hand tip is 228 mm along local +Y
    # - right_hand_link is the functional elbow: hand tip is 228 mm along local -Y
    # - Head_2          is the topmost head body: head tip is 100 mm up in local +Z
    extend_body_parent_names = ["left_hand_link", "right_hand_link", "Head_2"]
    extend_body_names = ["left_hand_tip", "right_hand_tip", "head_tip"]
    extend_body_pos = torch.tensor([[0.0, 0.228, 0.0], [0.0, -0.228, 0.0], [0.0, 0.0, 0.1]])

    # Only the 23 physical bodies are tracked: the reference motion lib cannot
    # compute virtual extend-body positions for K1 (see motion_lib_extend_hand).
    # The sim-side robot_state obs still uses all 26 bodies (extend included).
    tracked_body_names = [
        "Trunk",
        "Head_1",
        "Head_2",
        "Left_Arm_1",
        "Left_Arm_2",
        "Left_Arm_3",
        "left_hand_link",
        "Right_Arm_1",
        "Right_Arm_2",
        "Right_Arm_3",
        "right_hand_link",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Shank",
        "Left_Ankle_Cross",
        "left_foot_link",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Shank",
        "Right_Ankle_Cross",
        "right_foot_link",
    ]

    # PD gains for position-control mode.
    stiffness = {
        "AAHead_yaw": 40.0,
        "Head_pitch": 40.0,
        "ALeft_Shoulder_Pitch": 40.0,
        "Left_Shoulder_Roll": 40.0,
        "Left_Elbow_Pitch": 40.0,
        "Left_Elbow_Yaw": 40.0,
        "ARight_Shoulder_Pitch": 40.0,
        "Right_Shoulder_Roll": 40.0,
        "Right_Elbow_Pitch": 40.0,
        "Right_Elbow_Yaw": 40.0,
        "Left_Hip_Pitch": 200.0,
        "Left_Hip_Roll": 150.0,
        "Left_Hip_Yaw": 150.0,
        "Left_Knee_Pitch": 200.0,
        "Left_Ankle_Pitch": 20.0,
        "Left_Ankle_Roll": 20.0,
        "Right_Hip_Pitch": 200.0,
        "Right_Hip_Roll": 150.0,
        "Right_Hip_Yaw": 150.0,
        "Right_Knee_Pitch": 200.0,
        "Right_Ankle_Pitch": 20.0,
        "Right_Ankle_Roll": 20.0,
    }

    damping = {
        "AAHead_yaw": 5.0,
        "Head_pitch": 5.0,
        "ALeft_Shoulder_Pitch": 10.0,
        "Left_Shoulder_Roll": 10.0,
        "Left_Elbow_Pitch": 10.0,
        "Left_Elbow_Yaw": 10.0,
        "ARight_Shoulder_Pitch": 10.0,
        "Right_Shoulder_Roll": 10.0,
        "Right_Elbow_Pitch": 10.0,
        "Right_Elbow_Yaw": 10.0,
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

    mass_randomized_body_names = [
        "Trunk",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
    ]

    undesired_contact_body_names = [
        "Trunk",
        "Left_Arm_1",
        "Left_Arm_2",
        "Left_Arm_3",
        "left_hand_link",
        "Right_Arm_1",
        "Right_Arm_2",
        "Right_Arm_3",
        "right_hand_link",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
    ]

    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    def __post_init__(self):
        super().__post_init__()

        self.reference_motion_manager.motion_path = get_data_path("motions/cmu_punch.pkl")
        self.reference_motion_manager.skeleton_path = get_data_path("motion_lib/k1.xml")

        # Override reward limits for K1's 22 joints (base NeuralWBCRewardCfg has 19 for H1).
        # Order follows joint_names: head(2) + l_arm(4) + r_arm(4) + l_leg(6) + r_leg(6).
        self.rewards.torque_limits = [
            6.0,   # AAHead_yaw
            6.0,   # Head_pitch
            14.0,  # ALeft_Shoulder_Pitch
            14.0,  # Left_Shoulder_Roll
            14.0,  # Left_Elbow_Pitch
            14.0,  # Left_Elbow_Yaw
            14.0,  # ARight_Shoulder_Pitch
            14.0,  # Right_Shoulder_Roll
            14.0,  # Right_Elbow_Pitch
            14.0,  # Right_Elbow_Yaw
            30.0,  # Left_Hip_Pitch
            35.0,  # Left_Hip_Roll
            20.0,  # Left_Hip_Yaw
            40.0,  # Left_Knee_Pitch
            20.0,  # Left_Ankle_Pitch
            20.0,  # Left_Ankle_Roll
            30.0,  # Right_Hip_Pitch
            35.0,  # Right_Hip_Roll
            20.0,  # Right_Hip_Yaw
            40.0,  # Right_Knee_Pitch
            20.0,  # Right_Ankle_Pitch
            20.0,  # Right_Ankle_Roll
        ]
        self.rewards.joint_pos_limits = [
            (-1.0,    1.0),     # AAHead_yaw
            (-0.349,  0.855),   # Head_pitch
            (-3.316,  1.22),    # ALeft_Shoulder_Pitch
            (-1.74,   1.57),    # Left_Shoulder_Roll
            (-2.27,   2.27),    # Left_Elbow_Pitch
            (-2.44,   0.0),     # Left_Elbow_Yaw
            (-3.316,  1.22),    # ARight_Shoulder_Pitch
            (-1.57,   1.74),    # Right_Shoulder_Roll
            (-2.27,   2.27),    # Right_Elbow_Pitch
            (0.0,     2.44),    # Right_Elbow_Yaw
            (-3.0,    2.21),    # Left_Hip_Pitch
            (-0.4,    1.57),    # Left_Hip_Roll
            (-1.0,    1.0),     # Left_Hip_Yaw
            (0.0,     2.23),    # Left_Knee_Pitch
            (-0.87,   0.345),   # Left_Ankle_Pitch
            (-0.345,  0.345),   # Left_Ankle_Roll
            (-3.0,    2.21),    # Right_Hip_Pitch
            (-1.57,   0.4),     # Right_Hip_Roll
            (-1.0,    1.0),     # Right_Hip_Yaw
            (0.0,     2.23),    # Right_Knee_Pitch
            (-0.87,   0.345),   # Right_Ankle_Pitch
            (-0.345,  0.345),   # Right_Ankle_Roll
        ]
        # Velocity limits from official K1 URDF (booster_assets/robots/K1/K1_22dof.urdf).
        self.rewards.joint_vel_limits = [
            18.0,  # AAHead_yaw
            18.0,  # Head_pitch
            18.0,  # ALeft_Shoulder_Pitch
            18.0,  # Left_Shoulder_Roll
            18.0,  # Left_Elbow_Pitch
            18.0,  # Left_Elbow_Yaw
            18.0,  # ARight_Shoulder_Pitch
            18.0,  # Right_Shoulder_Roll
            18.0,  # Right_Elbow_Pitch
            18.0,  # Right_Elbow_Yaw
            7.1,   # Left_Hip_Pitch
            12.9,  # Left_Hip_Roll
            18.1,  # Left_Hip_Yaw
            12.5,  # Left_Knee_Pitch
            18.1,  # Left_Ankle_Pitch
            18.1,  # Left_Ankle_Roll
            7.1,   # Right_Hip_Pitch
            12.9,  # Right_Hip_Roll
            18.1,  # Right_Hip_Yaw
            12.5,  # Right_Knee_Pitch
            18.1,  # Right_Ankle_Pitch
            18.1,  # Right_Ankle_Roll
        ]

        # K1-specific reward scale and sigma overrides.
        #
        # penalize_joint_accelerations: K1's arm joints have very small distal inertia
        # (~0.002 kg·m²) with 14 Nm actuators → max accel ~7000 rad/s² → sum(acc²) up to
        # 49M per arm joint. The default -0.000011 was calibrated for H1's larger-inertia
        # joints and over-penalizes K1 by ~300× relative to the torque penalty. Reduce 10×.
        self.rewards.scales["penalize_joint_accelerations"] = -1e-7

        # penalize_torques: K1 max torques (6–40 Nm) are 5–10× smaller than H1 (200+ Nm),
        # so sum(torque²) is ~100× smaller at the default weight -0.0001. Increase 10× so
        # the torque-smoothness signal remains meaningful relative to other penalties.
        self.rewards.scales["penalize_torques"] = -0.001

        # joint_vel_sigma: default 1.0 rad/s is too narrow for locomotion — running joints
        # reach 5–15 rad/s, making exp(-err²/1²) ≈ 0 for any real motion. Widen to 5.0 so
        # the velocity tracking reward provides a useful gradient signal.
        self.rewards.joint_vel_sigma = 5.0

        # Swap position/velocity tracking weights so the policy can't exploit pose-matching.
        # Default: joint_positions=32, joint_velocities=16. Halve positions, double velocities.
        self.rewards.scales["reward_track_joint_positions"] = 16.0
        self.rewards.scales["reward_track_joint_velocities"] = 32.0

        if self.terrain.terrain_generator == HARD_ROUGH_TERRAINS_CFG:
            self.events.update_curriculum.params["penalty_level_up_threshold"] = 125

        if self.mode == NeuralWBCModes.TRAIN:
            self.episode_length_s = 20.0
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "Trunk"
        elif self.mode == NeuralWBCModes.DISTILL:
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "Trunk"
            self.add_policy_obs_noise = False
            self.reset_mask = True
            num_regions = len(self.distill_mask_modes)
            if num_regions == 1:
                region_modes = list(self.distill_mask_modes.values())[0]
                if len(region_modes) == 1:
                    self.reset_mask = False
        elif self.mode == NeuralWBCModes.TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"upper_body": DISTILL_MASK_MODES_ALL["upper_body"]}
        elif self.mode == NeuralWBCModes.DISTILL_TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.distill_teleop_selected_keypoints_names = []
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.default_rfi_lim = 0.0
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"upper_body": DISTILL_MASK_MODES_ALL["upper_body"]}
        else:
            raise ValueError(f"Unsupported mode {self.mode}")


class NeuralWBCEnvCfgK1Aggressive(NeuralWBCEnvCfgK1):
    """K1 config with aggressive velocity-first reward structure.

    Heavily penalises pose-only solutions by making joint velocity tracking
    4× more valuable than position tracking and widening the velocity kernel
    to sigma=10 so large velocity errors still produce a meaningful gradient.
    Body velocity tracking weight is also boosted to reinforce COM dynamics.
    """

    def __post_init__(self):
        super().__post_init__()

        # Override the moderate Run-1 swap with a more extreme split.
        self.rewards.scales["reward_track_joint_positions"] = 8.0
        self.rewards.scales["reward_track_joint_velocities"] = 64.0

        # Wider kernel: 10 rad/s velocity error → exp(-100/100)=0.37 (real gradient).
        self.rewards.joint_vel_sigma = 10.0

        # Boost body velocity tracking to reinforce COM-level dynamics.
        self.rewards.scales["reward_track_body_velocities"] = 20.0
