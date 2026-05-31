"""K1 student policy observation computation for mjlab distillation."""

import torch

from neural_wbc.core.mask import create_mask, create_mask_element_names
from neural_wbc.core.math_utils import (
    euler_xyz_from_quat,
    quat_inv,
    quat_rotate_inverse,
    yaw_quat,
)

# K1 body/joint names in MJCF tree order (matching MotionCommand body_names and JOINT_NAMES).
K1_BODY_NAMES = [
    "Trunk",
    "Head_1", "Head_2",
    "Left_Arm_1", "Left_Arm_2", "Left_Arm_3", "left_hand_link",
    "Right_Arm_1", "Right_Arm_2", "Right_Arm_3", "right_hand_link",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Shank",
    "Left_Ankle_Cross", "left_foot_link",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Shank",
    "Right_Ankle_Cross", "right_foot_link",
]  # 23 bodies

K1_JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
]  # 22 joints

K1_MASK_ELEMENT_NAMES = create_mask_element_names(K1_BODY_NAMES, K1_JOINT_NAMES)
# → 23 body slots + 22 joint slots + 7 root slots = 52 total

# Mask modes: each mode selects which reference states the student can observe.
# Patterns are matched against K1_MASK_ELEMENT_NAMES via regex (see neural_wbc.core.mask).
K1_DISTILL_MASK_MODES = {
    "locomotion": {
        "lower_body": [
            ".*Hip.*", ".*Knee.*", ".*Ankle.*", ".*foot.*",
            "root.*",
        ],
    },
    "upper_body": {
        "upper_body": [
            ".*Shoulder.*", ".*Elbow.*", ".*Arm.*", ".*Head.*",
        ],
        "lower_body": ["root.*"],
    },
    "full": {
        "all": [".*"],
    },
}

# Observation dimensions:
#   distilled_robot_state:  joint_pos(22) + joint_vel(22) + ang_vel(3) + gravity(3) = 50
#   distilled_imitation:    kinematic(23×3=69) + joint_cmd(22) + root_cmd(7) + mask(52) = 150
#   distilled_last_action:  22
#   total:                  222
NUM_STUDENT_OBS = 222
NUM_TEACHER_OBS = 125  # see tracking_env_cfg.py actor terms


def compute_k1_student_obs(
    motion_cmd,
    robot,
    last_actions: torch.Tensor,
    mask: torch.Tensor,
    device: str,
) -> dict[str, torch.Tensor]:
    """Compute K1 student policy observations from current env state.

    Args:
        motion_cmd: MotionCommand instance (has body_pos_w, joint_pos, etc.)
        robot: mjlab Entity for the K1 robot
        last_actions: [N, 22] last applied actions
        mask: [N, 52] boolean mask (float 0/1) selecting tracked reference states
        device: torch device string

    Returns:
        dict with keys: distilled_robot_state (50D), distilled_imitation (150D),
                        distilled_last_action (22D)
    """
    N = motion_cmd.num_envs
    num_bodies = 23
    num_joints = 22

    # --- Robot state (50D) ---
    joint_pos = robot.data.joint_pos        # [N, 22]
    joint_vel = robot.data.joint_vel        # [N, 22]

    anchor_idx = motion_cmd.robot_anchor_body_index
    root_quat_w = robot.data.body_link_quat_w[:, anchor_idx]       # [N, 4] wxyz
    body_ang_vel_w = robot.data.body_link_ang_vel_w[:, anchor_idx]  # [N, 3]

    local_ang_vel = quat_rotate_inverse(root_quat_w, body_ang_vel_w)  # [N, 3]

    gravity_w = torch.zeros(N, 3, device=device)
    gravity_w[:, 2] = -1.0
    projected_gravity = quat_rotate_inverse(root_quat_w, gravity_w)  # [N, 3]

    robot_state = torch.cat([joint_pos, joint_vel, local_ang_vel, projected_gravity], dim=-1)  # [N, 50]

    # --- Reference motion state ---
    ref_body_pos_w = motion_cmd.body_pos_w        # [N, 23, 3] world-frame (includes env_origins)
    ref_body_quat_w = motion_cmd.body_quat_w      # [N, 23, 4] wxyz
    ref_body_lin_vel_w = motion_cmd.body_lin_vel_w  # [N, 23, 3]
    ref_joint_pos = motion_cmd.joint_pos          # [N, 22]

    anchor_motion_idx = motion_cmd.motion_anchor_body_index   # 0 for Trunk
    ref_root_pos_w = ref_body_pos_w[:, anchor_motion_idx]     # [N, 3]
    ref_root_quat_w = ref_body_quat_w[:, anchor_motion_idx]   # [N, 4]
    ref_root_lin_vel_w = ref_body_lin_vel_w[:, anchor_motion_idx]  # [N, 3]

    root_pos_w = robot.data.body_link_pos_w[:, anchor_idx]    # [N, 3] current root pos

    # --- Kinematic command: reference body positions in heading frame (69D) ---
    # Remove yaw from current root orientation, rotate body deltas into that frame.
    heading_yaw_q = yaw_quat(root_quat_w)   # [N, 4] wxyz yaw-only
    delta = ref_body_pos_w - root_pos_w.unsqueeze(1)  # [N, 23, 3]
    heading_q_exp = heading_yaw_q.unsqueeze(1).expand(-1, num_bodies, -1)  # [N, 23, 4]
    local_ref_body_pos = quat_rotate_inverse(
        heading_q_exp.reshape(-1, 4),
        delta.reshape(-1, 3),
    ).reshape(N, num_bodies, 3)
    kinematic_cmd = local_ref_body_pos.reshape(N, -1)   # [N, 69]

    # --- Joint command: delta joint positions (22D) ---
    joint_cmd = ref_joint_pos - joint_pos               # [N, 22]

    # --- Root command: target velocity, orientation, height (7D) ---
    target_root_lin_vel = quat_rotate_inverse(ref_root_quat_w, ref_root_lin_vel_w)  # [N, 3]

    ref_rpy = euler_xyz_from_quat(ref_root_quat_w)   # (roll, pitch, yaw) each [N]
    cur_rpy = euler_xyz_from_quat(root_quat_w)
    target_root_rot = torch.stack(
        [ref_rpy[0], ref_rpy[1], ref_rpy[2] - cur_rpy[2]], dim=-1
    )                                                           # [N, 3]
    target_root_height = ref_root_pos_w[:, 2:3]                # [N, 1]

    root_cmd = torch.cat(
        [target_root_lin_vel, target_root_rot, target_root_height], dim=-1
    )                                                           # [N, 7]

    # --- Apply mask to imitation observations (150D) ---
    kinematic_mask = mask[:, :num_bodies].repeat_interleave(3, dim=-1)       # [N, 69]
    joint_mask = mask[:, num_bodies:num_bodies + num_joints]                  # [N, 22]
    root_mask = mask[:, num_bodies + num_joints:]                             # [N, 7]

    kinematic_cmd = kinematic_cmd * kinematic_mask
    joint_cmd = joint_cmd * joint_mask
    root_cmd = root_cmd * root_mask

    imitation_obs = torch.cat(
        [kinematic_cmd, joint_cmd, root_cmd, mask], dim=-1
    )                                                           # [N, 150]

    return {
        "distilled_robot_state": robot_state,    # [N, 50]
        "distilled_imitation": imitation_obs,    # [N, 150]
        "distilled_last_action": last_actions,   # [N, 22]
    }
