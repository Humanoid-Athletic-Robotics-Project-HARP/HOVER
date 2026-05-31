"""K1 flat-terrain motion tracking environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from .custom_rewards import leg_action_rate_l2, upper_body_action_rate_l2
from .k1_constants import K1_ACTION_SCALE, get_k1_robot_cfg

# K1 body split for HOVER-style upper/lower rewards.
# Upper body: head + 4 arm links per side = 10 bodies (excl. Trunk).
# Lower body: 3 hip links + shank + ankle + foot per side = 12 bodies.
# VR keypoints (head-top + hands): highest-priority end-effectors.
_UPPER_BODY_NAMES = (
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
)
_LOWER_BODY_NAMES = (
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
)
_VR_KEYPOINT_NAMES = ("Head_2", "left_hand_link", "right_hand_link")
_FOOT_BODY_NAMES = ("left_foot_link", "right_foot_link")


def k1_flat_tracking_env_cfg(
    motion_file: str = "",
    play: bool = False,
    has_state_estimation: bool = True,
) -> ManagerBasedRlEnvCfg:
    """Create K1 flat-terrain motion tracking configuration."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_k1_robot_cfg()}

    # Self-collision sensor: left-leg subtree vs right-leg subtree only.
    # Arms excluded — K1's shoulder-roll pose causes constant arm-torso contact
    # in the reference motion, swamping the tracking reward.
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="Left_Hip_Pitch", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="Right_Hip_Pitch", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    # Apply per-joint action scales.
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = K1_ACTION_SCALE

    # Motion command — body names to track (all 23 physical K1 bodies).
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "Trunk"
    motion_cmd.body_names = (
        "Trunk",
        *_UPPER_BODY_NAMES,
        *_LOWER_BODY_NAMES,
    )
    if motion_file:
        motion_cmd.motion_file = motion_file

    # Uniform sampling: fixes the adaptive curriculum feedback loop that caused
    # oscillating episode length across v1–v3.
    motion_cmd.sampling_mode = "uniform"

    # Domain randomization.
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r"^(left|right)_foot_1_collision$",
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

    # Remove EE body pos termination — foot Z exceeds 0.25m during walking.
    cfg.terminations.pop("ee_body_pos", None)
    cfg.terminations["anchor_pos"] = TerminationTermCfg(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.5},
    )

    # Root position: wider std so gradient doesn't vanish at ~0.5m XY error.
    cfg.rewards["motion_global_root_pos"] = RewardTermCfg(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.6},
    )

    # --- Upper/lower body split (HOVER-inspired, tuned for K1 cold-start) ---
    # v5 lesson: std=0.05 + weight=3/4 destabilised orientation from cold-start.
    # Use std=0.1 and moderate weights so locomotion can emerge before arm
    # tracking tightens; lower body mirrors HOVER sigma exactly.
    cfg.rewards.pop("motion_body_pos", None)
    cfg.rewards["motion_upper_body_pos"] = RewardTermCfg(
        func=mdp.motion_relative_body_position_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.1,
            "body_names": _UPPER_BODY_NAMES,
        },
    )
    cfg.rewards["motion_lower_body_pos"] = RewardTermCfg(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.5,
        params={
            "command_name": "motion",
            "std": 0.5,
            "body_names": _LOWER_BODY_NAMES,
        },
    )

    # VR keypoints: head-top + hands. Relaxed vs v5 (std=0.1, w=2.0) so trunk
    # orientation isn't pulled into unstable poses before gait is stable.
    cfg.rewards["motion_keypoints"] = RewardTermCfg(
        func=mdp.motion_relative_body_position_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.1,
            "body_names": _VR_KEYPOINT_NAMES,
        },
    )

    # Foot-specific rewards: explicit foot placement drives stumble prevention.
    cfg.rewards["motion_feet_pos"] = RewardTermCfg(
        func=mdp.motion_relative_body_position_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.15,
            "body_names": _FOOT_BODY_NAMES,
        },
    )
    cfg.rewards["motion_feet_lin_vel"] = RewardTermCfg(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.5,
        params={
            "command_name": "motion",
            "std": 0.5,
            "body_names": _FOOT_BODY_NAMES,
        },
    )

    # Action smoothness: HOVER upper split (-0.625), v4-level leg (-0.5).
    # v5 lesson: leg=-3.0 from cold-start made legs too stiff to balance.
    cfg.rewards.pop("action_rate_l2", None)
    cfg.rewards["upper_body_action_rate_l2"] = RewardTermCfg(
        func=upper_body_action_rate_l2,
        weight=-0.625,
    )
    cfg.rewards["leg_action_rate_l2"] = RewardTermCfg(
        func=leg_action_rate_l2,
        weight=-0.5,
    )

    cfg.viewer.body_name = "Trunk"

    if not has_state_estimation:
        new_actor_terms = {
            k: v
            for k, v in cfg.observations["actor"].terms.items()
            if k not in ["motion_anchor_pos_b", "base_lin_vel"]
        }
        cfg.observations["actor"] = ObservationGroupCfg(
            terms=new_actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg
