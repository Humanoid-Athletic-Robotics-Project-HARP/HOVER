"""Kinova K1 humanoid robot constants for mjlab."""

import re
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

K1_XML: Path = Path(
    "/workspace/testing-grounds/projects/hover/neural_wbc/data/data/mujoco/models/k1.xml"
)
assert K1_XML.exists(), f"K1 XML not found at {K1_XML}"


def get_k1_spec() -> mujoco.MjSpec:
    """Load K1 MJCF with mjlab-compatible modifications:
    - Strip pre-existing motor actuators
    - Name geoms for collision/friction config (foot geoms get positional names)
    - Rename/add IMU sensors to match G1 naming (imu_ang_vel, imu_lin_vel)
    """
    with open(K1_XML) as f:
        xml = f.read()
    xml = re.sub(r"<actuator>.*?</actuator>", "<actuator/>", xml, flags=re.DOTALL)
    spec = mujoco.MjSpec.from_string(xml)

    # Name geoms so SceneEntityCfg regex can target foot vs. body geoms.
    foot_body_names = ("left_foot_link", "right_foot_link")
    for body in spec.bodies:
        if body.name in foot_body_names:
            side = "left" if "left" in body.name else "right"
            for i, geom in enumerate(body.geoms):
                geom.name = f"{side}_foot_{i + 1}_collision"
        else:
            for i, geom in enumerate(body.geoms):
                if geom.name == "":
                    geom.name = f"{body.name}_col_{i}"

    # Rename existing IMU sensors and add velocimeter to match G1 convention.
    for sensor in spec.sensors:
        if sensor.type == mujoco.mjtSensor.mjSENS_GYRO:
            sensor.name = "imu_ang_vel"
        elif sensor.type == mujoco.mjtSensor.mjSENS_ACCELEROMETER:
            sensor.name = "imu_lin_acc"

    vel_sensor = spec.add_sensor()
    vel_sensor.name = "imu_lin_vel"
    vel_sensor.type = mujoco.mjtSensor.mjSENS_VELOCIMETER
    vel_sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
    vel_sensor.objname = "imu"
    vel_sensor.reftype = mujoco.mjtObj.mjOBJ_UNKNOWN

    return spec


# Position-actuator gains calibrated to match G1's kp/effort ratio (~0.5-0.7).
# G1 uses kp=40 Nm/rad for hip pitch (effort=88) and kp=99 for knee (effort=139).
# K1 effort limits are 2-3× lower, so we scale down: hip kp≈15, knee kp≈30.
# This keeps zero-action PD error below ~15% of effort limit at typical RSI offsets,
# leaving room for policy authority (action scale = 0.25 × effort / kp).

K1_ACTUATOR_HEAD = BuiltinPositionActuatorCfg(
    target_names_expr=("AAHead_yaw", "Head_pitch"),
    stiffness=5.0,
    damping=0.5,
    effort_limit=6.0,
    armature=0.002,
)

K1_ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "ALeft_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "ARight_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
    ),
    stiffness=14.0,
    damping=1.5,
    effort_limit=14.0,
    armature=0.001,
)

K1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
    target_names_expr=("Left_Hip_Pitch", "Right_Hip_Pitch"),
    stiffness=15.0,
    damping=2.5,
    effort_limit=30.0,
    armature=0.048,
)

K1_ACTUATOR_HIP_ROLL_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "Left_Hip_Roll", "Left_Hip_Yaw",
        "Right_Hip_Roll", "Right_Hip_Yaw",
    ),
    stiffness=25.0,
    damping=4.0,
    effort_limit=35.0,
    armature=0.031,
)

K1_ACTUATOR_KNEES = BuiltinPositionActuatorCfg(
    target_names_expr=("Left_Knee_Pitch", "Right_Knee_Pitch"),
    stiffness=30.0,
    damping=5.0,
    effort_limit=40.0,
    armature=0.096,
)

K1_ACTUATOR_ANKLES = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ),
    stiffness=15.0,
    damping=2.5,
    effort_limit=20.0,
    armature=0.057,
)

K1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        K1_ACTUATOR_HEAD,
        K1_ACTUATOR_ARMS,
        K1_ACTUATOR_HIP_PITCH,
        K1_ACTUATOR_HIP_ROLL_YAW,
        K1_ACTUATOR_KNEES,
        K1_ACTUATOR_ANKLES,
    ),
    soft_joint_pos_limit_factor=0.9,
)

# Default standing keyframe matching HOVER's _DEFAULT_Q.
K1_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.62),
    joint_pos={
        # Arms match reference motion mean (CMU data): shoulder_roll ≈ ±1.2 rad
        "Left_Shoulder_Roll": -1.2,
        "Right_Shoulder_Roll": 1.2,
        # Legs: reference motion has ankle_pitch=0 always (no ankle DOF in CMU→K1 transfer)
        "Left_Hip_Pitch": -0.25,
        "Left_Knee_Pitch": 0.5,
        "Left_Ankle_Pitch": 0.0,
        "Right_Hip_Pitch": -0.25,
        "Right_Knee_Pitch": 0.5,
        "Right_Ankle_Pitch": 0.0,
    },
    joint_vel={".*": 0.0},
)

# Collision config: foot geoms get condim=3; all others get condim=1.
# Foot geom names are set by get_k1_spec(): left_foot_1_collision, right_foot_1_collision.
K1_FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot_1_collision$", ".*_col_.*"),
    condim={
        r"^(left|right)_foot_1_collision$": 3,
        ".*_col_.*": 1,
    },
    priority={r"^(left|right)_foot_1_collision$": 1},
    friction={r"^(left|right)_foot_1_collision$": (0.6,)},
)


def get_k1_robot_cfg() -> EntityCfg:
    """Return a fresh K1 robot EntityCfg."""
    return EntityCfg(
        init_state=K1_HOME_KEYFRAME,
        collisions=(K1_FULL_COLLISION,),
        spec_fn=get_k1_spec,
        articulation=K1_ARTICULATION,
    )


# Action scale: 0.25 * effort_limit / stiffness  (same formula as G1).
K1_ACTION_SCALE: dict[str, float] = {}
for _act in K1_ARTICULATION.actuators:
    assert isinstance(_act, BuiltinPositionActuatorCfg)
    _e = _act.effort_limit
    _s = _act.stiffness
    assert _e is not None
    for _n in _act.target_names_expr:
        K1_ACTION_SCALE[_n] = 0.25 * _e / _s
