"""Convert HOVER K1 pkl motion data to mjlab npz format.

mjlab motion format (npz keys):
  joint_pos    [T, 22]    joint angles in K1 joint order (rad)
  joint_vel    [T, 22]    joint velocities (rad/s)
  body_pos_w   [T, 23, 3] world-frame body link positions (entity body order, skips world)
  body_quat_w  [T, 23, 4] world-frame body quaternions wxyz (entity body order)
  body_lin_vel_w [T, 23, 3] world-frame body link linear velocities
  body_ang_vel_w [T, 23, 3] world-frame body angular velocities

Entity body order (0-indexed, body 0 = Trunk, ..., body 22 = right_foot_link).
Computed via FK on the K1 MJCF using MuJoCo, resampled to 50 Hz.
"""

import argparse
import re
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.interpolate import interp1d

K1_XML = Path(
    "/workspace/testing-grounds/projects/hover/neural_wbc/data/data/mujoco/models/k1.xml"
)

SRC_FPS = 30.0
DST_FPS = 50.0  # mjlab control freq (dt=0.005, decimation=4)

JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
]


def load_model() -> mujoco.MjModel:
    with open(K1_XML) as f:
        xml = f.read()
    xml = re.sub(r"<actuator>.*?</actuator>", "<actuator/>", xml, flags=re.DOTALL)
    spec = mujoco.MjSpec.from_string(xml)
    return spec.compile()


def resample(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample array along axis 0 from src_fps to dst_fps."""
    T_src = arr.shape[0]
    t_src = np.arange(T_src) / src_fps
    t_dst = np.arange(0, t_src[-1], 1.0 / dst_fps)
    shape_rest = arr.shape[1:]
    arr_flat = arr.reshape(T_src, -1)
    fn = interp1d(t_src, arr_flat, axis=0, kind="linear", fill_value="extrapolate")
    return fn(t_dst).reshape(-1, *shape_rest)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, w] → [w, x, y, z]."""
    return q[..., [3, 0, 1, 2]]


def finite_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """Central finite difference along axis 0, forward/backward at endpoints."""
    vel = np.empty_like(arr)
    vel[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt)
    vel[0] = (arr[1] - arr[0]) / dt
    vel[-1] = (arr[-1] - arr[-2]) / dt
    return vel


def quat_to_ang_vel(q: np.ndarray, dt: float) -> np.ndarray:
    """Approximate world-frame angular velocity from quaternion time series.

    q: [T, 4] in wxyz format.
    Returns omega: [T, 3] in world frame.
    """
    T = q.shape[0]
    omega = np.zeros((T, 3), dtype=np.float32)

    for t in range(1, T - 1):
        q1 = q[t - 1]
        q2 = q[t + 1]
        # dq ≈ q2 * conj(q1) in wxyz
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        # conj(q1) = (w1, -x1, -y1, -z1)
        dw = w2 * w1 + x2 * x1 + y2 * y1 + z2 * z1
        dx = -w2 * x1 + x2 * w1 - y2 * z1 + z2 * y1
        dy = -w2 * y1 + x2 * z1 + y2 * w1 - z2 * x1
        dz = -w2 * z1 - x2 * y1 + y2 * x1 + z2 * w1
        # omega ≈ 2 * [dx,dy,dz] / (2*dt)
        omega[t] = np.array([dx, dy, dz], dtype=np.float32) / (2 * dt)

    omega[0] = omega[1]
    omega[-1] = omega[-2]
    return omega


def compute_body_link_lin_vel(
    xpos: np.ndarray,    # [T, 23, 3] entity body link positions
    cvel: np.ndarray,    # [T, 24, 6] MuJoCo cvel (body 0=world)
    subtree_com: np.ndarray,  # [T, 3] root body subtree COM
) -> np.ndarray:
    """Compute world-frame link linear velocities matching mjlab's formula.

    mjlab: lin_vel_w = cvel[3:6] - cross(cvel[0:3], subtree_com - xpos)
    """
    # xpos has entity body indices (0-based), cvel has MuJoCo indices (1-based).
    ang_vel_c = cvel[:, 1:, 0:3]   # [T, 23, 3]
    lin_vel_c = cvel[:, 1:, 3:6]   # [T, 23, 3]
    offset = subtree_com[:, None, :] - xpos  # [T, 23, 3]
    cross = np.cross(ang_vel_c, offset)      # [T, 23, 3]
    return (lin_vel_c - cross).astype(np.float32)


def process_clip(
    clip: dict,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    dt: float,
) -> dict:
    """Process one motion clip dict → dict of resampled numpy arrays."""
    root_trans = clip["root_trans_offset"].astype(np.float64)  # [T, 3]
    root_rot_xyzw = clip["root_rot"].astype(np.float64)         # [T, 4] xyzw
    dof = clip["dof"].astype(np.float64)                         # [T, 22]

    root_rot_wxyz = quat_xyzw_to_wxyz(root_rot_xyzw)

    # Resample everything to DST_FPS.
    src_fps = float(clip.get("fps", SRC_FPS))
    root_trans = resample(root_trans, src_fps, DST_FPS)
    root_rot_wxyz = resample(root_rot_wxyz, src_fps, DST_FPS)
    dof = resample(dof, src_fps, DST_FPS)

    # Renormalise quaternions after interpolation.
    norms = np.linalg.norm(root_rot_wxyz, axis=1, keepdims=True)
    root_rot_wxyz = root_rot_wxyz / np.clip(norms, 1e-8, None)

    T = root_trans.shape[0]

    body_pos_w   = np.zeros((T, 23, 3), dtype=np.float32)
    body_quat_w  = np.zeros((T, 23, 4), dtype=np.float32)
    cvel_all     = np.zeros((T, 24, 6), dtype=np.float32)
    subtree_coms = np.zeros((T, 3),     dtype=np.float32)

    # Run FK + comVel for each timestep.
    dof_vel = finite_diff(dof, dt).astype(np.float64)
    root_lin_vel = finite_diff(root_trans, dt).astype(np.float64)

    # Root angular velocity in world frame via quaternion finite diff.
    root_ang_vel = quat_to_ang_vel(root_rot_wxyz, dt).astype(np.float64)

    for t in range(T):
        data.qpos[0:3] = root_trans[t]
        data.qpos[3:7] = root_rot_wxyz[t]
        data.qpos[7:29] = dof[t]

        data.qvel[0:3] = root_lin_vel[t]
        data.qvel[3:6] = root_ang_vel[t]
        data.qvel[6:28] = dof_vel[t]

        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)

        # Entity bodies skip world (index 0): entity body i = MuJoCo body i+1.
        body_pos_w[t]  = data.xpos[1:].copy().astype(np.float32)
        body_quat_w[t] = data.xquat[1:].copy().astype(np.float32)
        cvel_all[t]    = data.cvel.copy().astype(np.float32)

        # Root body (entity body 0 = MuJoCo body 1) subtree COM.
        subtree_coms[t] = data.subtree_com[1].copy().astype(np.float32)

    body_ang_vel_w = cvel_all[:, 1:, 0:3].astype(np.float32)  # [T, 23, 3]
    body_lin_vel_w = compute_body_link_lin_vel(body_pos_w, cvel_all, subtree_coms)

    # Joint pos/vel in entity joint order (same as MuJoCo joint 1..22).
    joint_pos = dof.astype(np.float32)
    joint_vel  = dof_vel.astype(np.float32)

    return {
        "joint_pos":     joint_pos,
        "joint_vel":     joint_vel,
        "body_pos_w":    body_pos_w,
        "body_quat_w":   body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert K1 pkl motion to mjlab npz")
    parser.add_argument(
        "--input", default=(
            "/workspace/testing-grounds/projects/hover"
            "/neural_wbc/data/data/motions/cmu_simple.pkl"
        ),
        help="Input pkl file",
    )
    parser.add_argument(
        "--output", default=(
            "/workspace/testing-grounds/projects/hover"
            "/neural_wbc/data/data/mujoco/motions/k1_cmu_simple.npz"
        ),
        help="Output npz file",
    )
    parser.add_argument("--max_clips", type=int, default=None, help="Limit clips for testing")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    clips = joblib.load(args.input)
    clip_names = list(clips.keys())
    if args.max_clips is not None:
        clip_names = clip_names[: args.max_clips]
    print(f"Found {len(clips)} clips, processing {len(clip_names)} ...")

    model = load_model()
    data = mujoco.MjData(model)
    dt = 1.0 / DST_FPS

    arrays = {k: [] for k in [
        "joint_pos", "joint_vel",
        "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w",
    ]}

    for i, name in enumerate(clip_names):
        clip = clips[name]
        print(f"  [{i+1}/{len(clip_names)}] {name} ({clip['root_trans_offset'].shape[0]} frames)")
        result = process_clip(clip, model, data, dt)
        for k in arrays:
            arrays[k].append(result[k])

    # Concatenate all clips.
    combined = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}

    T_total = combined["joint_pos"].shape[0]
    print(f"\nTotal frames: {T_total} ({T_total / DST_FPS:.1f} s at {DST_FPS:.0f} Hz)")
    print(f"joint_pos  : {combined['joint_pos'].shape}")
    print(f"body_pos_w : {combined['body_pos_w'].shape}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **combined)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
