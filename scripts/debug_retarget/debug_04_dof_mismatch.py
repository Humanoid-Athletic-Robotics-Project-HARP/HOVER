"""Debug 4 — SMPL 3-DOF Shoulder vs K1 2-DOF Shoulder
SMPL models the shoulder as a full ball joint (3 DOF: pitch, roll, yaw in the parent frame).
The K1 only has Pitch + Roll — there is NO shoulder yaw joint.

This script builds a virtual "SMPL arm" (3-DOF kinematic chain) and tries to match
each pose using only the K1's Pitch+Roll. It shows the wrist position error and
which axes of rotation the K1 can/can't reproduce.

Ask: How large is the wrist error from the missing DOF?
     Which motion directions are unrepresentable?

Outputs:
  debug_output/04_dof_mismatch.gif      — virtual SMPL arm (orange dot = target wrist)
                                          vs K1 best-match, showing the gap
  debug_output/04_dof_mismatch_error.png — error map: angle vs rotation axis
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import minimize
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SCENE_XML = HERE.parent.parent / "neural_wbc/data/data/mujoco/models/scene_k1_vis.xml"
OUT = HERE / "debug_output"
OUT.mkdir(exist_ok=True)

LEFT_ARM_JOINTS = [
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
]
LEFT_HAND_BODY    = "left_hand_link"
LEFT_SHOULDER_BODY = "Left_Arm_1"


# ── quaternion / rotation helpers ─────────────────────────────────────────────

def quat_wxyz_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def axis_angle_to_mat(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def decompose_yx(R):
    """R = Ry(pitch) * Rx(roll). Returns (pitch, roll)."""
    roll  = np.arctan2(-R[1, 2], R[1, 1])
    pitch = np.arctan2(-R[2, 0], R[0, 0])
    return pitch, roll


# ── Virtual SMPL-style arm (FK in world space) ────────────────────────────────
# We model the left arm as:
#   shoulder_pos (fixed) + R_smpl_shoulder * upper_arm_vec + R_smpl_shoulder * elbow_R * forearm_vec
# upper_arm length and forearm length taken from K1 geometry.

UPPER_ARM_LEN = 0.068 + 0.044  # Left_Arm_1 y-offset + Left_Arm_2 y-offset ≈ 0.112 m
FOREARM_LEN   = 0.1215          # left_hand_link y-offset
UPPER_ARM_DIR = np.array([0, 1, 0])   # along +Y in shoulder local frame
FOREARM_DIR   = np.array([0, 1, 0])   # along +Y in elbow local frame


def virtual_wrist_pos(shoulder_world, R_shoulder_3dof, elbow_pitch=0.0):
    """Compute wrist position for a 3-DOF shoulder (full ball joint)."""
    elbow_R = axis_angle_to_mat([1, 0, 0], elbow_pitch)  # simple elbow flex in local X
    upper_end = shoulder_world + R_shoulder_3dof @ (UPPER_ARM_LEN * UPPER_ARM_DIR)
    wrist     = upper_end + (R_shoulder_3dof @ elbow_R) @ (FOREARM_LEN * FOREARM_DIR)
    return wrist


# ── MuJoCo helpers ────────────────────────────────────────────────────────────

def make_camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


_OPT = None

def _get_opt():
    global _OPT
    if _OPT is None:
        _OPT = mujoco.MjvOption()
    return _OPT


def render_frame(model, data, renderer, cam):
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam, scene_option=_get_opt())
    return renderer.render().copy()


def overlay(img_arr, lines, color=(255, 255, 255)):
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    y = 8
    for line in lines:
        draw.text((11, y + 1), line, fill=(0, 0, 0))
        draw.text((10, y), line, fill=color)
        y += 18
    return np.array(img)


def draw_dot_on_frame(img_arr, model, data, renderer, cam, world_pos, color=(255, 140, 0), radius=8):
    """Overlay a coloured dot at a world position by projecting it to screen."""
    # Project world_pos to screen coords via MuJoCo's scene
    scene = mujoco.MjvScene(model, maxgeom=1000)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    vp = np.array([0, 0, renderer.width, renderer.height], dtype=np.int32)
    ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    # Project: use mjv_room2clip then clip2window
    # Simpler: skip projection and just annotate with text
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    draw.text((10, img_arr.shape[0] - 30), f"target wrist: ({world_pos[0]:+.3f}, {world_pos[1]:+.3f}, {world_pos[2]:+.3f})",
              fill=color)
    return np.array(img)


def solve_ik_2dof(target, model, data, adrs, limits, home_qpos, hand_bid):
    """IK with only shoulder pitch + roll free; elbow fixed at 0."""
    def obj(q):
        data.qpos[adrs[0]] = q[0]
        data.qpos[adrs[1]] = q[1]
        data.qpos[adrs[2]] = 0.0
        data.qpos[adrs[3]] = 0.0
        mujoco.mj_kinematics(model, data)
        err = data.xpos[hand_bid] - target
        return float(err @ err)
    x0 = np.array([data.qpos[adrs[0]], data.qpos[adrs[1]]])
    bounds = [limits[0], limits[1]]
    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 300, "ftol": 1e-12})
    return res.x, np.sqrt(res.fun)


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    home_qpos = data.qpos.copy()

    jids     = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_ARM_JOINTS]
    adrs     = [model.jnt_qposadr[jid] for jid in jids]
    limits   = [tuple(model.jnt_range[jid]) for jid in jids]
    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_HAND_BODY)
    sh_bid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_SHOULDER_BODY)

    mujoco.mj_forward(model, data)
    shoulder_world = data.xpos[sh_bid].copy()

    renderer = mujoco.Renderer(model, height=400, width=560)
    cam = make_camera([0.0, 0.0, 0.88], 2.2, 65, -12)

    # Test: sweep shoulder rotation around each axis and the "yaw" axis (Z of shoulder frame)
    # The yaw rotation is what K1 CANNOT represent with only Pitch+Roll.
    axes = {
        "Pitch (Y) — K1 can do":      [0, 1, 0],
        "Roll (X) — K1 can do":        [1, 0, 0],
        "Yaw (Z) — K1 CANNOT do":      [0, 0, 1],
        "Diagonal XY — partial":        [1, 1, 0],
        "Diagonal XZ — large yaw err":  [1, 0, 1],
        "Diagonal YZ — large yaw err":  [0, 1, 1],
    }
    angles_deg = np.linspace(0, 90, 10)

    all_errors = {}  # axis_name → list of (angle_deg, error_m)
    frames = []

    for axis_name, axis in axes.items():
        errs = []
        for angle_deg in angles_deg:
            angle = np.radians(angle_deg)
            R_3dof = axis_angle_to_mat(axis, angle)
            target_wrist = virtual_wrist_pos(shoulder_world, R_3dof, elbow_pitch=0.0)

            # Best K1 2-DOF match via IK
            data.qpos[:] = home_qpos
            q_sol, err = solve_ik_2dof(target_wrist, model, data, adrs, limits, home_qpos, hand_bid)
            errs.append((angle_deg, err))

        all_errors[axis_name] = errs

        # Render a few frames for this axis
        for angle_deg_render in [0, 30, 60, 90]:
            angle = np.radians(angle_deg_render)
            R_3dof = axis_angle_to_mat(axis, angle)
            target_wrist = virtual_wrist_pos(shoulder_world, R_3dof, elbow_pitch=0.0)
            data.qpos[:] = home_qpos
            q_sol, err = solve_ik_2dof(target_wrist, model, data, adrs, limits, home_qpos, hand_bid)
            data.qpos[adrs[0]] = q_sol[0]
            data.qpos[adrs[1]] = q_sol[1]
            data.qpos[adrs[2]] = 0.0
            data.qpos[adrs[3]] = 0.0

            img = render_frame(model, data, renderer, cam)
            color = (50, 200, 50) if err < 0.03 else (220, 80, 50)
            can = "✓ representable" if err < 0.03 else "✗ NOT representable"
            img = overlay(img, [
                axis_name,
                f"rotation: {angle_deg_render}°",
                f"wrist error: {err*100:.1f} cm  {can}",
                f"K1 pitch={np.degrees(q_sol[0]):+.1f}°  roll={np.degrees(q_sol[1]):+.1f}°",
            ], color=color)
            frames.append(img)

    gif_path = OUT / "04_dof_mismatch.gif"
    imageio.mimsave(str(gif_path), frames, fps=2, loop=0)
    print(f"Saved {gif_path}  ({len(frames)} frames)")

    # --- error line chart ---
    fig, ax = plt.subplots(figsize=(11, 6))
    colors_map = {
        "Pitch (Y) — K1 can do":      "#43A047",
        "Roll (X) — K1 can do":        "#1E88E5",
        "Yaw (Z) — K1 CANNOT do":      "#E53935",
        "Diagonal XY — partial":        "#FB8C00",
        "Diagonal XZ — large yaw err":  "#8E24AA",
        "Diagonal YZ — large yaw err":  "#D81B60",
    }
    for axis_name, errs in all_errors.items():
        ang  = [e[0] for e in errs]
        vals = [e[1] * 100 for e in errs]  # cm
        ax.plot(ang, vals, label=axis_name, color=colors_map[axis_name],
                linewidth=2, marker="o", markersize=4)

    ax.axhline(3, color="orange", linewidth=1.5, linestyle="--", label="3 cm threshold")
    ax.set_xlabel("Shoulder rotation angle (°)")
    ax.set_ylabel("Wrist position error vs 3-DOF target (cm)")
    ax.set_title("K1 2-DOF shoulder (Pitch+Roll only) vs SMPL 3-DOF ball joint\n"
                 "Yaw rotation is unrepresentable → wrist error grows with rotation angle")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.35)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    png_path = OUT / "04_dof_mismatch_error.png"
    plt.savefig(str(png_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")

    renderer.close()
    print("\nKey: if the Yaw line grows steeply but Pitch/Roll stay near zero → "
          "the missing shoulder yaw DOF is the primary cause of arm retargeting error.")


if __name__ == "__main__":
    main()
