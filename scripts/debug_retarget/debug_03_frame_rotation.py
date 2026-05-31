"""Debug 3 — Frame Rotation Effect on Arms
The fk_frame_rotation quaternion aligns SMPL's coordinate frame to the K1 MuJoCo frame.
If it's wrong for the arms (even if correct for the legs), arm poses will be systematically off.

This script generates synthetic ball-joint shoulder rotations (as if from SMPL) and shows
what the K1 arm looks like when you apply different fk_frame_rotation values.
Side-by-side: identity rotation vs [0.5,0.5,0.5,0.5] (the typical AMASS→K1 correction).

Ask: Does the arm move in the expected direction under the current frame correction?
     If raising the arm forward in SMPL maps to raising sideways on K1 → wrong frame rotation.

Outputs:
  debug_output/03_frame_rotation.gif      — side-by-side comparison, L=identity, R=corrected
  debug_output/03_frame_rotation_error.png — wrist position difference for each test rotation
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import mujoco
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SCENE_XML = HERE.parent.parent / "neural_wbc/data/data/mujoco/models/scene_k1_vis.xml"
OUT = HERE / "debug_output"
OUT.mkdir(exist_ok=True)

# The quaternion used in the hover codebase as fk_frame_rotation (wxyz)
FK_FRAME_ROTATION_WXYZ = np.array([0.5, 0.5, 0.5, 0.5])

LEFT_ARM_JOINTS = [
    "ALeft_Shoulder_Pitch",   # Y-axis
    "Left_Shoulder_Roll",     # X-axis
    "Left_Elbow_Pitch",       # Y-axis
    "Left_Elbow_Yaw",         # Z-axis
]


# ── quaternion helpers ────────────────────────────────────────────────────────

def quat_wxyz_to_mat(q):
    """wxyz → 3×3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def axis_angle_to_mat(axis, angle):
    """Rodrigues' formula."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def decompose_yx(R):
    """Decompose R = Ry(pitch) * Rx(roll) → (pitch, roll)."""
    # R[2,0] = -sin(pitch),  R[0,0] = cos(pitch)*cos(roll) [ignore roll for atan2]
    # More robust: R[1,2] = -sin(roll), R[1,1] = cos(roll)
    roll  = np.arctan2(-R[1, 2], R[1, 1])
    pitch = np.arctan2(-R[2, 0], R[0, 0])
    return pitch, roll


def apply_frame_rotation(R_smpl, R_frame):
    """Rotate a SMPL joint rotation into the robot frame."""
    return R_frame @ R_smpl @ R_frame.T


def clamp(val, lo, hi):
    return float(np.clip(val, lo, hi))


# ── MuJoCo helpers ───────────────────────────────────────────────────────────

def make_camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def set_arm_from_rotation(model, data, adrs, limits, R_shoulder, elbow_pitch=0.0):
    """Decompose R_shoulder → Pitch + Roll, set arm joints."""
    pitch, roll = decompose_yx(R_shoulder)
    j_limits = {a: (lo, hi) for a, (lo, hi) in zip(adrs, limits)}
    data.qpos[adrs[0]] = clamp(pitch, *limits[0])
    data.qpos[adrs[1]] = clamp(roll,  *limits[1])
    data.qpos[adrs[2]] = clamp(elbow_pitch, *limits[2])
    data.qpos[adrs[3]] = 0.0


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


def side_by_side(left_img, right_img, left_label, right_label):
    """Concatenate two images horizontally with labels."""
    h = max(left_img.shape[0], right_img.shape[0])
    combined = np.concatenate([left_img, right_img], axis=1)
    img = Image.fromarray(combined)
    draw = ImageDraw.Draw(img)
    w = left_img.shape[1]
    for x, label in [(10, left_label), (w + 10, right_label)]:
        draw.text((x + 1, h - 22), label, fill=(0, 0, 0))
        draw.text((x, h - 23), label, fill=(255, 220, 0))
    return np.array(img)


# ── Test rotations (SMPL ball-joint shoulder, expressed as axis+angle) ────────

TEST_ROTATIONS = [
    # (description, axis, angle_deg)  — these are in the SMPL/AMASS world frame
    ("Arm forward  (+X in SMPL)",   [1, 0, 0],  60),
    ("Arm sideways (+Y in SMPL)",   [0, 1, 0],  60),
    ("Arm upward   (+Z in SMPL)",   [0, 0, 1],  60),
    ("Diagonal XY",                 [1, 1, 0],  60),
    ("Diagonal XZ",                 [1, 0, 1],  60),
    ("Diagonal YZ",                 [0, 1, 1],  60),
    ("Full diagonal",               [1, 1, 1],  50),
    ("Arm forward large",           [1, 0, 0], 100),
    ("Arm sideways large",          [0, 1, 0], 100),
    ("Arm upward large",            [0, 0, 1],  90),
]


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    home_qpos = data.qpos.copy()

    jids    = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_ARM_JOINTS]
    adrs    = [model.jnt_qposadr[jid] for jid in jids]
    limits  = [tuple(model.jnt_range[jid]) for jid in jids]
    hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_link")

    R_identity = np.eye(3)
    R_frame    = quat_wxyz_to_mat(FK_FRAME_ROTATION_WXYZ)

    renderer = mujoco.Renderer(model, height=360, width=480)
    cam = make_camera([0.0, 0.0, 0.88], 2.2, 75, -12)

    frames = []
    errors = []  # (desc, wrist_pos_identity, wrist_pos_corrected, diff_norm)

    for desc, axis, angle_deg in TEST_ROTATIONS:
        R_smpl = axis_angle_to_mat(axis, np.radians(angle_deg))

        # --- identity (no frame correction) ---
        data.qpos[:] = home_qpos
        set_arm_from_rotation(model, data, adrs, limits, R_identity @ R_smpl)
        img_identity = render_frame(model, data, renderer, cam)
        mujoco.mj_forward(model, data)
        wrist_no_corr = data.xpos[hand_bid].copy()
        pitch_id, roll_id = decompose_yx(R_identity @ R_smpl)

        # --- with fk_frame_rotation ---
        data.qpos[:] = home_qpos
        R_corrected = apply_frame_rotation(R_smpl, R_frame)
        set_arm_from_rotation(model, data, adrs, limits, R_corrected)
        img_corrected = render_frame(model, data, renderer, cam)
        mujoco.mj_forward(model, data)
        wrist_corr = data.xpos[hand_bid].copy()
        pitch_co, roll_co = decompose_yx(R_corrected)

        diff = np.linalg.norm(wrist_corr - wrist_no_corr)
        errors.append((desc, wrist_no_corr, wrist_corr, diff))

        # Annotate each side
        img_identity  = overlay(img_identity, [
            "No frame correction", f"pitch={np.degrees(pitch_id):+.1f}°  roll={np.degrees(roll_id):+.1f}°"])
        img_corrected = overlay(img_corrected, [
            "With fk_frame_rotation=[0.5,0.5,0.5,0.5]",
            f"pitch={np.degrees(pitch_co):+.1f}°  roll={np.degrees(roll_co):+.1f}°"])

        combined = side_by_side(img_identity, img_corrected,
                                "identity", "corrected")
        combined = overlay(combined, [desc, f"wrist diff: {diff*100:.1f} cm"], color=(255, 220, 50))
        frames.append(combined)

    gif_path = OUT / "03_frame_rotation.gif"
    imageio.mimsave(str(gif_path), frames, fps=2, loop=0)
    print(f"Saved {gif_path}  ({len(frames)} frames)")

    # --- error bar chart ---
    fig, ax = plt.subplots(figsize=(11, 6))
    descs = [e[0] for e in errors]
    diffs = [e[3] * 100 for e in errors]  # cm
    colors = ["#E53935" if d > 5 else "#43A047" for d in diffs]
    bars = ax.barh(range(len(descs)), diffs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(descs)))
    ax.set_yticklabels(descs, fontsize=9)
    ax.set_xlabel("Wrist position difference: identity vs fk_frame_rotation (cm)")
    ax.set_title("Frame rotation effect on left wrist position\n"
                 "Large bar = correction substantially changes arm pose\n"
                 "Small bar = correction has little effect (may be misconfigured)")
    ax.axvline(5, color="orange", linewidth=1.5, linestyle="--", label="5 cm threshold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, diffs):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f} cm", va="center", fontsize=8)
    plt.tight_layout()
    png_path = OUT / "03_frame_rotation_error.png"
    plt.savefig(str(png_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")

    renderer.close()
    print("\nKey question: if all bars are small → frame rotation has no effect on arm "
          "(decomposition ignores the 3rd DOF regardless).\n"
          "If bars are large only for Z/diagonal rotations → the yaw component is being "
          "absorbed into the wrong joints.")


if __name__ == "__main__":
    main()
