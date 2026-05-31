"""Debug 1 — Joint Range of Motion
Sweeps each arm joint through its full range while holding everything else at the home pose.
Ask: Can each joint actually move? Does the motion look physically right?

Outputs:
  debug_output/01_joint_rom.gif   — animation cycling through each joint sweep
  debug_output/01_joint_limits.png — bar chart of arm joint ranges
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

# Each entry: joint_name → (camera_args, {extra_joint: angle} pre-pose)
# The elbow joints need the shoulder pre-set to a non-zero angle first,
# because at home pose the forearm is aligned with the joint's Y rotation axis,
# making elbow flexion produce zero visible motion (spinning around its own axis).
ARM_JOINT_CONFIGS = [
    # (joint_name, camera(lookat, dist, azimuth, elev), pre_pose_overrides)
    ("ALeft_Shoulder_Pitch",  ([0.0,  0.0, 0.88], 2.2,  80, -12), {}),
    ("Left_Shoulder_Roll",    ([0.0,  0.0, 0.88], 2.2,  20, -12), {}),
    ("Left_Elbow_Pitch",      ([0.0,  0.0, 0.88], 2.2,  60, -12),
        # pre-pitch the shoulder so arm points forward → elbow flexion now visible
        {"ALeft_Shoulder_Pitch": -1.2}),
    ("Left_Elbow_Yaw",        ([0.0,  0.0, 0.88], 2.2,  60, -12),
        {"ALeft_Shoulder_Pitch": -1.2}),
    ("ARight_Shoulder_Pitch", ([0.0,  0.0, 0.88], 2.2, 100, -12), {}),
    ("Right_Shoulder_Roll",   ([0.0,  0.0, 0.88], 2.2, 160, -12), {}),
    ("Right_Elbow_Pitch",     ([0.0,  0.0, 0.88], 2.2, 120, -12),
        {"ARight_Shoulder_Pitch": -1.2}),
    ("Right_Elbow_Yaw",       ([0.0,  0.0, 0.88], 2.2, 120, -12),
        {"ARight_Shoulder_Pitch": -1.2}),
]


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


def render(model, data, renderer, cam):
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


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    home_qpos = data.qpos.copy()

    renderer = mujoco.Renderer(model, height=480, width=640)

    N = 24  # frames per sweep (min→max→min)
    frames = []

    for jname, cam_args, pre_pose in ARM_JOINT_CONFIGS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        adr = model.jnt_qposadr[jid]
        lo, hi = model.jnt_range[jid]
        angles = np.concatenate([np.linspace(lo, hi, N // 2),
                                 np.linspace(hi, lo, N // 2)])
        cam = make_camera(*cam_args)

        # Build the base pose for this joint's sweep (home + any pre-pose overrides)
        base_qpos = home_qpos.copy()
        for pre_jname, pre_angle in pre_pose.items():
            pre_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pre_jname)
            base_qpos[model.jnt_qposadr[pre_jid]] = pre_angle

        pre_note = (f"  (shoulder pre-set to {list(pre_pose.values())[0]:.1f} rad)"
                    if pre_pose else "")

        for angle in angles:
            data.qpos[:] = base_qpos
            data.qpos[adr] = angle
            img = render(model, data, renderer, cam)
            pct = (angle - lo) / (hi - lo + 1e-9) * 100
            img = overlay(img, [
                jname + pre_note,
                f"angle: {angle:+.3f} rad  ({pct:.0f}% of range)",
                f"limits: [{lo:.3f}, {hi:.3f}] rad",
            ])
            frames.append(img)

    gif_path = OUT / "01_joint_rom.gif"
    imageio.mimsave(str(gif_path), frames, fps=12, loop=0)
    print(f"Saved {gif_path}  ({len(frames)} frames)")

    # --- limit bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1565C0"] * 4 + ["#B71C1C"] * 4
    for i, (jname, _, _) in enumerate(ARM_JOINT_CONFIGS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        lo, hi = model.jnt_range[jid]
        ax.barh(i, hi - lo, left=lo, color=colors[i], alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.text((lo + hi) / 2, i, f"[{lo:.2f}, {hi:.2f}] ({np.degrees(hi-lo):.0f}°)",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    jnames = [cfg[0] for cfg in ARM_JOINT_CONFIGS]
    ax.set_yticks(range(len(jnames)))
    ax.set_yticklabels(jnames, fontsize=9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Joint angle (rad)")
    ax.set_title("K1 arm joint limits  —  blue = left, red = right\n"
                 "Note: shoulder has only 2 DOF (Pitch + Roll), no Yaw")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    png_path = OUT / "01_joint_limits.png"
    plt.savefig(str(png_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")

    renderer.close()


if __name__ == "__main__":
    main()
