"""Debug 5 — Joint Limit Saturation
Simulates typical arm motions (reaching forward, sideways, across body, overhead)
and shows which joints hit their limits.  If a joint is frequently clamped,
retargeting will produce stiff/frozen behaviour for that DOF.

Optionally loads a retargeted motion PKL (pass --pkl path/to/motion.pkl).
Without a PKL, uses synthetic reaching trajectories that stress the arm.

Joint colour coding in the GIF:
  green  — within 20% of range from center
  yellow — within 10% of limit
  red    — at or beyond limit (clamped)

Outputs:
  debug_output/05_limits.gif      — animated robot with colour-coded joints
  debug_output/05_limits_plot.png — time-series of arm joint angles with limit bands
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path
import mujoco
import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
RIGHT_ARM_JOINTS = [
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]
ALL_ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS


_OPT = None

def _get_opt():
    global _OPT
    if _OPT is None:
        _OPT = mujoco.MjvOption()
    return _OPT


def make_camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def overlay(img_arr, lines, color=(255, 255, 255)):
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    y = 8
    for line in lines:
        draw.text((11, y + 1), line, fill=(0, 0, 0))
        draw.text((10, y), line, fill=color)
        y += 18
    return np.array(img)


def limit_color(val, lo, hi):
    """Return an RGB tuple: green=safe, yellow=near limit, red=at limit."""
    span = hi - lo
    margin = 0.10 * span
    if val <= lo + margin or val >= hi - margin:
        return (220, 50, 50)    # red — at limit
    if val <= lo + 0.20 * span or val >= hi - 0.20 * span:
        return (240, 180, 0)    # yellow — close
    return (50, 200, 50)        # green — safe


def draw_joint_status(img_arr, joint_vals, joint_names, limits):
    """Overlay a small status bar for each arm joint."""
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    x0, y0, bar_w, bar_h = 8, img_arr.shape[0] - 10 - len(joint_vals) * 14, 160, 10
    for i, (jname, val, (lo, hi)) in enumerate(zip(joint_names, joint_vals, limits)):
        y = y0 + i * 14
        # Background bar
        draw.rectangle([x0, y, x0 + bar_w, y + bar_h], fill=(60, 60, 60))
        # Filled portion
        pct = (val - lo) / (hi - lo + 1e-9)
        pct_clamped = max(0.0, min(1.0, pct))
        fill_w = int(pct_clamped * bar_w)
        color = limit_color(val, lo, hi)
        draw.rectangle([x0, y, x0 + fill_w, y + bar_h], fill=color)
        # Label
        short = jname.replace("ALeft_", "L_").replace("Left_", "L_").replace("ARight_", "R_").replace("Right_", "R_")
        draw.text((x0 + bar_w + 4, y), f"{short}: {np.degrees(val):+.0f}°", fill=(220, 220, 220))
    return np.array(img)


# ── Synthetic motion generation ───────────────────────────────────────────────

def sinusoid(lo, hi, t, freq=1.0, phase=0.0):
    mid = (lo + hi) / 2
    amp = (hi - lo) / 2
    return mid + amp * np.sin(2 * np.pi * freq * t + phase)


def make_synthetic_trajectories(limits_left, limits_right, n_frames=180):
    """Generate stress-test arm trajectories that push into corners of the joint space."""
    T = np.linspace(0, 1, n_frames)

    # Scenario: sweep shoulder pitch while modulating roll; elbow flexion follows
    lo_l = [lim[0] for lim in limits_left]
    hi_l = [lim[1] for lim in limits_left]
    lo_r = [lim[0] for lim in limits_right]
    hi_r = [lim[1] for lim in limits_right]

    traj_left = np.zeros((n_frames, 4))
    traj_right = np.zeros((n_frames, 4))

    for i, t in enumerate(T):
        # Left arm: reach forward and across body (stresses shoulder pitch and roll)
        traj_left[i, 0] = sinusoid(lo_l[0], hi_l[0] * 0.9, t, freq=0.7, phase=0)        # shoulder pitch full sweep
        traj_left[i, 1] = sinusoid(lo_l[1], hi_l[1], t, freq=1.3, phase=np.pi / 3)       # shoulder roll
        traj_left[i, 2] = sinusoid(0, hi_l[2] * 0.8, t, freq=1.0, phase=np.pi / 2)       # elbow pitch
        traj_left[i, 3] = sinusoid(lo_l[3], 0, t, freq=0.5, phase=np.pi)                  # elbow yaw

        # Right arm: mirrored + phase shifted
        traj_right[i, 0] = sinusoid(lo_r[0], hi_r[0] * 0.9, t, freq=0.9, phase=np.pi / 4)
        traj_right[i, 1] = sinusoid(lo_r[1], hi_r[1], t, freq=1.1, phase=np.pi * 0.7)
        traj_right[i, 2] = sinusoid(0, hi_r[2] * 0.8, t, freq=1.2, phase=0)
        traj_right[i, 3] = sinusoid(0, hi_r[3], t, freq=0.6, phase=np.pi * 1.5)

    return traj_left, traj_right


def load_pkl_trajectories(pkl_path, model, adrs_left, adrs_right, max_frames=200):
    """Load arm joint angles from a retargeted motion PKL."""
    try:
        import joblib
    except ImportError:
        print("[WARN] joblib not installed, cannot load PKL. Using synthetic motion.")
        return None, None

    data_pkl = joblib.load(pkl_path)
    first_key = next(iter(data_pkl))
    clip = data_pkl[first_key]

    if "dof_pos" not in clip:
        print("[WARN] PKL clip missing 'dof_pos'. Using synthetic motion.")
        return None, None

    dof_pos = np.array(clip["dof_pos"])  # (T, n_joints)
    n_frames = min(len(dof_pos), max_frames)
    # Map qpos addresses to PKL joint indices (assume same order as MuJoCo)
    # This is a best-effort mapping; may need adjustment for specific PKL formats.
    traj_left  = dof_pos[:n_frames, [a - 7 for a in adrs_left]]   # subtract freejoint offset
    traj_right = dof_pos[:n_frames, [a - 7 for a in adrs_right]]
    return traj_left, traj_right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, default=None,
                        help="Path to retargeted motion PKL (optional). "
                             "If not provided, synthetic trajectories are used.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    home_qpos = data.qpos.copy()

    def jinfo(names):
        jids  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        adrs  = [model.jnt_qposadr[jid] for jid in jids]
        lims  = [tuple(model.jnt_range[jid]) for jid in jids]
        return adrs, lims

    adrs_l, lims_l = jinfo(LEFT_ARM_JOINTS)
    adrs_r, lims_r = jinfo(RIGHT_ARM_JOINTS)

    # Load or synthesise trajectories
    traj_l, traj_r = None, None
    if args.pkl:
        traj_l, traj_r = load_pkl_trajectories(args.pkl, model, adrs_l, adrs_r)
    if traj_l is None:
        print("Using synthetic stress-test trajectories...")
        traj_l, traj_r = make_synthetic_trajectories(lims_l, lims_r, n_frames=160)

    n_frames = len(traj_l)

    # Clamp trajectories to limits (what the retargeting pipeline would do)
    traj_l_clamped = np.clip(traj_l, [lo for lo, _ in lims_l], [hi for _, hi in lims_l])
    traj_r_clamped = np.clip(traj_r, [lo for lo, _ in lims_r], [hi for _, hi in lims_r])

    # Track saturation: fraction of time each joint is clamped
    saturated_l = np.abs(traj_l - traj_l_clamped) > 1e-6
    saturated_r = np.abs(traj_r - traj_r_clamped) > 1e-6

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = make_camera([0.0, 0.0, 0.88], 2.5, 120, -15)

    frames = []
    stride = max(1, n_frames // 80)   # keep GIF under ~80 frames

    for i in range(0, n_frames, stride):
        data.qpos[:] = home_qpos
        for adr, val in zip(adrs_l, traj_l_clamped[i]):
            data.qpos[adr] = val
        for adr, val in zip(adrs_r, traj_r_clamped[i]):
            data.qpos[adr] = val

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=_get_opt())
        img = renderer.render().copy()

        # Joint status bars
        img = draw_joint_status(img, traj_l_clamped[i], LEFT_ARM_JOINTS, lims_l)

        # Count saturations
        n_sat_l = int(saturated_l[i].sum())
        n_sat_r = int(saturated_r[i].sum())
        t_pct = i / n_frames * 100
        img = overlay(img, [
            f"frame {i}/{n_frames}  ({t_pct:.0f}%)",
            f"L arm saturated joints: {n_sat_l}/4   R arm: {n_sat_r}/4",
        ], color=(255, 220, 50) if (n_sat_l + n_sat_r) > 0 else (180, 255, 180))
        frames.append(img)

    gif_path = OUT / "05_limits.gif"
    imageio.mimsave(str(gif_path), frames, fps=12, loop=0)
    print(f"Saved {gif_path}  ({len(frames)} frames)")

    # --- time-series plot ---
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    T = np.arange(n_frames)
    colors_l = ["#1565C0", "#0288D1", "#00838F", "#00695C"]
    colors_r = ["#B71C1C", "#C62828", "#AD1457", "#6A1B9A"]

    for col, (traj_raw, traj_cl, lims, jnames, cols, side) in enumerate([
        (traj_l, traj_l_clamped, lims_l, LEFT_ARM_JOINTS, colors_l, "Left"),
        (traj_r, traj_r_clamped, lims_r, RIGHT_ARM_JOINTS, colors_r, "Right"),
    ]):
        for row, (jname, lo_hi, c) in enumerate(zip(jnames, lims, cols)):
            lo, hi = lo_hi
            ax = axes[row, col]
            ax.fill_between(T, lo, hi, alpha=0.08, color=c, label="valid range")
            ax.axhline(lo, color="red", linewidth=0.8, linestyle="--")
            ax.axhline(hi, color="red", linewidth=0.8, linestyle="--")
            ax.plot(T, traj_raw[:, row], color=c, linewidth=1.2, alpha=0.5, label="raw (before clamp)")
            ax.plot(T, traj_cl[:, row], color=c, linewidth=1.8, label="clamped")

            # Highlight saturated regions
            sat = np.abs(traj_raw[:, row] - traj_cl[:, row]) > 1e-6
            if sat.any():
                ax.fill_between(T, lo, hi, where=sat, color="red", alpha=0.25, label="CLAMPED")

            short = jname.replace("ALeft_", "").replace("Left_", "").replace("ARight_", "").replace("Right_", "")
            sat_pct = sat.mean() * 100
            ax.set_ylabel(f"{short}\n(rad)", fontsize=7)
            ax.set_title(f"{side} {short}  — clamped {sat_pct:.0f}% of time",
                         fontsize=8, color="red" if sat_pct > 10 else "black")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.legend(fontsize=6, loc="upper right")

    axes[-1, 0].set_xlabel("Frame")
    axes[-1, 1].set_xlabel("Frame")
    fig.suptitle("K1 arm joint angles vs limits  —  red shading = clamped (retargeting failure)\n"
                 "Dashed red lines = joint limits", fontsize=10)
    plt.tight_layout()
    png_path = OUT / "05_limits_plot.png"
    plt.savefig(str(png_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")

    renderer.close()

    print("\nSaturation summary:")
    for jname, sat in zip(LEFT_ARM_JOINTS, saturated_l.mean(axis=0)):
        print(f"  {jname:35s}  clamped {sat*100:.1f}% of frames")
    for jname, sat in zip(RIGHT_ARM_JOINTS, saturated_r.mean(axis=0)):
        print(f"  {jname:35s}  clamped {sat*100:.1f}% of frames")


if __name__ == "__main__":
    main()
