#!/usr/bin/env python3
"""
Sweep each of the 22 K1 DOFs independently and render a labeled GIF.
Run from hover root:
  python3 scripts/data_process/test_elbow_k1.py [--out /tmp/joint_sweep.gif]

Each segment of the GIF shows one joint sweeping from its min to max range
while all other joints are at zero. Joint index + name are printed on every frame.
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

K1_MJCF = "third_party/booster_assets/robots/K1/K1_22dof.xml"

DOF_NAMES = [
    "AAHead_yaw",          # 0
    "Head_pitch",          # 1
    "ALeft_Shoulder_Pitch",# 2
    "Left_Shoulder_Roll",  # 3
    "Left_Elbow_Pitch",    # 4
    "Left_Elbow_Yaw",      # 5
    "ARight_Shoulder_Pitch",# 6
    "Right_Shoulder_Roll", # 7
    "Right_Elbow_Pitch",   # 8
    "Right_Elbow_Yaw",     # 9
    "Left_Hip_Pitch",      # 10
    "Left_Hip_Roll",       # 11
    "Left_Hip_Yaw",        # 12
    "Left_Knee_Pitch",     # 13
    "Left_Ankle_Pitch",    # 14
    "Left_Ankle_Roll",     # 15
    "Right_Hip_Pitch",     # 16
    "Right_Hip_Roll",      # 17
    "Right_Hip_Yaw",       # 18
    "Right_Knee_Pitch",    # 19
    "Right_Ankle_Pitch",   # 20
    "Right_Ankle_Roll",    # 21
]


def label_frame(img_arr, text, color=(255, 255, 64)):
    """Stamp bold text onto the top-left corner of a numpy HxWx3 image."""
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    # Shadow for readability
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        draw.text((10 + dx, 10 + dy), text, fill=(0, 0, 0))
    draw.text((10, 10), text, fill=color)
    return np.array(img)


def render_frame(model, mdata, azimuths, dist=2.5, el=-15, lookat_z=0.85):
    panels = []
    for az in azimuths:
        r = mujoco.Renderer(model, height=360, width=360)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = dist
        cam.azimuth = az
        cam.elevation = el
        cam.lookat[:] = [0, 0, lookat_z]
        r.update_scene(mdata, camera=cam)
        panels.append(r.render().copy())
        r.close()
    return np.hstack(panels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/joint_sweep",
                        help="output directory (one GIF per DOF saved here)")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--steps", type=int, default=20,
                        help="frames per joint sweep (forward + back)")
    parser.add_argument("--dofs", default=None,
                        help="comma-separated DOF indices to test, e.g. 2,3,4,5 (default: all)")
    parser.add_argument("--every", type=int, default=1,
                        help="render every Nth DOF (e.g. --every 2 skips every other)")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(K1_MJCF)
    mdata = mujoco.MjData(model)

    # Read joint ranges from model
    joint_ranges = []
    for ji in range(model.njnt):
        if model.jnt_type[ji] == 3:  # hinge
            lo = model.jnt_range[ji, 0]
            hi = model.jnt_range[ji, 1]
            joint_ranges.append((lo, hi))

    azimuths = [0, 90, 180]   # front, right-side, back

    os.makedirs(args.out, exist_ok=True)

    dof_indices = list(range(22))
    if args.dofs:
        dof_indices = [int(x) for x in args.dofs.split(",")]
    dof_indices = dof_indices[::args.every]

    for dof_i in tqdm(dof_indices, desc="DOFs"):
        lo, hi = joint_ranges[dof_i]
        name = DOF_NAMES[dof_i]

        angles = np.concatenate([
            np.linspace(lo, hi, args.steps),
            np.linspace(hi, lo, args.steps),
        ])

        frames = []
        for angle in tqdm(angles, desc=f"DOF {dof_i:02d} {name}", leave=False):
            mujoco.mj_resetData(model, mdata)
            mdata.qpos[0:3] = [0, 0, 0.56]
            mdata.qpos[3:7] = [1, 0, 0, 0]
            mdata.qpos[7 + dof_i] = angle
            mujoco.mj_forward(model, mdata)

            frame = render_frame(model, mdata, azimuths)
            text = f"DOF {dof_i}: {name}\n{np.degrees(angle):+.1f} deg"
            frame = label_frame(frame, text)
            frames.append(frame)

        out_path = os.path.join(args.out, f"dof{dof_i:02d}_{name}.gif")
        imageio.mimsave(out_path, frames, fps=args.fps, loop=0)
        tqdm.write(f"  -> {out_path}")


if __name__ == "__main__":
    main()
