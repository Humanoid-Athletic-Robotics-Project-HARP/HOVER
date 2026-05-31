"""Render a recorded K1 episode using MuJoCo EGL offscreen rendering.

Usage:
    MUJOCO_GL=egl python3 scripts/render_episode_mujoco.py \
        --episode /tmp/k1_episode.npz \
        --out /tmp/k1_render.mp4
"""
import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--episode", type=str, default="/tmp/k1_episode.npz")
parser.add_argument("--urdf", type=str,
    default="third_party/booster_assets/robots/K1/K1_22dof.urdf")
parser.add_argument("--out", type=str, default="/tmp/k1_render.mp4")
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args = parser.parse_args()

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

data_npz = np.load(args.episode)
joint_pos  = data_npz["joint_pos"]   # [T, 22]
root_pos   = data_npz["root_pos"]    # [T, 3]
root_quat  = data_npz["root_quat"]   # [T, 4]  wxyz
T = len(joint_pos)
print(f"Loaded {T} frames from {args.episode}")

# ── Step 1: load URDF as fixed-base, save to MJCF ──────────────────────────
urdf_abs = os.path.abspath(args.urdf)
meshdir  = os.path.dirname(urdf_abs)
_base_model = mujoco.MjModel.from_xml_path(urdf_abs)
tmp_xml = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, dir="/tmp")
tmp_xml.close()
mujoco.mj_saveLastXML(tmp_xml.name, _base_model)

# ── Step 2: inject freejoint by wrapping worldbody children ────────────────
tree = ET.parse(tmp_xml.name)
root_el = tree.getroot()

# Set absolute meshdir so the /tmp-saved XML can resolve relative mesh paths
compiler = root_el.find("compiler")
if compiler is None:
    compiler = ET.SubElement(root_el, "compiler")
compiler.set("meshdir", meshdir)

worldbody = root_el.find("worldbody")

# Move every existing worldbody child into a new "base" body with a freejoint
base_body = ET.Element("body", name="base")
ET.SubElement(base_body, "freejoint", name="base_freejoint")
for child in list(worldbody):
    worldbody.remove(child)
    base_body.append(child)
worldbody.append(base_body)

# Add a ground plane so the robot doesn't fall into the void visually
ET.SubElement(worldbody, "geom", name="ground",
              type="plane", size="10 10 0.1",
              rgba="0.8 0.8 0.8 1", pos="0 0 0")

modified_xml = tmp_xml.name.replace(".xml", "_freejoint.xml")
tree.write(modified_xml)

# ── Step 3: reload with freejoint ─────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(modified_xml)
data  = mujoco.MjData(model)
print(f"Model with freejoint: nq={model.nq}, njnt={model.njnt}")
assert model.nq == 29, f"Expected nq=29 (7+22), got {model.nq}"

os.unlink(tmp_xml.name)
os.unlink(modified_xml)

renderer = mujoco.Renderer(model, height=args.height, width=args.width)

cam = mujoco.MjvCamera()
cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance  = 3.5
cam.azimuth   = 90.0
cam.elevation = -15.0
# Fix lookat at the first-frame root position so the camera never moves
cam.lookat[0] = root_pos[0, 0]
cam.lookat[1] = root_pos[0, 1]
cam.lookat[2] = 0.5  # mid-body height

frames = []
for t in range(T):
    # qpos layout: pos(3) + quat_wxyz(4) + joints(22)
    data.qpos[:3]  = root_pos[t]
    data.qpos[3:7] = root_quat[t]   # wxyz, matches MuJoCo convention
    data.qpos[7:]  = joint_pos[t]
    data.qvel[:]   = 0.0
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera=cam)
    frames.append(renderer.render().copy())

renderer.close()
print(f"Rendered {len(frames)} frames")

# Save as mp4 via imageio
try:
    import imageio
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"Saved video: {args.out}")
except Exception as e:
    out_gif = args.out.replace(".mp4", ".gif")
    import imageio
    imageio.mimsave(out_gif, frames, fps=args.fps)
    print(f"Saved gif: {out_gif} (mp4 failed: {e})")
