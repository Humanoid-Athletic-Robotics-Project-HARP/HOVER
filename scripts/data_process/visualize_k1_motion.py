"""
Render a retargeted K1 motion clip to a GIF using MuJoCo offscreen renderer.

The K1 MJCF has no visual geometry, so the skeleton is drawn as capsule + sphere
geoms injected into the MjvScene after each update_scene call via mjv_makeConnector.

Usage (from hover root):
    python3 scripts/data_process/visualize_k1_motion.py
    python3 scripts/data_process/visualize_k1_motion.py --clip 0-CMU_09_09_12_poses --every 2
    python3 scripts/data_process/visualize_k1_motion.py --clip 0-CMU_09_09_12_poses --out /tmp/walk.gif
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import joblib
import mujoco
import imageio
from tqdm import tqdm

_HOVER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K1_MJCF   = os.path.join(_HOVER_DIR, "neural_wbc/data/data/motion_lib/k1.xml")
AMASS_PKL = os.path.join(_HOVER_DIR, "third_party/human2humanoid/data/k1/amass_all.pkl")

WIDTH  = 640
HEIGHT = 480

BONE_RADIUS  = 0.025
JOINT_RADIUS = 0.035


def xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


def build_skeleton_edges(model):
    edges = []
    for i in range(1, model.nbody):
        p = int(model.body(i).parentid)
        if p >= 1:
            edges.append((p, i))
    return edges


def _body_rgba(idx, left_bodies, right_bodies):
    if idx in left_bodies:
        return np.array([0.2, 0.55, 1.0, 1.0], dtype=np.float32)
    if idx in right_bodies:
        return np.array([1.0, 0.3, 0.3, 1.0], dtype=np.float32)
    return np.array([0.85, 0.85, 0.85, 1.0], dtype=np.float32)


def _add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        pos.astype(np.float64),
        np.eye(3).flatten(),
        rgba,
    )
    g.matid = -1
    scene.ngeom += 1


def _add_capsule(scene, p0, p1, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba,
    )
    mujoco.mjv_makeConnector(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
        p0[0], p0[1], p0[2],
        p1[0], p1[1], p1[2],
    )
    g.rgba[:] = rgba
    g.matid = -1
    scene.ngeom += 1


def _draw_skeleton(scene, xpos, edges, left_bodies, right_bodies):
    # Spheres at each joint
    for i in range(1, len(xpos)):
        _add_sphere(scene, xpos[i], JOINT_RADIUS, _body_rgba(i, left_bodies, right_bodies))

    # Capsule bones
    for p, c in edges:
        rgba = _body_rgba(c, left_bodies, right_bodies)
        _add_capsule(scene, xpos[p], xpos[c], BONE_RADIUS, rgba)


def _inject_lights(scene, lookat):
    """Inject three directional lights relative to the robot's current position."""
    base = scene.nlight
    new_lights = [
        dict(pos=[lookat[0],     lookat[1],     lookat[2] + 5],   dir=[0, 0, -1],     amb=0.45, dif=0.75),
        dict(pos=[lookat[0] + 4, lookat[1],     lookat[2] + 3],   dir=[-0.8, 0, -0.6], amb=0.20, dif=0.55),
        dict(pos=[lookat[0] - 4, lookat[1],     lookat[2] + 3],   dir=[0.8,  0, -0.6], amb=0.20, dif=0.55),
    ]
    scene.nlight = min(base + len(new_lights), 99)
    for k, cfg in enumerate(new_lights):
        if base + k >= scene.nlight:
            break
        lt = scene.lights[base + k]
        lt.directional = True
        lt.castshadow  = False
        lt.pos[:]      = cfg["pos"]
        lt.dir[:]      = cfg["dir"]
        lt.ambient[:]  = [cfg["amb"]] * 3
        lt.diffuse[:]  = [cfg["dif"]] * 3
        lt.specular[:] = [0.05, 0.05, 0.05]
    for i in range(base):
        scene.lights[i].ambient[:] = np.maximum(scene.lights[i].ambient, 0.3)
        scene.lights[i].diffuse[:] = np.maximum(scene.lights[i].diffuse, 0.6)


def render_clip(clip_key, out_path, fps_out=30, every_nth=1, data_all=None,
                azimuth=135.0, elevation=-20.0, distance=3.5):
    if data_all is None:
        print("Loading motion data...", flush=True)
        data_all = joblib.load(AMASS_PKL)
    if clip_key not in data_all:
        raise ValueError(f"Clip '{clip_key}' not found.")

    clip   = data_all[clip_key]
    trans  = clip["root_trans_offset"]
    dof    = clip["dof"]
    root_q = clip["root_rot"]
    N      = trans.shape[0]
    frames = list(range(0, N, every_nth))

    print("Loading MuJoCo model...", flush=True)
    model = mujoco.MjModel.from_xml_path(K1_MJCF)
    data  = mujoco.MjData(model)
    edges = build_skeleton_edges(model)

    left_bodies  = {i for i in range(model.nbody) if "left"  in model.body(i).name.lower()}
    right_bodies = {i for i in range(model.nbody) if "right" in model.body(i).name.lower()}

    # K1 has 0 physics geoms; allocate plenty of slots for our injected skeleton
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH, max_geom=300)

    cam           = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = distance
    cam.azimuth   = azimuth
    cam.elevation = elevation

    print("Rendering frames...", flush=True)
    images = []
    for i in tqdm(frames, desc="Render"):
        data.qpos[0:3] = trans[i]
        data.qpos[3:7] = xyzw_to_wxyz(root_q[i])
        data.qpos[7:]  = dof[i]
        mujoco.mj_forward(model, data)

        trunk = data.xpos[1].copy()
        cam.lookat[:] = [trunk[0], trunk[1], trunk[2] * 0.55]

        renderer.update_scene(data, camera=cam)

        _inject_lights(renderer.scene, np.array(cam.lookat))
        _draw_skeleton(renderer.scene, data.xpos, edges, left_bodies, right_bodies)

        pixels = renderer.render()
        images.append(pixels.copy())

    renderer.close()
    imageio.mimsave(out_path, images, fps=max(1, fps_out // every_nth), loop=0)
    print(f"Saved {len(images)} frames → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip",  default=None,
                        help="e.g. 0-CMU_09_09_12_poses (default: first clip)")
    parser.add_argument("--out",   default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "out.gif"))
    parser.add_argument("--fps",   type=int, default=30)
    parser.add_argument("--every", type=int, default=1,
                        help="Use every Nth frame (2 = half-length GIF)")
    parser.add_argument("--azimuth",   type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--distance",  type=float, default=3.5)
    args = parser.parse_args()

    print("Loading motion data...", flush=True)
    data_all = joblib.load(AMASS_PKL)
    all_keys = list(data_all.keys())

    clip_key = args.clip or all_keys[0]
    if clip_key not in data_all:
        print(f"Clip '{clip_key}' not found. Available clips:")
        for k in all_keys:
            print(f"  {k}")
        raise SystemExit(1)

    N = len(data_all[clip_key]["root_trans_offset"])
    print(f"Clip: {clip_key}  ({N} frames @ 30 fps = {N/30:.1f}s)")

    render_clip(clip_key, args.out,
                fps_out=args.fps, every_nth=args.every, data_all=data_all,
                azimuth=args.azimuth, elevation=args.elevation, distance=args.distance)

    print("\nAvailable clips:")
    for k in all_keys:
        print(f"  {k}")
