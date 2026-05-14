"""
Overlay the SMPL reference skeleton on top of the K1 MuJoCo rendering.

SMPL joints and bones are injected as custom MuJoCo scene geoms so they share
the same camera, lighting, and depth buffer as the robot.

Usage (from hover root):
    python3 scripts/data_process/visualize_k1_smpl_overlay.py \\
        --clip 0-CMU_13_13_17_poses \\
        --amass_npz third_party/human2humanoid/data/AMASS/AMASS_Complete/CMU/13/13_17_poses.npz \\
        --every 15 --out /tmp/overlay.gif
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import joblib
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.append(os.getcwd())
from phc.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES

_HERE      = os.path.dirname(os.path.abspath(__file__))
_HOVER_DIR = os.path.normpath(os.path.join(_HERE, "..", ".."))
_H2H_ROOT  = os.path.join(_HOVER_DIR, "third_party", "human2humanoid")

K1_MJCF   = os.path.join(_HOVER_DIR, "third_party/booster_assets/robots/K1/K1_22dof.xml")
AMASS_PKL = os.path.join(_H2H_ROOT, "data", "k1", "amass_all.pkl")
SHAPE_PKL = os.path.join(_H2H_ROOT, "data", "k1", "shape_optimized_v1.pkl")
SMPL_DATA = os.path.join(_H2H_ROOT, "data", "smpl")

WIDTH, HEIGHT = 640, 480

# Exactly the 9 joints used as IK targets in grad_fit_k1.py, in the same order.
# Scaling: first 5 → leg_scale, last 4 → arm_scale  (mirrors _s in grad_fit_k1.py)
_SMPL_PICK_NAMES = [
    'Pelvis', 'L_Knee', 'L_Ankle', 'R_Knee', 'R_Ankle',   # leg_scale
    'L_Elbow', 'L_Hand', 'R_Elbow', 'R_Hand',              # arm_scale
]
SMPL_PICK_IDX = [SMPL_BONE_ORDER_NAMES.index(n) for n in _SMPL_PICK_NAMES]

# Connections between pick-array indices (0-8) for drawing bones
PICK_EDGES = [
    (0, 1), (1, 2),   # Pelvis→L_Knee→L_Ankle
    (0, 3), (3, 4),   # Pelvis→R_Knee→R_Ankle
    (0, 5), (5, 6),   # Pelvis→L_Elbow→L_Hand
    (0, 7), (7, 8),   # Pelvis→R_Elbow→R_Hand
]

# Colors per pick index: 0=center, 1-2=left, 3-4=right, 5-6=left, 7-8=right
_PICK_COLORS = [
    np.array([0.90, 0.90, 0.90, 0.9], dtype=np.float32),  # 0 Pelvis
    np.array([0.25, 0.65, 1.00, 0.9], dtype=np.float32),  # 1 L_Knee
    np.array([0.25, 0.65, 1.00, 0.9], dtype=np.float32),  # 2 L_Ankle
    np.array([1.00, 0.35, 0.35, 0.9], dtype=np.float32),  # 3 R_Knee
    np.array([1.00, 0.35, 0.35, 0.9], dtype=np.float32),  # 4 R_Ankle
    np.array([0.25, 0.65, 1.00, 0.9], dtype=np.float32),  # 5 L_Elbow
    np.array([0.25, 0.65, 1.00, 0.9], dtype=np.float32),  # 6 L_Hand
    np.array([1.00, 0.35, 0.35, 0.9], dtype=np.float32),  # 7 R_Elbow
    np.array([1.00, 0.35, 0.35, 0.9], dtype=np.float32),  # 8 R_Hand
]

JOINT_RADIUS = 0.030   # metres
BONE_WIDTH   = 0.015   # capsule radius


def _capsule_mat(a, b):
    """Row-major 3x3 rotation matrix that aligns the capsule's local Z with (b-a)."""
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.eye(3, dtype=np.float64).flatten()
    z = d / n
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = ref - np.dot(ref, z) * z;  x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z]).flatten()   # column-major = MuJoCo mat layout


def add_smpl_overlay(scene, picked_joints):
    """
    Inject the 9 IK-target joints into an mjvScene as spheres + capsule bones.
    picked_joints: (9, 3) — already scaled and translated to K1 world position.
    """
    I3 = np.eye(3, dtype=np.float64).flatten()

    for k, pos in enumerate(picked_joints):
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([JOINT_RADIUS, 0.0, 0.0]),
            pos.astype(np.float64),
            I3,
            _PICK_COLORS[k],
        )
        scene.ngeom += 1

    for a, b in PICK_EDGES:
        if scene.ngeom >= scene.maxgeom:
            break
        pa, pb = picked_joints[a], picked_joints[b]
        half_len = np.linalg.norm(pb - pa) / 2.0
        if half_len < 1e-6:
            continue
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([BONE_WIDTH, half_len, 0.0]),
            ((pa + pb) / 2.0).astype(np.float64),
            _capsule_mat(pa, pb),
            _PICK_COLORS[b],
        )
        scene.ngeom += 1


def _inject_lights(scene, lookat):
    base = scene.nlight
    cfgs = [
        dict(pos=[lookat[0],   lookat[1],   lookat[2]+5],  dir=[0,0,-1],      amb=0.45, dif=0.75),
        dict(pos=[lookat[0]+4, lookat[1],   lookat[2]+3],  dir=[-0.8,0,-0.6], amb=0.20, dif=0.55),
        dict(pos=[lookat[0]-4, lookat[1],   lookat[2]+3],  dir=[0.8, 0,-0.6], amb=0.20, dif=0.55),
    ]
    scene.nlight = min(base + len(cfgs), 99)
    for k, cfg in enumerate(cfgs):
        if base + k >= scene.nlight:
            break
        lt = scene.lights[base + k]
        lt.pos[:]     = cfg["pos"]
        lt.dir[:]     = cfg["dir"]
        lt.ambient[:] = [cfg["amb"]] * 3
        lt.diffuse[:] = [cfg["dif"]] * 3
        lt.specular[:] = [0.05] * 3
    for i in range(base):
        scene.lights[i].ambient[:] = np.maximum(scene.lights[i].ambient, 0.3)
        scene.lights[i].diffuse[:] = np.maximum(scene.lights[i].diffuse, 0.6)


def xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


def find_npz(amass_root, clip_key):
    stem = clip_key[2:]
    for npz in glob.glob(os.path.join(amass_root, "**", "*.npz"), recursive=True):
        candidate = "_".join(os.path.relpath(npz, amass_root).replace(".npz", "").split(os.sep))
        if candidate == stem:
            return npz
    return None


def label(arr, text, color=(255, 255, 64)):
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        draw.text((8+dx, 8+dy), text, fill=(0,0,0))
    draw.text((8, 8), text, fill=color)
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip",       default=None)
    parser.add_argument("--amass_npz",  default=None)
    parser.add_argument("--amass_root", default=None)
    parser.add_argument("--out",        default=os.path.join(_HERE, "overlay.gif"))
    parser.add_argument("--fps",        type=int,   default=30)
    parser.add_argument("--every",      type=int,   default=10)
    parser.add_argument("--azimuth",    type=float, default=135.0)
    parser.add_argument("--elevation",  type=float, default=-20.0)
    parser.add_argument("--distance",   type=float, default=3.5)
    args = parser.parse_args()

    # ── K1 data ───────────────────────────────────────────────────────────────
    print("Loading K1 pkl …")
    data_all = joblib.load(AMASS_PKL)
    clip_key = args.clip or list(data_all.keys())[0]
    if clip_key not in data_all:
        raise SystemExit(f"Clip '{clip_key}' not found.")
    clip     = data_all[clip_key]
    k1_trans = clip["root_trans_offset"]
    k1_dof   = clip["dof"]
    k1_rootq = clip["root_rot"]
    N = k1_trans.shape[0]
    print(f"Clip: {clip_key}  ({N} frames @ 30 fps = {N/30:.1f}s)")

    # ── SMPL data ─────────────────────────────────────────────────────────────
    npz_path = args.amass_npz
    if npz_path is None and args.amass_root:
        npz_path = find_npz(args.amass_root, clip_key)
    if npz_path and not os.path.exists(npz_path):
        raise FileNotFoundError(f"AMASS npz not found: {npz_path}")

    smpl_joints_all = None
    if npz_path:
        print(f"Loading SMPL from {npz_path} …")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        smpl_parser = SMPL_Parser(model_path=SMPL_DATA, gender="neutral").to(device)
        shape_new, leg_scale, arm_scale = joblib.load(SHAPE_PKL)
        shape_new = shape_new.to(device)
        leg_scale = float(leg_scale.mean().item())
        arm_scale = float(arm_scale.mean().item())
        print(f"  leg_scale={leg_scale:.4f}  arm_scale={arm_scale:.4f}")

        amass = dict(np.load(npz_path, allow_pickle=True))
        skip  = max(1, int(amass["mocap_framerate"] // 30))
        smpl_t = torch.from_numpy(amass["trans"][::skip]).float().to(device)
        smpl_p = torch.from_numpy(
            np.concatenate([amass["poses"][::skip, :66],
                            np.zeros((smpl_t.shape[0], 6))], axis=-1)
        ).float().to(device)
        n = min(smpl_t.shape[0], N)
        with torch.no_grad():
            _, joints = smpl_parser.get_joints_verts(smpl_p[:n], shape_new, smpl_t[:n])
        joints_np = joints.cpu().numpy()   # (n, J, 3), already Z-up

        # Replicate grad_fit_k1.py exactly:
        #   _s[:5] = leg_scale  (Pelvis, L_Knee, L_Ankle, R_Knee, R_Ankle)
        #   _s[5:] = arm_scale  (L_Elbow, L_Hand, R_Elbow, R_Hand)
        pick_scales = np.array([leg_scale]*5 + [arm_scale]*4)  # (9,)
        pelvis      = joints_np[:, 0:1, :]                     # (n, 1, 3)
        picked      = joints_np[:, SMPL_PICK_IDX, :]           # (n, 9, 3)
        smpl_joints_all = (picked - pelvis) * pick_scales[None, :, None] + pelvis
        print(f"SMPL target joints ready: {smpl_joints_all.shape}  "
              f"(9 IK targets, leg×{leg_scale:.3f} arm×{arm_scale:.3f})")
    else:
        print("No AMASS npz — rendering K1 only.")

    # ── MuJoCo ───────────────────────────────────────────────────────────────
    print("Loading MuJoCo model …")
    model    = mujoco.MjModel.from_xml_path(K1_MJCF)
    mjdata   = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    cam           = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = args.distance
    cam.azimuth   = args.azimuth
    cam.elevation = args.elevation

    # ── Render ────────────────────────────────────────────────────────────────
    print("Rendering …")
    frame_indices = list(range(0, N, args.every))
    images = []

    for fi in tqdm(frame_indices, desc="frames"):
        mjdata.qpos[0:3] = k1_trans[fi]
        mjdata.qpos[3:7] = xyzw_to_wxyz(k1_rootq[fi])
        mjdata.qpos[7:]  = k1_dof[fi]
        mujoco.mj_forward(model, mjdata)

        trunk = mjdata.xpos[1].copy()
        cam.lookat[:] = [trunk[0], trunk[1], trunk[2] * 0.55]

        renderer.update_scene(mjdata, camera=cam)
        # renderer.scene.ngeom = 0  # uncomment to hide K1 and show only SMPL
        _inject_lights(renderer.scene, np.array(cam.lookat))

        if smpl_joints_all is not None and fi < len(smpl_joints_all):
            sj = smpl_joints_all[fi].copy()          # (9, 3)
            sj = sj - sj[0] + k1_trans[fi]           # align Pelvis → K1 root
            add_smpl_overlay(renderer.scene, sj)

        frame = renderer.render().copy()
        frame = label(frame, f"K1 + SMPL  frame {fi}")
        images.append(frame)

    renderer.close()
    fps_out = max(1, args.fps // args.every)
    imageio.mimsave(args.out, images, fps=fps_out, loop=0)
    print(f"Saved {len(images)} frames → {args.out}")


if __name__ == "__main__":
    main()
