"""
Side-by-side comparison GIF: K1 robot (MuJoCo) | SMPL reference skeleton.

Usage (from hover root):
    python3 scripts/data_process/visualize_k1_smpl_compare.py \\
        --clip 0-CMU_13_13_17_poses \\
        --amass_npz data/AMASS/AMASS_Complete/CMU/13/13_17_poses.npz \\
        --every 15

    # auto-locate npz from root:
    python3 scripts/data_process/visualize_k1_smpl_compare.py \\
        --clip 0-CMU_13_13_17_poses \\
        --amass_root data/AMASS/AMASS_Complete \\
        --every 15
"""
import argparse
import glob
import io
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.append(os.getcwd())
from phc.smpllib.smpl_parser import SMPL_Parser

_HERE      = os.path.dirname(os.path.abspath(__file__))
_HOVER_DIR = os.path.normpath(os.path.join(_HERE, "..", ".."))
_H2H_ROOT  = os.path.join(_HOVER_DIR, "third_party", "human2humanoid")

K1_MJCF   = os.path.join(_HOVER_DIR, "third_party/booster_assets/robots/K1/K1_22dof.xml")
AMASS_PKL = os.path.join(_H2H_ROOT, "data", "k1", "amass_all.pkl")
SHAPE_PKL = os.path.join(_H2H_ROOT, "data", "k1", "shape_optimized_v1.pkl")
SMPL_DATA = os.path.join(_H2H_ROOT, "data", "smpl")

WIDTH, HEIGHT = 640, 480

# Standard 24-joint SMPL kinematic tree (matches SMPL_BONE_ORDER_NAMES)
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
_LEFT  = {1, 4, 7, 10, 13, 16, 18, 20, 22}   # L_Hip…L_Hand
_RIGHT = {2, 5, 8, 11, 14, 17, 19, 21, 23}   # R_Hip…R_Hand


def xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


def _inject_lights(scene, lookat):
    base = scene.nlight
    cfgs = [
        dict(pos=[lookat[0],     lookat[1],     lookat[2]+5],  dir=[0,0,-1],      amb=0.45, dif=0.75),
        dict(pos=[lookat[0]+4,   lookat[1],     lookat[2]+3],  dir=[-0.8,0,-0.6], amb=0.20, dif=0.55),
        dict(pos=[lookat[0]-4,   lookat[1],     lookat[2]+3],  dir=[0.8, 0,-0.6], amb=0.20, dif=0.55),
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


def render_smpl_frame(joints, azimuth, elevation, width=WIDTH, height=HEIGHT):
    """
    Render one SMPL skeleton frame as an HxWx3 uint8 numpy array.
    joints: (J, 3) in sim Z-up world frame, already translated to match K1 position.
    """
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#1e1e1e")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#1e1e1e")

    # Bones
    for j, p in enumerate(SMPL_PARENTS):
        if p < 0:
            continue
        if j in _LEFT:
            col = "#4fc3f7"   # blue = left
        elif j in _RIGHT:
            col = "#ef9a9a"   # red = right
        else:
            col = "#eeeeee"   # white = spine/head
        ax.plot(
            [joints[j, 0], joints[p, 0]],
            [joints[j, 1], joints[p, 1]],
            [joints[j, 2], joints[p, 2]],
            color=col, linewidth=2.5, solid_capstyle="round",
        )

    # Joint dots
    colors = ["#4fc3f7" if j in _LEFT else "#ef9a9a" if j in _RIGHT else "#eeeeee"
              for j in range(len(SMPL_PARENTS))]
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c=colors, s=18, depthshade=True, zorder=5)

    # Camera: MuJoCo elevation<0 means camera above looking down → matplotlib elev>0
    ax.view_init(elev=-elevation, azim=azimuth - 90)

    # Bounds centred on pelvis
    c    = joints[0]
    span = 0.9
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    ax.set_zlim(c[2] - 0.1,  c[2] + 1.8)
    ax.set_axis_off()
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))
    # Crop / resize to exact (H, W, 3)
    img = np.array(Image.fromarray(img).resize((width, height), Image.LANCZOS))
    return img


def label(arr, text, color=(255, 255, 64)):
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
        draw.text((8+dx, 8+dy), text, fill=(0,0,0))
    draw.text((8, 8), text, fill=color)
    return np.array(img)


def find_npz(amass_root, clip_key):
    """Try to locate the AMASS npz for this clip key by glob + stem matching."""
    stem = clip_key[2:]   # strip "0-"
    for npz in glob.glob(os.path.join(amass_root, "**", "*.npz"), recursive=True):
        rel  = os.path.relpath(npz, amass_root).replace(".npz", "")
        stem_candidate = "_".join(rel.split(os.sep))
        if stem_candidate == stem:
            return npz
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip",       default=None)
    parser.add_argument("--amass_npz",  default=None,
                        help="Direct path to the original AMASS .npz for this clip")
    parser.add_argument("--amass_root", default=None,
                        help="AMASS dataset root; script will auto-locate the .npz")
    parser.add_argument("--out",        default=os.path.join(_HERE, "compare.gif"))
    parser.add_argument("--fps",        type=int,   default=30)
    parser.add_argument("--every",      type=int,   default=10)
    parser.add_argument("--azimuth",    type=float, default=135.0)
    parser.add_argument("--elevation",  type=float, default=-20.0)
    parser.add_argument("--distance",   type=float, default=3.5)
    args = parser.parse_args()

    # ── K1 data ───────────────────────────────────────────────────────────────
    print("Loading K1 pkl …", flush=True)
    data_all = joblib.load(AMASS_PKL)
    clip_key = args.clip or list(data_all.keys())[0]
    if clip_key not in data_all:
        print(f"Clip '{clip_key}' not found. Available:\n  " +
              "\n  ".join(list(data_all.keys())[:10]))
        raise SystemExit(1)
    clip   = data_all[clip_key]
    k1_trans  = clip["root_trans_offset"]
    k1_dof    = clip["dof"]
    k1_rootq  = clip["root_rot"]
    N = k1_trans.shape[0]
    print(f"Clip: {clip_key}  ({N} frames @ 30 fps = {N/30:.1f}s)")

    # ── SMPL data (optional) ──────────────────────────────────────────────────
    npz_path = args.amass_npz
    if npz_path is None and args.amass_root:
        npz_path = find_npz(args.amass_root, clip_key)
        if npz_path:
            print(f"Found AMASS npz: {npz_path}")
        else:
            print("Could not auto-locate .npz — rendering K1 only.")

    smpl_joints_sim = None
    if npz_path and not os.path.exists(npz_path):
        raise FileNotFoundError(f"AMASS npz not found: {npz_path}")
    if npz_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        smpl_parser = SMPL_Parser(model_path=SMPL_DATA, gender="neutral").to(device)
        shape_new, _, _ = joblib.load(SHAPE_PKL)
        shape_new = shape_new.to(device)

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
        # AMASS root orientations already encode Y-up→Z-up, so FK output
        # is in sim Z-up world space — no extra rotation needed.
        smpl_joints_sim = joints.cpu().numpy()                  # (n, J, 3)
        print(f"SMPL joints ready: {smpl_joints_sim.shape}")

    # ── MuJoCo ───────────────────────────────────────────────────────────────
    print("Loading MuJoCo model …", flush=True)
    model    = mujoco.MjModel.from_xml_path(K1_MJCF)
    mjdata   = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam           = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance  = args.distance
    cam.azimuth   = args.azimuth
    cam.elevation = args.elevation

    # ── Render ────────────────────────────────────────────────────────────────
    print("Rendering …", flush=True)
    frame_indices = list(range(0, N, args.every))
    images = []

    for fi in tqdm(frame_indices, desc="frames"):
        # K1 MuJoCo frame
        mjdata.qpos[0:3] = k1_trans[fi]
        mjdata.qpos[3:7] = xyzw_to_wxyz(k1_rootq[fi])
        mjdata.qpos[7:]  = k1_dof[fi]
        mujoco.mj_forward(model, mjdata)

        trunk = mjdata.xpos[1].copy()
        cam.lookat[:] = [trunk[0], trunk[1], trunk[2] * 0.55]
        renderer.update_scene(mjdata, camera=cam)
        _inject_lights(renderer.scene, np.array(cam.lookat))
        k1_frame = renderer.render().copy()
        k1_frame = label(k1_frame, "K1 retargeted")

        if smpl_joints_sim is not None and fi < len(smpl_joints_sim):
            sj = smpl_joints_sim[fi].copy()               # (J, 3) in sim frame
            # Translate: SMPL pelvis → K1 trunk position
            sj = sj - sj[0] + trunk
            smpl_frame = render_smpl_frame(sj, args.azimuth, args.elevation)
            smpl_frame = label(smpl_frame, "SMPL reference", color=(180, 255, 180))
            combined = np.hstack([k1_frame, smpl_frame])
        else:
            combined = k1_frame

        images.append(combined)

    renderer.close()
    fps_out = max(1, args.fps // args.every)
    imageio.mimsave(args.out, images, fps=fps_out, loop=0)
    print(f"Saved {len(images)} frames → {args.out}")


if __name__ == "__main__":
    main()
