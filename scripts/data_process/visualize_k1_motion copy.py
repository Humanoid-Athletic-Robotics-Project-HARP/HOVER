"""
Render a retargeted K1 motion clip to a GIF using matplotlib 3D skeleton.
MuJoCo GPU rendering is unavailable in this environment (Isaac Sim conflict),
so body world positions are computed via mj_forward (CPU) and drawn as a stick figure.

Usage (from hover root):
    python3 scripts/visualize_k1_motion.py
    python3 scripts/visualize_k1_motion.py --clip 0-CMU_09_09_01_poses --out /tmp/walk.gif
"""
import argparse
import os
import numpy as np
import joblib
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import imageio
from tqdm import tqdm

_HOVER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K1_MJCF   = os.path.join(_HOVER_DIR, "neural_wbc/data/data/motion_lib/k1.xml")
AMASS_PKL = os.path.join(_HOVER_DIR, "third_party/human2humanoid/data/k1/amass_all.pkl")


def xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


def build_skeleton_edges(model):
    """Return list of (parent_idx, child_idx) body pairs, skipping world body."""
    edges = []
    for i in range(1, model.nbody):
        p = int(model.body(i).parentid)
        if p >= 1:  # skip world→trunk edge for clarity
            edges.append((p, i))
    return edges


def get_body_positions(model, data, trans, root_q, dof_frame):
    """Set qpos and run FK; return (nbody, 3) world positions."""
    data.qpos[0:3] = trans
    data.qpos[3:7] = xyzw_to_wxyz(root_q)
    data.qpos[7:]  = dof_frame
    mujoco.mj_forward(model, data)
    return data.xpos.copy()  # (nbody, 3)


def render_clip(clip_key, out_path, fps_out=30, every_nth=1, data_all=None):
    if data_all is None:
        print("Loading motion data...", flush=True)
        data_all = joblib.load(AMASS_PKL)
    if clip_key not in data_all:
        raise ValueError(f"Clip '{clip_key}' not found.")

    clip   = data_all[clip_key]
    trans  = clip["root_trans_offset"]   # (N, 3)
    dof    = clip["dof"]                 # (N, 22)
    root_q = clip["root_rot"]            # (N, 4) xyzw
    N      = trans.shape[0]
    frames = list(range(0, N, every_nth))

    print("Loading MuJoCo model...", flush=True)
    model = mujoco.MjModel.from_xml_path(K1_MJCF)
    data  = mujoco.MjData(model)
    edges = build_skeleton_edges(model)

    # Compute all body positions upfront
    all_pos = []
    for i in tqdm(frames, desc="FK"):
        all_pos.append(get_body_positions(model, data, trans[i], root_q[i], dof[i]))
    all_pos = np.stack(all_pos)  # (F, nbody, 3)

    # Scene bounds for stable axes
    lo = all_pos[:, 1:].min(axis=(0, 1))
    hi = all_pos[:, 1:].max(axis=(0, 1))
    pad = 0.3
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    half = max((hi - lo)[:2].max() / 2 + pad, 1.0)

    # Limb colours: torso/head grey, left blue, right red
    LEFT_BODIES  = {b for b in range(model.nbody) if "left" in model.body(b).name.lower() or "Left" in model.body(b).name}
    RIGHT_BODIES = {b for b in range(model.nbody) if "right" in model.body(b).name.lower() or "Right" in model.body(b).name}

    def edge_color(p, c):
        if c in LEFT_BODIES or p in LEFT_BODIES:   return "#4a90d9"
        if c in RIGHT_BODIES or p in RIGHT_BODIES: return "#e05252"
        return "#aaaaaa"

    print("Rendering frames...", flush=True)
    images = []
    fig = plt.figure(figsize=(6, 7), facecolor="#1a1a1a")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#1a1a1a")

    for f_idx in tqdm(range(len(frames)), desc="Render"):
        pos = all_pos[f_idx]
        ax.cla()
        ax.set_facecolor("#1a1a1a")

        # Draw bones
        for p, c in edges:
            color = edge_color(p, c)
            ax.plot([pos[p, 0], pos[c, 0]],
                    [pos[p, 1], pos[c, 1]],
                    [pos[p, 2], pos[c, 2]],
                    color=color, linewidth=2.5)

        # Draw joints as dots
        ax.scatter(pos[1:, 0], pos[1:, 1], pos[1:, 2],
                   c="#ffffff", s=12, zorder=5, depthshade=False)

        # Fixed axes so the robot doesn't swim
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(lo[2] - pad, hi[2] + pad)
        ax.set_xlabel("X", color="#888888", fontsize=8)
        ax.set_ylabel("Y", color="#888888", fontsize=8)
        ax.set_zlabel("Z", color="#888888", fontsize=8)
        ax.tick_params(colors="#555555", labelsize=6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333333")
        ax.view_init(elev=15, azim=0) # + f_idx * 0.5)  # slow orbit

        ax.set_title(clip_key.replace("0-", ""), color="#cccccc", fontsize=8, pad=4)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        images.append(buf.copy())

    plt.close(fig)
    imageio.mimsave(out_path, images, fps=fps_out // every_nth, loop=0)
    print(f"Saved {len(images)} frames → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip",  default=None, help="Clip name (default: first clip)")
    parser.add_argument("--out",   default="/workspace/testing-grounds/projects/hover/scripts/data_process/out.gif")
    parser.add_argument("--fps",   type=int, default=30)
    parser.add_argument("--every", type=int, default=1,
                        help="Render every Nth frame (2 = half-speed GIF)")
    args = parser.parse_args()

    print("Loading motion data...", flush=True)
    data_all = joblib.load(AMASS_PKL)
    all_keys = list(data_all.keys())
    clip_key = args.clip or all_keys[0]
    N = len(data_all[clip_key]["root_trans_offset"])
    print(f"Clip: {clip_key}  ({N} frames @ 30fps = {N/30:.1f}s)")

    render_clip(clip_key, args.out, fps_out=args.fps, every_nth=args.every, data_all=data_all)

    print("\nAvailable clips:")
    for k in all_keys:
        print(f"  {k}")
