#!/usr/bin/env python3
"""
Overlay SMPL target joints vs K1 retargeted FK joints for a given clip/frame.

Run from human2humanoid dir:
  python3 ../../scripts/data_process/viz_retarget_k1.py
  python3 ../../scripts/data_process/viz_retarget_k1.py --clip 0-CMU_... --frame 30
  python3 ../../scripts/data_process/viz_retarget_k1.py --list          # print all clip keys
"""
import argparse
import os, sys, os.path as osp
sys.path.append(os.getcwd())

import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.transform import Rotation as sRot

from phc.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch

_HERE  = osp.dirname(osp.abspath(__file__))
_HOVER = osp.normpath(osp.join(_HERE, "..", ".."))
_H2H   = osp.join(_HOVER, "third_party", "human2humanoid")
K1_MJCF        = osp.join(_HOVER, "neural_wbc", "data", "data", "motion_lib", "k1.xml")
_SMPL_DATA_PATH = osp.join(_H2H, "data", "smpl")
_K1_DATA_DIR    = osp.join(_H2H, "data", "k1")

K1_ROTATION_AXIS = torch.tensor([[
    [0, 0, 1], [0, 1, 0],
    [0, 1, 0], [1, 0, 0],
    [1, 0, 0], [0, 0, 1],   # L_Elbow_Pitch X-axis
    [0, 1, 0], [1, 0, 0],
    [1, 0, 0], [0, 0, 1],   # R_Elbow_Pitch X-axis
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],
    [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0],
]])

K1_BODY_NAMES = [
    'Trunk', 'Head_1', 'Head_2',
    'Left_Arm_1', 'Left_Arm_2', 'Left_Arm_3', 'left_hand_link',
    'Right_Arm_1', 'Right_Arm_2', 'Right_Arm_3', 'right_hand_link',
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw',
    'Left_Shank', 'Left_Ankle_Cross', 'left_foot_link',
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw',
    'Right_Shank', 'Right_Ankle_Cross', 'right_foot_link',
]

# K1 limb pairs (body index pairs)
K1_LINKS = [
    (0, 1),   # trunk→head1
    (1, 2),   # head1→head2
    (0, 3),   # trunk→L_Arm_1
    (3, 4), (4, 5), (5, 6),    # L arm chain
    (0, 7),   # trunk→R_Arm_1
    (7, 8), (8, 9), (9, 10),   # R arm chain
    (0, 11),  # trunk→L_Hip
    (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),  # L leg
    (0, 17),  # trunk→R_Hip
    (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),  # R leg
]

# SMPL skeleton (24 joints)
SMPL_LINKS = [
    (0, 1), (0, 2),            # pelvis→hips
    (1, 4), (2, 5),            # hips→knees
    (4, 7), (5, 8),            # knees→ankles
    (7, 10), (8, 11),          # ankles→feet
    (0, 3), (3, 6), (6, 9),    # spine
    (9, 12), (12, 15),         # neck→head
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),   # L arm
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),   # R arm
]

# Matched joint pairs used in the optimizer
k1_joint_pick   = ['Trunk', 'Left_Shank', 'left_foot_link', 'Right_Shank', 'right_foot_link',
                    'Left_Arm_3', 'left_hand_link', 'Right_Arm_3', 'right_hand_link']
smpl_joint_pick = ['Pelvis', 'L_Knee', 'L_Ankle', 'R_Knee', 'R_Ankle',
                    'L_Elbow', 'L_Hand', 'R_Elbow', 'R_Hand']
k1_pick_idx   = [K1_BODY_NAMES.index(j)          for j in k1_joint_pick]
smpl_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j)  for j in smpl_joint_pick]

SMPL_ROOT_ALIGN_QUAT = [0.5, 0.5, 0.5, 0.5]


def load_amass_npz(path):
    entry = dict(np.load(open(path, "rb"), allow_pickle=True))
    if 'mocap_framerate' not in entry:
        return None
    fps  = entry['mocap_framerate']
    skip = max(1, int(fps // 30))
    trans   = entry['trans'][::skip]
    pose_aa = np.concatenate([entry['poses'][::skip, :66], np.zeros((trans.shape[0], 6))], axis=-1)
    return trans, pose_aa, entry.get('betas', np.zeros(10))


def key_to_npz(key, amass_root):
    """
    Reverse the key construction from grad_fit_k1.py:
      key = "0-" + "_".join(path_components_without_root).replace(".npz","")
    Try every possible split of the underscore-joined string into dir/file parts.
    Also searches AMASS_Filtered alongside AMASS_Complete.
    """
    inner = key[2:]  # strip "0-"
    parts = inner.split("_")

    # Candidate roots to search (Complete and Filtered side by side)
    roots = [amass_root]
    alt = amass_root.replace("AMASS_Complete", "AMASS_Filtered") \
                    .replace("AMASS_Filtered", "AMASS_Complete")
    if alt != amass_root:
        roots.append(alt)

    for root in roots:
        for split in range(1, len(parts)):
            subdir = "/".join(parts[:split])
            fname  = "_".join(parts[split:]) + ".npz"
            candidate = osp.join(root, subdir, fname)
            if osp.isfile(candidate):
                return candidate

    # Last resort: glob for the filename anywhere under any root
    import glob as _glob
    for root in roots:
        for split in range(1, len(parts)):
            fname = "_".join(parts[split:]) + ".npz"
            hits = _glob.glob(osp.join(root, "**", fname), recursive=True)
            if hits:
                return hits[0]
    return None


def run_k1_fk(pose_aa_np, root_trans_np, device):
    """pose_aa_np: (N,23,3), root_trans_np: (N,3) → (N, n_bodies, 3)"""
    k1_fk = Humanoid_Batch(mjcf_file=K1_MJCF, extend_hand=False, extend_head=False, device=device)
    pa = torch.from_numpy(pose_aa_np).float().to(device).unsqueeze(0)   # (1,N,23,3)
    rt = torch.from_numpy(root_trans_np).float().to(device).unsqueeze(0)  # (1,N,3)
    fk = k1_fk.fk_batch(pa, rt)
    return fk.global_translation[0].cpu().numpy()  # (N, n_bodies, 3)


def plot_frame(smpl_j, k1_j, smpl_j_scaled, frame_idx, out_path):
    """
    smpl_j        : (24, 3)  raw SMPL joints for this frame
    smpl_j_scaled : (9, 3)   scaled target joints (what optimizer matched to)
    k1_j          : (23, 3)  K1 FK body positions for this frame
    """
    views = [
        ("Front (X-Z)",  0, 2, "X", "Z"),
        ("Side (Y-Z)",   1, 2, "Y", "Z"),
        ("Top  (X-Y)",   0, 1, "X", "Y"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(f"SMPL targets (blue) vs K1 retargeted FK (red) — frame {frame_idx}", fontsize=13)

    for ax, (title, xi, yi, xl, yl) in zip(axes, views):
        # --- SMPL skeleton (thin blue) ---
        for a, b in SMPL_LINKS:
            ax.plot([smpl_j[a, xi], smpl_j[b, xi]],
                    [smpl_j[a, yi], smpl_j[b, yi]],
                    color='steelblue', lw=1.0, alpha=0.5)
        ax.scatter(smpl_j[:, xi], smpl_j[:, yi],
                   s=12, color='steelblue', alpha=0.5, zorder=3)

        # --- scaled target joints (solid blue circles, optimizer used these) ---
        ax.scatter(smpl_j_scaled[:, xi], smpl_j_scaled[:, yi],
                   s=60, color='blue', marker='o', zorder=5, label='SMPL target (scaled)')
        for i, name in enumerate(smpl_joint_pick):
            ax.annotate(name, (smpl_j_scaled[i, xi], smpl_j_scaled[i, yi]),
                        fontsize=6, color='blue', alpha=0.8,
                        xytext=(3, 3), textcoords='offset points')

        # --- K1 skeleton (thin red) ---
        for a, b in K1_LINKS:
            ax.plot([k1_j[a, xi], k1_j[b, xi]],
                    [k1_j[a, yi], k1_j[b, yi]],
                    color='salmon', lw=1.0, alpha=0.5)
        ax.scatter(k1_j[:, xi], k1_j[:, yi],
                   s=12, color='salmon', alpha=0.5, zorder=3)

        # --- K1 matched joints (solid red squares) ---
        k1_matched = k1_j[k1_pick_idx]
        ax.scatter(k1_matched[:, xi], k1_matched[:, yi],
                   s=60, color='red', marker='s', zorder=5, label='K1 FK (matched)')
        for i, name in enumerate(k1_joint_pick):
            ax.annotate(name, (k1_matched[i, xi], k1_matched[i, yi]),
                        fontsize=6, color='red', alpha=0.8,
                        xytext=(3, -8), textcoords='offset points')

        # --- error lines between matched pairs ---
        for i in range(len(k1_pick_idx)):
            ax.plot([smpl_j_scaled[i, xi], k1_matched[i, xi]],
                    [smpl_j_scaled[i, yi], k1_matched[i, yi]],
                    color='orange', lw=1.0, linestyle='--', alpha=0.7, zorder=4)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    legend = [
        mpatches.Patch(color='blue',   label='SMPL target (scaled)'),
        mpatches.Patch(color='red',    label='K1 FK (matched bodies)'),
        mpatches.Patch(color='orange', label='Error (target→K1)'),
        mpatches.Patch(color='steelblue', alpha=0.4, label='Full SMPL skeleton'),
        mpatches.Patch(color='salmon',    alpha=0.4, label='Full K1 skeleton'),
    ]
    axes[-1].legend(handles=legend, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",        default=osp.join(_K1_DATA_DIR, "amass_all.pkl"))
    parser.add_argument("--amass_root", default=osp.join(_H2H, "data", "AMASS", "AMASS_Complete"))
    parser.add_argument("--clip",       default=None, help="data key; default=first clip")
    parser.add_argument("--frame",      type=int, default=0)
    parser.add_argument("--out",        default="retarget_viz.png")
    parser.add_argument("--list",       action="store_true", help="print clip keys and exit")
    parser.add_argument("--gif",        action="store_true", help="produce animated GIF")
    parser.add_argument("--step",       type=int, default=10, help="frame step for GIF")
    parser.add_argument("--fps",        type=int, default=3,  help="GIF playback fps")
    args = parser.parse_args()

    print(f"Loading {args.pkl} ...")
    data = joblib.load(args.pkl)

    if args.list:
        for k in data.keys():
            print(" ", k)
        return

    clip_key = args.clip or next(iter(data.keys()))
    clip     = data[clip_key]
    N        = clip['pose_aa'].shape[0]
    frame    = min(args.frame, N - 1)
    print(f"Clip : {clip_key}  ({N} frames, using frame {frame})")

    device = torch.device('cpu')

    # ── K1 FK ─────────────────────────────────────────────────────────────────
    pose_aa_k1 = clip['pose_aa']  # (N, 23, 3)

    # ── SMPL joints ───────────────────────────────────────────────────────────
    npz_path = key_to_npz(clip_key, args.amass_root)
    if npz_path is None:
        print(f"[warn] Could not locate AMASS npz for key '{clip_key}' under {args.amass_root}")
        print("       SMPL targets will be omitted. Check --amass_root.")
        # Fall back: just plot K1 FK
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        for a, b in K1_LINKS:
            ax.plot([k1_f[a, 1], k1_f[b, 1]], [k1_f[a, 2], k1_f[b, 2]], 'r-', lw=1)
        ax.scatter(k1_f[:, 1], k1_f[:, 2], s=20, c='red')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_title(f"K1 FK only — frame {frame}")
        plt.savefig(args.out, dpi=130, bbox_inches='tight')
        plt.close()
        return

    print(f"AMASS npz: {npz_path}")
    trans_all, pose_aa_smpl_all, betas = load_amass_npz(npz_path)
    if trans_all is None:
        print("Failed to load AMASS npz (no mocap_framerate). Aborting.")
        return

    frame = min(frame, trans_all.shape[0] - 1, N - 1)

    smpl_parser = SMPL_Parser(model_path=_SMPL_DATA_PATH, gender='neutral')
    shape_new, leg_scale, arm_scale = joblib.load(osp.join(_K1_DATA_DIR, "shape_optimized_v1.pkl"))

    pose_t  = torch.from_numpy(pose_aa_smpl_all).float()
    trans_t = torch.from_numpy(trans_all).float()
    betas_t = torch.zeros(1, 10)

    with torch.no_grad():
        _, joints_raw    = smpl_parser.get_joints_verts(pose_t, betas_t,  trans_t)
        _, joints_shaped = smpl_parser.get_joints_verts(pose_t, shape_new, trans_t)

    # Scale arm/leg targets the same way as in grad_fit_k1.py
    root_pos = joints_shaped[:, 0:1]  # (N, 1, 3) — SMPL pelvis
    _s = torch.ones(9)
    _s[:5] = leg_scale
    _s[5:] = arm_scale
    joints_scaled = (joints_shaped[:, smpl_pick_idx] - root_pos) * _s[None, :, None] + root_pos

    smpl_f        = joints_raw[frame].numpy()     # (24, 3) raw SMPL full skeleton
    smpl_scaled_f = joints_scaled[frame].numpy()  # (9, 3)  scaled targets

    # K1 FK using SMPL pelvis as root — same frame as the optimizer used.
    # (The pkl stores a floor-adjusted root; we undo that here so both
    #  SMPL targets and K1 FK bodies are in the same coordinate frame.)
    root_trans_viz = joints_raw[:, 0].numpy()  # (N, 3) — SMPL pelvis = K1 root during optim
    k1_joints_all  = run_k1_fk(pose_aa_k1, root_trans_viz, device)  # (N, 23, 3)
    k1_f = k1_joints_all[frame]  # (23, 3)

    # ── Error report ──────────────────────────────────────────────────────────
    k1_matched = k1_f[k1_pick_idx]  # (9, 3)
    errs = np.linalg.norm(smpl_scaled_f - k1_matched, axis=-1) * 1000
    print(f"\nPer-joint errors at frame {frame} (mm):")
    for name, err in zip(k1_joint_pick, errs):
        print(f"  {name:25s}  {err:.1f}")
    print(f"  {'mean':25s}  {errs.mean():.1f}")
    leg_err = errs[:5].mean()
    arm_err = errs[5:].mean()
    print(f"\n  leg mean={leg_err:.1f}mm   arm mean={arm_err:.1f}mm")

    # ── DOF values at this frame ───────────────────────────────────────────────
    if 'dof' in clip:
        dof = clip['dof']  # (N, 22) or (N, 22, 1)
        dof = dof.reshape(N, 22)
        elbow_l = dof[frame, 4]
        elbow_r = dof[frame, 8]
        shou_l  = dof[frame, 3]
        shou_r  = dof[frame, 7]
        print(f"\n  DOF at frame {frame}:")
        print(f"    L shoulder_roll={shou_l:.3f} rad  L elbow_pitch={elbow_l:.3f} rad ({np.degrees(elbow_l):.1f}°)")
        print(f"    R shoulder_roll={shou_r:.3f} rad  R elbow_pitch={elbow_r:.3f} rad ({np.degrees(elbow_r):.1f}°)")

    if not args.gif:
        plot_frame(smpl_f, k1_f, smpl_scaled_f, frame, args.out)
        return

    # ── GIF mode ──────────────────────────────────────────────────────────────
    import io
    from PIL import Image

    gif_out = args.out.replace(".png", ".gif") if args.out.endswith(".png") else args.out + ".gif"
    frame_indices = range(0, min(N, joints_raw.shape[0]), args.step)
    dof_arr = clip['dof'].reshape(N, 22) if 'dof' in clip else None

    pil_frames = []
    for fi in frame_indices:
        smpl_fi   = joints_raw[fi].numpy()
        scaled_fi = joints_scaled[fi].numpy()
        k1_fi     = k1_joints_all[fi]

        ep_l = f"{np.degrees(dof_arr[fi,4]):.0f}°" if dof_arr is not None else "?"
        ep_r = f"{np.degrees(dof_arr[fi,8]):.0f}°" if dof_arr is not None else "?"

        views = [
            ("Front (X-Z)", 0, 2, "X", "Z"),
            ("Side (Y-Z)",  1, 2, "Y", "Z"),
            ("Top  (X-Y)",  0, 1, "X", "Y"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"frame {fi}/{N-1}   L_elbow={ep_l}  R_elbow={ep_r}  "
            f"(step={args.step})",
            fontsize=11
        )

        for ax, (title, xi, yi, xl, yl) in zip(axes, views):
            for a, b in SMPL_LINKS:
                ax.plot([smpl_fi[a,xi], smpl_fi[b,xi]], [smpl_fi[a,yi], smpl_fi[b,yi]],
                        color='steelblue', lw=0.8, alpha=0.4)
            ax.scatter(smpl_fi[:,xi], smpl_fi[:,yi], s=8, color='steelblue', alpha=0.4, zorder=3)
            ax.scatter(scaled_fi[:,xi], scaled_fi[:,yi], s=50, color='blue', zorder=5)

            for a, b in K1_LINKS:
                ax.plot([k1_fi[a,xi], k1_fi[b,xi]], [k1_fi[a,yi], k1_fi[b,yi]],
                        color='salmon', lw=0.8, alpha=0.5)
            ax.scatter(k1_fi[:,xi], k1_fi[:,yi], s=8, color='salmon', alpha=0.5, zorder=3)
            k1_m = k1_fi[k1_pick_idx]
            ax.scatter(k1_m[:,xi], k1_m[:,yi], s=50, color='red', marker='s', zorder=5)

            for i in range(len(k1_pick_idx)):
                ax.plot([scaled_fi[i,xi], k1_m[i,xi]], [scaled_fi[i,yi], k1_m[i,yi]],
                        color='orange', lw=0.8, linestyle='--', alpha=0.6, zorder=4)

            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())
        buf.close()

    duration_ms = int(1000 / args.fps)
    pil_frames[0].save(
        gif_out,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
    )
    print(f"GIF ({len(pil_frames)} frames, {args.fps} fps) → {gif_out}")


if __name__ == "__main__":
    main()
