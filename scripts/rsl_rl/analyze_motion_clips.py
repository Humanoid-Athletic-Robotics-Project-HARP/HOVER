# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Offline analysis for retargeted motion PKLs (e.g. amass_all.pkl): per-clip height / floor stats.
#
# Usage (from repo root, same Python env as human2humanoid / optional neural_wbc):
#   python scripts/rsl_rl/analyze_motion_clips.py --motion_path neural_wbc/data/data/motions/amass_all.pkl
#   python scripts/rsl_rl/analyze_motion_clips.py --motion_path ... --fk --device cuda
#   python scripts/rsl_rl/analyze_motion_clips.py --motion_path ... --runtime_ref --runtime_max_clips 200
#
# Outputs CSV rows you can sort by min_fk_z / min_runtime_foot_z to find bad height clips.

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

HOVER_ROOT = Path(__file__).resolve().parents[2]


def _ensure_h2h_path() -> None:
    h2h = HOVER_ROOT / "third_party" / "human2humanoid"
    if h2h.is_dir() and str(h2h) not in sys.path:
        sys.path.insert(0, str(h2h))


def _default_motion_path() -> str:
    p = HOVER_ROOT / "neural_wbc" / "data" / "data" / "motions" / "amass_all.pkl"
    return str(p)


def _default_mjcf() -> str:
    try:
        from neural_wbc.data import get_data_path

        return get_data_path("motion_lib/k1.xml")
    except Exception:
        return str(HOVER_ROOT / "neural_wbc" / "data" / "data" / "motion_lib" / "k1.xml")


def clip_raw_stats(key: str, clip: dict) -> dict:
    rt = np.asarray(clip["root_trans_offset"], dtype=np.float64)
    z = rt[:, 2]
    return {
        "key": key,
        "frames": int(rt.shape[0]),
        "fps": int(clip.get("fps", 30)),
        "root_z_min": float(z.min()),
        "root_z_max": float(z.max()),
        "root_z_mean": float(z.mean()),
        "root_z_p01": float(np.percentile(z, 1)),
        "root_z_p99": float(np.percentile(z, 99)),
    }


def fk_height_stats(clip: dict, hb, device: str, stride: int, max_frames: int) -> dict:
    import torch

    pose = torch.as_tensor(clip["pose_aa"], dtype=torch.float32, device=device)
    trans = torch.as_tensor(clip["root_trans_offset"], dtype=torch.float32, device=device)
    t_all = pose.shape[0]
    idx = np.arange(0, t_all, stride, dtype=np.int64)
    if idx.size == 0:
        idx = np.array([0], dtype=np.int64)
    if idx.size > max_frames:
        idx = np.linspace(0, t_all - 1, max_frames).astype(np.int64)
    pose = pose[idx].unsqueeze(0)
    trans = trans[idx].unsqueeze(0)
    fps = int(clip.get("fps", 30))
    dt = 1.0 / max(fps, 1)
    with torch.no_grad():
        out = hb.fk_batch(pose, trans, return_full=True, dt=dt)
    g = out["global_translation"]
    # (1, Ts, J, 3)
    z_bodies = g[0, :, :, 2]
    min_z_all_bodies = float(z_bodies.min().item())
    min_z_per_frame = z_bodies.min(dim=-1).values
    root_z = g[0, :, 0, 2]

    names = list(hb.model_names)
    foot_idx = [names.index(n) for n in ("left_foot_link", "right_foot_link") if n in names]
    if len(foot_idx) == 2:
        fz = z_bodies[:, foot_idx]
        min_foot_z = float(fz.min().item())
        mean_foot_z = float(fz.mean().item())
    else:
        min_foot_z = float("nan")
        mean_foot_z = float("nan")

    return {
        "fk_min_z_all_bodies": min_z_all_bodies,
        "fk_min_z_feet": min_foot_z,
        "fk_mean_z_feet": mean_foot_z,
        "fk_min_root_z": float(root_z.min().item()),
        "fk_max_root_z": float(root_z.max().item()),
        "fk_frames_sampled": int(z_bodies.shape[0]),
    }


def runtime_ref_height_stats(
    clip: dict,
    key: str,
    skeleton_path: str,
    fk_q_wxyz,
    policy_dt: float,
    device: str,
    num_samples: int,
) -> dict:
    """Mirror training ReferenceMotionManager + get_state_from_motion_lib_cache (no terrain offset)."""
    import torch

    from neural_wbc.core.reference_motion import ReferenceMotionManager, ReferenceMotionManagerCfg

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        joblib.dump({key: clip}, tmp_path)
        cfg = ReferenceMotionManagerCfg(
            motion_path=tmp_path,
            skeleton_path=skeleton_path,
            fk_frame_rotation=list(fk_q_wxyz),
        )
        mgr = ReferenceMotionManager(
            cfg=cfg,
            device=torch.device(device),
            num_envs=1,
            random_sample=False,
            extend_head=False,
            extend_hand=False,
            dt=policy_dt,
        )
        n_steps_t = mgr.get_motion_num_steps()
        n_steps = int(n_steps_t.reshape(-1)[0].item())
        if n_steps <= 0:
            return {k: float("nan") for k in ("rt_min_root_z", "rt_min_foot_z", "rt_min_body_z")}
        idxs = np.linspace(0, n_steps - 1, num=min(num_samples, n_steps)).astype(np.int64)
        dev = torch.device(device)
        mins_root = []
        mins_foot = []
        mins_body = []
        names = list(mgr.body_extended_names)
        foot_ids = [names.index(n) for n in ("left_foot_link", "right_foot_link") if n in names]

        for si in idxs:
            buf = torch.full((1,), float(si), device=dev, dtype=torch.float32)
            st = mgr.get_state_from_motion_lib_cache(
                episode_length_buf=buf,
                terrain_heights=None,
                offset=None,
            )
            rp = st.root_pos[0, 2].item()
            mins_root.append(rp)
            be = st.body_pos_extend[0]
            mins_body.append(float(be[:, 2].min().item()))
            if len(foot_ids) == 2:
                fz = be[foot_ids, 2]
                mins_foot.append(float(fz.min().item()))
        return {
            "rt_min_root_z": float(min(mins_root)),
            "rt_max_root_z": float(max(mins_root)),
            "rt_min_body_z": float(min(mins_body)),
            "rt_min_foot_z": float(min(mins_foot)) if mins_foot else float("nan"),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze per-clip height statistics in a retargeted motion PKL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--motion_path", type=str, default=None, help="Path to joblib PKL (dict of clips).")
    parser.add_argument("--mjcf", type=str, default=None, help="MJCF for FK / runtime ref (default: data path).")
    parser.add_argument("--max_clips", type=int, default=None, help="Only process first N keys (stable sort by key).")
    parser.add_argument("--csv", type=str, default=None, help="Write CSV summary path.")
    parser.add_argument("--fk", action="store_true", help="Run Humanoid_Batch FK subsample (retarget frame).")
    parser.add_argument("--fk_stride", type=int, default=8, help="Frame stride for FK subsampling.")
    parser.add_argument("--fk_max_frames", type=int, default=120, help="Cap subsampled frames per clip for FK.")
    parser.add_argument("--device", type=str, default="cpu", help="torch device for FK / runtime (cuda:0, cpu).")
    parser.add_argument(
        "--runtime_ref",
        action="store_true",
        help="Load each clip via ReferenceMotionManager (matches training FK + fk_frame_rotation). Slow.",
    )
    parser.add_argument("--runtime_max_clips", type=int, default=100, help="Max clips for --runtime_ref (sorted keys).")
    parser.add_argument("--runtime_samples", type=int, default=12, help="Time samples per clip for runtime_ref.")
    parser.add_argument(
        "--policy_dt",
        type=float,
        default=0.02,
        help="Policy step dt (decimation * sim dt), default 4 * 0.005.",
    )
    parser.add_argument(
        "--fk_frame_rotation",
        type=float,
        nargs=4,
        default=[0.5, 0.5, 0.5, 0.5],
        metavar=("W", "X", "Y", "Z"),
        help="wxyz quaternion for runtime_ref.",
    )
    args = parser.parse_args()

    if str(HOVER_ROOT) not in sys.path:
        sys.path.insert(0, str(HOVER_ROOT))

    motion_path = os.path.abspath(args.motion_path or _default_motion_path())
    if not os.path.isfile(motion_path):
        print(f"[ERROR] Not found: {motion_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading {motion_path}")
    data: dict = joblib.load(motion_path)
    keys = sorted(data.keys())
    if args.max_clips is not None:
        keys = keys[: args.max_clips]
    print(f"[INFO] Clips to process: {len(keys)}")

    mjcf_path = os.path.abspath(args.mjcf or _default_mjcf())
    if not os.path.isfile(mjcf_path):
        print(f"[WARN] MJCF not found at {mjcf_path}; --fk / --runtime_ref need a valid path.")

    hb = None
    if args.fk:
        _ensure_h2h_path()
        import torch
        from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch

        hb = Humanoid_Batch(mjcf_file=mjcf_path, extend_hand=False, extend_head=False, device=torch.device(args.device))

    rows = []
    runtime_keys = keys
    if args.runtime_ref:
        runtime_keys = keys[: args.runtime_max_clips]

    for i, key in enumerate(keys):
        clip = data[key]
        row = clip_raw_stats(key, clip)
        if hb is not None:
            row.update(fk_height_stats(clip, hb, args.device, args.fk_stride, args.fk_max_frames))
        if args.runtime_ref and key in runtime_keys:
            try:
                row.update(
                    runtime_ref_height_stats(
                        clip,
                        key,
                        mjcf_path,
                        tuple(args.fk_frame_rotation),
                        args.policy_dt,
                        args.device,
                        args.runtime_samples,
                    )
                )
            except Exception as e:
                row["runtime_error"] = str(e)
        rows.append(row)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(keys)}] {key[:80]} root_z_min={row['root_z_min']:.4f}", flush=True)

    if rows:
        sort_key = "rt_min_foot_z" if args.runtime_ref and "rt_min_foot_z" in rows[0] else (
            "fk_min_z_feet" if args.fk and "fk_min_z_feet" in rows[0] else "root_z_min"
        )

        def _sort_val(r: dict) -> float:
            v = r.get(sort_key, float("nan"))
            return float(v) if not np.isnan(v) else 1e9

        ranked = sorted(rows, key=_sort_val)
        print(f"\n[INFO] Lowest 15 clips by {sort_key} (most suspicious floor / float):")
        for r in ranked[:15]:
            print(
                f"  {sort_key}={r.get(sort_key, float('nan')):.4f}  "
                f"root_z_min={r['root_z_min']:.4f}  frames={r['frames']}  {r['key'][:100]}"
            )

    if args.csv:
        out_path = os.path.abspath(args.csv)
        if rows:
            fieldnames = list(rows[0].keys())
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(f"[INFO] Wrote {out_path}")
        else:
            print("[WARN] No rows for CSV")


if __name__ == "__main__":
    main()
