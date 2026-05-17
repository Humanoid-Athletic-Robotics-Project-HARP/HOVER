"""Retarget an AMASS motion clip to the Booster K1 robot via GMR IK.

Accepts AMASS SMPL .npz files directly (e.g. CMU/144/144_01_poses.npz).
The SMPL body model at --smpl_model_dir is the same one used by the H1
pipeline (set up by retarget_k1.sh or retarget_h1.sh).

Usage:
    python smplx_to_k1.py \
        --amass_file third_party/human2humanoid/data/AMASS/AMASS_Complete/CMU/144/144_01_poses.npz \
        --save_path output/k1_motion.pkl

    python smplx_to_k1.py \
        --amass_file CMU/144/144_01_poses.npz \
        --amass_root third_party/human2humanoid/data/AMASS/AMASS_Complete \
        --save_path output/k1_motion.pkl \
        --actual_human_height 1.75
"""

import argparse
import pathlib
import sys
import time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]   # hover project root (scripts/ → gmr_k1/ → third_party/ → hover/)
ASSETS = HERE.parent / "assets"
IK_CONFIGS = HERE.parent / "ik_configs"

# Add phc to path so SMPL_Parser is importable (same package the H1 pipeline uses).
_PHC_ROOT = REPO_ROOT / "third_party" / "human2humanoid" / "phc"
if _PHC_ROOT.exists() and str(_PHC_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHC_ROOT))

# Override GMR's path registry before importing GeneralMotionRetargeting.
import general_motion_retargeting.params as _params
_params.ROBOT_XML_DICT["booster_k1"] = ASSETS / "booster_k1" / "K1_serial.xml"
_params.IK_CONFIG_DICT["smplx"]["booster_k1"] = IK_CONFIGS / "smplx_to_k1.json"

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer

import joblib
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# SMPL kinematic tree parent indices (24 joints, standard SMPL).
_SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# Map PHC SMPL_BONE_ORDER_NAMES → SMPL-X joint names used in the GMR IK config.
_PHC_TO_SMPLX = {
    "Pelvis":     "pelvis",
    "L_Hip":      "left_hip",
    "R_Hip":      "right_hip",
    "L_Knee":     "left_knee",
    "R_Knee":     "right_knee",
    "L_Ankle":    "left_ankle",
    "R_Ankle":    "right_ankle",
    "L_Toe":      "left_foot",
    "R_Toe":      "right_foot",
    "Head":       "head",
    "L_Shoulder": "left_shoulder",
    "R_Shoulder": "right_shoulder",
    "L_Elbow":    "left_elbow",
    "R_Elbow":    "right_elbow",
    "L_Wrist":    "left_wrist",
    "R_Wrist":    "right_wrist",
}


def load_amass_frames(npz_path: pathlib.Path, smpl_model_dir: pathlib.Path, tgt_fps: int = 30):
    """Load an AMASS .npz clip and return a list of per-frame joint dicts.

    Each frame dict maps SMPL-X joint names to (world_pos, world_quat_wxyz)
    tuples, matching the format expected by GMR's retarget().
    """
    from phc.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES  # noqa: F401

    data = dict(np.load(str(npz_path), allow_pickle=True))

    fps_key = "mocap_framerate" if "mocap_framerate" in data else "mocap_frame_rate"
    if fps_key not in data:
        raise ValueError(f"No mocap_framerate key in {npz_path}")
    src_fps = float(data[fps_key])
    frame_skip = max(1, int(round(src_fps / tgt_fps)))

    poses = data["poses"]          # (N, 72 or 156)
    if poses.shape[1] > 72:
        poses = poses[:, :72]
    trans = data["trans"]          # (N, 3)
    betas = np.asarray(data.get("betas", np.zeros(10)), dtype=np.float32).ravel()[:10]

    idx = list(range(0, poses.shape[0], frame_skip))
    poses_s = poses[idx].astype(np.float32)   # (M, 72)
    trans_s = trans[idx].astype(np.float32)   # (M, 3)
    M = len(idx)

    smpl = SMPL_Parser(model_path=str(smpl_model_dir), gender="neutral")

    with torch.no_grad():
        _, joints_w = smpl.get_joints_verts(
            torch.from_numpy(poses_s),
            torch.from_numpy(betas).unsqueeze(0),
            torch.from_numpy(trans_s),
        )
    joints_w = joints_w.numpy()  # (M, 24+, 3)

    all_rotvecs = poses_s.reshape(M, -1, 3)  # (M, 24, 3) local axis-angles

    frames = []
    for f in range(M):
        frame = {}
        global_rots = []
        for i, phc_name in enumerate(SMPL_BONE_ORDER_NAMES):
            if i >= all_rotvecs.shape[1]:
                break
            local_rv = all_rotvecs[f, i]
            rot = R.from_rotvec(local_rv) if i == 0 else global_rots[_SMPL_PARENTS[i]] * R.from_rotvec(local_rv)
            global_rots.append(rot)
            smplx_name = _PHC_TO_SMPLX.get(phc_name)
            if smplx_name is not None:
                frame[smplx_name] = (joints_w[f, i].copy(), rot.as_quat(scalar_first=True))
        frames.append(frame)

    return frames


def parse_args():
    p = argparse.ArgumentParser(description="Retarget AMASS motion to Booster K1.")
    p.add_argument("--amass_file", required=True,
                   help="Path to an AMASS .npz clip. Can be absolute or relative to --amass_root.")
    p.add_argument("--amass_root", default=None,
                   help="Root of the extracted AMASS dataset. "
                        "Defaults to third_party/human2humanoid/data/AMASS/AMASS_Complete.")
    p.add_argument("--smpl_model_dir", default=None,
                   help="Dir containing SMPL_NEUTRAL.pkl etc. "
                        "Defaults to third_party/human2humanoid/data/smpl/.")
    p.add_argument("--save_path", default=None,
                   help="Output .pkl path for retargeted joint angles.")
    p.add_argument("--actual_human_height", type=float, default=None,
                   help="Subject height in metres for proportional scaling.")
    p.add_argument("--visualize", action="store_true",
                   help="Show MuJoCo viewer after retargeting.")
    p.add_argument("--record_video", action="store_true",
                   help="Record a video of the retargeted motion.")
    p.add_argument("--video_path", default="k1_retarget.mp4",
                   help="Output video path (used with --record_video).")
    return p.parse_args()


def main():
    args = parse_args()

    amass_root = pathlib.Path(args.amass_root) if args.amass_root else \
        REPO_ROOT / "third_party" / "human2humanoid" / "data" / "AMASS" / "AMASS_Complete"

    smpl_model_dir = pathlib.Path(args.smpl_model_dir) if args.smpl_model_dir else \
        REPO_ROOT / "third_party" / "human2humanoid" / "data" / "smpl"

    amass_file = pathlib.Path(args.amass_file)
    if not amass_file.is_absolute():
        amass_file = amass_root / amass_file

    for path, label in [(amass_file, "amass_file"), (smpl_model_dir, "smpl_model_dir")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                "Run './retarget_k1.sh --setup' to extract AMASS and SMPL data."
            )

    gmr = GMR(
        src_human="smplx",
        tgt_robot="booster_k1",
        actual_human_height=args.actual_human_height,
    )

    print(f"Loading {amass_file} ...")
    motion_frames = load_amass_frames(amass_file, smpl_model_dir)

    _ARM_BODIES = {"left_hand_link", "right_hand_link", "left_hand_tip", "right_hand_tip"}
    _arm_tasks2 = [t for t in gmr.tasks2 if t.frame_name in _ARM_BODIES]
    _leg_tasks2  = [t for t in gmr.tasks2 if t.frame_name not in _ARM_BODIES]

    def _task_err_mm(tasks):
        if not tasks:
            return 0.0
        errs = np.concatenate([t.compute_error(gmr.configuration) for t in tasks])
        return float(np.linalg.norm(errs) * 1000)

    t0 = time.time()
    qpos_seq = []
    smpl_elbows, smpl_wrists = [], []
    arm_errs, leg_errs = [], []
    pbar = tqdm(motion_frames, desc="Retargeting", unit="frame")
    for frame in pbar:
        gmr.retarget(frame, offset_to_ground=True)
        qpos_seq.append(gmr.configuration.data.qpos.copy())
        smpl_elbows.append(np.stack([
            frame.get("left_elbow",  (np.zeros(3), None))[0],
            frame.get("right_elbow", (np.zeros(3), None))[0],
        ]))
        smpl_wrists.append(np.stack([
            frame.get("left_wrist",  (np.zeros(3), None))[0],
            frame.get("right_wrist", (np.zeros(3), None))[0],
        ]))
        arm_mm = _task_err_mm(_arm_tasks2)
        leg_mm = _task_err_mm(_leg_tasks2)
        arm_errs.append(arm_mm)
        leg_errs.append(leg_mm)
        pbar.set_postfix(arm_mm=f"{arm_mm:.0f}", leg_mm=f"{leg_mm:.0f}")
    elapsed = time.time() - t0
    fps_out = len(motion_frames) / elapsed
    print(f"Done — {len(motion_frames)} frames in {elapsed:.1f}s ({fps_out:.1f} FPS)")
    print(f"  mean IK err  arm: {np.mean(arm_errs):.1f} mm   leg: {np.mean(leg_errs):.1f} mm")

    qpos_seq = np.array(qpos_seq)

    if args.save_path:
        out = pathlib.Path(args.save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "k1": {
                "qpos":        qpos_seq,
                "smpl_elbow":  np.array(smpl_elbows),  # (M, 2, 3) left then right
                "smpl_wrist":  np.array(smpl_wrists),  # (M, 2, 3) left then right
                "source_npz":  str(amass_file),
            }
        }, out)
        print(f"Saved → {out}")

    if args.visualize or args.record_video:
        viewer = RobotMotionViewer(
            robot="booster_k1",
            robot_motion=qpos_seq,
            record_video=args.record_video,
            video_path=args.video_path,
        )
        viewer.run()


if __name__ == "__main__":
    main()
