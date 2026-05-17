"""Render a retargeted K1 motion pkl to a GIF or MP4.

Usage:
    python render_k1.py --pkl data/k1/CMU_144_144_01_poses.pkl
    python render_k1.py --pkl data/k1/CMU_144_144_01_poses.pkl --out k1_motion.gif
    python render_k1.py --pkl data/k1/CMU_144_144_01_poses.pkl --out k1_motion.mp4
    python render_k1.py --pkl data/k1/CMU_144_144_01_poses.pkl --fps 30 --width 640 --height 480
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pathlib
import sys
import numpy as np
import mujoco
import imageio
import joblib
from tqdm import tqdm

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_XML = HERE.parents[3] / "neural_wbc/data/data/mujoco/models/scene_k1_vis.xml"
FALLBACK_XML = HERE.parent / "assets/booster_k1/K1_serial.xml"


def make_camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def get_scene_option():
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1  # visual meshes
    opt.geomgroup[0] = 1  # floor
    return opt


_GREEN       = np.array([0.1, 0.9, 0.1, 0.85], dtype=np.float32)
_RED         = np.array([0.9, 0.1, 0.1, 0.85], dtype=np.float32)
_BLUE        = np.array([0.1, 0.1, 0.9, 0.85], dtype=np.float32)
_LIGHT_GREEN = np.array([0.5, 1.0, 0.5, 0.6],  dtype=np.float32)  # SMPL target hand
_LIGHT_RED   = np.array([1.0, 0.5, 0.5, 0.6],  dtype=np.float32)  # SMPL target elbow
_SPHERE_R = 0.04

# left_hand_tip / right_hand_tip site offsets in each body's local frame (from K1_serial.xml)
_L_TIP_LOCAL = np.array([0,  0.228, 0])
_R_TIP_LOCAL = np.array([0, -0.228, 0])


def _add_sphere(scene, pos, rgba=_GREEN, radius=_SPHERE_R):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius),
        np.asarray(pos, dtype=np.float64),
        np.eye(3).flatten(),
        rgba,
    )
    scene.ngeom += 1


def render_motion(model, data, qpos_seq, renderer, cam, opt, fps,
                  smpl_elbow=None, smpl_wrist=None):
    try:
        # left_hand_link xpos = Left_Elbow_Pitch joint position = elbow
        l_elbow_id    = model.body("left_hand_link").id
        r_elbow_id    = model.body("right_hand_link").id
        l_shoulder_id = model.body("Left_Arm_1").id
        r_shoulder_id = model.body("Right_Arm_1").id
        have_markers  = True
    except Exception:
        have_markers  = False

    frames = []
    for i, qpos in enumerate(tqdm(qpos_seq, desc="Rendering")):
        data.qpos[:len(qpos)] = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        if have_markers:
            l_elbow_pos = data.xpos[l_elbow_id]
            r_elbow_pos = data.xpos[r_elbow_id]
            l_hand_pos  = l_elbow_pos + data.xmat[l_elbow_id].reshape(3, 3) @ _L_TIP_LOCAL
            r_hand_pos  = r_elbow_pos + data.xmat[r_elbow_id].reshape(3, 3) @ _R_TIP_LOCAL
            _add_sphere(renderer.scene, data.xpos[l_shoulder_id], _BLUE)
            _add_sphere(renderer.scene, data.xpos[r_shoulder_id], _BLUE)
            _add_sphere(renderer.scene, l_elbow_pos, _RED)
            _add_sphere(renderer.scene, r_elbow_pos, _RED)
            _add_sphere(renderer.scene, l_hand_pos,  _GREEN)
            _add_sphere(renderer.scene, r_hand_pos,  _GREEN)
        if smpl_elbow is not None:
            _add_sphere(renderer.scene, smpl_elbow[i, 0], _LIGHT_RED)
            _add_sphere(renderer.scene, smpl_elbow[i, 1], _LIGHT_RED)
        if smpl_wrist is not None:
            _add_sphere(renderer.scene, smpl_wrist[i, 0], _LIGHT_GREEN)
            _add_sphere(renderer.scene, smpl_wrist[i, 1], _LIGHT_GREEN)
        frames.append(renderer.render().copy())
    return frames


def parse_args():
    p = argparse.ArgumentParser(description="Render K1 retargeted motion to GIF or MP4.")
    p.add_argument("--pkl", required=True, help="Path to individual or combined .pkl produced by retarget_k1.sh")
    p.add_argument("--key", default=None,
                   help="Clip key to load from a combined pkl (e.g. CMU_144_144_01_poses). "
                        "If omitted and pkl is combined, lists available keys.")
    p.add_argument("--out", default=None,
                   help="Output path (.gif or .mp4). Defaults to <pkl_stem>.gif next to the pkl.")
    p.add_argument("--xml", default=None,
                   help="MuJoCo scene XML. Defaults to scene_k1_vis.xml if available.")
    p.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
    p.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    p.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    p.add_argument("--azimuth", type=float, default=160, help="Camera azimuth degrees (default: 160)")
    p.add_argument("--elevation", type=float, default=-15, help="Camera elevation degrees (default: -15)")
    p.add_argument("--distance", type=float, default=2.5, help="Camera distance (default: 2.5)")
    p.add_argument("--max-frames", type=int, default=None, help="Only render the first N frames")
    return p.parse_args()


def main():
    args = parse_args()

    pkl_path = pathlib.Path(args.pkl)
    data_dict = joblib.load(pkl_path)

    # Resolve which clip dict to load.
    # Formats:
    #   GMR individual pkl  : {"k1": {"qpos": ..., "smpl_elbow": ..., ...}}
    #   grad-fit pkl        : {"0-CMU_...": {"dof": ..., "root_trans_offset": ..., "root_rot": ..., "fps": 30}}
    if "k1" in data_dict:
        k1 = data_dict["k1"]
    else:
        keys = list(data_dict.keys())
        if args.key is None:
            print("Combined pkl detected. Available keys:")
            for k in keys:
                print(f"  {k}")
            print("\nRe-run with --key <stem> to render a clip.")
            sys.exit(0)
        if args.key not in data_dict:
            print(f"Key '{args.key}' not found. Available: {keys}")
            sys.exit(1)
        k1 = data_dict[args.key]

    if "qpos" in k1:
        # GMR format — qpos already built
        qpos_seq = k1["qpos"]
    else:
        # grad-fit format: reconstruct qpos = [root_xyz | root_wxyz | dof]
        root_pos  = k1["root_trans_offset"]          # (T, 3)
        root_xyzw = k1["root_rot"]                   # (T, 4) scipy xyzw
        root_wxyz = root_xyzw[:, [3, 0, 1, 2]]       # → wxyz for MuJoCo
        dof       = k1["dof"]                        # (T, 22)
        qpos_seq  = np.concatenate([root_pos, root_wxyz, dof], axis=-1)

    smpl_elbow = k1.get("smpl_elbow")  # (M, 2, 3) or None
    smpl_wrist = k1.get("smpl_wrist")  # (M, 2, 3) or None
    source_npz = k1.get("source_npz", "unknown")
    total_frames = len(qpos_seq)
    if args.max_frames is not None:
        qpos_seq = qpos_seq[:args.max_frames]
        if smpl_elbow is not None:
            smpl_elbow = smpl_elbow[:args.max_frames]
        if smpl_wrist is not None:
            smpl_wrist = smpl_wrist[:args.max_frames]
    render_frames = len(qpos_seq)
    print(f"Source NPZ : {source_npz}")
    print(f"Frames     : {render_frames} / {total_frames} total")

    # Pick XML
    if args.xml:
        xml_path = pathlib.Path(args.xml)
    elif DEFAULT_XML.exists():
        xml_path = DEFAULT_XML
    else:
        xml_path = FALLBACK_XML
    print(f"Using scene: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    cam = make_camera(
        lookat=[0.0, 0.0, 0.7],
        distance=args.distance,
        azimuth=args.azimuth,
        elevation=args.elevation,
    )
    opt = get_scene_option()

    out_path = pathlib.Path(args.out) if args.out else pkl_path.with_suffix(".gif")

    with mujoco.Renderer(model, height=args.height, width=args.width) as renderer:
        print(f"Rendering {len(qpos_seq)} frames at {args.fps} FPS ...")
        frames = render_motion(model, data, qpos_seq, renderer, cam, opt, args.fps,
                               smpl_elbow=smpl_elbow, smpl_wrist=smpl_wrist)

    print(f"Writing {out_path} ...")
    if out_path.suffix == ".mp4":
        imageio.mimwrite(str(out_path), frames, fps=args.fps, codec="libx264")
    else:
        duration_ms = int(1000 / args.fps)
        imageio.mimwrite(str(out_path), frames, duration=duration_ms, loop=0)

    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
