import argparse
import glob
import os
import sys
import os.path as osp
sys.path.append(os.getcwd())

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from torch.autograd import Variable
from tqdm import tqdm

from phc.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch
import phc.utils.rotation_conversions as tRot

_HERE = osp.dirname(osp.abspath(__file__))
_HOVER_ROOT = osp.normpath(osp.join(_HERE, "..", ".."))
_H2H_ROOT = osp.join(_HOVER_ROOT, "third_party", "human2humanoid")
K1_MJCF = osp.join(_HOVER_ROOT, "neural_wbc", "data", "data", "motion_lib", "k1.xml")
_SMPL_DATA_PATH = osp.join(_H2H_ROOT, "data", "smpl")
_K1_DATA_DIR = osp.join(_H2H_ROOT, "data", "k1")

K1_ROTATION_AXIS = torch.tensor([[
    [0, 0, 1],  # AAHead_yaw
    [0, 1, 0],  # Head_pitch
    [0, 1, 0],  # ALeft_Shoulder_Pitch
    [1, 0, 0],  # Left_Shoulder_Roll
    [0, 1, 0],  # Left_Elbow_Pitch
    [0, 0, 1],  # Left_Elbow_Yaw
    [0, 1, 0],  # ARight_Shoulder_Pitch
    [1, 0, 0],  # Right_Shoulder_Roll
    [0, 1, 0],  # Right_Elbow_Pitch
    [0, 0, 1],  # Right_Elbow_Yaw
    [0, 1, 0],  # Left_Hip_Pitch
    [1, 0, 0],  # Left_Hip_Roll
    [0, 0, 1],  # Left_Hip_Yaw
    [0, 1, 0],  # Left_Knee_Pitch
    [0, 1, 0],  # Left_Ankle_Pitch
    [1, 0, 0],  # Left_Ankle_Roll
    [0, 1, 0],  # Right_Hip_Pitch
    [1, 0, 0],  # Right_Hip_Roll
    [0, 0, 1],  # Right_Hip_Yaw
    [0, 1, 0],  # Right_Knee_Pitch
    [0, 1, 0],  # Right_Ankle_Pitch
    [1, 0, 0],  # Right_Ankle_Roll
]])  # (1, 22, 3)

K1_BODY_NAMES = [
    'Trunk',
    'Head_1', 'Head_2',
    'Left_Arm_1', 'Left_Arm_2', 'Left_Arm_3', 'left_hand_link',
    'Right_Arm_1', 'Right_Arm_2', 'Right_Arm_3', 'right_hand_link',
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw', 'Left_Shank', 'Left_Ankle_Cross', 'left_foot_link',
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw', 'Right_Shank', 'Right_Ankle_Cross', 'right_foot_link',
]

# Left_Arm_1 / Right_Arm_1 are fixed relative to Trunk — no arm DOF changes
# them, so they only contribute irreducible ~20mm constant error and zero
# arm gradient.  Use 2 targets per side instead:
#   Left_Arm_3     -> L_Elbow   (~35 mm T-pose error, moveable via shoulder)
#   left_hand_link -> L_Hand    (~44 mm T-pose error, moveable via shoulder)
k1_joint_pick  = ['Trunk', 'Left_Shank', 'left_foot_link', 'Right_Shank', 'right_foot_link',
                  'Left_Arm_3', 'left_hand_link',
                  'Right_Arm_3', 'right_hand_link']
smpl_joint_pick = ['Pelvis', 'L_Knee', 'L_Ankle', 'R_Knee', 'R_Ankle',
                   'L_Elbow', 'L_Hand',
                   'R_Elbow', 'R_Hand']
k1_joint_pick_idx   = [K1_BODY_NAMES.index(j) for j in k1_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

SMPL_ROOT_ALIGN_QUAT = [0.5, 0.5, 0.5, 0.5]

SMPL_L_WRIST_IDX    = SMPL_BONE_ORDER_NAMES.index('L_Wrist')     # 20
SMPL_R_WRIST_IDX    = SMPL_BONE_ORDER_NAMES.index('R_Wrist')     # 21
SMPL_L_SHOULDER_IDX = SMPL_BONE_ORDER_NAMES.index('L_Shoulder')
SMPL_R_SHOULDER_IDX = SMPL_BONE_ORDER_NAMES.index('R_Shoulder')
K1_L_HAND_BODY_IDX  = K1_BODY_NAMES.index('left_hand_link')      # 6
K1_R_HAND_BODY_IDX  = K1_BODY_NAMES.index('right_hand_link')     # 10
ROT_WEIGHT = 0.005


def _smpl_global_rot(gt_root_rot_aa, pose_aa_walk, parents):
    """SMPL global rotation matrices in the sim world frame.

    Uses the already-aligned root (gt_root_rot_aa) so the resulting matrices
    are directly comparable to K1 FK global_rotation_mat output.
    """
    N = pose_aa_walk.shape[0]
    J = pose_aa_walk.shape[1] // 3
    local_aa = pose_aa_walk.reshape(N, J, 3).clone()
    local_aa[:, 0] = gt_root_rot_aa
    local_mat = tRot.axis_angle_to_matrix(local_aa)   # (N, J, 3, 3)
    g = [None] * J
    g[0] = local_mat[:, 0]
    plist = parents.tolist()
    for j in range(1, J):
        g[j] = g[plist[j]] @ local_mat[:, j]
    return torch.stack(g, dim=1)   # (N, J, 3, 3)


def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if 'mocap_framerate' not in entry_data:
        return None
    fps = entry_data['mocap_framerate']
    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    return {
        "pose_aa": pose_aa,
        "trans": root_trans,
        "betas": entry_data['betas'],
        "fps": fps,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", type=str, default="data/AMASS/AMASS_Complete")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smpl_parser_n = SMPL_Parser(model_path=_SMPL_DATA_PATH, gender="neutral")
    smpl_parser_n.to(device)

    shape_new, leg_scale, arm_scale = joblib.load(osp.join(_K1_DATA_DIR, "shape_optimized_v1.pkl"))
    shape_new = shape_new.to(device)
    leg_scale = leg_scale.to(device)
    arm_scale = arm_scale.to(device)

    k1_fk = Humanoid_Batch(mjcf_file=K1_MJCF, extend_hand=False, extend_head=False, device=device)
    k1_rot_axis = K1_ROTATION_AXIS.to(device)

    amass_root = args.amass_root
    all_npz = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {
        "0-" + "_".join(p.split("/")[split_len:]).replace(".npz", ""): p
        for p in all_npz
    }

    if len(key_name_to_pkls) == 0:
        raise ValueError(f"No motion files found in {amass_root}")

    data_dump = {}
    pbar = tqdm(key_name_to_pkls.keys())
    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if amass_data is None:
            continue

        skip = max(1, int(amass_data['fps'] // 30))
        trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(device)
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(
            np.concatenate((amass_data['pose_aa'][::skip, :66], np.zeros((N, 6))), axis=-1)
        ).float().to(device)

        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, torch.zeros((1, 10)).to(device), trans)
        root_trans_offset = trans + (joints[:, 0] - trans)

        gt_root_rot = torch.from_numpy(
            (sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat(SMPL_ROOT_ALIGN_QUAT).inv()).as_rotvec()
        ).float().to(device)

        with torch.no_grad():
            _, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            smpl_g_rot = _smpl_global_rot(gt_root_rot, pose_aa_walk, smpl_parser_n.parents)
        smpl_l_wrist_rot = smpl_g_rot[:, SMPL_L_WRIST_IDX]   # (N, 3, 3) in sim world frame
        smpl_r_wrist_rot = smpl_g_rot[:, SMPL_R_WRIST_IDX]   # (N, 3, 3)
        root_pos = joints[:, 0:1]
        # Leg IK targets (5 joints, scaled to K1 proportions)
        _s_leg = torch.ones(5, device=device) * leg_scale
        joints_leg_scaled = (joints[:, smpl_joint_pick_idx[:5]] - root_pos) * _s_leg[None, :, None] + root_pos

        # ── Analytical shoulder roll ──────────────────────────────────────────
        # K1 arm anatomy: every child body is offset in +Y from its parent, so
        # shoulder_pitch and elbow_pitch (Y-axis) have zero positional effect on
        # descendants.  The ONLY DOF that moves the hand in space is
        # Left/Right_Shoulder_Roll (X-axis), which swings the forearm in the Y-Z
        # plane.  There is no DOF for forward (X) arm extension — boxing punches
        # are geometrically unreachable.  We therefore set shoulder roll
        # analytically from the SMPL upper-arm elevation angle and let the
        # gradient optimizer handle everything else (legs + wrist orientation).
        with torch.no_grad():
            trunk_R = tRot.axis_angle_to_matrix(gt_root_rot)  # (N,3,3) local→world
            # Use pick indices for elbow (smpl_joint_pick_idx[5]=L_Elbow, [7]=R_Elbow)
            l_upper_world = joints[:, smpl_joint_pick_idx[5]] - joints[:, SMPL_L_SHOULDER_IDX]
            r_upper_world = joints[:, smpl_joint_pick_idx[7]] - joints[:, SMPL_R_SHOULDER_IDX]
            # Rotate to trunk frame
            l_upper_trunk = torch.einsum('nij,nj->ni', trunk_R.mT, l_upper_world)
            r_upper_trunk = torch.einsum('nij,nj->ni', trunk_R.mT, r_upper_world)
            # Elevation = atan2(Z, sqrt(X²+Y²))
            l_horiz = l_upper_trunk[:, :2].norm(dim=-1).clamp(min=1e-6)
            r_horiz = r_upper_trunk[:, :2].norm(dim=-1).clamp(min=1e-6)
            # Left roll: positive = arm UP (+Z).  Right roll: positive = arm DOWN,
            # so negate to keep elevation semantics consistent.
            target_l_roll =  torch.atan2(l_upper_trunk[:, 2], l_horiz)
            target_r_roll = -torch.atan2(r_upper_trunk[:, 2], r_horiz)
            target_l_roll = target_l_roll.clamp(k1_fk.joints_range[3, 0], k1_fk.joints_range[3, 1])
            target_r_roll = target_r_roll.clamp(k1_fk.joints_range[7, 0], k1_fk.joints_range[7, 1])
            print(f"  shoulder_roll L: min={target_l_roll.min():.2f} max={target_l_roll.max():.2f} mean={target_l_roll.mean():.2f}"
                  f"  R: min={target_r_roll.min():.2f} max={target_r_roll.max():.2f} mean={target_r_roll.mean():.2f}")

        init = torch.zeros((1, N, 22, 1), device=device)
        init[0, :, 3, 0] = target_l_roll   # Left_Shoulder_Roll
        init[0, :, 7, 0] = target_r_roll   # Right_Shoulder_Roll
        dof_pos_new = Variable(init.clone(), requires_grad=True)
        optimizer = torch.optim.Adadelta([dof_pos_new], lr=100)

        I3 = torch.eye(3, device=device).unsqueeze(0).expand(N, -1, -1)

        for iteration in range(2000):
            pose_aa_k1_new = torch.cat(
                [gt_root_rot[None, :, None], k1_rot_axis * dof_pos_new], dim=2
            )
            fk_return = k1_fk.fk_batch(pose_aa_k1_new, root_trans_offset[None,])

            leg_loss = (fk_return.global_translation[:, :, k1_joint_pick_idx[:5]] - joints_leg_scaled).norm(dim=-1).mean()

            k1_l_hand_rot = fk_return.global_rotation_mat[0, :, K1_L_HAND_BODY_IDX]
            k1_r_hand_rot = fk_return.global_rotation_mat[0, :, K1_R_HAND_BODY_IDX]
            rot_loss = (
                (k1_l_hand_rot.mT @ smpl_l_wrist_rot - I3).pow(2).sum((-2, -1)).mean() +
                (k1_r_hand_rot.mT @ smpl_r_wrist_rot - I3).pow(2).sum((-2, -1)).mean()
            )
            loss = leg_loss + ROT_WEIGHT * rot_loss
            pbar.set_description_str(
                f"{iteration} legs={leg_loss.item()*1000:.1f}mm rot={rot_loss.item():.3f}"
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dof_pos_new.data.clamp_(k1_fk.joints_range[:, 0, None], k1_fk.joints_range[:, 1, None])
            # Restore analytically computed shoulder rolls after each step
            dof_pos_new.data[:, :, 3, 0] = target_l_roll
            dof_pos_new.data[:, :, 7, 0] = target_r_roll

        dof_pos_new.data.clamp_(k1_fk.joints_range[:, 0, None], k1_fk.joints_range[:, 1, None])
        dof_pos_new.data[:, :, 3, 0] = target_l_roll
        dof_pos_new.data[:, :, 7, 0] = target_r_roll
        pose_aa_k1_new = torch.cat(
            [gt_root_rot[None, :, None], k1_rot_axis * dof_pos_new], dim=2
        )
        fk_return = k1_fk.fk_batch(pose_aa_k1_new, root_trans_offset[None,])

        root_trans_offset_dump = root_trans_offset.clone()
        root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08

        data_dump[data_key] = {
            "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
            "pose_aa": pose_aa_k1_new.squeeze().cpu().detach().numpy(),
            "dof": dof_pos_new.squeeze().detach().cpu().numpy(),
            "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
            "fps": 30,
        }

    for dof_i, dof_name in [(3, "L_shoulder_roll"), (7, "R_shoulder_roll"),
                            (4, "L_elbow_pitch"), (5, "L_elbow_yaw"), (8, "R_elbow_pitch"), (9, "R_elbow_yaw")]:
        vals = dof_pos_new[0, :, dof_i, 0].detach().cpu().numpy()
        print(f"  {dof_name:20s} min={vals.min():.3f}  max={vals.max():.3f}  mean={vals.mean():.3f}")


    os.makedirs(_K1_DATA_DIR, exist_ok=True)
    out_path = osp.join(_K1_DATA_DIR, "amass_all.pkl")
    joblib.dump(data_dump, out_path)
    print(f"Retargeted {len(data_dump)} clips -> {out_path}")
