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

k1_joint_pick  = ['Trunk', 'Left_Shank', 'left_foot_link', 'Right_Shank', 'right_foot_link',
                  'Left_Arm_2', 'Left_Arm_3', 'left_hand_link', 'Right_Arm_2', 'Right_Arm_3', 'right_hand_link']
smpl_joint_pick = ['Pelvis', 'L_Knee', 'L_Ankle', 'R_Knee', 'R_Ankle',
                   'L_Shoulder', 'L_Elbow', 'L_Hand', 'R_Shoulder', 'R_Elbow', 'R_Hand']
k1_joint_pick_idx   = [K1_BODY_NAMES.index(j) for j in k1_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

# SMPL root alignment used during retargeting (same as H1).
SMPL_ROOT_ALIGN_QUAT = [0.5, 0.5, 0.5, 0.5]


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

    shape_new, scale = joblib.load(osp.join(_K1_DATA_DIR, "shape_optimized_v1.pkl"))
    shape_new = shape_new.to(device)

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

        dof_pos_new = Variable(torch.zeros((1, N, 22, 1)).to(device), requires_grad=True)
        optimizer = torch.optim.Adadelta([dof_pos_new], lr=100)

        for iteration in range(500):
            verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            pose_aa_k1_new = torch.cat(
                [gt_root_rot[None, :, None], k1_rot_axis * dof_pos_new], dim=2
            ).to(device)
            fk_return = k1_fk.fk_batch(pose_aa_k1_new, root_trans_offset[None,])

            diff = fk_return.global_translation[:, :, k1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]
            loss = diff.norm(dim=-1).mean()
            pbar.set_description_str(f"{iteration} {loss.item() * 1000:.2f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dof_pos_new.data.clamp_(k1_fk.joints_range[:, 0, None], k1_fk.joints_range[:, 1, None])

        dof_pos_new.data.clamp_(k1_fk.joints_range[:, 0, None], k1_fk.joints_range[:, 1, None])
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

    os.makedirs(_K1_DATA_DIR, exist_ok=True)
    out_path = osp.join(_K1_DATA_DIR, "amass_all.pkl")
    joblib.dump(data_dump, out_path)
    print(f"Retargeted {len(data_dump)} clips -> {out_path}")
