import os
import sys
import os.path as osp
sys.path.append(os.getcwd())

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from torch.autograd import Variable

from phc.utils import torch_utils  # noqa: F401 — registers torch extensions
from phc.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch

_HERE = osp.dirname(osp.abspath(__file__))
_HOVER_ROOT = osp.normpath(osp.join(_HERE, "..", ".."))
_H2H_ROOT = osp.join(_HOVER_ROOT, "third_party", "human2humanoid")
K1_MJCF = osp.join(_HOVER_ROOT, "neural_wbc", "data", "data", "motion_lib", "k1.xml")
_SMPL_DATA_PATH = osp.join(_H2H_ROOT, "data", "smpl")
_K1_OUT_DIR = osp.join(_H2H_ROOT, "data", "k1")

# Joint rotation axes in MJCF depth-first order (matches Humanoid_Batch.from_mjcf).
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

# Body names in MJCF depth-first traversal order.
K1_BODY_NAMES = [
    'Trunk',
    'Head_1', 'Head_2',
    'Left_Arm_1', 'Left_Arm_2', 'Left_Arm_3', 'left_hand_link',
    'Right_Arm_1', 'Right_Arm_2', 'Right_Arm_3', 'right_hand_link',
    'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw', 'Left_Shank', 'Left_Ankle_Cross', 'left_foot_link',
    'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw', 'Right_Shank', 'Right_Ankle_Cross', 'right_foot_link',
]

# Shape fitting uses only pelvis+legs. Arms are excluded because K1's zero-pose
# arms don't match SMPL T-pose, which would bias the limb-length fit.
k1_joint_pick   = ['Trunk', 'Left_Hip_Pitch', 'Left_Shank', 'left_foot_link',
                   'Right_Hip_Pitch', 'Right_Shank', 'right_foot_link']
smpl_joint_pick = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle',
                   'R_Hip', 'R_Knee', 'R_Ankle']
k1_joint_pick_idx   = [K1_BODY_NAMES.index(j) for j in k1_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k1_fk = Humanoid_Batch(mjcf_file=K1_MJCF, extend_hand=False, extend_head=False, device=device)

# K1 at zero pose (standing, all DOFs = 0): pose is (1, 23, 3) — root + 22 joints.
dof_pos = torch.zeros((1, 22))
pose_aa_k1 = torch.cat([torch.zeros((1, 1, 3)), K1_ROTATION_AXIS * dof_pos[..., None]], dim=1)

# SMPL standing pose with root aligned to K1 convention.
pose_aa_stand = np.zeros((1, 72))
pose_aa_stand[:, :3] = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

smpl_parser_n = SMPL_Parser(model_path=_SMPL_DATA_PATH, gender="neutral")
trans = torch.zeros([1, 3])

verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, torch.zeros([1, 10]), trans)
root_trans_offset = trans + (joints[:, 0] - trans)

fk_return = k1_fk.fk_batch(pose_aa_k1[None,].to(device), root_trans_offset[None, 0:1].to(device))

shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
scale     = Variable(torch.ones([1]).to(device),      requires_grad=True)
optimizer = torch.optim.Adam([shape_new, scale], lr=0.1)

for i in range(1000):
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new.cpu(), trans[0:1])
    joints = joints.to(device)
    root_pos = joints[:, 0]
    joints = (joints - joints[:, 0]) * scale + root_pos
    loss = (fk_return.global_translation[:, :, k1_joint_pick_idx] - joints[:, smpl_joint_pick_idx]).norm(dim=-1).mean()
    if i % 100 == 0:
        print(i, loss.item() * 1000)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

os.makedirs(_K1_OUT_DIR, exist_ok=True)
out_path = osp.join(_K1_OUT_DIR, "shape_optimized_v1.pkl")
joblib.dump((shape_new.detach().cpu(), scale.detach().cpu()), out_path)
print(f"shape fitted and saved to {out_path}")
