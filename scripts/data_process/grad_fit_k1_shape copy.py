"""
Compute per-limb scale factors that map SMPL joint positions to K1 proportions.

K1 is ~58% of human height for legs and ~39% for arms — a single global scale
can't serve both, so we compute leg_scale and arm_scale separately from the
robot's actual MJCF geometry vs default SMPL joint positions.

No gradient-based optimization is used; the scales are derived analytically.
"""
import os
import sys
import os.path as osp
sys.path.append(os.getcwd())

import joblib
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot

from phc.utils import torch_utils  # noqa: F401 — registers torch extensions
from phc.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES

_HERE = osp.dirname(osp.abspath(__file__))
_HOVER_ROOT = osp.normpath(osp.join(_HERE, "..", ".."))
_H2H_ROOT = osp.join(_HOVER_ROOT, "third_party", "human2humanoid")
K1_MJCF = osp.join(_HOVER_ROOT, "neural_wbc", "data", "data", "motion_lib", "k1.xml")
_SMPL_DATA_PATH = osp.join(_H2H_ROOT, "data", "smpl")
_K1_OUT_DIR = osp.join(_H2H_ROOT, "data", "k1")


def _k1_body_pos(model, data, name):
    return data.xpos[model.body(name).id].copy()


# --- K1 geometry at zero pose ---
m = mujoco.MjModel.from_xml_path(K1_MJCF)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

trunk  = _k1_body_pos(m, d, "Trunk")
shank  = _k1_body_pos(m, d, "Left_Shank")      # knee
foot   = _k1_body_pos(m, d, "left_foot_link")   # ankle
arm1   = _k1_body_pos(m, d, "Left_Arm_1")       # shoulder complex
arm3   = _k1_body_pos(m, d, "Left_Arm_3")       # upper arm
hand   = _k1_body_pos(m, d, "left_hand_link")   # hand

k1_knee_z   = shank[2] - trunk[2]   # negative (below trunk)
k1_ankle_z  = foot[2]  - trunk[2]   # negative (below trunk)
k1_arm_len  = float(np.linalg.norm(arm3 - arm1) + np.linalg.norm(hand - arm3))

# --- SMPL geometry at zero betas, standing ---
smpl = SMPL_Parser(model_path=_SMPL_DATA_PATH, gender="neutral")
pose_aa_stand = np.zeros((1, 72))
pose_aa_stand[:, :3] = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
_, joints = smpl.get_joints_verts(
    torch.from_numpy(pose_aa_stand), torch.zeros(1, 10), torch.zeros(1, 3)
)
j = joints[0]

def _ji(name): return SMPL_BONE_ORDER_NAMES.index(name)

pelvis   = j[_ji("Pelvis")]
smpl_knee_z  = float(j[_ji("L_Knee")][2]  - pelvis[2])   # negative
smpl_ankle_z = float(j[_ji("L_Ankle")][2] - pelvis[2])   # negative
smpl_arm_len = float(
    (j[_ji("L_Elbow")] - j[_ji("L_Shoulder")]).norm() +
    (j[_ji("L_Hand")]  - j[_ji("L_Elbow")]).norm()
)

# --- Analytical scales ---
# Use average of knee- and ankle-based ratios for leg scale stability.
leg_scale = float(
    0.5 * (k1_knee_z / smpl_knee_z) + 0.5 * (k1_ankle_z / smpl_ankle_z)
)
arm_scale = float(k1_arm_len / smpl_arm_len)

print(f"K1  knee z-offset from trunk : {k1_knee_z:.4f}m  |  ankle: {k1_ankle_z:.4f}m")
print(f"SMPL knee z-offset from pelvis: {smpl_knee_z:.4f}m  |  ankle: {smpl_ankle_z:.4f}m")
print(f"K1  arm chain : {k1_arm_len:.4f}m  |  SMPL arm chain: {smpl_arm_len:.4f}m")
print(f"leg_scale = {leg_scale:.4f}   arm_scale = {arm_scale:.4f}")

# shape_new is zeros — neutral human shape; per-limb scale handles size difference.
shape_new = torch.zeros(1, 10)
leg_scale_t = torch.tensor([leg_scale])
arm_scale_t = torch.tensor([arm_scale])

os.makedirs(_K1_OUT_DIR, exist_ok=True)
out_path = osp.join(_K1_OUT_DIR, "shape_optimized_v1.pkl")
joblib.dump((shape_new, leg_scale_t, arm_scale_t), out_path)
print(f"Scales saved to {out_path}")
