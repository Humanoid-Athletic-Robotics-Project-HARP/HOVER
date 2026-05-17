"""
Compute per-limb scale factors that map SMPL joint positions to K1 proportions,
matching the 3-target-per-arm strategy used in grad_fit_k1.py.
grad_fit_k1.py matches these chains:
  K1 body          SMPL joint
  -----------      ----------
  Left_Arm_1    -> L_Shoulder   (shoulder body)
  Left_Arm_3    -> L_Elbow      (elbow body — just past Left_Elbow_Pitch)
  left_hand_link-> L_Hand       (end-effector)
arm_scale is the ratio of K1's full arm chain length to SMPL's:
  K1:   |Arm1->Arm3| + |Arm3->hand|
  SMPL: |L_Shoulder->L_Elbow| + |L_Elbow->L_Hand|
This single scale maps all three SMPL arm targets proportionally, giving
T-pose geometric errors of ~20 mm (shoulder), ~35 mm (elbow), ~44 mm (hand)
— all reachable by the optimizer, vs the previous 143 mm for hand->L_Elbow.
leg_scale is the average of knee- and ankle-based height ratios.
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

_HERE = osp.dirname(osp.abspath(__file__))
_HOVER_ROOT = osp.normpath(osp.join(_HERE, "..", ".."))
_H2H_ROOT = osp.join(_HOVER_ROOT, "third_party", "human2humanoid")
sys.path.insert(0, osp.join(_H2H_ROOT, "phc"))

from phc.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES

K1_MJCF = osp.join(_HOVER_ROOT, "neural_wbc", "data", "data", "motion_lib", "k1.xml")
_SMPL_DATA_PATH = osp.join(_H2H_ROOT, "data", "smpl")
_K1_OUT_DIR = osp.join(_H2H_ROOT, "data", "k1")

SMPL_ROOT_ALIGN_QUAT = [0.5, 0.5, 0.5, 0.5]


def _body_pos(model, data, name):
    return data.xpos[model.body(name).id].copy()


# ---------------------------------------------------------------------------
# K1 geometry at zero pose
# ---------------------------------------------------------------------------
m = mujoco.MjModel.from_xml_path(K1_MJCF)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

trunk  = _body_pos(m, d, "Trunk")
shank  = _body_pos(m, d, "Left_Shank")       # knee proxy
foot   = _body_pos(m, d, "left_foot_link")    # ankle proxy
arm1  = _body_pos(m, d, "Left_Arm_1")       # shoulder body
elbow = _body_pos(m, d, "left_hand_link")   # Left_Elbow_Pitch joint position = true elbow
tip   = elbow + np.array([0, 0.228, 0])     # hand tip (left_hand_tip site offset in K1_serial.xml)

k1_knee_z  = shank[2] - trunk[2]   # negative (below trunk)
k1_ankle_z = foot[2]  - trunk[2]   # negative
k1_arm_len = float(np.linalg.norm(elbow - arm1) + np.linalg.norm(tip - elbow))

# ---------------------------------------------------------------------------
# SMPL geometry at zero betas, T-pose aligned to robot frame
# ---------------------------------------------------------------------------
smpl = SMPL_Parser(model_path=_SMPL_DATA_PATH, gender="neutral")

# Apply root alignment so SMPL is in the same frame as the robot FK output.
pose_tpose = np.zeros((1, 72))
pose_tpose[:, :3] = sRot.from_quat(SMPL_ROOT_ALIGN_QUAT).as_rotvec()
_, joints = smpl.get_joints_verts(
    torch.from_numpy(pose_tpose).float(),
    torch.zeros(1, 10),
    torch.zeros(1, 3),
)
j = joints[0]  # (24, 3) in robot world frame

def _ji(name):
    return SMPL_BONE_ORDER_NAMES.index(name)

pelvis    = j[_ji("Pelvis")]
smpl_knee_z  = float(j[_ji("L_Knee")][2]  - pelvis[2])
smpl_ankle_z = float(j[_ji("L_Ankle")][2] - pelvis[2])
smpl_arm_len = float(
    (j[_ji("L_Elbow")] - j[_ji("L_Shoulder")]).norm() +
    (j[_ji("L_Hand")]  - j[_ji("L_Elbow")]).norm()
)

# ---------------------------------------------------------------------------
# Analytical scales
# ---------------------------------------------------------------------------
leg_scale = float(
    0.5 * (k1_knee_z / smpl_knee_z) + 0.5 * (k1_ankle_z / smpl_ankle_z)
)
arm_scale = float(k1_arm_len / smpl_arm_len)

print(f"K1   knee z  : {k1_knee_z:.4f} m   ankle z : {k1_ankle_z:.4f} m")
print(f"SMPL knee z  : {smpl_knee_z:.4f} m   ankle z : {smpl_ankle_z:.4f} m")
print(f"K1   arm chain: {k1_arm_len:.4f} m   SMPL arm chain: {smpl_arm_len:.4f} m")
print(f"leg_scale = {leg_scale:.4f}   arm_scale = {arm_scale:.4f}")

# ---------------------------------------------------------------------------
# T-pose error verification for all arm targets (diagnostic)
# ---------------------------------------------------------------------------
root_pos = j[_ji("Pelvis")].numpy()

def _tpose_err_body(k1_body, smpl_joint):
    k1_v = _body_pos(m, d, k1_body) - trunk
    sm_v = (j[_ji(smpl_joint)].numpy() - root_pos) * arm_scale
    return np.linalg.norm(k1_v - sm_v) * 1000

def _tpose_err_tip(smpl_joint):
    k1_v = tip - trunk
    sm_v = (j[_ji(smpl_joint)].numpy() - root_pos) * arm_scale
    return np.linalg.norm(k1_v - sm_v) * 1000

print("\nT-pose geometric errors with arm_scale:")
print(f"  {'Left_Arm_1':20s} -> {'L_Shoulder':12s}  {_tpose_err_body('Left_Arm_1', 'L_Shoulder'):.1f} mm")
print(f"  {'left_hand_link':20s} -> {'L_Elbow':12s}  {_tpose_err_body('left_hand_link', 'L_Elbow'):.1f} mm")
print(f"  {'left_hand_tip':20s} -> {'L_Hand':12s}  {_tpose_err_tip('L_Hand'):.1f} mm")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
shape_new   = torch.zeros(1, 10)
leg_scale_t = torch.tensor([leg_scale])
arm_scale_t = torch.tensor([arm_scale])

os.makedirs(_K1_OUT_DIR, exist_ok=True)
out_path = osp.join(_K1_OUT_DIR, "shape_optimized_v1.pkl")
joblib.dump((shape_new, leg_scale_t, arm_scale_t), out_path)
print(f"\nSaved to {out_path}")
