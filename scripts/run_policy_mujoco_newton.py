"""Run a trained K1 teacher policy inside MuJoCo Newton physics (no IsaacLab).

Loads the actor MLP directly from a checkpoint, builds teacher observations
from MuJoCo state, applies PD control, and renders to mp4.  No IsaacLab
process needed — safe to run alongside a live training job.

Usage:
    MUJOCO_GL=egl python3 scripts/run_policy_mujoco_newton.py \
        --checkpoint logs/teacher/26_05_22_07-10-41/model_500.pt \
        --motion neural_wbc/data/data/motions/cmu_simple.pkl \
        --max_steps 500 \
        --video /tmp/k1_policy_newton.mp4
"""
import argparse, os, sys
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--scene", type=str,
    default="neural_wbc/data/data/mujoco/models/scene_k1.xml")
parser.add_argument("--motion", type=str,
    default="neural_wbc/data/data/motions/cmu_simple.pkl")
parser.add_argument("--skeleton", type=str,
    default="neural_wbc/data/data/motion_lib/k1.xml")
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--video", type=str, default="/tmp/k1_policy_newton.mp4")
parser.add_argument("--out", type=str, default="/tmp/k1_policy_newton.npz")
args = parser.parse_args()

import mujoco
from neural_wbc.core.body_state import BodyState
from neural_wbc.core.observations import compute_teacher_observations
from neural_wbc.core.reference_motion import ReferenceMotionManager, ReferenceMotionManagerCfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── K1 config constants (from NeuralWBCEnvCfgK1) ─────────────────────────────
BODY_NAMES = [
    "Trunk", "Head_1", "Head_2",
    "Left_Arm_1", "Left_Arm_2", "Left_Arm_3", "left_hand_link",
    "Right_Arm_1", "Right_Arm_2", "Right_Arm_3", "right_hand_link",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Shank",
    "Left_Ankle_Cross", "left_foot_link",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Shank",
    "Right_Ankle_Cross", "right_foot_link",
]
JOINT_NAMES = [
    "AAHead_yaw", "Head_pitch",
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
]
# virtual tip bodies extended from physical parents
EXTEND_PARENT_NAMES = ["left_hand_link", "right_hand_link", "Head_2"]
EXTEND_BODY_POS = torch.tensor([[0.0, 0.228, 0.0], [0.0, -0.228, 0.0], [0.0, 0.0, 0.1]], device=device)
# all 23 physical bodies are tracked
TRACKED_BODY_IDS = list(range(23))
# default joint positions (action=0 → these targets)
DEFAULT_Q = {
    "AAHead_yaw": 0.0, "Head_pitch": 0.0,
    "ALeft_Shoulder_Pitch": 0.0, "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Pitch": 0.0, "Left_Elbow_Yaw": 0.0,
    "ARight_Shoulder_Pitch": 0.0, "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Pitch": 0.0, "Right_Elbow_Yaw": 0.0,
    "Left_Hip_Pitch": -0.25, "Left_Hip_Roll": 0.0, "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.5, "Left_Ankle_Pitch": -0.25, "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.25, "Right_Hip_Roll": 0.0, "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.5, "Right_Ankle_Pitch": -0.25, "Right_Ankle_Roll": 0.0,
}
KP = {  # stiffness
    "AAHead_yaw": 40, "Head_pitch": 40,
    "ALeft_Shoulder_Pitch": 40, "Left_Shoulder_Roll": 40, "Left_Elbow_Pitch": 40, "Left_Elbow_Yaw": 40,
    "ARight_Shoulder_Pitch": 40, "Right_Shoulder_Roll": 40, "Right_Elbow_Pitch": 40, "Right_Elbow_Yaw": 40,
    "Left_Hip_Pitch": 200, "Left_Hip_Roll": 150, "Left_Hip_Yaw": 150,
    "Left_Knee_Pitch": 200, "Left_Ankle_Pitch": 20, "Left_Ankle_Roll": 20,
    "Right_Hip_Pitch": 200, "Right_Hip_Roll": 150, "Right_Hip_Yaw": 150,
    "Right_Knee_Pitch": 200, "Right_Ankle_Pitch": 20, "Right_Ankle_Roll": 20,
}
KD = {  # damping
    "AAHead_yaw": 5, "Head_pitch": 5,
    "ALeft_Shoulder_Pitch": 10, "Left_Shoulder_Roll": 10, "Left_Elbow_Pitch": 10, "Left_Elbow_Yaw": 10,
    "ARight_Shoulder_Pitch": 10, "Right_Shoulder_Roll": 10, "Right_Elbow_Pitch": 10, "Right_Elbow_Yaw": 10,
    "Left_Hip_Pitch": 5, "Left_Hip_Roll": 5, "Left_Hip_Yaw": 5,
    "Left_Knee_Pitch": 5, "Left_Ankle_Pitch": 4, "Left_Ankle_Roll": 4,
    "Right_Hip_Pitch": 5, "Right_Hip_Roll": 5, "Right_Hip_Yaw": 5,
    "Right_Knee_Pitch": 5, "Right_Ankle_Pitch": 4, "Right_Ankle_Roll": 4,
}
EFFORT_LIMIT = {
    "AAHead_yaw": 6, "Head_pitch": 6,
    "ALeft_Shoulder_Pitch": 14, "Left_Shoulder_Roll": 14, "Left_Elbow_Pitch": 14, "Left_Elbow_Yaw": 14,
    "ARight_Shoulder_Pitch": 14, "Right_Shoulder_Roll": 14, "Right_Elbow_Pitch": 14, "Right_Elbow_Yaw": 14,
    "Left_Hip_Pitch": 30, "Left_Hip_Roll": 35, "Left_Hip_Yaw": 20,
    "Left_Knee_Pitch": 40, "Left_Ankle_Pitch": 20, "Left_Ankle_Roll": 20,
    "Right_Hip_Pitch": 30, "Right_Hip_Roll": 35, "Right_Hip_Yaw": 20,
    "Right_Knee_Pitch": 40, "Right_Ankle_Pitch": 20, "Right_Ankle_Roll": 20,
}
ACTION_SCALE = 0.25
DECIMATION    = 4   # policy runs at 50 Hz, physics at 1 kHz

# build ordered tensors
kp_vec     = torch.tensor([KP[j]           for j in JOINT_NAMES], device=device)
kd_vec     = torch.tensor([KD[j]           for j in JOINT_NAMES], device=device)
effort_vec = torch.tensor([EFFORT_LIMIT[j] for j in JOINT_NAMES], device=device)
default_q  = torch.tensor([DEFAULT_Q[j]    for j in JOINT_NAMES], device=device)
extend_parent_ids = [BODY_NAMES.index(n) for n in EXTEND_PARENT_NAMES]

# ── Load actor MLP ────────────────────────────────────────────────────────────
ckpt = torch.load(args.checkpoint, map_location=device)
sd   = ckpt["model_state_dict"]
actor = torch.nn.Sequential(
    torch.nn.Linear(961, 512), torch.nn.ELU(),
    torch.nn.Linear(512, 256), torch.nn.ELU(),
    torch.nn.Linear(256, 128), torch.nn.ELU(),
    torch.nn.Linear(128, 22),
)
actor.load_state_dict({
    "0.weight": sd["actor.0.weight"], "0.bias": sd["actor.0.bias"],
    "2.weight": sd["actor.2.weight"], "2.bias": sd["actor.2.bias"],
    "4.weight": sd["actor.4.weight"], "4.bias": sd["actor.4.bias"],
    "6.weight": sd["actor.6.weight"], "6.bias": sd["actor.6.bias"],
})
actor.to(device).eval()
print(f"[INFO] Loaded actor from {args.checkpoint}  (iter={ckpt.get('iter', '?')})")

# ── Load MuJoCo model ─────────────────────────────────────────────────────────
scene_path = os.path.abspath(args.scene)
model = mujoco.MjModel.from_xml_path(scene_path)
data  = mujoco.MjData(model)
solver_names = {0: "PGS", 1: "CG", 2: "Newton"}
print(f"[INFO] MuJoCo solver: {solver_names.get(model.opt.solver, model.opt.solver)}")

opt = mujoco.MjvOption()
opt.geomgroup[3] = 1  # show collision geometry (k1.xml uses group 3)

# ── Reference motion manager ──────────────────────────────────────────────────
rmm_cfg = ReferenceMotionManagerCfg()
rmm_cfg.motion_path   = os.path.abspath(args.motion)
rmm_cfg.skeleton_path = os.path.abspath(args.skeleton)
rmm_cfg.extend_hand   = False
rmm_cfg.extend_head   = False
rmm = ReferenceMotionManager(
    cfg=rmm_cfg, device=device, num_envs=1,
    random_sample=False,
    extend_hand=False, extend_head=False,
    dt=DECIMATION * model.opt.timestep,
)

# ── Set initial state from RSI frame 0 ───────────────────────────────────────
episode_buf = torch.zeros(1, device=device, dtype=torch.long)
start_pos   = torch.zeros(1, 3, device=device)
rmm.reset_motion_start_times(env_ids=torch.tensor([0], device=device), sample=False)
ref0 = rmm.get_state_from_motion_lib_cache(episode_buf, offset=start_pos)
mujoco.mj_resetData(model, data)
root_pos0 = ref0.root_pos[0].cpu().numpy()
root_pos0[2] = max(root_pos0[2], 0.61)   # ensure above ground
data.qpos[:3]  = root_pos0
data.qpos[3:7] = ref0.root_rot[0].cpu().numpy()   # wxyz
data.qpos[7:]  = ref0.joint_pos[0].cpu().numpy()
data.qvel[:]   = 0.0
mujoco.mj_forward(model, data)
print(f"[INFO] Init root_z={data.qpos[2]:.4f}  ncon={data.ncon}")

# ── Renderer ──────────────────────────────────────────────────────────────────
renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance  = 3.5
cam.azimuth   = 90.0
cam.elevation = -15.0
cam.lookat[0] = root_pos0[0]
cam.lookat[1] = root_pos0[1]
cam.lookat[2] = 0.5

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_body_state_from_mujoco() -> BodyState:
    """Read all 23 K1 body states from MuJoCo data into a BodyState."""
    # body_pos, body_rot from xpos/xquat (global frame, 0=world → skip)
    body_pos = torch.tensor(data.xpos[1:].copy(),  dtype=torch.float32, device=device).unsqueeze(0)  # [1,23,3]
    body_rot = torch.tensor(data.xquat[1:].copy(), dtype=torch.float32, device=device).unsqueeze(0)  # [1,23,4] wxyz

    # velocities: mj_objectVelocity returns [ang(3), lin(3)] in world frame
    lin_vel = torch.zeros(1, 23, 3, device=device)
    ang_vel = torch.zeros(1, 23, 3, device=device)
    buf = np.zeros(6)
    for i, _ in enumerate(BODY_NAMES):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, i+1, buf, 0)
        ang_vel[0, i] = torch.tensor(buf[:3], dtype=torch.float32)
        lin_vel[0, i] = torch.tensor(buf[3:], dtype=torch.float32)

    joint_pos = torch.tensor(data.qpos[7:].copy(), dtype=torch.float32, device=device).unsqueeze(0)
    joint_vel = torch.tensor(data.qvel[6:].copy(), dtype=torch.float32, device=device).unsqueeze(0)

    bs = BodyState(
        body_pos=body_pos, body_rot=body_rot,
        body_lin_vel=lin_vel, body_ang_vel=ang_vel,
        joint_pos=joint_pos, joint_vel=joint_vel,
        root_id=0,
    )
    bs.extend_body_states(
        extend_body_pos=EXTEND_BODY_POS.unsqueeze(0),
        extend_body_parent_ids=extend_parent_ids,
    )
    return bs

# ── Main loop ─────────────────────────────────────────────────────────────────
RENDER_EVERY = DECIMATION  # one frame per policy step
frames, joint_positions, root_positions, root_quaternions = [], [], [], []
last_actions = torch.zeros(1, 22, device=device)

for step in range(args.max_steps):
    # -- reference motion state
    ref_state = rmm.get_state_from_motion_lib_cache(episode_buf, offset=start_pos)

    # -- teacher observations
    bs = get_body_state_from_mujoco()
    with torch.no_grad():
        obs, _ = compute_teacher_observations(
            body_state=bs,
            ref_motion_state=ref_state,
            tracked_body_ids=TRACKED_BODY_IDS,
            last_actions=last_actions,
            ref_episodic_offset=None,
        )
        raw_actions = actor(obs)            # [1, 22]
    last_actions = raw_actions.clone()

    # -- PD control: policy outputs delta from default; convert to torques
    pos_target = raw_actions[0] * ACTION_SCALE + default_q
    q   = torch.tensor(data.qpos[7:].copy(), dtype=torch.float32, device=device)
    qd  = torch.tensor(data.qvel[6:].copy(), dtype=torch.float32, device=device)
    torques = kp_vec * (pos_target - q) - kd_vec * qd
    torques = torch.clamp(torques, -effort_vec, effort_vec)
    torques_np = torques.cpu().numpy()

    # -- physics sub-steps at MuJoCo dt
    for _ in range(DECIMATION):
        data.ctrl[:] = torques_np
        mujoco.mj_step(model, data)

    # -- advance reference motion by one policy step
    episode_buf += 1

    # -- record
    joint_positions.append(data.qpos[7:].copy())
    root_positions.append(data.qpos[:3].copy())
    root_quaternions.append(data.qpos[3:7].copy())

    # -- render
    renderer.update_scene(data, camera=cam, scene_option=opt)
    frames.append(renderer.render().copy())

    if step % 50 == 0:
        print(f"  step {step:4d}  root_z={data.qpos[2]:.4f}  ncon={data.ncon}")

renderer.close()

# ── Save data ─────────────────────────────────────────────────────────────────
np.savez(args.out,
         joint_pos=np.array(joint_positions),
         root_pos=np.array(root_positions),
         root_quat=np.array(root_quaternions))
print(f"\nSaved {len(joint_positions)} steps → {args.out}")
print(f"  root_z: {np.array(root_positions)[:,2].min():.4f} → {np.array(root_positions)[:,2].max():.4f}")

# ── Save video ────────────────────────────────────────────────────────────────
fps = int(1.0 / (model.opt.timestep * DECIMATION))
print(f"Saving {len(frames)} frames at {fps} fps → {args.video}")
try:
    import imageio
    writer = imageio.get_writer(args.video, fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"Saved: {args.video}")
except Exception as e:
    gif = args.video.replace(".mp4", ".gif")
    imageio.mimsave(gif, frames, fps=fps)
    print(f"Saved gif: {gif}  (mp4 failed: {e})")
