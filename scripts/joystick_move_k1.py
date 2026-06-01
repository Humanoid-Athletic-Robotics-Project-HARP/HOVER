"""Joystick → K1 policy command pipeline.

Reads a connected gamepad via pygame and maps axes to K1 hand/head position
targets (OmniH2O mode) or upper-body joint angle targets, writing commands in
the same format as live_camera_k1.py so both can feed the same deployment
pipeline.

Controller layout — OmniH2O mode (default):
    Left  stick  X / Y  → left  hand lateral / forward
    Right stick  X / Y  → right hand lateral / forward
    L2 / R2             → left / right hand height
    Hat up / down       → head height (integrated)
    Hat left / right    → head yaw   (integrated)
    L1 (button 4)       → head pitch up
    R1 (button 5)       → head pitch down
    L3 (button 10)      → reset all targets to neutral
    Select (button 8)   → toggle mode (oh2o ↔ joint_angles)

Controller layout — joint_angles mode:
    Left  stick  X / Y  → Left_Shoulder_Roll  / ALeft_Shoulder_Pitch
    Right stick  X / Y  → Right_Shoulder_Roll / ARight_Shoulder_Pitch
    L2 / R2             → Left_Elbow_Pitch    / Right_Elbow_Pitch
    Hat up / down       → Head_pitch (integrated)
    Hat left / right    → AAHead_yaw (integrated)
    L3 (button 10)      → reset to neutral pose

Note: axis indices follow PS4/DualSense layout on Linux. Xbox controllers
swap axes 2↔3 and 4↔5 — use --axis-map to remap if needed.

Requirements:
    pip install pygame numpy

Usage:
    python scripts/joystick_move_k1.py
    python scripts/joystick_move_k1.py --mode joint_angles
    python scripts/joystick_move_k1.py --output socket:tcp://localhost:5555
"""

import argparse
import time

import numpy as np

try:
    import pygame
except ImportError:
    raise ImportError("Please install pygame: pip install pygame")


# ─── Constants ───────────────────────────────────────────────────────────────

# Matches live_camera_k1.py and the env config joint ordering
K1_JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

# From NeuralWBCEnvCfgK1.robot_init_state
K1_NEUTRAL_JOINTS = np.array([
    0.0,    # AAHead_yaw
    0.0,    # Head_pitch
    0.0,    # ALeft_Shoulder_Pitch
    0.0,    # Left_Shoulder_Roll
    0.0,    # Left_Elbow_Pitch
    0.0,    # Left_Elbow_Yaw
    0.0,    # ARight_Shoulder_Pitch
    0.0,    # Right_Shoulder_Roll
    0.0,    # Right_Elbow_Pitch
    0.0,    # Right_Elbow_Yaw
    -0.28,  # Left_Hip_Pitch
    0.0,    # Left_Hip_Roll
    0.0,    # Left_Hip_Yaw
    0.56,   # Left_Knee_Pitch
    -0.28,  # Left_Ankle_Pitch
    0.0,    # Left_Ankle_Roll
    -0.28,  # Right_Hip_Pitch
    0.0,    # Right_Hip_Roll
    0.0,    # Right_Hip_Yaw
    0.56,   # Right_Knee_Pitch
    -0.28,  # Right_Ankle_Pitch
    0.0,    # Right_Ankle_Roll
], dtype=np.float32)

# Neutral target positions in robot frame (x=forward, y=left, z=up from root).
# Tracked bodies per NeuralWBCEnvCfgK1: left_hand_link, right_hand_link, Head_2.
NEUTRAL_OH2O = {
    "head_pos":       np.array([0.00,  0.00, 0.40], dtype=np.float32),
    "left_hand_pos":  np.array([0.30,  0.20, 0.00], dtype=np.float32),
    "right_hand_pos": np.array([0.30, -0.20, 0.00], dtype=np.float32),
}

# Joint limits (radians) — from NeuralWBCEnvCfgK1.position_limit
JOINT_LIMITS = {
    "AAHead_yaw":           (-1.0,    1.0),
    "Head_pitch":           (-0.349,  0.855),
    "ALeft_Shoulder_Pitch": (-3.316,  1.22),
    "Left_Shoulder_Roll":   (-1.74,   1.57),
    "Left_Elbow_Pitch":     (-2.27,   2.27),
    "Right_Shoulder_Roll":  (-1.57,   1.74),
    "ARight_Shoulder_Pitch":(-3.316,  1.22),
    "Right_Elbow_Pitch":    (-2.27,   2.27),
}

# Hand workspace limits (metres from root)
HAND_X_RANGE = (-0.10, 0.55)
HAND_Y_RANGE = (-0.50, 0.50)
HAND_Z_RANGE = (-0.30, 0.40)
HEAD_Z_RANGE = (0.20,  0.55)

DEADZONE    = 0.08
HAND_SPEED  = 0.30   # m/s per full stick deflection
HEIGHT_RATE = 0.20   # m/s per full trigger pull
HEAD_RATE   = 0.60   # rad/s per full hat press

# PS4 / DualSense axis indices (default)
AX_LEFT_X  = 0
AX_LEFT_Y  = 1
AX_L2      = 2
AX_RIGHT_X = 3
AX_RIGHT_Y = 4
AX_R2      = 5

BTN_L1     = 4
BTN_R1     = 5
BTN_SELECT = 8
BTN_L3     = 10


# ─── Joystick ────────────────────────────────────────────────────────────────

class JoystickController:
    def __init__(self, device_index=0):
        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError("No gamepad detected — plug one in and retry.")

        idx = min(device_index, count - 1)
        self.joy = pygame.joystick.Joystick(idx)
        self.joy.init()
        print(f"Controller: {self.joy.get_name()}  "
              f"(axes={self.joy.get_numaxes()}, buttons={self.joy.get_numbuttons()}, "
              f"hats={self.joy.get_numhats()})")

    def read(self):
        pygame.event.pump()
        axes    = [self.joy.get_axis(i) for i in range(self.joy.get_numaxes())]
        buttons = [self.joy.get_button(i) for i in range(self.joy.get_numbuttons())]
        hat     = self.joy.get_hat(0) if self.joy.get_numhats() > 0 else (0, 0)
        return axes, buttons, hat


def _ax(axes, idx):
    v = axes[idx] if idx < len(axes) else 0.0
    return v if abs(v) > DEADZONE else 0.0


def _btn(buttons, idx):
    return bool(buttons[idx]) if idx < len(buttons) else False


# ─── Command computation ──────────────────────────────────────────────────────

def update_oh2o(targets, axes, buttons, hat, dt):
    """Integrate joystick inputs into absolute hand/head position targets."""
    lx = -_ax(axes, AX_LEFT_Y)   # stick up   → hand forward (+x)
    ly = -_ax(axes, AX_LEFT_X)   # stick left → hand left    (+y)
    rx = -_ax(axes, AX_RIGHT_Y)
    ry = -_ax(axes, AX_RIGHT_X)

    # Triggers: remap -1..1 → 0..1, then centre so idle=0
    l2 = (_ax(axes, AX_L2) + 1.0) / 2.0 - 0.5
    r2 = (_ax(axes, AX_R2) + 1.0) / 2.0 - 0.5

    targets["left_hand_pos"][0]  += lx * HAND_SPEED  * dt
    targets["left_hand_pos"][1]  += ly * HAND_SPEED  * dt
    targets["left_hand_pos"][2]  += l2 * HEIGHT_RATE * dt * 2

    targets["right_hand_pos"][0] += rx * HAND_SPEED  * dt
    targets["right_hand_pos"][1] += ry * HAND_SPEED  * dt
    targets["right_hand_pos"][2] += r2 * HEIGHT_RATE * dt * 2

    # Hat integrates head yaw (x) and height (z)
    targets["head_pos"][1] += hat[0] * HEAD_RATE * dt
    targets["head_pos"][2] += hat[1] * HEAD_RATE * dt

    # Clamp to workspace
    targets["left_hand_pos"][0]  = np.clip(targets["left_hand_pos"][0],  *HAND_X_RANGE)
    targets["left_hand_pos"][1]  = np.clip(targets["left_hand_pos"][1],  *HAND_Y_RANGE)
    targets["left_hand_pos"][2]  = np.clip(targets["left_hand_pos"][2],  *HAND_Z_RANGE)
    targets["right_hand_pos"][0] = np.clip(targets["right_hand_pos"][0], *HAND_X_RANGE)
    targets["right_hand_pos"][1] = np.clip(targets["right_hand_pos"][1], *HAND_Y_RANGE)
    targets["right_hand_pos"][2] = np.clip(targets["right_hand_pos"][2], *HAND_Z_RANGE)
    targets["head_pos"][1]       = np.clip(targets["head_pos"][1],       *HAND_Y_RANGE)
    targets["head_pos"][2]       = np.clip(targets["head_pos"][2],       *HEAD_Z_RANGE)


def update_joint_angles(joints, axes, buttons, hat, dt):
    """Integrate joystick inputs into upper-body joint angle targets."""
    MAX_SHOULDER = 1.0
    MAX_ELBOW    = 1.5

    joints[2] += -_ax(axes, AX_LEFT_Y)  * MAX_SHOULDER * dt * 2  # ALeft_Shoulder_Pitch
    joints[3] +=  _ax(axes, AX_LEFT_X)  * MAX_SHOULDER * dt * 2  # Left_Shoulder_Roll
    joints[6] += -_ax(axes, AX_RIGHT_Y) * MAX_SHOULDER * dt * 2  # ARight_Shoulder_Pitch
    joints[7] +=  _ax(axes, AX_RIGHT_X) * MAX_SHOULDER * dt * 2  # Right_Shoulder_Roll

    l2 = (_ax(axes, AX_L2) + 1.0) / 2.0
    r2 = (_ax(axes, AX_R2) + 1.0) / 2.0
    joints[4] += l2 * MAX_ELBOW * dt * 2   # Left_Elbow_Pitch
    joints[8] += r2 * MAX_ELBOW * dt * 2   # Right_Elbow_Pitch

    joints[0] += hat[0] * HEAD_RATE * dt   # AAHead_yaw
    joints[1] -= hat[1] * HEAD_RATE * dt   # Head_pitch

    # Clamp to joint limits
    for name, (lo, hi) in JOINT_LIMITS.items():
        idx = K1_JOINT_NAMES.index(name)
        joints[idx] = np.clip(joints[idx], lo, hi)


# ─── Output ──────────────────────────────────────────────────────────────────

class CommandPublisher:
    """Same interface as live_camera_k1.py so both can feed the same pipeline."""

    def __init__(self, output_method="stdout", output_path="k1_joystick_commands.csv"):
        self.method = output_method
        self._socket = None

        if output_method == "file":
            self._file = open(output_path, "w")
            self._file.write("timestamp," + ",".join(K1_JOINT_NAMES) + "\n")
        elif output_method == "socket":
            try:
                import zmq
                context = zmq.Context()
                self._socket = context.socket(zmq.PUB)
                self._socket.bind(output_path)
                print(f"Publishing on {output_path}")
            except ImportError:
                raise ImportError("ZMQ output requires pyzmq: pip install pyzmq")

    def publish(self, timestamp, joint_angles=None, oh2o_commands=None):
        if self.method == "file" and joint_angles is not None:
            line = f"{timestamp:.4f}," + ",".join(f"{a:.6f}" for a in joint_angles)
            self._file.write(line + "\n")
            self._file.flush()
        elif self.method == "socket" and self._socket:
            msg = {"timestamp": timestamp}
            if joint_angles is not None:
                msg["joint_angles"] = joint_angles.tolist()
            if oh2o_commands:
                msg["oh2o"] = {k: v.tolist() for k, v in oh2o_commands.items()}
            self._socket.send_json(msg)
        elif self.method == "stdout":
            if oh2o_commands:
                h  = oh2o_commands["head_pos"]
                lh = oh2o_commands["left_hand_pos"]
                rh = oh2o_commands["right_hand_pos"]
                print(f"t={timestamp:.2f}  "
                      f"head=[{h[0]:.2f},{h[1]:.2f},{h[2]:.2f}]  "
                      f"L=[{lh[0]:.2f},{lh[1]:.2f},{lh[2]:.2f}]  "
                      f"R=[{rh[0]:.2f},{rh[1]:.2f},{rh[2]:.2f}]")
            elif joint_angles is not None:
                print(f"t={timestamp:.2f}  joints={np.round(joint_angles[:6], 2)}...")

    def close(self):
        if hasattr(self, "_file"):
            self._file.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Joystick → K1 pose commands")
    parser.add_argument("--controller", type=int, default=0, help="Gamepad device index")
    parser.add_argument("--mode", choices=["oh2o", "joint_angles"], default="oh2o",
                        help="oh2o: head+hand position targets  joint_angles: direct upper-body joints")
    parser.add_argument("--output", type=str, default="stdout",
                        help="stdout | file:path.csv | socket:tcp://host:port")
    parser.add_argument("--hz", type=int, default=50, help="Publish rate (Hz)")
    parser.add_argument("--ax-right-x", type=int, default=AX_RIGHT_X,
                        help="Axis index for right stick X (default 3, Xbox: 3)")
    parser.add_argument("--ax-right-y", type=int, default=AX_RIGHT_Y,
                        help="Axis index for right stick Y (default 4, Xbox: 4)")
    parser.add_argument("--ax-l2", type=int, default=AX_L2,
                        help="Axis index for L2 trigger (default 2, Xbox: 2)")
    parser.add_argument("--ax-r2", type=int, default=AX_R2,
                        help="Axis index for R2 trigger (default 5, Xbox: 5)")
    args = parser.parse_args()

    global AX_RIGHT_X, AX_RIGHT_Y, AX_L2, AX_R2
    AX_RIGHT_X = args.ax_right_x
    AX_RIGHT_Y = args.ax_right_y
    AX_L2      = args.ax_l2
    AX_R2      = args.ax_r2

    if args.output.startswith("file:"):
        publisher = CommandPublisher("file", args.output[5:])
    elif args.output.startswith("socket:"):
        publisher = CommandPublisher("socket", args.output[7:])
    else:
        publisher = CommandPublisher("stdout")

    joy  = JoystickController(args.controller)
    dt   = 1.0 / args.hz
    mode = args.mode

    oh2o_targets  = {k: v.copy() for k, v in NEUTRAL_OH2O.items()}
    joint_targets = K1_NEUTRAL_JOINTS.copy()

    prev_select = False
    prev_l3     = False
    start       = time.time()

    print(f"\nRunning in '{mode}' mode at {args.hz} Hz.  Ctrl-C to quit.")
    print("  Left  stick      → left hand / left shoulder")
    print("  Right stick      → right hand / right shoulder")
    print("  L2 / R2          → hand height / elbow pitch")
    print("  Hat              → head yaw + height (integrated)")
    print("  L3  (btn 10)     → reset to neutral")
    print("  Select (btn  8)  → toggle oh2o ↔ joint_angles\n")

    try:
        while True:
            t0 = time.time()
            axes, buttons, hat = joy.read()
            timestamp = t0 - start

            select_now = _btn(buttons, BTN_SELECT)
            if select_now and not prev_select:
                mode = "joint_angles" if mode == "oh2o" else "oh2o"
                print(f"  → switched to mode: {mode}")
            prev_select = select_now

            l3_now = _btn(buttons, BTN_L3)
            if l3_now and not prev_l3:
                oh2o_targets  = {k: v.copy() for k, v in NEUTRAL_OH2O.items()}
                joint_targets = K1_NEUTRAL_JOINTS.copy()
                print("  → reset to neutral")
            prev_l3 = l3_now

            oh2o_cmd     = None
            joint_angles = None

            if mode == "oh2o":
                update_oh2o(oh2o_targets, axes, buttons, hat, dt)
                oh2o_cmd = oh2o_targets
            else:
                update_joint_angles(joint_targets, axes, buttons, hat, dt)
                joint_angles = joint_targets

            publisher.publish(timestamp, joint_angles=joint_angles, oh2o_commands=oh2o_cmd)

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        publisher.close()
        pygame.quit()


if __name__ == "__main__":
    main()
