"""Debug 2 — IK Reachability (uniform lattice)
Samples a uniform 3-D grid of wrist targets around the left shoulder and solves IK for each.
Specifically tests forward reach (positive X) vs lateral reach (positive Y).

Ask: Can the arm reach forward at all? Where is the reachable shell in workspace?

Outputs:
  debug_output/02_ik_reach.gif      — robot arm at each IK solution (green=reached, red=failed)
  debug_output/02_ik_reach_map.png  — 3-D scatter of the lattice coloured by result
  debug_output/02_ik_reach_topdown.png — top-down XY slice at shoulder height (forward vs lateral)
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import minimize
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SCENE_XML = HERE.parent.parent / "neural_wbc/data/data/mujoco/models/scene_k1_vis.xml"
OUT = HERE / "debug_output"
OUT.mkdir(exist_ok=True)

# Joints used for left-arm IK (4-DOF: shoulder pitch, roll, elbow pitch, elbow yaw)
LEFT_ARM_JOINTS = [
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
]
LEFT_HAND_BODY    = "left_hand_link"
LEFT_HAND_SITE    = "left_hand_tip"
LEFT_SHOULDER_BODY = "Left_Arm_1"


def make_camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def overlay(img_arr, lines, color=(255, 255, 255)):
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    y = 8
    for line in lines:
        draw.text((11, y + 1), line, fill=(0, 0, 0))
        draw.text((10, y), line, fill=color)
        y += 18
    return np.array(img)


def setup():
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEFT_ARM_JOINTS]
    adrs = [model.jnt_qposadr[jid] for jid in jids]
    limits = [model.jnt_range[jid] for jid in jids]
    hand_bid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_HAND_BODY)
    hand_sid    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_HAND_SITE)
    shoulder_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_SHOULDER_BODY)

    return model, data, adrs, limits, hand_bid, hand_sid, shoulder_bid


_OPT = None

def _get_opt():
    global _OPT
    if _OPT is None:
        _OPT = mujoco.MjvOption()
    return _OPT


def fk_hand(model, data, adrs, q):
    for adr, val in zip(adrs, q):
        data.qpos[adr] = val
    mujoco.mj_kinematics(model, data)
    return data.xpos[hand_bid].copy()


def hand_tip_world(data, hand_sid):
    """World position of the hand tip site."""
    return data.site_xpos[hand_sid].copy()


def solve_ik(target, model, data, adrs, limits, home_q, hand_sid, warm_q=None):
    def objective(q):
        for adr, val in zip(adrs, q):
            data.qpos[adr] = val
        mujoco.mj_kinematics(model, data)
        err = hand_tip_world(data, hand_sid) - target
        return float(err @ err)

    bounds = [(lo, hi) for lo, hi in limits]
    rng = np.random.default_rng(0)
    # Candidates: home, previous solution (spatial warm start), 3 random
    x0_candidates = [home_q]
    if warm_q is not None:
        x0_candidates.append(warm_q)
    for _ in range(3):
        x0_candidates.append(rng.uniform([lo for lo, _ in limits], [hi for _, hi in limits]))
    best_res = None
    for x0 in x0_candidates:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    dist = np.sqrt(best_res.fun)
    return best_res.x, dist, dist < 0.04


def draw_target_sphere(model, data, renderer, cam, target_pos, solved, home_qpos, adrs, q_sol):
    """Render with arm at IK solution and a coloured dot for the target."""
    data.qpos[:] = home_qpos
    for adr, val in zip(adrs, q_sol):
        data.qpos[adr] = val
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam, scene_option=_get_opt())
    pixels = renderer.render().copy()
    color = (50, 220, 50) if solved else (220, 50, 50)
    label = "REACHED" if solved else "FAILED"
    target_str = f"target: ({target_pos[0]:+.3f}, {target_pos[1]:+.3f}, {target_pos[2]:+.3f})"
    return overlay(pixels, [f"IK  {label}", target_str], color=color)


ARM_REACH = 0.462   # shoulder-to-hand-tip length (m), Left_Arm_4 mesh Y max = 0.228 m
LATTICE_STEPS = 7   # per axis → 7³ = 343 raw points before filtering

def sample_lattice_targets(shoulder_pos):
    """Uniform 3-D lattice centred on the shoulder, returned in snake-scan order.

    Snake scan: sweep X for each Y, flip X direction each row, flip Y direction
    each Z slice — so consecutive targets are always adjacent → smooth arm motion.
    """
    half = ARM_REACH * 1.2
    coords = np.linspace(-half, half, LATTICE_STEPS)

    ordered = []
    for iz, z in enumerate(coords):
        ys_this = coords if iz % 2 == 0 else coords[::-1]
        for iy, y in enumerate(ys_this):
            xs_this = coords if (iz + iy) % 2 == 0 else coords[::-1]
            for x in xs_this:
                pt = shoulder_pos + np.array([x, y, z])
                d = np.linalg.norm([x, y, z])
                if 0.05 < d < ARM_REACH * 1.05:
                    ordered.append((pt, d))

    targets = np.array([o[0] for o in ordered])
    dists   = np.array([o[1] for o in ordered])
    return targets, dists


INTERP_N = 5   # transition frames between consecutive IK solutions


def _add_sphere(scene, pos, rgba, radius=0.025):
    if scene.ngeom < scene.maxgeom:
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([radius, radius, radius]),
                            np.array(pos, dtype=np.float64),
                            np.eye(3).flatten(),
                            np.array(rgba, dtype=np.float32))
        scene.ngeom += 1


def render_q(model, data, renderer, cam, home_qpos, adrs, q, hand_sid,
             target_pos=None, solved=None):
    data.qpos[:] = home_qpos
    for adr, val in zip(adrs, q):
        data.qpos[adr] = val
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam, scene_option=_get_opt())

    scene = renderer._scene
    if target_pos is not None:
        # Green/red sphere = IK target (where hand SHOULD go)
        target_rgba = [0.1, 0.9, 0.1, 0.9] if solved else [0.9, 0.15, 0.15, 0.9]
        _add_sphere(scene, target_pos, target_rgba, radius=0.025)
        # White sphere = actual hand tip site position after IK
        _add_sphere(scene, hand_tip_world(data, hand_sid), [1.0, 1.0, 1.0, 0.85], radius=0.015)

    return renderer.render().copy()


def main():
    model, data, adrs, limits, hand_bid, hand_sid, shoulder_bid = setup()
    home_qpos = data.qpos.copy()
    home_q = np.array([data.qpos[a] for a in adrs])

    mujoco.mj_forward(model, data)
    shoulder_pos = data.xpos[shoulder_bid].copy()
    print(f"Left shoulder pos: {shoulder_pos}")

    targets, target_dists = sample_lattice_targets(shoulder_pos)
    print(f"Lattice: {len(targets)} targets (snake-scan order)")

    cam = make_camera([0.0, 0.1, 0.85], 1.8, 180, -10)
    renderer = mujoco.Renderer(model, height=480, width=640)
    tip_home = hand_tip_world(data, hand_sid)
    reach = np.linalg.norm(tip_home - shoulder_pos)
    print(f"IK site: left_hand_tip  pos={np.round(tip_home,4)}  reach={reach:.4f} m")

    # ── Phase 1: solve IK for every target ───────────────────────────────────
    results = []   # (target, solved, ik_dist, q_sol)
    prev_q = home_q.copy()
    print(f"Solving IK for {len(targets)} targets...")
    for i, (tgt, tdist) in enumerate(zip(targets, target_dists)):
        q_sol, dist, solved = solve_ik(tgt, model, data, adrs, limits, home_q, hand_sid,
                                       warm_q=prev_q)
        results.append((tgt, solved, dist, q_sol))
        prev_q = q_sol   # spatial warm start for next adjacent target
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(targets)}  solved={solved}  ik_err={dist:.4f} m")

    # ── Phase 2: build smooth GIF with interpolated transitions ──────────────
    frames = []
    prev_q_frame = home_q.copy()

    for i, (tgt, solved, dist, q_sol) in enumerate(results):
        rel = tgt - shoulder_pos
        color = (50, 220, 50) if solved else (220, 50, 50)
        label = "REACHED" if solved else "FAILED"

        # Interpolate joint angles from previous solution to this one
        for t in np.linspace(0, 1, INTERP_N + 1)[1:]:   # skip t=0 (already rendered as prev endpoint)
            q_interp = (1.0 - t) * prev_q_frame + t * q_sol
            # Show the target sphere throughout the transition, not just at the endpoint
            img = render_q(model, data, renderer, cam, home_qpos, adrs, q_interp,
                           hand_sid, target_pos=tgt, solved=solved)
            # Annotate at the endpoint frame
            if t >= 1.0 - 1e-6:
                img = overlay(img, [
                    f"{i+1}/{len(results)}  {label}",
                    f"X={rel[0]:+.2f} Y={rel[1]:+.2f} Z={rel[2]:+.2f} m",
                    f"ik err: {dist*100:.1f} cm",
                ], color=color)
            frames.append(img)

        prev_q_frame = q_sol

    data.qpos[:] = home_qpos
    mujoco.mj_forward(model, data)

    gif_path = OUT / "02_ik_reach.gif"
    imageio.mimsave(str(gif_path), frames, fps=8, loop=0)
    print(f"Saved {gif_path}")

    reached = [r for r in results if r[1]]
    failed  = [r for r in results if not r[1]]
    n_ok = len(reached)
    print(f"\nResults: {n_ok}/{len(results)} targets reached ({100*n_ok/len(results):.0f}%)")
    if failed:
        print(f"Mean IK error (failed): {np.mean([r[2] for r in failed]):.4f} m")

    # ── 3-D scatter plot ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if reached:
        pts = np.array([r[0] for r in reached])
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="green", s=50,
                   label=f"Reached ({n_ok})", alpha=0.8)
    if failed:
        pts = np.array([r[0] for r in failed])
        dists = np.array([r[2] for r in failed])
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=dists, cmap="Reds", s=60,
                        label=f"Failed ({len(failed)})", alpha=0.9, vmin=0.04, vmax=0.25)
        plt.colorbar(sc, ax=ax, label="IK residual (m)", shrink=0.6)
    ax.scatter(*shoulder_pos, c="blue", s=160, marker="*", label="Left shoulder", zorder=5)
    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (left)")
    ax.set_zlabel("Z (up)")
    ax.set_title(f"K1 left arm IK reachability — uniform lattice\n"
                 f"{n_ok}/{len(results)} reached  |  shoulder: {np.array2string(shoulder_pos, precision=2)}")
    ax.legend()
    plt.tight_layout()
    png_path = OUT / "02_ik_reach_map.png"
    plt.savefig(str(png_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path}")

    # ── Top-down XY view (forward vs lateral) ────────────────────────────────
    # Collapse all Z layers: show every point projected onto the XY plane,
    # coloured green/red and sized by how close to shoulder height.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: top-down XY projection (all Z levels)
    ax = axes[0]
    if reached:
        pts = np.array([r[0] for r in reached]) - shoulder_pos
        ax.scatter(pts[:, 0], pts[:, 1], c="green", s=80, alpha=0.7, label=f"Reached ({n_ok})", zorder=3)
    if failed:
        pts_f = np.array([r[0] for r in failed]) - shoulder_pos
        dists_f = np.array([r[2] for r in failed])
        ax.scatter(pts_f[:, 0], pts_f[:, 1], c=dists_f, cmap="Reds", s=80,
                   alpha=0.8, label=f"Failed ({len(failed)})", vmin=0.04, vmax=0.25, zorder=2)
    ax.scatter(0, 0, c="blue", s=200, marker="*", label="Shoulder", zorder=5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("X — forward →  (m)")
    ax.set_ylabel("Y — left →  (m)")
    ax.set_title("Top-down XY view (all Z levels)\nGreen = can reach, Red = cannot")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    # Label quadrants
    lim = ARM_REACH * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.text( lim * 0.6,  lim * 0.85, "forward-left",  ha="center", fontsize=8, color="gray")
    ax.text( lim * 0.6, -lim * 0.85, "forward-right", ha="center", fontsize=8, color="gray")
    ax.text(-lim * 0.6,  lim * 0.85, "back-left",     ha="center", fontsize=8, color="gray")
    ax.text(-lim * 0.6, -lim * 0.85, "back-right",    ha="center", fontsize=8, color="gray")

    # Right: side XZ view (forward vs height)
    ax2 = axes[1]
    if reached:
        pts = np.array([r[0] for r in reached]) - shoulder_pos
        ax2.scatter(pts[:, 0], pts[:, 2], c="green", s=80, alpha=0.7, label=f"Reached ({n_ok})", zorder=3)
    if failed:
        pts_f = np.array([r[0] for r in failed]) - shoulder_pos
        dists_f = np.array([r[2] for r in failed])
        sc2 = ax2.scatter(pts_f[:, 0], pts_f[:, 2], c=dists_f, cmap="Reds", s=80,
                          alpha=0.8, label=f"Failed ({len(failed)})", vmin=0.04, vmax=0.25, zorder=2)
        plt.colorbar(sc2, ax=ax2, label="IK residual (m)", shrink=0.8)
    ax2.scatter(0, 0, c="blue", s=200, marker="*", label="Shoulder", zorder=5)
    ax2.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("X — forward →  (m)")
    ax2.set_ylabel("Z — up →  (m)")
    ax2.set_title("Side XZ view (all Y levels)\nForward reach vs height")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)

    fig.suptitle(f"K1 left arm forward reach test — uniform {LATTICE_STEPS}³ lattice\n"
                 f"Arm reach = {ARM_REACH:.3f} m  |  {n_ok}/{len(results)} reachable",
                 fontsize=11)
    plt.tight_layout()
    png2_path = OUT / "02_ik_reach_topdown.png"
    plt.savefig(str(png2_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {png2_path}")

    renderer.close()


if __name__ == "__main__":
    main()
