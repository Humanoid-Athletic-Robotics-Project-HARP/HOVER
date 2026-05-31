"""Convert K1_22dof.urdf to USD for use with IsaacLab.

Run from the hover project root:
    python scripts/tools/convert_k1_urdf.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import pathlib
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

URDF = pathlib.Path(__file__).resolve().parents[2] / \
    "third_party/booster_assets/robots/K1/K1_22dof.urdf"
OUT  = URDF.with_suffix(".usd")

cfg = UrdfConverterCfg(
    asset_path=str(URDF),
    usd_dir=str(OUT.parent),
    usd_file_name=OUT.name,
    fix_base=False,
    merge_fixed_joints=True,
    self_collision=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
    ),
)

print(f"Converting: {URDF}")
converter = UrdfConverter(cfg)
print(f"Saved USD : {OUT}")

simulation_app.close()
