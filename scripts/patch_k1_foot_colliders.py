"""Patch K1 USD: replace mesh foot collision shapes with thin boxes.

Mesh colliders in PhysX cause erratic contacts for humanoid feet.
A thin box matching the foot footprint gives stable, flat ground contacts.

Usage:
    python3 scripts/patch_k1_foot_colliders.py
Output:
    third_party/booster_assets/robots/K1/K1_22dof_simple_feet.usd
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

URDF_DIR = os.path.abspath("third_party/booster_assets/robots/K1")
IN_USD    = os.path.join(URDF_DIR, "K1_22dof.usd")
OUT_USD   = os.path.join(URDF_DIR, "K1_22dof_simple_feet.usd")

# Foot box dimensions (from mesh bounding box + inertia analysis):
#   length (x, fore-aft): 0.16m  → half 0.08m
#   width  (y, lateral):  0.08m  → half 0.04m
#   height (z):           0.03m  → half 0.015m
# Origin offset: bottom of box at z=-0.084 (mesh bottom), so center at z=-0.069
FOOT_BOX_HALF = Gf.Vec3f(0.08, 0.04, 0.015)
FOOT_BOX_OFFSET = Gf.Vec3d(0.018, 0.0, -0.069)

stage = Usd.Stage.Open(IN_USD)

foot_links = ["left_foot_link", "right_foot_link"]

for prim in stage.Traverse():
    prim_name = prim.GetName()
    if prim_name not in foot_links:
        continue

    print(f"Patching: {prim.GetPath()}")

    # Find and remove existing collision children (mesh-based)
    for child in list(prim.GetChildren()):
        child_prim = stage.GetPrimAtPath(child.GetPath())
        if child_prim.HasAPI(UsdPhysics.CollisionAPI):
            print(f"  Removing old collision prim: {child.GetPath()}")
            stage.RemovePrim(child.GetPath())

    # Add a new Cube collision prim
    col_path = prim.GetPath().AppendChild("foot_collision_box")
    cube = UsdGeom.Cube.Define(stage, col_path)
    cube.GetSizeAttr().Set(1.0)  # unit cube, scaled via xformOp

    # Scale to box dimensions
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(FOOT_BOX_OFFSET)
    xform.AddScaleOp().Set(Gf.Vec3f(
        FOOT_BOX_HALF[0] * 2,
        FOOT_BOX_HALF[1] * 2,
        FOOT_BOX_HALF[2] * 2,
    ))

    # Make it a collision shape (not visible)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()

    print(f"  Added box collider at {col_path}")

stage.Export(OUT_USD)
print(f"\nSaved: {OUT_USD}")

simulation_app.close()
