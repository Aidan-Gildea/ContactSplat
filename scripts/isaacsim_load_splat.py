# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Load an exported NuRec splat USDZ into Isaac Sim and verify it resolves.

Headless smoke test for the export: opens a stage, references the USDZ, steps
the renderer, and reports the resulting prims and world-space bounds. Use it to
confirm an asset is loadable before opening it in the GUI.

Run with Isaac Sim's own interpreter:
    ~/isaac-sim/python.sh scripts/isaacsim_load_splat.py <path to .usdz>
"""

import argparse
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("usdz", help="path to the exported .usdz")
parser.add_argument(
    "--gui", action="store_true", help="open the full GUI instead of running headless"
)
parser.add_argument(
    "--report",
    default=None,
    help="write the verification result here as JSON. Kit captures Python stdout into "
    "its own logger and force-exits on shutdown, so neither printed output nor the "
    "process exit code is reliable evidence -- write to a file instead.",
)
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402  (must precede omni imports)

simulation_app = SimulationApp({"headless": not args.gui, "renderer": "RaytracedLighting"})

import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

# Reference the splat under its own prim so it can sit alongside robots/props.
prim = stage.DefinePrim("/World/AriaSplat", "Xform")
prim.GetReferences().AddReference(args.usdz)

# Let the renderer resolve the payload.
for _ in range(60):
    simulation_app.update()

print(f"\nstage upAxis      : {UsdGeom.GetStageUpAxis(stage)}")
print(f"stage metersPerUnit: {UsdGeom.GetStageMetersPerUnit(stage)}")

found = []
for p in Usd.PrimRange(prim):
    found.append((p.GetPath().pathString, p.GetTypeName()))
for path, type_name in found:
    print(f"  {path}  {type_name}")

nurec = [t for _, t in found if "NuRec" in str(t)]
print(f"\nNuRec field prims resolved: {len(nurec)}")

# Verify the splat's frame. 3dgrut's ply_to_usd.py bakes a Y-down -> Z-up
# conversion rotation into the Volume prim, which is wrong for PLYs already in
# the MPS world frame (Z-up metres) and lays the scene on its side; the export
# script strips it via scripts/fix_nurec_usdz_frame.py. Report the composed
# volume transform so a regressed asset is caught here rather than in the GUI.
volume_xform = None
volume_xform_is_identity = None
for p in Usd.PrimRange(prim):
    if p.GetTypeName() == "Volume":
        m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        volume_xform = [list(row) for row in m]
        volume_xform_is_identity = Gf.IsClose(m, Gf.Matrix4d(1.0), 1e-9)
        print(f"volume local-to-world: {m}")
        print(
            "volume frame identity :",
            volume_xform_is_identity,
            "" if volume_xform_is_identity else "<-- baked conversion rotation present, "
            "splat will NOT be upright; re-export or run scripts/fix_nurec_usdz_frame.py",
        )
        break

bounds = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
).ComputeWorldBound(prim)
r = bounds.ComputeAlignedRange()
if not r.IsEmpty():
    print(f"world AABB min: {r.GetMin()}")
    print(f"world AABB max: {r.GetMax()}")

ok = len(nurec) >= 1
print("\nRESULT:", "OK - splat loaded and resolved" if ok else "FAILED - no NuRec prims")

if args.report:
    import json

    with open(args.report, "w") as f:
        json.dump(
            {
                "ok": ok,
                "usdz": args.usdz,
                "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                "prims": [{"path": p, "type": str(t)} for p, t in found],
                "nurec_prim_count": len(nurec),
                "volume_local_to_world": volume_xform,
                "volume_frame_is_identity": volume_xform_is_identity,
                "world_aabb_min": list(r.GetMin()) if not r.IsEmpty() else None,
                "world_aabb_max": list(r.GetMax()) if not r.IsEmpty() else None,
            },
            f,
            indent=2,
        )

simulation_app.close()
sys.exit(0 if ok else 1)
