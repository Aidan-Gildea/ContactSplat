#!/usr/bin/env python
"""Add a mesh to a NuRec splat USDZ as a hidden collider, producing one self-contained USDZ.

Input
    splat.usdz   NuRec Gaussian volume from 3dgrut's ply_to_usd.py, with the frame fix from
                 scripts/fix_nurec_usdz_frame.py already applied (volume transform = identity)
    mesh.ply     triangle mesh in the same frame, e.g. photogrammetry/run_photogrammetry.sh's
                 mesh_delaunay.ply or mesh_delaunay_decimated.ply

Both are in the MPS world frame (gravity-aligned, Z-up, metres), so the mesh is added with no
transform and lines up with the splat exactly.

Output: one USDZ whose /World holds
    /World/gauss      the splat, visible (unchanged from the input package)
    /World/collider   the mesh: invisible, static triangle-mesh collision (no simplification)

Spike removal. COLMAP's Delaunay mesher connects some surface points to far-away noise and sky,
leaving long thin triangles tens of metres long. As an invisible collider these would be
invisible walls and ceilings, so every triangle with an edge longer than MAX_EDGE_M is dropped.
On the Outside_20260812_141244 mesh that removes 0.17% of triangles, all far from or high above
the walked path, and leaves floor coverage along the path unchanged (0.880 before and after).
Typical triangles there are about 3 cm across. Nothing else is removed or simplified.

Run with a Python that has pxr (usd-core), numpy and plyfile, e.g. the 3dgrut env:

    python isaacsim/add_mesh_collider.py splat.usdz mesh.ply out.usdz
"""

import argparse
import os
import sys
import tempfile
import zipfile

import numpy as np
from plyfile import PlyData
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

MAX_EDGE_M = 2.0  # drop triangles with any edge longer than this (Delaunay spikes)
COLLIDER_NAME = "collider"
COLLIDER_FILE = "collider.usdc"


def face_index_property(path):
    """Name of the face index list: COLMAP writes 'vertex_index', Open3D 'vertex_indices'."""
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("property list") and line.split()[-1] in ("vertex_index", "vertex_indices"):
                return line.split()[-1]
            if line == "end_header":
                break
    raise SystemExit(f"{path}: no face vertex_index / vertex_indices property")


def read_triangle_mesh(path):
    """Return (points float32 (N,3), triangles int64 (M,3)) from a binary or ASCII PLY."""
    index_name = face_index_property(path)
    try:  # fast path for fixed-size face lists (plyfile >= 0.8)
        ply = PlyData.read(path, known_list_len={"face": {index_name: 3}})
    except TypeError:
        ply = PlyData.read(path)
    v = ply["vertex"]
    points = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    faces = ply["face"][index_name]
    tris = np.stack(faces) if faces.dtype == object else np.asarray(faces)
    tris = tris.astype(np.int64).reshape(-1, 3)
    return points, tris


def drop_long_triangles(points, tris, max_edge):
    """Remove triangles with an edge longer than max_edge, then drop unused vertices."""
    p = points[tris]  # (M, 3 corners, 3 xyz)
    longest = np.max(
        np.stack([np.linalg.norm(p[:, i] - p[:, (i + 1) % 3], axis=1) for i in range(3)], 1),
        axis=1,
    )
    keep = longest <= max_edge
    kept = tris[keep]
    used, remapped = np.unique(kept, return_inverse=True)
    return points[used], remapped.reshape(-1, 3).astype(np.int32), int((~keep).sum())


def write_collider_layer(path, points, tris):
    """Write the mesh as an invisible static triangle-mesh collider, default prim /collider."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    mesh = UsdGeom.Mesh.Define(stage, f"/{COLLIDER_NAME}")
    stage.SetDefaultPrim(mesh.GetPrim())
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.ravel()))
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([points.min(0), points.max(0)])))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)

    # Hidden: the splat does all the rendering. Collision ignores visibility.
    mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    # Static collider (no RigidBodyAPI). Approximation "none" = the exact triangle mesh,
    # which PhysX supports for static geometry.
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
        UsdPhysics.Tokens.none
    )
    stage.GetRootLayer().Save()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("splat_usdz", help="NuRec splat .usdz (frame already fixed)")
    ap.add_argument("mesh_ply", help="triangle mesh .ply in the same frame")
    ap.add_argument("out_usdz", help="combined .usdz to write")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(args.splat_usdz) as z:
            names = z.namelist()
            z.extractall(tmp)
        root_name = names[0]  # USDZ rule: the first entry is the root layer
        if COLLIDER_FILE in names:
            sys.exit(f"{args.splat_usdz} already contains {COLLIDER_FILE}")

        root = Sdf.Layer.FindOrOpen(os.path.join(tmp, root_name))
        world_path = Sdf.Path(f"/{root.defaultPrim}") if root.defaultPrim else Sdf.Path("/World")
        world = root.GetPrimAtPath(world_path)
        if world is None:
            sys.exit(f"no {world_path} prim in {root_name}")

        # --- mesh -> collider layer ------------------------------------------------------
        points, tris = read_triangle_mesh(args.mesh_ply)
        n_in = len(tris)
        points, tris, n_dropped = drop_long_triangles(points, tris, MAX_EDGE_M)
        print(f"mesh: {n_in:,} triangles in, {n_dropped:,} longer than {MAX_EDGE_M} m dropped "
              f"({100.0 * n_dropped / n_in:.2f}%), {len(tris):,} kept, {len(points):,} vertices")
        print(f"collider bounds (m): {points.min(0).round(2)} .. {points.max(0).round(2)}")
        write_collider_layer(os.path.join(tmp, COLLIDER_FILE), points, tris)

        # --- reference it from the root layer, next to the splat ------------------------
        spec = Sdf.PrimSpec(world, COLLIDER_NAME, Sdf.SpecifierDef)
        spec.referenceList.Prepend(Sdf.Reference(f"./{COLLIDER_FILE}"))
        root.Save()

        # --- repackage: root layer first, then everything else --------------------------
        out_tmp = args.out_usdz + ".partial"
        writer = Sdf.ZipFileWriter.CreateNew(out_tmp)
        for name in names + [COLLIDER_FILE]:
            writer.AddFile(os.path.join(tmp, name), name)
        writer.Save()
        os.replace(out_tmp, args.out_usdz)

    print(f"wrote {args.out_usdz}")
    print(f"  {world_path}/gauss      splat, visible")
    print(f"  {world_path}/{COLLIDER_NAME}   mesh, invisible static collider")


if __name__ == "__main__":
    main()
