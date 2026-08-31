#!/usr/bin/env python3
"""Build the Isaac Sim environment: visual splat + invisible collision mesh.

Loads (optionally) the NuRec splat USDZ under /World/AriaSplat (visual only)
and a collision mesh under /World/AriaCollision with visibility=invisible,
UsdPhysics.CollisionAPI, and a chosen physics approximation
(sdf | convexDecomposition | none). PLY input is converted to a cached USD
sibling file on first use.

Both assets are in the MPS world frame (gravity-aligned, Z-up, metres) after
the E1 fix, so no registration transform is applied anywhere.

With --bench it also measures collision cooking/setup time (world.reset())
and steady-state physics step time, using a dynamic box dropped onto the
mesh so a collision pair is actually active.

Run with Isaac Sim's own interpreter:
    ~/isaac-sim/python.sh scripts/isaacsim_build_env.py \
        --collision output/.../fused_best_stereoNear_2dgsFar_trajfloor.ply \
        --approximation none --bench --report /path/report.json

Kit swallows Python stdout into its own logger and force-exits on shutdown:
neither printed output nor the process exit code is reliable evidence.
Always pass --report and read the JSON.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

DEFAULT_SPLAT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/isaacsim/"
    "Outside_20260812_141244.usdz",
)

APPROXIMATIONS = ("none", "sdf", "convexDecomposition")


# ---------------------------------------------------------------------------
# PLY reading (pure numpy -- no open3d in the Isaac Sim env)
# ---------------------------------------------------------------------------

_PLY_DTYPES = {
    "double": "<f8",
    "float": "<f4",
    "uchar": "u1",
    "uint8": "u1",
    "int": "<i4",
    "uint": "<u4",
}


def read_ply(path):
    """Read a binary_little_endian PLY (Open3D style) -> (verts Nx3 f64, tris Mx3 i64).

    Supports arbitrary scalar vertex properties (x/y/z extracted) and a pure
    triangle face list `property list uchar uint vertex_indices`.
    """
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        fmt = [l for l in header if l.startswith("format ")]
        if not fmt or "binary_little_endian" not in fmt[0]:
            raise ValueError(f"{path}: only binary_little_endian PLY supported: {fmt}")

        n_vert = n_face = 0
        vert_props = []  # (name, dtype-str)
        face_list = None  # (count_dtype, index_dtype)
        cur = None
        for line in header:
            t = line.split()
            if not t:
                continue
            if t[0] == "element":
                cur = t[1]
                if cur == "vertex":
                    n_vert = int(t[2])
                elif cur == "face":
                    n_face = int(t[2])
            elif t[0] == "property":
                if cur == "vertex":
                    if t[1] == "list":
                        raise ValueError(f"{path}: list property on vertex unsupported")
                    vert_props.append((t[2], _PLY_DTYPES[t[1]]))
                elif cur == "face":
                    if t[1] != "list":
                        raise ValueError(f"{path}: non-list face property unsupported")
                    face_list = (_PLY_DTYPES[t[2]], _PLY_DTYPES[t[3]])

        vdtype = np.dtype([(n, d) for n, d in vert_props])
        vbuf = f.read(n_vert * vdtype.itemsize)
        verts_rec = np.frombuffer(vbuf, dtype=vdtype, count=n_vert)
        verts = np.stack(
            [verts_rec["x"], verts_rec["y"], verts_rec["z"]], axis=1
        ).astype(np.float64)

        cd, idx = face_list
        fdtype = np.dtype([("n", cd), ("v", idx, (3,))])
        fbuf = f.read()
        if len(fbuf) < n_face * fdtype.itemsize:
            raise ValueError(f"{path}: truncated face block")
        faces_rec = np.frombuffer(fbuf, dtype=fdtype, count=n_face)
        if not np.all(faces_rec["n"] == 3):
            raise ValueError(f"{path}: non-triangle faces present")
        tris = faces_rec["v"].astype(np.int64)
    return verts, tris


# ---------------------------------------------------------------------------
# PLY -> USD conversion (requires pxr; call only after SimulationApp init,
# or from any interpreter that can `from pxr import Usd`)
# ---------------------------------------------------------------------------

def convert_ply_to_usd(ply_path, usd_path=None, force=False):
    """Convert a triangle PLY to a USD mesh file (Z-up, metres). Cached."""
    from pxr import Usd, UsdGeom, Vt, Sdf, Gf

    ply_path = os.path.abspath(ply_path)
    if usd_path is None:
        usd_path = os.path.splitext(ply_path)[0] + ".collision.usd"
    if os.path.exists(usd_path) and not force:
        if os.path.getmtime(usd_path) >= os.path.getmtime(ply_path):
            return usd_path, None  # cache hit
    t0 = time.perf_counter()
    verts, tris = read_ply(ply_path)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    # write to a temp file and atomically rename, so a crashed conversion can
    # never leave a truncated file that a later run would take as a cache hit
    tmp_path = usd_path + ".tmp.usd"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    stage = Usd.Stage.CreateNew(tmp_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/AriaCollisionMesh")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/AriaCollisionMesh/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(tris), 3, np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(tris.astype(np.int32).ravel())
    )
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    mesh.CreateExtentAttr(
        Vt.Vec3fArray(
            [Gf.Vec3f(*[float(v) for v in lo]), Gf.Vec3f(*[float(v) for v in hi])]
        )
    )
    mesh.CreateSubdivisionSchemeAttr("none")
    # provenance for downstream sanity checks
    prim = mesh.GetPrim()
    prim.SetCustomDataByKey("sourcePly", ply_path)
    prim.SetCustomDataByKey("sourceAabbMin", Gf.Vec3d(*[float(v) for v in lo]))
    prim.SetCustomDataByKey("sourceAabbMax", Gf.Vec3d(*[float(v) for v in hi]))
    stage.GetRootLayer().Save()
    del stage
    os.replace(tmp_path, usd_path)
    t_write = time.perf_counter() - t0
    info = {
        "n_vertices": int(len(verts)),
        "n_triangles": int(len(tris)),
        "aabb_min": [float(v) for v in lo],
        "aabb_max": [float(v) for v in hi],
        "read_s": round(t_read, 3),
        "write_s": round(t_write, 3),
    }
    return usd_path, info


def resolve_collision_usd(collision_path, force=False):
    """Accept .ply or .usd; return (usd_path, convert_info_or_None)."""
    collision_path = os.path.abspath(collision_path)
    if collision_path.lower().endswith((".usd", ".usda", ".usdc")):
        return collision_path, None
    return convert_ply_to_usd(collision_path, force=force)


# ---------------------------------------------------------------------------
# Stage setup helpers (require pxr after SimulationApp init)
# ---------------------------------------------------------------------------

def setup_collision_prim(
    stage,
    usd_path,
    xform_path="/World/AriaCollision",
    approximation="none",
    sdf_resolution=256,
    friction=(0.8, 0.7),
):
    """Reference the mesh USD, make it invisible, apply collision APIs.

    approximation: 'none' (static triangle mesh) | 'sdf' | 'convexDecomposition'.
    Returns an info dict.
    """
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    xform = stage.DefinePrim(xform_path, "Xform")
    xform.GetReferences().AddReference(usd_path)
    UsdGeom.Imageable(xform).MakeInvisible()

    mesh_prim = None
    for p in Usd.PrimRange(xform):
        if p.GetTypeName() == "Mesh":
            mesh_prim = p
            break
    if mesh_prim is None:
        raise RuntimeError(f"no Mesh prim found under {xform_path} ({usd_path})")

    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
    mesh_col.CreateApproximationAttr().Set(approximation)
    if approximation == "sdf":
        sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
        sdf_api.CreateSdfResolutionAttr().Set(int(sdf_resolution))

    # friction material so slopes are climbable (terrain has ~1.3 m of grade)
    mat_path = xform_path + "/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys_mat.CreateStaticFrictionAttr().Set(float(friction[0]))
    phys_mat.CreateDynamicFrictionAttr().Set(float(friction[1]))
    binding = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
    binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

    return {
        "xform_path": xform_path,
        "mesh_prim": str(mesh_prim.GetPath()),
        "collision_usd": usd_path,
        "approximation": approximation,
        "sdf_resolution": int(sdf_resolution) if approximation == "sdf" else None,
        "friction_static_dynamic": [float(friction[0]), float(friction[1])],
    }


def compute_world_aabb(stage, prim_path):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
        ignoreVisibility=True,  # collision prim is invisible on purpose
    )
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if r.IsEmpty():
        return None
    return [list(r.GetMin()), list(r.GetMax())]


def load_splat(stage, simulation_app, usdz_path, xform_path="/World/AriaSplat"):
    """Reference the NuRec USDZ, resolve it, and run the E1 frame checks."""
    from pxr import Usd, UsdGeom, Gf

    prim = stage.DefinePrim(xform_path, "Xform")
    prim.GetReferences().AddReference(usdz_path)
    for _ in range(60):
        simulation_app.update()

    found = [(str(p.GetPath()), str(p.GetTypeName())) for p in Usd.PrimRange(prim)]
    nurec = [t for _, t in found if "NuRec" in t]
    volume_identity = None
    for p in Usd.PrimRange(prim):
        if p.GetTypeName() == "Volume":
            m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            volume_identity = Gf.IsClose(m, Gf.Matrix4d(1.0), 1e-9)
            break
    return {
        "xform_path": xform_path,
        "usdz": usdz_path,
        "nurec_prim_count": len(nurec),
        "volume_frame_is_identity": volume_identity,
        "world_aabb": compute_world_aabb(stage, xform_path),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision", required=True, help="collision mesh (.ply or .usd)")
    parser.add_argument(
        "--approximation", default="none", choices=APPROXIMATIONS,
        help="physics:approximation for the collision mesh "
        "(none = static triangle mesh / trimesh)",
    )
    parser.add_argument("--sdf-resolution", type=int, default=256)
    parser.add_argument(
        "--splat", default=None,
        help=f"NuRec USDZ to load under /World/AriaSplat (default: none; pass "
        f"'default' for {DEFAULT_SPLAT})",
    )
    parser.add_argument("--bench", action="store_true", help="run the physics benchmark")
    parser.add_argument("--bench-steps", type=int, default=300)
    parser.add_argument("--bench-warmup", type=int, default=20)
    parser.add_argument(
        "--box-pos", type=float, nargs=3, default=[-1.167, 1.552, -1.30],
        help="drop-box start (near trajectory start; floor there is ~-1.83 m)",
    )
    parser.add_argument("--report", required=True, help="write results JSON here")
    parser.add_argument("--force-convert", action="store_true")
    args = parser.parse_args()

    report = {
        "ok": False,
        "task": "isaacsim_build_env",
        "collision_input": os.path.abspath(args.collision),
        "approximation": args.approximation,
        "argv": sys.argv[1:],
    }

    def write_report():
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)

    write_report()

    from isaacsim import SimulationApp  # noqa: E402

    t_app0 = time.perf_counter()
    simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    report["app_startup_s"] = round(time.perf_counter() - t_app0, 2)

    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid

        # convert / resolve collision USD
        t0 = time.perf_counter()
        usd_path, conv_info = resolve_collision_usd(
            args.collision, force=args.force_convert
        )
        report["convert_s"] = round(time.perf_counter() - t0, 3)
        report["collision_usd"] = usd_path
        report["convert_info"] = conv_info  # None on cache hit / direct USD

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()

        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0,
                      rendering_dt=1.0 / 60.0)

        t0 = time.perf_counter()
        col_info = setup_collision_prim(
            stage, usd_path,
            approximation=args.approximation,
            sdf_resolution=args.sdf_resolution,
        )
        report["stage_setup_s"] = round(time.perf_counter() - t0, 3)
        report["collision"] = col_info
        report["collision_world_aabb"] = compute_world_aabb(
            stage, col_info["xform_path"]
        )

        if args.splat:
            splat_path = DEFAULT_SPLAT if args.splat == "default" else args.splat
            report["splat"] = load_splat(stage, simulation_app, splat_path)
            # overlap consistency: collision AABB should sit inside the splat AABB
            s = report["splat"]["world_aabb"]
            c = report["collision_world_aabb"]
            if s and c:
                inside = all(
                    s[0][k] <= c[0][k] and c[1][k] <= s[1][k] for k in range(3)
                )
                report["collision_inside_splat_aabb"] = bool(inside)
                report["aabb_center_offset_m"] = [
                    round((c[0][k] + c[1][k]) / 2 - (s[0][k] + s[1][k]) / 2, 3)
                    for k in range(3)
                ]
        write_report()

        if args.bench:
            box = DynamicCuboid(
                prim_path="/World/BenchBox", name="bench_box",
                position=np.array(args.box_pos), size=0.3,
            )
            world.scene.add(box)

            t0 = time.perf_counter()
            world.reset()  # physics init + collision cooking happens here
            report["bench_reset_s"] = round(time.perf_counter() - t0, 3)

            t0 = time.perf_counter()
            world.step(render=False)
            report["bench_first_step_s"] = round(time.perf_counter() - t0, 3)

            for _ in range(args.bench_warmup):
                world.step(render=False)

            times = []
            for _ in range(args.bench_steps):
                t0 = time.perf_counter()
                world.step(render=False)
                times.append(time.perf_counter() - t0)
            times_ms = np.array(times) * 1e3
            report["bench_steps"] = args.bench_steps
            report["step_ms_mean"] = round(float(times_ms.mean()), 3)
            report["step_ms_median"] = round(float(np.median(times_ms)), 3)
            report["step_ms_p95"] = round(float(np.percentile(times_ms, 95)), 3)
            report["step_ms_max"] = round(float(times_ms.max()), 3)

            pos, _ = box.get_world_pose()
            report["box_final_pos"] = [round(float(v), 4) for v in pos]
            # floor under the default box XY is ~-1.83 m; a 0.3 box resting on it
            # has center ~-1.68. Grossly higher means the approximation filled in
            # the terrain (expected failure mode of convexDecomposition).
            report["box_final_z"] = round(float(pos[2]), 4)
            world.stop()

        report["ok"] = True
        write_report()
    except Exception as e:  # noqa: BLE001
        import traceback

        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        write_report()
    finally:
        write_report()
        simulation_app.close()


if __name__ == "__main__":
    main()
