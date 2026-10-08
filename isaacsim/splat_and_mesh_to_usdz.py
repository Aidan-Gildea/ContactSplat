#!/usr/bin/env python
"""Package a trained Gaussian splat and a collision mesh into ONE .usdz for NVIDIA Isaac Sim.

    python isaacsim/splat_and_mesh_to_usdz.py \
        --splat output/<SCENE>/<RECT>/point_cloud/iteration_30000/point_cloud.ply \
        --mesh  output/photogrammetry/<SCENE>/<RECT>/mesh_delaunay_decimated.ply \
        --out   output/<SCENE>/<RECT>/isaacsim/<SCENE>_with_collider.usdz

Both inputs are in the MPS world frame (gravity-aligned, Z-up, metres), so nothing is moved:
the splat and the collider line up by construction.

What it does, in order
  1. read the splat PLY (positions, wxyz quaternions, log scales, opacity logits, SH colour)
  2. drop stray Gaussians farther than --radius (50 m) from the median position
  3. write the splat as a NuRec volume (what Isaac Sim's RTX renderer draws), identity transform
  4. read the mesh PLY, drop triangles with an edge longer than --max-edge (2 m), and write it
     as an invisible static triangle-mesh collider
  5. pack everything into one USDZ (root layer first, 64-byte aligned by Sdf.ZipFileWriter)

USDZ contents
  default.usda   root layer: /World (Xform, defaultPrim), over /World/gauss -> @gauss.usda@,
                 def /World/collider -> @./collider.usdc@, customLayerData["renderSettings"]
  splat.nurec    gzip (level 0) of msgpack(NuRec "3dgut-nrend" template), arrays as float16
  gauss.usda     /World/gauss UsdVol.Volume with omni:nurec:* attributes, xformOp = identity
  collider.usdc  /collider UsdGeom.Mesh, invisible, PhysicsCollisionAPI + MeshCollisionAPI(none)

Needs numpy, plyfile, msgpack and usd-core (pxr); the 3dgrut conda env has all four. No GPU,
no torch and no 3dgrut checkout are needed.

This does in one step what isaacsim/export_isaacsim_usdz.sh did with four:
scripts/filter_splat_outliers.py (step 2), 3dgrut's threedgrut/export/scripts/ply_to_usd.py
(step 3), scripts/fix_nurec_usdz_frame.py (step 3, transform) and isaacsim/add_mesh_collider.py
(steps 4-5). 3dgrut's ply_to_usd.py needs a GPU only because it builds a 3DGUT renderer it never
uses (model.py MixtureOfGaussians); the USDZ writing itself is numpy and pxr.

--compare OLD.usdz checks the new file against one made by the old four-step path: zip entries,
layer metadata, every composed prim, attribute and relationship (arrays by SHA-256), and the
decoded NuRec payload. Expected output: "identical (apart from the .nurec file name)".

Portions of steps 3 and 5 (the NuRec payload template, the volume layer and the root layer) are
adapted from NVIDIA 3dgrut, https://github.com/nv-tlabs/3dgrut at commit
70604495bfbebd5add0d3ffdc8a57bc14a63a94f, files threedgrut/export/nurec_templates.py,
threedgrut/export/usd_util.py and threedgrut/export/usdz_exporter.py.
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. Licensed under the Apache License,
Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0); distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
Changes from 3dgrut: the PLY is read with plyfile instead of through a torch model; the volume
transform is identity instead of 3dgrut's baked ((-1,0,0),(0,0,-1),(0,-1,0)) conversion
(usd_util.py:141-152), which lays an MPS-frame scene on its side; renderer parameters are
constants copied from 3dgrut's default config instead of read through Hydra; the gzip header
timestamp is fixed to 0, so every file inside the USDZ is byte-identical from run to run (only the
zip entries' modification times differ).
"""

import argparse
import gzip
import hashlib
import io
import os
import sys
import tempfile
import time
import zipfile

import msgpack
import numpy as np
from plyfile import PlyData
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils, UsdVol, Vt

NUREC_FILE = "splat.nurec"  # the old path named it after its temporary splat.usdz
GAUSS_FILE = "gauss.usda"
ROOT_FILE = "default.usda"
COLLIDER_NAME = "collider"
COLLIDER_FILE = "collider.usdc"
SH_DEGREE = 3  # NuRec radiance_sph_degree; our splats store 45 f_rest values (degree 3)

# Renderer parameters that 3dgrut's ply_to_usd.py writes into the .nurec payload. They come from
# its default Hydra config apps/colmap_3dgut.yaml (render/3dgut.yaml over render/3dgrt.yaml over
# base_gs.yaml) at commit 70604495, not from fill_3dgut_template's own defaults. Keep the Python
# types: msgpack encodes 1.0 and 1 differently.
NUREC_PARAMS = {
    "density_activation": "sigmoid",  # base_gs.yaml model.density_activation
    "scale_activation": "exp",  # base_gs.yaml model.scale_activation
    "rotation_activation": "normalize",  # usdz_exporter.py:94
    "density_kernel_degree": 2,  # render/3dgut.yaml:8
    "density_kernel_density_clamping": True,  # render/3dgrt.yaml:6
    "density_kernel_min_response": 0.0113,  # render/3dgut.yaml:9
    "radiance_sph_degree": 3,  # render/3dgrt.yaml:10
    "transmittance_threshold": 0.0001,  # render/3dgut.yaml:10
    "global_z_order": True,  # render/3dgut.yaml:26
    "n_rolling_shutter_iterations": 5,  # render/3dgut.yaml:18
    "ut_alpha": 1.0,  # render/3dgut.yaml:19
    "ut_beta": 2.0,  # render/3dgut.yaml:20
    "ut_kappa": 0.0,  # render/3dgut.yaml:21
    "ut_require_all_sigma_points": False,  # render/3dgut.yaml:23
    "image_margin_factor": 0.1,  # render/3dgut.yaml:22
    "rect_bounding": True,  # render/3dgut.yaml:14
    "tight_opacity_bounding": True,  # render/3dgut.yaml:15
    "tile_based_culling": True,  # render/3dgut.yaml:16
    "k_buffer_size": 0,  # render/3dgut.yaml:25
}

# Isaac Sim RTX settings, stored in the root layer's customLayerData (3dgrut usd_util.py:123-134).
# Isaac Sim applies them when the USDZ is opened directly, not when it is referenced into a stage.
RENDER_SETTINGS = {
    "rtx:rendermode": "RaytracedLighting",
    "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
    "rtx:post:histogram:enabled": False,
    "rtx:post:registeredCompositing:invertToneMap": True,
    "rtx:post:registeredCompositing:invertColorCorrection": True,
    "rtx:material:enableRefraction": False,
    "rtx:post:tonemap:op": 2,
    "rtx:raytracing:fractionalCutoutOpacity": False,
    "rtx:matteObject:visibility:secondaryRays": True,
}


# --- 1. splat PLY ------------------------------------------------------------------------------
def read_splat_ply(path):
    """Return the splat as float32 arrays in NuRec's layout (pre-activation values).

    PLY layout (model/vanilla_gsplat.py save_ply): x y z, nx ny nz, f_dc_0..2, f_rest_0..44 stored
    channel-major (all 15 red coefficients, then green, then blue), opacity (logit),
    scale_0..2 (log), rot_0..3 (w x y z, not normalised), plus a one-row 'metadata' element.
    NuRec wants f_rest coefficient-major (r,g,b of coefficient 1, then coefficient 2, ...), the
    same reshuffle as 3dgrut model.py:705-706.
    """
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = [p.name for p in v.properties]
    if "metadata" in [e.name for e in ply.elements]:
        fmt = "".join(chr(c) for c in ply["metadata"]["color_format"][0])
        if fmt != "rgb":
            sys.exit(f"{path}: colour format '{fmt}'; only 'rgb' splats are supported")

    def stack(prefix):
        cols = sorted((n for n in names if n.startswith(prefix)), key=lambda n: int(n.split("_")[-1]))
        return np.column_stack([np.asarray(v[c], dtype=np.float32) for c in cols]) if cols else None

    n = len(v.data)
    n_coeffs = (SH_DEGREE + 1) ** 2 - 1  # 15
    rest = stack("f_rest_")
    if rest is None:  # DC-only splat: zeros, as 3dgrut does (model.py:707-709)
        specular = np.zeros((n, 3 * n_coeffs), np.float32)
    elif rest.shape[1] == 3 * n_coeffs:
        specular = rest.reshape(n, 3, n_coeffs).transpose(0, 2, 1).reshape(n, 3 * n_coeffs)
    else:
        sys.exit(f"{path}: {rest.shape[1]} f_rest values, expected {3 * n_coeffs} or 0")

    return {
        "positions": np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32),
        "rotations": stack("rot"),  # 3dgrut matches the 'rot' prefix (model.py:722)
        "scales": stack("scale_"),
        "densities": np.asarray(v["opacity"], dtype=np.float32)[:, None],
        "features_albedo": np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).astype(np.float32),
        "features_specular": np.ascontiguousarray(specular),
    }


# --- 2. stray Gaussians --------------------------------------------------------------------------
def drop_far_gaussians(splat, radius):
    """Keep Gaussians within `radius` m of the median position.

    MCMC densification leaves a few hundred nearly transparent Gaussians kilometres away. They are
    invisible but would set the asset's bounding box and the NuRec crop bounds. The median, not
    the origin, is the centre because the MPS world origin sits wherever tracking started.
    """
    xyz = splat["positions"]
    centre = np.median(xyz, axis=0)
    keep = np.linalg.norm(xyz - centre, axis=1) <= radius
    dropped = int((~keep).sum())
    print(f"splat: kept {int(keep.sum()):,}/{keep.size:,} Gaussians, dropped {dropped:,} beyond "
          f"{radius} m of the median position {np.round(centre, 3).tolist()}")
    if dropped:
        opacity = 1.0 / (1.0 + np.exp(-splat["densities"][~keep, 0]))
        print(f"       dropped Gaussians' median opacity {np.median(opacity):.4f}")
    return {k: a[keep] for k, a in splat.items()}


# --- 3. NuRec payload and volume layer -----------------------------------------------------------
def nurec_payload(splat):
    """gzip(msgpack(template)) as 3dgrut nurec_templates.py:101-251 and usdz_exporter.py:124-128.

    The dict is written in the template's key order, so the msgpack bytes match 3dgrut's.
    """
    p = NUREC_PARAMS
    f16 = {k: a.astype(np.float16) for k, a in splat.items()}  # nurec_templates.py:68-77
    extra_signal = np.zeros((f16["positions"].shape[0], 0), dtype=np.float16)  # :80
    g = ".gaussians_nodes.gaussians."
    template = {
        "nre_data": {
            "version": "0.2.576",
            "model": "nre",
            "config": {
                "layers": {
                    "gaussians": {
                        "name": "sh-gaussians",
                        "device": "cuda",
                        "density_activation": p["density_activation"],
                        "scale_activation": p["scale_activation"],
                        "rotation_activation": p["rotation_activation"],
                        "precision": 16,
                        "particle": {
                            "density_kernel_planar": False,
                            "density_kernel_degree": p["density_kernel_degree"],
                            "density_kernel_density_clamping": p["density_kernel_density_clamping"],
                            "density_kernel_min_response": p["density_kernel_min_response"],
                            "radiance_sph_degree": p["radiance_sph_degree"],
                        },
                        "transmittance_threshold": p["transmittance_threshold"],
                    }
                },
                "renderer": {
                    "name": "3dgut-nrend",
                    "log_level": 3,
                    "force_update": False,
                    "update_step_train_batch_end": False,
                    "per_ray_features": False,
                    "global_z_order": p["global_z_order"],
                    "projection": {
                        "n_rolling_shutter_iterations": p["n_rolling_shutter_iterations"],
                        "ut_dim": 3,
                        "ut_alpha": p["ut_alpha"],
                        "ut_beta": p["ut_beta"],
                        "ut_kappa": p["ut_kappa"],
                        "ut_require_all_sigma_points": p["ut_require_all_sigma_points"],
                        "image_margin_factor": p["image_margin_factor"],
                        "min_projected_ray_radius": 0.5477225575051661,
                    },
                    "culling": {
                        "rect_bounding": p["rect_bounding"],
                        "tight_opacity_bounding": p["tight_opacity_bounding"],
                        "tile_based": p["tile_based_culling"],
                        "near_clip_distance": 0.2,
                        "far_clip_distance": 3.402823466e38,
                    },
                    "render": {"mode": "kbuffer", "k_buffer_size": p["k_buffer_size"]},
                },
                "name": "gaussians_primitive",
                "appearance_embedding": {"name": "skip-appearance", "embedding_dim": 0, "device": "cuda"},
                "background": {"name": "skip-background", "device": "cuda", "composite_in_linear_space": False},
            },
            "state_dict": {
                "._extra_state": {"obj_track_ids": {"gaussians": []}},
                g + "positions": f16["positions"].tobytes(),
                g + "rotations": f16["rotations"].tobytes(),
                g + "scales": f16["scales"].tobytes(),
                g + "densities": f16["densities"].tobytes(),
                g + "extra_signal": extra_signal.tobytes(),
                g + "features_albedo": f16["features_albedo"].tobytes(),
                g + "features_specular": f16["features_specular"].tobytes(),
                # 3dgrut sets n_active_features to the max SH degree after loading a PLY (model.py:739)
                g + "n_active_features": np.array([SH_DEGREE], dtype=np.int64).tobytes(),
                g + "positions.shape": list(f16["positions"].shape),
                g + "rotations.shape": list(f16["rotations"].shape),
                g + "scales.shape": list(f16["scales"].shape),
                g + "densities.shape": list(f16["densities"].shape),
                g + "extra_signal.shape": list(extra_signal.shape),
                g + "features_albedo.shape": list(f16["features_albedo"].shape),
                g + "features_specular.shape": list(f16["features_specular"].shape),
                g + "n_active_features.shape": [],
            },
        }
    }
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0, mtime=0) as f:
        f.write(msgpack.packb(template))
    return buffer.getvalue()


def new_stage():
    """Z-up, metres, /World Xform as defaultPrim (3dgrut usd_util.py:50-66)."""
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1)
    stage.SetMetadata("upAxis", "Z")
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetMetadata("defaultPrim", "World")
    return stage


def write_gauss_layer(path, positions):
    """NuRec volume /World/gauss (3dgrut usd_util.py:91-208), with an identity transform."""
    lo = [float(x) for x in positions.min(axis=0)]
    hi = [float(x) for x in positions.max(axis=0)]
    print(f"splat bounds (m): {np.round(lo, 2).tolist()} .. {np.round(hi, 2).tolist()}")

    stage = new_stage()
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", RENDER_SETTINGS)
    volume = UsdVol.Volume.Define(stage, "/World/gauss")
    prim = volume.GetPrim()
    # The MPS frame already is the stage frame (Z-up, metres), so no rotation.
    volume.AddTransformOp().Set(Gf.Matrix4d(1.0))
    prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    fields = {}
    for name, prim_name, data_type in (("density", "density_field", "float"),
                                       ("emissiveColor", "emissive_color_field", "float3")):
        field = stage.DefinePrim(f"/World/gauss/{prim_name}", "OmniNuRecFieldAsset")
        volume.CreateFieldRelationship(name, field.GetPath())
        field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set("./" + NUREC_FILE)
        field.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set(name)
        field.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set(data_type)
        field.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set(name)
        fields[name] = field
    for row, attr in enumerate(("omni:nurec:ccmR", "omni:nurec:ccmG", "omni:nurec:ccmB")):
        ccm = [0.0, 0.0, 0.0, 0.0]
        ccm[row] = 1.0  # identity colour-correction matrix
        fields["emissiveColor"].CreateAttribute(attr, Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(ccm))

    prim.GetAttribute("extent").Set([lo, hi])
    prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*lo))
    prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*hi))
    prim.CreateRelationship("proxy")
    stage.GetRootLayer().Export(path)


# --- 4. collider --------------------------------------------------------------------------------
def face_index_property(path):
    """Name of the face index list: COLMAP writes 'vertex_index', Open3D 'vertex_indices'."""
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("property list") and line.split()[-1] in ("vertex_index", "vertex_indices"):
                return line.split()[-1]
            if line == "end_header":
                break
    sys.exit(f"{path}: no face vertex_index / vertex_indices property")


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
    return points, tris.astype(np.int64).reshape(-1, 3)


def drop_long_triangles(points, tris, max_edge):
    """Drop triangles with an edge longer than max_edge, then the vertices no longer used.

    COLMAP's Delaunay mesher joins some surface points to far-away noise and sky with long thin
    triangles. As an invisible collider these would be invisible walls and ceilings. On the
    Outside_20260812_141244 MVS mesh this removes 0.17% of triangles, all far from or high above
    the walked path, and leaves floor coverage along the path unchanged (0.880).
    """
    p = points[tris]  # (M, 3 corners, 3 xyz)
    longest = np.max(np.stack([np.linalg.norm(p[:, i] - p[:, (i + 1) % 3], axis=1) for i in range(3)], 1), axis=1)
    keep = longest <= max_edge
    used, remapped = np.unique(tris[keep], return_inverse=True)
    return points[used], remapped.reshape(-1, 3).astype(np.int32), int((~keep).sum())


def write_collider_layer(path, points, tris):
    """Invisible static triangle-mesh collider, defaultPrim /collider."""
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
    mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)  # the splat does all the rendering
    # Static collider (no RigidBodyAPI). Approximation "none" = the exact triangle mesh,
    # which PhysX supports for static geometry.
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    stage.GetRootLayer().Save()


# --- 5. root layer and package ------------------------------------------------------------------
def write_root_layer(path):
    """default.usda: references the splat and the collider (3dgrut usd_util.py:231-257)."""
    stage = new_stage()
    delegate = UsdUtils.CoalescingDiagnosticDelegate()  # silences 'unresolved reference' in memory
    stage.OverridePrim("/World/gauss").GetReferences().AddReference(GAUSS_FILE)
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", dict(RENDER_SETTINGS))
    layer = stage.GetRootLayer()
    spec = Sdf.PrimSpec(layer.GetPrimAtPath("/World"), COLLIDER_NAME, Sdf.SpecifierDef)
    spec.referenceList.Prepend(Sdf.Reference(f"./{COLLIDER_FILE}"))
    layer.Export(path)
    del delegate


def package_usdz(out_path, work_dir):
    """Root layer first (USDZ rule), then payload, splat layer, collider; 64-byte aligned."""
    tmp = out_path + ".partial"
    writer = Sdf.ZipFileWriter.CreateNew(tmp)
    for name in (ROOT_FILE, NUREC_FILE, GAUSS_FILE, COLLIDER_FILE):
        writer.AddFile(os.path.join(work_dir, name), name)
    writer.Save()
    os.replace(tmp, out_path)


# --- optional: check against a USDZ made by the old four-step path ------------------------------
def _digest(value):
    if isinstance(value, (bytes, bytearray)):
        return hashlib.sha256(value).hexdigest()
    if isinstance(value, dict):
        return {k: _digest(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_digest(v) for v in value]
    return value


def _attr_value(attr):
    value = attr.Get()
    if hasattr(value, "__len__") and not isinstance(value, str) and type(value).__name__.endswith("Array"):
        a = np.asarray(value)
        return (a.dtype.str, a.shape, hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    return value


def _describe(usdz):
    """Everything that should match, as plain Python values."""
    with zipfile.ZipFile(usdz) as z:
        names = z.namelist()
        nurec = [n for n in names if n.endswith(".nurec")]
        payload = msgpack.unpackb(gzip.decompress(z.read(nurec[0])), raw=False) if nurec else None
    stage = Usd.Stage.Open(usdz)
    root = stage.GetRootLayer()
    prims = {}
    for prim in stage.TraverseAll():
        prims[str(prim.GetPath())] = {
            "type": str(prim.GetTypeName()),
            "specifier": str(prim.GetSpecifier()),
            "apis": list(prim.GetAppliedSchemas()),
            "attrs": {a.GetName(): (str(a.GetTypeName()), _attr_value(a)) for a in prim.GetAttributes()
                      if a.HasAuthoredValue()},
            "rels": {r.GetName(): [str(t) for t in r.GetTargets()] for r in prim.GetRelationships()},
        }
    return {
        "zip_entries": [n if not n.endswith(".nurec") else "*.nurec" for n in names],
        "root_layer": {"upAxis": UsdGeom.GetStageUpAxis(stage), "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage),
                       "defaultPrim": root.defaultPrim, "customLayerData": dict(root.customLayerData)},
        "prims": prims,
        "nurec": _digest(payload),
    }


def compare_usdz(new, old):
    a, b = _describe(new), _describe(old)
    problems = []

    def walk(x, y, where):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y), key=str):
                if k not in x or k not in y:
                    problems.append(f"{where}/{k}: only in {'old' if k in y else 'new'}")
                else:
                    walk(x[k], y[k], f"{where}/{k}")
        elif x != y:
            problems.append(f"{where}: new={x!r} old={y!r}")

    # The one intended difference: the old path named the payload after its temporary file.
    for p in (a, b):
        for prim in p["prims"].values():
            if "filePath" in prim["attrs"]:
                t, _ = prim["attrs"]["filePath"]
                prim["attrs"]["filePath"] = (t, "<nurec>")
    walk(a, b, "")
    for line in problems:
        print("DIFF", line)
    print("compare:", "identical (apart from the .nurec file name)" if not problems else f"{len(problems)} differences")
    return not problems


# --- main ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splat", required=True, help="trained splat PLY (point_cloud.ply)")
    ap.add_argument("--mesh", required=True, help="triangle mesh PLY in the same frame")
    ap.add_argument("--out", required=True, help="USDZ to write")
    ap.add_argument("--radius", type=float, default=50.0, help="keep Gaussians within this many m of the median")
    ap.add_argument("--max-edge", type=float, default=2.0, help="drop collider triangles with a longer edge (m)")
    ap.add_argument("--compare", metavar="OLD_USDZ", help="check the result against a USDZ from the old path")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as work:
        splat = drop_far_gaussians(read_splat_ply(args.splat), args.radius)
        with open(os.path.join(work, NUREC_FILE), "wb") as f:
            f.write(nurec_payload(splat))
        write_gauss_layer(os.path.join(work, GAUSS_FILE), splat["positions"])

        points, tris = read_triangle_mesh(args.mesh)
        n_in = len(tris)
        points, tris, n_dropped = drop_long_triangles(points, tris, args.max_edge)
        print(f"mesh: {n_in:,} triangles in, {n_dropped:,} longer than {args.max_edge} m dropped "
              f"({100.0 * n_dropped / max(n_in, 1):.2f}%), {len(tris):,} kept, {len(points):,} vertices")
        write_collider_layer(os.path.join(work, COLLIDER_FILE), points, tris)

        write_root_layer(os.path.join(work, ROOT_FILE))
        package_usdz(args.out, work)

    print(f"wrote {args.out} in {time.time() - t0:.1f} s")
    print("  /World/gauss      splat, visible")
    print(f"  /World/{COLLIDER_NAME}   mesh, invisible static collider")
    if args.compare and not compare_usdz(args.out, args.compare):
        sys.exit(1)


if __name__ == "__main__":
    main()
