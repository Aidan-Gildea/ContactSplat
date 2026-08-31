#!/usr/bin/env python
"""Derive a walkable-floor surface from the MPS closed-loop trajectory (task E5).

The closed-loop trajectory is a dense (1 kHz) metric record of where a human
head was, in a gravity-aligned Z-up world frame. A head is a roughly constant
height above the walkable floor, so

    floor_z(x, y) ~= device_z(x, y) - h        (h ~ eye height, ~1.5-1.7 m)

This floor has no holes and needs no visual observations: it covers exactly the
region a robot must drive, it is a traversability guarantee (a human physically
walked there), and it follows ramps and grade changes automatically, unlike a
single fitted plane.

Honest limitation: a trajectory is a *curve*, not a surface. It gives floor
height along the walked path (laterally dilated by a clearance radius), not
across the whole scene. Cells away from the path are inferred and flagged as
such. For anything beyond the walked corridor, use --fuse-with to combine this
prior with a mesh from an observation-based pipeline.

Outputs (all in the MPS world frame, Z-up, metres):
  floor_mesh.ply          triangle mesh of the floor corridor
  floor_heightfield.npz   height (nx,ny) float32, NaN outside footprint;
                          category (nx,ny) uint8: 0=outside 1=observed
                          2=interpolated 3=extrapolated; origin_xy; cell_size
  floor_heightfield.json  metadata sidecar (grid convention, parameters, stats)

Eye height h defaults to 1.60 m (assumed). --calibrate-with <mesh> fits h
against floor geometry recovered by another pipeline instead — see
calibrate_eye_height(). Fitted value and spread are reported; a large spread
means gait/posture varied and the prior is weaker than it looks.

--fuse-with <mesh> uses the trajectory floor as a *prior* over another
pipeline's mesh: estimate and remove the mesh's global floor-height bias (pin
absolute floor height), reject mesh floor that disagrees with the prior by
more than --fuse-threshold (default 10 cm), and fill holes with prior heights.
Emits fused_floor_mesh.ply / fused_heightfield.npz plus fused_full_mesh.ply
(the input mesh, bias-corrected, with rejected floor triangles stripped, plus
the fused floor patch).

Run in the ego_splats env, CPU only:

    /home/sun/miniforge3/envs/ego_splats/bin/python \
        scripts/floor_from_trajectory.py --output-dir output/.../floor
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_MPS_DIR = "/home/sun/aria/mps_Outside_20260812_141244_vrs/slam"
DEFAULT_OUTPUT_DIR = "output/Outside_20260812_141244/floor_from_trajectory"

CAT_OUTSIDE = 0
CAT_OBSERVED = 1
CAT_INTERPOLATED = 2
CAT_EXTRAPOLATED = 3
CAT_NAMES = {0: "outside", 1: "observed", 2: "interpolated", 3: "extrapolated"}

# Fused-cell provenance (fused_heightfield.npz "source" array)
FUSE_OUTSIDE = 0
FUSE_MESH = 1        # mesh floor agreed with prior -> mesh height kept
FUSE_HOLE_FILLED = 2  # mesh had no floor hit -> prior height
FUSE_REJECTED = 3    # mesh floor disagreed > threshold -> prior height


# ---------------------------------------------------------------------------
# Trajectory loading (projectaria_tools, per the brief)
# ---------------------------------------------------------------------------

def load_trajectory(mps_dir: Path) -> np.ndarray:
    """Device positions (N,3) in the MPS world frame, via projectaria_tools."""
    from projectaria_tools.core import mps as aria_mps

    csv_path = Path(mps_dir) / "closed_loop_trajectory.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path}")
    traj = aria_mps.read_closed_loop_trajectory(str(csv_path))
    xyz = np.empty((len(traj), 3), dtype=np.float64)
    for i, pose in enumerate(traj):
        xyz[i] = np.asarray(pose.transform_world_device.translation()).reshape(3)
    return xyz


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

class FloorGrid:
    """Regular XY grid holding the derived floor heightfield.

    Convention (matches scripts/eval_mesh.py): arrays are indexed [ix, iy];
    cell (ix, iy) has its center at
        x = origin_xy[0] + (ix + 0.5) * cell
        y = origin_xy[1] + (iy + 0.5) * cell
    """

    def __init__(self, height, category, origin_xy, cell, eye_height):
        self.height = height          # (nx,ny) float32, NaN outside footprint
        self.category = category      # (nx,ny) uint8
        self.origin_xy = origin_xy    # (xmin, ymin) of grid corner
        self.cell = cell
        self.eye_height = eye_height

    @property
    def footprint(self):
        return self.category != CAT_OUTSIDE

    def cell_centers(self, ix, iy):
        x = self.origin_xy[0] + (np.asarray(ix) + 0.5) * self.cell
        y = self.origin_xy[1] + (np.asarray(iy) + 0.5) * self.cell
        return x, y

    def stats(self) -> dict:
        fp = self.footprint
        cats, counts = np.unique(self.category, return_counts=True)
        return {
            "grid_shape": list(self.height.shape),
            "origin_xy": [float(v) for v in self.origin_xy],
            "cell_m": self.cell,
            "eye_height_m": self.eye_height,
            "footprint_cells": int(fp.sum()),
            "footprint_area_m2": round(float(fp.sum()) * self.cell**2, 3),
            "cells_by_category": {
                CAT_NAMES[int(c)]: int(n) for c, n in zip(cats, counts)
            },
            "floor_z_min_m": round(float(np.nanmin(self.height)), 3),
            "floor_z_max_m": round(float(np.nanmax(self.height)), 3),
        }


def build_floor_grid(
    traj_xyz: np.ndarray,
    eye_height: float = 1.60,
    cell: float = 0.10,
    clearance: float = 0.75,
    smooth_sigma: float = 1.0,
) -> FloorGrid:
    """Trajectory -> floor heightfield.

    1. Rasterize trajectory XY to `cell`-sized cells; per-cell robust
       (median) device Z -> observed floor = median_z - eye_height.
    2. Dilate laterally to `clearance` (robot half-width) around the path.
    3. Fill non-observed footprint cells: linear interpolation between passes
       where the cell is inside the convex hull of observed cells
       (flag: interpolated), nearest-neighbour otherwise (flag: extrapolated
       -- mostly the lateral dilation ring).
    4. Optional mild Gaussian smoothing of the height (footprint-masked),
       to suppress residual gait bounce. Categories are never smoothed.
    """
    import pandas as pd
    from scipy import ndimage
    from scipy.interpolate import griddata

    xy = traj_xyz[:, :2]
    margin = clearance + 2 * cell
    xmin, ymin = xy.min(axis=0) - margin
    xmax, ymax = xy.max(axis=0) + margin
    nx = int(np.ceil((xmax - xmin) / cell)) + 1
    ny = int(np.ceil((ymax - ymin) / cell)) + 1

    ix = np.clip(((xy[:, 0] - xmin) / cell).astype(np.int64), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - ymin) / cell).astype(np.int64), 0, ny - 1)

    # Robust per-cell Z: median of the 1 kHz samples that fall in the cell.
    flat = ix * ny + iy
    med = pd.Series(traj_xyz[:, 2]).groupby(flat).median()
    obs_flat = med.index.to_numpy()
    obs_ix, obs_iy = obs_flat // ny, obs_flat % ny

    observed = np.zeros((nx, ny), dtype=bool)
    observed[obs_ix, obs_iy] = True
    height = np.full((nx, ny), np.nan, dtype=np.float64)
    height[obs_ix, obs_iy] = med.to_numpy() - eye_height

    # Lateral dilation to the robot clearance radius.
    dist, (ni, nj) = ndimage.distance_transform_edt(~observed, return_indices=True)
    footprint = (dist * cell) <= clearance

    # Fill non-observed footprint cells.
    category = np.zeros((nx, ny), dtype=np.uint8)
    category[observed] = CAT_OBSERVED
    fill = footprint & ~observed
    fi, fj = np.nonzero(fill)
    if len(fi):
        interp = griddata(
            np.column_stack([obs_ix, obs_iy]).astype(np.float64),
            height[obs_ix, obs_iy],
            np.column_stack([fi, fj]).astype(np.float64),
            method="linear",
        )
        got = np.isfinite(interp)
        height[fi[got], fj[got]] = interp[got]
        category[fi[got], fj[got]] = CAT_INTERPOLATED
        # Outside the convex hull of observations: nearest observed cell.
        miss = ~got
        height[fi[miss], fj[miss]] = height[ni[fi[miss], fj[miss]],
                                            nj[fi[miss], fj[miss]]]
        category[fi[miss], fj[miss]] = CAT_EXTRAPOLATED

    # Masked Gaussian smoothing (normalized convolution) inside the footprint.
    if smooth_sigma > 0:
        h0 = np.where(footprint, height, 0.0)
        w = footprint.astype(np.float64)
        num = ndimage.gaussian_filter(h0, smooth_sigma)
        den = ndimage.gaussian_filter(w, smooth_sigma)
        with np.errstate(invalid="ignore"):
            sm = num / den
        height = np.where(footprint, sm, np.nan)
    else:
        height = np.where(footprint, height, np.nan)

    return FloorGrid(height.astype(np.float32), category,
                     (float(xmin), float(ymin)), cell, eye_height)


# ---------------------------------------------------------------------------
# Heightfield -> triangle mesh
# ---------------------------------------------------------------------------

def grid_to_mesh(grid: FloorGrid, height=None, footprint=None):
    """Build an Open3D mesh whose quads fully cover every footprint cell.

    Vertices sit on cell *corners* (node grid), so a ray cast at any footprint
    cell center hits strictly inside a quad -- no boundary-edge ambiguity.
    Node height = mean of the adjacent footprint cells' heights.
    """
    import open3d as o3d

    H = grid.height if height is None else height
    fp = grid.footprint if footprint is None else footprint
    nx, ny = fp.shape

    # Node (corner) grid is (nx+1, ny+1); node (i,j) touches cells
    # (i-1..i, j-1..j). Accumulate adjacent cell heights.
    num = np.zeros((nx + 1, ny + 1), dtype=np.float64)
    cnt = np.zeros((nx + 1, ny + 1), dtype=np.int64)
    ci, cj = np.nonzero(fp)
    hv = H[ci, cj].astype(np.float64)
    for di in (0, 1):
        for dj in (0, 1):
            np.add.at(num, (ci + di, cj + dj), hv)
            np.add.at(cnt, (ci + di, cj + dj), 1)
    node_used = cnt > 0
    node_z = np.where(node_used, num / np.maximum(cnt, 1), 0.0)

    node_idx = np.full((nx + 1, ny + 1), -1, dtype=np.int64)
    ui, uj = np.nonzero(node_used)
    node_idx[ui, uj] = np.arange(len(ui))
    verts = np.column_stack([
        grid.origin_xy[0] + ui * grid.cell,
        grid.origin_xy[1] + uj * grid.cell,
        node_z[ui, uj],
    ])

    # Two triangles per footprint cell, CCW seen from +Z (upward normals).
    a = node_idx[ci, cj]
    b = node_idx[ci + 1, cj]
    c = node_idx[ci + 1, cj + 1]
    d = node_idx[ci, cj + 1]
    tris = np.concatenate([
        np.column_stack([a, b, c]),
        np.column_stack([a, c, d]),
    ])

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(tris),
    )
    mesh.compute_vertex_normals()
    return mesh


def save_heightfield(grid: FloorGrid, npz_path: Path, extra_meta: dict = None,
                     source=None):
    npz_path = Path(npz_path)
    arrays = dict(
        height=grid.height,
        category=grid.category,
        origin_xy=np.asarray(grid.origin_xy, dtype=np.float64),
        cell_size=np.float64(grid.cell),
    )
    if source is not None:
        arrays["source"] = source
    np.savez_compressed(npz_path, **arrays)
    meta = {
        "frame": "MPS world (gravity-aligned, Z-up, metres)",
        "convention": "arrays indexed [ix,iy]; cell center at "
                      "origin_xy + (index + 0.5) * cell_size",
        "category_codes": CAT_NAMES,
        **grid.stats(),
    }
    if source is not None:
        meta["source_codes"] = {
            FUSE_OUTSIDE: "outside", FUSE_MESH: "mesh_kept",
            FUSE_HOLE_FILLED: "hole_filled_from_prior",
            FUSE_REJECTED: "rejected_replaced_by_prior",
        }
    if extra_meta:
        meta.update(extra_meta)
    npz_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------------------
# Eye-height calibration against another pipeline's recovered floor
# ---------------------------------------------------------------------------

def calibrate_eye_height(
    traj_xyz: np.ndarray,
    mesh_path: Path,
    cell: float = 0.10,
    h_range: tuple = (1.0, 2.5),
    min_cells: int = 50,
) -> dict:
    """Fit eye height h by comparing trajectory Z against floor geometry
    another pipeline actually recovered.

    For every trajectory-observed cell, cast a ray straight down from the
    per-cell median device position onto `mesh_path`; h = device_z - hit_z.
    Restricted to hits with h inside `h_range` (rejects rays that hit
    overhangs or floaters instead of floor). Returns the fitted value
    (median) and its spread -- a large spread means the wearer's gait or
    posture varied and the constant-offset prior is weaker than it looks.

    NOTE (2026-08-20): no observation-based pipeline mesh exists yet for this
    scene, so this cannot be run against real recovered floor geometry today.
    It is validated against a synthetic mesh; rerun with --calibrate-with once
    E2/E3/E4 produce meshes.
    """
    import open3d as o3d
    import pandas as pd

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"{mesh_path} has no triangles")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    # Per-cell median device position (same gridding as build_floor_grid).
    xy = traj_xyz[:, :2]
    xmin, ymin = xy.min(axis=0)
    iy_n = int(np.ceil((xy[:, 1].max() - ymin) / cell)) + 2
    flat = ((xy[:, 0] - xmin) / cell).astype(np.int64) * iy_n + \
           ((xy[:, 1] - ymin) / cell).astype(np.int64)
    df = pd.DataFrame({"f": flat, "x": traj_xyz[:, 0],
                       "y": traj_xyz[:, 1], "z": traj_xyz[:, 2]})
    med = df.groupby("f").median()

    origins = med[["x", "y", "z"]].to_numpy()
    rays = np.zeros((len(origins), 6), dtype=np.float32)
    rays[:, :3] = origins
    rays[:, 5] = -1.0
    t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()

    h_all = t_hit[np.isfinite(t_hit)].astype(np.float64)
    h = h_all[(h_all >= h_range[0]) & (h_all <= h_range[1])]
    result = {
        "mesh": str(mesh_path),
        "n_cells_total": int(len(origins)),
        "n_cells_hit": int(np.isfinite(t_hit).sum()),
        "n_cells_used": int(len(h)),
        "h_range_accepted_m": list(h_range),
    }
    if len(h) < min_cells:
        result["ok"] = False
        result["error"] = (
            f"only {len(h)} usable cells (< {min_cells}); "
            "mesh does not cover enough of the walked path to calibrate"
        )
        return result
    result.update({
        "ok": True,
        "h_fitted_m": round(float(np.median(h)), 4),
        "h_mean_m": round(float(h.mean()), 4),
        "h_std_m": round(float(h.std()), 4),
        "h_iqr_m": round(float(np.percentile(h, 75) - np.percentile(h, 25)), 4),
        "h_p5_m": round(float(np.percentile(h, 5)), 4),
        "h_p95_m": round(float(np.percentile(h, 95)), 4),
    })
    return result


# ---------------------------------------------------------------------------
# Fusion: trajectory floor as a prior over another pipeline's mesh
# ---------------------------------------------------------------------------

def fuse_with_mesh(
    grid: FloorGrid,
    mesh_path: Path,
    threshold: float = 0.10,
    ray_height: float = 0.50,
    strip_band: float = 0.40,
    strip_dilate: int = 1,
):
    """Fuse the trajectory-floor prior with another pipeline's mesh.

    1. Raycast the input mesh straight down at every footprint cell center
       from `ray_height` above the prior floor -> mesh floor height per cell.
    2. Pin absolute floor height: dz = median(prior - mesh) over hit cells;
       the whole input mesh is translated by dz (MPS-frame meshes should give
       dz ~ 0; a large dz means the pipeline had a global height bias).
    3. Classify cells: mesh kept (|mesh+dz - prior| <= threshold),
       hole filled (no hit -> prior height),
       rejected (disagreement > threshold -> prior height).
    4. Outputs:
       - fused floor grid (heights: mesh where kept, prior otherwise)
         + per-cell provenance array
       - fused floor mesh over the footprint
       - fused full mesh: input mesh translated by dz, with triangles over
         rejected cells (mask dilated by `strip_dilate` cells, so partial-cell
         skirt triangles whose centroid falls in an agreeing neighbour cell go
         too) inside the floor band (prior +/- strip_band) removed,
         concatenated with the fused floor mesh.
    """
    import open3d as o3d
    from scipy import ndimage

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"{mesh_path} has no triangles")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    fi, fj = np.nonzero(grid.footprint)
    cx, cy = grid.cell_centers(fi, fj)
    prior = grid.height[fi, fj].astype(np.float64)

    rays = np.zeros((len(fi), 6), dtype=np.float32)
    rays[:, 0] = cx
    rays[:, 1] = cy
    rays[:, 2] = prior + ray_height
    rays[:, 5] = -1.0
    t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
    hit = np.isfinite(t_hit)
    mesh_z = np.where(hit, prior + ray_height - t_hit, np.nan)

    # Pin absolute floor height: robust global offset over all hit cells.
    if hit.any():
        dz = float(np.median(prior[hit] - mesh_z[hit]))
    else:
        dz = 0.0
    mesh_z_corr = mesh_z + dz

    agree = hit & (np.abs(mesh_z_corr - prior) <= threshold)
    rejected = hit & ~agree
    hole = ~hit

    fused_h = np.where(agree, mesh_z_corr, prior)
    fused_height = np.full_like(grid.height, np.nan, dtype=np.float64)
    fused_height[fi, fj] = fused_h
    source = np.zeros_like(grid.category)
    source[fi[agree], fj[agree]] = FUSE_MESH
    source[fi[hole], fj[hole]] = FUSE_HOLE_FILLED
    source[fi[rejected], fj[rejected]] = FUSE_REJECTED

    fused_grid = FloorGrid(fused_height.astype(np.float32), grid.category.copy(),
                           grid.origin_xy, grid.cell, grid.eye_height)
    fused_floor_mesh = grid_to_mesh(fused_grid)

    # Full fused mesh: shift input by dz, strip floor-band triangles over
    # rejected cells, then add the fused floor patch.
    shifted = o3d.geometry.TriangleMesh(mesh)
    shifted.translate((0.0, 0.0, dz))
    v = np.asarray(shifted.vertices)
    t = np.asarray(shifted.triangles)
    centroids = v[t].mean(axis=1)

    nx, ny = grid.footprint.shape
    gx = ((centroids[:, 0] - grid.origin_xy[0]) / grid.cell).astype(np.int64)
    gy = ((centroids[:, 1] - grid.origin_xy[1]) / grid.cell).astype(np.int64)
    in_grid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    rej_grid = np.zeros((nx, ny), dtype=bool)
    rej_grid[fi[rejected], fj[rejected]] = True
    if strip_dilate > 0 and rej_grid.any():
        rej_grid = ndimage.binary_dilation(
            rej_grid, structure=np.ones((3, 3), dtype=bool),
            iterations=strip_dilate,
        )
    prior_full = np.where(np.isfinite(grid.height), grid.height, 0.0)

    drop = np.zeros(len(t), dtype=bool)
    ig = np.nonzero(in_grid)[0]
    over_rej = rej_grid[gx[ig], gy[ig]]
    band = np.abs(centroids[ig, 2] - prior_full[gx[ig], gy[ig]]) <= strip_band
    drop[ig[over_rej & band]] = True

    shifted.remove_triangles_by_mask(drop)
    shifted.remove_unreferenced_vertices()
    fused_full = shifted + fused_floor_mesh

    report = {
        "input_mesh": str(mesh_path),
        "threshold_m": threshold,
        "ray_height_m": ray_height,
        "strip_band_m": strip_band,
        "strip_dilate_cells": strip_dilate,
        "global_z_offset_applied_m": round(dz, 4),
        "n_footprint_cells": int(len(fi)),
        "n_cells_mesh_kept": int(agree.sum()),
        "n_cells_hole_filled": int(hole.sum()),
        "n_cells_rejected": int(rejected.sum()),
        "frac_mesh_kept": round(float(agree.mean()), 4),
        "frac_hole_filled": round(float(hole.mean()), 4),
        "frac_rejected": round(float(rejected.mean()), 4),
        "n_input_triangles": int(len(t)),
        "n_triangles_stripped": int(drop.sum()),
        "mesh_vs_prior_abs_disagreement_kept_cells_mean_m":
            round(float(np.abs(mesh_z_corr[agree] - prior[agree]).mean()), 4)
            if agree.any() else None,
    }
    return fused_grid, source, fused_floor_mesh, fused_full, report


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mps-dir", type=Path, default=Path(DEFAULT_MPS_DIR),
                    help=f"MPS slam directory (default {DEFAULT_MPS_DIR})")
    ap.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
                    help=f"output directory (default {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--eye-height", type=float, default=1.60,
                    help="assumed device height above the floor [m] "
                         "(default 1.60; overridden by --calibrate-with)")
    ap.add_argument("--cell", type=float, default=0.10, help="grid cell [m]")
    ap.add_argument("--clearance", type=float, default=0.75,
                    help="lateral dilation radius around the walked path [m]")
    ap.add_argument("--smooth-sigma", type=float, default=1.0,
                    help="Gaussian smoothing of heights, in cells (0 = off)")
    ap.add_argument("--calibrate-with", type=Path, default=None, metavar="MESH",
                    help="fit eye height against this mesh's recovered floor "
                         "and use the fitted value instead of --eye-height")
    ap.add_argument("--fuse-with", type=Path, default=None, metavar="MESH",
                    help="also fuse the floor prior with this mesh")
    ap.add_argument("--fuse-threshold", type=float, default=0.10,
                    help="max |mesh - prior| floor disagreement kept [m]")
    ap.add_argument("--fuse-strip-band", type=float, default=0.40,
                    help="floor band half-width for stripping rejected input "
                         "mesh triangles [m]")
    ap.add_argument("--fuse-strip-dilate", type=int, default=1,
                    help="dilate the rejected-cell strip mask by this many "
                         "cells (catches partial-cell skirt triangles)")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    traj = load_trajectory(args.mps_dir)
    print(f"[traj] {len(traj)} poses, "
          f"Z {traj[:, 2].min():.3f}..{traj[:, 2].max():.3f} m "
          f"[{time.time()-t0:.1f}s]")

    calib = None
    eye_height = args.eye_height
    if args.calibrate_with is not None:
        calib = calibrate_eye_height(traj, args.calibrate_with, cell=args.cell)
        print(f"[calib] {json.dumps(calib)}")
        if calib.get("ok"):
            eye_height = calib["h_fitted_m"]
            print(f"[calib] using fitted eye height {eye_height} m "
                  f"(std {calib['h_std_m']} m)")
        else:
            print(f"[calib] FAILED ({calib.get('error')}); "
                  f"falling back to assumed {eye_height} m")

    t0 = time.time()
    grid = build_floor_grid(traj, eye_height=eye_height, cell=args.cell,
                            clearance=args.clearance,
                            smooth_sigma=args.smooth_sigma)
    stats = grid.stats()
    print(f"[grid] {json.dumps(stats)}  [{time.time()-t0:.1f}s]")

    import open3d as o3d
    mesh = grid_to_mesh(grid)
    mesh_path = out / "floor_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    print(f"[mesh] {len(mesh.triangles)} triangles -> {mesh_path}")

    meta = {
        "mps_dir": str(args.mps_dir),
        "trajectory_poses": int(len(traj)),
        "smooth_sigma_cells": args.smooth_sigma,
        "clearance_m": args.clearance,
        "eye_height_source": "calibrated" if (calib and calib.get("ok"))
                             else "assumed",
    }
    if calib is not None:
        meta["calibration"] = calib
    save_heightfield(grid, out / "floor_heightfield.npz", extra_meta=meta)
    print(f"[hf]   heightfield -> {out / 'floor_heightfield.npz'} (+ .json)")

    if args.fuse_with is not None:
        t0 = time.time()
        fused_grid, source, fused_floor, fused_full, rep = fuse_with_mesh(
            grid, args.fuse_with, threshold=args.fuse_threshold,
            strip_band=args.fuse_strip_band,
            strip_dilate=args.fuse_strip_dilate,
        )
        print(f"[fuse] {json.dumps(rep)}  [{time.time()-t0:.1f}s]")
        o3d.io.write_triangle_mesh(str(out / "fused_floor_mesh.ply"), fused_floor)
        o3d.io.write_triangle_mesh(str(out / "fused_full_mesh.ply"), fused_full)
        save_heightfield(fused_grid, out / "fused_heightfield.npz",
                         extra_meta={"fusion": rep, **meta}, source=source)
        (out / "fusion_report.json").write_text(json.dumps(rep, indent=2))
        print(f"[fuse] fused_floor_mesh.ply, fused_full_mesh.ply, "
              f"fused_heightfield.npz -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
