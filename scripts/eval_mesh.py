#!/usr/bin/env python
"""Evaluate a candidate collision mesh against MPS ground-truth signals.

Several pipelines (TSDF-from-splat-depth, stereo depth fusion, COLMAP MVS,
trajectory floor) produce candidate collision meshes for the same Aria scene.
This script scores one mesh objectively so the candidates can be ranked.

All inputs are expected in the MPS world frame: gravity-aligned, Z-up, metres.

Metrics
-------
1. Floor coverage: project the MPS closed-loop trajectory (1 kHz, ground truth
   for where a human walked) to the XY plane, buffer it laterally (default
   0.75 m robot clearance), rasterize to a grid (default 10 cm). For each cell
   raycast straight down from `--ray-height` (default 0.5 m) above the expected
   floor (local trajectory Z minus eye height, default 1.6 m). A cell counts as
   covered when the hit lands within +/- `--z-tol` (default 15 cm) of expected.
2. Largest hole: connected components (8-connectivity) of the *missing* cells;
   report the area of the largest one in m^2. A robot does not care about many
   small holes; it cares about the one it falls through.
3. Agreement with the MPS semi-dense point cloud: load
   `semidense_points.csv.gz` with projectaria_tools and filter it with the same
   confidence thresholds `scripts/extract_aria_vrs.py` uses
   (inverse_distance_std <= 0.005, distance_std <= 0.01), then report
   mean / median / p95 unsigned point-to-mesh distance. This is an independent
   geometry source, the closest thing to ground truth available.
4. Mesh hygiene: triangle count, bounding box, non-manifold edge count, number
   of connected components, and total area of components smaller than 0.1 m^2
   (floater junk that wrecks collision performance). Topology metrics are
   computed after an exact duplicate-vertex merge so triangle-soup exports
   (one vertex per corner) are judged on their real connectivity.

Output: a JSON scorecard (`--output`, default `<mesh>.scorecard.json`) plus a
human-readable summary on stdout.

Run with the ego_splats env, CPU only:

    /home/sun/miniforge3/envs/ego_splats/bin/python scripts/eval_mesh.py \
        candidate.ply --mps-dir /home/sun/aria/mps_Outside_20260812_141244_vrs/slam

`scripts/compare_meshes.py` runs this over several meshes and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_MPS_DIR = "/home/sun/aria/mps_Outside_20260812_141244_vrs/slam"

# Same thresholds as scripts/extract_aria_vrs.py (filter_points_from_confidence)
SEMIDENSE_INV_DIST_STD = 0.005
SEMIDENSE_DIST_STD = 0.01


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectory_xyz(mps_dir: Path) -> np.ndarray:
    """Device positions (N,3) in the MPS world frame from closed_loop_trajectory.csv."""
    csv_path = Path(mps_dir) / "closed_loop_trajectory.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"missing {csv_path}")
    import pandas as pd

    df = pd.read_csv(
        csv_path, usecols=["tx_world_device", "ty_world_device", "tz_world_device"]
    )
    return df.to_numpy(dtype=np.float64)


def load_semidense_points(mps_dir: Path) -> tuple[np.ndarray, int]:
    """Filtered semi-dense world points (M,3); also returns the raw count.

    Uses projectaria_tools and the exact confidence thresholds used in
    scripts/extract_aria_vrs.py so this is consistent with what the splat was
    initialised from.
    """
    from projectaria_tools.core import mps as aria_mps
    from projectaria_tools.core.mps.utils import filter_points_from_confidence

    pts_path = Path(mps_dir) / "semidense_points.csv.gz"
    if not pts_path.exists():
        raise FileNotFoundError(f"missing {pts_path}")
    raw = aria_mps.read_global_point_cloud(str(pts_path))
    filtered = filter_points_from_confidence(
        raw, SEMIDENSE_INV_DIST_STD, SEMIDENSE_DIST_STD
    )
    positions = np.asarray([p.position_world for p in filtered], dtype=np.float64)
    return positions, len(raw)


# ---------------------------------------------------------------------------
# Evaluation context: everything derived from MPS only (mesh-independent),
# built once and reused across meshes by compare_meshes.py.
# ---------------------------------------------------------------------------

@dataclass
class EvalContext:
    mps_dir: Path
    cell: float = 0.10          # grid cell size [m]
    buffer: float = 0.75        # lateral buffer around the walked path [m]
    eye_height: float = 1.60    # device height above the walkable floor [m]
    z_tol: float = 0.15         # accepted |hit - expected floor| [m]
    ray_height: float = 0.50    # ray origin height above expected floor [m]
    max_points: int = 0         # subsample semi-dense cloud (0 = use all)
    skip_semidense: bool = False

    # filled by build()
    traj_xyz: np.ndarray = field(default=None, repr=False)
    footprint: np.ndarray = field(default=None, repr=False)      # (nx,ny) bool
    expected_floor: np.ndarray = field(default=None, repr=False) # (nx,ny) float
    origin_xy: tuple = None
    semidense: np.ndarray = field(default=None, repr=False)
    n_semidense_raw: int = 0

    def build(self) -> "EvalContext":
        from scipy import ndimage

        self.mps_dir = Path(self.mps_dir)
        t0 = time.time()
        self.traj_xyz = load_trajectory_xyz(self.mps_dir)
        xy = self.traj_xyz[:, :2]
        margin = self.buffer + 2 * self.cell
        xmin, ymin = xy.min(axis=0) - margin
        xmax, ymax = xy.max(axis=0) + margin
        nx = int(np.ceil((xmax - xmin) / self.cell)) + 1
        ny = int(np.ceil((ymax - ymin) / self.cell)) + 1
        self.origin_xy = (float(xmin), float(ymin))

        ix = np.clip(((xy[:, 0] - xmin) / self.cell).astype(np.int64), 0, nx - 1)
        iy = np.clip(((xy[:, 1] - ymin) / self.cell).astype(np.int64), 0, ny - 1)

        occupied = np.zeros((nx, ny), dtype=bool)
        occupied[ix, iy] = True

        # Per-cell mean device Z (at 1 kHz the samples inside a 10 cm cell are
        # dense; mean vs median is indistinguishable here and mean vectorises).
        z_sum = np.zeros((nx, ny), dtype=np.float64)
        z_cnt = np.zeros((nx, ny), dtype=np.int64)
        np.add.at(z_sum, (ix, iy), self.traj_xyz[:, 2])
        np.add.at(z_cnt, (ix, iy), 1)
        with np.errstate(invalid="ignore"):
            z_mean = np.where(z_cnt > 0, z_sum / np.maximum(z_cnt, 1), np.nan)

        # Buffer the path: cells within `buffer` of any occupied cell. The
        # distance transform also hands us the nearest occupied cell, which is
        # how each buffered cell inherits its local trajectory height.
        dist, (near_i, near_j) = ndimage.distance_transform_edt(
            ~occupied, return_indices=True
        )
        self.footprint = (dist * self.cell) <= self.buffer
        self.expected_floor = z_mean[near_i, near_j] - self.eye_height

        n_cells = int(self.footprint.sum())
        print(
            f"[ctx] trajectory {len(self.traj_xyz)} poses, grid {nx}x{ny} "
            f"@ {self.cell} m, footprint {n_cells} cells "
            f"({n_cells * self.cell**2:.1f} m^2)  [{time.time()-t0:.1f}s]"
        )

        if not self.skip_semidense:
            t0 = time.time()
            self.semidense, self.n_semidense_raw = load_semidense_points(self.mps_dir)
            if self.max_points and len(self.semidense) > self.max_points:
                rng = np.random.default_rng(0)
                sel = rng.choice(len(self.semidense), self.max_points, replace=False)
                self.semidense = self.semidense[sel]
            print(
                f"[ctx] semi-dense points: {self.n_semidense_raw} raw -> "
                f"{len(self.semidense)} used  [{time.time()-t0:.1f}s]"
            )
        return self

    def params_dict(self) -> dict:
        return {
            "cell_m": self.cell,
            "buffer_m": self.buffer,
            "eye_height_m": self.eye_height,
            "z_tol_m": self.z_tol,
            "ray_height_m": self.ray_height,
            "semidense_inverse_distance_std_threshold": SEMIDENSE_INV_DIST_STD,
            "semidense_distance_std_threshold": SEMIDENSE_DIST_STD,
            "semidense_max_points": self.max_points,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _raycasting_scene(mesh):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def floor_metrics(scene, ctx: EvalContext) -> dict:
    """Metrics 1 + 2: floor coverage and hole analysis via downward raycasts."""
    import open3d as o3d
    from scipy import ndimage

    fp_i, fp_j = np.nonzero(ctx.footprint)
    n_cells = len(fp_i)
    if n_cells == 0:
        return {"error": "empty trajectory footprint"}

    xmin, ymin = ctx.origin_xy
    cx = xmin + (fp_i + 0.5) * ctx.cell
    cy = ymin + (fp_j + 0.5) * ctx.cell
    expected = ctx.expected_floor[fp_i, fp_j]
    oz = expected + ctx.ray_height

    rays = np.zeros((n_cells, 6), dtype=np.float32)
    rays[:, 0] = cx
    rays[:, 1] = cy
    rays[:, 2] = oz
    rays[:, 5] = -1.0  # straight down
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = ans["t_hit"].numpy().astype(np.float64)

    hit = np.isfinite(t_hit)
    hit_z = np.where(hit, oz - t_hit, np.nan)
    ok = hit & (np.abs(hit_z - expected) <= ctx.z_tol)

    # Holes: connected components of NOT-ok cells inside the footprint.
    ok_grid = np.zeros_like(ctx.footprint)
    ok_grid[fp_i[ok], fp_j[ok]] = True
    missing = ctx.footprint & ~ok_grid
    labels, n_holes = ndimage.label(missing, structure=np.ones((3, 3), dtype=int))
    cell_area = ctx.cell ** 2
    if n_holes > 0:
        hole_sizes = np.bincount(labels.ravel())[1:]  # skip background
        largest_hole_m2 = float(hole_sizes.max() * cell_area)
    else:
        largest_hole_m2 = 0.0

    hit_err = np.abs(hit_z[ok] - expected[ok])
    return {
        "n_cells": int(n_cells),
        "footprint_area_m2": round(n_cells * cell_area, 3),
        "n_cells_hit": int(hit.sum()),
        "n_cells_hit_within_tol": int(ok.sum()),
        "coverage": round(float(ok.sum() / n_cells), 4),
        "n_holes": int(n_holes),
        "largest_hole_m2": round(largest_hole_m2, 3),
        "total_missing_area_m2": round(float(missing.sum() * cell_area), 3),
        "hit_abs_error_mean_m": round(float(hit_err.mean()), 4) if ok.any() else None,
    }


def semidense_metrics(scene, ctx: EvalContext) -> dict:
    """Metric 3: unsigned point-to-mesh distance of the filtered MPS cloud."""
    import open3d as o3d

    pts = ctx.semidense
    if pts is None or len(pts) == 0:
        return {"error": "no semi-dense points loaded"}
    q = o3d.core.Tensor(pts.astype(np.float32))
    d = scene.compute_distance(q).numpy().astype(np.float64)
    return {
        "n_points_raw": int(ctx.n_semidense_raw),
        "n_points_used": int(len(pts)),
        "mean_m": round(float(d.mean()), 4),
        "median_m": round(float(np.median(d)), 4),
        "p95_m": round(float(np.percentile(d, 95)), 4),
        "max_m": round(float(d.max()), 4),
        "frac_within_5cm": round(float((d <= 0.05).mean()), 4),
        "frac_within_10cm": round(float((d <= 0.10).mean()), 4),
        "frac_within_25cm": round(float((d <= 0.25).mean()), 4),
    }


def hygiene_metrics(mesh) -> dict:
    """Metric 4: mesh hygiene. Topology after exact duplicate-vertex merge."""
    n_tri_raw = len(mesh.triangles)
    n_vert_raw = len(mesh.vertices)
    out = {
        "n_vertices": int(n_vert_raw),
        "n_triangles": int(n_tri_raw),
    }
    if n_tri_raw == 0:
        out["error"] = "mesh has no triangles"
        return out

    bbox = mesh.get_axis_aligned_bounding_box()
    out["bbox_min_m"] = [round(float(v), 3) for v in bbox.min_bound]
    out["bbox_max_m"] = [round(float(v), 3) for v in bbox.max_bound]
    out["total_area_m2"] = round(float(mesh.get_surface_area()), 3)

    # Merge exactly-duplicated vertices so triangle-soup exports are judged on
    # real connectivity, not on the accident of per-corner vertex duplication.
    topo = type(mesh)(mesh)  # copy
    topo.remove_duplicated_vertices()
    out["topology_after_exact_vertex_merge"] = True
    out["non_manifold_edges"] = int(
        len(topo.get_non_manifold_edges(allow_boundary_edges=True))
    )
    _, cluster_n_tri, cluster_areas = topo.cluster_connected_triangles()
    areas = np.asarray(cluster_areas, dtype=np.float64)
    small = areas < 0.1
    out["n_components"] = int(len(areas))
    out["n_small_components"] = int(small.sum())            # area < 0.1 m^2
    out["small_component_area_m2"] = round(float(areas[small].sum()), 4)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evaluate_mesh(mesh_path: Path, ctx: EvalContext) -> dict:
    import open3d as o3d

    mesh_path = Path(mesh_path)
    t0 = time.time()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    t_load = time.time() - t0

    scorecard = {
        "mesh": str(mesh_path),
        "mps_dir": str(ctx.mps_dir),
        "params": ctx.params_dict(),
    }

    t0 = time.time()
    scorecard["hygiene"] = hygiene_metrics(mesh)
    t_hyg = time.time() - t0

    if scorecard["hygiene"].get("error"):
        scorecard["floor"] = {"error": "mesh has no triangles"}
        scorecard["semidense"] = {"error": "mesh has no triangles"}
        return scorecard

    scene = _raycasting_scene(mesh)

    t0 = time.time()
    scorecard["floor"] = floor_metrics(scene, ctx)
    t_floor = time.time() - t0

    t_sd = 0.0
    if ctx.skip_semidense:
        scorecard["semidense"] = {"skipped": True}
    else:
        t0 = time.time()
        scorecard["semidense"] = semidense_metrics(scene, ctx)
        t_sd = time.time() - t0

    scorecard["timing_s"] = {
        "load_mesh": round(t_load, 2),
        "hygiene": round(t_hyg, 2),
        "floor_raycast": round(t_floor, 2),
        "semidense_distance": round(t_sd, 2),
    }
    return scorecard


def print_summary(sc: dict) -> None:
    f, s, h = sc.get("floor", {}), sc.get("semidense", {}), sc.get("hygiene", {})
    print(f"\n=== {sc['mesh']} ===")
    if "error" in h:
        print(f"  hygiene ERROR: {h['error']}")
        return
    print(
        f"  floor coverage : {f.get('coverage')}  "
        f"({f.get('n_cells_hit_within_tol')}/{f.get('n_cells')} cells, "
        f"footprint {f.get('footprint_area_m2')} m^2)"
    )
    print(
        f"  largest hole   : {f.get('largest_hole_m2')} m^2  "
        f"({f.get('n_holes')} holes, {f.get('total_missing_area_m2')} m^2 missing)"
    )
    if s.get("skipped"):
        print("  semi-dense     : skipped")
    elif "error" in s:
        print(f"  semi-dense     : ERROR {s['error']}")
    else:
        print(
            f"  semi-dense d   : mean {s['mean_m']} m, median {s['median_m']} m, "
            f"p95 {s['p95_m']} m  ({s['n_points_used']} pts)"
        )
    print(
        f"  hygiene        : {h['n_triangles']} tris, "
        f"{h.get('n_components')} components, "
        f"{h.get('non_manifold_edges')} non-manifold edges, "
        f"small(<0.1 m^2) area {h.get('small_component_area_m2')} m^2"
    )
    print(
        f"  bbox           : {h.get('bbox_min_m')} .. {h.get('bbox_max_m')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mesh", type=Path, help="mesh file (PLY/OBJ), MPS world frame")
    ap.add_argument("--mps-dir", type=Path, default=Path(DEFAULT_MPS_DIR),
                    help=f"MPS slam directory (default {DEFAULT_MPS_DIR})")
    ap.add_argument("--output", type=Path, default=None,
                    help="scorecard JSON path (default <mesh>.scorecard.json)")
    ap.add_argument("--cell", type=float, default=0.10, help="grid cell [m]")
    ap.add_argument("--buffer", type=float, default=0.75,
                    help="lateral buffer around trajectory [m]")
    ap.add_argument("--eye-height", type=float, default=1.60,
                    help="assumed device height above floor [m]")
    ap.add_argument("--z-tol", type=float, default=0.15,
                    help="accepted |hit - expected floor| [m]")
    ap.add_argument("--ray-height", type=float, default=0.50,
                    help="ray origin height above expected floor [m]")
    ap.add_argument("--max-points", type=int, default=0,
                    help="subsample semi-dense cloud to N points (0 = all)")
    ap.add_argument("--no-semidense", action="store_true",
                    help="skip the semi-dense distance metric (fast runs)")
    args = ap.parse_args()

    if not args.mesh.exists():
        print(f"error: mesh not found: {args.mesh}", file=sys.stderr)
        return 1

    ctx = EvalContext(
        mps_dir=args.mps_dir, cell=args.cell, buffer=args.buffer,
        eye_height=args.eye_height, z_tol=args.z_tol, ray_height=args.ray_height,
        max_points=args.max_points, skip_semidense=args.no_semidense,
    ).build()

    scorecard = evaluate_mesh(args.mesh, ctx)
    out = args.output or args.mesh.with_suffix(args.mesh.suffix + ".scorecard.json")
    out.write_text(json.dumps(scorecard, indent=2))
    print_summary(scorecard)
    print(f"\nscorecard written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
