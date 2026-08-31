#!/usr/bin/env python
"""Compose a near-field mesh with the far field of a second mesh (task E6a).

Used to build the FUSED-BEST collision candidate input: the Gen 2 stereo TSDF
mesh is the best geometry inside its own XY extent (voxel-accurate floor, clean
topology) but is blind beyond ~4 m; the 2DGS TSDF mesh has far-field
walls/structure the stereo mesh lacks. This script keeps the near mesh intact
and adds only the far mesh's triangles whose centroids fall OUTSIDE the near
mesh's XY axis-aligned bounding box, after removing small floater components
(< --junk-area, default 0.1 m^2 -- the E0 harness's junk definition) from that
cropped far-field piece. The near mesh is never modified.

The output is a *pre-fusion* composite: pipe it through
    scripts/floor_from_trajectory.py --eye-height 1.6683 --fuse-with <output>
to pin the floor height and fill the remaining floor holes with the
trajectory prior (see docs/experiments/E6_candidates_report.md).

Both inputs must be in the MPS world frame (Z-up, metres); the composite
inherits it. CPU only, ego_splats env.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--near", type=Path, required=True,
                    help="near-field mesh, kept intact (authoritative inside its XY AABB)")
    ap.add_argument("--far", type=Path, required=True,
                    help="far-field mesh; only triangles outside the near XY AABB are kept")
    ap.add_argument("--output", type=Path, required=True, help="composite mesh PLY")
    ap.add_argument("--meta", type=Path, default=None,
                    help="composition metadata JSON (default: <output>.meta.json)")
    ap.add_argument("--junk-area", type=float, default=0.1,
                    help="remove far-field components smaller than this [m^2] (0 = keep all)")
    args = ap.parse_args()
    meta_path = args.meta or args.output.with_suffix(args.output.suffix + ".meta.json")

    near = o3d.io.read_triangle_mesh(str(args.near))
    far = o3d.io.read_triangle_mesh(str(args.far))
    if len(near.triangles) == 0 or len(far.triangles) == 0:
        raise SystemExit("empty input mesh")

    nb = near.get_axis_aligned_bounding_box()
    min_xy = np.asarray(nb.min_bound)[:2]
    max_xy = np.asarray(nb.max_bound)[:2]

    v = np.asarray(far.vertices)
    t = np.asarray(far.triangles)
    c = v[t].mean(axis=1)
    inside = ((c[:, 0] >= min_xy[0]) & (c[:, 0] <= max_xy[0]) &
              (c[:, 1] >= min_xy[1]) & (c[:, 1] <= max_xy[1]))
    n_inside = int(inside.sum())
    far.remove_triangles_by_mask(inside)
    far.remove_unreferenced_vertices()

    n_junk_comps = n_junk_tris = 0
    junk_area = 0.0
    if args.junk_area > 0 and len(far.triangles):
        labels, _, areas = far.cluster_connected_triangles()
        labels = np.asarray(labels)
        areas = np.asarray(areas)
        junk = areas[labels] < args.junk_area
        n_junk_comps = int((areas < args.junk_area).sum())
        n_junk_tris = int(junk.sum())
        junk_area = float(areas[areas < args.junk_area].sum())
        far.remove_triangles_by_mask(junk)
        far.remove_unreferenced_vertices()

    combined = near + far
    args.output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.output), combined)

    meta = {
        "near_mesh": str(args.near),
        "far_mesh": str(args.far),
        "near_xy_aabb_min": min_xy.tolist(),
        "near_xy_aabb_max": max_xy.tolist(),
        "near_triangles": len(near.triangles),
        "far_triangles_total": int(len(t)),
        "far_triangles_inside_near_aabb_dropped": n_inside,
        "far_junk_components_removed": n_junk_comps,
        "far_junk_triangles_removed": n_junk_tris,
        "far_junk_area_removed_m2": round(junk_area, 3),
        "far_triangles_kept": len(far.triangles),
        "combined_triangles": len(combined.triangles),
        "junk_area_threshold_m2": args.junk_area,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
