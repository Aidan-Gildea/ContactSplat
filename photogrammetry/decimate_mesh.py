#!/usr/bin/env python3
"""
Simplify the Delaunay mesh for simulation use, keeping it within about 1 cm of the original.

Two steps, in this order:

1. Drop spike triangles: any triangle with an edge longer than MAX_EDGE_M. COLMAP's Delaunay
   mesher connects real surfaces to far-away noise and sky with long thin triangles (0.17% of
   triangles on Outside_20260812_141244, all far from or high above the walked path). They must
   go first: simplifying with them present lets flat ground merge into triangles over 2 m long,
   which a later spike filter would then delete, punching holes in the floor.

2. Quadric error metric decimation (Garland & Heckbert, SIGGRAPH 1997), via Open3D. Vertices
   are merged in order of least added error, and merging stops once the next merge would exceed
   TOLERANCE_M. The quadric error is a sum of squared distances to the original surface's
   planes, so the limit is approximate, not a hard bound on every point. Flat ground collapses
   to few large triangles while edges and small objects keep their detail, which is why this
   method suits a collider: the robot's contact surface stays where it was.

Measured on Outside_20260812_141244 (2.48 M triangles):
    tolerance 0.5 cm -> 18% of triangles kept, 1 cm -> 13%, 2 cm -> 9%, 5 cm -> 5%
    at 1 cm the typical (median) shift near the walked path is 0.8 cm, and floor coverage is
    unchanged (0.879 -> 0.887); the mesh's own error against MPS points is 1.5 cm.

    python photogrammetry/decimate_mesh.py mesh_delaunay.ply mesh_delaunay_decimated.ply
"""

import argparse
import time

import numpy as np
import open3d as o3d

MAX_EDGE_M = 2.0     # spike triangles: longest edge above this is removed before simplifying
TOLERANCE_M = 0.01   # stop merging once the added error would exceed about 1 cm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh_in", help="full mesh, e.g. mesh_delaunay.ply")
    ap.add_argument("mesh_out", help="simplified mesh to write")
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.mesh_in)
    n_in = len(mesh.triangles)
    if n_in == 0:
        raise SystemExit(f"{args.mesh_in} holds no triangles")

    # 1. spikes
    v = np.asarray(mesh.vertices)
    p = v[np.asarray(mesh.triangles)]
    longest = np.max(
        np.stack([np.linalg.norm(p[:, i] - p[:, (i + 1) % 3], axis=1) for i in range(3)], 1), axis=1
    )
    spikes = longest > MAX_EDGE_M
    mesh.remove_triangles_by_mask(spikes)
    mesh.remove_unreferenced_vertices()

    # 2. quadric decimation, bounded by error rather than by a triangle count
    t0 = time.time()
    simple = mesh.simplify_quadric_decimation(target_number_of_triangles=1, maximum_error=TOLERANCE_M ** 2)
    simple.remove_unreferenced_vertices()
    dt = time.time() - t0

    if not o3d.io.write_triangle_mesh(args.mesh_out, simple, write_ascii=False):
        raise SystemExit(f"failed to write {args.mesh_out}")

    n_out = len(simple.triangles)
    print(f"input     : {n_in:,} triangles")
    print(f"spikes    : {int(spikes.sum()):,} triangles with an edge > {MAX_EDGE_M} m removed")
    print(f"simplified: {n_out:,} triangles ({100.0 * n_out / n_in:.1f}% of input), "
          f"tolerance about {TOLERANCE_M * 100:.0f} cm, {dt:.0f} s")
    print(f"wrote {args.mesh_out}")


if __name__ == "__main__":
    main()
