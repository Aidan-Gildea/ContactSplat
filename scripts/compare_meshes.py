#!/usr/bin/env python
"""Run scripts/eval_mesh.py metrics over several candidate meshes, print a table.

The MPS-derived context (trajectory grid + filtered semi-dense cloud) is built
once and shared, so adding meshes to the comparison is cheap.

    /home/sun/miniforge3/envs/ego_splats/bin/python scripts/compare_meshes.py \
        meshA.ply meshB.ply meshC.ply \
        --mps-dir /home/sun/aria/mps_Outside_20260812_141244_vrs/slam \
        --output comparison.json

All eval_mesh.py grid/tolerance flags are accepted and forwarded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mesh import DEFAULT_MPS_DIR, EvalContext, evaluate_mesh  # noqa: E402


COLUMNS = [
    # (header, width, extractor)
    ("mesh",        28, lambda s: Path(s["mesh"]).name),
    ("cover",        7, lambda s: _fmt(s["floor"].get("coverage"), "{:.3f}")),
    ("hole m2",      8, lambda s: _fmt(s["floor"].get("largest_hole_m2"), "{:.2f}")),
    ("sd-mean",      8, lambda s: _fmt(s["semidense"].get("mean_m"), "{:.3f}")),
    ("sd-med",       7, lambda s: _fmt(s["semidense"].get("median_m"), "{:.3f}")),
    ("sd-p95",       7, lambda s: _fmt(s["semidense"].get("p95_m"), "{:.3f}")),
    ("<10cm",        6, lambda s: _fmt(s["semidense"].get("frac_within_10cm"), "{:.2f}")),
    ("tris",         9, lambda s: _fmt(s["hygiene"].get("n_triangles"), "{:d}")),
    ("comps",        6, lambda s: _fmt(s["hygiene"].get("n_components"), "{:d}")),
    ("nonmanif",     8, lambda s: _fmt(s["hygiene"].get("non_manifold_edges"), "{:d}")),
    ("junk m2",      7, lambda s: _fmt(s["hygiene"].get("small_component_area_m2"), "{:.3f}")),
]


def _fmt(v, spec):
    if v is None:
        return "-"
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def print_table(scorecards: list[dict]) -> None:
    header = "  ".join(f"{name:>{w}}" for name, w, _ in COLUMNS)
    print("\n" + header)
    print("-" * len(header))
    for sc in scorecards:
        cells = []
        for name, w, fn in COLUMNS:
            try:
                cells.append(f"{fn(sc):>{w}}")
            except Exception:
                cells.append(f"{'ERR':>{w}}")
        print("  ".join(cells))
    print()
    print("cover    : fraction of buffered-trajectory floor cells hit within z-tol")
    print("hole m2  : largest connected missing floor region")
    print("sd-*     : point-to-mesh distance vs filtered MPS semi-dense cloud [m]")
    print("<10cm    : fraction of semi-dense points within 10 cm of the mesh")
    print("junk m2  : total area of connected components smaller than 0.1 m2")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("meshes", type=Path, nargs="+", help="mesh files (PLY/OBJ)")
    ap.add_argument("--mps-dir", type=Path, default=Path(DEFAULT_MPS_DIR))
    ap.add_argument("--output", type=Path, default=None,
                    help="write all scorecards + table inputs to this JSON")
    ap.add_argument("--cell", type=float, default=0.10)
    ap.add_argument("--buffer", type=float, default=0.75)
    ap.add_argument("--eye-height", type=float, default=1.60)
    ap.add_argument("--z-tol", type=float, default=0.15)
    ap.add_argument("--ray-height", type=float, default=0.50)
    ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--no-semidense", action="store_true")
    args = ap.parse_args()

    missing = [m for m in args.meshes if not m.exists()]
    if missing:
        for m in missing:
            print(f"error: mesh not found: {m}", file=sys.stderr)
        return 1

    ctx = EvalContext(
        mps_dir=args.mps_dir, cell=args.cell, buffer=args.buffer,
        eye_height=args.eye_height, z_tol=args.z_tol, ray_height=args.ray_height,
        max_points=args.max_points, skip_semidense=args.no_semidense,
    ).build()

    scorecards = []
    for mesh_path in args.meshes:
        print(f"[eval] {mesh_path}")
        scorecards.append(evaluate_mesh(mesh_path, ctx))

    print_table(scorecards)

    if args.output:
        args.output.write_text(json.dumps(
            {"params": ctx.params_dict(), "scorecards": scorecards}, indent=2
        ))
        print(f"full scorecards written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
