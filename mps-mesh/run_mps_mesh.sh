#!/bin/bash
# Mesh from the MPS semi-dense point cloud instead of patch-match MVS. The points and the
# SLAM-camera views that observed them are written as a COLMAP dense workspace, then the
# same `colmap delaunay_mesher` (default options) and the same evaluator as
# photogrammetry/run_photogrammetry.sh are run, so the two meshes differ only in their points.
#
#   conda activate ego_splats
#   bash mps-mesh/run_mps_mesh.sh <rectified_dir> [out_dir]
#
#   <rectified_dir>  processed/<recording>/camera-rgb-rectified-<focal>-h<height>
#                    (its semidense_points.csv.gz symlink locates the MPS slam/ folder)
#   [out_dir]        default <repo>/output/mps-mesh/<recording>/<rectified folder>
#
# Stages (each skipped when its output already exists; FORCE=1 reruns everything):
#   prepare             mps-mesh/mps_points_to_colmap.py  -> dense/fused.ply + fused.ply.vis + dense/sparse
#   mesh                colmap delaunay_mesher            -> mesh_delaunay.ply
#   evaluate            photogrammetry/evaluate_mesh.py   -> report.json / report.md
#   decimate            photogrammetry/decimate_mesh.py   -> mesh_delaunay_decimated.ply
#   evaluate_decimated  photogrammetry/evaluate_mesh.py   -> report_decimated.json / .md
#
# Knobs (environment variables, all optional):
#   EYE_HEIGHT=1.6683   trajectory-to-floor distance used by the evaluator
#   DECIMATE=1          also write the simplified copy that isaacsim/export_isaacsim_usdz.sh
#                       uses as the collider (0 = full mesh only). On by default here, unlike
#                       run_photogrammetry.sh, because this is the default collider route.
#   FORCE=0             1 = ignore existing outputs and redo every stage
#   PYTHON=python       interpreter with numpy, pandas, open3d (ego_splats env)

set -euo pipefail

RECT_DIR="${1:?usage: run_mps_mesh.sh <rectified_dir> [out_dir]}"
RECT_DIR="$(cd "$RECT_DIR" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

RECT_NAME="$(basename "$RECT_DIR")"
SCENE_NAME="$(basename "$(dirname "$RECT_DIR")")"
OUT_DIR="${2:-$REPO_ROOT/output/mps-mesh/$SCENE_NAME/$RECT_NAME}"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/.done"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

EYE_HEIGHT="${EYE_HEIGHT:-1.6683}"
DECIMATE="${DECIMATE:-1}"
FORCE="${FORCE:-0}"
PYTHON="${PYTHON:-python}"

DENSE="$OUT_DIR/dense"
MESH="$OUT_DIR/mesh_delaunay.ply"
MESH_DECIMATED="$OUT_DIR/mesh_delaunay_decimated.ply"

# --- preflight ---------------------------------------------------------------------------
[ -e "$RECT_DIR/semidense_points.csv.gz" ] || { echo "no semidense_points.csv.gz in $RECT_DIR" >&2; exit 1; }
[ -f "$REPO_ROOT/photogrammetry/evaluate_mesh.py" ] || { echo "photogrammetry/evaluate_mesh.py not found" >&2; exit 1; }
command -v colmap >/dev/null || { echo "colmap not on PATH" >&2; exit 1; }
"$PYTHON" -c "import numpy, pandas, open3d" 2>/dev/null \
    || { echo "PYTHON=$PYTHON lacks numpy/pandas/open3d; conda activate ego_splats" >&2; exit 1; }

echo "rectified : $RECT_DIR"
echo "output    : $OUT_DIR"
echo

# --- stage helpers -----------------------------------------------------------------------
done_marker() { echo "$OUT_DIR/.done/$1"; }
is_done()     { [ "$FORCE" != "1" ] && [ -f "$(done_marker "$1")" ]; }
mark_done()   { date -Is > "$(done_marker "$1")"; }

run_stage() {   # run_stage <name> <command...>
    local name="$1"; shift
    if is_done "$name"; then
        echo "== $name: already done ($(cat "$(done_marker "$name")")), skipping"
        return 0
    fi
    echo "== $name: $(date -Is)"
    local t0=$SECONDS
    "$@" 2>&1 | tee "$OUT_DIR/logs/$name.log"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "== $name FAILED (exit $rc), see $OUT_DIR/logs/$name.log" >&2
        exit "$rc"
    fi
    mark_done "$name"
    echo "== $name: done in $(( SECONDS - t0 )) s"
    echo
}

# --- 1. prepare --------------------------------------------------------------------------
run_stage prepare "$PYTHON" "$HERE/mps_points_to_colmap.py" \
    --rectified_dir "$RECT_DIR" --out_dir "$OUT_DIR"

# --- 2. mesh -----------------------------------------------------------------------------
run_stage mesh colmap delaunay_mesher \
    --input_path "$DENSE" \
    --input_type dense \
    --output_path "$MESH"

# --- 3. evaluate -------------------------------------------------------------------------
# --fused points at the MPS cloud, so the report's "fused" rows describe the mesher's input.
run_stage evaluate "$PYTHON" "$REPO_ROOT/photogrammetry/evaluate_mesh.py" \
    --mesh "$MESH" \
    --rectified_dir "$RECT_DIR" \
    --fused "$DENSE/fused.ply" \
    --eye_height "$EYE_HEIGHT" \
    --out "$OUT_DIR/report.json"

# --- 4. decimate (default on) ------------------------------------------------------------
# Same simplification as run_photogrammetry.sh DECIMATE=1: drop >2 m spike triangles, then
# quadric-error decimation at about 1 cm. On the 12 recordings this kept 4-17% of the
# triangles with floor coverage within 0.011 of the full mesh.
if [ "$DECIMATE" = "1" ]; then
    run_stage decimate "$PYTHON" "$REPO_ROOT/photogrammetry/decimate_mesh.py" "$MESH" "$MESH_DECIMATED"
    run_stage evaluate_decimated "$PYTHON" "$REPO_ROOT/photogrammetry/evaluate_mesh.py" \
        --mesh "$MESH_DECIMATED" \
        --rectified_dir "$RECT_DIR" \
        --fused "$DENSE/fused.ply" \
        --eye_height "$EYE_HEIGHT" \
        --out "$OUT_DIR/report_decimated.json"
fi

echo "Mesh (MPS world frame, metres, Z-up):"
echo "  $MESH"
echo "Report:"
echo "  $OUT_DIR/report.md"
if [ "$DECIMATE" = "1" ]; then
    echo "Simplified mesh, used as the Isaac Sim collider:"
    echo "  $MESH_DECIMATED"
    echo "  $OUT_DIR/report_decimated.md"
fi
