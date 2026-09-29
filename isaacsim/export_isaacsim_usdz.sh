#!/bin/bash
# Package a trained Gaussian splat and its COLMAP mesh into ONE .usdz for Isaac Sim.
#
#   splat  -> visible   (NuRec Gaussian volume, rendered by Isaac Sim's RTX renderer)
#   mesh   -> invisible static collider (robots drive on it and bump into it)
#
# This script only converts and packages. It trains nothing and meshes nothing; both inputs
# must already exist. Both are in the MPS world frame (Z-up, metres), so they line up with
# no alignment step.
#
#   export SCENE=Outside_20260812_141244
#   bash isaacsim/export_isaacsim_usdz.sh
#
# Inputs
#   splat  output/<SCENE>/<RECT>/point_cloud/iteration_30000/point_cloud.ply
#          (from scripts/bash_local/train_gen2_outside.sh)
#   mesh   output/photogrammetry/<SCENE>/<RECT>/mesh_delaunay.ply
#          (from photogrammetry/run_photogrammetry.sh)
# Output
#   output/<SCENE>/<RECT>/isaacsim/<SCENE>_with_collider.usdz
#
# Steps
#   1. drop far-away stray Gaussians (> 50 m)           scripts/filter_splat_outliers.py
#   2. splat PLY -> NuRec USDZ                           3dgrut threedgrut/export/scripts/ply_to_usd.py
#   3. remove the rotation 3dgrut bakes in               scripts/fix_nurec_usdz_frame.py
#   4. add the mesh as a hidden collider, one USDZ       isaacsim/add_mesh_collider.py
#
# Needs two conda envs: ego_splats (step 1) and 3dgrut (steps 2-4), plus a 3dgrut checkout.
# Optional overrides, only if your layout differs:
#   RGB_FOCAL=1008 RGB_HEIGHT=1512   or RECT   rectified folder name
#   OUTPUT_ROOT   default <repo>/output
#   GRUT_REPO     default $HOME/3dgrut
#   GRUT_PY / SPLATS_PY / CONDA_BASE   interpreters, derived from your conda install

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCENE="${SCENE:?set SCENE to the recording name, e.g. export SCENE=Outside_20260812_141244}"
RGB_FOCAL="${RGB_FOCAL:-1008}"
RGB_HEIGHT="${RGB_HEIGHT:-1512}"
RECT="${RECT:-camera-rgb-rectified-${RGB_FOCAL}-h${RGB_HEIGHT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output}"

conda_base() {
    if [ -n "${CONDA_PREFIX:-}" ]; then echo "${CONDA_PREFIX%%/envs/*}"; else conda info --base 2>/dev/null || true; fi
}
CONDA_BASE="${CONDA_BASE:-$(conda_base)}"
GRUT_REPO="${GRUT_REPO:-$HOME/3dgrut}"
GRUT_PY="${GRUT_PY:-$CONDA_BASE/envs/3dgrut/bin/python}"
SPLATS_PY="${SPLATS_PY:-$CONDA_BASE/envs/ego_splats/bin/python}"

SPLAT_PLY="$OUTPUT_ROOT/$SCENE/$RECT/point_cloud/iteration_30000/point_cloud.ply"
MESH_PLY="$OUTPUT_ROOT/photogrammetry/$SCENE/$RECT/mesh_delaunay.ply"
OUT_DIR="$OUTPUT_ROOT/$SCENE/$RECT/isaacsim"
OUT_USDZ="$OUT_DIR/${SCENE}_with_collider.usdz"

# --- preflight ---------------------------------------------------------------------------
[ -f "$SPLAT_PLY" ] || { echo "No trained splat at: $SPLAT_PLY" >&2; echo "Run scripts/bash_local/train_gen2_outside.sh first." >&2; exit 1; }
[ -f "$MESH_PLY" ]  || { echo "No mesh at: $MESH_PLY" >&2; echo "Run photogrammetry/run_photogrammetry.sh first." >&2; exit 1; }
[ -f "$GRUT_REPO/threedgrut/export/scripts/ply_to_usd.py" ] || { echo "No 3dgrut checkout at: $GRUT_REPO (set GRUT_REPO)" >&2; exit 1; }
for py in "$GRUT_PY" "$SPLATS_PY"; do
    [ -x "$py" ] || { echo "No python interpreter at: $py (set GRUT_PY / SPLATS_PY or CONDA_BASE)" >&2; exit 1; }
done

mkdir -p "$OUT_DIR"
WORK="$(mktemp -d "$OUT_DIR/.work.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "splat : $SPLAT_PLY"
echo "mesh  : $MESH_PLY"
echo

# --- 1. drop far-away stray Gaussians --------------------------------------------------
# A few hundred nearly transparent Gaussians drift kilometres away during training. They are
# invisible but would make the asset's bounding box enormous in Isaac Sim.
echo "== 1/4 filter stray Gaussians"
"$SPLATS_PY" "$REPO_ROOT/scripts/filter_splat_outliers.py" \
    "$SPLAT_PLY" --radius 50 --output "$WORK/splat.ply"

# --- 2. splat -> NuRec USDZ (must run from the 3dgrut repo root) ------------------------
echo "== 2/4 convert splat to NuRec"
(cd "$GRUT_REPO" && "$GRUT_PY" threedgrut/export/scripts/ply_to_usd.py \
    "$WORK/splat.ply" --output_file "$WORK/splat.usdz")

# --- 3. undo 3dgrut's baked rotation, which would lay the scene on its side --------------
echo "== 3/4 fix splat orientation"
"$GRUT_PY" "$REPO_ROOT/scripts/fix_nurec_usdz_frame.py" "$WORK/splat.usdz"

# --- 4. add the mesh as a hidden collider and write the single file ---------------------
echo "== 4/4 add mesh collider"
"$GRUT_PY" "$REPO_ROOT/isaacsim/add_mesh_collider.py" \
    "$WORK/splat.usdz" "$MESH_PLY" "$OUT_USDZ"

echo
echo "Isaac Sim file:"
echo "  $OUT_USDZ"
echo "Open it with File > Open, or drag it onto a stage. Press Play to enable collisions."
