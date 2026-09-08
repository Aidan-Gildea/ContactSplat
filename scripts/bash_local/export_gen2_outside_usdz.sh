#!/bin/bash
# Convert a trained egocentric_splats PLY into a NuRec USDZ that Isaac Sim can render.
#
# This step runs in the *3dgrut* env, not ego_splats -- it uses NVIDIA's
# USDZ/NuRec exporter, which lives in the 3dgrut repo.
#
# Reads the same variables as the README (sections 1.2, 1.4 and Block 2):
#
#   export SCENE="my_recording"
#   export RGB_FOCAL=1008 RGB_HEIGHT=1512      # or set RECT directly
#   bash scripts/bash_local/export_gen2_outside_usdz.sh
#
# Optional overrides:
#   RECT          rectified folder name, default camera-rgb-rectified-${RGB_FOCAL}-h${RGB_HEIGHT}
#   OUTPUT_ROOT   where trained models live, default <repo>/output
#   ITER          training iteration to export, default 30000
#   GRUT_REPO     3dgrut checkout, default $HOME/3dgrut
#   GRUT_PY       python of the 3dgrut conda env, default <conda base>/envs/3dgrut/bin/python
#   SPLATS_PY     python of the ego_splats conda env, default <conda base>/envs/ego_splats/bin/python
#   CONDA_BASE    conda install root, derived from CONDA_PREFIX or `conda info --base`
#
# The first run compiles 3dgrut's CUDA extensions (a few minutes); later runs
# are seconds.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SCENE="${SCENE:?set SCENE to the recording name (README 1.2)}"
RGB_FOCAL="${RGB_FOCAL:-1008}"
RGB_HEIGHT="${RGB_HEIGHT:-1512}"
RECT="${RECT:-camera-rgb-rectified-${RGB_FOCAL}-h${RGB_HEIGHT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output}"
ITER="${ITER:-30000}"

# Resolve the two conda env interpreters without hardcoding an install location.
conda_base() {
    if [ -n "${CONDA_PREFIX:-}" ]; then
        echo "${CONDA_PREFIX%%/envs/*}"
    else
        conda info --base 2>/dev/null || true
    fi
}
CONDA_BASE="${CONDA_BASE:-$(conda_base)}"
GRUT_REPO="${GRUT_REPO:-$HOME/3dgrut}"
GRUT_PY="${GRUT_PY:-$CONDA_BASE/envs/3dgrut/bin/python}"
SPLATS_PY="${SPLATS_PY:-$CONDA_BASE/envs/ego_splats/bin/python}"

PLY="$OUTPUT_ROOT/$SCENE/$RECT/point_cloud/iteration_${ITER}/point_cloud.ply"
OUT_DIR="$OUTPUT_ROOT/$SCENE/$RECT/isaacsim"
USDZ="$OUT_DIR/${SCENE}.usdz"

if [ ! -f "$PLY" ]; then
    echo "No trained PLY at: $PLY" >&2
    echo "Train first: bash scripts/bash_local/train_gen2_outside.sh" >&2
    exit 1
fi
if [ ! -f "$GRUT_REPO/threedgrut/export/scripts/ply_to_usd.py" ]; then
    echo "No 3dgrut checkout at: $GRUT_REPO (set GRUT_REPO)" >&2
    exit 1
fi
for py in "$GRUT_PY" "$SPLATS_PY"; do
    if [ ! -x "$py" ]; then
        echo "No python interpreter at: $py (set GRUT_PY / SPLATS_PY or CONDA_BASE)" >&2
        exit 1
    fi
done

mkdir -p "$OUT_DIR"

# Strip far-field outliers before export.
#
# MCMC drifts a few hundred low-opacity Gaussians arbitrarily far from the
# scene. They are invisible (median opacity ~0.01) but they define the asset's
# bounding box: unfiltered, one scene exported with a ~120 km AABB around 10 m
# of actual content, which ruins framing and depth precision in a USD stage.
# 50 m keeps ~98.7% of Gaussians and bounds the AABB to +/-50 m.
FILTER_RADIUS="${FILTER_RADIUS:-50}"
FILTERED="${PLY%.ply}_filtered.ply"
"$SPLATS_PY" "$REPO_ROOT/scripts/filter_splat_outliers.py" \
    "$PLY" --radius "$FILTER_RADIUS" --output "$FILTERED"

# ply_to_usd.py must run from the 3dgrut repo root: it resolves its hydra
# config with a path relative to the script's own location.
cd "$GRUT_REPO"
"$GRUT_PY" threedgrut/export/scripts/ply_to_usd.py "$FILTERED" --output_file "$USDZ"

# Strip the frame-conversion rotation 3dgrut bakes into the volume prim.
#
# ply_to_usd.py assumes its input is in 3dgrut's normalized frame (Y-down) and
# unconditionally authors a Y-down -> Z-up rotation, (x,y,z) -> (-x,-z,-y), as
# xformOp:transform on /World/gauss. Our PLY is *already* Z-up metres in the
# MPS world frame, so that rotation lays the scene on its side in Isaac Sim
# (up ends along stage -Y). The derivation is in the docstring of
# scripts/fix_nurec_usdz_frame.py. The fix sets the transform to identity, in
# place, and refuses to touch any transform it does not recognise.
"$GRUT_PY" "$REPO_ROOT/scripts/fix_nurec_usdz_frame.py" "$USDZ"

echo
echo "USDZ written to:"
echo "  $USDZ"
echo
echo "Open it in Isaac Sim (File > Open, or drag onto the stage) with an RTX"
echo "renderer selected. The asset content is in the MPS world frame (Z-up,"
echo "metres) and the exporter's baked rotation has been stripped, so it needs"
echo "no extra transform. Note: 'F' framing frames the +/-${FILTER_RADIUS} m outlier-filter"
echo "AABB while most of the opacity mass sits within ~10 m of centre, so the"
echo "content looks small until you zoom in -- that is framing, not a unit error."
