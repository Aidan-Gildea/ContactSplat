#!/bin/bash
# Convert a trained egocentric_splats PLY into a NuRec USDZ that Isaac Sim can render.
#
# This step runs in the *3dgrut* env, not ego_splats -- it uses NVIDIA's
# USDZ/NuRec exporter, which lives in the 3dgrut repo.
#
#   bash scripts/bash_local/export_gen2_outside_usdz.sh
#
# The first run compiles 3dgrut's CUDA extensions (a few minutes); later runs
# are seconds.

set -euo pipefail

SPLATS_REPO="/home/sun/Desktop/aria_proj/egocentric_splats"
GRUT_REPO="/home/sun/3dgrut"
GRUT_PY="/home/sun/miniforge3/envs/3dgrut/bin/python"

SCENE="Outside_20260812_141244"
RECTIFIED="camera-rgb-rectified-1008-h1512"
ITER=30000

PLY="$SPLATS_REPO/output/$SCENE/$RECTIFIED/point_cloud/iteration_${ITER}/point_cloud.ply"
OUT_DIR="$SPLATS_REPO/output/$SCENE/$RECTIFIED/isaacsim"
USDZ="$OUT_DIR/${SCENE}.usdz"

if [ ! -f "$PLY" ]; then
    echo "No trained PLY at: $PLY" >&2
    echo "Train first: bash scripts/bash_local/train_gen2_outside.sh" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# Strip far-field outliers before export.
#
# MCMC drifts a few hundred low-opacity Gaussians arbitrarily far from the
# scene. They are invisible (median opacity ~0.01) but they define the asset's
# bounding box: unfiltered, this scene exported with a ~120 km AABB around 10 m
# of actual content, which ruins framing and depth precision in a USD stage.
# 50 m keeps ~98.7% of Gaussians and bounds the AABB to +/-50 m.
FILTER_RADIUS=50
FILTERED="${PLY%.ply}_filtered.ply"
/home/sun/miniforge3/envs/ego_splats/bin/python \
    "$SPLATS_REPO/scripts/filter_splat_outliers.py" \
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
# (up ends along stage -Y). Root-caused in docs/experiments/E1_report.md.
# The fix sets the transform to identity, in place, and refuses to touch any
# transform it does not recognise.
"$GRUT_PY" "$SPLATS_REPO/scripts/fix_nurec_usdz_frame.py" "$USDZ"

echo
echo "USDZ written to:"
echo "  $USDZ"
echo
echo "Open it in Isaac Sim (File > Open, or drag onto the stage) with an RTX"
echo "renderer selected. The asset content is in the MPS world frame (Z-up,"
echo "metres) and the exporter's baked rotation has been stripped, so it needs"
echo "no extra transform. Note: 'F' framing frames the +/-${FILTER_RADIUS} m outlier-filter"
echo "AABB while ~85% of the opacity mass sits within 10 m of centre, so the"
echo "content looks small until you zoom in -- that is framing, not a unit error."
