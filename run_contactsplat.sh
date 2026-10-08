#!/bin/bash
# Turn one Aria Gen 2 recording into an Isaac Sim scene: a Gaussian splat for appearance and a
# collision mesh for physics, packaged together in one USDZ.
#
#   bash run_contactsplat.sh /path/to/<recording>.vrs
#
# The MPS SLAM output must be next to the recording, in mps_<recording>_vrs/slam/, which is
# where aria_mps writes it. Stages that already finished are skipped on re-run.
#
# Stages
#   1. frames  scripts/bash_local/run_gen2_outside.sh    -> <recordings>/processed/<recording>/<RECT>/
#   2. splat   scripts/bash_local/train_gen2_outside.sh  -> output/<recording>/<RECT>/point_cloud/iteration_30000/
#   3. mesh    mps-mesh/run_mps_mesh.sh                  -> output/mps-mesh/<recording>/<RECT>/
#   4. scene   isaacsim/splat_and_mesh_to_usdz.py        -> output/<recording>/<RECT>/isaacsim/<recording>_with_collider.usdz
#
# Optional environment variables
#   MESH_SOURCE=mps     mvs = COLMAP multi-view stereo mesh instead (photogrammetry/, takes hours)
#   EYE_HEIGHT=1.6683   glasses-to-floor height in metres, only used to score the mesh
#   CAP_MAX=1500000     largest number of Gaussians the splat may grow to
#   DECIMATE=1          0 = use the full mesh as the collider instead of the simplified copy
#   MPS_FOLDER          default <recordings>/mps_<recording>_vrs/slam
#   OUT_ROOT            preprocessed frames, default <recordings>/processed
#   OUTPUT_ROOT         splat, mesh and scene, default <repo>/output

set -euo pipefail

VRS="${1:?usage: bash run_contactsplat.sh /path/to/<recording>.vrs}"
[ -f "$VRS" ] || { echo "No recording at: $VRS" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REC_ROOT="$(cd "$(dirname "$VRS")" && pwd)"
export SCENE="$(basename "$VRS" .vrs)"
export MPS_FOLDER="${MPS_FOLDER:-$REC_ROOT/mps_${SCENE}_vrs/slam}"
export OUT_ROOT="${OUT_ROOT:-$REC_ROOT/processed}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output}"
export RGB_FOCAL="${RGB_FOCAL:-1008}"
export RGB_HEIGHT="${RGB_HEIGHT:-1512}"
export RECT="camera-rgb-rectified-${RGB_FOCAL}-h${RGB_HEIGHT}"
export DECIMATE="${DECIMATE:-1}"

RECT_DIR="$OUT_ROOT/$SCENE/$RECT"
SPLAT_PLY="$OUTPUT_ROOT/$SCENE/$RECT/point_cloud/iteration_30000/point_cloud.ply"
USDZ="$OUTPUT_ROOT/$SCENE/$RECT/isaacsim/${SCENE}_with_collider.usdz"

MESH_SOURCE="${MESH_SOURCE:-mps}"
case "$MESH_SOURCE" in
    mps) MESH_SCRIPT="mps-mesh/run_mps_mesh.sh";             MESH_DIR="$OUTPUT_ROOT/mps-mesh/$SCENE/$RECT" ;;
    mvs) MESH_SCRIPT="photogrammetry/run_photogrammetry.sh"; MESH_DIR="$OUTPUT_ROOT/photogrammetry/$SCENE/$RECT" ;;
    *)   echo "MESH_SOURCE must be mps or mvs, not '$MESH_SOURCE'" >&2; exit 1 ;;
esac

# --- preflight: fail now, not after hours of training ---------------------------------------
[ -f "$MPS_FOLDER/closed_loop_trajectory.csv" ] || {
    echo "No MPS SLAM output at: $MPS_FOLDER" >&2
    echo "Run: aria_mps single -i \"$VRS\" --features SLAM" >&2
    exit 1
}
command -v colmap >/dev/null || { echo "colmap is not on PATH" >&2; exit 1; }
python -c "import torch, gsplat, open3d, pxr, msgpack, plyfile" 2>/dev/null || {
    echo "The active Python is missing packages. Run: pip install -r requirements.txt" >&2
    exit 1
}

echo "recording : $VRS"
echo "frames    : $RECT_DIR"
echo "outputs   : $OUTPUT_ROOT"
echo

echo "== 1/4 frames"
if [ -f "$RECT_DIR/transforms_with_sparse_depth.json" ]; then
    echo "already done"
else
    bash "$REPO_ROOT/scripts/bash_local/run_gen2_outside.sh"
fi

echo "== 2/4 splat"
if [ -f "$SPLAT_PLY" ]; then
    echo "already done"
else
    bash "$REPO_ROOT/scripts/bash_local/train_gen2_outside.sh"
fi

echo "== 3/4 mesh ($MESH_SOURCE)"
bash "$REPO_ROOT/$MESH_SCRIPT" "$RECT_DIR" "$MESH_DIR"
if [ "$DECIMATE" = "1" ]; then
    MESH_PLY="$MESH_DIR/mesh_delaunay_decimated.ply"
else
    MESH_PLY="$MESH_DIR/mesh_delaunay.ply"
fi

echo "== 4/4 scene"
python "$REPO_ROOT/isaacsim/splat_and_mesh_to_usdz.py" \
    --splat "$SPLAT_PLY" --mesh "$MESH_PLY" --out "$USDZ"

echo
echo "Isaac Sim scene:"
echo "  $USDZ"
echo "Open it with File > Open and press Play to turn on collisions."
