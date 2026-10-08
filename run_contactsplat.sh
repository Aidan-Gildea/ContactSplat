#!/bin/bash
# Turn one Aria Gen 2 recording into an Isaac Sim scene: a Gaussian splat for appearance and a
# collision mesh for physics, packaged together in one USDZ.
#
#   bash run_contactsplat.sh /path/to/<recording>.vrs
#
# The MPS SLAM output must be next to the recording, in mps_<recording>_vrs/slam/, which is
# where aria_mps writes it (or set MPS_FOLDER). The frames, mesh and splat stages are skipped
# when they already finished; the scene file is rebuilt every run (a few seconds).
#
# Stages
#   1. frames  scripts/bash_local/run_gen2_outside.sh    -> <recordings>/processed/<recording>/<RECT>/
#   2. mesh    mps-mesh/run_mps_mesh.sh                  -> output/mps-mesh/<recording>/<RECT>/
#   3. splat   scripts/bash_local/train_gen2_outside.sh  -> output/<recording>/<RECT>/point_cloud/iteration_<ITERATIONS>/
#   4. scene   isaacsim/splat_and_mesh_to_usdz.py        -> output/<recording>/<RECT>/isaacsim/<recording>_with_collider.usdz
#
# The mesh runs before the splat so that a COLMAP problem shows up in a minute, not after training.

set -euo pipefail

# --- Settings ----------------------------------------------------------------------------------
# Edit the values here, or set any of them before the command, e.g. ITERATIONS=15000 bash run_contactsplat.sh ...
# A changed setting only affects stages that have not run yet. To redo the mesh, add FORCE=1. To
# retrain the splat, delete output/<recording>/<RECT>/point_cloud/.

# Splat training (stage 3)
# Training steps. Time grows about linearly with this. The Gaussian count stops growing at
# step 25,000, so going lower mostly costs sharpness.
export ITERATIONS="${ITERATIONS:-30000}"
# Most Gaussians training may grow to. Higher gives finer detail but uses more GPU memory and
# makes each step slower. It caps growth only: a recording with more MPS points starts above it.
export CAP_MAX="${CAP_MAX:-1500000}"
# Model the RGB camera reading out row by row while you walk. Sharper splats; steps after
# RS_START are slower because each frame is rendered as several row bands.
export ROLLING_SHUTTER="${ROLLING_SHUTTER:-true}"
# Step where rolling-shutter modelling starts. Earlier makes training slower.
export RS_START="${RS_START:-10000}"
# Also fit the MPS sparse depth. Can help flat, textureless surfaces. Untested on our recordings.
export DEPTH_LOSS="${DEPTH_LOSS:-false}"
# 7-1 holds out every 8th frame and scores the splat on them (test_logs.json). all trains on
# every frame; the score is then measured on training frames, so it reads higher than it is.
export TRAIN_SPLIT="${TRAIN_SPLIT:-7-1}"

# Collision mesh (stage 2)
# mps builds the mesh from the MPS points in about a minute. mvs uses COLMAP multi-view stereo
# instead: hours, and outdoors only slightly fuller floors.
export MESH_SOURCE="${MESH_SOURCE:-mps}"
# 1 simplifies the mesh to within about 1 cm (4-17% of the triangles, faster physics) and uses
# that as the collider. 0 uses the full mesh.
export DECIMATE="${DECIMATE:-1}"
# Height of the glasses above the floor in metres. Only used to score the mesh, not to build it.
export EYE_HEIGHT="${EYE_HEIGHT:-1.6683}"

# Scene file (stage 4)
# Drop Gaussians farther than this many metres from the scene centre. A few drift far away
# during training and would make the scene's bounding box huge.
SPLAT_RADIUS="${SPLAT_RADIUS:-50}"
# Drop collider triangles with an edge longer than this many metres: spikes across open space.
MAX_EDGE="${MAX_EDGE:-2.0}"

# Recording
# Rectified RGB size. 1008/1512 matches profile10 (2016x1512). For profile8 (2560x1920) use
# 1280/1920; for other profiles see docs/rectification.md.
export RGB_FOCAL="${RGB_FOCAL:-1008}"
export RGB_HEIGHT="${RGB_HEIGHT:-1512}"

# --- Paths ---------------------------------------------------------------------------------------
VRS="${1:?usage: bash run_contactsplat.sh /path/to/<recording>.vrs}"
[ -f "$VRS" ] || { echo "No recording at: $VRS" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REC_ROOT="$(cd "$(dirname "$VRS")" && pwd)"
export SCENE="$(basename "$VRS" .vrs)"
export RECT="camera-rgb-rectified-${RGB_FOCAL}-h${RGB_HEIGHT}"
MPS_FOLDER="${MPS_FOLDER:-$REC_ROOT/mps_${SCENE}_vrs/slam}"
OUT_ROOT="${OUT_ROOT:-$REC_ROOT/processed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output}"

case "$MESH_SOURCE" in
    mps) MESH_SCRIPT="mps-mesh/run_mps_mesh.sh";             MESH_SUBDIR="mps-mesh" ;;
    mvs) MESH_SCRIPT="photogrammetry/run_photogrammetry.sh"; MESH_SUBDIR="photogrammetry" ;;
    *)   echo "MESH_SOURCE must be mps or mvs, not '$MESH_SOURCE'" >&2; exit 1 ;;
esac

# --- Preflight: fail now, not after hours of work ------------------------------------------------
[ -f "$MPS_FOLDER/closed_loop_trajectory.csv" ] || {
    echo "No MPS SLAM output at: $MPS_FOLDER" >&2
    echo "Run aria_mps on the recording, or set MPS_FOLDER to its slam/ folder." >&2
    exit 1
}
command -v colmap >/dev/null || { echo "colmap is not on PATH" >&2; exit 1; }
python -c "import torch, lightning, hydra, gsplat, open3d, pxr, msgpack, plyfile" 2>/dev/null || {
    echo "The active Python is missing packages. Run: pip install -r requirements.txt" >&2
    exit 1
}
echo "Checking gsplat's CUDA kernels (the first run compiles them, a few minutes)"
python -c "from gsplat.cuda._backend import _C; assert _C is not None" >/dev/null 2>&1 || {
    echo "gsplat could not load or build its CUDA kernels." >&2
    echo "Install the CUDA toolkit (nvcc) that matches your PyTorch build and set CUDA_HOME." >&2
    exit 1
}

# The stage scripts change directory, so hand them absolute paths.
mkdir -p "$OUT_ROOT" "$OUTPUT_ROOT"
export MPS_FOLDER="$(cd "$MPS_FOLDER" && pwd)"
export OUT_ROOT="$(cd "$OUT_ROOT" && pwd)"
export OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

RECT_DIR="$OUT_ROOT/$SCENE/$RECT"
MESH_DIR="$OUTPUT_ROOT/$MESH_SUBDIR/$SCENE/$RECT"
SPLAT_PLY="$OUTPUT_ROOT/$SCENE/$RECT/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
USDZ="$OUTPUT_ROOT/$SCENE/$RECT/isaacsim/${SCENE}_with_collider.usdz"
if [ "$DECIMATE" = "1" ]; then
    MESH_PLY="$MESH_DIR/mesh_delaunay_decimated.ply"
else
    MESH_PLY="$MESH_DIR/mesh_delaunay.ply"
fi

echo "recording : $VRS"
echo "frames    : $RECT_DIR"
echo "outputs   : $OUTPUT_ROOT"
echo

# --- Stages --------------------------------------------------------------------------------------
echo "== 1/4 frames"
if [ -f "$RECT_DIR/transforms_with_sparse_depth.json" ]; then
    echo "already done"
else
    bash "$REPO_ROOT/scripts/bash_local/run_gen2_outside.sh"
fi

echo "== 2/4 mesh ($MESH_SOURCE)"
bash "$REPO_ROOT/$MESH_SCRIPT" "$RECT_DIR" "$MESH_DIR"

echo "== 3/4 splat"
if [ -f "$SPLAT_PLY" ]; then
    echo "already done"
else
    bash "$REPO_ROOT/scripts/bash_local/train_gen2_outside.sh"
fi

echo "== 4/4 scene"
python "$REPO_ROOT/isaacsim/splat_and_mesh_to_usdz.py" \
    --splat "$SPLAT_PLY" --mesh "$MESH_PLY" --out "$USDZ" \
    --radius "$SPLAT_RADIUS" --max-edge "$MAX_EDGE"

echo
echo "Isaac Sim scene:"
echo "  $USDZ"
echo "Open it with File > Open and press Play to turn on collisions."
