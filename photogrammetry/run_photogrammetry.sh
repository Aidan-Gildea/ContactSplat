#!/bin/bash
# Photogrammetry mesh from a posed, rectified Aria dataset, with COLMAP's pose solver
# bypassed so the mesh lands in the MPS world frame -- the same frame as the trained splat.
#
#   conda activate ego_splats
#   bash photogrammetry/run_photogrammetry.sh <rectified_dir> [out_dir]
#
#   <rectified_dir>  processed/<recording>/camera-rgb-rectified-<focal>-h<height>
#                    (output of scripts/extract_aria_vrs.py / run_gen2_outside.sh)
#   [out_dir]        default <repo>/output/photogrammetry/<recording>/<rectified folder>
#
# Stages (each skipped when its output already exists; FORCE=1 reruns everything):
#   prepare      transforms.json -> keyframes + COLMAP cameras.txt / images.txt (fixed MPS poses)
#   features     colmap feature_extractor        (SIFT, GPU, keyframes only)
#   match        colmap sequential_matcher       (continuous walk -> sequential, not exhaustive)
#   triangulate  colmap point_triangulator       (poses FIXED; mapper is never run)
#   undistort    colmap image_undistorter        (PINHOLE already; only lays out the dense workspace)
#   stereo       colmap patch_match_stereo       (GPU, the slow one)
#   fuse         colmap stereo_fusion            -> dense/fused.ply
#   mesh         colmap delaunay_mesher          -> mesh_delaunay.ply
#   evaluate     photogrammetry/evaluate_mesh.py -> report.json / report.md
#
# Knobs (environment variables, all optional):
#   START=0 DURATION=-1        time window in seconds from the first posed frame (-1 = to end)
#   MIN_TRANSLATION=0.10       keyframe spacing, metres
#   MIN_ROTATION_DEG=10        keyframe spacing, degrees
#   MAX_KEYFRAMES=-1           hard cap on keyframes (uniform thinning), -1 = none
#   MAX_IMAGE_SIZE=1008        longest image side for dense stereo (2016 = native, 4x slower)
#   NUM_SRC_IMAGES=20          patch-match source images per reference image
#   SEQ_OVERLAP=20             sequential matcher window (plus powers of two beyond it)
#   LOOP_DETECTION=0           1 = vocab-tree loop detection (downloads the tree once)
#   GPU_INDEX=0                CUDA device for SIFT and patch-match
#   CACHE_GB=16                patch-match / fusion image cache (machine has 31 GB RAM)
#   EYE_HEIGHT=1.6683          trajectory-to-floor distance used by the evaluator
#   STOP_AFTER=<stage>         stop after that stage (e.g. STOP_AFTER=triangulate to check
#                              the sparse alignment before spending GPU hours)
#   FORCE=0                    1 = ignore existing outputs and redo every stage
#   PYTHON=python              interpreter with numpy, pandas, open3d (ego_splats env)

set -euo pipefail

RECT_DIR="${1:?usage: run_photogrammetry.sh <rectified_dir> [out_dir]}"
RECT_DIR="$(cd "$RECT_DIR" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

RECT_NAME="$(basename "$RECT_DIR")"
SCENE_NAME="$(basename "$(dirname "$RECT_DIR")")"
OUT_DIR="${2:-$REPO_ROOT/output/photogrammetry/$SCENE_NAME/$RECT_NAME}"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/.done"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

START="${START:-0}"
DURATION="${DURATION:--1}"
MIN_TRANSLATION="${MIN_TRANSLATION:-0.10}"
MIN_ROTATION_DEG="${MIN_ROTATION_DEG:-10}"
MAX_KEYFRAMES="${MAX_KEYFRAMES:--1}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-1008}"
NUM_SRC_IMAGES="${NUM_SRC_IMAGES:-20}"
SEQ_OVERLAP="${SEQ_OVERLAP:-20}"
LOOP_DETECTION="${LOOP_DETECTION:-0}"
GPU_INDEX="${GPU_INDEX:-0}"
CACHE_GB="${CACHE_GB:-16}"
EYE_HEIGHT="${EYE_HEIGHT:-1.6683}"
STOP_AFTER="${STOP_AFTER:-}"
FORCE="${FORCE:-0}"
PYTHON="${PYTHON:-python}"

IMAGES="$RECT_DIR/images"
DB="$OUT_DIR/database.db"
SPARSE_IN="$OUT_DIR/sparse_in"
SPARSE="$OUT_DIR/sparse"
DENSE="$OUT_DIR/dense"
MESH="$OUT_DIR/mesh_delaunay.ply"

# --- preflight ---------------------------------------------------------------------------
[ -d "$IMAGES" ] || { echo "no images/ in $RECT_DIR" >&2; exit 1; }
[ -f "$RECT_DIR/transforms.json" ] || { echo "no transforms.json in $RECT_DIR" >&2; exit 1; }
command -v colmap >/dev/null || { echo "colmap not on PATH" >&2; exit 1; }
"$PYTHON" -c "import numpy, pandas, open3d" 2>/dev/null \
    || { echo "PYTHON=$PYTHON lacks numpy/pandas/open3d; conda activate ego_splats" >&2; exit 1; }
if command -v nvidia-smi >/dev/null; then
    BUSY="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
    [ -n "$BUSY" ] && echo "WARNING: GPU already in use:" && echo "$BUSY"
fi

echo "rectified : $RECT_DIR"
echo "output    : $OUT_DIR"
echo "colmap    : $(colmap -h 2>&1 | head -1)"
echo

# --- stage helpers -----------------------------------------------------------------------
done_marker() { echo "$OUT_DIR/.done/$1"; }
is_done()     { [ "$FORCE" != "1" ] && [ -f "$(done_marker "$1")" ]; }
mark_done()   { date -Is > "$(done_marker "$1")"; }
maybe_stop()  { [ "$STOP_AFTER" = "$1" ] && { echo; echo "STOP_AFTER=$1 reached."; exit 0; } || true; }
fail_stage()  { rm -f "$(done_marker "$1")"; echo "== $1 FAILED: $2" >&2; exit 1; }

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
run_stage prepare "$PYTHON" "$HERE/colmap_from_transforms.py" \
    --rectified_dir "$RECT_DIR" --out_dir "$OUT_DIR" \
    --start "$START" --duration "$DURATION" \
    --min_translation "$MIN_TRANSLATION" --min_rotation_deg "$MIN_ROTATION_DEG" \
    --max_keyframes "$MAX_KEYFRAMES"
maybe_stop prepare

N_KEYFRAMES="$(grep -c . "$OUT_DIR/keyframes.txt")"
CAM_PARAMS="$(awk '!/^#/ {print $5","$6","$7","$8; exit}' "$SPARSE_IN/cameras.txt")"

# --- 2. features -------------------------------------------------------------------------
# One shared PINHOLE camera whose parameters are the rectified intrinsics, so the database
# camera and the fixed-pose model agree exactly. Only the keyframes are extracted.
if [ "$FORCE" = "1" ] && [ -f "$DB" ]; then rm -f "$DB"; fi
run_stage features colmap feature_extractor \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --image_list_path "$OUT_DIR/keyframes.txt" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_params "$CAM_PARAMS" \
    --FeatureExtraction.use_gpu 1 \
    --FeatureExtraction.gpu_index "$GPU_INDEX"
maybe_stop features

# --- 3. match ----------------------------------------------------------------------------
# Sequential: keyframes are a continuous walk. quadratic_overlap also matches frame i with
# i + 2^k, which catches revisits along the same path without a vocabulary tree.
run_stage match colmap sequential_matcher \
    --database_path "$DB" \
    --SequentialMatching.overlap "$SEQ_OVERLAP" \
    --SequentialMatching.quadratic_overlap 1 \
    --SequentialMatching.loop_detection "$LOOP_DETECTION" \
    --FeatureMatching.use_gpu 1 \
    --FeatureMatching.gpu_index "$GPU_INDEX"
maybe_stop match

# --- 4. triangulate ----------------------------------------------------------------------
# point_triangulator keeps every image pose constant and only solves for 3D points; the
# three ba_refine flags keep the intrinsics constant as well. This is the step that
# replaces `mapper`, and it is why the result stays in the MPS frame.
mkdir -p "$SPARSE"
run_stage triangulate colmap point_triangulator \
    --database_path "$DB" \
    --image_path "$IMAGES" \
    --input_path "$SPARSE_IN" \
    --output_path "$SPARSE" \
    --clear_points 1 \
    --refine_intrinsics 0 \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0
# Human-readable copy + statistics (reprojection error, points, observations per image).
colmap model_converter --input_path "$SPARSE" --output_path "$SPARSE" --output_type TXT >/dev/null 2>&1 || true
colmap model_analyzer --path "$SPARSE" 2>&1 | tee "$OUT_DIR/logs/triangulate_stats.log" || true
N_REG="$(grep -oE "Registered images: [0-9]+" "$OUT_DIR/logs/triangulate_stats.log" | grep -oE "[0-9]+$" || echo 0)"
N_PTS="$(grep -oE "Points: [0-9]+" "$OUT_DIR/logs/triangulate_stats.log" | grep -oE "[0-9]+$" || echo 0)"
if [ "$N_REG" -lt "$N_KEYFRAMES" ] || [ "$N_PTS" -eq 0 ]; then
    fail_stage triangulate "$N_REG of $N_KEYFRAMES keyframes registered, $N_PTS points; see logs/triangulate_stats.log"
fi
echo "   $N_REG keyframes, $N_PTS points, $(grep -oE "Mean reprojection error: [0-9.]+px" "$OUT_DIR/logs/triangulate_stats.log")"
maybe_stop triangulate

# --- 5. undistort ------------------------------------------------------------------------
# Images are already pinhole, so this only lays out the dense workspace (dense/images,
# dense/sparse, dense/stereo/patch-match.cfg) and links the images in at native size.
# Do NOT pass --max_image_size here: COLMAP 3.13 rescales the images but leaves the camera
# at the original size, and patch_match_stereo then rejects every view with
# "Check failed: width_ == bitmap_.Width()". Downscaling is done inside stereo and fusion
# instead, which rescale the cameras consistently.
run_stage undistort colmap image_undistorter \
    --image_path "$IMAGES" \
    --input_path "$SPARSE" \
    --output_path "$DENSE" \
    --output_type COLMAP \
    --copy_policy soft-link \
    --num_patch_match_src_images "$NUM_SRC_IMAGES"
maybe_stop undistort

# --- 6. stereo ---------------------------------------------------------------------------
run_stage stereo colmap patch_match_stereo \
    --workspace_path "$DENSE" \
    --workspace_format COLMAP \
    --PatchMatchStereo.max_image_size "$MAX_IMAGE_SIZE" \
    --PatchMatchStereo.geom_consistency 1 \
    --PatchMatchStereo.gpu_index "$GPU_INDEX" \
    --PatchMatchStereo.cache_size "$CACHE_GB"
# COLMAP logs per-view failures and still exits 0, so count the depth maps ourselves.
N_DEPTH="$(ls "$DENSE/stereo/depth_maps"/*.geometric.bin 2>/dev/null | wc -l)"
if [ "$N_DEPTH" -lt "$N_KEYFRAMES" ]; then
    fail_stage stereo "only $N_DEPTH of $N_KEYFRAMES geometric depth maps were written; see logs/stereo.log"
fi
echo "   $N_DEPTH geometric depth maps"
maybe_stop stereo

# --- 7. fuse -----------------------------------------------------------------------------
run_stage fuse colmap stereo_fusion \
    --workspace_path "$DENSE" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "$DENSE/fused.ply" \
    --StereoFusion.max_image_size "$MAX_IMAGE_SIZE" \
    --StereoFusion.cache_size "$CACHE_GB"
N_FUSED="$(head -c 400 "$DENSE/fused.ply" | awk '/^element vertex/ {print $3; exit}')"
if [ -z "$N_FUSED" ] || [ "$N_FUSED" -eq 0 ]; then
    fail_stage fuse "fused.ply holds no points; see logs/fuse.log"
fi
echo "   fused cloud: $N_FUSED points"
maybe_stop fuse

# --- 8. mesh -----------------------------------------------------------------------------
run_stage mesh colmap delaunay_mesher \
    --input_path "$DENSE" \
    --input_type dense \
    --output_path "$MESH"
maybe_stop mesh

# --- 9. evaluate -------------------------------------------------------------------------
run_stage evaluate "$PYTHON" "$HERE/evaluate_mesh.py" \
    --mesh "$MESH" \
    --rectified_dir "$RECT_DIR" \
    --fused "$DENSE/fused.ply" \
    --keyframes_json "$OUT_DIR/keyframes.json" \
    --eye_height "$EYE_HEIGHT" \
    --out "$OUT_DIR/report.json"

echo "Mesh (MPS world frame, metres, Z-up):"
echo "  $MESH"
echo "Report:"
echo "  $OUT_DIR/report.md"
