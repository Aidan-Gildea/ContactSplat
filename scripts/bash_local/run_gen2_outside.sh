#!/bin/bash
# Preprocess the Aria Gen 2 "Outside_20260812_141244" recording.
#
# Concrete, runnable version of run_vrs_preprocessing_gen2.sh with this
# machine's real paths and rectification parameters matched to the recording's
# actual sensor mode (profile10, RGB 2016x1512 @ 30 Hz).
#
# Run from the repo root, in the ego_splats env:
#   conda activate ego_splats && bash scripts/bash_local/run_gen2_outside.sh

set -euo pipefail

REC_ROOT="/home/sun/aria"
SCENE="Outside_20260812_141244"
MPS_FOLDER="$REC_ROOT/mps_${SCENE}_vrs/slam"
OUT_ROOT="$REC_ROOT/processed"

mkdir -p "$OUT_ROOT"

# --- Rectification parameters ------------------------------------------------
#
# This recording's RGB is 2016x1512 (profile10), NOT the 2560x1920 of profile8
# that run_vrs_preprocessing_gen2.sh was tuned for. Its defaults (focal 1280 /
# height 1920) would resample every frame UP to 2560x1920 -- 4.19 MB per PNG
# instead of 2.94 MB, for no extra information.
#
#   horizontal FOV = 2 * atan((width / 2) / focal)
#
# --rectified_rgb_size is the output HEIGHT; width follows the 4:3 source
# aspect ratio. focal 1008 at the resulting width 2016 gives exactly 90
# degrees horizontal, matching the effective FOV of the Gen 1 defaults, at the
# sensor's native sampling.
#
# SLAM is 512x512 with a 119 degree FOV; focal 180 at width 512 keeps ~110.
RGB_FOCAL=1008
RGB_HEIGHT=1512
SLAM_FOCAL=180
SLAM_HEIGHT=512

python scripts/extract_aria_vrs.py \
    --input_root  "$REC_ROOT" \
    --output_root "$OUT_ROOT" \
    --vrs_file    "$SCENE.vrs" \
    --rectified_rgb_focal        "$RGB_FOCAL" \
    --rectified_rgb_size         "$RGB_HEIGHT" \
    --rectified_monochrome_focal "$SLAM_FOCAL" \
    --rectified_monochrome_height "$SLAM_HEIGHT" \
    --online_calib_file           "$MPS_FOLDER/online_calibration.jsonl" \
    --trajectory_file             "$MPS_FOLDER/closed_loop_trajectory.csv" \
    --semi_dense_points_file      "$MPS_FOLDER/semidense_points.csv.gz" \
    --semi_dense_observation_file "$MPS_FOLDER/semidense_observations.csv.gz"
    # --overwrite                          # regenerate instead of skipping existing output
    # --visualize                          # stream each stage to a rerun viewer
    # --timestamp_convention readout_start # reproduce upstream's half-readout bias

echo
echo "Done. Training folder:"
echo "  $OUT_ROOT/$SCENE/camera-rgb-rectified-${RGB_FOCAL}-h${RGB_HEIGHT}"
