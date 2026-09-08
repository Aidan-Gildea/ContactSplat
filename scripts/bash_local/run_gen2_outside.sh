#!/bin/bash
# Preprocess an Aria Gen 2 recording made with profile10 (RGB 2016x1512 @ 30 Hz).
#
# Concrete, runnable version of run_vrs_preprocessing_gen2.sh with rectification
# parameters matched to profile10. Paths come from the environment, set exactly
# as in README section 1.2:
#
#   export REC_ROOT="/path/to/recordings"   # folder holding <SCENE>.vrs and the MPS output
#   export SCENE="my_recording"             # .vrs filename WITHOUT the extension
#   conda activate ego_splats && bash scripts/bash_local/run_gen2_outside.sh
#
# Optional overrides (defaults match the README):
#   MPS_FOLDER   default $REC_ROOT/mps_${SCENE}_vrs/slam
#   OUT_ROOT     default $REC_ROOT/processed
#   RGB_FOCAL / RGB_HEIGHT / SLAM_FOCAL / SLAM_HEIGHT   see below

set -euo pipefail

REC_ROOT="${REC_ROOT:?set REC_ROOT to the folder containing \$SCENE.vrs (README 1.2)}"
SCENE="${SCENE:?set SCENE to the .vrs filename without the extension (README 1.2)}"
MPS_FOLDER="${MPS_FOLDER:-$REC_ROOT/mps_${SCENE}_vrs/slam}"
OUT_ROOT="${OUT_ROOT:-$REC_ROOT/processed}"

# Run from the repo root regardless of where the script was invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$REC_ROOT/$SCENE.vrs" ]; then
    echo "No recording at: $REC_ROOT/$SCENE.vrs" >&2
    exit 1
fi
if [ ! -f "$MPS_FOLDER/closed_loop_trajectory.csv" ]; then
    echo "No MPS output at: $MPS_FOLDER (expected closed_loop_trajectory.csv)" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

# --- Rectification parameters ------------------------------------------------
#
# profile10 RGB is 2016x1512, NOT the 2560x1920 of profile8 that
# run_vrs_preprocessing_gen2.sh defaults to. Those defaults (focal 1280 /
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
# For any other sensor mode see docs/rectification.md and debug_scripts/solve_focal.py.
RGB_FOCAL="${RGB_FOCAL:-1008}"
RGB_HEIGHT="${RGB_HEIGHT:-1512}"
SLAM_FOCAL="${SLAM_FOCAL:-180}"
SLAM_HEIGHT="${SLAM_HEIGHT:-512}"

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
