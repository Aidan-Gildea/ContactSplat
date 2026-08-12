#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Preprocess an Aria **Gen 2** recording.
#
# Same script as run_vrs_preprocessing.sh, retuned for Gen 2 optics. See that
# file for the Gen 1 settings. The Python side auto-detects the generation and
# resolves camera labels from the device, so the only differences here are the
# rectification parameters and the paths.

set -euo pipefail

# NOTE: these paths contain spaces, so every expansion below is quoted.
DATA_INPUT_DIR="/Users/patrick/aria"
DATA_PROCESSED_DIR="/Users/patrick/aria/processed"
VRS_FILE="Test recording_20260804_195027.vrs"

# MPS folder
MPS_FOLDER="$DATA_INPUT_DIR/mps_Test recording_20260804_195027_vrs/slam"
# Ensures the MPS folder contains the following structure
# $MPS_FOLDER
# - closed_loop_trajectory.csv
# - semidense_points.csv.gz
# - semidense_observations.csv.gz
# - online_calibration.jsonl      <- Gen 2 also sources readout time from here

mkdir -p "$DATA_PROCESSED_DIR"

# --- Rectification parameters, derived from the Gen 2 calibration ------------
#
# Gen 2 RGB is 2560x1920 (profile8) with a 133x99 degree FOV and a native
# fisheye focal of ~1115. A pinhole cannot cover 133 degrees, so as on Gen 1 we
# target a narrower rectified FOV.
#
#   horizontal FOV = 2 * atan((width / 2) / focal)
#
# focal 1280 at width 2560 gives exactly 90 degrees horizontal (73.7 vertical),
# matching the effective FOV the Gen 1 defaults produced (2400 wide @ focal
# 1200). --rectified_rgb_size is the output HEIGHT; width follows the 4:3
# source aspect ratio automatically.
#
# Gen 2 SLAM is 512x512 with a 119 degree FOV. focal 180 at width 512 gives
# ~110 degrees, which is the same trade the Gen 1 defaults made.
python scripts/extract_aria_vrs.py \
    --input_root "$DATA_INPUT_DIR" \
    --output_root "$DATA_PROCESSED_DIR" \
    --vrs_file "$VRS_FILE" \
    --rectified_rgb_focal 1280 \
    --rectified_rgb_size 1920 \
    --rectified_monochrome_focal 180 --rectified_monochrome_height 512 \
    --online_calib_file "$MPS_FOLDER/online_calibration.jsonl" \
    --trajectory_file "$MPS_FOLDER/closed_loop_trajectory.csv" \
    --semi_dense_points_file "$MPS_FOLDER/semidense_points.csv.gz" \
    --semi_dense_observation_file "$MPS_FOLDER/semidense_observations.csv.gz"
    # --timestamp_convention readout_start   # reproduce upstream's half-readout bias
    # --visualize                            # stream each stage to a rerun viewer
    # --extract_fisheye
    # --use_factory_calib
