#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Render a trajectory video from a trained model.
#
# Point RESULTS_FOLDER at the trained model directory (the one holding
# point_cloud/ and cameras.json). If unset, it is built from the README's
# Block 2 variables as output/$SCENE/$RECT.
results_folder="${RESULTS_FOLDER:-output/${SCENE:?set RESULTS_FOLDER, or SCENE and RECT (README Block 2)}/${RECT:?set RESULTS_FOLDER, or SCENE and RECT (README Block 2)}}"

gain_amplify=${GAIN_AMPLIFY:-1}  # amplify the analog gain, compared to the original video
render_fps=${RENDER_FPS:-10}     # render a 10 fps video. Will interpolate the poses from the keyframes

ply_file=$results_folder/point_cloud/iteration_30000/point_cloud.ply
json_file=$results_folder/cameras.json
render_output=$results_folder/traj_render_fps_"$render_fps"_gain_"$gain_amplify"

python render_lightning.py \
    scene.load_ply="$ply_file" \
    render.render_only=true \
    render.render_json="$json_file" \
    render.render_output="$render_output" \
    render.render_fps=$render_fps \
    render.gain_amplify=$gain_amplify
