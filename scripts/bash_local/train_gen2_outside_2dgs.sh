#!/bin/bash
# Train a 2DGS (2D Gaussian surfel) model on the preprocessed Gen 2 "Outside"
# recording, for the E2 mesh-extraction comparison against the 3DGS baseline.
#
# Run from the repo root, in the ego_splats env (needs CUDA):
#   conda activate ego_splats && bash scripts/bash_local/train_gen2_outside_2dgs.sh
#
# Differences from train_gen2_outside.sh:
#   - train_model=2dgs           flat surfel Gaussians with real normals
#   - EXP_NAME gets a -2dgs suffix so the 3DGS output is not overwritten
#   - scene.data_factor / scene.pcd_stride dropped: verified inert -- no code
#     in the repo reads either key, and the 3DGS run's own cameras.json proves
#     it trained at full 2016x1512. VRAM is controlled by cap_max alone.
#   - opt.handle_rolling_shutter=false: MEASURED to be required for 2DGS on
#     16 GB. With it true, every validation/test render uses the rolling-
#     shutter motion array (_render_motion_array's eval branch ignores
#     handle_rolling_shutter_start_iter), which renders 4-8 full-res cameras
#     in ONE rasterization_2dgs call. The first attempt OOM'd at epoch-0
#     validation (~13.2 GiB in use + a 1.98 GiB isect_tiles alloc); training
#     itself would hit the same wall with gradients at iter 10k. 3DGS survives
#     this path; 2DGS's rasterizer does not. RS-off costs the model rolling-
#     shutter compensation (~1-5 px of motion across the 10.1 ms readout);
#     compare PSNR only under a matched RS-off evaluation.

set -euo pipefail

DATA_ROOT="/home/sun/aria/processed"
SCENE="Outside_20260812_141244"
RECTIFIED="camera-rgb-rectified-1008-h1512"   # must match the preprocessing output folder
EXP_NAME="$SCENE/${RECTIFIED}-2dgs"           # model_path = output_root/exp_name

# --- Memory ------------------------------------------------------------------
# 16 GB (RTX 4080 SUPER): the `default` densification strategy runs away on
# this outdoor scene (~15M Gaussians, OOM). MCMC bounds the count via cap_max;
# 1.5M matched the 3DGS run (peak ~4 GB there), so the comparison is
# budget-matched. Note the 2DGS MCMC branch only honors this since the
# _create_strategy fix -- before it, MCMCStrategy() was built with no args and
# silently used cap_max=1M.
#
# expandable_segments addresses allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_lightning.py \
    train_model=2dgs \
    opt=simple_gsplat_30K \
    opt.densification_strategy=MCMC \
    opt.mcmc_strategy.cap_max=1500000 \
    opt.handle_rolling_shutter=false \
    scene.data_root="$DATA_ROOT" \
    scene.scene_name="$SCENE/$RECTIFIED" \
    scene.input_format="aria" \
    exp_name="$EXP_NAME" \
    output_root=./output \
    viewer.use_trainer_viewer=false

echo
echo "Trained PLY:"
echo "  ./output/$EXP_NAME/point_cloud/iteration_30000/point_cloud.ply"
echo
echo "NOTE: hydra swallows exception status -- check for the PLY on disk,"
echo "not the exit code."
