#!/bin/bash
# Train a 3DGS model on the preprocessed Gen 2 "Outside" recording.
#
# Run from the repo root, in the ego_splats env (needs CUDA):
#   conda activate ego_splats && bash scripts/bash_local/train_gen2_outside.sh

set -euo pipefail

DATA_ROOT="/home/sun/aria/processed"
SCENE="Outside_20260812_141244"
RECTIFIED="camera-rgb-rectified-1008-h1512"   # must match the preprocessing output folder
EXP_NAME="$SCENE/$RECTIFIED"                  # model_path = output_root/exp_name

# --- Memory ------------------------------------------------------------------
#
# 16 GB (RTX 4080 SUPER) is not enough for this scene with the `default`
# densification strategy. On 4,675 frames at 2016x1512 it grew to roughly 15M
# Gaussians and died in spherical_harmonics_bwd around iteration 10k -- which is
# also where handle_rolling_shutter starts rendering several samples per frame,
# compounding the peak.
#
# MCMC bounds the Gaussian count outright via cap_max, which is the reliable fix
# on a fixed VRAM budget. 3M Gaussians is ample for a 10x10 m scene and leaves
# headroom for the rolling-shutter sampling.
#
# expandable_segments addresses fragmentation: the OOM reported 7.3 GB reserved
# but unallocated.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Peak memory is driven by the rasteriser's Gaussian-tile intersection buffers,
# which scale with BOTH the Gaussian count and the pixel count. Capping only the
# Gaussian count (cap_max=3M at full 2016x1512) still OOM'd at 14.3 GB genuinely
# allocated. These three settings together leave real headroom on 16 GB:
#
#   data_factor=2   1008x756 -- 4x fewer pixels, so ~4x fewer intersections
#   cap_max=1.5M    half the Gaussian ceiling
#   pcd_stride=2    start from ~2.1M init points instead of ~4.3M
#
# On a 24 GB+ GPU, drop data_factor and raise cap_max for a sharper result.
python train_lightning.py \
    train_model=3dgs \
    opt=simple_gsplat_30K \
    opt.densification_strategy=MCMC \
    opt.mcmc_strategy.cap_max=1500000 \
    opt.handle_rolling_shutter=true \
    scene.data_root="$DATA_ROOT" \
    scene.scene_name="$SCENE/$RECTIFIED" \
    scene.input_format="aria" \
    scene.data_factor=2 \
    scene.pcd_stride=2 \
    exp_name="$EXP_NAME" \
    output_root=./output \
    viewer.use_trainer_viewer=false

echo
echo "Trained PLY:"
echo "  ./output/$EXP_NAME/point_cloud/iteration_30000/point_cloud.ply"
