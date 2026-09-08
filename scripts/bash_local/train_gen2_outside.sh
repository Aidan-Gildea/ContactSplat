#!/bin/bash
# Train a 3DGS model on a preprocessed Gen 2 recording.
#
# Reads the same variables as the README (sections 1.2, 1.4 and Block 2):
#
#   export OUT_ROOT="/path/to/recordings/processed"   # or REC_ROOT; OUT_ROOT defaults to $REC_ROOT/processed
#   export SCENE="my_recording"
#   export RGB_FOCAL=1008 RGB_HEIGHT=1512              # or set RECT directly
#   conda activate ego_splats && bash scripts/bash_local/train_gen2_outside.sh
#
# Optional overrides:
#   RECT          rectified folder name, default camera-rgb-rectified-${RGB_FOCAL}-h${RGB_HEIGHT}
#   OUTPUT_ROOT   where trained models go, default <repo>/output
#   CAP_MAX       MCMC Gaussian ceiling, default 1500000 (see the memory notes below)

set -euo pipefail

SCENE="${SCENE:?set SCENE to the recording name (README 1.2)}"
OUT_ROOT="${OUT_ROOT:-${REC_ROOT:+$REC_ROOT/processed}}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT (or REC_ROOT) to the preprocessed data root (README 1.2)}"
RGB_FOCAL="${RGB_FOCAL:-1008}"
RGB_HEIGHT="${RGB_HEIGHT:-1512}"
RECT="${RECT:-camera-rgb-rectified-${RGB_FOCAL}-h${RGB_HEIGHT}}"   # must match the preprocessing output folder
EXP_NAME="$SCENE/$RECT"                                            # model_path = OUTPUT_ROOT/EXP_NAME
CAP_MAX="${CAP_MAX:-1500000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output}"

if [ ! -f "$OUT_ROOT/$SCENE/$RECT/transforms_with_sparse_depth.json" ]; then
    echo "No preprocessed data at: $OUT_ROOT/$SCENE/$RECT" >&2
    echo "Run scripts/bash_local/run_gen2_outside.sh first, or check RECT." >&2
    exit 1
fi

# --- Memory ------------------------------------------------------------------
#
# 16 GB (RTX 4080 SUPER) is not enough for a ~4,700-frame outdoor scene with the
# `default` densification strategy. At 2016x1512 it grew to roughly 15M
# Gaussians and died in spherical_harmonics_bwd around iteration 10k -- which is
# also where handle_rolling_shutter starts rendering several samples per frame,
# compounding the peak.
#
# MCMC bounds the Gaussian count outright via cap_max, which is the reliable fix
# on a fixed VRAM budget.
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
# On a 24 GB+ GPU, drop data_factor and raise CAP_MAX for a sharper result.
python train_lightning.py \
    train_model=3dgs \
    opt=simple_gsplat_30K \
    opt.densification_strategy=MCMC \
    opt.mcmc_strategy.cap_max="$CAP_MAX" \
    opt.handle_rolling_shutter=true \
    scene.data_root="$OUT_ROOT" \
    scene.scene_name="$SCENE/$RECT" \
    scene.input_format="aria" \
    scene.data_factor=2 \
    scene.pcd_stride=2 \
    exp_name="$EXP_NAME" \
    output_root="$OUTPUT_ROOT" \
    viewer.use_trainer_viewer=false

echo
echo "Trained PLY:"
echo "  $OUTPUT_ROOT/$EXP_NAME/point_cloud/iteration_30000/point_cloud.ply"
