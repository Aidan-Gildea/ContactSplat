# Running on Aria Gen 2 recordings

This repo was written and benchmarked for **Aria Gen 1**. This page covers the changes
needed for **Gen 2** and how to run the pipeline end to end.

The Python code auto-detects the device generation and resolves camera labels from the
recording, so the same commands work for both — only the rectification parameters and
paths differ.

## What is different on Gen 2

| | Gen 1 | Gen 2 |
|---|---|---|
| RGB camera | `camera-rgb`, **square** (2880² / 1408²) | `camera-rgb`, **4:3** (2560×1920 or 2016×1512) |
| SLAM cameras | `camera-slam-left`, `camera-slam-right` | `slam-front-left`, `slam-front-right`, `slam-side-left`, `slam-side-right` |
| SLAM resolution | 640×480 | 512×512 |
| Rolling-shutter readout | published per resolution | **read from MPS online calibration** |
| Devignetting assets | shipped in `data/` | none exist (ISP does lens shading on-device) |
| Projection model | `FISHEYE624` | `FISHEYE624` — unchanged, no math changes |

Readout time is no longer hardcoded anywhere. It is read per camera from
`online_calibration.jsonl` (`ReadoutTimesSec`), which makes the pipeline
**profile-agnostic** — profile8, profile10 and any custom profile self-describe.

---

## Setup

Preprocessing is **CPU only** and runs anywhere, including Apple Silicon. Training
requires an NVIDIA GPU: `gsplat`'s rasterizer is CUDA and has no CPU or Metal backend.

### Preprocessing only (any machine, no GPU)

```bash
conda create -n aria python=3.11
conda activate aria

# projectaria-tools brings the VRS reader, MPS loaders and the aria_* CLI tools
pip install projectaria-tools projectaria-mps
pip install opencv-python numpy pandas pillow rerun-sdk tqdm
```

### Full pipeline (Linux + NVIDIA GPU)

```bash
conda create -n ego_splats python=3.10
conda activate ego_splats

# Install pytorch (tested version). Choose a version compatible with your system.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

`gsplat` compiles CUDA extensions on first use, so `nvcc` must be on `PATH` and
`CUDA_HOME` must be set. Verify before training:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Record and process

Record with **profile10** (RGB 2016×1512 @ 30 Hz) — the highest RGB frame rate, which
matters most for splat coverage. profile8 (2560×1920 @ 10 Hz) also works.

```bash
conda activate aria

# 1. Validate the recording before spending time on it
run_vrs_health_check --path "my_recording.vrs"

# 2. Submit to MPS for bundle-adjusted poses + semi-dense points
aria_mps single -i "my_recording.vrs" --features SLAM
```

This produces the four inputs preprocessing needs:

```
mps_my_recording_vrs/slam/
├── closed_loop_trajectory.csv     # 1 kHz bundle-adjusted poses -- use this one
├── semidense_points.csv.gz        # sparse 3D points
├── semidense_observations.csv.gz  # which pixel saw which point
└── online_calibration.jsonl       # time-varying calibration + READOUT TIMES
```

---

## Preprocess the VRS

```bash
bash scripts/bash_local/run_vrs_preprocessing_gen2.sh
```

Edit the paths at the top of that script first. Or call the tool directly:

```bash
python scripts/extract_aria_vrs.py \
    --input_root "/path/to/recordings" \
    --output_root "/path/to/processed" \
    --vrs_file "my_recording.vrs" \
    --rectified_rgb_focal 1280 \
    --rectified_rgb_size 1920 \
    --rectified_monochrome_focal 180 --rectified_monochrome_height 512 \
    --online_calib_file  "$MPS_FOLDER/online_calibration.jsonl" \
    --trajectory_file    "$MPS_FOLDER/closed_loop_trajectory.csv" \
    --semi_dense_points_file      "$MPS_FOLDER/semidense_points.csv.gz" \
    --semi_dense_observation_file "$MPS_FOLDER/semidense_observations.csv.gz"
```

> **Quote your paths.** Aria Studio names recordings with spaces in them
> (e.g. `My recording_20260101_120000.vrs`), which breaks unquoted shell expansion.

### Choosing the rectification parameters

`--rectified_rgb_size` is the output **height**; the width follows the source aspect
ratio (on Gen 1 it forced a square, which was correct only for a square sensor).

Horizontal FOV of the rectified pinhole image is:

```
FOV = 2 * atan((width / 2) / focal)
```

Gen 2 RGB has a 133°×99° field of view, far more than a pinhole can represent, so pick a
narrower target. `--rectified_rgb_focal 1280 --rectified_rgb_size 1920` gives 2560×1920 at
**90° horizontal**, matching what the Gen 1 defaults produced (2400 wide at focal 1200).
For SLAM, focal 180 at 512 wide gives ~110° of the sensor's 119°.

### Useful flags

```bash
--visualize                          # stream each stage to a rerun viewer
--overwrite                          # regenerate instead of skipping existing output
--extract_fisheye                    # rectify to equidistant fisheye instead of pinhole
--timestamp_convention readout_start # reproduce upstream's half-readout pose bias
```

`--timestamp_convention` defaults to `center`. Aria documents `capture_timestamp_ns` as
the center of exposure of the middle row; this repo originally treated it as the start of
readout, which biases every pose by half a readout — 8 ms on Gen 1 full-res RGB, **19 ms
on Gen 2 profile8**. Use `readout_start` only to A/B against the published results.

### Expected output

```
processed/my_recording/
├── camera-rgb-images/                       # raw extracted frames
├── camera-rgb-transforms.json
├── camera-rgb-rectified-1280-h1920/         # <- training consumes this
│   ├── images/                              # rectified PNGs
│   ├── sparse_depth/                        # per-frame sparse depth
│   ├── transforms.json
│   ├── transforms_with_sparse_depth.json
│   ├── vignette.png, mask.png, image_index.png
│   ├── semidense_points.csv.gz       -> symlink
│   └── closed_loop_trajectory.csv    -> symlink
└── slam-{front,side}-{left,right}-rectified-180-h512/
```

Sanity-check the run — `vignette.png`, `mask.png` and `image_index.png` must match the
rectified image size exactly, or training fails later with a broadcast error:

```bash
cd processed/my_recording/camera-rgb-rectified-1280-h1920
python - <<'EOF'
from PIL import Image
import glob, json
print('image   ', Image.open(sorted(glob.glob('images/*.png'))[0]).size)
for f in ['vignette.png', 'mask.png', 'image_index.png']:
    print(f'{f:16s}', Image.open(f).size)
d = json.load(open('transforms_with_sparse_depth.json'))
fr = d['frames'][0]
print('readout ns', fr['timestamp_read_end'] - fr['timestamp_read_start'])
print('frames    ', len(d['frames']))
EOF
```

Watch the console for the observation match rate. `0%` means sparse depth is silently
empty and depth supervision is a no-op:

```
==> slam-front-left: matched observations for 271/813 frames (33.3%)
```

A third is normal — MPS tracks at 10 Hz while the SLAM cameras record at 30 Hz.

Fewer output frames than the VRS contains is also normal: the MPS trajectory starts after
tracking initialises and ends early, so RGB frames outside that window have no pose.

---

## Train

Requires CUDA. `scene_name` must match the rectified folder name from preprocessing.

```bash
conda activate ego_splats

python train_lightning.py \
    train_model=3dgs opt=simple_gsplat_30K \
    opt.densification_strategy=default \
    opt.handle_rolling_shutter=true \
    scene.data_root=/path/to/processed \
    scene.scene_name="my_recording/camera-rgb-rectified-1280-h1920" \
    scene.input_format="aria" \
    output_root=./output \
    viewer.use_trainer_viewer=true
```

A viewer serves at `http://0.0.0.0:8080` during training.

**Train on RGB only** if the goal is exporting to another tool. The SLAM-camera and joint
modes emit PLYs with 1 or 4 colour channels (`f_dc_0..3`) instead of the standard 3, which
no external viewer or converter reads — including `3dgrut` for NuRec/Isaac Sim export.

The SLAM cameras still earn their place in preprocessing, where they generate the RGB
sparse depth. Measured contribution on a Gen 2 profile8 recording:

| Camera | points in SLAM view | landing in RGB frustum | yield |
|---|---|---|---|
| `slam-front-left` | 35,827 | 21,036 | 58.7% |
| `slam-front-right` | 33,542 | 15,869 | 47.3% |
| `slam-side-left` | 28,717 | 4,421 | 15.4% |
| `slam-side-right` | 37,896 | 1,860 | 4.9% |

The side pair adds ~17% more depth points than the front pair alone.

---

## Capture technique

Reconstruction quality is set at capture time, not by any flag:

- **Translate, don't just rotate.** Rotating in place gives no baseline, so depth is close
  to unobservable. Walk.
- **2–5 minutes**, covering multiple heights and angles.
- **Revisit viewpoints** so loop closure has something to close on.
- **Avoid motion blur.** Gen 2 has no equivalent of Gen 1's Profile 31 exposure cap —
  `auto_exposure` sub-fields are ignored. A custom profile with `fixed_exposure`
  (`exposure_us: 3000`) or `blur_filter_config` is the Gen 2 answer.

## Known gaps

- **No Gen 2 devignetting assets exist.** The port applies a neutral (all-ones) vignette
  rather than stretching the Gen 1 IMX577/OV7251 masks, which are the wrong sensors at the
  wrong resolution and aspect. Gen 2 does lens-shading correction in the ISP instead
  (`rgb_camera.lsc` in a custom profile).
- **profile10 readout time is undocumented**, but it does not need to be: MPS reports
  whatever the recording used, so it arrives in the data.
- **Whether `capture_timestamp_ns` marks the middle-row exposure centre on Gen 2 is
  documented only for Gen 1.** The Gen 2 MPS spec says "center of exposure" without
  specifying the row. Confirm with AriaOps@meta.com if you need certainty at the ~19 ms
  level.
