# egocentric_splats — Aria Gen 2

Gaussian Splatting reconstruction from Project Aria recordings. Fork of
[facebookresearch/egocentric_splats](https://github.com/facebookresearch/egocentric_splats)
with Aria **Gen 2** support added.

Camera labels and rolling-shutter readout times are resolved from the recording, so Gen 1
and Gen 2 both work with the same commands. See [docs/gen2.md](docs/gen2.md) for what
changed, and [docs/course.html](docs/course.html) for a ground-up walkthrough of how the
pipeline works.

---

## Pipeline

```
┌─ Block 1 ─────────────────────────────────────┐   ┌─ Block 2 ──┐   ┌─ Block 3 ────┐
│ env → record → MPS → extract_aria_vrs → verify│ → │   train    │ → │ view · render │
│ CPU · ~3 h · ~50 GB                           │   │ GPU · ~6 h │   │ export        │
└───────────────────────────────────────────────┘   └────────────┘   └──────────────┘
```

| Block | Needs | Produces |
|---|---|---|
| **1 · Dataset** | CPU only (any machine, incl. Apple Silicon) | `processed/<recording>/camera-rgb-rectified-*/` |
| **2 · Train** | Linux + NVIDIA GPU | `point_cloud.ply` |
| **3 · Use** | GPU for the viewer | video, USDZ / NuRec |

Training needs CUDA. The `gsplat` rasterizer has no CPU or Metal backend.

---

## Block 1: Recording to training-ready dataset
The purpose of this block is to generate a training-ready dataset from your recorded vrs data. 

### 1.1 Environment

Only one conda environment is used in the whole pipeline. `requirements.txt` already includes
`projectaria-tools`, `opencv-python`, `pandas`, `rerun-sdk` and `gsplat`.

```bash
conda create -n ego_splats python=3.10 -y
conda activate ego_splats
cd /path/to/egocentric_splats

# torch FIRST, from the CUDA index matching your driver (see `nvidia-smi`).
# requirements.txt asks only for torch>=2.6.0, so installing it first keeps this build.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install projectaria-mps          # the aria_mps CLI, not in requirements.txt

# verify
python -c "import torch, cv2, projectaria_tools; print(torch.__version__, torch.cuda.is_available())"
```

That last line must print your pytorch version and `True`. `False` means PyTorch cannot see
the GPU, which almost always means you installed wrong torch build, so reinstall it from the CUDA index
that matches the version `nvidia-smi` reports. 


### 1.2 Set your paths
You must re-set your paths once per shell instance. 

```bash
conda activate ego_splats
cd /path/to/egocentric_splats

export REC_ROOT="/path/to/recordings" # Path to the folder containing vrs and mps outputs
export SCENE="my_recording" # .vrs filename WITHOUT the extension
export MPS_FOLDER="$REC_ROOT/mps_${SCENE}_vrs/slam" # a derived value using the REC_ROOT
export OUT_ROOT="$REC_ROOT/processed"  
mkdir -p "$OUT_ROOT"

ls -la "$REC_ROOT/$SCENE.vrs"     # confirm before spending hours
```

Quote every path as Aria Studio puts spaces in recording names. `${SCENE}` needs
braces because `_vrs` follows immediately, whereas `$REC_ROOT/...` does not, because `/`
cannot be part of a variable name.


### 1.3 Record and run MPS

<u>**If you have already generated your MPS data using Aria Studio or some other method, you can skip this step.**</u>

**Record with profile10** (RGB 2016x1512 at 30 Hz). It gives the highest RGB frame rate,
which matters more than resolution for splat coverage, and it is the mode the default
values in 1.4 are calculated for. profile8 and custom profiles work too, but you will have
to recalculate the two rectification values first. See
[docs/rectification.md](docs/rectification.md).

```bash
run_vrs_health_check --path "$REC_ROOT/$SCENE.vrs"

# bundle-adjusted poses + semi-dense points (cloud, ~30 min)
aria_mps single -i "$REC_ROOT/$SCENE.vrs" --features SLAM
```

Produces `$MPS_FOLDER` containing four files, **all required**:

```
closed_loop_trajectory.csv      1 kHz bundle-adjusted device poses
online_calibration.jsonl        time-varying calibration + readout times
semidense_points.csv.gz         3D points, Gaussian init and depth values
semidense_observations.csv.gz   which point was seen in which frame
```

Confirm the recording and its MPS output together:

```bash
python debug_scripts/recording_info.py --vrs "$REC_ROOT/$SCENE.vrs" 2>/dev/null
```

That reports the device generation, the RGB resolution and frame count, the readout time
of every camera, which of the four MPS files are present, and the rectification flags to
use in the next step.

### 1.4 Preprocess

If you recorded profile10, use the values below as they are. For any other sensor mode,
work out your own pair first with
[docs/rectification.md](docs/rectification.md) and `debug_scripts/solve_focal.py`.

```bash
export RGB_FOCAL=1008        # profile10 at 2016x1512, 90 degrees horizontal
export RGB_HEIGHT=1512       # output HEIGHT, width follows the source aspect ratio

python scripts/extract_aria_vrs.py \
    --input_root  "$REC_ROOT" \
    --output_root "$OUT_ROOT" \
    --vrs_file    "$SCENE.vrs" \
    --rectified_rgb_focal         "$RGB_FOCAL" \
    --rectified_rgb_size          "$RGB_HEIGHT" \
    --rectified_monochrome_focal  180 \
    --rectified_monochrome_height 512 \
    --online_calib_file           "$MPS_FOLDER/online_calibration.jsonl" \
    --trajectory_file             "$MPS_FOLDER/closed_loop_trajectory.csv" \
    --semi_dense_points_file      "$MPS_FOLDER/semidense_points.csv.gz" \
    --semi_dense_observation_file "$MPS_FOLDER/semidense_observations.csv.gz" \
    2>&1 | tee "$OUT_ROOT/preprocess_$SCENE.log"
```

Or edit the paths at the top of
[`scripts/bash_local/run_vrs_preprocessing_gen2.sh`](scripts/bash_local/run_vrs_preprocessing_gen2.sh)
and run that instead. Be aware that it ships the profile8 pair, `1280` and `1920`, which
would resample a profile10 recording upward for no benefit.

Both focal flags default to `-1`, which means skip that camera. Omit them and the run
finishes in seconds having produced nothing.

Optional flags:

```bash
--visualize                          # stream each stage to a rerun viewer
--overwrite                          # regenerate instead of skipping existing output
--extract_fisheye                    # equidistant fisheye output instead of pinhole
--timestamp_convention readout_start # reproduce upstream's half-readout pose bias
```

Re-running is safe, because every stage skips output that already exists. 
Three console lines are worth keeping, which is what the `tee` is for:

```
Detected Gen2 device: rgb=camera-rgb, slam=[...]        labels resolved
camera-rgb: a total of 4676 number of frames.           slightly under the VRS count is normal
==> slam-front-left: matched observations for 1571/4717 frames (33.3%)
```

About 33% is correct, because MPS tracks at 10 Hz while the cameras record at
30 Hz. **`0%` means sparse depth is empty and depth supervision does nothing.** There is
one such line per SLAM camera.

### 1.5 Verify before spending GPU hours

```
processed/my_recording/
├── camera-rgb-rectified-1008-h1512/        # training reads this
│   ├── images/  sparse_depth/
│   ├── transforms.json  transforms_with_sparse_depth.json
│   ├── vignette.png  mask.png  image_index.png
│   ├── semidense_points.csv.gz     -> symlink
│   └── closed_loop_trajectory.csv  -> symlink
└── slam-{front,side}-{left,right}-rectified-180-h512/
```

```bash
python debug_scripts/verify_preprocessing.py \
    "$OUT_ROOT/$SCENE/camera-rgb-rectified-${RGB_FOCAL}-h${RGB_HEIGHT}"
```

It checks the four things that actually break training: 
- that `vignette.png`, `mask.png`
and `image_index.png` match the rectified image size
- that `transforms_with_sparse_depth.json` exists and holds frames
- that the readout time is
non-zero for an RGB camera
- that sparse depth files hold points. 

It samples frames
across the recording rather than trusting frame zero, which sits at the edge of the
trajectory window and is the least representative. The exit code is non-zero if anything
fails.

Fewer frames than the VRS contains is normal. The MPS trajectory starts after tracking
initialises, so frames outside that window have no pose and are dropped.

---

## Block 2: Train

Same environment as Block 1, but this half needs CUDA. `gsplat` compiles its kernels on the
first run, which adds several minutes once.

```bash
conda activate ego_splats

export RECT="camera-rgb-rectified-${RGB_FOCAL}-h${RGB_HEIGHT}"   # must match Block 1 exactly

# scene_name must match the rectified folder name from Block 1
python train_lightning.py \
    train_model=3dgs \
    opt=simple_gsplat_30K \
    opt.densification_strategy=default \
    opt.handle_rolling_shutter=true \
    scene.data_root="$OUT_ROOT" \
    scene.scene_name="$SCENE/$RECT" \
    scene.input_format="aria" \
    exp_name="$SCENE/$RECT" \
    output_root=./output \
    viewer.use_trainer_viewer=true
```

Viewer at `http://0.0.0.0:8080` during training. Set `viewer.use_trainer_viewer=false` for
a production run. The PLY lands in
`output/<exp_name>/point_cloud/iteration_30000/point_cloud.ply`.

Held-out PSNR, SSIM and LPIPS come free, because `scene.train_split: "7-1"` holds out every
8th frame automatically. Record them.

**Train on RGB only if you plan to export elsewhere.** The SLAM-camera and joint RGB+mono
modes emit PLYs with 1 or 4 colour channels instead of 3, which no external tool reads,
including `3dgrut` for NuRec and Isaac Sim. Colour format is chosen from the cameras
present in `scene.scene_name`, so pointing it at a single RGB rectified folder is what
selects it.

SLAM cameras still matter in Block 1, though, where they are used to generate the RGB sparse depth point cloud. 

## Block 3: View, render, export

```bash
# interactive viewer on a trained model
python launch_viewer.py model_root="output/$SCENE/$RECT/"

# render a video
bash scripts/bash_local/run_aria_render.sh
```

The PLY is the handoff point to `3dgrut` for NuRec and Isaac Sim.

---

## Helper scripts

Small standalone tools, none of which need arguments beyond a path.

| Script | Purpose |
|---|---|
| [`debug_scripts/recording_info.py`](debug_scripts/recording_info.py) | Device generation, RGB resolution and frame count, readout times, MPS file check, suggested flags |
| [`debug_scripts/solve_focal.py`](debug_scripts/solve_focal.py) | Solve focal length for a target field of view, or the reverse |
| [`debug_scripts/verify_preprocessing.py`](debug_scripts/verify_preprocessing.py) | Check a rectified folder before training on it |

---

## Capture notes

Reconstruction quality is defined by what you do while capturing your vrs.

- Walk around - rotating in place gives no baseline and depth becomes unobservable.
- 2 to 5 minutes, multiple heights and angles.
- Revisit viewpoints so loop closure has something to close on.
- Gen 2 has no exposure-capped profile, where Gen 1 used Profile 31. For motion blur, use
  a custom profile with `fixed_exposure` or `blur_filter_config`.
. 