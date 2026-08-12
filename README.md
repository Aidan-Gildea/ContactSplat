# egocentric_splats — Aria Gen 2

Gaussian Splatting reconstruction from Project Aria recordings. Fork of
[facebookresearch/egocentric_splats](https://github.com/facebookresearch/egocentric_splats)
with Aria **Gen 2** support added.

Camera labels and rolling-shutter readout times are resolved from the recording, so Gen 1
and Gen 2 both work with the same commands. See [docs/gen2.md](docs/gen2.md) for what
changed.

Preprocessing runs on CPU. Training needs an NVIDIA GPU — `gsplat`'s rasterizer is CUDA,
with no CPU or Metal backend.

---

## 1. Setup

### Preprocessing only (any machine, no GPU)

```bash
conda create -n aria python=3.11
conda activate aria

pip install projectaria-tools projectaria-mps
pip install opencv-python numpy pandas pillow rerun-sdk tqdm
```

### Training (Linux + NVIDIA GPU)

```bash
conda create -n ego_splats python=3.10
conda activate ego_splats

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# gsplat builds CUDA extensions on first use
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. Record and run MPS

Record with **profile10** (RGB 2016x1512 @ 30 Hz). profile8 also works.

```bash
conda activate aria

# validate the recording first
run_vrs_health_check --path "my_recording.vrs"

# bundle-adjusted poses + semi-dense points (cloud, ~30 min)
aria_mps single -i "my_recording.vrs" --features SLAM
```

Produces `mps_my_recording_vrs/slam/` containing `closed_loop_trajectory.csv`,
`semidense_points.csv.gz`, `semidense_observations.csv.gz` and `online_calibration.jsonl`.
All four are required below.

---

## 3. Preprocess

```bash
# edit the paths at the top of the script first
bash scripts/bash_local/run_vrs_preprocessing_gen2.sh
```

Or call it directly:

```bash
MPS_FOLDER="/path/to/mps_my_recording_vrs/slam"

python scripts/extract_aria_vrs.py \
    --input_root  "/path/to/recordings" \
    --output_root "/path/to/processed" \
    --vrs_file    "my_recording.vrs" \
    --rectified_rgb_focal 1280 \
    --rectified_rgb_size 1920 \
    --rectified_monochrome_focal 180 --rectified_monochrome_height 512 \
    --online_calib_file  "$MPS_FOLDER/online_calibration.jsonl" \
    --trajectory_file    "$MPS_FOLDER/closed_loop_trajectory.csv" \
    --semi_dense_points_file      "$MPS_FOLDER/semidense_points.csv.gz" \
    --semi_dense_observation_file "$MPS_FOLDER/semidense_observations.csv.gz"
```

Quote all paths — Aria Studio puts spaces in recording names.

`--rectified_rgb_size` is the output **height**; width follows the source aspect ratio.
Horizontal FOV is `2 * atan((width / 2) / focal)`, so focal 1280 at width 2560 gives 90
degrees. Gen 2 RGB covers 133 degrees, more than a pinhole can represent, so some
cropping is unavoidable.

Optional flags:

```bash
--visualize                          # stream each stage to a rerun viewer
--overwrite                          # regenerate instead of skipping existing output
--extract_fisheye                    # equidistant fisheye output instead of pinhole
--timestamp_convention readout_start # reproduce upstream's half-readout pose bias
```

Output:

```
processed/my_recording/
├── camera-rgb-rectified-1280-h1920/        # training reads this
│   ├── images/  sparse_depth/
│   ├── transforms.json  transforms_with_sparse_depth.json
│   └── vignette.png  mask.png  image_index.png
└── slam-{front,side}-{left,right}-rectified-180-h512/
```

Check the run before training — these three must match the image size exactly:

```bash
cd processed/my_recording/camera-rgb-rectified-1280-h1920
python - <<'EOF'
from PIL import Image
import glob, json
print('image   ', Image.open(sorted(glob.glob('images/*.png'))[0]).size)
for f in ['vignette.png', 'mask.png', 'image_index.png']:
    print(f'{f:16s}', Image.open(f).size)
d = json.load(open('transforms_with_sparse_depth.json'))
print('frames  ', len(d['frames']))
print('readout ', d['frames'][0]['timestamp_read_end'] - d['frames'][0]['timestamp_read_start'], 'ns')
EOF
```

Console prints an observation match rate per SLAM camera. `0%` means sparse depth is empty
and depth supervision does nothing. ~33% is normal (MPS tracks at 10 Hz, SLAM records at
30 Hz).

---

## 4. Train

```bash
conda activate ego_splats

# scene_name must match the rectified folder name from step 3
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

Viewer at `http://0.0.0.0:8080` during training. Output PLY lands in
`output/<scene_name>/<exp>/point_cloud/iteration_30000/point_cloud.ply`.

Train on RGB only if you plan to export elsewhere. The SLAM-camera and joint RGB+mono
modes emit PLYs with 1 or 4 colour channels instead of 3, which no external tool reads —
including `3dgrut` for NuRec / Isaac Sim.

SLAM cameras are still used during preprocessing, where they generate the RGB sparse
depth.

---

## 5. View and render

```bash
# interactive viewer on a trained model
python launch_viewer.py model_root=output/my_recording/camera-rgb-rectified-1280-h1920/

# render a video
bash scripts/bash_local/run_aria_render.sh
```

---

## Capture notes

Reconstruction quality is set at capture time, not by any flag.

- Walk. Rotating in place gives no baseline and depth becomes unobservable.
- 2–5 minutes, multiple heights and angles.
- Revisit viewpoints so loop closure has something to close on.
- Gen 2 has no exposure-capped profile (Gen 1 used Profile 31). For motion blur, use a
  custom profile with `fixed_exposure` or `blur_filter_config`.
