# Full pass: Aria Gen 2 VRS → 3D Gaussian Splat → Isaac Sim

End-to-end run of this repo on a real Gen 2 recording, on this machine.
Assumes MPS has **already** been run and its output is on disk.

Everything below was actually executed, not written from the docs. Scripts live in
`scripts/bash_local/`.

## The whole thing, if you just want the commands

```bash
conda activate ego_splats
cd /home/sun/Desktop/aria_proj/egocentric_splats

bash scripts/bash_local/run_gen2_outside.sh          # VRS + MPS -> rectified frames + sparse depth  (~13 min)
bash scripts/bash_local/train_gen2_outside.sh        # -> point_cloud.ply                            (30k iters)
bash scripts/bash_local/export_gen2_outside_usdz.sh  # -> Isaac-Sim-ready .usdz                      (uses the 3dgrut env)

~/isaac-sim/python.sh scripts/isaacsim_load_splat.py \
    output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/isaacsim/Outside_20260812_141244.usdz --gui
```

Each script has its paths at the top. Sections below explain every parameter, and the
sharp edges are in **VRAM** (§3), **outlier filtering** (Results) and **depth supervision
is off by default**.

---

## 0. What you need before starting

### Input data

```
/home/sun/aria/
├── Outside_20260812_141244.vrs                    # 2.1 GB raw recording
└── mps_Outside_20260812_141244_vrs/
    ├── vrs_health_check.json
    └── slam/
        ├── closed_loop_trajectory.csv             # required
        ├── online_calibration.jsonl               # required
        ├── semidense_points.csv.gz                # required
        ├── semidense_observations.csv.gz          # required
        ├── open_loop_trajectory.csv               # not used by this pipeline
        └── summary.json
```

Only the four marked *required* are consumed. `open_loop_trajectory.csv` is ignored —
closed-loop is the bundle-adjusted one.

### Environments

Two conda envs, used at different steps. This matters:

| Env | Used for | Why |
|---|---|---|
| `ego_splats` | preprocessing **and** training | has `projectaria_tools`, `cv2`, `torch`+CUDA |
| `3dgrut` | USDZ export only | NVIDIA's NuRec exporter lives in the 3dgrut repo |

The `aria` / `aria2` envs **cannot** run preprocessing — they have `projectaria_tools`
but no `cv2`, and `scripts/aria_utils.py` imports OpenCV at module load, so it dies
immediately. Use `ego_splats` throughout.

### This recording

Read out of the VRS/MPS rather than assumed:

| | |
|---|---|
| Device | Aria **Gen 2** |
| Duration / path length | 2:37 / 100.2 m |
| RGB | `camera-rgb`, 2016×1512 @ 30 Hz → **profile10** |
| RGB frames | 4,719 |
| SLAM | 4 cameras, 512×512, 4,717 frames each |
| RGB rolling-shutter readout | **10.10 ms** (from `online_calibration.jsonl`) |
| RGB native fisheye focal | 878.3 |
| Trajectory | 155,800 poses @ 1 kHz, 161.53 s → 317.33 s |

---

## 1. Preprocess the VRS

```bash
conda activate ego_splats
cd /home/sun/Desktop/aria_proj/egocentric_splats
bash scripts/bash_local/run_gen2_outside.sh
```

That script wraps one call to `scripts/extract_aria_vrs.py`:

```bash
python scripts/extract_aria_vrs.py \
    --input_root  "/home/sun/aria" \
    --output_root "/home/sun/aria/processed" \
    --vrs_file    "Outside_20260812_141244.vrs" \
    --rectified_rgb_focal         1008 \
    --rectified_rgb_size          1512 \
    --rectified_monochrome_focal  180 \
    --rectified_monochrome_height 512 \
    --online_calib_file           "$MPS_FOLDER/online_calibration.jsonl" \
    --trajectory_file             "$MPS_FOLDER/closed_loop_trajectory.csv" \
    --semi_dense_points_file      "$MPS_FOLDER/semidense_points.csv.gz" \
    --semi_dense_observation_file "$MPS_FOLDER/semidense_observations.csv.gz"
```

### What each parameter does

| Parameter | Meaning |
|---|---|
| `--input_root` | Folder containing the `.vrs`. Joined with `--vrs_file` to locate it. |
| `--vrs_file` | Recording filename. Its stem also names the output folder. |
| `--output_root` | Output goes to `<output_root>/<vrs stem>/`. |
| `--rectified_rgb_focal` | Focal length (px) of the **output pinhole** RGB camera. Sets FOV. `<0` skips RGB entirely. |
| `--rectified_rgb_size` | Output **height** in px. Width is derived from the source aspect ratio (4:3 here → 2016). |
| `--rectified_monochrome_focal` | Same, for the SLAM cameras. `<0` skips all SLAM cameras. |
| `--rectified_monochrome_height` | Output height for SLAM. |
| `--online_calib_file` | MPS time-varying calibration. Source of intrinsics, extrinsics **and readout time**. |
| `--trajectory_file` | Closed-loop trajectory; interpolated to each frame's timestamp for pose. |
| `--semi_dense_points_file` | Global 3D point cloud. Becomes the splat initialisation and depth values. |
| `--semi_dense_observation_file` | uid ↔ (camera, frame, u, v) link table. Drives sparse depth. Optional — omitting it skips depth. |

Optional flags worth knowing:

| Flag | Effect |
|---|---|
| `--overwrite` | Regenerate instead of skipping existing output. Needed to re-run after changing parameters. |
| `--visualize` | Stream each stage to a rerun viewer. |
| `--extract_fisheye` | Emit equidistant fisheye instead of pinhole. |
| `--timestamp_convention readout_start` | Reproduce upstream's half-readout pose bias. Default `center` is correct. |
| `--use_factory_calib` | Use factory instead of online calibration. |

### Choosing the rectification numbers

The horizontal FOV of the output pinhole is:

```
FOV = 2 * atan((width / 2) / focal)
```

`--rectified_rgb_size` is the **height**; width follows the source aspect ratio. At
2016 wide, focal 1008 gives exactly 90°.

The checked-in `run_vrs_preprocessing_gen2.sh` uses `focal 1280 / size 1920`, which was
derived for **profile8** (2560×1920). This recording is **profile10** (2016×1512), so
those numbers resample every frame *upward*. Measured on a real frame from this
recording:

| focal / height | output | horiz FOV | PNG per frame |
|---|---|---|---|
| 1280 / 1920 | 2560×1920 | 90.0° | 4.19 MB |
| **1008 / 1512** | **2016×1512** | **90.0°** | **2.94 MB** |
| 756 / 1512 | 2016×1512 | 106.3° | 3.18 MB |

Same FOV, 30 % less disk and training cost, no information lost. Match the height to
the sensor's native height and scale the focal with it.

### What it produces

```
/home/sun/aria/processed/Outside_20260812_141244/
├── camera-rgb-images/                        # raw extracted frames (jpg)
├── camera-rgb-transforms.json                # fisheye intrinsics + poses
├── camera-rgb-rectified-1008-h1512/          # <- training consumes this
│   ├── images/                               # rectified pinhole PNGs
│   ├── sparse_depth/                         # per-frame projected depth
│   ├── transforms.json
│   ├── transforms_with_sparse_depth.json     # <- the file training reads
│   ├── vignette.png  mask.png  image_index.png
│   ├── semidense_points.csv.gz     -> symlink into the MPS folder
│   └── closed_loop_trajectory.csv  -> symlink into the MPS folder
└── slam-{front,side}-{left,right}-rectified-180-h512/
```

Those last two symlinks matter: **training reads the point cloud and trajectory from
the rectified folder, not from your MPS directory.** Moving or renaming the MPS folder
later breaks training.

---

## 2. Verify before training

```bash
cd /home/sun/aria/processed/Outside_20260812_141244/camera-rgb-rectified-1008-h1512
python - <<'EOF'
from PIL import Image
import glob, json
imgs = sorted(glob.glob('images/*.png'))
size = Image.open(imgs[0]).size
print('images   ', len(imgs), size)
for f in ['vignette.png', 'mask.png', 'image_index.png']:
    s = Image.open(f).size
    print(f'{f:17s}', s, 'OK' if s == size else '<-- MISMATCH, training will crash')
d = json.load(open('transforms_with_sparse_depth.json'))
fr = d['frames'][0]
print('frames   ', len(d['frames']))
print('readout  ', (fr['timestamp_read_end'] - fr['timestamp_read_start']) / 1e6, 'ms')
sd = json.load(open(fr['sparse_depth']))
print('depth pts', len(sd['z']))
EOF
```

`vignette.png`, `mask.png` and `image_index.png` must match the image size **exactly** —
they are multiplied against the image during training, and a mismatch is a broadcast
error thousands of iterations in.

Also check the console output of step 1 for the per-camera match rate:

```
==> slam-front-left: matched observations for 1559/4674 frames (33.4%)
```

~33 % is correct (MPS tracks at 10 Hz, the cameras record at 30 Hz). **0 % means sparse
depth is empty and depth supervision is silently doing nothing.**

Fewer output frames than the VRS contains is also expected: the MPS trajectory starts
after tracking initialises, so RGB frames outside that window have no pose and are
dropped (4,719 → 4,675 here).

---

## 3. Train

```bash
conda activate ego_splats
bash scripts/bash_local/train_gen2_outside.sh
```

which runs:

```bash
python train_lightning.py \
    train_model=3dgs \
    opt=simple_gsplat_30K \
    opt.densification_strategy=default \
    opt.handle_rolling_shutter=true \
    scene.data_root="/home/sun/aria/processed" \
    scene.scene_name="Outside_20260812_141244/camera-rgb-rectified-1008-h1512" \
    scene.input_format="aria" \
    exp_name="Outside_20260812_141244/camera-rgb-rectified-1008-h1512" \
    output_root=./output \
    viewer.use_trainer_viewer=false
```

### What each parameter does

| Parameter | Meaning |
|---|---|
| `train_model=3dgs` | Selects `VanillaGSplat` (3D Gaussians). `2dgs` selects the 2D surfel model. |
| `opt=simple_gsplat_30K` | Optimisation preset in `conf/opt/`. 30,000 iterations, SH degree 3. |
| `opt.densification_strategy=MCMC` | Bounds the Gaussian count via `cap_max`. **Required on a 16 GB GPU** — see below. The documented `default` strategy OOMs on this scene. |
| `opt.mcmc_strategy.cap_max=3000000` | Hard ceiling on Gaussian count. |
| `opt.handle_rolling_shutter=true` | Model the RGB rolling shutter. Uses the per-frame readout time from `transforms.json` (10.10 ms here). Kicks in at iteration 10,000. |
| `scene.data_root` | Root of preprocessed output. |
| `scene.scene_name` | **Must** be `<recording>/<rectified folder>` — exactly two path components, and the folder name encodes the focal/height from step 1. |
| `scene.input_format="aria"` | Use the Aria reader (poses, rolling shutter, vignette) rather than COLMAP. |
| `exp_name` | Output goes to `output_root/exp_name`. Leave it unset and everything lands in `./output/` root. |
| `output_root` | Base output directory. |
| `viewer.use_trainer_viewer` | Live viewer on `0.0.0.0:8080` during training. Off for an unattended run. |

Useful extras:

| Parameter | Effect |
|---|---|
| `scene.data_factor=2` | Downscale images 2× at load. Cuts VRAM and time if you OOM. |
| `scene.train_split="7-1"` | Default. Holds out every 8th frame for validation/test. `4-1` holds out every 5th. |
| `opt.iterations=7000` | Shorter run for a quick sanity check. |
| `opt.depth_loss=true` | Actually supervise on the sparse depth (off by default — see below). |

### VRAM: the documented command does not fit in 16 GB

Running exactly the README command on this scene (4,675 frames at 2016×1512, RTX 4080
SUPER, 16 GB) **fails**:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.88 GiB
  ... in gsplat/cuda/_wrapper.py, spherical_harmonics_bwd
```

It died about an hour in. A 2.88 GiB allocation in the SH backward pass at SH degree 3
implies roughly **15 M Gaussians** — `default` densification ran away on a foliage-heavy
outdoor scene. It also compounded at iteration 10,000, which is where
`handle_rolling_shutter_start_iter` starts rendering several samples per frame.

Fragmentation was a factor but not the cause: 15 M Gaussians need ~14 GB for parameters,
gradients and Adam state alone. The fix is to bound the count:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # OOM reported 7.3 GB reserved-but-unallocated
opt.densification_strategy=MCMC opt.mcmc_strategy.cap_max=3000000
```

**Watch out:** the run exits with **code 0** even when it crashes this way — Hydra
swallows the exception's status, so `set -e` does not catch it and a wrapper script
reports success. Check for a PLY on disk, not the exit code.

Training holds out every 8th frame automatically, and since there is no `test_views/`
folder it reuses that validation split as the test set — so the final `trainer.test()`
reports held-out metrics. That is your validation number.

Output:

```
output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/
└── point_cloud/iteration_30000/point_cloud.ply
```

**Train on RGB only.** The SLAM-camera and joint modes write PLYs with 1 or 4 colour
channels (`f_dc_0..3`) instead of 3, which the 3dgrut importer — and every external
viewer — cannot read. The SLAM cameras still earn their place in preprocessing, where
they generate the RGB sparse depth.

---

## 4. Export to USDZ for Isaac Sim

```bash
bash scripts/bash_local/export_gen2_outside_usdz.sh
```

which runs, **in the 3dgrut env**:

```bash
cd /home/sun/3dgrut
/home/sun/miniforge3/envs/3dgrut/bin/python \
    threedgrut/export/scripts/ply_to_usd.py \
    "<path>/point_cloud/iteration_30000/point_cloud.ply" \
    --output_file "<path>/isaacsim/Outside_20260812_141244.usdz"
```

| Parameter | Meaning |
|---|---|
| positional arg | Input PLY. Must be standard 3-channel 3DGS. |
| `--output_file` | Output USDZ. Defaults to the input path with `.usdz`. |

Notes:

- Must run from the 3dgrut repo root — the script resolves its hydra config relative to
  its own location.
- The first run compiles 3dgrut's CUDA extensions (a few minutes). Later runs are seconds.
- `ply_to_usd.py` passes `dataset=None`, so the normalizing transform is **not** applied.
  The splat keeps its original world coordinates.

### Why no coordinate conversion is needed

The MPS world frame is gravity-aligned and Z-up — confirmed on this recording, where the
trajectory spans ~10 m in X and Y but only 1.66 m in Z (head height while walking). The
exported USDZ declares `upAxis = Z` and `metersPerUnit = 1.0`, which is Isaac Sim's
default stage convention. So the splat drops in at the right scale and orientation with
no extra transform.

This is the one place a conversion *could* have been needed. Aria's **camera** convention
(X right, Y down, Z forward) matches COLMAP's, so there is no conversion inside the repo
either.

---

## 5. Import into Isaac Sim

The USDZ contains a `Volume` prim with `OmniNuRecFieldAsset` children. NuRec rendering is
built into the RTX Hydra delegate in Isaac Sim 5.1 (this machine has
`5.1.0-rc.19`), so no extra extension is required.

Manually:

1. Launch Isaac Sim: `~/isaac-sim/isaac-sim.sh`
2. **File → Open** the `.usdz`, or drag it from the Content browser onto the stage to
   reference it into an existing scene.
3. Select an **RTX** renderer (RTX – Real-Time or RTX – Interactive). The splat will not
   appear under the Storm/preview delegate.
4. Frame the stage (`F`). The splat sits at MPS world coordinates, which are not
   centred on the origin — if you see nothing, frame the prim rather than assuming it
   failed.

The asset is at:

```
output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/isaacsim/Outside_20260812_141244.usdz
```

Or scripted, which also serves as a smoke test that the asset resolves in the renderer:

```bash
~/isaac-sim/python.sh scripts/isaacsim_load_splat.py <path to .usdz>
~/isaac-sim/python.sh scripts/isaacsim_load_splat.py <path to .usdz> --gui   # with the GUI
```

It references the USDZ under `/World/AriaSplat`, steps the renderer, and reports the
resolved prims and world-space bounds.

Pass `--report out.json` to get the result as a file. Do that rather than reading the
console: **Kit captures Python `stdout` into its own logger and force-exits on shutdown**,
so on a headless run neither the printed output nor the process exit code is trustworthy
evidence. The first run here exited 0 having printed nothing at all.

Verified result for this asset:

```json
{ "ok": true, "up_axis": "Z", "meters_per_unit": 1.0, "nurec_prim_count": 2,
  "world_aabb_min": [-49.07, -47.63, -48.66],
  "world_aabb_max": [ 48.06,  49.76,  48.53] }
```

Both `OmniNuRecFieldAsset` prims resolve, and the world bounds agree with the filtered
PLY — so the asset loads in Isaac Sim's renderer at the right scale.

Splats render as a volume: they have no collision geometry. To use it as a robot
environment, add collision proxies separately.

---

## Bugs fixed during this pass

Committed alongside this document.

**1. Behind-camera points were not rejected** (`utils/point_utils.py`)

```python
# before
if u < 0 or u >= w-1 or v < 0 or v >= h-1 and z > 0:
```

`and` binds tighter than `or`, so `z > 0` applied only to the final clause. A point behind
the camera projects to a sign-flipped `u,v` that can land inside the image, and would be
accepted with a **negative depth**. The vectorised branch immediately above tests `z > 0`
separately, which is what makes the scalar version unintended.

**Measured impact on this recording: zero.** The fix is instrumented, and it rejected no
points on any of the four SLAM cameras. That is expected in hindsight — the MPS
observation table only lists points a camera actually *saw*, so they are in front of it by
construction. Verified independently: no negative `z` in any sampled sparse-depth file.

So this is a latent correctness fix, not a repair of corrupted data. It matters for any
code path that projects arbitrary points rather than observed ones — which is exactly what
the RGB stage does when it re-projects one camera's points into another's frustum.

**2. `project()` misrouted small batches** (`utils/point_utils.py`)

It branched on `point3d.shape[-1] > 1`, so a batch that happened to contain exactly one
point — or zero — fell into the scalar branch and returned `mask=None`, which every
batched caller then used as an index. Now branches on rank. The scalar path was also
dead-broken for genuine 1-D input (`T_w2c[:3, 3:]` is a column, so `(3,) + (3,1)`
broadcast to `(3,3)`); it is now handled properly.

**3. Sparse depth was computed twice per SLAM camera** (`scripts/extract_aria_vrs.py`)

`create_visible_depth_map` ran once inside the rectification loop and again in the
dedicated pass below it, and the first result was written to a `transforms.json` that the
second pass immediately re-read and recomputed from scratch. This doubled the single most
expensive stage of preprocessing. Now computed once. (Pre-existing upstream, not
introduced by the Gen 2 port.)

**4. The per-point projection loop is now vectorised** (`scripts/extract_aria_vrs.py`)

It called `project()` once per observed point — ~5,700 points × ~6,300 matched frames.
The semi-dense map is now flattened into parallel arrays once and each frame projects in
a single batched operation.

**5. `densification_strategy=MCMC` could not start at all** (`model/vanilla_gsplat.py`)

`_create_strategy` read `opt.refine_start_iter`, `opt.refine_stop_iter` and
`opt.min_opacity` out of the `mcmc_strategy` config block, but that block defines them as
`mcmc_refine_start_iter`, `mcmc_refine_stop_iter` and `mcmc_min_opacity`. Selecting MCMC
therefore raised:

```
omegaconf.errors.ConfigAttributeError: Key 'refine_start_iter' is not in struct
    full_key: opt.mcmc_strategy.refine_start_iter
```

before training could begin — so this code path had never run. `mcmc_refine_every` was
also defined in the config but never passed to the strategy. Both fixed.

This one is load-bearing here: MCMC is the *only* way to bound the Gaussian count, which
is what makes this scene trainable on a 16 GB GPU at all.

---

## A finding worth knowing: depth supervision is off by default

`conf/opt/simple_gsplat_30K.yaml` sets `depth_loss: false`. With the documented training
command, the sparse depth this pipeline spends most of its preprocessing effort building is
**not used as a training signal**. It is still used for per-camera near-plane estimation
(`model/vanilla_gsplat.py`, `camera.sparse_depth.min()`), but
`calculate_inverse_depth_loss` is gated off.

To actually supervise on it:

```bash
opt.depth_loss=true opt.depth_lambda=2e-4
```

This is upstream's default, not something the Gen 2 port changed. Worth stating explicitly
because the four Gen 2 SLAM cameras exist in this pipeline almost entirely to produce that
depth — so whether it is enabled decides whether that work affects the result at all.

---

## Results

### Preprocessing

Wall clock **~13 minutes** (00:25 → 00:38), 23 GB output, on the recording described above.

| Camera | frames | rectified | sparse-depth frames | observation match |
|---|---|---|---|---|
| `camera-rgb` | 4,719 → **4,675** | 2016×1512 | 4,675 | n/a (derived from SLAM) |
| `slam-front-left` | 4,717 → 4,674 | 512×512 | 4,674 | 1,556 / 4,674 (**33.3 %**) |
| `slam-front-right` | 4,717 → 4,674 | 512×512 | 4,674 | 1,556 / 4,674 (**33.3 %**) |
| `slam-side-left` | 4,717 → 4,674 | 512×512 | 4,674 | 1,556 / 4,674 (**33.3 %**) |
| `slam-side-right` | 4,717 → 4,674 | 512×512 | 4,674 | 1,556 / 4,674 (**33.3 %**) |

33.3 % is the expected ratio — MPS tracks at 10 Hz, the cameras record at 30 Hz. The
4,719 → 4,675 drop is frames falling outside the MPS trajectory window.

Semi-dense cloud: **4,292,410 points**. 2D tracked observations per camera: 7.7 M – 9.1 M.

Sampled sparse depth (every 200th frame): median **2.81 m** for `slam-front-left`,
**5.11 m** for `camera-rgb`, no negative depths. RGB depth is denser than any single SLAM
camera's because it merges all four.

### Verification

| Check | Result |
|---|---|
| `vignette.png` / `mask.png` / `image_index.png` vs image size | 2016×1512, all match |
| symlinks resolve to the MPS folder | yes |
| readout time carried per frame | 10.10 ms |
| PLY colour format | `rgb`, 3 channels, SH degree 3 |
| PLY → USDZ (real trained PLY) | succeeds |
| exported USDZ | `upAxis=Z`, `metersPerUnit=1.0`, `OmniNuRecFieldAsset` |

### Training

30,000 iterations, MCMC, `cap_max=1.5M`, `data_factor=2` (1008×756), rolling shutter on.
Peak VRAM stayed near 4 GB of 16 GB.

Held-out metrics on the 585-frame validation split (every 8th frame):

| Metric | Value |
|---|---|
| **PSNR** | **25.55** |
| **SSIM** | **0.846** |
| **LPIPS** | **0.389** |

Final model: **1,500,000 Gaussians** — exactly `cap_max`, so the cap was the binding
constraint and a larger budget would have used it.

Reaching this took four attempts; the three failures are documented above and in the
VRAM section, since they are the parts most likely to bite on a different machine.

### Export

| | |
|---|---|
| PLY (raw) | 372 MB, 1,500,000 Gaussians |
| PLY (filtered, ±50 m) | 1,481,184 Gaussians — 1.254 % dropped, median opacity 0.0115 |
| AABB before filtering | −31403 … +114038 m (**~120 km**) |
| AABB after filtering | −49.8 … +49.1 m |
| USDZ | 175 MB, `upAxis=Z`, `metersPerUnit=1.0` |

### Outlier filtering is required, not optional

MCMC drifts a few hundred low-opacity Gaussians arbitrarily far from the scene. They are
invisible — the ones beyond 5 km had a median opacity of **0.011** — but they define the
exported asset's bounding box. Unfiltered, a 10 m driveway exported with a **~120 km**
AABB, which ruins framing and depth precision in a USD stage.

`scripts/filter_splat_outliers.py` drops Gaussians beyond a radius of the **median**
Gaussian position (not the origin — the MPS world origin sits wherever tracking
initialised, not at the scene centre). The export script runs it automatically.

```bash
python scripts/filter_splat_outliers.py in.ply --radius 50 --output out.ply
```

This is a property of MCMC, so it did not arise on the `default`-strategy attempts — but
those could not complete on this GPU at all.
