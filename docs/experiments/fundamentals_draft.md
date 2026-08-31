# From the ground up: how this pipeline actually works

Draft of the fundamentals section for the final report. Everything below is grounded in
the code of this repo as it exists on this machine (branch `gen2-port`, working tree of
2026-08-20, including the uncommitted Gen 2 fixes), with `file:line` citations. Where the
code contradicts the folklore in the docs, the code wins and the discrepancy is called
out. Companion documents: `docs/full_pass.md` (the as-run commands and numbers) and
`docs/experiments/DOC_report.md` (surprises found while writing this).

The pipeline in one sentence: a walk through a space wearing Aria Gen 2 glasses becomes
(1) posed, undistorted photographs plus a metric sparse point cloud (preprocessing), which
become (2) an optimized set of ~1.5 M colored, semi-transparent 3D Gaussians that
re-render those photographs (training), which becomes (3) a USD asset Isaac Sim can render
(export) — and at no point does anything in this chain produce a *surface*.

---

## 1. The inputs: VRS and the four MPS files

### 1.1 The VRS recording

`/home/sun/aria/Outside_20260812_141244.vrs` is the raw multi-sensor recording: for this
profile (profile10), one rolling-shutter RGB camera (`camera-rgb`, 2016×1512 @ 30 Hz,
fisheye) and four global-shutter monochrome SLAM cameras (512×512 @ 30 Hz), plus IMUs and
other streams this pipeline never touches. The repo reads it exclusively through
`projectaria_tools`' `data_provider` (`scripts/extract_aria_vrs.py:431`), pulling per
frame: the pixel data, `capture_timestamp_ns`, exposure duration, and gain
(`scripts/extract_aria_vrs.py:93-127`).

Cameras are enumerated from the device calibration, not hardcoded, because Gen 1 and
Gen 2 name their SLAM cameras differently (`camera-slam-left/right` vs
`slam-front-left/right` + `slam-side-left/right`) and reuse StreamIds inconsistently
between generations (`scripts/aria_utils.py:22-59`, `scripts/extract_aria_vrs.py:435-448`).

A key device fact that shapes the entire pipeline: the RGB camera is **rolling shutter**
(rows are read out sequentially over ~10 ms) while all four SLAM cameras are **global
shutter**. On a head-mounted camera at walking speed, 10 ms of head rotation is a visible
geometric error, so the RGB rolling shutter is modeled explicitly — twice, in fact: once
in preprocessing (per-frame start/middle/end poses, §2.3) and once at render time during
training (§3.5).

### 1.2 What MPS adds, file by file

MPS (Meta's cloud/desktop Machine Perception Services) runs visual-inertial SLAM plus
offline bundle adjustment over the VRS and emits the `slam/` folder. This pipeline
consumes exactly four of its files; everything else (`open_loop_trajectory.csv`,
`summary.json`) is ignored. The world frame of all four files is the same: **gravity
aligned, Z-up, metric metres**, with the origin wherever tracking initialised. That
shared frame is the load-bearing property of the whole project — the splat, the sparse
depth, and every candidate collision mesh live in it without any registration step.

**`closed_loop_trajectory.csv`** — the device pose over time, ~1 kHz (155,800 rows over
~156 s here). Columns (verified against the actual file):
`tracking_timestamp_us`, translation `t*_world_device`, quaternion `q*_world_device`,
linear/angular velocity, gravity vector in world (0, 0, −9.81 — direct confirmation the
frame is gravity-aligned Z-up), and a `quality_score`. "Closed-loop" means loop-closed
and bundle-adjusted — globally consistent, unlike the drift-prone open-loop file. It is
read with `mps.read_closed_loop_trajectory` (`scripts/extract_aria_vrs.py:410`) and is
the **only source of camera poses anywhere in the pipeline**: preprocessing interpolates
it per frame (§2.2), and training re-reads it again at camera-load time to interpolate
per rolling-shutter-row poses (`scene/dataset_readers.py:565-566`,
`scene/cameras.py:910-918`). Nothing in this repo ever solves for a pose.

**`online_calibration.jsonl`** — time-varying calibration, one JSON record per SLAM
keyframe (1,557 records ≈ 10 Hz here). Each record carries, per camera: the FISHEYE624
projection parameters (focal, principal point, distortion — first param here 878.3, the
"native fisheye focal" of full_pass.md), the camera-to-device extrinsic
`T_Device_Camera`, and — critically — a top-level `ReadoutTimesSec` field. On this
recording that field is `[[4, 0.0101]]`: stream index 4 (camera-rgb) reads out in
10.1 ms, and the SLAM cameras are absent because they are global shutter. This file is
needed for three distinct things:

1. *Intrinsics/extrinsics that follow thermal drift*: for each frame, preprocessing
   bisects to the nearest-in-time calibration record and uses that camera's parameters
   (`scripts/extract_aria_vrs.py:153-165`), rather than the fixed factory calibration
   (available via `--use_factory_calib` for comparison).
2. *The rolling-shutter readout time*: factory calibration does not carry readout time at
   all (`get_readout_time_sec()` returns None there), so MPS online calibration is the
   only machine-readable source (`scripts/aria_utils.py:100-140`). Reading it from data
   also makes the pipeline profile-agnostic — Gen 2 readout time varies by recording
   profile (38 ms on profile8, 10.1 ms here on profile10).
3. *The stereo baseline* (used by experiment E3, not by this pipeline directly) comes
   from the same per-camera `T_Device_Camera` entries.

**`semidense_points.csv.gz`** — the global semi-dense point cloud: 4,292,410 rows of
`uid, graph_uid, p*_world, inv_dist_std, dist_std`. These are the 3D positions of the
tracked feature points from SLAM, with per-point uncertainty (standard deviation of the
distance and inverse-distance estimates). It is used twice, with *different* confidence
thresholds:

- Preprocessing keeps points with `inverse_distance_std < 0.005, distance_std < 0.01`
  via `filter_points_from_confidence` (`scripts/extract_aria_vrs.py:417-421`) and builds
  a `uid → point` map (`scripts/extract_aria_vrs.py:429`) used to give sparse-depth
  points their 3D positions.
- Training re-reads the same file (through a symlink, §2.6) with looser thresholds
  (`inverse_distance_std < 0.01, distance_std < 0.02`,
  `scene/dataset_readers.py:689-694`) and uses the surviving points as the
  **initialization of the Gaussians** — one Gaussian seeded per point (§3.2).

**`semidense_observations.csv.gz`** — the link table connecting the point cloud to
images: rows of `uid, frame_tracking_timestamp_us, camera_serial, u, v`, i.e. "point
`uid` was observed by camera `camera_serial` at time `t`, at pixel `(u,v)`" (7.7–9.1 M
rows per camera here). This is what makes *per-frame sparse depth* possible: it tells
you which subset of the 4.3 M points a given camera actually saw in a given frame, so
depth maps contain only genuinely observed geometry instead of points projected through
walls. It is loaded as a pandas DataFrame (`scripts/extract_aria_vrs.py:510-513`),
filtered per camera by serial number (`scripts/extract_aria_vrs.py:721-724`), and joined
to frames by timestamp (§2.5). It is the only optional input: omitting it skips sparse
depth generation entirely (`scripts/extract_aria_vrs.py:715-717, 763-764`).

Two facts about the observations worth knowing because they explain numbers in the logs:
MPS tracks at ~10 Hz while the cameras record at 30 Hz, so only ~33 % of frames have
observations (the "matched observations for 1556/4674 frames (33.3%)" lines); and only
SLAM cameras appear in the table — the RGB camera has no observations of its own, which
is why RGB sparse depth must be built indirectly (§2.5).

---

## 2. Preprocessing: `scripts/extract_aria_vrs.py`

One command (`scripts/bash_local/run_gen2_outside.sh`) turns VRS + MPS into a training
folder. Internally it is four passes: raw extraction, rectification, SLAM sparse depth,
RGB sparse depth (`run_single_sequence`, `scripts/extract_aria_vrs.py:374-774`).

### 2.1 Pass 1 — raw frames with per-frame calibration and poses

For every frame of every selected camera, `to_aria_image_frame`
(`scripts/extract_aria_vrs.py:66-247`) writes the raw (still-fisheye) image to
`<camera>-images/<camera>_<timestamp>.jpg` and records a JSON entry
(`to_frame_json`, `scripts/extract_aria_vrs.py:250-311`) with the FISHEYE624 intrinsics
and distortion params of the nearest online calibration, the camera→device extrinsic,
and **three** world-from-camera poses: at readout start, middle, and end. The result is
`<camera>-transforms.json`, one per camera.

### 2.2 Pose interpolation from the 1 kHz trajectory

A camera frame's timestamp almost never coincides with a trajectory sample, so poses are
interpolated: `interpolate_aria_pose` (`scripts/aria_utils.py:211-234`) bisects the
closed-loop trajectory for the bracketing pair of poses and calls
`interpolate_closeloop_trajectory_pose` (`scripts/aria_utils.py:175-208`), which does a
proper SE3 interpolation (`sophus.interpolate` — slerp on rotation, lerp on translation)
plus linear interpolation of velocities and quality score. At 1 kHz the bracketing poses
are ≤1 ms apart, so this is effectively exact. Frames whose timestamp falls outside the
trajectory return None and are dropped (`scripts/extract_aria_vrs.py:131-142`) — that is
the 4,719 → 4,675 frame loss: MPS starts tracking only after initialization. The camera
pose is then composed as `T_world_camera = T_world_device @ T_device_camera`
(`scripts/extract_aria_vrs.py:173-184`).

### 2.3 Rolling-shutter timestamps: which instant does a frame represent?

For a rolling-shutter camera "the pose of the frame" is ill-defined — every row was
captured at a different time. The code brackets the readout with three timestamps
(`scripts/extract_aria_vrs.py:96-124`): under the default `center` convention,
`capture_timestamp_ns` is taken to be the center-of-exposure of the *middle* row (which
is what Aria documents), and start/end are ± half the readout time (10.1 ms / 2 here),
where readout time comes from MPS `ReadoutTimesSec` via `get_readout_time_ns`
(`scripts/aria_utils.py:100-140`; resolved once per camera at
`scripts/extract_aria_vrs.py:84-88`). The alternative `--timestamp_convention
readout_start` reproduces the original upstream behaviour — which treated the timestamp
as the readout start and therefore biased every pose by half a readout
(`scripts/extract_aria_vrs.py:1224-1234`). All three timestamps, and the poses
interpolated at each, are stored per frame (`scripts/extract_aria_vrs.py:198-210`); the
`timestamp_read_end − timestamp_read_start` difference is how training later recovers
the readout time without ever seeing the MPS file
(`scene/dataset_readers.py:483` — `readout_time_ns=frame["timestamp_read_end"] -
frame["timestamp_read_start"]`). For global-shutter SLAM cameras the readout time is 0
and all three poses coincide.

### 2.4 Fisheye → pinhole rectification

Gaussian rasterizers want simple projection models; Aria lenses are heavily distorted
fisheyes (Gen 2 RGB covers 133°×99°). So every frame is resampled into an ideal
distortion-free **pinhole** image:

- `undistort_image` (`scripts/aria_utils.py:347-387`) reconstructs the source camera as
  a FISHEYE624 `CameraCalibration` from the per-frame params, builds the target as
  `calibration.get_linear_camera_calibration(ow, oh, f)` — "linear" = pinhole — and
  resamples with `calibration.distort_by_calibration`. (`--extract_fisheye` instead
  targets a spherical/equidistant model; nothing downstream in the 3DGS path uses it,
  and 2DGS explicitly cannot, §3.7.)
- **`--rectified_rgb_focal` (1008 here) is the focal length in pixels of the *output*
  pinhole camera.** It does not need to match the native fisheye focal; it *chooses the
  output FOV* via `FOV = 2·atan((width/2)/focal)`. Smaller focal = wider FOV = more of
  the fisheye's view squeezed in (with stretching at the edges); larger focal = narrower
  crop. 1008 at width 2016 gives exactly 90° horizontal.
- **`--rectified_rgb_size` (1512 here) is the output *height*; width is derived from
  the source aspect ratio** (`rectified_w = (rectified_h / frame.h) * frame.w`,
  `scripts/aria_utils.py:430-431`) — 4:3 here, so 2016×1512. The same pair of flags
  exists for the SLAM cameras (`--rectified_monochrome_focal/height`, 180/512 here).
  Negative focal skips that camera class entirely
  (`scripts/extract_aria_vrs.py:452-455`).
- The output intrinsics are exact by construction: `fx = fy = focal`,
  `cx = (w−1)/2`, `cy = (h−1)/2` (`scripts/aria_utils.py:437-441`). This is why the
  training-side transforms carry clean round numbers and why downstream consumers (e.g.
  COLMAP in experiment E4) can use a `PINHOLE` model with no distortion.

Three auxiliary images are rectified through the *same* warp so they stay pixel-aligned
with the images (`scripts/extract_aria_vrs.py:568-632`):

- `vignette.png` — lens-shading falloff, multiplied into the rendered image during
  training's exposure model. For Gen 2 no vignette assets exist and the ISP already
  corrects shading on-device, so a neutral all-ones vignette is used
  (`scripts/aria_utils.py:575-591`).
- `mask.png` — valid-pixel mask. For Gen 2, a full-frame mask; the fisheye boundary is
  handled by `undistort_image` zeroing unmapped pixels (`scripts/aria_utils.py:644-650`).
- `image_index.png` — the clever one: a synthetic image whose pixel value encodes the
  **source (fisheye) row** each rectified pixel came from, built by warping a vertical
  0→1 ramp through the rectification (`scripts/aria_utils.py:525-550`). Rolling shutter
  is a *raw-sensor-row* phenomenon, and rectification bends rows into curves — this map
  is what lets training assign each rectified pixel to the correct readout-time slice
  (§3.5).

The output folder is named `<camera>-rectified-<focal>-h<height>` — the flags are baked
into the path, which is why `scene.scene_name` must name the exact folder.

### 2.5 Sparse depth: which points did this frame actually see?

**SLAM cameras** (`create_visible_depth_map`, `scripts/extract_aria_vrs.py:952-1121`):
for each camera, observations are filtered to that camera's serial
(`scripts/extract_aria_vrs.py:721-724`), then each frame is joined to the observation
table by timestamp — nearest observation timestamp within 16 ms, i.e. half a frame
period (`scripts/extract_aria_vrs.py:981-1016`; an exact-equality join here previously
made sparse depth silently empty whenever half-readout wasn't a whole microsecond). The
matched observation rows give the `uid`s seen in this frame; their 3D positions are
looked up in the semi-dense map (flattened once into parallel arrays for vectorization,
`scripts/extract_aria_vrs.py:925-949`) and projected in one batched `project()` call
into the frame's rectified pinhole camera using `K` and the inverse of the frame's
`transform_matrix` (`scripts/extract_aria_vrs.py:1019-1041`). In-frustum survivors are
written per frame as `sparse_depth/<camera>_<timestamp>.json` with parallel arrays
`u, v, z, inverseDistanceStd, distanceStd, uid` (`scripts/extract_aria_vrs.py:1054-1061`)
— pixel coordinates, metric depth along the camera Z axis, and the MPS per-point
uncertainties. The per-camera match rate is printed at the end
(`scripts/extract_aria_vrs.py:1094-1108`); ~33 % is healthy, 0 % means depth supervision
would silently be a no-op.

**The RGB camera** has no observations of its own, so
`fetch_visible_depth_map_for_RGB` (`scripts/extract_aria_vrs.py:777-922`) builds its
depth secondhand: for each RGB frame, take the temporally nearest frame of *each* SLAM
camera (`scripts/extract_aria_vrs.py:843-848`), unproject that frame's sparse depth
through its pinhole intrinsics back to 3D world points
(`scripts/extract_aria_vrs.py:866-876`), and reproject them into the RGB frustum,
keeping in-frustum hits (`scripts/extract_aria_vrs.py:878-887`). Gen 2's side cameras
mostly look away from the RGB frustum; their points simply get culled, so extra cameras
can only add coverage (`scripts/extract_aria_vrs.py:783-796`). The result: RGB sparse
depth is denser than any single SLAM camera's because it merges all four.

### 2.6 What `transforms_with_sparse_depth.json` contains

This is the one file training reads (`conf/config.yaml:44` sets it as `data_format`).
Verified structure on disk: top-level `camera_model` ("linear"), `camera_label`,
`transform_cpf`, and `frames` — one entry per frame with:

| key | meaning |
|---|---|
| `fx, fy, cx, cy, w, h` | rectified pinhole intrinsics (1008/1008/1007.5/755.5/2016/1512 here) |
| `image_width_raw/height_raw` | pre-rectification sensor size |
| `image_path` | `images/<camera>_<timestamp>.png` |
| `camera2device` | camera→device extrinsic (4×4) |
| `transform_matrix` | camera→world at the readout **center** (4×4) |
| `timestamp`, `timestamp_read_start`, `timestamp_read_end` | ns; end−start = readout time (10.1 ms here) |
| `exposure_duration_s`, `gain` | per-frame exposure state, used by the training-time exposure model |
| `sparse_depth` | relative path to this frame's sparse depth JSON |

(The per-frame read-start/read-end *poses* computed in pass 1 are not carried into this
file — `process_frame` drops them, `scripts/aria_utils.py:465-466` — because training
re-derives poses at arbitrary sub-readout timestamps from the trajectory itself, §3.5.)

Two symlinks are also planted in the rectified folder
(`scripts/extract_aria_vrs.py:546-563`): `semidense_points.csv.gz` and
`closed_loop_trajectory.csv`, pointing into the MPS directory. Training reads both
through the symlinks (`scene/dataset_readers.py:565, 687`), so the training folder is
self-contained *only as long as the MPS folder stays put*.

---

## 3. Training: `train_lightning.py`, `model/vanilla_gsplat.py`, `model/GS2D_gsplat.py`

`train_lightning.py` is a thin PyTorch-Lightning driver: it builds the scene
(`initialize_scene_info`, `train_lightning.py:25`), picks the model class by
`train_model` — `"3dgs"` → `VanillaGSplat`, `"2dgs"` → `Gaussians2D`
(`train_lightning.py:27-40`) — makes train/valid/test loaders
(`train_lightning.py:58-67`), runs `trainer.fit` then `trainer.test`
(`train_lightning.py:81-91`). The rendering backend is NVIDIA-maintained **gsplat
1.5.3** (`gsplat.rendering.rasterization`, `model/vanilla_gsplat.py:26`); this repo's
contribution is everything around it (Aria cameras, exposure model, rolling shutter,
losses, strategies).

### 3.1 What a 3D Gaussian is, in this codebase's exact terms

The entire scene model is a flat `ParameterDict` of per-Gaussian tensors
(`SceneSplats.create_splats_with_optimizers`, `model/vanilla_gsplat.py:55-169`):

| parameter | shape | storage → activation | meaning |
|---|---|---|---|
| `means` | (N,3) | raw | 3D center, in MPS world metres |
| `scales` | (N,3) | log → `exp` (`model/vanilla_gsplat.py:184-185`) | per-axis standard deviations of the ellipsoid |
| `quats` | (N,4) | unnormalized → normalized inside the rasterizer (`model/vanilla_gsplat.py:944-946`) | ellipsoid orientation |
| `opacities` | (N,) | logit → `sigmoid` (`model/vanilla_gsplat.py:195-197`) | peak alpha of the Gaussian |
| `rgb_sh0` | (N,1,3) | SH DC term | base color |
| `rgb_shN` | (N,15,3) | SH degrees 1–3 | view-dependent color variation |

Each Gaussian is an anisotropic 3D Gaussian density: an ellipsoid-shaped, soft-edged
blob with orientation (quat), per-axis size (scales), transparency (opacity), and a
view-dependent color given by a degree-3 spherical-harmonics expansion — 1 DC + 15
higher-order coefficients per channel, hence `sh_degree: 3` ⇒ (3+1)² = 16 coefficients
(`model/vanilla_gsplat.py:134`). Rendering projects every ellipsoid into the image as a
2D Gaussian ("splat"), sorts by depth, and alpha-composites front to back — that is all
`rasterization()` does (`model/vanilla_gsplat.py:965-983`). `render_mode="RGB+ED"`
additionally returns per-pixel **expected depth** — the alpha-weighted mean depth of the
Gaussians along each ray (`model/vanilla_gsplat.py:982, 987`) — which matters later:
expected depth over fuzzy blobs is smooth and can average across depth discontinuities.

Color is stored as SH around a 0.5 gray offset (`color_to_sh`,
`model/vanilla_gsplat.py:460-462`). Depending on which cameras train the model, the
color format is `rgb`, `m` (monochrome), or `rgbm` (both,
`model/vanilla_gsplat.py:642-671`) — this is why full_pass.md warns to train RGB-only:
`m`/`rgbm` PLYs have 1 or 4 `f_dc_*` channels and external tools only read 3.

### 3.2 Initialization: the semi-dense cloud becomes the first Gaussians

`readAriaSceneInfo` loads the semi-dense points through the symlink, filters them
(§1.2), and hands them over as `SceneInfo.point_cloud`
(`scene/dataset_readers.py:687-703`). One Gaussian is created per point: mean = point
position; scale = the average distance to its 3 nearest neighbors (a kNN over the whole
cloud, `model/vanilla_gsplat.py:74-77, 465-469`) so that neighboring Gaussians initially
overlap; random orientation; opacity `init_opa`=0.1; SH DC = 0.5 gray (the MPS cloud
has no color — `colors=None` at `scene/dataset_readers.py:703`, so `color_init` falls
back to 0.5 at `model/vanilla_gsplat.py:126-127`). This is `init_type: "sfm"`
(`conf/opt/simple_gsplat_30K.yaml:43`): geometry starts where SLAM saw geometry, which
is the reason training converges at all in 30 k iterations.

`scene_scale` — the max camera distance from the mean camera center
(`scene/dataset_readers.py:71-81, 683`), ×1.1 (`model/vanilla_gsplat.py:492`) — scales
only the position learning rate (`1.6e-4 * scene_scale`, `model/vanilla_gsplat.py:84-86`)
and normalizes rendered depth in the depth loss (`model/vanilla_gsplat.py:1118`); it
does not rescale the scene.

### 3.3 Cameras, the split, and the exposure model

Every frame entry becomes an `AriaCamera` (`scene/dataset_readers.py:459-486`) carrying
its rectified pinhole intrinsics, its camera→device extrinsic, its timestamp, its
per-frame exposure/gain, its sparse depth, and a handle to the full 1 kHz trajectory.
Camera labels are normalized so downstream code can dispatch on `"rgb"`/`"slam"`
substrings (`scene/dataset_readers.py:414-434`; the prefix test that recognizes both
Gen 1 and Gen 2 SLAM names is `scene/cameras.py:38-43`).

`scene.train_split: "7-1"` (`conf/config.yaml:46`) holds out every 8th camera as the
validation set (`scene/dataset_readers.py:663-675`: `valid_idx = arange(0, N, 8)`,
train = the complement; `"4-1"` holds out every 5th). With no `test_views/` folder on
disk, the validation cameras are reused as the test set
(`scene/dataset_readers.py:678-681`), which is why the final `trainer.test()` numbers
(PSNR 25.55 / SSIM 0.846 / LPIPS 0.389 on 585 held-out frames, verified from
`output/.../test_logs.json`) are genuinely held-out metrics.

The model does not directly predict the photograph; it predicts scene *irradiance*, and
each camera exposes it: `expose_image` (`scene/cameras.py:496-543`) multiplies by
`exposure_duration × gain` (normalized by the recording's median,
`scene/dataset_readers.py:509-514`) and by the per-pixel vignette, operating in gamma
space. This makes Gaussian color independent of the camera's auto-exposure flicker —
without it, a scene recorded under varying exposure would bake brightness changes into
geometry.

### 3.4 The loss and the optimization loop

Manual-optimization Lightning: one Adam optimizer per parameter group with per-group
learning rates (`model/vanilla_gsplat.py:160-169`), plus an exponential decay schedule
on `means` only (down to 1 % over the run, `model/vanilla_gsplat.py:702-708`).

Per step (`training_step`, `model/vanilla_gsplat.py:1027-1193`): render one training
camera, expose it, then

```
loss = 0.8 * L1(render, photo) + 0.2 * DSSIM(render, photo)        # :1105-1108
     [+ depth_lambda * Huber(1/render_depth vs 1/sparse_depth)     # :1116-1128, if opt.depth_loss
     [+ opacity/scale regularizers if enabled]                     # :1131-1139
```

- The pixel term (`pixel_lambda` 0.8, `conf/opt/simple_gsplat_30K.yaml:124`) matches
  colors; DSSIM (structural dissimilarity) matches local contrast/structure. (The 0.2 is
  hard-coded at `model/vanilla_gsplat.py:1107`; the `ssim_lambda`/`dssim_lambda` config
  keys are decorative — they happen to equal 0.2 anyway.)
- **The depth loss is off by default** (`depth_loss: false`,
  `conf/opt/simple_gsplat_30K.yaml:135`) — the sparse depth that preprocessing works so
  hard to build is, by default, *not* a training signal. When enabled,
  `calculate_inverse_depth_loss` (`model/loss.py:22-82`) samples the rendered depth map
  at each sparse point's pixel via `grid_sample`, compares *inverse* depths (so near
  points count more, and the comparison is robust to far-field error), weights by the
  MPS per-point `inverseDistanceStd`, and takes a Huber loss. Even when disabled, sparse
  depth is still used: as the per-camera minimum depth for the 3D smooth filter
  (`model/vanilla_gsplat.py:808-810`) and for the rolling-shutter motion estimate
  (§3.5).
- The 3D smooth filter (`use_3d_smooth_filter: true`, `conf/config.yaml:70`) is a
  Mip-Splatting-inspired anti-aliasing measure: it widens every Gaussian by the world
  size of one pixel at the camera's minimum observed depth and compensates opacity by
  the determinant ratio, using a single per-view filter rather than Mip-Splatting's
  per-point one (`_apply_3d_depth_based_filter`, `model/vanilla_gsplat.py:899-929`,
  applied at `model/vanilla_gsplat.py:955-959`). It suppresses the shimmering/aliasing
  of sub-pixel Gaussians when rendering from novel distances.

### 3.5 Rolling shutter at render time (`opt.handle_rolling_shutter`)

Preprocessing gave each frame a single center-of-readout pose, but the photograph was
really taken by a camera that *moved during* the 10.1 ms readout: each image row saw the
world from a slightly different pose. From iteration 10,000
(`handle_rolling_shutter_start_iter`, `conf/opt/simple_gsplat_30K.yaml:33`; gating in
`_render_motion_array`, `model/vanilla_gsplat.py:773-788`), training models this
exactly:

1. **How many poses are needed?** At camera construction,
   `_estimate_pixel_motion_offset` (`scene/cameras.py:949-1004`) reprojects the frame's
   own sparse depth through the relative pose across a 2 ms window and measures the
   median pixel displacement — i.e. "how many pixels does the world slide per 2 ms for
   *this* frame". The number of readout time-slices is chosen so each slice moves ≲1
   pixel, capped at 8 (`_max_rolling_shutter_sample`, `scene/cameras.py:746`;
   computation at `scene/cameras.py:822-841`). A slow frame gets 1 sample (no rolling
   shutter cost); a fast head-turn gets 8.
2. **Sample poses.** The readout interval is split into that many brackets and the pose
   is interpolated from the 1 kHz trajectory at each bracket's center
   (`scene/cameras.py:843-855`, `sample_viewmatrices` at `scene/cameras.py:920-947`).
   This is where training needs the raw trajectory again — the reason for the
   closed-loop symlink.
3. **Render and reassemble.** `render()` stacks all sampled poses into one batched
   rasterization call (`model/vanilla_gsplat.py:814-842`) — N_rs full images rendered
   per training step — and then *composites a single image by picking, for every pixel,
   the time-slice its source sensor row was actually read in*, using the rectified
   row-index map from §2.4 quantized to the slice count
   (`rolling_shutter_index_image`, `scene/cameras.py:1029-1035`; the `torch.gather` at
   `model/vanilla_gsplat.py:870-891`). The same machinery has an exposure-blur axis
   (average over samples within the exposure window) that is compiled in but disabled by
   default (`_max_exposure_sample = 1`, `scene/cameras.py:747`).

This is why VRAM spikes at iteration 10 k (up to 8 renders per step instead of 1), and
why the OOM in the original full-resolution run occurred there. SLAM cameras are global
shutter, so for them the array is always 1×1 and this machinery is inert.

### 3.6 Densification: default strategy vs MCMC, and why `cap_max` exists

The initial ~2–4 M Gaussians cannot represent fine detail everywhere, so gsplat
*changes N during training* via a pluggable strategy (`_create_strategy`,
`model/vanilla_gsplat.py:581-625`), stepped before/after every backward pass
(`model/vanilla_gsplat.py:1152-1185`):

- **`default`** (`DefaultStrategy`, the original 3DGS recipe,
  `model/vanilla_gsplat.py:583-603` with knobs from
  `conf/opt/simple_gsplat_30K.yaml:66-83`): every `refine_every`=100 steps between
  iterations 500 and 15,000, Gaussians whose accumulated image-plane position gradient
  exceeds `grow_grad2d` are **duplicated** (if small: scale < `grow_scale3d`·scene) or
  **split** (if large) — high gradient means "the photometric loss wants more capacity
  here". Gaussians with opacity < `prune_opa`=0.005 are **pruned** (they contribute
  nothing), as are those larger than `prune_scale3d`. Every 3,000 steps all opacities
  are **reset** low so pruning can re-evaluate honestly. `absgrad: true`
  (`conf/opt/simple_gsplat_30K.yaml:90`) uses absolute rather than net gradients
  (AbsGS), which detects need-to-densify regions whose gradients cancel. The failure
  mode: **N is unbounded**. On this foliage-heavy outdoor scene it ran to ~15 M
  Gaussians and OOM'd a 16 GB card (full_pass.md §3) — every Gaussian costs ~59 floats
  of parameters plus gradients plus two Adam moments.
- **`MCMC`** (`MCMCStrategy`, `model/vanilla_gsplat.py:604-621`, knobs from
  `conf/opt/simple_gsplat_30K.yaml:102-108`): reframes densification as sampling — the
  Gaussian set is treated as an MCMC sample from a distribution concentrated where the
  scene needs capacity. Dead (low-opacity) Gaussians are *relocated* onto high-opacity
  ones rather than deleted-and-grown, new ones are introduced by sampling the opacity
  distribution, and every step adds decaying positional noise (`noise_lr`, scaled by the
  current means learning rate, `model/vanilla_gsplat.py:1177-1185`). Crucially it takes
  **`cap_max` — a hard ceiling on N** (1.5 M in the real run). That cap is the entire
  reason MCMC is used here: it converts "hope the scene fits" into a memory budget set
  in advance. The final model landed exactly at `cap_max`, i.e. the budget, not the
  scene, was the binding constraint. Two costs: the injected noise is what sprays a few
  hundred low-opacity Gaussians kilometres away (§4.1), and a too-small cap costs
  sharpness.

Config-plumbing note: the MCMC block's keys are `mcmc_refine_start_iter`,
`mcmc_refine_stop_iter`, `mcmc_refine_every`, `mcmc_min_opacity`
(`conf/opt/simple_gsplat_30K.yaml:105-108`) and are mapped to the strategy's argument
names at `model/vanilla_gsplat.py:613-620` — an uncommitted fix; before it, selecting
MCMC crashed at startup on missing keys, so the MCMC path had never actually run
(full_pass.md, bug 5).

### 3.7 The key config flags, honestly

From `conf/config.yaml` / `conf/opt/simple_gsplat_30K.yaml`, the ones the briefs
mention:

| flag | what the code actually does with it |
|---|---|
| `model.sh_degree: 3` (`conf/config.yaml:64`) | SH order for view-dependent color; 16 coeffs/channel. Raised one degree per 1,000 iters (`sh_degree_interval`, `model/vanilla_gsplat.py:1017-1021`) so color complexity grows after geometry settles. |
| `scene.train_split: "7-1"` (`conf/config.yaml:46`) | every-8th-frame holdout; see §3.3. |
| `gs_default_strategy.prune_opa: 0.005` (`conf/opt/simple_gsplat_30K.yaml:68`) | opacity floor below which the default strategy deletes a Gaussian; MCMC's equivalent is `mcmc_min_opacity` (relocation threshold). |
| `model.use_3d_smooth_filter: true` (`conf/config.yaml:70`) | the depth-based Mip-Splatting-style 3D filter of §3.4. |
| `opt.handle_rolling_shutter: true` + `handle_rolling_shutter_start_iter: 10000` | §3.5. |
| `scene.data_factor` (`conf/config.yaml:61`) | **inert.** Declared in the config (comment says "downscale the image by a factor of 1,2,4,8") and set to 2 in the real run — but no code reads it: the only consumers of `factor`/`stride` config keys are steps_scaler math (`grep` over `scene/`, `model/`, `train_lightning.py`). Verified empirically: the completed run's `cameras.json` and rendered test images are full 2016×1512 with fx = 1008. |
| `scene.pcd_stride` (`conf/config.yaml:41`) | **also inert.** A `stride` parameter exists in `utils/point_utils.py:17-33` (`fetchPly`) but the Aria reader builds its point cloud via `mps.read_global_point_cloud` and never applies it (`scene/dataset_readers.py:687-703`). |

The last two rows contradict `docs/full_pass.md` (which credits `data_factor=2` +
`pcd_stride=2` with the VRAM fix). What the evidence supports: the run that succeeded
trained at **full resolution** with `cap_max=1.5M` and MCMC, at ~4 GB peak; the two
scene flags were accepted by Hydra (they exist in the config schema) but changed
nothing. This matters for experiment E2: retraining "with the same settings" means
MCMC + cap_max is the setting that counts, and re-running at a genuinely reduced
resolution would require implementing `data_factor`, not just passing it.

### 3.8 3DGS vs 2DGS (`model/GS2D_gsplat.py`)

`Gaussians2D` subclasses `VanillaGSplat` (`model/GS2D_gsplat.py:33`) and swaps the
rasterizer for `rasterization_2dgs` (`model/GS2D_gsplat.py:10, 278-293`). The
parameterization is identical (it even keeps 3 scale components; the rasterizer uses
two of them) — the difference is the primitive:

- **3DGS: an ellipsoid** (a volumetric density blob). Its "depth" is the expected depth
  of translucent volumes blended along the ray — soft, and free to be wrong between
  surfaces, because the photometric loss only constrains the composited *color*.
- **2DGS: a flat, oriented disk** (a surfel — a 2D Gaussian embedded in 3D). A disk has
  a well-defined tangent plane, so the rasterizer returns per-pixel **analytic surface
  normals** (`render_normals`) alongside color, plus a depth computed from exact
  ray-disk intersection, a `normals_from_depth` estimate, and a per-ray **depth
  distortion** measure (`model/GS2D_gsplat.py:270-278, 306-314`).

Its training step adds two geometry regularizers that 3DGS cannot have
(`model/GS2D_gsplat.py:423-462`, weights at `conf/opt/simple_gsplat_30K.yaml:143-153`):
a **normal-consistency loss** (rendered disk normals must agree with normals implied by
the depth map — forces disks to lie *flat on* the surface they render, from iter 7,000)
and a **depth-distortion loss** (penalizes each ray's alpha mass being spread across
depth — pulls each pixel's contributing splats into one thin shell, from iter 3,000).
Those two losses are the actual mechanism behind "2DGS depth/normals are crisper": the
representation admits a well-defined surface, and the losses actively concentrate the
model onto it, where 3DGS's expected depth may blend a leaf and the wall behind it into
a depth that corresponds to no physical surface. This is why experiment E2 expects
cleaner TSDF fusion from a 2DGS retrain.

The constraint: `assert camera.camera_projection_model_gsplat == "pinhole"` — "2dgs will
only support pinhole camera model" (`model/GS2D_gsplat.py:62-64`; the mapping
linear→pinhole, spherical→fisheye is `scene/cameras.py:91-94`). gsplat's 2DGS
rasterizer has no fisheye ray-disk intersection path. Since this pipeline rectifies
everything to pinhole anyway (`camera_model: "linear"` in every transforms.json), the
assert is satisfied here — but it forecloses training 2DGS on `--extract_fisheye`
output.

### 3.9 What lands on disk

At test end (`on_test_epoch_end`, `model/vanilla_gsplat.py:1348-1357`), the model is
written to `point_cloud/iteration_30000/point_cloud.ply` via `save_ply`
(`model/vanilla_gsplat.py:237-313`): per vertex `x y z`, dummy normals, `f_dc_0..2`
(SH DC), `f_rest_0..44` (15 SH coeffs × 3 channels), `opacity`, `scale_0..2`,
`rot_0..3` — **all stored pre-activation** (log-scales, logit-opacities,
`model/vanilla_gsplat.py:272-275`), which every 3DGS-ecosystem tool expects — plus a
nonstandard second PLY element `metadata` recording color format and SH degree
(`model/vanilla_gsplat.py:292-302`) that this repo's own loader reads back and that
tolerant readers (like the 3dgrut importer) ignore.

---

## 4. Post: filtering, USDZ export, Isaac Sim

### 4.1 `scripts/filter_splat_outliers.py` — why far outliers exist and why they matter

MCMC's exploration noise (§3.6) leaves a small population of near-invisible Gaussians
scattered arbitrarily far away — on this run, 18,816 of 1.5 M (1.25 %) beyond 50 m,
median opacity ~0.011, some past 100 km. Visually they are nothing. But a USD asset's
**AABB is computed over all primitives regardless of opacity**, so the unfiltered export
declared bounds of −31,403…+114,038 m — a ~120 km box around a 10 m driveway. That
wrecks any consumer that uses bounds: viewport framing (`f` flies to the box, not the
scene), camera near/far and hence **depth precision**, culling, and any
collision-placement logic keyed to asset extents.

The filter (`filter_ply`, `scripts/filter_splat_outliers.py:32-64`) computes the
**median** Gaussian position — not the origin, because the MPS world origin sits
wherever tracking initialised, generally off-center — and drops everything beyond
`--radius` (default 50 m) of it, preserving all vertex properties and the `metadata`
element. The export wrapper runs it automatically before conversion
(`scripts/bash_local/export_gen2_outside_usdz.sh:36-46`). Post-filter AABB:
±50 m. This is MCMC-specific debris; the `default` strategy would not produce it (but
could not finish on this GPU).

### 4.2 USDZ export (3dgrut, `ply_to_usd.py`)

`scripts/bash_local/export_gen2_outside_usdz.sh` runs NVIDIA's converter from the
**3dgrut** repo in its own conda env:
`threedgrut/export/scripts/ply_to_usd.py <filtered.ply> --output_file <out.usdz>`. What
it does (read from `/home/sun/3dgrut/threedgrut/export/scripts/ply_to_usd.py:80-96`):
loads the PLY into 3dgrut's `MixtureOfGaussians` via `init_from_ply`, then calls
`USDZExporter.export(model, path, dataset=None, conf=conf)`. Because `dataset=None`, the
exporter's optional recentering/upright "normalizing transform" stays identity
(`threedgrut/export/usdz_exporter.py:70-79`) — the Gaussian *payload* keeps its raw MPS
world coordinates, which is what the shared-frame invariant requires. Practical
constraints: it must run from the 3dgrut repo root (its hydra config path is relative to
the script, `ply_to_usd.py:34-48`), the first run compiles CUDA extensions, and the
input must be a standard 3-channel 3DGS PLY (§3.9). The output is a `.usdz` containing a
`Volume` prim with `OmniNuRecFieldAsset` children — NuRec is the neural-rendering
representation the RTX renderer in Isaac Sim ≥5.1 consumes natively. The exported layer
declares `upAxis = Z`, `metersPerUnit = 1.0` (verified via the loader's report), which
matches the MPS frame convention.

**One correction the export script now applies** (root-caused by experiment E1 during
this program): even with the normalizing transform disabled, 3dgrut's
`serialize_nurec_usd` *unconditionally* authors a frame-conversion rotation
`(x,y,z) → (−x,−z,−y)` as `xformOp:transform` on the Volume prim — the conversion from
3dgrut's normalized Y-down frame to the Z-up stage. For a PLY that is already Z-up MPS
metres, that rotation lays the scene on its side in Isaac Sim. The export script
therefore strips it after conversion with `scripts/fix_nurec_usdz_frame.py`
(`scripts/bash_local/export_gen2_outside_usdz.sh:52-61`), which rewrites the volume's
transform to identity inside the USDZ — and only if it exactly matches the known 3dgrut
conversion matrix, aborting on anything unexpected
(`scripts/fix_nurec_usdz_frame.py:48-54, 102-113`). Full root-causing is in
`docs/experiments/E1_report.md`. So the corrected invariant is: the
*payload* was always in the MPS frame; the *prim transform* needed one deterministic
fix at the export layer.

### 4.3 Loading into Isaac Sim (`scripts/isaacsim_load_splat.py`)

The script is a headless smoke test run under Isaac Sim's own interpreter
(`~/isaac-sim/python.sh`). It boots a `SimulationApp` with the RTX renderer
(`scripts/isaacsim_load_splat.py:34-36`), creates a fresh stage, defines
`/World/AriaSplat` as an Xform and **references** the USDZ under it
(`scripts/isaacsim_load_splat.py:41-46`) — referencing rather than opening, so the splat
can coexist with robots and props (experiment E6 adds a collision mesh as a sibling
prim). It then steps the app 60 frames to let the renderer resolve the NuRec payload
(`scripts/isaacsim_load_splat.py:49-50`), walks the prim tree counting NuRec-typed
prims, verifies the Volume prim's composed local-to-world transform is identity — i.e.
that the baked 3dgrut rotation of §4.2 has been stripped, so a regressed asset is caught
here rather than by eyeballing the GUI (`scripts/isaacsim_load_splat.py:64-83`) — and
computes the world AABB with a `UsdGeom.BBoxCache` over the default+render purposes
(`scripts/isaacsim_load_splat.py:85-91`). Success = at least one NuRec prim resolved.

The `--report` flag exists because of a Kit platform gotcha the docstring spells out
(`scripts/isaacsim_load_splat.py:26-31`): Kit captures Python stdout into its own logger
and force-exits on shutdown, so on a headless run neither printed output nor the process
exit code is trustworthy — the JSON file is the only reliable evidence. Verified result
for this asset: `ok: true`, `up_axis: "Z"`, `meters_per_unit: 1.0`, 2 NuRec prims, AABB
≈ ±49 m, agreeing with the filtered PLY.

Splats render as a *volume*: the stage gains no meshes, no collision geometry, nothing a
physics scene can touch. Which brings us to the point of the whole experiment program:

---

## 5. Mental model: a splat is not a surface

**What the trained artifact is.** `point_cloud.ply` is a list of 1.5 M soft ellipsoids
optimized for exactly one objective: *when alpha-composited from the training
viewpoints, reproduce the training photographs* (§3.4's loss is photometric only —
recall depth supervision is off by default). It is a **renderable primitive set** — a
view-synthesis machine — not a geometric model. Nothing in the objective demands that
Gaussians lie *on* surfaces:

- A wall can be represented as several semi-transparent layers straddling the true
  surface whose blend looks right, with no individual Gaussian at the wall's depth.
- Fog-like low-opacity Gaussians can hang in free space to fake soft shadows or sky
  gradients; MCMC's noise leaves genuinely meaningless ones kilometres away (§4.1).
- The rendered "depth" is *expected depth* along each ray (`RGB+ED`,
  `model/vanilla_gsplat.py:982-987`) — an opacity-weighted average that can land between
  two real surfaces where nothing physically exists.

**Why mesh extraction is always a separate step.** Since no surface exists anywhere in
the representation, one must be *decided*, by imposing extra assumptions the training
never enforced. Every route in the experiment program is one such decision procedure:
E2 renders depth from training views and fuses into a TSDF (assumption: expected depth ≈
first opaque surface — helped substantially by 2DGS, whose disk primitive plus
normal-consistency and distortion losses build the surface bias into training itself,
§3.8); E3 and E4 bypass the splat and estimate geometry directly from images; E5 uses
the trajectory as a direct prior on the floor. None of these is "reading the mesh out of
the splat", because there is no mesh in the splat. This is also why the USDZ has no
collision geometry to offer Isaac Sim (§4.3) and why E6 pairs the visual splat with a
separately-derived invisible collision mesh in the same MPS frame.

**Why unobserved floor cannot be recovered by any splat variant.** The optimization's
only gradient source is reprojection error against captured pixels
(`model/vanilla_gsplat.py:1101-1108`). A region that appears in **zero** training images
contributes zero terms to the loss; every Gaussian there (there are none — initialization
comes from SLAM points, which also require observation, §3.2) has exactly zero gradient.
The model is not interpolating a surface prior; it is unconstrained there, and renders
background/nothing. This is structural for *egocentric* capture: a head-mounted camera
pitched at the horizon rarely images the floor within ~1–2 m of the wearer's feet, and
the patch directly underfoot is occluded by the wearer's own body at every instant — so
the very strip a wheeled robot must drive on is systematically the least-observed
geometry in the scene. No splat variant — 3DGS, 2DGS, Mip-, MCMC- or otherwise — changes
this, because all of them share the same data term; a fancier primitive cannot conjure
gradients from pixels that were never captured. Filling that hole requires information
from outside the images. That is precisely the asymmetric value of E5: the closed-loop
trajectory is a 1 kHz record that a human *physically stood on* every point of the walked
path — a dense traversability certificate for exactly the region the cameras never saw.

**Frame discipline, restated.** Everything — trajectory, semi-dense points, sparse
depth, Gaussians, USDZ, candidate meshes — shares the MPS closed-loop world frame:
gravity-aligned, Z-up, metres, origin at tracking start. The camera-side convention (X
right, Y down, Z forward) matches COLMAP's, so the repo performs no coordinate
conversion anywhere (`scripts/extract_aria_vrs.py:171-172`,
`scripts/aria_utils.py:445-447`); the one place a conversion crept in — the rotation
3dgrut bakes into the exported Volume prim — is now stripped deterministically at export
(§4.2). Any pipeline that re-solves poses or re-centers geometry breaks the invariant
that lets the splat and the collision mesh coexist in Isaac Sim without registration.
