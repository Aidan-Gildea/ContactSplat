# From one walk to a drivable world: the Aria Gen 2 → Isaac Sim environment pipeline

**Scene:** `Outside_20260812_141244` — one 2:37, 100.2 m walk through an outdoor space,
recorded on Aria Gen 2 glasses (profile10: one 2016×1512 rolling-shutter RGB camera,
four 512×512 global-shutter SLAM cameras), with Meta MPS run on the recording.
**Deliverable:** two aligned assets loaded together in Isaac Sim — a photoreal Gaussian
splat (USDZ, visual only) and an invisible collision mesh (USD, physics only) — plus the
evaluation and acceptance machinery that picked the collision mesh.

This report names every method in plain language. The underlying run reports (where the
experiment codes live) are:

| Plain name | Report file (`docs/experiments/`) |
|---|---|
| The eval harness | `E0_report.md` |
| The Isaac frame fix | `E1_report.md` |
| The visual splat | `../full_pass.md` (end-to-end run) + `E1_report.md` (frame fix) |
| Splat-depth meshing, 3DGS variant | `E2_3dgs_report.md` |
| Splat-depth meshing, 2DGS variant | `E2_2dgs_report.md` |
| Stereo depth | `E3_report.md` |
| Trajectory floor | `E5_report.md` |
| Photogrammetry (designed, reserved) | brief in `../mesh_experiments.md` (E4); no run report yet |
| The fusion | `E6_candidates_report.md` |
| The drive test | `E6_drive_report.md` |
| Fundamentals / code audit | `fundamentals_draft.md`, `DOC_report.md` |

Numeric ground truth throughout is the on-disk scorecard JSONs under
`output/Outside_20260812_141244/` (in particular
`collision_candidates/comparison_e6a_h16683.json` and the per-mesh
`*.scorecard*.json` files); where a report's prose rounds differently, the JSON wins.

---

## 1. Executive summary

**The goal.** Turn a single head-mounted recording into a simulation environment a
wheeled robot can actually drive: photoreal appearance for rendering, and collision
geometry that is complete (no hole the robot falls through), metrically placed, and
clean enough for PhysX.

**The final recipe.**

- *Visuals:* a 3D Gaussian splat (1.5 M Gaussians, trained 30 k iterations with MCMC
  densification capped at 1.5 M — the cap is what makes a 16 GB GPU sufficient),
  outlier-filtered to ±50 m, exported to USDZ with NVIDIA's 3dgrut converter, and then
  passed through the frame fix that strips the one incorrect rotation the converter
  bakes in. Held-out quality: PSNR 25.55 / SSIM 0.846 / LPIPS 0.389.
- *Collision:* the fused-best mesh
  (`collision_candidates/fused_best_stereoNear_2dgsFar_trajfloor.ply`, 4.12 M
  triangles): stereo-depth TSDF geometry in the near field (best floor accuracy, floor
  at the metrically correct height), 2DGS splat-depth geometry in the far field (walls
  and structure the stereo sensor is blind to), and the trajectory-derived floor prior
  fused underneath (fills every unobserved floor cell at calibrated eye height
  h = 1.6683 m). Scorecard: floor coverage 0.970, largest hole 0.40 m², semi-dense
  median error 2.9 cm, 26.7 m² of floater junk. Mounted invisible as a **static
  triangle-mesh collider** (`physics:approximation = "none"`) — benchmarking showed
  SDF buys nothing for a static mesh and convex decomposition produces wrong geometry.

**The drive-test verdict.** A Nova Carter driven along the demonstrator's own path
(363 waypoints, 90.64 m) records **zero fall-throughs on every candidate** — the
failure class the program set out to eliminate is gone. But only the fused candidates
finish the course: the raw stereo mesh ends in an unrecoverable rollover at 91 %, and
the floor-only prior — despite the best per-cell scorecard — rolls the robot at 20 %.
The fused-best mesh completes 100 % at both tested speeds with the fewest rescues (11)
and zero tip-overs, and is the recommended collision asset.

**The one-sentence alignment principle.** Every asset — splat, depth maps, floor prior,
every candidate mesh — is built directly in the MPS closed-loop world frame
(gravity-aligned, Z-up, metric metres), so the visual and collision assets coexist in
Isaac Sim **without a single registration step**; the only frame error ever found was
introduced by the exporter, and it is now deterministically stripped.

---

## 2. Fundamentals: how the pipeline actually works

This section is the ground-up mental model, condensed from
`docs/experiments/fundamentals_draft.md`; every mechanism cites the code
(`file:line`, branch `gen2-port` working tree). The pipeline in one sentence: a walk
becomes (1) posed, undistorted photographs plus a metric sparse point cloud
(preprocessing), which become (2) ~1.5 M colored, semi-transparent 3D Gaussians
optimized to re-render those photographs (training), which become (3) a USD asset
Isaac Sim renders (export) — and at no point does this chain produce a *surface*.
That last fact is why the whole experiment program of Section 3 exists.

### 2.1 The inputs: one VRS, four MPS files

The VRS is the raw multi-sensor recording, read exclusively through
`projectaria_tools`' data provider (`scripts/extract_aria_vrs.py:431`): per frame the
pixels, `capture_timestamp_ns`, exposure and gain
(`scripts/extract_aria_vrs.py:93-127`). Cameras are enumerated from the device
calibration rather than hardcoded, because Gen 1 and Gen 2 name their SLAM cameras
differently and reuse stream IDs inconsistently (`scripts/aria_utils.py:22-59`,
`scripts/extract_aria_vrs.py:435-448`). One device fact shapes everything: the RGB
camera is **rolling shutter** (10.1 ms readout on this profile) while the four SLAM
cameras are global shutter — so RGB geometry is modeled per-row, twice (§2.2, §2.3).

MPS runs visual-inertial SLAM plus offline bundle adjustment and emits `slam/`. The
pipeline consumes exactly four files, all in one shared world frame — **gravity
aligned, Z-up, metric metres**, origin wherever tracking initialized. That shared
frame is the load-bearing property of the whole project.

- **`closed_loop_trajectory.csv`** — device pose at ~1 kHz (155,800 rows / ~156 s
  here), loop-closed and bundle-adjusted; its gravity column reads (0, 0, −9.81),
  direct confirmation the frame is gravity-aligned Z-up. It is the **only source of
  camera poses anywhere in the pipeline** (`scripts/extract_aria_vrs.py:410`;
  re-read at training time for per-row poses, `scene/dataset_readers.py:565-566`,
  `scene/cameras.py:910-918`). Nothing in this repo ever solves for a pose.
- **`online_calibration.jsonl`** — time-varying calibration (1,557 records ≈ 10 Hz):
  per-camera FISHEYE624 intrinsics and camera-to-device extrinsics that follow thermal
  drift (`scripts/extract_aria_vrs.py:153-165`), plus `ReadoutTimesSec` — on this
  recording `[[4, 0.0101]]`, i.e. the RGB stream reads out in 10.1 ms. Factory
  calibration does not carry readout time at all, so this file is the only
  machine-readable source (`scripts/aria_utils.py:100-140`); reading it from data also
  keeps the pipeline profile-agnostic (38 ms on profile8 vs 10.1 ms here). The front
  SLAM pair's extrinsics in this file also provide the stereo baseline used by the
  stereo-depth method (§3.4).
- **`semidense_points.csv.gz`** — the global semi-dense point cloud: 4,292,410 points
  with per-point uncertainty. Used twice with different confidence thresholds:
  preprocessing keeps `inv_dist_std < 0.005, dist_std < 0.01` for sparse depth
  (`scripts/extract_aria_vrs.py:417-421`); training re-reads it with looser thresholds
  and seeds **one Gaussian per surviving point** (`scene/dataset_readers.py:689-694`).
- **`semidense_observations.csv.gz`** — the link table `uid ↔ (camera, frame, u, v)`
  (7.7–9.1 M rows per camera): which of the 4.3 M points a given camera actually saw
  in a given frame. This is what makes per-frame sparse depth possible without
  projecting points through walls (`scripts/extract_aria_vrs.py:510-513, 721-724`).
  MPS tracks at ~10 Hz against 30 Hz video, so ~33 % of frames have observations —
  the "1556/4674 frames (33.3 %)" lines in the logs are healthy, and 0 % would mean
  depth supervision silently does nothing. The RGB camera never appears in this table,
  which is why its sparse depth is built secondhand (§2.2).

### 2.2 Preprocessing (`scripts/extract_aria_vrs.py`, ~13 min)

One command turns VRS + MPS into a training folder; internally four passes
(`run_single_sequence`, `scripts/extract_aria_vrs.py:374-774`).

**Poses.** Each frame's pose is SE3-interpolated from the 1 kHz trajectory
(`scripts/aria_utils.py:175-234` — slerp on rotation, lerp on translation; bracketing
poses ≤1 ms apart, so effectively exact), then composed as
`T_world_camera = T_world_device @ T_device_camera`
(`scripts/extract_aria_vrs.py:173-184`). Frames outside the trajectory window are
dropped — the 4,719 → 4,675 frame loss is MPS initialization, not a bug.

**Rolling shutter bookkeeping.** For a rolling-shutter camera "the frame's pose" is
ill-defined, so three timestamps are stored per frame — readout start / center / end
(`scripts/extract_aria_vrs.py:96-124`), the center convention matching Aria's
documentation (the `--timestamp_convention readout_start` flag reproduces upstream's
half-readout pose bias for comparison). Training later recovers the readout time as
`timestamp_read_end − timestamp_read_start` (`scene/dataset_readers.py:483`) without
ever seeing an MPS file.

**Rectification.** Every fisheye frame is resampled to an ideal **pinhole** image
(`undistort_image`, `scripts/aria_utils.py:347-387`). `--rectified_rgb_focal` (1008)
is the *output* pinhole focal and therefore chooses the FOV
(`FOV = 2·atan((w/2)/f)`; 1008 at width 2016 gives exactly 90° horizontal);
`--rectified_rgb_size` (1512) is the output height, width derived from aspect ratio
(`scripts/aria_utils.py:430-431`). Output intrinsics are exact by construction:
`fx = fy = focal`, `cx = (w−1)/2`, `cy = (h−1)/2` (`scripts/aria_utils.py:437-441`) —
which is also why downstream consumers (COLMAP in the photogrammetry design, §3.6) can
use a clean `PINHOLE` model. Three auxiliary images ride the same warp so they stay
pixel-aligned: a vignette (all-ones on Gen 2 — the ISP already corrects shading,
`scripts/aria_utils.py:575-591`), a valid-pixel mask, and `image_index.png` — a warped
row-index ramp that records which *raw sensor row* each rectified pixel came from
(`scripts/aria_utils.py:525-550`), the key to rolling-shutter handling at render time.

**Sparse depth.** For each SLAM camera, observations are joined to frames by nearest
timestamp within 16 ms, the observed points projected in one batched call into the
rectified camera, and written per frame as `u, v, z` plus MPS uncertainties
(`scripts/extract_aria_vrs.py:952-1121`). The RGB camera gets its depth secondhand:
each RGB frame takes the temporally nearest frame of *each* SLAM camera, unprojects
that camera's sparse depth to world points, and reprojects into the RGB frustum
(`scripts/extract_aria_vrs.py:777-922`) — so RGB sparse depth is denser than any
single SLAM camera's, and reaches deeper (median depth ~5.1 m vs 2.8 m here).

**The one file training reads** is `transforms_with_sparse_depth.json`
(`conf/config.yaml:44`): per frame the rectified pinhole intrinsics, camera→device
extrinsic, camera→world transform at readout center, the three timestamps,
per-frame exposure/gain, and the relative path to that frame's sparse-depth JSON.
Two symlinks planted next to it (`scripts/extract_aria_vrs.py:546-563`) point back
into the MPS folder — training re-reads the trajectory and point cloud through them,
so the training folder is self-contained only while the MPS folder stays put.

### 2.3 Training (`train_lightning.py`, `model/vanilla_gsplat.py`, `model/GS2D_gsplat.py`)

The rendering backend is NVIDIA's **gsplat 1.5.3**; this repo's contribution is
everything around it (Aria cameras, exposure model, rolling shutter, losses,
strategies).

**What a Gaussian is here.** A flat `ParameterDict` of per-Gaussian tensors
(`model/vanilla_gsplat.py:55-169`): 3D mean (in MPS world metres), log-scales,
quaternion, logit-opacity, and degree-3 spherical-harmonics color (16 coefficients per
channel). Rendering projects every ellipsoid to a 2D "splat", sorts by depth, and
alpha-composites front to back (`model/vanilla_gsplat.py:965-983`).
`render_mode="RGB+ED"` additionally returns **expected depth** — the alpha-weighted
mean depth along each ray (`model/vanilla_gsplat.py:982-987`) — which can legally land
*between* two real surfaces; this matters for meshing later.

**Initialization.** One Gaussian per filtered semi-dense point
(`scene/dataset_readers.py:687-703`, `model/vanilla_gsplat.py:74-77`): geometry starts
where SLAM saw geometry, which is why 30 k iterations converge at all.

**The split and the exposure model.** `train_split: "7-1"` holds out every 8th frame
(`scene/dataset_readers.py:663-675`); with no separate test folder those 585 frames
are the test set, so the final metrics are genuinely held out. The model predicts
scene *irradiance*; each camera then exposes it by its own
`exposure_duration × gain` and vignette (`scene/cameras.py:496-543`) — without this,
auto-exposure flicker would be baked into geometry.

**The loss** (`model/vanilla_gsplat.py:1027-1193`):
`0.8·L1 + 0.2·DSSIM` on the exposed render. **Depth supervision is off by default**
(`depth_loss: false`, `conf/opt/simple_gsplat_30K.yaml:135`) — the sparse depth that
preprocessing works hard to build is not a training signal in the runs reported here;
it still feeds the per-camera near-plane of the anti-aliasing filter and the
rolling-shutter motion estimate.

**Rolling shutter at render time** (from iteration 10,000): each frame estimates how
fast its pixels move from its own sparse depth (`scene/cameras.py:949-1004`), picks
1–8 readout time-slices so each slice moves ≲1 pixel, renders all slice poses in one
batched call, and composites the final image by picking, per pixel, the slice its
*source sensor row* was actually read in — using the rectified row-index map from
§2.2 (`scene/cameras.py:1029-1035`, `model/vanilla_gsplat.py:814-891`). This is why
VRAM spikes at iteration 10 k, and why the 2DGS retrain had to run RS-off (§3.3).

**Densification and the reason the scene trains at all.** gsplat changes the Gaussian
count during training via a pluggable strategy (`model/vanilla_gsplat.py:581-625`).
The classic `default` strategy (split/duplicate on high image-plane gradient, prune
low opacity) has **unbounded N** — on this foliage-heavy outdoor scene it ran to
~15 M Gaussians and OOM'd the 16 GB card. The `MCMC` strategy reframes densification
as sampling and, crucially, takes **`cap_max` — a hard ceiling on N** (1.5 M in the
real run). The final model landed exactly at the cap: the budget, not the scene, was
the binding constraint. MCMC's exploration noise is also what sprays a few thousand
near-invisible Gaussians kilometres away — the reason the outlier filter exists (§2.4).

**3DGS vs 2DGS** (`model/GS2D_gsplat.py`). Same parameterization, different primitive:
3DGS is a volumetric ellipsoid whose "depth" is a soft expected value, free to be
wrong between surfaces because only the composited *color* is supervised. 2DGS is a
**flat oriented disk** (surfel) with a well-defined tangent plane; its rasterizer
returns analytic surface normals and ray-disk intersection depth, and its training
adds two geometry regularizers 3DGS cannot have (`model/GS2D_gsplat.py:423-462`): a
normal-consistency loss (disks must lie flat on the surface they render) and a
depth-distortion loss (each ray's alpha mass pulled into one thin shell). Those two
losses are the mechanism behind "2DGS depth is crisper" — and the reason 2DGS becomes
the geometry asset while 3DGS stays the visual asset (§3.3). Constraint: gsplat's 2DGS
rasterizer supports only pinhole cameras (`model/GS2D_gsplat.py:62-64`) — satisfied
here because everything is rectified to pinhole anyway.

### 2.4 Post: filter, export, load

**Outlier filtering is required, not optional.** MCMC leaves 18,816 of 1.5 M Gaussians
(1.25 %) beyond 50 m, median opacity ~0.011, some past 100 km. Visually nothing — but
a USD asset's bounding box is computed over *all* primitives regardless of opacity, so
the unfiltered export declared a **~120 km AABB around a 10 m driveway**, wrecking
viewport framing, camera near/far (hence depth precision), and any logic keyed to
extents. `scripts/filter_splat_outliers.py:32-64` drops everything beyond `--radius`
(50 m) of the **median** Gaussian position (not the origin — the MPS origin sits
wherever tracking initialized). Post-filter: 1,481,184 Gaussians, AABB ±50 m.

**USDZ export and the Isaac frame fix.** The export wrapper runs 3dgrut's
`ply_to_usd.py` with `dataset=None`, so the Gaussian *payload* keeps raw MPS world
coordinates (`threedgrut/export/usdz_exporter.py:70-79`). But 3dgrut *unconditionally*
authors a Y-down→Z-up conversion rotation `(x,y,z) → (−x,−z,−y)` onto the Volume prim
— correct for COLMAP-frame PLYs, exactly one conversion too many for a PLY that is
already Z-up: it lays the scene on its side. The export script now strips it with
`scripts/fix_nurec_usdz_frame.py` (identity-rewrite inside the USDZ; refuses any
unexpected transform, `scripts/fix_nurec_usdz_frame.py:48-54, 102-113`). Full story
in §3.1.

**Loading.** `scripts/isaacsim_load_splat.py` is a headless smoke test: it references
the USDZ under `/World/AriaSplat`, verifies the NuRec prims resolve in the RTX
renderer, verifies the Volume's composed transform is **identity** (so a regressed
asset is caught mechanically, `scripts/isaacsim_load_splat.py:64-83`), and writes a
`--report` JSON — the only trustworthy evidence, because Kit swallows stdout and
force-exits (`scripts/isaacsim_load_splat.py:26-31`).

### 2.5 The core mental model: a splat is not a surface

`point_cloud.ply` is 1.5 M soft ellipsoids optimized for exactly one objective:
reproduce the training photographs when alpha-composited from the training viewpoints.
It is a view-synthesis machine, not a geometric model. Nothing in the objective
demands Gaussians lie *on* surfaces: a wall can be several translucent layers
straddling the true surface; fog-like Gaussians can hang in free space; the rendered
"depth" is an opacity-weighted average that can land where nothing physically exists.

Consequently **mesh extraction is always a separate decision**, imposing assumptions
training never enforced — and every method in Section 3 is one such decision
procedure. Two structural facts follow:

1. **Unobserved floor cannot be recovered by any splat variant.** The only gradient
   source is reprojection error against captured pixels
   (`model/vanilla_gsplat.py:1101-1108`); a region in zero training images contributes
   zero loss terms and stays unconstrained. Egocentric capture makes this systematic:
   a head-mounted camera pitched at the horizon rarely images the floor within 1–2 m
   of the feet, and the patch directly underfoot is occluded by the wearer's own body
   — so the strip a robot must drive on is the *least-observed* geometry in the scene.
   No fancier primitive changes this; the information must come from outside the
   images. That is precisely the asymmetric value of the trajectory floor (§3.5): the
   closed-loop trajectory is a 1 kHz record that a human *physically stood on* every
   point of the walked path.
2. **Fixed poses, no gauge freedom.** A pipeline that solves its own poses (monocular
   SfM, for instance) recovers geometry only up to an arbitrary similarity transform —
   unknown global rotation, translation, and scale — which would then need registering
   to the splat. This pipeline never solves a pose: everything inherits the MPS
   closed-loop frame, metric and gravity-aligned by construction, and the camera-side
   convention (X right, Y down, Z forward) matches COLMAP's, so the repo performs no
   coordinate conversion anywhere (`scripts/extract_aria_vrs.py:171-172`,
   `scripts/aria_utils.py:445-447`). The one place a conversion crept in — 3dgrut's
   baked rotation — is stripped deterministically at export. This invariant is what
   lets the splat and the collision mesh coexist in Isaac Sim without registration,
   and it is why the photogrammetry design (§3.6) forbids COLMAP's `mapper`.

### 2.6 What the code actually does (verified findings)

Reading the code against the folklore turned up findings the rest of the report
depends on (full audit: `DOC_report.md`):

- **`scene.data_factor` and `scene.pcd_stride` are inert.** Both keys exist in
  `conf/config.yaml` (lines 61 and 41) and the training wrapper passes
  `data_factor=2 pcd_stride=2` — but no code reads either key (grep over `scene/`,
  `model/`, `train_lightning.py`; the one `stride` parameter in
  `utils/point_utils.py:17-33` is never called on the Aria path). Verified
  empirically: the completed run's `cameras.json` and rendered test images are full
  2016×1512 at fx = 1008. So the successful run trained at **full resolution**, and
  what actually fixed the OOM was MCMC + `cap_max=1.5M` (plus
  `expandable_segments`), not downscaling.[^df]
- **The 2DGS class ignored the MCMC config block** — `Gaussians2D._create_strategy`
  constructed `MCMCStrategy()` with no arguments, silently using gsplat's default
  `cap_max=1,000,000` and ignoring `opt.mcmc_strategy.*` entirely. Found in the code
  audit, fixed before the 2DGS retrain (the fix mirrors
  `model/vanilla_gsplat.py:613`), and verified to work: the retrained 2DGS PLY
  contains exactly 1,500,000 Gaussians.
- **The 3DGS MCMC path itself had never run** before this program: `_create_strategy`
  read config keys under the wrong names and crashed at startup. The (uncommitted) key
  mapping at `model/vanilla_gsplat.py:613-620` is load-bearing — MCMC's `cap_max` is
  the only Gaussian-count bound, i.e. the only reason this scene trains in 16 GB.
- Smaller confirmations: the DSSIM weight 0.2 is hard-coded
  (`model/vanilla_gsplat.py:1107`; the `ssim_lambda`/`dssim_lambda` config keys are
  decorative); depth supervision off by default; preprocessing still imports the old
  `project()` from `scripts/aria_utils.py` rather than the fixed copy in
  `utils/point_utils.py` (measured impact on this recording: zero, but the latent bad
  path exists); `l1_grad=true` would crash on a config-key typo
  (`model/vanilla_gsplat.py:1110`).

---

## 3. The methods

Framing for every section below: input is always the same VRS + MPS pair; the product
is always something Isaac Sim loads in the MPS world frame. The visual splat produces
the USDZ; every mesh method produces a PLY that
`scripts/isaacsim_build_env.py` converts to a `.collision.usd` sibling (cached,
Z-up/metres, provenance recorded in `customData`) and mounts invisible under
`/World/AriaCollision`.

**How candidates are scored (the eval harness).** `scripts/eval_mesh.py` /
`scripts/compare_meshes.py` (built first, validated on synthetic known-good/known-bad
meshes; `E0_report.md`) score every mesh in the MPS frame: **floor coverage** —
the trajectory rasterized to a 10 cm XY grid, buffered 0.75 m, one downward ray per
cell, covered = hit within ±15 cm of local trajectory Z minus eye height (7,144 cells
= 71.44 m² footprint); **largest hole** — biggest connected component of missed cells
(a robot cares about the one hole it falls through, not many small ones);
**semi-dense agreement** — point-to-mesh distance against the 938,113
confidence-filtered MPS points (independent geometry, but wall/foliage-dominated, so
use it to *rank*, not as absolute floor accuracy); **hygiene** — triangles,
components, non-manifold edges, and "junk" = total area of components < 0.1 m².
Two eval-harness facts recur below: the scene has **~1.6 m of real grade** (a flat
plane scores only 0.22), and the default eye height 1.6 m was an assumption until the
splat-depth mesh calibrated it to **h = 1.6683 m** (§3.2).

### 3.1 The visual splat

**How do I run this (VRS/MPS → USDZ in Isaac Sim):**

```bash
conda activate ego_splats
cd /home/sun/Desktop/aria_proj/egocentric_splats

bash scripts/bash_local/run_gen2_outside.sh          # VRS+MPS -> rectified frames + sparse depth (~13 min, 23 GB)
bash scripts/bash_local/train_gen2_outside.sh        # 30k iters -> point_cloud.ply
bash scripts/bash_local/export_gen2_outside_usdz.sh  # filter -> ply_to_usd (3dgrut env) -> frame fix -> .usdz

~/isaac-sim/python.sh scripts/isaacsim_load_splat.py \
    output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/isaacsim/Outside_20260812_141244.usdz \
    --report /tmp/usd_report.json     # read the JSON, not stdout
```

The training call that matters inside the wrapper:
`train_lightning.py train_model=3dgs opt=simple_gsplat_30K
opt.densification_strategy=MCMC opt.mcmc_strategy.cap_max=1500000
opt.handle_rolling_shutter=true scene.input_format="aria"` with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Train **RGB-only**: the
SLAM/joint modes write 1- or 4-channel PLYs no external tool reads.

**Differentiating factor.** This is the photoreal half of the deliverable and the
scene's appearance ground truth. It needs no geometry method to exist — but it offers
physics nothing: the USDZ contains a NuRec `Volume`, no meshes, no collision geometry.
Everything else in this report exists because of that.

**How it went.** Four attempts to train: the documented `default`-strategy command
OOM'd (~15 M Gaussians in the SH backward pass); the MCMC path crashed on the config
bug (§2.6) until fixed; the final run completed at full 2016×1512 with
`cap_max=1.5M`, peaking near 4 GB of 16 GB. Held-out (585 frames): **PSNR 25.55 /
SSIM 0.846 / LPIPS 0.389** (25.42 / 0.843 / 0.390 under the matched RS-off protocol
of §3.3 — rolling-shutter handling at eval is worth only ~0.13 dB).[^psnr] Export:
372 MB PLY → 1,481,184 Gaussians after the ±50 m filter → 175 MB USDZ.

**The Isaac frame fix (and the obsolete manual-rotation ritual).** The first import
into Isaac Sim looked *very small and incorrectly oriented*, and the working
assumption was to find it with `F` and rotate it by hand — a per-import ritual that
would have had to be repeated by every consumer. The diagnosis
(`E1_report.md`) rejected the unit-mismatch hypothesis with direct evidence (asset and
stage both `metersPerUnit=1.0`, `upAxis=Z`) and root-caused the orientation: 3dgrut
bakes the `(x,y,z)→(−x,−z,−y)` rotation onto the Volume prim unconditionally (§2.4);
the world AABB of the loaded asset matched the local extent pushed through exactly
that rotation, axis-for-axis. The "very small" appearance was a framing artifact, not
scale: ~85 % of the splat's opacity mass sits within 10 m of the scene centre, but
`F` frames the ±50 m filter-radius AABB. The fix landed at the **asset layer**
(`scripts/fix_nurec_usdz_frame.py`, run automatically by the export script), so every
consumer — File→Open, drag-and-drop, the drive-test stage — gets a correct asset, and
the loader now asserts `volume_frame_is_identity` so a regressed export is caught by
the smoke test rather than by eyeballing. After the fix the composed world bounds
equal the PLY bounds verbatim: [−48.06, −48.53, −49.76] … [49.07, 48.66, 47.63]. The
manual rotation is obsolete; performing it now would *break* the asset.

**Failure modes (measured).** `default` densification is unusable on this scene/GPU
(unbounded N → OOM ~an hour in); the training process exits code 0 even when it
crashes (Hydra swallows the status — check for the PLY, not the exit code); unfiltered
export produces the 120 km AABB; and the splat contributes no collision geometry by
construction.

### 3.2 Splat-depth meshing, 3DGS variant

**How do I run this (trained splat → collision-candidate mesh):**

```bash
PY=/home/sun/miniforge3/envs/ego_splats/bin/python
BASE=output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512
DATA=/home/sun/aria/processed/Outside_20260812_141244/camera-rgb-rectified-1008-h1512

# aggressive depth-render copy (visual asset untouched)
$PY scripts/filter_splat_outliers.py $BASE/point_cloud/iteration_30000/point_cloud.ply \
    --radius 50 --min-opacity 0.3 --output $BASE/mesh_tsdf/depth_render_op03.ply

# render depth from all training poses, fuse into a TSDF, marching-cubes
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/extract_mesh_tsdf.py \
    --ply $BASE/mesh_tsdf/depth_render_op03.ply --data-dir $DATA \
    --output $BASE/mesh_tsdf/mesh_tsdf_op03.ply

$PY scripts/eval_mesh.py $BASE/mesh_tsdf/mesh_tsdf_op03.ply
```

`extract_mesh_tsdf.py` renders expected depth (`RGB+ED`) from the 4,675 camera poses
subsampled by pose delta (≥0.10 m / ≥10°: **1,031 frames kept**), masks invalid/sky
pixels (alpha < 0.5) and depth outside [0.1, 6.0] m, and fuses with Open3D's
`ScalableTSDFVolume` at voxel 2 cm / truncation 8 cm. ~90 s of fusion at ~11 fps.
The 6 m cutoff is deliberate: splat depth was only ever trained/consumed inside the
render config's 7 m clip, and the whole walkable footprint is seen from ~2 m by some
camera.

**Differentiating factor.** The zero-extra-capture route: it reuses the already-trained
visual model, so a collision candidate costs ~2 minutes of filtering plus ~2 minutes
of fusion. Its baseline comes from the splat itself — which is exactly its weakness,
because expected depth over translucent blobs inherits every place the splat faked
appearance without geometry (§2.5).

**How it went vs the others.** Scorecard (default h = 1.60): floor coverage
**0.5013**, largest hole **19.94 m²**, semi-dense median **3.19 cm** (79.8 % of points
within 10 cm) — locally accurate where the camera looked, and worst-in-program
coverage everywhere else. Hygiene is the worst of any candidate: 10.7 M triangles,
767,859 components, **341 m² of junk** (splat fuzz around vegetation and far-field
smear — raising the opacity filter to 0.5 left the junk unchanged while eroding real
surface, so 0.3 is the kept setting). At the calibrated h = 1.6683 it rescores to
0.478 / 18.88 m². Its one lasting contribution: casting rays from the trajectory onto
this mesh **calibrated the eye height — h = 1.6683 m** (median over 856 usable walked
cells; mean 1.7145, std 0.2754) — the constant every later experiment uses.

**Failure modes (measured).** The failure decomposition shows through-hits *below* the
expected floor dominate (1,623 cells, 975 of them >0.5 m deep): rays pass through
floor gaps onto smeared under-floor junk — the literal fall-through geometry. The
19.94 m² largest hole sits in the central steep-grade region where the harness's own
buffered-cell expectation is partly a guess (1.3 m of grade inside one hole; 85 % of
its cells are buffer cells)[^hole] — but the 2DGS variant halved it under identical
scoring, so much of it is real. Not an eye-height offset: rescoring at the calibrated
h *reduces* covered cells (3,405 vs 3,581) — the misses are scatter, not shift.

### 3.3 Splat-depth meshing, 2DGS variant

**How do I run this (retrain as surfels, then the identical mesh pipeline):**

```bash
bash scripts/bash_local/train_gen2_outside_2dgs.sh   # train_model=2dgs, MCMC cap_max=1.5M, RS off (see below)

B2=output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512-2dgs
$PY scripts/filter_splat_outliers.py $B2/point_cloud/iteration_30000/point_cloud.ply \
    --radius 50 --min-opacity 0.3 --output $B2/mesh_tsdf/depth_render_op03.ply
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/extract_mesh_tsdf.py \
    --ply $B2/mesh_tsdf/depth_render_op03.ply --data-dir $DATA \
    --train-model 2dgs --output $B2/mesh_tsdf/mesh_tsdf_2dgs_op03.ply
```

Retrain cost: **~3 h 47 min** wall for 30 k iterations on the RTX 4080 SUPER (one
additional ~30 min attempt lost, next paragraph). The MCMC-config fix from the code
audit (§2.6) made the run honor `cap_max=1500000` — verified: exactly 1,500,000
Gaussians on disk.

**RS-off was forced, not chosen.** The first attempt ran with rolling shutter on per
the brief and OOM'd at the first validation epoch: the rolling-shutter deferral
(`handle_rolling_shutter_start_iter=10000`) applies only to the *training* branch, so
every eval frame rendered the full 4–8-pose motion array, and `rasterization_2dgs`
carries far more per-camera state (surfel intersection, normal/distortion buffers)
than the 3DGS rasterizer — a 1.98 GiB `isect_tiles` allocation tipped 16 GB.
Retrained RS-off; the evaluation asymmetry was then removed by re-evaluating **both**
models RS-off with `scripts/eval_splat_test.py` (same test path, masking, exposure).
Geometry impact is negligible for meshing: at walking speed the camera moves ~1.5 cm
during the 10.1 ms readout, far under the 8 cm TSDF truncation, and mesh extraction
renders RS-off for both models anyway.

**Differentiating factor.** The only method that changes the *representation* to favor
geometry: flat disks with normal-consistency and depth-distortion losses concentrate
the model onto actual surfaces (§2.3), so the same TSDF pipeline receives crisper
depth. Baseline: the identical fusion parameters, filter settings, and (near-identical)
Gaussian budget as the 3DGS variant — 884,386 vs 904,972 Gaussians into fusion — so
the comparison isolates the primitive.

**How it went vs the others.** Decisively better than 3DGS as geometry, at a real
visual cost. Under identical everything (default h): coverage **0.640 vs 0.501**
(on-path 0.716 vs 0.560), largest hole **8.86 vs 19.94 m²** (−56 %, same central
steep-grade region), deep (>0.5 m) through-hole cells **150 vs 975** (−85 % of the
robot-falls-through class), junk **86 vs 341 m²**, 6.6 M vs 10.7 M triangles, and
2.7 m less under-floor smear in the AABB. Semi-dense *median* is statistically the
same (3.47 vs 3.19 cm); the worse mean/p95 is an extent artifact — the 2DGS mesh has
~31 % less total surface because it models less far-field smear, which is exactly
what a collision mesh should not have. The cost: **PSNR 23.69 vs 25.42 (matched
RS-off protocol)**, SSIM 0.822 vs 0.843, LPIPS 0.453 vs 0.390 — flat disks hurt most
on thin structures and view-dependent vegetation. Hence the role split the rest of
the program adopts: **3DGS remains the visual asset; 2DGS is the geometry asset.**
At the calibrated h it rescores to 0.616 / 7.21 m² (the 2DGS floor floats ~3–4 cm
high, so raising the expectation *lowers* its score — see §3.7's fusion pins).

**Failure modes (measured).** Still only 64 % floor-complete — surfels cannot conjure
gradients from unobserved pixels (§2.5); the largest hole persists in the same
steep-grade region; ~2.4× slower depth rendering at fusion time (222.6 s vs ~91 s,
irrelevant offline); and rolling-shutter handling is untrainable within 16 GB for
this model (eval renders OOM), leaving a small protocol asymmetry that the matched
re-evaluation bounds at ~0.13 dB.

### 3.4 Stereo depth

**How do I run this (VRS/MPS → depth maps → collision-candidate mesh):**

```bash
# one-time setup, own env (depth_from_stereo: Python 3.11, torch 2.10+cu128)
git clone https://github.com/facebookresearch/projectaria_gen2_depth_from_stereo
cd projectaria_gen2_depth_from_stereo && git submodule update --init
conda env create -f environment.yml
# + FoundationStereo checkpoint 23-51-11 (ViT-Large, model_best_bp2.pth, 3.3 GB)

# depth export: MPS closed-loop poses + online calibration, LR-consistency masks
/home/sun/miniforge3/envs/depth_from_stereo/bin/python export_depth_from_stereo.py \
  --vrs /home/sun/aria/Outside_20260812_141244.vrs \
  --mps /home/sun/aria/mps_Outside_20260812_141244_vrs \
  --stereo_model ./FoundationStereo/ckpts/pretrained_models/23-51-11/model_best_bp2.pth \
  --output_dir <repo>/output/Outside_20260812_141244/stereo_depth/export_stride3 \
  --lr_check --stride 3

# fusion (repo, ego_splats env; CPU, ~12 s) — same shared TSDF stage as the splat meshes
python scripts/extract_mesh_stereo_tsdf.py \
  --export-dir output/Outside_20260812_141244/stereo_depth/export_stride3 \
  --depth-trunc 4.0 \
  --output output/Outside_20260812_141244/stereo_depth/mesh_tsdf_stereo_d40.ply
```

Meta's first-party Gen 2 tool rectifies the front SLAM pair and runs FoundationStereo
for zero-shot disparity; `--mps` makes `T_world_camera` MPS closed-loop and the
calibration online — the same frame as everything else, by construction. 1,558 frames
at stride 3 (10 Hz), 17.5 min at 1.5 fps / ~3.3 GB VRAM. Fusion reuses the shared
stage factored out of the splat-mesh script (`scripts/tsdf_fusion.py`), so parameters
are identical by construction (voxel 2 cm, trunc 8 cm, pose-delta subsample →
842 frames); LR-inconsistent pixels (11 %) are masked before integration.

**Differentiating factor.** The only *direct measurement* route: no training, no splat
in the loop — metric depth from a calibrated stereo rig, with a closed-form error
model. Baseline read from the MPS online calibration: **B = 134.961 mm**,
rectified **f = 305.867 px**, so σ_Z = Z²·σ_d/(f·B) with f·B = 41.28 m·px:

| Z | σ_Z @ σ_d = 0.5 px | σ_Z @ σ_d = 1.0 px |
|---|---|---|
| 1 m | 1.2 cm | 2.4 cm |
| 2 m | 4.8 cm | 9.7 cm |
| 5 m | 30.3 cm | 60.6 cm |
| 10 m | 1.21 m | 2.42 m |
| 20 m | 4.85 m | 9.69 m |

σ_Z crosses the 2 cm TSDF voxel at ~1.3 m and the 8 cm truncation band at ~2.6 m
(optimistic σ_d = 0.5 px) — this rig is a near-field instrument.

**Sanity gate before fusing.** Projected the 938,113 filtered semi-dense points into
five frames spread across the walk and compared against the tool's depth at
LR-consistent pixels: **pooled signed median dz = −0.5 mm** — pose, intrinsics, and
depth scale all agree with the MPS world (abort criterion was >20 cm). Measured
accuracy 5.2 cm at ~2 m matches the model's 4.8 cm prediction: FoundationStereo really
delivers ~half-pixel disparity on these 512×512 monochrome images.

**How it went vs the others.** Best observation-based candidate on nearly everything.
Depth cutoff swept (2.5 / 4.0 / 6.0 m): 4.0 m wins exactly as the error model
predicts — the 2.5–4 m band still adds real surface, 4–6 m mostly adds smear (junk
×3.3, coverage down). Kept mesh (default h): coverage **0.864**, largest hole
**0.91 m²**, semi-dense median **2.18 cm** (87.1 % within 10 cm — best of all
candidates), **0 non-manifold edges**, 27.2 m² junk, 3.26 M triangles. On-path
coverage 0.946 vs 0.716 (2DGS); **deep through-hole cells: 0** (vs 150 / 975). And
the stereo floor sits at the *calibrated* floor height: its apparent −7.6 cm offset
under the default harness is within 8 mm of what the calibrated eye height predicts —
rescoring at h = 1.6683 raises it to **0.903 / 1.50 m²**, while the splat meshes
(biased a few cm high by foliage/smear) drop. Cheapest GPU route by far: 17.5 min of
inference vs ~4 h of 2DGS retraining. The LR masks were ablated: they buy ~40 % junk
reduction for zero coverage cost — masks on.

**Failure modes (measured).** Blind past ~4 m *by construction* — the d60 sweep shows
including 4–6 m already degrades every metric, so distant walls, buildings, and canopy
cannot come from this sensor (its AABB is ±8–9 m vs the splat meshes' ±12 m);
far-field geometry must come from the splat side. Under-coverage where the wearer
never pointed the front cameras (3.3 % no-hit in the buffer ring — same failure class
as the splats, milder). And ~10 % of the footprint remains unobserved-floor holes —
the gap the trajectory floor exists to fill.

### 3.5 Trajectory floor

**How do I run this (MPS trajectory → floor mesh + heightfield):**

```bash
$PY scripts/floor_from_trajectory.py                          # defaults: h=1.6, cell 0.10 m, clearance 0.75 m (~0.5 s, CPU)
$PY scripts/floor_from_trajectory.py --calibrate-with <recovered mesh>   # fit h, rebuild at the fitted value
# products: floor_from_trajectory/floor_mesh.ply + floor_heightfield.npz (+ .json)
```

Rasterizes the 155,800-pose closed-loop trajectory into 10 cm cells, per-cell
**median** device Z minus eye height h = observed floor; dilates 0.75 m laterally;
fills unobserved footprint cells by linear interpolation between passes (flagged
`interpolated`) or nearest-observed (flagged `extrapolated`); light masked smoothing
to remove gait bounce. Emits both a triangle mesh and a PhysX-friendly heightfield
with a per-cell category array so consumers can weigh observed vs guessed cells.

**Differentiating factor.** The only method that needs **no images at all**, and the
only one with a traversability *guarantee*: a human physically stood on every observed
cell — a certificate for exactly the strip the cameras systematically never see
(§2.5). It also follows the scene's real ~1.6 m of grade automatically, where any
fitted plane fails (coverage 0.22). Its baseline requirement runs the other way:
h must be calibrated against geometry some *other* method recovered — done against the
3DGS splat-depth mesh, yielding **h = 1.6683 m** (median; std 0.2754 — the spread
comes largely from mesh defects, so the robust median is the usable value).

**How it went vs the others.** As a floor *scorecard* it is untouchable: coverage
**0.978, largest hole 0.48 m²** (rebuilt at the calibrated h, scored under the
default-h harness; 0.987 / 0.37 m² when built and scored at the same h)[^floorproto],
one single component, zero junk, 14,288 triangles, built in half a second. Semi-dense
median 0.41 m is expected and not an accuracy number — the MPS cloud is
wall/foliage-dominated, and a floor-only mesh is far from most of it. Per-cell
strength on the walked path: median 120 trajectory samples per cell, per-cell device-Z
std median 2.9 mm.

**But completeness is not drivability.** In the drive test (§3.8) the floor-only mesh
is catastrophic *as a collider*: the robot cruises at near-full speed for 18 m and
then **rolls over at 20 % of the path, at both tested speeds**, in a real ~15 cm
gutter that the smooth prior renders as a clean steep-walled dip — one wheel drops in
and nothing arrests the roll, because the prior contains no obstacles and no texture
of any kind. 0 stalls, then a flip. Its best-in-program scorecard and worst-in-program
drive result is the clearest demonstration in the project that per-cell metrics and
acceptance tests measure different things.

**Failure modes (measured).** A trajectory is a curve, not a surface: only 1,028 of
7,144 footprint cells (14 %) were actually walked; 4,315 are interpolated and 1,801
extrapolated — the guarantee degrades to an assumption off-path, and linear
interpolation between passes at different heights can bridge across a real drop the
walker went around. The 1.34 % missing cells under the harness are almost entirely
interpolated cells where the harness's nearest-neighbour expectation and the floor's
linear interpolation disagree — both are guesses; nobody walked there. And h is a
constant: posture changes (the wearer crouching produced up to 0.85 m dips in device
Z) enter the floor directly unless cleaned, which the drive test's waypoint
preparation had to do explicitly.

### 3.6 Photogrammetry — designed, reserved for the author

*This slot is deliberately left unimplemented; it is reserved as the author's own
hands-on deliverable. Everything below is design and hypothesis — there are no
results, and none are implied.*

**The design (COLMAP dense MVS with MPS-fixed poses).** The critical constraint:
**never let COLMAP solve poses.** MPS poses are bundle-adjusted, gravity-aligned, and
metric, in the same frame as everything else; monocular SfM would return an arbitrary
similarity frame (the gauge-freedom argument of §2.5) and the alignment invariant
would be lost. So the flow uses COLMAP's *second half* only, on the already-rectified
pinhole frames (whose exact intrinsics map cleanly to COLMAP's `PINHOLE` model with
zero distortion, §2.2):

1. Build `cameras.txt` / `images.txt` from `transforms_with_sparse_depth.json`
   (fixed poses, known intrinsics).
2. `feature_extractor` → `sequential_matcher` (a continuous walk needs sequential,
   not exhaustive, matching).
3. `point_triangulator` against the **fixed** poses — no `mapper`, ever.
4. `patch_match_stereo` → `stereo_fusion` → `delaunay_mesher` (with `poisson_mesher`
   as a comparison).

Practical notes carried over from the rest of the program: subsample the 4,675 frames
by pose delta (~0.10 m / 10°, the shared protocol) before MVS; single shared 16 GB
GPU, so serialize against training jobs.

**The hypothesis.** Patch-match MVS needs texture. The expected failure is exactly the
collision-critical region: blank concrete floor, where photometric matching returns
nothing — so it should *lose* to stereo depth and the trajectory floor on floor
coverage. The expected win is the far field: textured walls, facades, and structure
beyond the ~4 m stereo horizon, where MVS with wide effective baselines (the whole
walk) can outresolve both the stereo rig and splat expected-depth. The interesting
measurement is precisely that asymmetry: how much floor survives `stereo_fusion`
versus how much wall.

**How it will be judged.** By the same instruments as everything else: an
`eval_mesh.py` scorecard (floor coverage / largest hole / semi-dense / hygiene, at
h = 1.6683), entry into the scoreboard of Section 4, candidacy as the fusion's
far-field partner (replacing or complementing 2DGS in the composite), and — if it
earns it — a drive-test run. When built, its run report joins the mapping table on
page 1 (the reserved E4 slot).

### 3.7 The fusion

**How do I run this (best parents + floor prior → the final collision candidates):**

```bash
B=output/Outside_20260812_141244
OUT=$B/collision_candidates

# compose: stereo near field kept intact; 2DGS triangles outside the stereo XY AABB
# appended (far-field junk <0.1 m2 removed); floor prior fused last
$PY scripts/compose_near_far_mesh.py \
    --near $B/stereo_depth/mesh_tsdf_stereo_d40.ply \
    --far  $B/camera-rgb-rectified-1008-h1512-2dgs/mesh_tsdf/mesh_tsdf_2dgs_op03.ply \
    --output <scratch>/prefuse_stereo_near_2dgs_far.ply
$PY scripts/floor_from_trajectory.py --eye-height 1.6683 \
    --fuse-with <scratch>/prefuse_stereo_near_2dgs_far.ply --output-dir $OUT/fused_best
# (same --fuse-with call on the raw stereo and raw 2DGS meshes builds the two
#  single-parent fusions used as controls)
```

Fusion mechanics (`floor_from_trajectory.py --fuse-with`): raycast the candidate at
every footprint cell; **pin absolute floor height** by the median prior-vs-mesh
offset and shift the whole mesh by it; keep cells that agree within 10 cm; patch
holes and rejections at prior height; strip rejected floor-band triangles (0.40 m
band, 1-cell dilation). Output includes the fused mesh, a fused heightfield, and a
per-cell provenance array (`source` = mesh-kept / hole-filled / rejected).

**Differentiating factor.** The only method that is a *combination policy* rather than
a sensor: it spends each source where it is strong — stereo where measurement is
voxel-accurate, 2DGS where only the splat has ever seen geometry, the trajectory prior
where nothing observed the floor at all — and it is the only route with a mechanism
that *guarantees* a floor surface under every footprint cell.

**How it went vs the others.** The height pins independently confirm the bias story:
the stereo mesh needed only **+0.9 cm** to sit on the calibrated prior; the 2DGS mesh
needed **−3.4 cm** (it floats high). The fused-best composition is clean by
construction — its footprint classification is byte-identical to fused-stereo's
(6,294 mesh-kept / 229 hole-filled / 621 rejected of 7,144 cells) because the 2DGS
far field never touches the floor corridor. Scorecards at h = 1.6683: fused-best
**0.9696 coverage / 0.40 m² largest hole / 2.87 cm sd-median / 26.7 m² junk / 1
non-manifold edge**, with the best semi-dense mean/p95 of the fusions (7.4 / 23.7 cm)
because the far field supplies the walls the stereo mesh lacks. All three fusions
passed the sanity gate (beat the parent on coverage AND largest hole, sd-median
within 1 cm; the ~0.7 cm sd-median rise is the height pin itself moving the surface
relative to the semi-dense cloud — the expected, bounded price of an absolute pin).
The residual-failure decomposition shows the point of the whole exercise: **zero
no-hit cells and zero below-tolerance cells — the fall-through failure class is
structurally eliminated.** Every residual miss is a first hit 0.15–0.5 m *above* the
prior floor (~92 % in interpolated/extrapolated cells): partly real vegetation and
wall bases inside the clearance ring — genuine no-drive zones a collision mesh
*should* contain — and partly stereo smear.

**Failure modes (measured).** Where the fusion *rejects* the parent floor and patches
at prior height, the patch meets the kept mesh in a **step** — the seam-step hazard
the drive test later exposed (188 cells >8 cm, max 0.21 m on fused-best; §3.8).
Fusion trusts the prior across the whole footprint, so a genuine low obstacle inside
the never-walked clearance ring that sits within the strip band would be flattened
into floor (impossible over walked cells — the space was free by demonstration).
The deliberately-kept 26.7 m² of small components inside the stereo near field may
contain real thin obstacles; a blanket component filter was rejected as risking
exactly what a collision mesh must keep.

### 3.8 The drive test

**How do I run this (candidate mesh + splat → composed stage → acceptance run):**

```bash
# stage build + physics benchmark (PLY -> cached .collision.usd conversion inside)
~/isaac-sim/python.sh scripts/isaacsim_build_env.py \
    --collision $OUT/fused_best_stereoNear_2dgsFar_trajfloor.ply \
    --approximation none --splat default --report visual_check.json     # add --bench for timings

# the acceptance test itself
~/isaac-sim/python.sh scripts/sim_drive_test.py \
    --collision $OUT/fused_best_stereoNear_2dgsFar_trajfloor.ply \
    --report drive_fused_best.json
```

A Nova Carter (differential drive, wheel radius 0.14 m, base 0.4132 m; measured
effective chassis clearance **~1–3 cm** at settle) is driven along the demonstrator's
own path: the closed-loop trajectory subsampled to ≥0.25 m spacing → **363 waypoints,
90.64 m**, expected floor = trajectory z − 1.6683 with a rolling-median clean that
replaced 4 crouching-posture outliers. Follower: P-turn on heading error, waypoint
radius 0.40 m with lookahead; stall = commanded velocity with <6 cm displacement over
6 s; three stalls at one waypoint with failed reverse-recovery = a **wedge** (recorded,
then the robot is teleported 2 waypoints ahead so one wedge cannot blind the rest of
the census — "distance before first wedge" keeps the honest autonomous number).
Fall-through = base >0.5 m below expected local floor. All evidence from `--report`
JSONs (Kit swallows stdout).

**Differentiating factor.** Everything else in this report is a proxy metric; this is
the requirement. It is the only instrument sensitive to *lateral* geometry — steps,
trenches, obstacle standoff, tip-over dynamics — which the per-cell harness is
structurally blind to.

**The physics-approximation benchmark first.** On the 4.12 M-triangle fused-best mesh,
with a dynamic drop-box as probe: static trimesh (`none`) cooks in 7.6 s cold / 1.3 s
warm and steps at 0.44 ms; **SDF** (resolution 256) is identical in step time with
+1.1 s setup — SDF contact generation matters for *dynamic* SDF meshes, and this
collider is static, so it buys nothing; **convexDecomposition is disqualified on
correctness**, not speed — 32 convex hulls bridge every hollow of a concave 25 m
terrain, and the probe box comes to rest **0.31 m above the true floor** and slides
(bench JSONs: rest z −1.3604 vs −1.6675 on the trimesh; the drive report's prose
rounds this to 0.32 m).
The "never use raw trimesh collision" folklore does not apply to a static collider:
the robot articulation dominates the step (≈11 ms) regardless. All drive runs use
`--approximation none`.

**Run-by-run results** (identical follower, path, physics; vmax 0.8 m/s unless noted):

| candidate | outcome | completion | dist before 1st wedge | fall-throughs | stalls | wedges (rescues) | max roll / pitch |
|---|---|---|---|---|---|---|---|
| **fused-best** | **completed** | **100 %** | 17.5 m | **0** | 44 | 11 | 43° / 39° |
| fused-best @ 0.5 m/s | completed | 100 % | 18.5 m | 0 | 52 | 12 | 63° / 42° |
| fused-2DGS | completed | 100 % | 6.8 m | 0 | 58 | 13 | 55° / 33° |
| raw stereo | **tipped** (pitch 75°) | 91.4 % (82.9 m) | 17.5 m | 0 | 33 | 7 | 41° / 75° |
| trajectory floor | **tipped** (roll 70°) | **20.2 % (18.3 m)** | — | 0 | 0 | 0 | 70° / 21° |
| trajectory floor @ 0.5 m/s | tipped (roll 71°) | 20.2 % | — | 0 | 0 | 0 | 71° / 21° |

**Zero fall-throughs on every candidate at every speed** — the class the program set
out to eliminate stays eliminated under real physics. Only the fused candidates finish;
fused-best does it with the fewest rescues and its wedge sites are geometric, not
speed-induced (same sites at 0.5 m/s). "Completion 100 %" always reads as "completable
with N rescues" — the autonomous number is distance before first wedge.

**The key discovery: fusion seams are curb steps.** An early fused-best attempt (the
first to survive spawn) wedged immovably at waypoint 70. Fine raycast probing found
a ~25 cm-wide, 10–15 cm-deep slot running exactly along the walked line, with flat
kept-mesh floor on both sides — and the fused heightfield's provenance array explains it: those cells
are `source = rejected` (the parent mesh sat >10 cm above the prior), so the fusion
patched them at prior height while the neighbours stayed kept-mesh, leaving a
curb-edged trench one robot-track wide. Wheels drop in; a ~1–3 cm-clearance chassis
grounds on the edges; forward and reverse are both dead. **The per-cell harness
cannot see this**: every trench cell and every shelf cell individually passes its own
±15 cm check — lateral discontinuity is invisible to per-cell metrics, and it took
the acceptance test to expose it. Census over the fused heightfields: fused-best has
**188 patch cells stepping >8 cm** against an adjacent kept cell (max 0.21 m,
1.88 m²), and 8 of its 11 wedge sites are seam cells; fused-2DGS has 393 such cells
(max 0.25 m) — matching its 2× rougher ride. The v2 mitigation is specific: feather
the patch↔kept transition over ~3–5 cells in the fusion, then re-gate and re-drive.

**The anomalies are partly real.** The waypoint-70 area is the scene's acid test —
every candidate fails there first, each in its own way: the trajectory prior itself
says the walked line dips ~15 cm there (a real gutter), and the raw stereo run wedges
against a 1.8 m-tall, ~10 cm-wide pillar standing 0.2 m off-path — a real pole or
stereo smear, present in all stereo-based meshes; the report recommends verifying the
handful of on-corridor pillars against the splat rendering before deleting anything.
Human-walkable does not imply Carter-drivable: a ~1–3 cm-clearance differential robot
fails on honest cm-scale roughness a higher-clearance platform would shrug off.

**Failure modes of the test itself (measured).** Teleport recovery makes the census
complete but inflates "completion"; the stall counter needed a yaw-deadlock trigger
added mid-program (the headline run predates it but was verified not to have entered
that state); spawn required a settle-retry protocol after a first-attempt tip on a
stereo-smear blob; CPU governor was `powersave`, so step timings are conservative
upper bounds.

---

## 4. The scoreboard

All candidates, scored by the eval harness at the calibrated h = 1.6683
(`collision_candidates/comparison_e6a_h16683.json` — the numeric ground truth), with
each raw source's original default-h (h = 1.60) headline in brackets for continuity
with the per-method reports.[^scoring] Drive results from `E6_drive_report.md` /
scratchpad `e6b/drive_*.json`.

| Candidate | Floor coverage | Largest hole (m²) | sd-median (cm) | Junk (m²) | Drive test |
|---|---|---|---|---|---|
| **Fused-best (stereo near + 2DGS far + floor prior)** | **0.9696** | **0.40** | 2.87 | **26.7** | **100 %, 0 fall-throughs, 11 rescues — recommended** |
| Fused-stereo (stereo + floor prior) | 0.9696 | 0.40 | 2.88 | 26.7 | not driven (near-field twin of fused-best) |
| Fused-2DGS (2DGS + floor prior) | 0.9565 | 0.47 | 3.82 | 85.8 | 100 %, 0 fall-throughs, 13 rescues, first wedge 6.8 m |
| Raw stereo [0.864 / 0.91 at default h] | 0.9029 | 1.50 | **2.18** | 27.2 | tipped (pitch 75°) at 91.4 % |
| Raw 2DGS splat-depth [0.640 / 8.86] | 0.6163 | 7.21 | 3.47 | 86.0 | not driven (superseded by its fusion) |
| Raw 3DGS splat-depth [0.501 / 19.94] | 0.4782 | 18.88 | 3.19 | 341.0 | not driven (superseded) |
| Trajectory floor [0.978 / 0.48 at default h][^floorproto] | 0.9866 | 0.37 | 40.8 | 0.0 | tipped (roll 70°) at 20.2 %, both speeds |
| Photogrammetry | — | — | — | — | reserved (design in §3.6; enters this table when built) |

Reading order matters: the trajectory floor "wins" the scorecard and loses the drive
test; raw stereo wins sd-median and flips the robot; only the fusions survive both
instruments. The scorecard columns rank *floor completeness and local accuracy*; the
drive column is the requirement.

---

## 5. Known limitations & next steps

1. **Seam feathering (v2 of the fusion).** The drive test's central discovery: fused
   floor patches meet kept mesh in 8–21 cm curb-like steps (188 cells on fused-best;
   8 of 11 wedge sites). Feather the patch↔kept-mesh transition over ~3–5 cells in
   `floor_from_trajectory.py --fuse-with`, then re-run the sanity gate and the drive
   test. This is the highest-leverage single change available.
2. **Capture protocol for the next recording: scan first, one continuous recording.**
   Roughly a "scan minute" at the start of the same recording — walk the space slowly
   with the head pitched down, sweeping the floor corridor from both directions,
   before the natural walk. Every observation-based method failed the same way (floor
   seen only at grazing angles or not at all: the splats' 19.94/8.86 m² holes, the
   stereo buffer-ring no-hits); a deliberate scan pass converts the least-observed
   region into the best-observed one at zero pipeline cost. It must be the *same*
   recording so one MPS solve covers everything — splitting recordings would break
   the shared-frame invariant that lets every asset drop in unregistered.
3. **2DGS evaluation note.** The 2DGS model both trains and evaluates RS-off (forced
   by the 16 GB OOM); the 3DGS comparison uses the matched RS-off protocol (worth
   ~0.13 dB). Any future 2DGS-vs-3DGS visual claim should keep using
   `scripts/eval_splat_test.py` for like-for-like numbers, and a ≥24 GB GPU would
   remove the RS-off compromise entirely.
4. **Verify Gen 2 eye-gaze MPS availability.** The imitation-learning side of the
   project wants gaze; whether MPS ships eye-gaze output for Gen 2 recordings of this
   profile has not been verified on this machine. It gates the IL work, not this
   deliverable — but it should be checked before the next recording is planned.
5. **The photogrammetry slot is open** (§3.6) — designed, reserved for the author,
   and pre-wired to enter the scoreboard and the fusion as a far-field partner.
6. **Known residual hazards in the recommended asset:** the real ~15 cm gutter at
   waypoint 70 (route around it or accept it as a known hazard for low-clearance
   platforms); the handful of thin near-field pillars (verify against the splat
   rendering before any targeted removal); and the robot-envelope caveat — Nova
   Carter's ~1–3 cm effective clearance makes it a strict judge, and a
   higher-clearance platform (Jackal/Dingo are on the same asset server) would be a
   useful second data point.

---

## 6. Reproduction appendix

### 6.1 Environments

| Environment | Used for | Notes |
|---|---|---|
| `ego_splats` (conda; Python 3.10.20, torch + CUDA, `projectaria_tools`, OpenCV, Open3D 0.19) | preprocessing, training (3DGS & 2DGS), splat-depth meshing, stereo-mesh fusion, eval harness, trajectory floor, fusion, outlier filter | the workhorse; `pip install importlib_metadata` was needed once for Open3D |
| `3dgrut` (conda) | USDZ export (`ply_to_usd.py`) + frame fix (`pxr` available here) | run from `/home/sun/3dgrut` repo root; first run compiles CUDA extensions |
| `depth_from_stereo` (conda; Python 3.11.16, torch 2.10.0+cu128) | the stereo-depth tool + FoundationStereo | created by the tool's `environment.yml`; checkpoint `23-51-11` (3.3 GB) |
| Isaac Sim 5.1.0 (`~/isaac-sim/python.sh`) | splat smoke test, stage build, physics bench, drive test | always use `--report` JSONs; Kit swallows stdout and force-exits |

Hardware: single RTX 4080 SUPER (16 GB) — GPU jobs serialized throughout.

### 6.2 Script inventory (this repo)

| Script | Purpose |
|---|---|
| `scripts/extract_aria_vrs.py` | VRS + MPS → rectified pinhole frames, per-frame calibration/poses, sparse depth, `transforms_with_sparse_depth.json` |
| `scripts/aria_utils.py` | preprocessing helpers (calibration, undistortion, pose interpolation, readout time) |
| `train_lightning.py` | training driver; `train_model=3dgs\|2dgs` |
| `model/vanilla_gsplat.py` / `model/GS2D_gsplat.py` | the 3DGS / 2DGS models (losses, strategies, rolling shutter, PLY I/O) |
| `scripts/filter_splat_outliers.py` | radius + `--min-opacity` PLY filter (median-centred) |
| `scripts/fix_nurec_usdz_frame.py` | strip 3dgrut's baked frame rotation from the exported USDZ (refuses unknown transforms) |
| `scripts/isaacsim_load_splat.py` | headless Isaac smoke test; asserts NuRec prims resolve + volume transform identity; `--report` |
| `scripts/extract_mesh_tsdf.py` | trained splat → depth renders → TSDF → mesh (`--train-model 2dgs` supported) |
| `scripts/tsdf_fusion.py` | shared TSDF fusion stage (volume, integrate, subsample-by-pose-delta, marching cubes) |
| `scripts/extract_mesh_stereo_tsdf.py` | stereo-tool export dir → masked TSDF → mesh (same shared stage) |
| `scripts/eval_mesh.py` / `scripts/compare_meshes.py` | the scorecard: floor coverage, largest hole, semi-dense agreement, hygiene |
| `scripts/floor_from_trajectory.py` | trajectory floor mesh + heightfield; `--calibrate-with` (fit h), `--fuse-with` (prior fusion) |
| `scripts/compose_near_far_mesh.py` | stereo-near + 2DGS-far composite (AABB crop + far-field junk removal) |
| `scripts/eval_splat_test.py` | matched-protocol (RS-off) held-out splat evaluation |
| `scripts/isaacsim_build_env.py` | composed stage: splat + invisible collision mesh; PLY→USD conversion (cached); `--bench` physics timings |
| `scripts/sim_drive_test.py` | the acceptance test: Nova Carter along the MPS path; fall-throughs / stalls / wedges / completion to JSON |
| `scripts/bash_local/run_gen2_outside.sh`, `train_gen2_outside.sh`, `train_gen2_outside_2dgs.sh`, `export_gen2_outside_usdz.sh` | the as-run wrappers with this scene's paths and parameters |

### 6.3 End-to-end sequence for a new recording

```bash
# 0. Record (see §5.2: scan the floor corridor first, one continuous recording),
#    run MPS, and place:  <rec>.vrs  +  mps_<rec>_vrs/slam/{closed_loop_trajectory.csv,
#    online_calibration.jsonl, semidense_points.csv.gz, semidense_observations.csv.gz}

conda activate ego_splats
cd /home/sun/Desktop/aria_proj/egocentric_splats

# 1. Preprocess (copy run_gen2_outside.sh, point it at the new VRS/MPS;
#    match rectified height to the sensor, scale focal for 90 deg FOV)
bash scripts/bash_local/run_gen2_outside.sh
#    verify: aux images match image size; ~33% observation match rate; readout ms sane

# 2. Train the visual splat (3DGS, MCMC cap; full res)
bash scripts/bash_local/train_gen2_outside.sh        # check for the PLY, not exit code

# 3. Train the geometry splat (2DGS, RS off on 16 GB)
bash scripts/bash_local/train_gen2_outside_2dgs.sh

# 4. Filter + export + frame-fix the visual asset, then smoke-test it
bash scripts/bash_local/export_gen2_outside_usdz.sh
~/isaac-sim/python.sh scripts/isaacsim_load_splat.py <usdz> --report r.json   # volume_frame_is_identity must be true

# 5. Stereo depth (own env) + fusion
<depth_from_stereo>/python export_depth_from_stereo.py --vrs <vrs> --mps <mps_root> \
    --stereo_model <ckpt> --output_dir output/<rec>/stereo_depth/export_stride3 --lr_check --stride 3
python scripts/extract_mesh_stereo_tsdf.py --export-dir output/<rec>/stereo_depth/export_stride3 \
    --depth-trunc 4.0 --output output/<rec>/stereo_depth/mesh_tsdf_stereo_d40.ply

# 6. 2DGS splat-depth mesh (far-field partner)
python scripts/filter_splat_outliers.py <2dgs PLY> --radius 50 --min-opacity 0.3 --output <op03.ply>
python scripts/extract_mesh_tsdf.py --ply <op03.ply> --data-dir <processed rect dir> \
    --train-model 2dgs --output <mesh_tsdf_2dgs_op03.ply>

# 7. Calibrate eye height against recovered geometry, rebuild the floor
python scripts/floor_from_trajectory.py --calibrate-with <best observation mesh>

# 8. Compose + fuse the final candidate, then gate it
python scripts/compose_near_far_mesh.py --near <stereo mesh> --far <2dgs mesh> --output <prefuse.ply>
python scripts/floor_from_trajectory.py --eye-height <h_fit> --fuse-with <prefuse.ply> --output-dir <fused_best_dir>
python scripts/compare_meshes.py <fused> <raw parents...> --eye-height <h_fit> --output comparison.json
#    gate: fused beats each parent on coverage AND largest hole; sd-median within 1 cm

# 9. Build the composed Isaac stage (static trimesh collider) and bench it
~/isaac-sim/python.sh scripts/isaacsim_build_env.py --collision <fused.ply> \
    --approximation none --splat <usdz> --bench --report bench.json

# 10. The acceptance test
~/isaac-sim/python.sh scripts/sim_drive_test.py --collision <fused.ply> --report drive.json
#    accept: 0 fall-throughs, no tip-over, completion with a tolerable rescue census
```

---

### Notes

[^df]: `docs/full_pass.md` credits `data_factor=2` + `pcd_stride=2` with the VRAM fix
and describes the run as 1008×756; the code audit (`DOC_report.md` §1) and the run's
own `cameras.json` / rendered test images (2016×1512, fx = 1008) show both flags are
read by nothing and the run trained at full resolution. Per this report's rules the
more specific artifact wins; `full_pass.md` is stale on this point.

[^psnr]: Stored test metrics are PSNR 25.548 / SSIM 0.8457 / LPIPS 0.3887 (rounded to
25.55/0.846/0.389 in `full_pass.md`); the RS-off matched re-evaluation is
25.415 / 0.8430 / 0.3895. The 2DGS figure 23.689 / 0.8217 / 0.4529 is RS-off by
construction, so the honest visual gap is ~1.73 dB.

[^hole]: The 19.94 m² "largest hole" figure overstates the splat's failure somewhat:
85 % of that hole's cells are buffer-ring cells whose expected floor is the harness's
own nearest-neighbour guess under 1.3 m of grade (`E0_report.md` finding 1,
`E2_3dgs_report.md` failure analysis). The 2DGS variant halving it under identical
scoring shows a real component too.

[^floorproto]: Trajectory-floor numbers depend on the build/score protocol. The floor
*rebuilt at the calibrated h = 1.6683* scores 0.9780 / 0.48 m² under the default
h = 1.60 harness (the four-way tables in the splat-depth and stereo reports) and
0.9866 / 0.37 m² under the h = 1.6683 harness (`comparison_e6a_h16683.json`) — the
latter equals the original h = 1.60-build-and-score numbers, as expected when build
and harness use the same h. Both are quoted where each protocol applies.

[^scoring]: The default-h (1.60) bracketed values are each raw method's original
report scorecard (`mesh_tsdf_op03.ply.scorecard.json`,
`mesh_tsdf_2dgs_op03.ply.scorecard.json`, `mesh_tsdf_stereo_d40.ply.scorecard.json`);
the main columns are the like-for-like h = 1.6683 rescoring from
`comparison_e6a_h16683.json`. sd-median and junk are h-independent and identical in
both. Report tables round (e.g. 0.478 for 0.4782, 0.987 for 0.9866); this table uses
the JSON values at 4 decimals.
