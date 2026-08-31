# E2a — TSDF mesh from rendered 3DGS depth: run report

Date: 2026-08-20. Env: `ego_splats` (`/home/sun/miniforge3/envs/ego_splats/bin/python`).
GPU: RTX 4080 SUPER 16 GB (verified idle before starting: 569 MiB desktop use only).
Nothing committed or pushed.

This is the first half of brief E2 (baseline 3DGS mesh) plus two verified code fixes.
The 2DGS retrain + comparison is the next agent's task (E2b); fix 1 below is what makes
that retrain honor its configured Gaussian budget.

## Deliverables

| File | What |
|---|---|
| `model/GS2D_gsplat.py` (fix) | 2DGS `_create_strategy` now constructs `MCMCStrategy` from `opt.mcmc_strategy.*` |
| `scripts/filter_splat_outliers.py` (extended) | new `--min-opacity` flag for the aggressive depth-render copy |
| `scripts/extract_mesh_tsdf.py` (new) | trained splat -> depth renders -> Open3D ScalableTSDFVolume -> marching-cubes mesh, MPS world frame |
| `output/.../mesh_tsdf/depth_render_op03.ply` | depth-render copy, radius 50 + min-opacity 0.3 (904,972 Gaussians) |
| `output/.../mesh_tsdf/depth_render_op05.ply` | depth-render copy, radius 50 + min-opacity 0.5 (799,176 Gaussians) |
| `output/.../mesh_tsdf/mesh_tsdf_op03.ply` (+`.meta.json`, `.scorecard.json`) | TSDF mesh from the 0.3 copy |
| `output/.../mesh_tsdf/mesh_tsdf_op05.ply` (+`.meta.json`, `.scorecard.json`) | TSDF mesh from the 0.5 copy |

(`output/...` = `output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512`.)
The visual asset (`point_cloud/iteration_30000/point_cloud.ply` and the exported USDZ)
is untouched — the aggressively filtered PLYs are separate derived assets used only
for depth rendering, exactly as the brief requires.

## Fix 1 — 2DGS ignored the MCMC config block (`model/GS2D_gsplat.py`)

Verified stage-1 finding (DOC report §2): `Gaussians2D._create_strategy` constructed
`MCMCStrategy()` with **no arguments**, so a 2DGS run with
`opt.densification_strategy=MCMC` silently used gsplat's defaults —
`cap_max=1,000,000` — and ignored `opt.mcmc_strategy.cap_max` (and every other key)
from the config/command line.

Fix: the MCMC branch now mirrors `VanillaGSplat._create_strategy`
(`model/vanilla_gsplat.py:613`) exactly, passing `cap_max`, `noise_lr`,
`mcmc_refine_start_iter`, `mcmc_refine_stop_iter`, `mcmc_refine_every`,
`mcmc_min_opacity` out of `self.cfg.opt.mcmc_strategy`. Verified the constructor
signature against the installed gsplat 1.5.3 (`MCMCStrategy(cap_max=1500000, ...)`
constructs and reports the right values). The `default`-strategy branch was left
as is (out of scope; it still only sets `key_for_gradient="gradient_2dgs"`, which
E2b should keep in mind if it ever runs 2DGS with the default strategy).

## Fix 2 — `--min-opacity` on `scripts/filter_splat_outliers.py`

Why (brief E2): low-opacity floaters are visually negligible but they still write
into `RGB+ED` expected-depth renders, dragging depth off-surface and punching
spurious holes into a TSDF. The depth-render copy therefore gets filtered much more
aggressively than the visual copy; the visual asset is not touched.

Implementation: the PLY stores the pre-activation logit, so the flag thresholds the
**activated** (sigmoid) opacity; it composes with the existing radius filter and
`0.0` (default) disables it, so all existing callers (including the USDZ export
script) behave identically. The `metadata` PLY element (color format, SH degree) is
preserved as before.

Measured on the trained 1.5 M-Gaussian PLY (radius 50 in both cases):

| Setting | Dropped by radius | Dropped by opacity | Kept |
|---|---|---|---|
| `--min-opacity 0.3` | 18,816 (1.254 %) | 576,212 (38.4 %) | **904,972** |
| `--min-opacity 0.5` | 18,816 (1.254 %) | 682,008 (45.5 %) | **799,176** |

More than a third of all Gaussians sit below 0.3 activated opacity — consistent
with MCMC keeping a large population of near-transparent Gaussians alive.

## New script — `scripts/extract_mesh_tsdf.py`

Design, following the brief and the render_lightning.py loading pattern:

- **Model loading**: hydra `compose` of `conf/config.yaml` with the same overrides a
  render run would use, `initialize_eval_info(cfg)` for the scene/camera setup
  (which also disables the 3D smooth filter and viewer), dispatch on
  `cfg.train_model` (`3dgs` -> `VanillaGSplat`, `2dgs` -> `Gaussians2D`),
  `module.load_ply(cfg.scene.load_ply)`. `--train-model 2dgs` is already wired for
  E2b; both models render depth via gsplat `render_mode="RGB+ED"` (expected
  z-depth), which is exactly what Open3D's pinhole back-projection wants.
- **Poses**: `scene_info.all_cameras` — the 4,675 rectified RGB frames with
  MPS closed-loop center-row poses (train split ∪ held-out 1/8; same walk). Output
  is therefore in the MPS world frame; no registration.
- **Rolling shutter**: `opt.handle_rolling_shutter=false` for this pass — each
  frame renders once from its center-row pose. At walking speed the 10.1 ms RGB
  readout moves the camera ~1.5 cm, below the 8 cm truncation band; rendering the
  full per-row pose array would cost 3–8 renders per frame for nothing the TSDF
  can resolve.
- **Frame subsampling**: keep a frame after >=0.10 m translation or >=10°
  rotation since the last kept frame (defaults; CLI-overridable). On this
  recording: **kept 1,031 / 4,675 frames** (3,644 dropped as redundant — a 30 fps
  walk advances ~2 cm/frame). Logged and recorded in the `.meta.json`.
- **Per-pixel masking before integration**: pixels outside the rectified valid
  mask (`mask.png`); pixels with accumulated alpha < 0.5 (sky / unmodelled
  background — expected depth is meaningless there); depth outside
  [`--depth-min` 0.1, `--depth-trunc` 6.0]; non-finite depth.
- **Fusion**: `o3d.pipelines.integration.ScalableTSDFVolume`,
  `voxel_length=0.02`, `sdf_trunc=0.08` (the brief's starting values), RGB8 color
  (the color render comes free with `RGB+ED`), per-camera `PinholeCameraIntrinsic`
  from `camera.intrinsic_np` and extrinsic `camera.w2c_44_np`. Camera axes are
  X-right/Y-down/Z-forward on both sides, so no convention change.
- **Output**: marching-cubes mesh (`extract_triangle_mesh`) + a `.meta.json` with
  every parameter, frame counts, timings and mesh stats.

### Max fusion depth: 6.0 m, deliberately

`conf/config.yaml` clips rendered depth to `render.depth_min/depth_max` = 0.1/7.0,
so the trained model's depth was only ever consumed inside that band, and the RGB
sparse depth that shaped it has median ~5.1 m. Splat expected-depth error grows
with range (weaker parallax on a local walk, larger Gaussians, foliage), while
TSDF fusion with `sdf_trunc=0.08` assumes depth noise well under ~8 cm — far-field
depth smears the volume rather than adding usable surface. The trajectory covers
the entire evaluated footprint, so every walkable cell is seen from ~2 m by some
camera and a 6 m cutoff costs no floor coverage; what it costs is distant walls /
vegetation that the E0 harness does not score and a collision mesh does not need.
6.0 m also stays inside the 7.0 m the render config itself trusts.

## Commands actually run

```bash
cd /home/sun/Desktop/aria_proj/egocentric_splats
PY=/home/sun/miniforge3/envs/ego_splats/bin/python
BASE=output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512
DATA=/home/sun/aria/processed/Outside_20260812_141244/camera-rgb-rectified-1008-h1512

# depth-render copies (visual asset untouched)
$PY scripts/filter_splat_outliers.py $BASE/point_cloud/iteration_30000/point_cloud.ply \
    --radius 50 --min-opacity 0.3 --output $BASE/mesh_tsdf/depth_render_op03.ply
$PY scripts/filter_splat_outliers.py $BASE/point_cloud/iteration_30000/point_cloud.ply \
    --radius 50 --min-opacity 0.5 --output $BASE/mesh_tsdf/depth_render_op05.ply

# TSDF fusion (defaults: voxel 0.02, trunc 0.08, depth 0.1-6.0, alpha>=0.5,
# subsample 0.10 m / 10 deg)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/extract_mesh_tsdf.py \
    --ply $BASE/mesh_tsdf/depth_render_op03.ply --data-dir $DATA \
    --output $BASE/mesh_tsdf/mesh_tsdf_op03.ply
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/extract_mesh_tsdf.py \
    --ply $BASE/mesh_tsdf/depth_render_op05.ply --data-dir $DATA \
    --output $BASE/mesh_tsdf/mesh_tsdf_op05.ply

# scoring
$PY scripts/eval_mesh.py $BASE/mesh_tsdf/mesh_tsdf_op03.ply
$PY scripts/eval_mesh.py $BASE/mesh_tsdf/mesh_tsdf_op05.ply

# eye-height retro-calibration of the E5 floor
$PY scripts/floor_from_trajectory.py --calibrate-with <best mesh>
```

## Fusion runs

All three runs fused the same **1,031 / 4,675** pose-delta-subsampled frames.
GPU render + CPU integration ran at ~11.3 fps once warm (the 30-frame smoke test
reported 0.45 fps only because first-use CUDA/rasterizer warmup amortized over 30
frames); each full fusion took ~90 s plus ~16 s scene/model load and ~9 s marching
cubes. Peak GPU well under 16 GB; GPU verified idle before and after (~560 MiB
desktop only).

| Run | Source copy | Gaussians | `--alpha-thresh` | valid px (mean) | Triangles |
|---|---|---|---|---|---|
| `mesh_tsdf_op03` | op03 | 904,972 | 0.5 (default) | 66.9 % | 10,714,519 |
| `mesh_tsdf_op05` | op05 | 799,176 | 0.5 (default) | 65.5 % | 10,550,530 |
| `mesh_tsdf_op03_a03` (bonus) | op03 | 904,972 | 0.3 | 67.3 % | 10,946,457 |

## Scorecards (`scripts/eval_mesh.py`, default params: h=1.60, ±15 cm, 0.75 m buffer)

Full comparison (also in `mesh_tsdf/comparison_e2a.json`; the E5 floor row is the
**retro-calibrated** floor rebuilt at h=1.6683, see below):

| Mesh | Coverage | Largest hole m² | Missing m² | sd-mean m | sd-median m | sd-p95 m | <10 cm | Triangles | Components | Non-manifold | Junk m² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **mesh_tsdf_op03 (kept)** | **0.5013** | 19.94 | 35.63 | **0.0655** | **0.0319** | 0.2515 | **0.798** | 10.71 M | 767,859 | 13 | 341.0 |
| mesh_tsdf_op03_a03 | 0.5091 | 19.94 | 35.07 | 0.0655 | 0.0319 | 0.2525 | 0.797 | 10.95 M | 792,123 | 19 | 352.0 |
| mesh_tsdf_op05 | 0.4612 | 22.41 | 38.49 | 0.0683 | 0.0334 | 0.2551 | 0.782 | 10.55 M | 771,313 | 4 | 340.4 |
| E5 floor (recalibrated) | 0.9780 | 0.48 | 1.57 | 0.6162 | 0.408 | 1.8592 | 0.294 | 14,288 | 1 | 0 | 0.0 |

Full scorecard of the kept mesh (`mesh_tsdf_op03.ply.scorecard.json`):

```
floor:     n_cells 7144 (71.44 m²), n_cells_hit 6306, n_cells_hit_within_tol 3581,
           coverage 0.5013, n_holes 69, largest_hole 19.94 m², total_missing 35.63 m²,
           hit_abs_error_mean 0.0685 m
semidense: 938,113 points used (of 4,292,410), mean 0.0655 m, median 0.0319 m,
           p95 0.2515 m, max 3.231 m, within 5/10/25 cm = 0.634 / 0.798 / 0.949
hygiene:   10,714,519 tris, 8,395,333 vertices, 767,859 components,
           13 non-manifold edges, small(<0.1 m²) component area 341.04 m²,
           bbox [-12.86, -11.87, -5.61] .. [12.29, 13.01, 7.69]
```

**Filter-setting verdict:** min-opacity **0.3 wins** over 0.5 on every floor metric
(0.5013 vs 0.4612 coverage, 19.94 vs 22.41 m² largest hole) and slightly on
semi-dense agreement, at identical junk (341 vs 340 m²) — cutting 45 % of the
Gaussians starts eroding real surface while the junk evidently is not made of
low-opacity floaters. Per the brief, the worse mesh's PLY (`mesh_tsdf_op05.ply`)
was deleted; its scorecard, metadata and log are kept. The bonus alpha-0.3 variant
gains +0.8 pp coverage for +11 m² junk and does not move the largest hole; it is
kept on disk as a sensitivity check but **`mesh_tsdf_op03.ply` (brief-default
parameters) is the designated E2a baseline** and the calibration mesh.

## Failure-mode analysis (why coverage is 0.50 when geometry is 3 cm accurate)

Decomposition of the 7,144 footprint cells for op03 (downward rays, same
parameters as the harness):

| | cells | covered | no-hit | junk hit >15 cm above | through-hit >15 cm below |
|---|---|---|---|---|---|
| on-path (trajectory-occupied) | 1,028 | 0.560 | 8.3 % | 93 | 274 |
| buffer ring (0.75 m dilation) | 6,116 | 0.491 | 12.3 % | 1,009 | 1,349 |

- **Through-hits below dominate** (1,623 cells; 975 of them land >0.5 m below the
  expectation): the ray passes through a floor gap and hits smeared/junk surface
  beneath. dz percentiles over all hits: p25 −0.157, median −0.051, p75 +0.077,
  p95 +0.389 m.
- **The 19.94 m² largest hole** sits in the central area (x −3.4..3.3,
  y −3.5..5.3). Its expected floor spans **−2.43..−1.15 m — 1.3 m of grade inside
  one hole** — and 85 % of its cells are buffer cells; composition 1,323
  through-below / 447 no-hit / 224 junk-above, median dz −0.67 m. Much of this is
  the harness expectation itself being a guess in buffered/steep-grade cells
  (E0 finding 1; E5's interpolated cells failed the same way), on top of genuine
  splat-floor gaps on weakly textured ground.
- **Not an eye-height offset:** rescoring the same rays with the calibrated
  h = 1.6683 m *reduces* covered cells (3,405 vs 3,581) — the miss pattern is
  scatter, not a systematic shift.
- Junk (341 m² in 768 k tiny components) is splat fuzz around vegetation and
  noisy far-field surface, not low-opacity floaters (raising min-opacity to 0.5
  left it unchanged). E6 will need a component-size cleanup pass before collision
  use regardless of which mesh wins.

The headline read: **splat TSDF geometry is locally accurate (semi-dense median
3.2 cm, 80 % of 938 k points within 10 cm — 10× better than the E5 floor's 0.41 m
median) but only covers what the camera actually looked at, while the E5
trajectory floor is complete but geometrically coarse.** They are exact
complements; E5's `--fuse-with` mode (pin height, fill holes, reject
disagreements) is the obvious marriage and already exists — recommended as the
E6 candidate alongside the raw meshes.

## Eye-height calibration (task 6 — retro-calibrates E5)

```bash
$PY scripts/floor_from_trajectory.py --calibrate-with $BASE/mesh_tsdf/mesh_tsdf_op03.ply
```

Result (856/1,035 walked cells usable; acceptance band h ∈ [1.0, 2.5] m):

| | |
|---|---|
| **h fitted (median)** | **1.6683 m** |
| mean | 1.7145 m |
| std | 0.2754 m |
| IQR | 0.158 m |
| p5–p95 | 1.2589 – 2.3021 m |

The fitted median is only 6.8 cm above the assumed 1.60 m, but the spread is
large — 6.7× the std E5 measured on its synthetic validation mesh (0.041 m) — so
per the E5 report's own criterion the constant-offset prior is **weaker than it
looked**: the tails come from junk above the floor (h ≈ 1.26 at p5) and
through-holes below it (h ≈ 2.30 at p95), i.e. mesh defects as much as gait.
The median is robust to both, so 1.6683 is still the best available value.

Side effect (intended): the run rebuilt the E5 floor deliverables in
`output/Outside_20260812_141244/floor_from_trajectory/` (`floor_mesh.ply`,
`floor_heightfield.npz`, `.json`) at h = 1.6683; floor Z now spans
−2.519..−0.987 m. Under the default-h harness the recalibrated floor scores
coverage 0.9780 (was 0.9866 at h = 1.60) — the 6.8 cm shift stays well inside
the ±15 cm tolerance.

## Problems hit and fixed

- **Smoke-test throughput was misleading** (0.45 fps over 30 frames): CUDA context
  + rasterizer warmup dominated. Full runs sustain ~11.3 fps; no subsampling
  beyond the pose-delta filter was needed, and memory stayed comfortable
  (ScalableTSDF + 10.7 M-triangle marching cubes on 31 GB RAM).
- **A metadata field initially had a broken placeholder expression**
  (`valid_px_mean_fraction`); fixed to a proper tracked ratio before the real runs.
- A first look at the per-run `valid_px_mean_fraction` values seemed
  non-monotonic across variants; re-reading the labeled metadata showed they are
  exactly monotonic as expected (op05 65.5 % < op03 66.9 % < a03 67.3 %) — the
  confusion was a mislabeled grep, not a bug.

## Status

Complete. Both fixes verified (`MCMCStrategy` kwargs checked against the
installed gsplat 1.5.3; filter flag exercised on the real PLY).
`scripts/extract_mesh_tsdf.py` ran end-to-end three times; two min-opacity
settings compared and the better mesh kept (`mesh_tsdf_op03.ply`); full
scorecards above; eye height calibrated (h = 1.6683 m, std 0.275) and the E5
floor rebuilt with it. Nothing committed or pushed. Ready for E2b (2DGS retrain
— which will now actually honor `opt.mcmc_strategy.cap_max` thanks to fix 1,
and can reuse this script via `--train-model 2dgs`).
