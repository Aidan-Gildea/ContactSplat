# E2b — 2DGS retrain + TSDF mesh: run report

Date: 2026-08-20 (training) / 2026-08-26 (mesh + scoring; session resumed). Env: `ego_splats` (`/home/sun/miniforge3/envs/ego_splats/bin/python`).
GPU: RTX 4080 SUPER 16 GB (verified: only ~560 MiB desktop use before starting).
Nothing committed or pushed.

Second half of brief E2: retrain the scene as 2DGS (flat surfel Gaussians),
extract a TSDF mesh with the same pipeline/parameters as the winning 3DGS
variant (E2a `mesh_tsdf_op03`), score both plus the E5 floor, and give a
verdict on whether 2DGS is worth the retrain cost for collision geometry.

## Prerequisite verification (step 1)

All three prerequisites from E2a were on disk and correct before any GPU work:

- `model/GS2D_gsplat.py` `_create_strategy` MCMC branch passes
  `opt.mcmc_strategy.*` kwargs (cap_max, noise_lr, refine iters, min_opacity)
  — mirrors `model/vanilla_gsplat.py:613`.
- `scripts/extract_mesh_tsdf.py` exists, `--train-model 2dgs` already wired,
  and produced the scored E2a baseline
  (`output/.../mesh_tsdf/mesh_tsdf_op03.ply`, coverage 0.5013).
- E5 floor mesh at
  `output/Outside_20260812_141244/floor_from_trajectory/floor_mesh.ply`
  (rebuilt at the calibrated h=1.6683).

Also verified compatible before training: every `cfg.opt.*` key the 2DGS
class reads exists in `conf/opt/simple_gsplat_30K.yaml` (normal_loss 7k,
dist_loss 3k, opacity_reg/scale_reg 0.0, absgrad, ...); `train_lightning.py:33`
dispatches `train_model=2dgs` to `Gaussians2D`; the 2DGS pinhole assertion
holds (rectified RGB is pinhole); `Gaussians2D` inherits `save_ply`/`load_ply`
from `VanillaGSplat`, whose PLY reader is scale-count-agnostic, and
`scripts/filter_splat_outliers.py` masks vertex rows without assuming a
field layout — so the 2DGS PLY round-trips through the whole tool chain.

### Pre-launch fix: the masked except-path (known risk from DOC §4)

`model/GS2D_gsplat.py` `training_step` wrapped the render in
`try/except:` returning undefined `loss`/`image` — any real rendering error
would have surfaced as a NameError with no traceback of the true cause.
Fixed before launch to log the camera id and re-raise. This paid off on the
first crash (below), which produced a clean traceback.

## Training script (step 2)

`scripts/bash_local/train_gen2_outside_2dgs.sh` — copy of
`train_gen2_outside.sh` with:

- `train_model=2dgs`
- `EXP_NAME="$SCENE/${RECTIFIED}-2dgs"` (separate output dir; 3DGS untouched)
- `opt.densification_strategy=MCMC opt.mcmc_strategy.cap_max=1500000`
  (budget-matched to the 3DGS run; honored thanks to the E2a strategy fix —
  verified in the run's `cfg_args`)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `scene.data_factor` / `scene.pcd_stride` dropped (verified inert, stage-1)
- `opt.handle_rolling_shutter=false` — NOT the brief's setting; forced by a
  measured OOM, see next section.

## Attempt 1 (rolling shutter on): OOM at first validation — root cause

Launched with `opt.handle_rolling_shutter=true` per the brief. Training
itself ran fine: epoch 0 completed 4090/4090 steps in 24:51 (2.70 it/s),
steady ~6.5 GiB VRAM. At the FIRST validation epoch it crashed at
validation frame ~10/585:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.98 GiB
  ... gsplat/rendering.py:1486 rasterization_2dgs -> isect_tiles
process has 13.20 GiB in use; 12.59 GiB allocated by PyTorch
```

preceded by repeated
`expandable_segments: memory mapping failed with OOM ... (free: 88 MB)`.

Root cause (code, not luck): `VanillaGSplat._render_motion_array`
(`model/vanilla_gsplat.py:773-788`) returns
`camera.is_moving_camera and cfg.opt.handle_rolling_shutter` for
**every non-training render** — the `handle_rolling_shutter_start_iter=10000`
deferral applies only to the training branch. So with RS on, every
validation/test frame renders the full rolling-shutter motion array:
`Gaussians2D.render` stacks the 4–8 per-row-bracket poses (this recording:
"rs step" 4–8 at 10.1 ms readout) into ONE batched `rasterization_2dgs`
call — 4–8 full 2016×1512 cameras at once, each producing color+depth+
alpha+normals+distortion buffers plus tile-intersection arrays. The
validation frames 1–9 survived at the allocator's edge; frame 10's
1.98 GiB `isect_tiles` allocation did not.

Why 3DGS survived the identical eval path: gsplat's 3DGS `rasterization`
saves far less per camera (no surfel ray-intersection state, no
normal/distortion render targets). The 2DGS rasterizer is what tips it over
16 GB — forward-only. Training would have hit the same wall harder at
iteration 10,000, where the motion-array render runs **with gradients**.

Fix decision (minimal, honest): retrain with
`opt.handle_rolling_shutter=false`, and remove the resulting evaluation
asymmetry by re-evaluating BOTH models on the test split with RS off
(`scripts/eval_splat_test.py`, new — reuses the repo's own `test_step`,
`ImageLoss`, masking and exposure so numbers are directly comparable).
Alternatives rejected: chunking the motion-array batch (bounds the
transient but not the saved-for-backward footprint at iter 10k+, so the
run would still die hours in); capping RS samples (protocol deviation
anyway, memory still uncertain). Geometry impact is negligible for the
mesh: at walking speed the camera moves ~1.5 cm during the 10.1 ms
readout, far below the 8 cm TSDF truncation band, and
`extract_mesh_tsdf.py` renders with RS off for both models regardless
(E2a's protocol).

Cost of attempt 1: ~30 min GPU. Evidence preserved in
`output/.../camera-rgb-rectified-1008-h1512-2dgs/train_2dgs_attempt1_rs_oom.log`.

## Attempt 2 (rolling shutter off): COMPLETED

Launched 2026-08-20 17:19, PLY written 21:06 — **~3 h 47 min wall** for 30,000
iterations + validation epochs + the final 585-frame test pass (12:43 at
0.84 it/s). Training sustained ~2.4–2.7 it/s (last epoch: 1,370/1,370 steps at
2.38 it/s; 30,000 = 7 x 4,090 + 1,370 across 8 Lightning epochs). VRAM stayed
steady around ~6.5 GiB — the RS-off eval path renders one pose per frame, so
the attempt-1 wall never appeared. For scale: the 3DGS run took a similar
30k-iteration wall but each 2DGS step carries extra surfel state (normals,
distortion maps, depth-to-normal consistency) plus the normal (7k+) and
distortion (3k+) losses.

Verified after completion:

- `point_cloud/iteration_30000/point_cloud.ply`: **exactly 1,500,000
  Gaussians** — `cap_max=1500000` was honored (the E2a `_create_strategy` fix
  doing its job; before it this would have been 1.0 M). 372 MB, `color_format
  rgb`, SH degree 3, standard 65-property layout — round-trips through
  `filter_splat_outliers.py` and the PLY loader unchanged.
- `cfg_args` records `train_model: 2dgs`, `handle_rolling_shutter: False`,
  `mcmc_strategy.cap_max: 1500000`, normal_loss (7k+, lambda 0.05) and
  dist_loss (3k+, lambda 0.01) enabled — the 2DGS-specific regularizers ran.
- Held-out test (585 frames, every 8th; RS-off model, RS-off eval):
  **PSNR 23.689 / SSIM 0.8217 / LPIPS 0.4529**.

For reference, the 3DGS run's stored test metrics are PSNR 25.548 /
SSIM 0.8457 / LPIPS 0.3887 — but that evaluation rendered WITH the
rolling-shutter motion array, so it is not like-for-like; the matched RS-off
re-evaluation is in step 5 below.

## Mesh extraction (step 4)

Same pipeline and parameters as the E2a winner (`mesh_tsdf_op03`), so the
comparison is clean. Commands actually run (repo root, `ego_splats`):

```bash
PY=/home/sun/miniforge3/envs/ego_splats/bin/python
B2=output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512-2dgs
DATA=/home/sun/aria/processed/Outside_20260812_141244/camera-rgb-rectified-1008-h1512

# depth-render copy (visual asset untouched), E2a's winning filter settings
$PY scripts/filter_splat_outliers.py $B2/point_cloud/iteration_30000/point_cloud.ply \
    --radius 50 --min-opacity 0.3 --output $B2/mesh_tsdf/depth_render_op03.ply

# TSDF fusion, all defaults = E2a's parameters (voxel 0.02, trunc 0.08,
# depth 0.1-6.0 m, alpha>=0.5, subsample 0.10 m / 10 deg)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY scripts/extract_mesh_tsdf.py \
    --ply $B2/mesh_tsdf/depth_render_op03.ply --data-dir $DATA \
    --train-model 2dgs --output $B2/mesh_tsdf/mesh_tsdf_2dgs_op03.ply
```

Filtering (vs the 3DGS numbers in brackets): radius 50 dropped 6,274 = 0.418 %
[18,816 = 1.254 %] — MCMC drifted far fewer floaters out of a surfel model —
and min-opacity 0.3 dropped another 609,340 = 40.6 % [576,212 = 38.4 %],
keeping **884,386** Gaussians [904,972]. Near-identical Gaussian budgets into
the fusion, as intended. Raw-PLY AABB before filtering was again ~180 km of
low-opacity MCMC drift; 78 x 71 x 44 m after.

Fusion (`--train-model 2dgs` load path worked unmodified — `Gaussians2D`
inherits `load_ply`, and its `render()` returns the same `depth`/`alphas`/
`render` keys the script consumes): same **1,031 / 4,675** pose-delta-kept
frames as E2a, mean valid-pixel fraction **70.5 %** [66.9 %], fusion 222.6 s at
**4.63 fps** [~91 s at 11.3 fps — the 2DGS rasterizer with its extra normal/
distortion buffers renders ~2.4x slower], marching cubes 4.6 s. Result:
**6,581,948 triangles** [10,714,519], AABB [-12.81, -11.91, **-2.89**] ..
[12.43, 12.97, 7.49] [3DGS z: **-5.61** .. 7.69] — the 2DGS volume simply
contains far less smeared junk, 2.7 m less of it hanging below the floor.

## Scoring (step 5)

`scripts/eval_mesh.py` scorecard written next to the mesh;
`scripts/compare_meshes.py` output (default harness params h=1.60, +/-15 cm,
0.75 m buffer) over the three briefed meshes, verbatim
(`$B2/mesh_tsdf/comparison_e2b.json`):

```
                        mesh    cover   hole m2   sd-mean   sd-med   sd-p95   <10cm       tris   comps  nonmanif  junk m2
-------------------------------------------------------------------------------------------------------------------------
          mesh_tsdf_op03.ply    0.501     19.94     0.066    0.032    0.252    0.80   10714519  767859        13  341.038
     mesh_tsdf_2dgs_op03.ply    0.640      8.86     0.084    0.035    0.359    0.76    6581948  213708         5   86.039
              floor_mesh.ply    0.978      0.48     0.616    0.408    1.859    0.29      14288       1         0    0.000
```

(`floor_mesh.ply` is the E5 trajectory floor rebuilt at the calibrated
h=1.6683.) Full 2DGS scorecard: n_cells_hit 6,670/7,144, within-tol 4,576
(coverage **0.6405**), 44 holes, largest **8.86 m²**, total missing 25.68 m²,
hit |error| mean 6.29 cm; semi-dense mean/median/p95 = 8.40/**3.47**/35.9 cm,
within 5/10/25 cm = 60.3 %/76.3 %/90.6 %; hygiene 6.58 M tris, 213,708
components, 5 non-manifold edges, junk (<0.1 m² components) **86.0 m²**, total
area 862.9 m².

Failure-mode decomposition, same method as E2a (downward rays, cells split
into on-path = trajectory-occupied vs buffer ring; 3DGS numbers in brackets):

| | cells | covered | no-hit | junk-above | through-below |
|---|---|---|---|---|---|
| on-path | 1,028 | **0.716** [0.560] | **1.1 %** [8.3 %] | 79 [93] | 202 [274] |
| buffer | 6,116 | **0.628** [0.491] | 7.6 % [12.3 %] | 1,077 [1,009] | 736 [1,349] |

- **Through-hits collapse**: 938 total [1,623], and only **150** of them are
  >0.5 m deep [975]. dz percentiles over all hits p25/p50/p75/p95 =
  -0.097/-0.029/+0.071/+0.352 m [3DGS p95 +0.389, with a heavy deep-negative
  tail]. The surfel depth is crisp enough that the TSDF closes the floor
  instead of leaving leak-through onto smeared under-floor junk.
- **On-path no-hit nearly vanishes** (1.1 % vs 8.3 %): where the camera
  actually looked, 2DGS reconstructs the walked floor almost completely.
- **The largest hole (8.86 m²) is the same central steep-grade region** as
  3DGS's 19.94 m² hole (x -2.5..0.6, y -1.0..3.2; expected floor spans
  -2.43..-1.46 m — ~1 m of grade inside one hole), 77 % buffer cells,
  composition 862 through-below / 18 no-hit / 6 junk-above, median dz among
  hits -0.35 m. As in E2a, part of this is the harness's buffered-cell
  expectation being a guess under grade (E0 finding 1), on top of real splat
  floor gaps — but 2DGS shrank it by 56 % under identical scoring.
- **Semi-dense agreement is the one metric 2DGS loses**, and it is an extent
  artifact, not an accuracy one: median distance is statistically the same
  (3.47 vs 3.19 cm) but mean/p95 are worse (8.4/35.9 vs 6.6/25.2 cm) because
  the 2DGS mesh has ~31 % less total surface (863 vs 1,250 m²) — it models
  less far-field wall/foliage for the (wall-and-foliage-dominated, E0) filtered
  cloud to be near. The missing area is exactly the smear the collision use
  case does not want; 341 - 86 = 255 m² of the 3DGS mesh's "coverage" of those
  points is floater junk.

## Splat visual quality (matched protocol)

The 2DGS run's stored test metrics are RS-off; the 3DGS run's stored metrics
were evaluated WITH the rolling-shutter motion array. For a like-for-like
comparison the 3DGS PLY was re-evaluated with `scripts/eval_splat_test.py`
(same `test_step`/`ImageLoss`/masking/exposure path, RS off, same 585-frame
split; results in `output/.../camera-rgb-rectified-1008-h1512/test_rs_off/`):

| Model | Protocol | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| 3DGS | RS-on eval (stored) | 25.548 | 0.8457 | 0.3887 |
| **3DGS** | **RS-off eval (matched)** | **25.415** | **0.8430** | **0.3895** |
| **2DGS** | **RS-off train + eval** | **23.689** | **0.8217** | **0.4529** |

Rolling-shutter handling at eval is worth only ~0.13 dB to 3DGS, so the
protocol asymmetry was minor — the honest gap is **~1.73 dB PSNR / +0.021
SSIM / -0.063 LPIPS in favor of 3DGS**. The surfel model trades appearance for
geometry exactly as expected on this foliage-heavy outdoor scene (thin
structures and view-dependent vegetation are where flat disks hurt most).
As the visualization asset, the 3DGS splat remains the better USDZ source.

## Verdict (step 6)

**For collision geometry: yes — 2DGS wins decisively, and the retrain cost is
moderate (~3 h 47 min on this GPU, one attempt lost to the RS eval OOM).**
Under identical filtering, fusion parameters, and scoring:

- floor coverage **0.640 vs 0.501** (+14 pp; on-path 0.716 vs 0.560),
- largest hole **8.86 vs 19.94 m²** (-56 %), total missing 25.7 vs 35.6 m²,
- deep (>0.5 m) through-hole cells **150 vs 975** (-85 %) — the failure mode
  a robot actually falls into,
- hygiene: **86 vs 341 m² of floater junk**, 214 k vs 768 k components,
  6.6 M vs 10.7 M triangles, and 2.7 m less under-floor smear in the AABB —
  materially less cleanup for E6's collision-approximation step,
- local accuracy preserved: semi-dense median 3.5 vs 3.2 cm (the worse
  mean/p95 is a mesh-extent artifact, above).

The costs: ~1.7 dB PSNR on the splat as a visual asset, no rolling-shutter
compensation trainable within 16 GB (2DGS eval renders OOM with the motion
array), and ~2.4x slower depth rendering at fusion time (222.6 s vs ~91 s —
irrelevant offline).

**Recommendation for E6:** keep both trained models and split the roles —
**3DGS splat for the USDZ visualization asset** (it is already exported),
**2DGS TSDF mesh as the observation-based collision candidate**. The 2DGS
mesh is still only 64 % floor-complete with an 8.86 m² worst hole, so the E2a
conclusion stands, strengthened: fuse it with the E5 trajectory floor
(complete, coarse, coverage 0.978) via `floor_from_trajectory.py --fuse-with`
— the two remain exact complements, and 2DGS narrows the gap the floor prior
has to fill by half. A component-size cleanup pass before collision use is
still required (86 m² of junk), but is a quarter of the 3DGS problem.

## Artifacts

| Path | What |
|---|---|
| `output/.../camera-rgb-rectified-1008-h1512-2dgs/point_cloud/iteration_30000/point_cloud.ply` | trained 2DGS splat, 1.5 M Gaussians (visual/derivable asset, untouched) |
| `.../-2dgs/mesh_tsdf/depth_render_op03.ply` | aggressive depth-render copy (radius 50, min-opacity 0.3; 884,386 G) |
| `.../-2dgs/mesh_tsdf/mesh_tsdf_2dgs_op03.ply` (+`.meta.json`, `.scorecard.json`, `.log`) | the E2b mesh, 6.58 M tris, MPS world frame |
| `.../-2dgs/mesh_tsdf/comparison_e2b.json` | three-way scorecards (3DGS / 2DGS / E5 floor) |
| `.../camera-rgb-rectified-1008-h1512/test_rs_off/` | matched RS-off 3DGS test eval (`summary.json`, per-frame `test_logs.json`) |
| `.../-2dgs/train_2dgs_attempt1_rs_oom.log`, `.../-2dgs/train_2dgs_detached.log` | attempt-1 OOM evidence, attempt-2 full training log |
| `scripts/eval_splat_test.py` | matched-protocol splat test evaluator (new, E2b) |
| `scripts/bash_local/train_gen2_outside_2dgs.sh` | the 2DGS training script (step 2) |

## Status

Complete. 2DGS trained 30 k iterations at cap_max=1.5 M (honored — exactly
1,500,000 Gaussians on disk), meshed with the E2a pipeline at identical
parameters, scored, decomposed, and compared against the 3DGS baseline and the
E5 floor; matched RS-off visual metrics for both splats. GPU verified idle
before and after each job. Nothing committed or pushed.
