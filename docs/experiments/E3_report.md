# E3 — Gen 2 stereo depth -> TSDF mesh: run report

Date: 2026-08-26. GPU: RTX 4080 SUPER 16 GB (verified before starting: 647 MiB
desktop use only, no compute jobs). Nothing committed or pushed.

Task: set up Meta's first-party Aria Gen 2 stereo depth tool
(`projectaria_gen2_depth_from_stereo`), run it on the Outside recording with
MPS closed-loop poses, sanity-check the depth against the MPS semi-dense
points BEFORE fusing, fuse the LR-consistency-masked depth maps into a TSDF
with the shared E2 fusion stage, quantify the stereo error envelope, and score
against the E2 meshes and the E5 trajectory floor.

## Deliverables

| Path | What |
|---|---|
| `/home/sun/projectaria_gen2_depth_from_stereo/` | tool clone + FoundationStereo submodule + checkpoints (outside this repo) |
| conda env `depth_from_stereo` | new env for the tool (Python 3.11.16, torch 2.10.0+cu128); `ego_splats` / `3dgrut` / `aria` untouched |
| `scripts/tsdf_fusion.py` (new) | shared TSDF fusion stage factored out of `extract_mesh_tsdf.py` |
| `scripts/extract_mesh_tsdf.py` (refactored) | now delegates volume/integrate/mesh/subsample to `tsdf_fusion.py`; CLI unchanged |
| `scripts/extract_mesh_stereo_tsdf.py` (new) | fuses the tool's depth+mask output via the shared fusion stage |
| `output/Outside_20260812_141244/stereo_depth/export_stride3/` | rectified images, uint16 mm depth, LR masks, `pinhole_camera_parameters.json` |
| `output/Outside_20260812_141244/stereo_depth/sanity_stereo_vs_semidense.json` | pre-fusion sanity check numbers |
| `output/Outside_20260812_141244/stereo_depth/mesh_tsdf_stereo*.ply` (+meta/scorecards) | the E3 meshes |
| `output/Outside_20260812_141244/stereo_depth/comparison_e3.json` | four-way comparison scorecards |

## 1. Tool setup

```bash
cd /home/sun
git clone https://github.com/facebookresearch/projectaria_gen2_depth_from_stereo
cd projectaria_gen2_depth_from_stereo
git submodule update --init          # FoundationStereo @ 6e88068
/home/sun/miniforge3/bin/conda env create -f environment.yml   # -> depth_from_stereo
```

Env verified: Python 3.11.16, torch 2.10.0+cu128 (`cuda_available True`,
CUDA 12.8), projectaria_tools 2.1.0, xformers 0.0.35. Per the machine's
standing rule, no existing env was touched. (Note: the E0 `~/.local`
user-site hazard does not apply here — this env is Python 3.11, and the
polluted user site is `~/.local/lib/python3.10`.)

Checkpoint: the official FoundationStereo Google Drive folder
(`23-51-11`, ViT-Large, `model_best_bp2.pth` 3.30 GB + `cfg.yaml`) downloaded
to `FoundationStereo/ckpts/pretrained_models/23-51-11/`. The system `gdown`
in `~/.local/bin` is broken (system Python missing `tqdm`), so it was run
through the `ego_splats` interpreter — package from user site, deps from the
env, nothing installed anywhere:

```bash
cd FoundationStereo/ckpts
/home/sun/miniforge3/envs/ego_splats/bin/python -m gdown --folder \
    https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf
```

(The folder also carries the small `11-33-40` model and an ONNX export,
~4 GB total; only `23-51-11` was used.) First model load also pulls DINOv2
via torch hub, as the tool's README warns.

## 2. Export run

5-frame smoke test first (throughput + output format), then the real run:

```bash
conda activate depth_from_stereo   # actually: the env's python called directly
cd /home/sun/projectaria_gen2_depth_from_stereo
python export_depth_from_stereo.py \
  --vrs /home/sun/aria/Outside_20260812_141244.vrs \
  --mps /home/sun/aria/mps_Outside_20260812_141244_vrs \
  --stereo_model ./FoundationStereo/ckpts/pretrained_models/23-51-11/model_best_bp2.pth \
  --output_dir <repo>/output/Outside_20260812_141244/stereo_depth/export_stride3 \
  --lr_check --stride 3
```

Notes on the choices:

- `--mps` takes the MPS **root** (the dir containing `slam/`), not `slam/`
  itself. The tool reads the closed-loop trajectory (155,800 poses confirmed
  in its log) and **online calibration** through `MpsDataProvider` — so
  `T_world_camera` is MPS closed-loop and the intrinsics/extrinsics are the
  online (not factory) calibration, exactly as required. No
  factory-calibration flag exists in this tool; factory would only be used
  if MPS were absent.
- `--lr_check` runs FoundationStereo twice per frame (left-right and
  flipped right-left) and writes `masks/mask_*.png` (255 = consistent).
  `--zero_inconsistent_depth` was NOT used — raw depth + separate masks is
  strictly more information; masking is applied at fusion time.
- `--stride 3` = 10 Hz. Fusion subsamples by pose delta (0.10 m / 10 deg,
  the E2 protocol) anyway, and at this walk's ~0.64 m/s a 10 Hz stream
  advances ~6.4 cm per frame, so stride 3 still oversamples the pose-delta
  grid while cutting GPU time 3x.
- Throughput: 1.5 fps with LR check (0.67 s/frame, i.e. ~0.33 s per
  FoundationStereo pass at 512x512, valid_iters=32), ~3.3 GB VRAM.

Output format (verified on the smoke run): 512x512 rectified left grayscale
PNGs; uint16 millimetre depth PNGs; LR masks; per-frame JSON with
`Linear:fu,fv,u0,v0` intrinsics, `T_world_camera` and `T_device_rectCam`
(XYZW quaternions). Rectified focal fx = fy = **305.867 px** — exactly the
online fisheye focal (244.71 px) x the tool's hardcoded `focal_scale=1.25`.
Principal point (253.14, 257.81) = the fisheye's own PP carried over.

## 3. Stereo error model (task 5)

Baseline read from `online_calibration.jsonl` (`slam-front-left` /
`slam-front-right` `T_Device_Camera` translations, all 1,557 records):

| | |
|---|---|
| **B** (front stereo baseline) | **134.961 mm** mean, std 0.111 mm, range 134.792–135.163 mm |
| fisheye focal (front-left) | 244.710 px (std 0.013) |
| **f** (rectified, from tool output) | **305.867 px** (= 244.71 x 1.25) |
| **f·B** | **41.28 m·px** |

sigma_Z = Z² · sigma_d / (f·B):

| Z | sigma_Z @ sigma_d=0.5 px | sigma_Z @ sigma_d=1.0 px |
|---|---|---|
| 1 m | 1.2 cm | 2.4 cm |
| 2 m | 4.8 cm | 9.7 cm |
| 5 m | 30.3 cm | 60.6 cm |
| 10 m | 1.21 m | 2.42 m |
| 20 m | 4.85 m | 9.69 m |

Crossings against the TSDF parameters (voxel 0.02 m, trunc 0.08 m — E2's):

| | sigma_d = 0.5 px | sigma_d = 1.0 px |
|---|---|---|
| sigma_Z exceeds the 2 cm voxel beyond | **1.28 m** | 0.91 m |
| sigma_Z exceeds the 8 cm truncation beyond | **2.57 m** | 1.82 m |

So on this 512px / 13.5 cm-baseline rig, stereo depth is voxel-accurate only
inside ~1.3 m and truncation-band-accurate inside ~2.6 m (at an optimistic
0.5 px disparity error). For the suspected 15 m+ outdoor range: sigma_Z is
**2.7 m at 15 m** (0.5 px) — the route is not merely degraded there, it is
unusable. The empirical check below confirms the model almost exactly.

## 4. Pre-fusion sanity check (task 3)

Six frames spread across the whole walk were exported with `--stride 900`
(five valid — one fell outside the MPS trajectory window), and the filtered
MPS semi-dense points (938,113 of 4,292,410; same
`filter_points_from_confidence(0.005, 0.01)` loader as the E0 harness) were
projected into each frame using the tool's own `T_world_camera` and
intrinsics. A per-pixel z-buffer keeps the nearest point per pixel;
comparison only at pixels that are LR-consistent with valid depth. Script:
scratchpad `sanity_stereo_vs_semidense.py`; numbers:
`output/Outside_20260812_141244/stereo_depth/sanity_stereo_vs_semidense.json`.

Pooled over 5 frames / 329,935 point-pixel pairs:

| | |
|---|---|
| median signed dz (stereo − semidense) | **−0.0005 m** |
| median \|dz\| | 0.150 m |
| frac \|dz\| < 10 cm | 0.430 |

By semidense range (pooled):

| z range | n | median \|dz\| | median dz (bias) | <10 cm |
|---|---|---|---|---|
| 0–1.5 m | 43,657 | **0.033 m** | +0.011 | 0.775 |
| 1.5–2.5 m | 104,752 | **0.052 m** | +0.001 | 0.655 |
| 2.5–4 m | 83,232 | 0.312 m | −0.014 | 0.300 |
| 4–6 m | 67,637 | 0.371 m | −0.131 | 0.133 |
| 6–10 m | 29,227 | 0.307 m | +0.039 | 0.183 |
| 10–30 m | 1,430 | 0.514 m | −0.235 | 0.104 |

Verdict: **pass — no frame/convention problem.** The pooled signed median of
−0.5 mm means pose, intrinsics, and depth scale all agree with the MPS world;
the abort criterion (>20 cm median) is not met. The range structure matches
the sigma_Z model: measured 5.2 cm at ~2 m vs predicted 4.8 cm at
sigma_d = 0.5 px — FoundationStereo really is delivering ~half-pixel
disparity accuracy on these 512x512 monochrome images. The large 2.5–4 m
errors in two frames (median dz of −1.3 to −1.5 m) are dominated by
**occlusion**, not stereo failure: sparse semidense points behind a
foreground surface project into pixels where stereo correctly reports the
nearer surface, and a z-buffer over a sparse cloud cannot remove that. The
per-frame signed medians stay within ±7 cm on every frame.

## 5. Full export run

```bash
cd /home/sun/projectaria_gen2_depth_from_stereo
/home/sun/miniforge3/envs/depth_from_stereo/bin/python export_depth_from_stereo.py \
  --vrs /home/sun/aria/Outside_20260812_141244.vrs \
  --mps /home/sun/aria/mps_Outside_20260812_141244_vrs \
  --stereo_model ./FoundationStereo/ckpts/pretrained_models/23-51-11/model_best_bp2.pth \
  --output_dir <repo>/output/Outside_20260812_141244/stereo_depth/export_stride3 \
  --lr_check --stride 3
```

**1,558 frames** exported of 1,573 stride-3 candidates (the missing 15 are
SLAM frames before the MPS trajectory initialises at t = 161.53 s — the same
window drop preprocessing sees). 17.5 min wall at 1.5 fps, ~3.3 GB VRAM,
440 MB output. Full log: `stereo_depth/export_stride3.log`.

## 6. Fusion (task 4) — shared E2 fusion stage

The fusion stage of `scripts/extract_mesh_tsdf.py` was factored into
`scripts/tsdf_fusion.py` (`make_tsdf_volume` / `integrate_frame` /
`extract_mesh` / `subsample_indices_by_pose_delta`); `extract_mesh_tsdf.py`
now delegates to it with an unchanged CLI — **regression-verified** by
re-running the E2a command with `--max-frames 10` (same model load, same
pose-delta subsample counts, mesh written fine). The new
`scripts/extract_mesh_stereo_tsdf.py` consumes the tool's export directory
and drives the same shared stage, so E2/E3 fusion parameters are identical by
construction: voxel 0.02 m, trunc 0.08 m, pose-delta subsample 0.10 m/10 deg.

```bash
# repo root, ego_splats env (fusion is CPU-only, ~12 s)
python scripts/extract_mesh_stereo_tsdf.py \
  --export-dir output/Outside_20260812_141244/stereo_depth/export_stride3 \
  --depth-trunc 4.0 \
  --output output/Outside_20260812_141244/stereo_depth/mesh_tsdf_stereo_d40.ply
```

Masking before integration: LR-inconsistent pixels (11.0 % of all pixels
carried valid depth but failed the LR check and were dropped), depth outside
[0.3 m, depth-trunc]. Mean valid-pixel fraction 62.0 % at trunc 4.0.
Pose-delta kept **842/1,558** frames.

### Max fusion depth: swept, 4.0 m wins

The error model says the truncation band is exceeded beyond ~2.6 m
(sigma_d = 0.5 px), so three cutoffs were fused and scored (identical
everything else; `eval_mesh.py` defaults, h = 1.60):

| depth-trunc | coverage | largest hole m² | sd-median | junk m² | tris | components |
|---|---|---|---|---|---|---|
| 2.5 m | 0.8226 | 2.39 | 0.0220 | 11.3 | 1.68 M | 10,431 |
| **4.0 m (kept)** | **0.8638** | **0.91** | **0.0218** | 27.2 | 3.26 M | 25,411 |
| 6.0 m | 0.8408 | 1.20 | 0.0247 | 88.5 | 6.83 M | 79,124 |

Exactly the error model's shape: the 2.5–4 m band still adds real surface
(multi-view TSDF averaging tames the ~10–20 cm single-view noise; coverage
+4 pp, largest hole −62 %), while 4–6 m mostly adds smear (coverage down,
junk x3.3). **4.0 m is the kept setting**; per E2a's practice the losing
variants' PLYs were deleted, their scorecards/meta/logs kept.

The LR masks were also ablated at trunc 4.0 (`--no-mask`, scratchpad):
coverage 0.860 vs 0.864, sd-median 2.26 vs 2.18 cm, but junk 46.2 vs
27.2 m² and components 51 k vs 25 k — the masks buy a ~40 % junk reduction
for zero coverage cost. Masks on is the right call.

## 7. Scoring (task 6)

`scripts/compare_meshes.py` verbatim (default harness params h = 1.60,
±15 cm, 0.75 m buffer; full scorecards in
`stereo_depth/comparison_e3.json`):

```
                        mesh    cover   hole m2   sd-mean   sd-med   sd-p95   <10cm       tris   comps  nonmanif  junk m2
-------------------------------------------------------------------------------------------------------------------------
    mesh_tsdf_stereo_d40.ply    0.864      0.91     0.075    0.022    0.241    0.87    3259063   25411         0   27.235
     mesh_tsdf_2dgs_op03.ply    0.640      8.86     0.084    0.035    0.359    0.76    6581948  213708         5   86.039
          mesh_tsdf_op03.ply    0.501     19.94     0.066    0.032    0.252    0.80   10714519  767859        13  341.038
              floor_mesh.ply    0.978      0.48     0.616    0.408    1.859    0.29      14288       1         0    0.000
```

Failure decomposition (E2's method: downward rays, on-path =
trajectory-occupied cells vs 0.75 m buffer ring; 2DGS in brackets):

| | cells | covered | no-hit | junk-above | through-below |
|---|---|---|---|---|---|
| on-path | 1,028 | **0.946** [0.716] | 0.9 % [1.1 %] | 17 [79] | 30 [202] |
| buffer | 6,116 | **0.850** [0.628] | 3.3 % [7.6 %] | 368 [1,077] | 349 [736] |

- **Deep (>0.5 m) through-hole cells: 0** — versus 150 (2DGS) and 975
  (3DGS). The robot-falls-through failure mode is eliminated outright.
- **The stereo floor sits at the *calibrated* floor height.** All-hit dz
  percentiles p25/p50/p75 = −0.109/−0.076/−0.034 m against the default
  h = 1.60 expectation — i.e. a systematic −7.6 cm offset, which is within
  8 mm of the −6.8 cm shift the E2a-calibrated eye height (h = 1.6683 m)
  predicts. Rescoring at h = 1.6683 confirms it: stereo coverage **rises to
  0.9029** (largest hole 1.50 m², 6.94 m² missing;
  `mesh_tsdf_stereo_d40.ply.scorecard_h16683.json`) while 2DGS *drops*
  0.640 → 0.616. The splat meshes are biased a few cm high (foliage/smear
  pulls the fused surface up); the stereo mesh is where the floor actually
  is, and half its residual "misses" under the default harness are the
  harness's eye-height assumption, not mesh error.
- Semi-dense agreement: **median 2.18 cm, 87.1 % within 10 cm** — best of
  all candidates (2DGS 3.47 cm/76.3 %, 3DGS 3.19 cm/79.8 %). Only sd-*mean*
  is mid-pack (7.5 cm vs 3DGS's 6.6 cm) and that is the familiar extent
  artifact: the filtered cloud is wall/foliage-dominated (E0) and a 4 m
  cutoff models less far field than a 10.7 M-triangle splat mesh — by
  design.
- Hygiene: 0 non-manifold edges, 25 k components / 27 m² junk vs 214 k/86 m²
  (2DGS) and 768 k/341 m² (3DGS) — by far the least cleanup for E6.

## 8. Where stereo wins and loses

**Wins (as the brief predicted, but larger):** everything within ~4 m, which
on a head-mounted rig includes essentially the entire walkable floor
(eye height 1.67 m, so floor pixels sit at 1.7–2.6 m range where measured
accuracy is 3–5 cm). Best floor coverage of any observation-based route
(0.864/0.903 default/calibrated-h), best local accuracy, zero deep
through-holes, cleanest topology, and the floor at the metrically correct
height. Also the cheapest GPU route by far: no training — 17.5 min of
inference vs ~4 h of 2DGS retraining — and fusion is 12 s on CPU.

**Loses:** everything beyond ~4 m, by construction. sigma_Z = Z²·0.5px/41.28
crosses the 8 cm truncation at 2.6 m and reaches 1.2 m at 10 m — the d60
sweep shows including 4–6 m already degrades every metric, so distant walls,
buildings and canopy simply cannot come from this sensor. The mesh's AABB is
±8–9 m vs the splat meshes' ±12 m; anything needing far-field geometry
(visual context, distant obstacles) must come from the splat mesh or the
splat itself. Stereo also under-covers where the wearer never pointed the
front cameras (3.3 % no-hit in the buffer ring — glance coverage, same
failure class as the splats but milder because the SLAM cameras' FOV is
wide).

**Read for E6:** `mesh_tsdf_stereo_d40.ply` supersedes the 2DGS mesh as the
observation-based collision candidate (better on every metric that matters
for collision, 20x fewer junk components). The E5 floor remains the
complement for the remaining ~10 % of unobserved footprint —
`floor_from_trajectory.py --fuse-with` now has a much better partner: the
stereo mesh agrees with the calibrated floor height that the E5 floor is
built from, so fusing them should not fight over height the way the
slightly-high splat meshes would.

## Problems hit and fixed

1. **`~/.local/bin/gdown` is broken system-wide** (system Python 3.10 lacks
   `tqdm`); worked around by running the user-site gdown package through the
   `ego_splats` interpreter. Nothing installed or modified.
2. **One sanity frame missing:** `--stride 900` produced 5 frames, not 6 —
   VRS SLAM frame 0 (t ≈ 160.3 s) predates the MPS trajectory start
   (161.53 s) and the tool correctly skips poseless frames. Expected
   behavior, noted so the 5-frame sanity set is understood.
3. **Foreground-timeout handling:** the 17.5 min export was run detached
   with a log-tail waiter, per the machine's 10-min foreground limit.
   No failures in the run itself.

## Status

Complete. Tool set up in its own env (no existing env touched), full-walk
export with MPS closed-loop poses + LR masks, pre-fusion sanity check
passed (pooled signed median −0.5 mm vs semidense; abort criterion not
met), fusion via the shared (and regression-tested) E2 fusion stage, error
model quantified from measured B and f, depth-cutoff swept, mask ablation
done, scored and decomposed against all benchmarks. GPU idle before and
after. Nothing committed or pushed.
