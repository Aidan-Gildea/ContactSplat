# E5 — Trajectory-derived floor: run report

Date: 2026-08-20. Env: `ego_splats`, CPU only (no GPU touched). Nothing committed.

## Deliverables

| File | What |
|---|---|
| `scripts/floor_from_trajectory.py` | New script: trajectory → floor mesh + heightfield, `--calibrate-with`, `--fuse-with` |
| `output/Outside_20260812_141244/floor_from_trajectory/floor_mesh.ply` | Standalone floor mesh (MPS world frame, Z-up, metres) |
| `output/Outside_20260812_141244/floor_from_trajectory/floor_heightfield.npz` (+`.json`) | Heightfield: `height` (122×128 float32, NaN outside), `category` (0=outside 1=observed 2=interpolated 3=extrapolated), `origin_xy`, `cell_size` |
| `output/Outside_20260812_141244/floor_from_trajectory/floor_mesh.ply.scorecard.json` | E0-harness scorecard |

Validation assets (synthetic meshes, fusion outputs) live in the session scratchpad
(`.../scratchpad/{variants,fusion_test,fusion_test_h160}/`, `synth_*.ply`) and are
reproducible from the commands below.

## What the script does

1. Loads `closed_loop_trajectory.csv` via `projectaria_tools`
   (`mps.read_closed_loop_trajectory`) — 155,800 poses, device Z −0.977…0.686 m.
2. Rasterizes trajectory XY into 10 cm cells; per-cell **median** device Z − eye
   height h (default **1.6 m**, see calibration section) = observed floor height.
3. Dilates laterally to the robot clearance (default 0.75 m) via distance transform.
4. Fills non-observed footprint cells: `scipy.interpolate.griddata` **linear**
   between passes (flag `interpolated`), nearest-observed-cell outside the convex
   hull of observations — mostly the dilation ring (flag `extrapolated`).
5. Optional masked-Gaussian smoothing of heights (`--smooth-sigma`, default 1 cell)
   to suppress gait bounce. Category flags are never smoothed.
6. Emits **both** a triangle mesh (vertices on cell *corners*, so every footprint
   cell is fully covered by two triangles — no boundary raycast ambiguity) and the
   heightfield npz (PhysX-friendly: regular grid + origin + cell size + validity).

Grid construction (margin, cell, clearance) deliberately mirrors
`scripts/eval_mesh.py` so grids align exactly: same 122×128 grid, same origin,
same 7,144-cell / 71.44 m² footprint.

## Commands run

```bash
cd /home/sun/Desktop/aria_proj/egocentric_splats
PY=/home/sun/miniforge3/envs/ego_splats/bin/python

# standalone floor (defaults: h=1.6, cell=0.10, clearance=0.75, smooth=1)
$PY scripts/floor_from_trajectory.py            # ~0.5 s total
$PY scripts/eval_mesh.py output/Outside_20260812_141244/floor_from_trajectory/floor_mesh.ply

# variants / validation (scratchpad)
$PY scripts/floor_from_trajectory.py --smooth-sigma 0 --output-dir $SCR/variants/nosmooth
$PY scripts/floor_from_trajectory.py --calibrate-with $SCR/synth_calib_h162.ply \
    --fuse-with $SCR/synth_candidate.ply --output-dir $SCR/fusion_test
$PY scripts/floor_from_trajectory.py --fuse-with $SCR/synth_candidate.ply \
    --output-dir $SCR/fusion_test_h160
```

## Standalone floor scorecard (E0 harness, default params)

| Metric | Value |
|---|---|
| Floor coverage | **0.9866** (7,048 / 7,144 cells, 71.44 m² footprint) |
| Largest hole | **0.37 m²** (5 holes, 0.96 m² missing total) |
| Hit abs error (covered cells) | 0.0212 m mean |
| Semi-dense distance | mean 0.5874 m, median 0.3581 m, p95 1.8049 m (938,113 pts) |
| Hygiene | 14,288 tris, **1 component, 0 non-manifold edges, 0 floater junk** |
| Bbox | [−5.731, −4.675, −2.434] … [5.969, 7.625, −0.919] |

Reference points: E0's known-good synthetic terrain (built at the harness's own
expectation) scores 0.9945 / 0.20 m²; a flat plane scores 0.223 (E0 finding:
77.7 % of cells deviate >0.15 m from the median — the terrain has real grade,
which this floor follows automatically; floor Z spans −2.45…−0.92 m).

### Where the 1.34 % missing cells are

Of 96 failing cells: 79 `interpolated`, 17 `observed`, 0 `extrapolated`;
83 out-of-tolerance (|dz| median 0.21 m, max 0.50 m), 13 no-hit (my surface
> 0.5 m above the harness expectation, so the ray starts under it). Per-category
|mine − harness-expected|: observed median 0.009 m; interpolated median 0.020 m,
max 0.82 m; extrapolated max 0.11 m.

Root cause: between passes walked at different heights (the scene has ~1.6 m of
grade), my linear interpolation and the harness's nearest-neighbour expectation
are *both* guesses and can disagree by more than the 0.15 m tolerance. Neither is
ground truth there — nobody walked those cells. Smoothing is not the cause: a
`--smooth-sigma 0` build scores 0.9871 vs 0.9866, so the default (which removes
gait bounce) is kept.

Intrinsic strength of the prior along the walked path: median 120 trajectory
samples per observed cell; per-cell device-Z std median **0.0029 m** (p95
0.036 m, max 0.46 m on multi-pass cells at different grade heights).

## Eye-height calibration (amended scope)

**Amendment honoured:** no other pipeline (E2/E3/E4) has produced a mesh yet, so
h could not be calibrated against real recovered floor geometry. h is a
parameter (`--eye-height`, default 1.6 m — the value E0 assumed), and
calibration is implemented as a reusable function
`calibrate_eye_height(traj_xyz, mesh_path, ...)` exposed as
`--calibrate-with <mesh>`: it casts rays straight down from each per-cell median
device position onto the given mesh, takes h = device_z − hit_z per cell
(accepting 1.0–2.5 m to reject overhangs/floaters), and reports the median fit
plus spread (std / IQR / p5–p95). **Rerun with a real E2/E3/E4 mesh once one
exists.**

Validated on a synthetic mesh built at exactly h_true = 1.62 m below the
per-cell median trajectory: fitted **1.6203 m** (mean 1.6187, std 0.0411,
IQR 0.0142, p5–p95 1.582–1.652; 1,035/1,035 cells hit). The residual std comes
from mesh-node averaging over terrain, not the estimator. When calibration
succeeds the fitted h replaces the default for the floor build; on failure
(< 50 usable cells) it falls back to the assumed value and says so.

## Fusion mode (`--fuse-with`), validated synthetically

Design: raycast the candidate mesh downward at every footprint cell from
prior + 0.5 m → per-cell mesh floor height; **pin absolute floor height** by
dz = median(prior − mesh) over all hit cells and translate the whole input mesh
by dz; classify cells *kept* (|mesh+dz − prior| ≤ `--fuse-threshold`, default
0.10 m), *hole-filled* (no hit → prior height), *rejected* (disagree → prior
height). Outputs: fused heightfield (+ per-cell `source` provenance array),
fused floor mesh, and `fused_full_mesh.ply` = bias-corrected input with
rejected floor-band triangles stripped + the fused floor patch.

Synthetic candidate (built from the h=1.6 floor): global −5 cm Z bias, a 0.6 m
radius hole cut on the walked path, and a 1.5×1.5 m patch raised +0.30 m.

| Check | Expected | Measured |
|---|---|---|
| Pinned offset dz | +0.050 | **+0.0499** |
| dz under calibrated prior (h=1.6203) | 0.05−0.0203 = +0.0297 | **+0.0296** |
| Hole-filled cells | ≈ π·0.6²/0.01 ≈ 113 | **112** |
| Rejected cells | ≥ 225 (patch) + skirt | **252** |
| Kept-cell residual vs prior | ~0 | 0.0023 m mean |

Scorecards (semidense column on real MPS points):

| Mesh | Coverage | Largest hole | Components | Junk <0.1 m² |
|---|---|---|---|---|
| Synthetic candidate alone | 0.9486 | 1.87 m² | — | — |
| Fused (first attempt, band 0.30, no dilation) | 0.9735 | 1.27 m² | 59 | 0.465 m² |
| **Fused (final: band 0.40, strip-dilate 1)** | **0.9863** | **0.37 m²** | **2** | **0.0** |
| Prior alone (ceiling) | 0.9866 | 0.37 m² | 1 | 0.0 |

**Problem hit and fixed:** the first fusion attempt stripped rejected floor
triangles by centroid cell only. Skirt triangles of the raised patch whose
centroids fell in *agreeing* neighbour cells survived, hung above the fused
floor, and were hit first by the eval rays (59 fragments, 0.465 m² junk,
coverage stuck at 0.9735). Additionally the +0.30 m defect sat exactly on the
0.30 m strip-band float boundary. Fix: dilate the rejected-cell strip mask by
1 cell (`--fuse-strip-dilate`, default 1) and widen the default band to 0.40 m.
Fusion then recovers the prior's ceiling exactly.

## Honest limitations (state of the prior)

- **A trajectory is a curve, not a surface.** Only 1,028 of 7,144 footprint
  cells (14 %) were actually walked over; 4,315 are interpolated between passes
  and 1,801 extrapolated sideways into the clearance ring. The traversability
  guarantee is strongest at `observed` cells and is an *assumption* elsewhere —
  the category array exists precisely so downstream consumers can weigh this.
- Linear interpolation between passes at different heights can bridge across a
  real drop (curb, step) that the walker went around; the harness cannot detect
  this because it extrapolates the same way.
- Fusion trusts the prior across the whole footprint: genuine low obstacles
  inside the *dilated ring* (where nobody walked) that sit within the strip
  band would be rejected and flattened into floor. Over walked cells this
  cannot happen (the space was free by demonstration), but E6's drive test
  should be the judge for the ring.
- The semi-dense agreement numbers (median 0.36 m) are expected to look poor
  for any floor-only mesh: the MPS cloud is dominated by walls/foliage above
  the floor. Comparable to E0's synthetic terrain (mean 0.454 m). Use this
  metric only to compare floor meshes against each other, not against
  full-scene meshes.
- h = 1.6 m is still an assumption for the deliverable mesh. The scene's floor
  height is only known relative to that constant; a real-mesh calibration
  (function is ready) will shift the whole surface by (h_fit − 1.6).

## Runtime

Whole pipeline (load 155,800 poses → grid → mesh → heightfield) ≈ 0.5 s CPU.
Full eval-harness scoring ≈ 5 s (semidense load dominates). No GPU used.

## Remaining work for later tasks

1. When E2/E3/E4 meshes exist: `--calibrate-with <mesh>` to fit h for real, then
   rebuild; report the fitted value and spread.
2. `--fuse-with <real mesh>` and score `fused_full_mesh.ply` — expected to fix
   exactly the failure modes those pipelines are predicted to have (floor holes
   on textureless concrete for E4, long-range noise for E3).
3. E6: feed `floor_heightfield.npz` directly as a PhysX heightfield collider —
   it is strictly more robust than the triangle mesh for a wheeled robot
   (`height` + `origin_xy` + `cell_size`; NaN cells are outside the corridor).
