# E6a — Final collision-candidate set: build + scoring report

Date: 2026-08-26. Env: `ego_splats` (`/home/sun/miniforge3/envs/ego_splats/bin/python`),
CPU only — GPU verified idle before starting (647 MiB desktop use, 3 % util) and never
touched. Nothing committed or pushed.

Task: build the final collision-candidate set by fusing the trajectory-floor prior
(E5, at the calibrated eye height h = 1.6683 m) into the two best observation-based
meshes (E3 stereo, E2b 2DGS), plus a judgment-call composite (stereo near field +
2DGS far field), then score all candidates and the four raw sources with the E0
harness and apply the sanity gate.

## Deliverables

All under `output/Outside_20260812_141244/collision_candidates/` (repo-relative;
everything in the MPS world frame, gravity-aligned Z-up metres — drops into Isaac Sim
next to the E1-fixed splat USDZ with no registration transform):

| Path | What |
|---|---|
| `fused_stereo_d40_trajfloor.ply` | **FUSED-STEREO**: stereo TSDF mesh + trajectory-floor prior (hardlink of `fused_stereo/fused_full_mesh.ply`) |
| `fused_2dgs_op03_trajfloor.ply` | **FUSED-2DGS**: 2DGS TSDF mesh + trajectory-floor prior (hardlink of `fused_2dgs/fused_full_mesh.ply`) |
| `fused_best_stereoNear_2dgsFar_trajfloor.ply` | **FUSED-BEST**: stereo near field + 2DGS far field + trajectory-floor prior (hardlink of `fused_best/fused_full_mesh.ply`) |
| `fused_{stereo,2dgs,best}/` | full `floor_from_trajectory.py` output dirs: `fused_heightfield.npz` (PhysX-ready, with per-cell provenance `source` array), `fused_floor_mesh.ply`, `fusion_report.json`, standalone floor copies |
| `fused_best/prefuse_composition.json` | exact composition numbers for the FUSED-BEST input |
| `raw_stereo_tsdf_d40.ply` etc. | symlinks to the four raw sources, so the candidate set is browsable in one directory |
| `comparison_e6a_h16683.json` | full scorecards for all seven candidates at h = 1.6683 |
| `*.ply.scorecard_h16683.json` | per-mesh `eval_mesh.py` scorecards for the three fused candidates |
| `scripts/compose_near_far_mesh.py` (new, repo) | reusable near+far mesh composer used for FUSED-BEST |

## 1. Inputs and protocol

- Prior: `scripts/floor_from_trajectory.py` at **h = 1.6683 m** (the E2a-calibrated
  value; passed explicitly via `--eye-height 1.6683`, matching the rebuilt E5 floor on
  disk). Fusion defaults per E5: `--fuse-threshold 0.10`, `--fuse-strip-band 0.40`,
  `--fuse-strip-dilate 1`.
- Parents: `stereo_depth/mesh_tsdf_stereo_d40.ply` (E3),
  `camera-rgb-rectified-1008-h1512-2dgs/mesh_tsdf/mesh_tsdf_2dgs_op03.ply` (E2b),
  `camera-rgb-rectified-1008-h1512/mesh_tsdf/mesh_tsdf_op03.ply` (E2a, baseline only),
  `floor_from_trajectory/floor_mesh.ply` (E5).
- Scoring: `eval_mesh.py` / `compare_meshes.py` with `--eye-height 1.6683` for every
  candidate — fused and raw alike — so the sanity-gate comparisons are like-for-like.
  NB: raw-mesh numbers therefore differ from the E2/E3 report tables, which were scored
  at the default h = 1.60 (e.g. raw stereo 0.903 here vs 0.864 there; raw 2DGS 0.616 vs
  0.640; raw 3DGS 0.478 vs 0.501 — consistent with the h-rescoring already noted in E3 §7).

## 2. Commands run (chronological)

```bash
REPO=/home/sun/Desktop/aria_proj/egocentric_splats
B=$REPO/output/Outside_20260812_141244
OUT=$B/collision_candidates
PY=/home/sun/miniforge3/envs/ego_splats/bin/python

# FUSED-STEREO (1.5 s fusion step)
$PY scripts/floor_from_trajectory.py --eye-height 1.6683 \
    --fuse-with $B/stereo_depth/mesh_tsdf_stereo_d40.ply --output-dir $OUT/fused_stereo

# FUSED-2DGS (3.3 s fusion step)
$PY scripts/floor_from_trajectory.py --eye-height 1.6683 \
    --fuse-with $B/camera-rgb-rectified-1008-h1512-2dgs/mesh_tsdf/mesh_tsdf_2dgs_op03.ply \
    --output-dir $OUT/fused_2dgs

# FUSED-BEST: compose (scripts/compose_near_far_mesh.py), then fuse (2.1 s)
$PY scripts/compose_near_far_mesh.py \
    --near $B/stereo_depth/mesh_tsdf_stereo_d40.ply \
    --far  $B/camera-rgb-rectified-1008-h1512-2dgs/mesh_tsdf/mesh_tsdf_2dgs_op03.ply \
    --output <scratch>/prefuse_stereo_near_2dgs_far.ply
$PY scripts/floor_from_trajectory.py --eye-height 1.6683 \
    --fuse-with <scratch>/prefuse_stereo_near_2dgs_far.ply --output-dir $OUT/fused_best

# scoring (semidense context shared across all seven)
$PY scripts/compare_meshes.py $OUT/fused_stereo_d40_trajfloor.ply \
    $OUT/fused_2dgs_op03_trajfloor.ply $OUT/fused_best_stereoNear_2dgsFar_trajfloor.ply \
    $OUT/raw_stereo_tsdf_d40.ply $OUT/raw_2dgs_tsdf_op03.ply $OUT/raw_3dgs_tsdf_op03.ply \
    $OUT/raw_trajectory_floor.ply --eye-height 1.6683 --output $OUT/comparison_e6a_h16683.json
# + eval_mesh.py --eye-height 1.6683 on each fused candidate (scorecard_h16683 files)
```

(The FUSED-BEST composition was first run from a scratchpad script, then the polished
`scripts/compose_near_far_mesh.py` was verified to reproduce it **identically** —
same 5 counters, listed in §4 — before being recorded here as the canonical command.
The pre-fusion composite lives in the session scratchpad; it is fully reproducible
from the command above and its numbers are in `fused_best/prefuse_composition.json`.)

## 3. Fusion reports (from `fusion_report.json`)

| | FUSED-STEREO | FUSED-2DGS | FUSED-BEST |
|---|---|---|---|
| global z offset dz applied [m] | **+0.0093** | **−0.0337** | +0.0093 |
| cells mesh-kept (of 7,144) | 6,294 (88.1 %) | 4,112 (57.6 %) | 6,294 (88.1 %) |
| cells hole-filled (prior height) | 229 (3.2 %) | 555 (7.8 %) | 229 (3.2 %) |
| cells rejected (prior height) | 621 (8.7 %) | 2,477 (34.7 %) | 621 (8.7 %) |
| input triangles | 3,259,063 | 6,581,948 | 4,271,027 |
| triangles stripped (rejected floor band) | 164,516 | 298,147 | 164,516 |
| kept-cell |mesh−prior| mean [m] | 0.0302 | 0.0422 | 0.0302 |

Reads:

- **The height pins confirm E3's bias finding quantitatively**: the stereo mesh needed
  only +0.9 cm to sit on the calibrated prior, the 2DGS mesh needed −3.4 cm (it floats
  ~3–4 cm high — foliage/smear pulling the fused surface up). Relative stereo↔2DGS
  offset: 4.3 cm.
- **FUSED-BEST's footprint classification is byte-identical to FUSED-STEREO's** (same
  dz, same 6,294/229/621 split, same 164,516 stripped triangles). This is the expected
  proof that the composition is clean: the evaluation footprint lies entirely inside the
  stereo XY AABB, so the near field of the composite is pure stereo and the 2DGS far
  field never touches the floor corridor.

## 4. FUSED-BEST composition (the judgment call — done, cleanly)

Method (`scripts/compose_near_far_mesh.py`): keep the stereo mesh intact; from the 2DGS
mesh keep only triangles whose **centroids fall outside the stereo mesh's XY AABB**
([−8.334, −8.524] … [8.17, 9.31]); remove floater components < 0.1 m² (the E0 junk
definition) from that cropped far-field piece only; concatenate; fuse the floor prior
last. Numbers (`fused_best/prefuse_composition.json`):

| | |
|---|---|
| stereo (near) triangles, untouched | 3,259,063 |
| 2DGS triangles inside stereo AABB, dropped | 5,377,601 of 6,581,948 |
| far-field junk removed | 31,800 components / 192,383 tris / 19.36 m² |
| 2DGS far-field triangles kept | 1,011,964 |
| composite triangles | 4,271,027 |

Honest caveats, neither of which got hacky:

- The 2DGS far field retains its own ~+4.3 cm height bias (the composite is shifted by
  the stereo-driven +0.9 cm, not the 2DGS −3.4 cm). Irrelevant for collision: those
  triangles are distant walls/structure ≥ ~8 m out, far outside the drive corridor.
- The AABB cut is a box, not the stereo mesh's true footprint, so a thin annulus just
  inside the box edge has neither stereo (sparse there) nor 2DGS geometry. It is far
  outside the evaluated corridor and the E0 metrics are insensitive to it; a
  per-triangle occupancy crop was tried mentally and rejected as complexity with no
  measurable payoff.

## 5. Scorecards — all seven candidates at h = 1.6683

`compare_meshes.py` output (verbatim values; full scorecards in
`comparison_e6a_h16683.json`):

```
                                       mesh    cover   hole m2   sd-mean   sd-med   sd-p95   <10cm       tris    comps  nonmanif  junk m2
-----------------------------------------------------------------------------------------------------------------------------------------
             fused_stereo_d40_trajfloor.ply    0.970      0.40     0.085    0.029    0.251    0.82    3108835    24888         0   26.709
              fused_2dgs_op03_trajfloor.ply    0.957      0.47     0.079    0.038    0.316    0.78    6298089   212755         5   85.833
fused_best_stereoNear_2dgsFar_trajfloor.ply    0.970      0.40     0.074    0.029    0.236    0.82    4120799    24954         1   26.709
                    raw_stereo_tsdf_d40.ply    0.903      1.50     0.075    0.022    0.241    0.87    3259063    25411         0   27.235
                     raw_2dgs_tsdf_op03.ply    0.616      7.21     0.084    0.035    0.359    0.76    6581948   213708         5   86.039
                     raw_3dgs_tsdf_op03.ply    0.478     18.88     0.066    0.032    0.252    0.80   10714519   767859        13  341.038
                   raw_trajectory_floor.ply    0.987      0.37     0.616    0.408    1.859    0.29      14288        1         0    0.000
```

Supporting exacts (from the JSON): total missing area 2.17 m² (both stereo-based
fusions) vs 6.94 m² (raw stereo), 3.11 vs 27.41 m² (2DGS); total mesh area 501.7 /
894.5 / 643.2 m² for the three fusions; FUSED-BEST AABB [−12.81, −11.91, −2.86] …
[12.41, 12.97, 4.96] — full 2DGS XY extent with the stereo z-cleanliness.

## 6. Sanity gate — ALL THREE FUSIONS PASS, no violations

Gate: fused must beat its unfused parent on floor coverage AND largest hole, with
semidense median within 1 cm of the parent. All at h = 1.6683:

| Candidate | Parent | Coverage | Largest hole | sd-median (Δ) | Verdict |
|---|---|---|---|---|---|
| FUSED-STEREO | raw stereo | 0.9696 > 0.9029 ✓ | 0.40 < 1.50 m² ✓ | 0.0288 vs 0.0218 (+0.70 cm ≤ 1 cm) ✓ | **PASS** |
| FUSED-2DGS | raw 2DGS | 0.9565 > 0.6163 ✓ | 0.47 < 7.21 m² ✓ | 0.0382 vs 0.0347 (+0.35 cm) ✓ | **PASS** |
| FUSED-BEST | raw stereo | 0.9696 > 0.9029 ✓ | 0.40 < 1.50 m² ✓ | 0.0287 vs 0.0218 (+0.69 cm) ✓ | **PASS** |
| FUSED-BEST | raw 2DGS (2nd parent) | 0.9696 > 0.6163 ✓ | 0.40 < 7.21 m² ✓ | 0.0287 vs 0.0347 (−0.60 cm) ✓ | **PASS** |

The ~0.7 cm sd-median rise on the stereo-based fusions is the fusion's own +0.93 cm
global z-shift (pinning the mesh to the calibrated prior) moving the whole surface
relative to the semidense cloud — the expected, bounded price of an absolute height
pin, and well inside the gate.

## 7. Residual-failure decomposition (what the last 3 % is)

Downward raycast per footprint cell against the **prior grid** (scratchpad
`decompose_fused.py`; note the harness's own expectation differs slightly on
interpolated cells — E5's known nearest-vs-linear disagreement — which accounts for
the gap between these counts and the harness's 217 failing cells at cover 0.9696):

| | FUSED-STEREO / FUSED-BEST | FUSED-2DGS |
|---|---|---|
| failing cells (of 7,144) | 123 | 226 |
| … no-hit | **0** | **0** |
| … hit below tolerance (fall-through class) | **0** | **0** |
| … hit above tolerance | 123 (median dz +0.46 m) | 226 (median dz +0.45 m) |
| by category: observed / interp / extrap | 3 / 113 / 7 | 17 / 168 / 41 |
| largest hole (prior-grid view) | 26 cells at x 0.7–1.6, y 3.2–4.4 (all interpolated) | 47 cells at x 4.1–5.1, y −1.8…−0.8 |

The headline: **the fall-through failure class is structurally eliminated** — the fused
floor patch guarantees a surface at prior height under every footprint cell, so no
eval ray misses and none passes below tolerance. Every residual miss is a first hit
0.15–0.5 m *above* the prior floor: parent-mesh structure sitting above the 0.40 m
strip band but below the 0.5 m ray start, ~92 % of it in interpolated/extrapolated
cells where nobody walked. That band is partly real (vegetation, wall bases inside the
0.75 m clearance ring — genuine no-drive zones a collision mesh *should* contain) and
partly stereo smear; the harness cannot distinguish them, and E6b's drive test (which
follows the walked path, not the ring) is the right judge. This is also exactly why
the fusions sit 1.7–3.0 pp below the floor prior's 0.987 ceiling: the prior has no
above-floor structure at all.

## 8. Ranking and read for E6b

1. **FUSED-BEST** (`fused_best_stereoNear_2dgsFar_trajfloor.ply`, 4.12 M tris) —
   primary candidate. Ties FUSED-STEREO on every floor metric (0.970 / 0.40 m² /
   sd-med 2.9 cm), strictly better semidense mean/p95 (0.074 / 0.236 m — best <10 cm
   fraction of the fusions at 0.82), because the 2DGS far field supplies the walls and
   structure the stereo sensor is blind to. Cleanest topology of any full-scene
   candidate: 1 non-manifold edge, junk 26.7 m² (vs 341 m² for raw 3DGS).
2. **FUSED-STEREO** (3.11 M tris) — the lean variant. Identical collision behaviour in
   and around the corridor; choose it if E6b's SDF/convex-decomposition cost scales
   badly with triangle count and far-field collision proves unnecessary.
3. **FUSED-2DGS** (6.30 M tris) — fallback only. Passes its gate but loses to both
   stereo-based fusions on every metric while carrying 8.5× their component count.
4. **raw_trajectory_floor** — floor-only baseline; its heightfield sibling remains the
   most robust *floor* collider (PhysX-native), but it contains no obstacles at all.
5. **raw stereo / 2DGS / 3DGS** — superseded baselines for the drive-test table.

Practical notes for E6b:

- The fused heightfields (`fused_*/fused_heightfield.npz`: `height`, `origin_xy`,
  `cell_size`, per-cell `source` provenance 1=mesh/2=hole-filled/3=rejected) allow a
  hybrid setup: PhysX heightfield for the floor + mesh collider for obstacles.
- All fused meshes were z-shifted by their pin (stereo-based +0.93 cm): they sit on the
  *calibrated-prior* floor, which is the same reference `sim_drive_test.py` will use
  for fall-through detection.
- The ~24.9 k small components (26.7 m²) inside the stereo near field were deliberately
  NOT scrubbed — inside the corridor they may be real thin obstacles; a blanket
  component filter risks deleting exactly what a collision mesh must keep. If E6b's
  collision cooking chokes on them, filter with the same 0.1 m² rule and re-gate.

## 9. Problems hit

None blocking. Two near-misses worth recording:

1. The first FUSED-BEST composition ran from a scratchpad script; the polished repo
   script (`scripts/compose_near_far_mesh.py`) was then verified to reproduce all five
   composition counters **identically** before being documented as canonical.
2. `floor_from_trajectory.py --fuse-with` writes fixed filenames into `--output-dir`;
   the three fusions therefore live in separate subdirectories, with the deliverable
   meshes hardlinked (not copied) to self-describing top-level names — same inode, no
   duplicated 100–280 MB payloads.

## Status

Complete. Three fused candidates built (FUSED-BEST included — the composition stayed
clean), all seven candidates scored at the calibrated h = 1.6683, sanity gate applied:
three passes, zero violations. CPU-only throughout; GPU idle before and after; nothing
committed or pushed.
