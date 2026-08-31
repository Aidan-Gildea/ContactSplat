# E0 — Evaluation harness: run report

Date: 2026-08-20. Env: `ego_splats` (`/home/sun/miniforge3/envs/ego_splats/bin/python`,
Python 3.10.20). CPU only — no GPU job was started. Nothing committed.

## Deliverables

| File | What |
|---|---|
| `scripts/eval_mesh.py` | Scores one mesh (PLY/OBJ) against the MPS data → JSON scorecard |
| `scripts/compare_meshes.py` | Runs the same metrics over N meshes, prints a ranking table, shares the MPS context across meshes |

Validation assets (synthetic meshes + scorecards + generator script) are in the session
scratchpad, not the repo:
`/tmp/claude-1000/-home-sun-Desktop-aria-proj-egocentric-splats/83deeb14-482b-4b45-b1f1-27ad7da94749/scratchpad/`
(`make_synthetic_meshes.py`, `e0_meshes/*.ply`, `e0_meshes/comparison.json`).

## What the harness measures

All in the MPS world frame (gravity-aligned, Z-up, metres). Defaults are the brief's
numbers and are CLI-overridable (`--cell 0.10 --buffer 0.75 --eye-height 1.60
--z-tol 0.15 --ray-height 0.50`).

1. **Floor coverage** — trajectory from `closed_loop_trajectory.csv` (read via pandas,
   `t{x,y,z}_world_device` columns; 155,800 poses) rasterized to a 10 cm XY grid,
   buffered 0.75 m laterally via a distance transform. Per-cell expected floor = local
   trajectory Z (per-cell mean; nearest occupied cell for buffered cells, via
   `distance_transform_edt(return_indices=True)`) minus 1.6 m eye height. One downward
   ray per cell from 0.5 m above expected floor (`open3d.t.geometry.RaycastingScene`);
   covered = hit within ±15 cm of expected.
2. **Largest hole** — 8-connected components of the not-covered cells; area of the
   largest, in m². Also reports hole count and total missing area.
3. **Semi-dense agreement** — `mps.read_global_point_cloud` +
   `filter_points_from_confidence(pts, 0.005, 0.01)` — the exact thresholds
   `scripts/extract_aria_vrs.py` uses (its lines 417–421), so the reference cloud is the
   same one the splat was initialised from. Unsigned point-to-mesh distance via
   `RaycastingScene.compute_distance`; reports mean/median/p95/max and inlier fractions
   at 5/10/25 cm. `--max-points` subsamples, `--no-semidense` skips.
4. **Mesh hygiene** — triangle/vertex count, AABB, total area, non-manifold edge count
   (`get_non_manifold_edges(allow_boundary_edges=True)`, i.e. edges shared by >2
   triangles), connected-component count, and total area of components < 0.1 m²
   ("junk"). Topology metrics run after `remove_duplicated_vertices()` (exact merge) so
   a triangle-soup export with one vertex per corner is judged on its real
   connectivity instead of reporting one component per triangle. Raw counts and bbox
   are from the unmodified mesh.

Design choice worth flagging: per-cell trajectory Z uses the **mean**, not the median,
of the ~1 kHz samples in each 10 cm cell — at that density the two are
indistinguishable and the mean vectorises with `np.add.at`. Everything else follows the
brief literally.

## Environment problem hit and fixed

`import open3d` failed in `ego_splats` with
`ModuleNotFoundError: No module named 'importlib_metadata'`. Open3D 0.19.0 actually
lives in the **user site** (`~/.local/lib/python3.10/site-packages`), which precedes
the env's site-packages on `sys.path` (along with two ROS Humble paths); its
`open3d.visualization` → `draw_plotly` → `dash` import chain needs
`importlib_metadata`, which was missing everywhere on the path. Fix — minimal, no
touching `~/.local` or other envs:

```bash
/home/sun/miniforge3/envs/ego_splats/bin/pip install importlib_metadata   # got 9.0.0 (+zipp 4.1.0)
```

After that `open3d` 0.19.0 imports and `RaycastingScene` works in this env.

## Validation on synthetic known-good / known-bad meshes

Generator (scratchpad `make_synthetic_meshes.py`) builds, from the real trajectory:

- `terrain_good.ply` — grid mesh following the per-cell expected floor (known-good);
- `terrain_hole.ply` — same with a 1.0 m-radius disk of triangles removed at the path
  midpoint (expected hole ≈ π ≈ 3.14 m²);
- `terrain_hole_floaters.ply` — plus 20 floating 5 cm boxes (each 0.015 m² area →
  0.300 m² of junk expected);
- `terrain_offset1m.ply` — terrain shifted 1 m down (known-bad height);
- `plane_good.ply` / `plane_hole.ply` — the brief's flat plane at the median expected
  floor height (−1.719 m), with/without the hole.

Commands actually run:

```bash
PY=/home/sun/miniforge3/envs/ego_splats/bin/python
S=<scratchpad>/e0_meshes
$PY <scratchpad>/make_synthetic_meshes.py --out-dir $S
$PY scripts/eval_mesh.py $S/terrain_good.ply --output $S/terrain_good.scorecard.json
$PY scripts/compare_meshes.py $S/terrain_good.ply $S/terrain_hole.ply \
    $S/terrain_hole_floaters.ply $S/terrain_offset1m.ply \
    $S/plane_good.ply $S/plane_hole.ply --output $S/comparison.json
```

Result table (verbatim from `compare_meshes.py`):

```
                        mesh    cover   hole m2   sd-mean   sd-med   sd-p95   <10cm       tris   comps  nonmanif  junk m2
-------------------------------------------------------------------------------------------------------------------------
            terrain_good.ply    0.995      0.20     0.454    0.212    1.529    0.34      39182       1         0    0.000
            terrain_hole.ply    0.952      3.06     0.459    0.226    1.529    0.33      38555       1         0    0.000
   terrain_hole_floaters.ply    0.952      3.06     0.452    0.226    1.524    0.33      38795      21         0    0.300
        terrain_offset1m.ply    0.000     71.44     1.354    1.120    2.495    0.00      39182       1         0    0.000
              plane_good.ply    0.223     28.52     0.607    0.465    1.591    0.07      39182       1         0    0.000
              plane_hole.ply    0.202     30.00     0.611    0.472    1.592    0.06      38555       1         0    0.000
```

Every metric moves the right way, with quantitative agreement:

- **Known-good terrain**: coverage 0.9945 over the 7,144-cell / 71.44 m² footprint,
  mean |hit − expected| = 6.8 mm. Not exactly 1.0: 6 residual holes (largest 0.20 m²)
  sit where the expected floor changes steeply between adjacent passes, and
  nearest-cell vertex sampling in the synthetic mesh itself deviates >15 cm there —
  a property of the test mesh, not the metric.
- **Hole detection**: cutting a 1.0 m disk drops coverage by exactly the hole's share
  (0.995→0.952; 3.06/71.44 ≈ 0.043) and reports largest hole 3.06 m² vs π·1² = 3.14 m²
  (10 cm grid quantization).
- **Floaters**: floor metrics unchanged; components 1→21 and junk area 0.300 m² —
  exactly 20 × 0.015 m².
- **1 m height offset (known-bad)**: coverage 0.000, largest hole = the entire
  71.44 m² footprint, semi-dense mean distance 0.454→1.354 m, 10 cm-inlier fraction
  0.34→0.00.
- **Semi-dense metric is discriminative but scene-diluted**: even the perfect floor
  scores only 34 % of points within 10 cm, because the filtered cloud (938,113 of
  4,292,410 raw points) mostly lies on walls/foliage above the floor. Fine for
  *ranking* full-scene meshes; do not read it as an absolute floor-accuracy number for
  floor-only meshes like E5's.

Extra checks beyond the brief:

- **Non-manifold detection**: a hand-built 3-triangles-on-one-edge mesh reports exactly
  1 non-manifold edge; a triangle-soup version of it still reports 1 component and 1
  non-manifold edge after the exact-merge step.
- **Multi-million-triangle speed**: a 4,138,242-triangle plane (0.01 m grid) evaluates
  in 15.8 s wall total; breakdown: mesh load 0.26 s, hygiene 9.94 s (dominated by
  `cluster_connected_triangles`), floor raycast 0.80 s, 938k semi-dense distance
  queries 0.04 s. MPS context build adds ~4 s (semi-dense CSV load 3.5 s) and is done
  once per `compare_meshes.py` run.
- **OBJ input**: the same mesh saved as `.obj` produces identical numbers.
- **Resolution independence**: the 0.01 m-grid flat plane scores coverage 0.2226 vs
  0.223 for the 0.1 m-grid one.

## Findings that matter for the other experiments

1. **A flat floor plane is wrong for this scene.** The expected floor over the
   footprint spans −2.574…−0.916 m; 77.7 % of footprint cells deviate >0.15 m from the
   median. Either the walk has real grade or the wearer's height above ground varied
   substantially. Consequence: E5 must stay terrain-following (as briefed), and any
   evaluation assuming one floor height would be meaningless — the harness therefore
   uses local trajectory Z per cell, per the brief.
2. **Eye height 1.6 m is an assumption baked into `--eye-height`.** E5 is tasked with
   calibrating it; when it does, re-run comparisons with the fitted value.
3. Trajectory footprint at the default 0.75 m buffer: 7,144 cells = **71.44 m²**;
   grid 122×128 at 0.1 m; XY extent roughly [−6.0…6.2] × [−4.9…7.9] m.
4. Filtered semi-dense reference cloud: **938,113 points** (of 4,292,410 raw) at
   thresholds inv_dist_std ≤ 0.005, dist_std ≤ 0.01.

## Status

Complete. Both scripts work end-to-end in `ego_splats`, CPU-only, validated at the
known-good and known-bad extremes. Ready for E2–E5 to consume; interface is
`eval_mesh.EvalContext` + `evaluate_mesh()` for programmatic use, or the CLIs.
