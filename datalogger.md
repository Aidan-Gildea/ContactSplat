# ContactSplat data log

A running record of every experiment on this pipeline: what was run, on which data, with which settings, and what came out.
Newest entries at the bottom of the log. **Results tables are generated, never typed** — regenerate them with

```bash
python scripts/datalog_tables.py output > /tmp/tables.md      # on the machine that holds output/
```

and paste the output over the "Results" section. Raw artifacts live on the lab machine (`sun@lab`) under
`~/Desktop/aria_proj/egocentric_splats/output/`; job logs and queue scripts under `~/contactsplat_runs/`.

## Conventions

- **Floor metrics** (`photogrammetry/evaluate_mesh.py`): a 10 cm grid over the walked path buffered by 0.75 m; a cell is
  *covered* if a downward ray hits within ±15 cm of the expected floor. Expected floor = trajectory height minus the
  **glasses-to-floor height**, which is measured per recording (see 2026-10-03), not assumed.
- **Splat metrics**: mean PSNR / SSIM / LPIPS over the held-out eighth of frames.
- **Drive test** (`sim_drive_test.py`, from commit `badf881`): Nova Carter in Isaac Sim 5.1 follows the recorded walk
  (waypoints every 0.25 m, 0.8 m/s, triangle-mesh collider). One deterministic run per mesh unless stated.
- Wearer A is 6′1″, wearer B is 5′10″.

## Results (generated 2026-10-04 23:15 by `scripts/datalog_tables.py`)

### Recordings

| recording | environment | wearer | posed frames | glasses-to-floor (measured) |
|---|---|---|---|---|
| Outside | outdoor | A (6'1") | 4675 | 1.667 m |
| Hall | indoor | A (6'1") | 5698 | 1.667 m |
| Room | indoor | A (6'1") | 8221 | 1.667 m |
| park3_0 | outdoor | B (5′10″) | 702 | 1.588 m |
| park3_1 | outdoor | B (5′10″) | 686 | 1.583 m |
| park3_2 | outdoor | B (5′10″) | 731 | 1.586 m |
| park4_0 | outdoor | B (5′10″) | 668 | 1.615 m |
| park4_1 | outdoor | B (5′10″) | 741 | 1.600 m |
| park4_2 | outdoor | B (5′10″) | 713 | 1.596 m |
| park7_0 | outdoor | B (5′10″) | 603 | 1.590 m |
| park7_1 | outdoor | B (5′10″) | 598 | 1.593 m |
| park7_2 | outdoor | B (5′10″) | 640 | 1.589 m |

### Splat quality (held-out frames)

| recording | variant | Gaussian cap | Gaussians (final) | PSNR | SSIM | LPIPS | test frames |
|---|---|---|---|---|---|---|---|
| Outside | RS on | 1.5 M | 1.50 M | 25.45 | 0.845 | 0.390 | 585 |
| Outside | RS off | 1.5 M | 1.50 M | 25.33 | 0.843 | 0.393 | 585 |
| Hall | RS on | 1.5 M | 1.50 M | 31.52 | 0.871 | 0.585 | 713 |
| Room | RS on | 1.5 M | 2.63 M | 29.87 | 0.892 | 0.521 | 1028 |
| park3_0 | RS on | 1.5 M | 1.50 M | 32.05 | 0.867 | 0.605 | 88 |
| park3_1 | RS on | 1.5 M | 1.50 M | 32.32 | 0.868 | 0.592 | 86 |
| park3_2 | RS on | 1.5 M | 1.50 M | 30.95 | 0.862 | 0.605 | 92 |
| park4_0 | RS on | 1.5 M | 1.50 M | 32.27 | 0.869 | 0.642 | 84 |
| park4_0 | RS off | 1.5 M | 1.50 M | 31.55 | 0.865 | 0.650 | 84 |
| park4_1 | RS on | 1.5 M | 1.50 M | 31.76 | 0.867 | 0.629 | 93 |
| park4_2 | RS on | 1.5 M | 1.50 M | 31.42 | 0.865 | 0.654 | 90 |
| park7_0 | RS on | 1.5 M | 1.50 M | 30.21 | 0.836 | 0.668 | 76 |
| park7_1 | RS on | 1.5 M | 1.50 M | 30.61 | 0.839 | 0.659 | 75 |
| park7_2 | RS on | 1.5 M | 1.50 M | 30.82 | 0.841 | 0.646 | 80 |

### Collision meshes (scored with each recording's measured glasses-to-floor height)

| recording | method | triangles | floor coverage | largest hole (m²) | floor height error, median (cm) |
|---|---|---|---|---|---|
| Outside | MVS | 2,480,749 | 0.880 | 1.44 | 3.3 |
| Outside | MVS · decimated | 314,204 | 0.888 | 1.40 | 3.3 |
| Outside | MPS points | 471,360 | 0.873 | 2.42 | 3.7 |
| Outside | MPS points · decimated | 52,397 | 0.879 | 2.39 | 3.5 |
| Hall | MVS | 1,514 | 0.137 | 13.70 | 7.6 |
| Hall | MVS · decimated | 460 | 0.092 | 15.28 | 7.6 |
| Hall | MPS points | 312,180 | 0.414 | 6.69 | 5.4 |
| Hall | MPS points · decimated | 15,926 | 0.412 | 6.81 | 5.1 |
| Room | MVS | 22,178 | 0.083 | 18.16 | 8.2 |
| Room | MVS · decimated | 2,529 | 0.075 | 18.33 | 7.6 |
| Room | MPS points | 560,971 | 0.318 | 6.78 | 6.4 |
| Room | MPS points · decimated | 22,124 | 0.328 | 6.69 | 6.1 |
| park3_0 | MPS points | 173,246 | 0.920 | 1.09 | 2.7 |
| park3_0 | MPS points · decimated | 29,180 | 0.922 | 1.08 | 2.6 |
| park3_1 | MPS points | 174,671 | 0.921 | 0.91 | 2.7 |
| park3_1 | MPS points · decimated | 29,638 | 0.923 | 0.92 | 2.5 |
| park3_2 | MPS points | 193,178 | 0.898 | 1.44 | 2.9 |
| park3_2 | MPS points · decimated | 30,704 | 0.899 | 1.45 | 2.8 |
| park4_0 | MVS | 830,542 | 0.938 | 1.00 | 2.1 |
| park4_0 | MPS points | 164,902 | 0.934 | 0.99 | 2.6 |
| park4_0 | MPS points · decimated | 25,307 | 0.935 | 1.00 | 2.5 |
| park4_1 | MPS points | 190,887 | 0.945 | 0.86 | 3.3 |
| park4_1 | MPS points · decimated | 26,385 | 0.949 | 0.86 | 3.3 |
| park4_2 | MVS | 706,744 | 0.925 | 1.06 | 2.5 |
| park4_2 | MPS points | 168,771 | 0.918 | 1.03 | 2.9 |
| park4_2 | MPS points · decimated | 25,881 | 0.919 | 1.04 | 2.9 |
| park7_0 | MPS points | 176,726 | 0.936 | 0.89 | 2.5 |
| park7_0 | MPS points · decimated | 25,040 | 0.940 | 0.92 | 2.4 |
| park7_1 | MPS points | 153,220 | 0.892 | 1.25 | 2.7 |
| park7_1 | MPS points · decimated | 21,413 | 0.894 | 1.25 | 2.6 |
| park7_2 | MPS points | 164,362 | 0.920 | 1.16 | 2.7 |
| park7_2 | MPS points · decimated | 21,780 | 0.923 | 1.17 | 2.6 |

## Log

**2026-08-12 · Outside recorded.** Aria Gen 2, one outdoor walk, 4,675 posed frames, 90.6 m path.

**2026-08-16 → 08-26 · First full pass (Outside).** 3DGS and 2DGS trained at full 2016×1512; five TSDF collision
candidates (3DGS depth, 2DGS depth, stereo depth, trajectory floor prior, fused) scored and drive-tested. Drive results
were single deterministic runs and are recorded only as prose in `docs/experiments/final_report_content.md`
(removed in `6ab552e`; recover with `git show 6ab552e^:docs/experiments/final_report_content.md`); no drive-report
JSON survives. All August artifacts are in the lab machine's Trash (`~/.local/share/Trash/files/output`, trashed
2026-09-15) — recoverable, not yet restored.

**2026-09-27 → 10-02 · MVS pipeline and new recordings.** COLMAP MVS on fixed MPS poses (`photogrammetry/`), Isaac Sim
USDZ export with a hidden collider, optional decimation. New recordings: Hall and Room (indoor, wearer A) and nine
parking-lot walks (wearer B). Splats trained; MVS meshes for Outside, Hall, Room, park4_0, park4_2; MPS-points meshes
(`mps-mesh/`, semi-dense points straight to the Delaunay mesher) for ten scenes. Three trainings (Room, park4_2,
park7_0) died about a minute in with no log.

**2026-10-03 · Glasses-to-floor height measured per recording.** From MPS semi-dense points within 0.35 m of each
0.25 m trajectory sample, the median drop from the device to the points 1.0–2.2 m below. Outside gives **1.667 m**,
matching August's independent mesh-based calibration (1.6683 m) to 1 mm. Parking walks: **1.583–1.615 m**. The
7.7 cm gap matches the wearers' 7.6 cm height difference. Indoors the method is unreliable (furniture near the path:
Room gives 1.295 m), so indoor recordings use wearer A's outdoor value, 1.667 m. All meshes rescored: parking floor
error fell from 5.2–7.1 cm to 2.1–3.3 cm and coverage rose 2–5 points. Original reports kept alongside.

**2026-10-03 · Training crash found and fixed.** `~/.bashrc` puts CUDA 13.0 on `PATH` only for interactive shells; a
background job got `/usr/bin/nvcc`, which cannot find `cuda_runtime.h`, so gsplat's kernel rebuild failed. Queued jobs
now export `CUDA_HOME`, `PATH` and `LD_LIBRARY_PATH`. Results: park7_0, park4_2 and park7_2 trained (park7_2
preprocessed for the first time); Room ran out of memory at step 6,726 with a 1.5 M Gaussian cap.

**2026-10-03 · Drive test hangs.** The August script (copied from `badf881` into `~/contactsplat_runs/drive/`) spawned
and settled the robot on the Outside MPS-points mesh and planned 363 waypoints over 90.64 m, but simulated time never
reached 10 s in 44 minutes (the report writes every 10 simulated seconds). World reset took 43 s, against 7.6 s for
August's 4.12 M-triangle mesh. Physics stepping on the new mesh is the suspect; diagnostic queued.

**2026-10-04 · Room retried at a 1.0 M Gaussian cap: out of memory again**, at step 7,184 (14.0 GB allocated by
PyTorch on the 16 GB card), in the backward pass through gsplat. Both failures sit near the end of the first pass over
the 8,221 frames, so a few close-range indoor frames are the likely peak. Lowering the cap is not enough on 16 GB.

**2026-10-04 · Ablation: decimating the MPS-points meshes.** All twelve decimated to 11–17 % of their triangles at
~1 cm tolerance and rescored: coverage and largest hole change by at most 0.01.

**2026-10-04 · Ablation queue armed** (starts when Room finishes): drive-test diagnostic (new mesh vs August's, 3
simulated seconds each); re-preprocess six parking walks; MVS on the seven parking walks that lack it; rolling-shutter
modelling **off** on Outside and park4_0 (Lv et al.'s contribution); Meta's Gen 1 sample through the Gen 2 port.

**2026-10-04 · Drive-test diagnostic: the hang is the new mesh, not the Isaac Sim setup.** Same script, robot and
Outside trajectory, 3 simulated seconds each (`~/contactsplat_runs/drive/diag/`):

| collision mesh | triangles | world reset | 3 simulated s | physics step, mean |
|---|---|---|---|---|
| August fused (stereo + 2DGS + floor) | 4,120,799 | 2.8 s | done in 3.5 s wall | 19.2 ms |
| Outside MPS points (Delaunay) | 471,360 | 2.1 s | not reached in 20 min | > 6 s (est.) |

Both robots spawned and settled (ground clearance 1.4 and 1.5 cm). On August's mesh the robot reached 8 waypoints
(2.0 m) with no stalls. The 43 s world reset seen on 10-03 did not recur (2.1 s), so it was not the cause.
(Correction, same day: per-step timing below shows normal steps followed by one step that never returns, not
uniformly slow steps.)

**2026-10-04 · Drive-test hang narrowed down (Outside).** Per-step timing printed every 10 steps; a stack dump fires
if one physics step exceeds 60 s. Every hang is inside PhysX `simulate()`, at a fixed place for a given mesh:

| collision mesh | triangles | normal step | hangs at | reproducible |
|---|---|---|---|---|
| MPS points, full | 471,360 | ~8 ms (trimmed copy) | 0.9 s sim, near (−0.95, 1.61) | — |
| MPS points, decimated | 52,397 | ~3 ms | 6.0 s sim, waypoint 16, (−0.14, −1.17) | yes, same step with CCD off |
| MVS, decimated | 314,204 | ~10 ms | 33.8 s sim, waypoint 73, (3.13, 3.37) | — |
| MPS points, decimated, non-manifold edges removed | 50,905 | ~2.5 ms | passed waypoint 16; frozen at waypoint 28 from 17 s, then hung in wedge recovery | — |

Turning continuous collision detection off changed nothing. Mesh quality is the clearest difference from August's
mesh, which drives:

| mesh | edges shared by 3+ triangles | edges with inconsistent winding |
|---|---|---|
| August fused (TSDF) | 0 | 0 |
| MPS points, decimated | 839 | 1,672 |
| MPS points, full minus >0.5 m edges | 2,919 | 5,807 |
| MVS, decimated | 1,997 | 3,976 |

Each hang point has non-manifold edges within 0.5 m. Removing them (Open3D) moved the robot past the first hang, so
they are part of the cause but not all of it; 202 winding flips remain (the mesh is not orientable). Triangle shape
near the first hang point is unremarkable (no slivers or duplicates). Open: whether the drive test should use repaired
meshes, PhysX's SDF collision mode, or a re-meshed surface (TSDF / Poisson) for the Delaunay meshes. Drive tests on
the remaining scenes are on hold until that is decided.

**2026-10-04 · Room sent to a cloud GPU.** RunPod A40 (48 GB, $0.50/hr, on-demand), raw VRS + MPS uploaded from the
lab, preprocessed and trained there with the lab's exact code (`acbd483`, branch `mps-mesh`) at the full 1.5 M cap.
Software differs from the lab: Python 3.12, torch 2.8.0+cu128, CUDA 12.8, gsplat 1.5.3 (same), projectaria-tools 2.2.0
(same). Record the GPU wherever Room's splat is quoted.

**2026-10-04/05 · Room trained (cloud A40).** Preprocessing on the pod took 2 h 56 min (rectifying 8,221 full-resolution
frames on 9 vCPUs); training 30,000 steps took 2 h 53 min including test rendering, at 3–4 steps/s, peak GPU memory
under 9 GB of 46. Held-out frames (n = 1,028): **PSNR 29.87, SSIM 0.892, LPIPS 0.521**. Final model: **2,625,293
Gaussians, above the 1.5 M cap**: Room's MPS output has 4.66 M semi-dense points, so the filtered initial cloud already
exceeded the cap, and MCMC's cap limits growth but never removes Gaussians. That is the likely reason lowering the cap
to 1.0 M did not stop the lab's out-of-memory failures. Results (PLY, `test_logs.json`, `cameras.json`, `cfg_args`,
test renders and depth, pod logs) are on the Mac in `contactsplatai/runpod_results/Room_20260929_211650/`, PLY checksum
verified; copied to the lab's `output/` on 10-04 23:10 (checksum matches; the failed 1.0 M run's `cfg_args` and
`cameras.json` kept alongside with a `.lab-failed-cap1M` suffix). Pod terminated. Cost $3.62.

**2026-10-04 · Lab disk full; queue 2 lost.** At 11:23 the lab's 908 GB disk reached 0 bytes free. park3_0 and
park3_1 re-preprocessed; park3_2 failed partway (images written, no `transforms_with_sparse_depth.json`); every later
step failed instantly. The largest project folders: MVS workspaces 70 GB, recordings 165 GB (Room 42, parking 44),
`lightning_logs` 12 GB. Freed 31 GB by clearing the pip download cache and conda's package tarballs (re-downloadable;
no project data touched). Queue 6 (replacing short-lived queues 3–5) checks for 8 GB free before every step and runs:
park4_0 preprocessing, rolling-shutter-off on park4_0 and Outside, the Gen 1 sample, then MVS on park3_0, park3_1,
park7_0 and park7_2.

**2026-10-04 · Queue 6 results (lab, unattended 12:10–21:34).**
- **Rolling-shutter modelling off** (Lv et al.'s contribution; otherwise identical settings, single runs): Outside
  PSNR 25.45 → 25.33 (−0.12 dB), LPIPS 0.390 → 0.393; park4_0 PSNR 32.27 → 31.55 (−0.72 dB), LPIPS 0.642 → 0.650.
  Modelling rolling shutter helps on both, more on the short parking walk. Two scenes, one run each: an observation, not
  a significance claim.
- **Meta's Gen 1 sample through the Gen 2 port**: trains and scores normally (PSNR 28.95, SSIM 0.944, LPIPS 0.308,
  39 test frames, 1200 × 2400 rectification). The port did not break Gen 1 input.
- **Gaussian cap check**: every scene except Room ends at exactly 1.50 M Gaussians, so the cap binds there. Room
  (2.63 M) is the only scene whose initial cloud exceeded the cap. The table script now reads the cap from each run's
  `cfg_args` and the final count from its PLY instead of hard-coding them.
- **MVS on park3_0, park3_1, park7_0 failed instantly**: their workspaces date from 10-01 and keep `.done` markers for
  stages whose outputs (`database.db`, `dense/`) no longer exist, so the script skips to stereo and COLMAP aborts
  reading an empty workspace. They need a fresh workspace.
- **MVS on park7_2 stopped after stereo (77 min)**: 279 of 280 depth maps. COLMAP skipped one frame
  (`camera-rgb_678777965643.png`) because it has no source images, which is normal; the pipeline's check requires every
  frame, so fusion and meshing never ran. The 6.4 GB stereo workspace is kept. Lab disk now 9 GB free.

## Findings so far

1. **Outdoors, MPS-points meshes match MVS** (coverage 0.873 vs 0.880 on Outside, 0.934 vs 0.938 on park4_0, 0.918 vs
   0.925 on park4_2) with about a fifth of the triangles and no dense stereo step.
2. **Indoors, MVS collapses** (coverage 0.08–0.14 on Hall and Room): blank floors give patch-match nothing to match.
   MPS points do better (0.32–0.41) but not well enough to drive on.
3. **Decimation is free outdoors**: 8× fewer triangles on the Outside MVS mesh and 6–9× on MPS-points meshes with no
   loss in coverage or hole size.
4. **The glasses-to-floor height must be measured per recording.** A single constant made every parking mesh look 5–7 cm
   off; measured per walk, the error is 2–3 cm.
5. **Splat PSNR flatters the parking walks**: PSNR 30–32 but LPIPS 0.59–0.67, against 0.39 on Outside. Look at renders
   before quoting PSNR.
6. **Rolling-shutter modelling helps a little**: turning it off cost 0.12 dB on Outside and 0.72 dB on park4_0 (single
   runs).
7. **The new Delaunay meshes do not yet survive the drive test**: PhysX hangs at fixed spots on meshes with
   non-manifold edges; August's TSDF mesh, which has none, drives.

## Open

- Drive test on the new meshes: PhysX hangs on the Delaunay meshes, which have non-manifold edges and inconsistent
  winding (see "hang narrowed down"). No drive result yet exists for any recording made after August.
- August artifacts still in the Trash.
- MVS check in `photogrammetry/run_photogrammetry.sh` fails a run when COLMAP legitimately skips a frame (park7_2).
  Decide whether to accept COLMAP-skipped frames; then park7_2 needs only fusion and meshing.
- Stale 10-01 MVS workspaces for park3_0, park3_1 and park7_0 must be moved aside before rerunning.
- Lab disk space (9 GB free): each parking MVS needs ~7 GB; re-preprocessing park3_2, park4_1 and park7_1 ~15 GB.
