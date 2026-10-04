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

## Results (generated 2026-10-04)

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

| recording | variant | Gaussian cap | PSNR | SSIM | LPIPS | test frames |
|---|---|---|---|---|---|---|
| Outside | RS on | 1.5 M | 25.45 | 0.845 | 0.390 | 585 |
| Hall | RS on | 1.5 M | 31.52 | 0.871 | 0.585 | 713 |
| park3_0 | RS on | 1.5 M | 32.05 | 0.867 | 0.605 | 88 |
| park3_1 | RS on | 1.5 M | 32.32 | 0.868 | 0.592 | 86 |
| park3_2 | RS on | 1.5 M | 30.95 | 0.862 | 0.605 | 92 |
| park4_0 | RS on | 1.5 M | 32.27 | 0.869 | 0.642 | 84 |
| park4_1 | RS on | 1.5 M | 31.76 | 0.867 | 0.629 | 93 |
| park4_2 | RS on | 1.5 M | 31.42 | 0.865 | 0.654 | 90 |
| park7_0 | RS on | 1.5 M | 30.21 | 0.836 | 0.668 | 76 |
| park7_1 | RS on | 1.5 M | 30.61 | 0.839 | 0.659 | 75 |
| park7_2 | RS on | 1.5 M | 30.82 | 0.841 | 0.646 | 80 |

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

**2026-10-04 · Room retried at a 1.0 M Gaussian cap** (running). Report Room's lower cap wherever its splat is quoted.

**2026-10-04 · Ablation: decimating the MPS-points meshes.** All twelve decimated to 11–17 % of their triangles at
~1 cm tolerance and rescored: coverage and largest hole change by at most 0.01.

**2026-10-04 · Ablation queue armed** (starts when Room finishes): drive-test diagnostic (new mesh vs August's, 3
simulated seconds each); re-preprocess six parking walks; MVS on the seven parking walks that lack it; rolling-shutter
modelling **off** on Outside and park4_0 (Lv et al.'s contribution); Meta's Gen 1 sample through the Gen 2 port.

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

## Open

- Drive test on the new meshes (hang under diagnosis). No drive result yet exists for any recording made after August.
- August artifacts still in the Trash.
- Room splat at 1.0 M cap (running).
