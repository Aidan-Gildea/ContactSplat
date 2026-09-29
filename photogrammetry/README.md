# photogrammetry — COLMAP mesh with MPS poses fixed

Consumes the posed, rectified RGB dataset that trunk produces and turns it into a
triangle mesh in the **MPS world frame** (gravity-aligned, Z-up, metres). Because the
trained Gaussian splat lives in that same frame, the mesh and the splat coexist in a
simulator with no registration step.

```
processed/<recording>/camera-rgb-rectified-<focal>-h<height>/     (trunk output)
        │  transforms.json + images/
        ▼
photogrammetry/run_photogrammetry.sh
        │  keyframes → SIFT → sequential match → triangulate (poses FIXED)
        │           → patch-match stereo → fusion → Delaunay mesh
        ▼
output/photogrammetry/<recording>/<rectified folder>/mesh_delaunay.ply   + report.md
```

## Why COLMAP's pose solver is bypassed

MPS already produced bundle-adjusted, gravity-aligned, metric poses from four global-shutter
SLAM cameras plus an IMU. Monocular structure-from-motion would return poses in an
arbitrary frame at an arbitrary scale, and the mesh would no longer line up with the
splat. So `mapper` is never run. The pipeline uses only COLMAP's *second half*:

1. `cameras.txt` / `images.txt` are written directly from `transforms.json`. The rectified
   frames are exact pinhole images, so they map onto COLMAP's `PINHOLE` model with no
   distortion parameters. The principal point is shifted by half a pixel because
   projectaria_tools centres the top-left pixel at (0, 0) and COLMAP at (0.5, 0.5).
2. `feature_extractor` + `sequential_matcher` find 2D correspondences.
3. `point_triangulator` triangulates those correspondences **against the fixed poses**.
   Intrinsics are held constant too. This yields the sparse model dense stereo needs
   (per-image depth ranges and source-image selection).
4. `image_undistorter` → `patch_match_stereo` → `stereo_fusion` → `delaunay_mesher`.

## Usage

```bash
conda activate ego_splats                       # numpy, pandas, open3d; colmap must be on PATH
cd /path/to/egocentric_splats-photogrammetry

bash photogrammetry/run_photogrammetry.sh \
    "$OUT_ROOT/$SCENE/camera-rgb-rectified-1008-h1512"          # out_dir optional 2nd arg
```

Every stage is skipped when it has already completed, so a run can be resumed after an
interruption. `FORCE=1` redoes everything.

Check the sparse alignment **before** committing GPU hours:

```bash
STOP_AFTER=triangulate bash photogrammetry/run_photogrammetry.sh <rectified_dir>
cat output/photogrammetry/<recording>/<rect>/logs/triangulate_stats.log
```

A mean reprojection error of about one pixel with most keyframes holding hundreds of
observations means the poses and the convention are right. Garbage there (few points,
errors of many pixels) means something upstream is wrong; do not proceed to stereo.

Restrict to a time window (seconds from the first posed frame), e.g. the one minute the
wearer spent looking at the floor:

```bash
START=30 DURATION=60 bash photogrammetry/run_photogrammetry.sh <rectified_dir> <out_dir>
```

### Knobs

| Variable | Default | Effect |
|---|---|---|
| `START`, `DURATION` | `0`, `-1` | time window in seconds; `-1` runs to the end |
| `MIN_TRANSLATION`, `MIN_ROTATION_DEG` | `0.10`, `10` | keyframe spacing (a frame is kept when either is exceeded) |
| `MAX_KEYFRAMES` | `-1` | uniform thinning cap |
| `MAX_IMAGE_SIZE` | `1008` | longest side for dense stereo; `2016` is native and roughly 4x slower |
| `NUM_SRC_IMAGES` | `20` | patch-match source views per reference view |
| `SEQ_OVERLAP` | `20` | sequential matching window; powers of two beyond it are matched too |
| `LOOP_DETECTION` | `0` | `1` enables vocabulary-tree loop detection (downloads the tree) |
| `GPU_INDEX` | `0` | CUDA device |
| `CACHE_GB` | `16` | image cache for stereo and fusion |
| `EYE_HEIGHT` | `1.6683` | trajectory-to-floor distance used by the evaluator |
| `STOP_AFTER` | | stop after `prepare`, `features`, `match`, `triangulate`, `undistort`, `stereo`, `fuse` or `mesh` |
| `FORCE` | `0` | rerun every stage |
| `PYTHON` | `python` | interpreter for the two helper scripts |

## Outputs

```
output/photogrammetry/<recording>/<rectified folder>/
├── keyframes.txt, keyframes.json      which frames were used and why
├── sparse_in/                         cameras.txt / images.txt written from MPS poses
├── database.db                        SIFT features + matches
├── sparse/                            triangulated sparse model (bin + txt)
├── dense/                             COLMAP dense workspace; dense/fused.ply is the MVS cloud
├── mesh_delaunay.ply                  the deliverable, MPS world frame
├── report.json, report.md             evaluation
├── logs/<stage>.log                   one log per stage
└── .done/<stage>                      completion markers (delete one to redo that stage)
```

## Reading the report

`evaluate_mesh.py` scores the mesh against the two independent geometry sources MPS
provides, in the same frame:

- **floor coverage / largest hole** — 10 cm cells within 0.75 m of the walked path,
  downward ray from 0.5 m above the expected floor (trajectory height minus `EYE_HEIGHT`),
  hit within ±15 cm counts. The largest hole is the biggest connected block of misses.
  This is the collision-relevant number. For a windowed run only the part of the walk
  covered by the selected keyframes is scored.
- **semi-dense agreement** — point-to-mesh distance for confidence-filtered semi-dense
  points (same thresholds as `extract_aria_vrs.py`), reported for all points and for the
  points inside the mesh's bounding box. The second isolates accuracy from coverage.
- **fused asymmetry** — how much of the fused MVS cloud sits in the floor band versus
  above it. Patch-match needs texture, and untextured ground is exactly where it returns
  nothing, so this number says how much floor survived `stereo_fusion` before meshing.
- **hygiene** — triangle count, bounds, connected components, and the area in
  components smaller than 0.1 m² (floating junk that hurts collision performance).

## Known limits

- Frames are ordered for sequential matching by file name, which embeds the device
  timestamp. That is chronological as long as every timestamp has the same digit count.
- Delaunay meshing on a very large fused cloud is CPU and RAM bound; if it runs out of
  memory, raise `MIN_TRANSLATION` or lower `MAX_IMAGE_SIZE`.
- Spatial matching from the known poses (COLMAP's `spatial_matcher` with pose priors)
  would find cross-visit matches more reliably than sequential matching; it needs the
  priors written into the database and is not done here.
