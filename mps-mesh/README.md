# mps-mesh — Delaunay mesh from the MPS semi-dense points

Same mesher and same evaluator as `photogrammetry/`, different points. Instead of running
patch-match MVS on the RGB keyframes, the confident MPS semi-dense points are handed
straight to `colmap delaunay_mesher`, together with the SLAM-camera views that observed
each point. The result is in the MPS world frame (gravity-aligned, Z-up, metres), so it
lines up with the splat and with the photogrammetry mesh with no registration step.

```
processed/<recording>/camera-rgb-rectified-<focal>-h<height>/     (trunk output)
        │  semidense_points.csv.gz symlink → MPS slam/ folder
        ▼
mps-mesh/run_mps_mesh.sh
        │  prepare: confident semi-dense points + their SLAM-camera views → COLMAP dense workspace
        │  mesh:    colmap delaunay_mesher (default options)
        │  evaluate: photogrammetry/evaluate_mesh.py
        ▼
output/mps-mesh/<recording>/<rectified folder>/mesh_delaunay.ply   + report.md
```

## What `prepare` writes

`mps_points_to_colmap.py` reads four files from the MPS `slam/` folder, which it finds by
following the rectified folder's `semidense_points.csv.gz` symlink:

| MPS file | used for |
|---|---|
| `semidense_points.csv.gz` | point positions; kept if `inv_dist_std < 0.005` and `dist_std < 0.01` (the trunk's filter) |
| `semidense_observations.csv.gz` | which SLAM camera saw which point at which timestamp |
| `closed_loop_trajectory.csv` | device pose at each observation timestamp (nearest 1 kHz sample) |
| `online_calibration.jsonl` | `T_Device_Camera` and focal length per SLAM camera (first record) |

It writes them in the layout `stereo_fusion` produces, so the mesher cannot tell the
difference:

```
dense/fused.ply          the points
dense/fused.ply.vis      for each point, the indices of the views that observed it
dense/sparse/            one image per (timestamp, SLAM camera) view; points3D.txt empty
mps_points.json          counts: points kept, observations, views per camera
```

The mesher uses the views to carve free space: the segment from a camera to a point it saw
must be empty. It uses the camera model only to decide which nearby points to merge, so
each fisheye SLAM camera is written as a `SIMPLE_PINHOLE` with the same focal length and
principal point.

## Usage

```bash
conda activate ego_splats                       # numpy, pandas, open3d; colmap must be on PATH
bash mps-mesh/run_mps_mesh.sh "$OUT_ROOT/$SCENE/camera-rgb-rectified-1008-h1512"   # out_dir optional 2nd arg
```

The MPS `slam/` folder must still hold `semidense_observations.csv.gz` and
`online_calibration.jsonl` next to the points file. Each stage is skipped when it has
already completed; `FORCE=1` redoes everything. Other knobs: `EYE_HEIGHT` (default
`1.6683`, passed to the evaluator) and `PYTHON`.

## Reading the report

`report.md` has the same rows as the photogrammetry report, so the two can sit side by side.
The "fused" rows describe the MPS cloud that was meshed. Two caveats when comparing:

- **Semi-dense agreement is not independent here.** The evaluator measures the mesh against
  the same confident semi-dense points the mesh was built from. Floor coverage and largest
  hole come from the walked trajectory and remain independent.
- **Floor span.** The photogrammetry evaluator scores the trajectory span its keyframes
  cover; this one scores the whole recording. With the default keyframe settings the two
  spans are nearly the same.
