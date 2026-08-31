# Mesh extraction experiments — delegation briefs

Goal: produce a **collision mesh** for the Gen 2 "Outside" scene that lives in the same
world frame as the trained Gaussian splat, so Isaac Sim can render the splat and collide
against the mesh with no registration step.

Key facts every agent needs (repeated in each brief so they're self-contained):

| | |
|---|---|
| Repo | `/home/sun/Desktop/aria_proj/egocentric_splats` |
| Scene | `Outside_20260812_141244` (Aria **Gen 2**, 2:37, 100.2 m, 4,719 RGB frames) |
| Raw | `/home/sun/aria/Outside_20260812_141244.vrs` |
| MPS | `/home/sun/aria/mps_Outside_20260812_141244_vrs/slam/` |
| Preprocessed | `/home/sun/aria/processed/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/` |
| Trained PLY | `output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/point_cloud/iteration_30000/point_cloud.ply` |
| Envs | `ego_splats` (preprocess + train, CUDA), `3dgrut` (USDZ export only) |
| GPU | single RTX 4080 SUPER, **16 GB** — GPU tasks must be serialized |

**Shared invariant:** the MPS world frame is gravity-aligned, Z-up, metric metres. Every
output mesh MUST be in that frame. Any pipeline that solves its own poses is wrong.

---

## Order of execution

`E0` and `E1` first, alone. Then `E2`–`E6`, but **serialize anything touching the GPU**
(E2 retrain, E3 stereo, E4 COLMAP dense) — there is one 16 GB card and the existing
3DGS run already needed `data_factor=2` + `cap_max=1.5M` to fit.

`E5` (trajectory floor) is pure CPU/numpy and can run alongside anything.

---

## E0 — Evaluation harness (build this FIRST, nothing else is comparable without it)

> You are working in `/home/sun/Desktop/aria_proj/egocentric_splats` on an Aria Gen 2
> reconstruction project. Several different pipelines are going to produce candidate
> collision meshes for the same scene, and I need an objective way to rank them.
>
> Write `scripts/eval_mesh.py`. It takes a mesh file (PLY/OBJ) plus the MPS directory
> `/home/sun/aria/mps_Outside_20260812_141244_vrs/slam/` and emits a JSON scorecard with:
>
> 1. **Floor coverage.** Read `closed_loop_trajectory.csv` (MPS closed-loop, 1 kHz,
>    gravity-aligned Z-up metres — this is the ground truth for where a human walked).
>    Project the trajectory to the XY plane, buffer it laterally by 0.75 m, and rasterize
>    to a 10 cm grid. For each cell, raycast straight down from 0.5 m above the expected
>    floor height (trajectory Z minus an assumed eye height of 1.6 m). Report the fraction
>    of cells that return a hit within ±15 cm of expected.
> 2. **Largest hole.** Connected-component the *missing* cells from (1); report the area
>    of the largest component in m². A robot doesn't care about many small holes; it cares
>    about the one it falls through.
> 3. **Agreement with MPS semi-dense points.** Load `semidense_points.csv.gz` via
>    `projectaria_tools` (`mps.read_global_point_cloud`), filter by the existing confidence
>    thresholds used in `scripts/extract_aria_vrs.py` (`filter_points_from_confidence`),
>    then report mean/median/p95 point-to-mesh distance. This is an independent geometry
>    source, so it's the closest thing to ground truth available.
> 4. **Mesh hygiene.** Triangle count, bounding box, non-manifold edge count, number of
>    disconnected components, and total area of components smaller than 0.1 m² (floater
>    junk that will wreck collision performance).
>
> Use Open3D and numpy. Make it fast enough to run on a multi-million-triangle mesh.
> Also write `scripts/compare_meshes.py` that runs the above over several meshes and
> prints a comparison table.
>
> Verify it works by running it on a trivially-constructed test mesh (e.g. a flat plane at
> the expected floor height spanning the trajectory) so I can see the metrics behave
> sensibly at the known-good and known-bad extremes. Report the numbers you get.

---

## E1 — Isaac Sim units / orientation (diagnose, don't paper over)

> In `/home/sun/Desktop/aria_proj/egocentric_splats`, the file
> `scripts/bash_local/export_gen2_outside_usdz.sh` claims the exported USDZ is
> "Z-up in metres, matching the MPS world frame, so it needs no extra transform."
> But in practice, when the asset is loaded into Isaac Sim it appears **very small and
> incorrectly oriented**, and has to be found with `f` and fixed with a manual rotation.
>
> One of those two things is wrong. Find out which. Do not add a hardcoded fudge transform
> until you know the root cause.
>
> Strong hypothesis to test first: a `metersPerUnit` mismatch. USD stages carry a
> `metersPerUnit` value, and Isaac Sim's default stage is frequently centimetres (0.01)
> while a converted asset declares metres (1.0) — which makes the asset render 100× too
> small. `scripts/isaacsim_load_splat.py` **already prints both `upAxis` and
> `metersPerUnit`** for the stage. Run it and read those two lines:
>
>     ~/isaac-sim/python.sh scripts/isaacsim_load_splat.py \
>         output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/isaacsim/Outside_20260812_141244.usdz \
>         --report /tmp/usd_report.json
>
> Note the docstring's warning: Kit swallows stdout and force-exits, so use `--report` and
> read the JSON rather than trusting printed output or exit codes.
>
> Then inspect the USDZ's own layer metadata directly with `pxr.Usd` (open the file as a
> stage without referencing it) and compare its declared `upAxis` / `metersPerUnit` /
> world-space bounds against the stage's. Also check what NVIDIA's
> `threedgrut/export/scripts/ply_to_usd.py` (in `/home/sun/3dgrut`) actually writes for
> those two values.
>
> Deliverable: a root-cause explanation, plus a **deterministic** fix at the correct layer
> — either export-side (set the right metadata in the conversion) or load-side (set the
> stage to match). Update the export script and/or `isaacsim_load_splat.py` accordingly,
> and correct the comment in the export script if its claim turns out to be false.

---

## E2 — TSDF mesh from rendered splat depth (baseline + 2DGS comparison)

> In `/home/sun/Desktop/aria_proj/egocentric_splats`, I need to extract a triangle mesh
> from a trained 3D Gaussian Splatting model. Gaussians are not a surface, so this is a
> separate extraction step: render a depth map from every training camera pose, fuse them
> into a TSDF volume, and run marching cubes.
>
> Write `scripts/extract_mesh_tsdf.py`:
> - Load a trained model the same way `render_lightning.py` does (it already dispatches on
>   `cfg.train_model` and reuses the scene/camera setup — follow that pattern rather than
>   inventing a new loader).
> - Render depth at each training camera. Note `conf/config.yaml` already has
>   `render.render_depth: True` and `depth_min`/`depth_max` clipping — reuse that path.
> - Before rendering, **filter the Gaussians**: `scripts/filter_splat_outliers.py` already
>   does radius filtering; add an `--min-opacity` flag to it. Low-opacity floaters punch
>   spurious holes in a TSDF, so the depth-render copy should be filtered much more
>   aggressively than the copy used for visualization. Keep them as two separate derived
>   assets from the one training run — do NOT degrade the visual asset.
> - Fuse with Open3D's `ScalableTSDFVolume` (start at `voxel_length=0.02`,
>   `sdf_trunc=0.08`) using each camera's `T_world_camera`. If memory blows up on ~4,700
>   frames, subsample frames and/or switch to VDBFusion.
> - Output the mesh in the **MPS world frame** (gravity-aligned Z-up metres) so it aligns
>   with the splat with no registration.
>
> Run it twice and compare with `scripts/eval_mesh.py`:
> 1. On the existing trained 3DGS PLY at
>    `output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/point_cloud/iteration_30000/point_cloud.ply`
> 2. On a fresh **2DGS** model. 2DGS is already implemented in this repo —
>    `model/GS2D_gsplat.py`, selected via `train_model=2dgs` (see `train_lightning.py:33`).
>    It uses flat disks instead of ellipsoids and returns real surface normals, so its
>    depth should be markedly crisper. Retrain by copying
>    `scripts/bash_local/train_gen2_outside.sh` to a `_2dgs` variant and changing
>    `train_model=3dgs` to `train_model=2dgs`.
>
> **VRAM warning:** one RTX 4080 SUPER, 16 GB. The existing 3DGS run needed
> `data_factor=2`, `opt.mcmc_strategy.cap_max=1500000`, `pcd_stride=2`, and
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to fit — read the comments in that
> training script, they explain exactly why. Expect 2DGS to need similar or tighter caps.
>
> Report both scorecards and your read on whether 2DGS is worth the retrain cost.

---

## E3 — Gen 2 stereo depth → TSDF mesh

> Aria Gen 2 has four CV cameras with 80° stereo overlap (up from 35° on Gen 1), which
> makes it usable as a genuine depth capture device. Meta ships a first-party tool for
> this: https://github.com/facebookresearch/projectaria_gen2_depth_from_stereo — it
> rectifies the front-left/front-right SLAM pair and runs FoundationStereo for zero-shot
> disparity, converting to metric depth.
>
> Set it up and run it on `/home/sun/aria/Outside_20260812_141244.vrs` with MPS data at
> `/home/sun/aria/mps_Outside_20260812_141244_vrs/slam/`. It needs its own conda env
> (Python 3.11, PyTorch + CUDA 12.8) and 2–4 GB VRAM. Do not disturb the existing
> `ego_splats` or `3dgrut` envs.
>
> Its outputs are rectified left images, uint16 millimetre depth PNGs, optional
> left-right consistency masks, and a JSON with per-frame intrinsics, `T_world_camera`,
> and `T_device_rectCam`. **`T_world_camera` comes from MPS closed-loop**, which is the
> same frame the Gaussian splat lives in — that's the whole point of this route.
>
> The tool stops at a Rerun point cloud; the meshing step is yours. Reuse the TSDF fusion
> from `scripts/extract_mesh_tsdf.py` (task E2) rather than writing a second one — factor
> the fusion into a shared module if needed. **Use the consistency masks** to reject
> unreliable pixels before fusing.
>
> Then quantify the accuracy envelope, because this is the key limitation: these are
> 512×512 mono cameras on a short baseline, and stereo depth error goes as
> σ_Z ≈ Z²·σ_d/(f·B). Read the actual baseline B from the two front SLAM camera extrinsics
> in `online_calibration.jsonl` and the rectified focal f from the tool's output JSON, then
> report expected depth error at 1, 2, 5, 10 and 20 m. I need to know at what range this
> route stops being trustworthy — I suspect it's good indoors and poor on a 15 m+ outdoor
> scene, but I want the number, not the intuition.
>
> Score the resulting mesh with `scripts/eval_mesh.py`.

---

## E4 — COLMAP dense MVS with poses fixed from MPS

> I want a photogrammetry mesh (the Polycam-equivalent route) for the Aria Gen 2 scene at
> `/home/sun/aria/processed/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/`.
>
> **Critical constraint: do NOT let COLMAP solve for camera poses.** MPS already produced
> bundle-adjusted, gravity-aligned, metric poses using four global-shutter cameras plus
> IMU — far better than monocular SfM would manage, and crucially in the same world frame
> as the trained Gaussian splat. If COLMAP solves its own poses you get an arbitrary frame
> at an arbitrary scale and the whole point is lost. Skip `mapper` entirely.
>
> The correct flow is COLMAP's *second half* only:
> 1. Build a `cameras.txt` / `images.txt` from the existing transforms JSON in the
>    preprocessed folder (`transforms_with_sparse_depth.json`). These frames are already
>    **rectified pinhole with rolling shutter compensated** by
>    `scripts/extract_aria_vrs.py`, so they map cleanly to COLMAP's `PINHOLE` model and
>    need no undistortion.
> 2. `feature_extractor` + `sequential_matcher` (sequential, not exhaustive — this is a
>    continuous walk).
> 3. `point_triangulator` against the fixed poses to build a sparse model.
> 4. `patch_match_stereo` → `stereo_fusion` → `delaunay_mesher` (and try `poisson_mesher`
>    for comparison).
>
> Practical notes: 4,719 frames at 30 fps is heavily redundant and dense MVS is expensive.
> Subsample to keyframes by pose delta (e.g. every 10 cm of translation or 10° of rotation)
> rather than by fixed stride. Single 16 GB GPU, shared with other work — check whether
> another GPU job is running before you start.
>
> Report honestly on the known failure mode: MVS needs texture, and blank concrete floors
> and untextured walls are exactly where patch-match returns nothing. Quantify how much of
> the floor survives `stereo_fusion` versus how much of the walls do — that asymmetry is
> the thing I actually need to know about this route.
>
> Score with `scripts/eval_mesh.py`.

---

## E5 — Trajectory-derived floor (CPU only, run this alongside anything)

> This one exploits a free signal. `closed_loop_trajectory.csv` in
> `/home/sun/aria/mps_Outside_20260812_141244_vrs/slam/` contains **155,800 poses at 1 kHz
> over 100.2 m of walking**. It's a dense metric record of where a human head was, and a
> human head is a roughly constant offset above the walkable floor. The MPS world frame is
> gravity-aligned, so "down" is exactly known.
>
> Therefore `floor_z(x,y) ≈ device_z(x,y) − h`, with h ≈ eye height (~1.5–1.7 m).
>
> This is valuable because it has **no holes and needs no observations** — it covers
> exactly the region the robot must drive, and it's a traversability guarantee, since a
> human physically walked there. It also follows ramps and grade changes automatically,
> unlike a single fitted ground plane (which matters for infrastructure inspection).
>
> Write `scripts/floor_from_trajectory.py`:
> - Load the closed-loop trajectory with `projectaria_tools`
>   (`mps.read_closed_loop_trajectory`) — see how `scripts/extract_aria_vrs.py` does it.
> - **Calibrate h rather than assuming it.** Fit it by comparing trajectory Z against
>   whatever floor geometry the other pipelines DID recover in well-observed regions.
>   Report the fitted value and its spread — a large spread means the person's gait or
>   posture varied and the prior is weaker than it looks.
> - Grid the trajectory into 10 cm XY cells, take a robust per-cell Z (median), and
>   dilate laterally to a configurable robot clearance (default 0.75 m).
> - Fill gaps between passes by interpolation; flag extrapolated cells separately from
>   observed ones in the output.
> - Emit **both** a triangle mesh and a heightfield (a 2D array + origin + cell size).
>   PhysX handles heightfields natively and they are far more robust for a wheeled robot
>   than triangle soup — no thin-triangle tunnelling, no seams to fall through.
>
> Be explicit about the honest limitation: a trajectory is a *curve*, not a surface. It
> gives floor height along the walked path, not across the whole room. So also implement a
> `--fuse-with <mesh>` mode that uses this as a **prior** over another pipeline's mesh:
> pin absolute floor height, fill holes, and reject floor points that disagree by more
> than a configurable threshold (default 10 cm).
>
> Score both the standalone floor and a fused result with `scripts/eval_mesh.py`.

---

## E6 — Isaac Sim collision setup + the drive test (the actual acceptance test)

> This is the test that decides which of the other experiments won. Everything else is a
> proxy metric; this is the requirement.
>
> In `/home/sun/Desktop/aria_proj/egocentric_splats`, extend
> `scripts/isaacsim_load_splat.py` (or write a sibling script) that loads **both**:
> - the NuRec splat USDZ under `/World/AriaSplat` — visual only, as it already does
> - a collision mesh under `/World/AriaCollision` — `visibility = invisible`, with
>   `UsdPhysics.CollisionAPI` applied
>
> Both are in the MPS world frame, so neither needs a registration transform. Resolve the
> units/orientation question with task E1 before trusting the alignment.
>
> **Do not use raw triangle-mesh collision.** A 2 cm-voxel TSDF mesh is easily >1M
> triangles and triangle collision against it will dominate the sim step. Set
> `physics:approximation` to `sdf` or `convexDecomposition` and benchmark both — report
> collision setup time and steady-state sim step time for each.
>
> Then build the acceptance test, `scripts/sim_drive_test.py`:
> - Spawn a simple wheeled robot (a Carter or equivalent from the Isaac Sim asset library)
>   at the start of the demonstrator's trajectory.
> - Drive it along the XY path from `closed_loop_trajectory.csv`
>   (`/home/sun/aria/mps_Outside_20260812_141244_vrs/slam/`), subsampled to a sane waypoint
>   spacing.
> - Count and log: fall-throughs (robot Z drops below expected floor by >0.5 m), spurious
>   collisions (contact where the human walked freely — by definition traversable), and
>   how far along the path it got before failing.
> - Write results to JSON. Kit swallows stdout and force-exits, so file output is the only
>   reliable evidence — the existing script's docstring already warns about this.
>
> Run it against every candidate mesh from E2–E5 and produce a final ranking table.
