# DOC — run report: fundamentals draft

Task: write the from-the-ground-up understanding section of the final report
(`docs/experiments/fundamentals_draft.md`), grounded in the actual code with file:line
citations. No GPU work was run. This report records how the draft was produced and, more
importantly, what reading the code turned up that the existing docs do not say.

## What was done

- Read `docs/mesh_experiments.md` and `docs/full_pass.md` in full.
- Read, in full: `scripts/extract_aria_vrs.py` (1,317 lines), `scripts/aria_utils.py`
  (700), `utils/point_utils.py`, `utils/aria_utils.py`, `train_lightning.py`,
  `model/vanilla_gsplat.py` (1,536), `model/GS2D_gsplat.py` (534), `model/loss.py`,
  `scene/__init__.py`, `scene/cameras.py` (1,262), `scene/dataset_readers.py` (883),
  `conf/config.yaml`, `conf/opt/simple_gsplat_30K.yaml`,
  `scripts/filter_splat_outliers.py`, `scripts/isaacsim_load_splat.py`, the three
  `scripts/bash_local/*.sh` wrappers, and (in `/home/sun/3dgrut`)
  `threedgrut/export/scripts/ply_to_usd.py` plus the relevant parts of
  `threedgrut/export/usdz_exporter.py`.
- Verified claims against on-disk data rather than trusting docs:
  - MPS file schemas (column headers of `closed_loop_trajectory.csv`,
    `semidense_points.csv.gz`, `semidense_observations.csv.gz`; first record of
    `online_calibration.jsonl` — `ReadoutTimesSec: [[4, 0.0101]]`, 1,557 records).
  - `transforms_with_sparse_depth.json` structure and per-frame keys; readout time
    10.1 ms; 4,675 frames; sparse-depth JSON keys.
  - The completed training run's `cfg_args`, `cameras.json` (4,090/585/585 split,
    **width 2016 × height 1512, fx 1008**), `test_logs.json` (PSNR 25.548 /
    SSIM 0.846 / LPIPS 0.389 over 585 test frames), and a rendered test image
    (2016×1512).
  - gsplat version (1.5.3) and the `MCMCStrategy` / `DefaultStrategy` docstrings and
    signatures in the ego_splats env.
- Wrote `docs/experiments/fundamentals_draft.md` (~490 lines), citing file:line for
  every mechanism described.
- Mid-task, E1's fix landed on disk (`scripts/fix_nurec_usdz_frame.py`, updated
  `scripts/isaacsim_load_splat.py` and export script); read the new code and updated
  the draft's §4.2/§4.3/§5 to match the current pipeline (baked 3dgrut Volume-prim
  rotation, now stripped at export; loader now asserts the volume transform is
  identity).

No commits, no pushes, no GPU jobs.

## Surprising findings (things the code says that the docs don't)

Ordered by how much they matter to the rest of the experiment program.

### 1. `scene.data_factor` and `scene.pcd_stride` are inert — full_pass.md's VRAM story is partly wrong

`conf/config.yaml:41` (`pcd_stride`) and `conf/config.yaml:61` (`data_factor`) are
declared, and `scripts/bash_local/train_gen2_outside.sh` passes `scene.data_factor=2
scene.pcd_stride=2` — but **no code in the repo reads either key**. Grep over `scene/`,
`model/`, `train_lightning.py` finds no consumer; `utils/point_utils.py:17-33` has a
`stride` parameter on `fetchPly`, but the Aria path builds its point cloud via
`mps.read_global_point_cloud` (`scene/dataset_readers.py:687-703`) and never calls it.
Hydra accepts the overrides because the keys exist in the schema, so nothing errors.

Empirical confirmation from the completed run's own outputs: `cameras.json` records all
cameras at 2016×1512 with fx=1008 (not 1008×756/fx=504), and the rendered test images
are 2016×1512. So the run that succeeded trained at **full resolution**, and what
actually fixed the OOM was MCMC + `cap_max=1.5M` (plus `expandable_segments`), not the
image/point-cloud downscaling that `docs/full_pass.md` (§3, Results) and the comments in
`train_gen2_outside.sh` credit.

Consequence for E2: "retrain with the same settings" = MCMC + cap_max. Anyone who wants
genuinely reduced-resolution training must *implement* `data_factor` (e.g. resize in
`Camera.get_image` and scale intrinsics), not just pass the flag. It also means the
documented ~4 GB peak was achieved at full 2016×1512, which is better news for VRAM
headroom than the docs imply.

### 2. 2DGS ignores the MCMC config block entirely — including `cap_max`

`Gaussians2D._create_strategy` (`model/GS2D_gsplat.py:221-224`) constructs
`MCMCStrategy()` **with no arguments**, so a 2DGS run with
`opt.densification_strategy=MCMC` silently uses gsplat's defaults: `cap_max=1,000,000`,
`refine_stop_iter=25,000`, `min_opacity=0.005` — and ignores
`opt.mcmc_strategy.cap_max=1500000` (or 3M, or anything else) from the command line. The
`default` branch likewise ignores the `gs_default_strategy` knobs
(`model/GS2D_gsplat.py:210-217`, only `key_for_gradient="gradient_2dgs"` is set). The
3DGS class plumbs all of these through (`model/vanilla_gsplat.py:581-621`); the 2DGS
subclass was never given the same treatment.

Consequence for E2's 2DGS retrain: the Gaussian budget will be 1.0 M regardless of what
the command line says. That may be fine for VRAM (it is *lower* than the 3DGS run's cap)
but it is not the experiment the brief describes, and the PSNR comparison would be
budget-confounded. E2 should either patch `Gaussians2D._create_strategy` to mirror the
vanilla one, or report the actual cap used.

### 3. The preprocessing script still uses the *old* `project()` — the fixed copy lives elsewhere

There are two `project()` implementations: the fixed one in `utils/point_utils.py:66-110`
(rank-based single/batch dispatch, `z > 0` everywhere, NaN-safe division) used by the
training side (`scene/cameras.py:27`, `scene/dataset_readers.py:27`), and an old copy in
`scripts/aria_utils.py:263-297` that retains both original defects (the
`shape[-1] > 1` batch test and the `and`-binds-tighter-than-`or` scalar bug). 
`scripts/extract_aria_vrs.py:23-34` imports `project` **from `aria_utils`** — so sparse
depth generation runs on the old copy.

Why nothing broke: the old copy's *batched* branch does include `z > 0` in its mask, and
the vectorised preprocessing always calls it with (3, N) batches. The bad paths are only
reachable when a frame has exactly one observed point (shape (3,1) → misrouted to the
scalar branch → a behind-camera point with sign-flipped in-bounds (u,v) is accepted with
negative depth). The instrumentation added in `scripts/extract_aria_vrs.py:1044-1052`
measured zero such points on this recording, matching full_pass.md's "measured impact:
zero". But full_pass.md's implication that preprocessing now runs on the fixed function
is wrong — the fix exists, preprocessing just doesn't import it. One-line remedy if ever
touched again: import `project` from `utils.point_utils` in `extract_aria_vrs.py` (or
delete the stale copy in `scripts/aria_utils.py`).

### 4. Latent crashers in code paths not exercised by the documented run

- `model/vanilla_gsplat.py:1110`: `self.cfg.opt.l1_grad_lamda` (typo — config declares
  `l1_grad_lambda`, `conf/opt/simple_gsplat_30K.yaml:128`). Setting `opt.l1_grad=true`
  on a 3DGS run raises `ConfigAttributeError` at the first training step. The 2DGS path
  spells it correctly (`model/GS2D_gsplat.py:400`).
- `model/GS2D_gsplat.py:331-340`: the `try/except` around the training render returns
  `{"loss": loss, "render": image}` with `loss`/`image` **undefined** in the except
  branch → any real rendering exception is masked by a `NameError`. Also a bare
  `except`.
- `interpolate_aria_pose` — both copies (`scripts/aria_utils.py:225-226`,
  `utils/aria_utils.py:74-75`): the end-of-trajectory branch does
  `return closed_loop_traj[start_pose]`, indexing a list with a *pose object* →
  `TypeError` if a queried timestamp lands at/after the last trajectory sample.
  Training partially guards it by trimming frames near the trajectory edges
  (`scene/dataset_readers.py:395-410` — note the comments there say "100 ms" but the
  constant is `1e6` ns = **1 ms**); preprocessing avoids it in practice because frame
  timestamps sit inside the trajectory span. A 1 ms margin is thinner than the ±5 ms
  the rolling-shutter sampler can query around a frame (`scene/cameras.py:843-855`), so
  this can in principle still fire for the last RGB frame of a recording.
- `model/loss.py:191`: `if capture_luminance is None` — `capture_luminance` is unbound
  when `l1_grad` is requested in RGB color space (only assigned in the LUMINANCE
  branch), a second reason `l1_grad` cannot currently be used.

None of these affect the documented RGB / no-l1_grad / 3DGS run; all of them affect
someone flipping "harmless-looking" config switches. E2/E6 should know before touching
2DGS or extra losses.

### 5. Loss weights: the DSSIM weight is hard-coded, config keys are decorative

`loss = 0.8·L1 + 0.2·DSSIM` uses the literal `0.2` in both models
(`model/vanilla_gsplat.py:1107`, `model/GS2D_gsplat.py:397`); `ssim_lambda`
(`conf/opt/simple_gsplat_30K.yaml:57`) and `dssim_lambda`
(`conf/opt/simple_gsplat_30K.yaml:132`) are read by nothing. Benign at defaults (both
say 0.2), misleading for anyone sweeping them.

### 6. Three different confidence thresholds for the same semi-dense cloud

- Preprocessing depth generation: `inverse_distance_std < 0.005`, `distance_std < 0.01`
  (`scripts/extract_aria_vrs.py:417-421`).
- Training initialization: `< 0.01`, `< 0.02` (`scene/dataset_readers.py:691-694`) —
  looser, so the Gaussian init uses more (noisier) points than depth supervision would.
- The (dead) `visualize_frames` helper: `< 0.001`, `< 0.15`
  (`scripts/extract_aria_vrs.py:326`).

Not a bug, but E0's "filter by the existing confidence thresholds used in
`extract_aria_vrs.py`" needs to know there are several; the 0.005/0.01 pair at
`scripts/extract_aria_vrs.py:417-421` is the one that gates depth data.

### 7. Confirmations worth restating (no surprise, verified)

- Depth supervision is off by default (`depth_loss: false`,
  `conf/opt/simple_gsplat_30K.yaml:135`); sparse depth still feeds the 3D smooth
  filter's min-depth (`model/vanilla_gsplat.py:808-810`) and the rolling-shutter motion
  estimate (`scene/cameras.py:805-812`), so it is not wasted work even unsupervised.
- The MCMC config-key fix in the working tree (`model/vanilla_gsplat.py:604-620`) is
  load-bearing: without it `densification_strategy=MCMC` crashes at startup, and MCMC's
  `cap_max` is the only Gaussian-count bound — i.e. the only reason this scene trains
  in 16 GB.
- `ReadoutTimesSec` in `online_calibration.jsonl` lists only the RGB stream
  (`[[4, 0.0101]]`); SLAM cameras are global shutter and correctly report None →
  readout 0 (`scripts/aria_utils.py:100-140`).
- The exported USDZ payload keeps raw MPS world coordinates because `ply_to_usd.py`
  passes `dataset=None` (no normalizing transform) — but 3dgrut *does* bake a
  Y-down→Z-up rotation onto the Volume prim, which E1 root-caused during this program;
  the export script now strips it (`scripts/fix_nurec_usdz_frame.py`) and the loader
  verifies identity (`scripts/isaacsim_load_splat.py:64-83`). full_pass.md §4's "no
  coordinate conversion is needed" claim predates this and is stale as written.

## Problems hit

- **Concurrent edits**: `scripts/isaacsim_load_splat.py` and the export script changed
  on disk mid-task (E1's fix landing). Handled by re-reading the new code and updating
  the draft rather than describing the pre-fix behaviour; the draft now documents the
  post-fix pipeline and credits E1.
- **Docs-vs-code conflicts** (`data_factor`/`pcd_stride`, the `project()` import): per
  the task brief, resolved in favour of the code, with the on-disk evidence cited
  (cameras.json / render image sizes) so the final report can defend the claim.

## Deliverables

- `docs/experiments/fundamentals_draft.md` — the draft section (the deliverable).
- `docs/experiments/DOC_report.md` — this report.
