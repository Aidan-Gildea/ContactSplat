# E1 — Isaac Sim units / orientation diagnosis

Date: 2026-08-20. Task: root-cause why the exported NuRec USDZ
(`output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/isaacsim/Outside_20260812_141244.usdz`)
appears **very small and incorrectly oriented** in Isaac Sim, despite
`scripts/bash_local/export_gen2_outside_usdz.sh` claiming it is "Z-up in metres, matching
the MPS world frame, so it needs no extra transform". Land a deterministic fix at the
correct layer.

## Verdict up front

- **The `metersPerUnit` hypothesis is rejected.** Asset and Isaac Sim stage both use
  `metersPerUnit = 1.0`, `upAxis = Z`. There is no unit mismatch and no 100x scale error.
- **The misorientation is real and root-caused:** 3dgrut's exporter unconditionally bakes a
  Y-down-to-Z-up conversion rotation, `(x,y,z) -> (-x,-z,-y)`, into the NuRec `Volume`
  prim. Our PLY is *already* Z-up (MPS world frame), so the rotation sends up (+Z) to
  stage **-Y** and lays the scene on its side. The export script's claim was **false** for
  orientation (true for units).
- **"Very small" is a framing artifact, not a scale error:** ~85 % of the splat's opacity
  mass sits within 10 m of the scene centre, but the asset's AABB is the +/-50 m
  outlier-filter radius, so `F` frames a ~97 m box around ~20 m of content.
- **Fix landed export-side** (the asset itself is now correct for any consumer):
  `scripts/fix_nurec_usdz_frame.py` strips the baked rotation from the USDZ; the export
  script runs it automatically; `scripts/isaacsim_load_splat.py` now reports the volume
  transform so a regressed asset is caught. Verified end-to-end in Isaac Sim.

---

## 1. Evidence chain

### 1.1 What the exporter writes (code inspection)

`/home/sun/3dgrut/threedgrut/export/scripts/ply_to_usd.py` calls
`USDZExporter.export(model, output_path, dataset=None, conf=conf)`.

- `MixtureOfGaussians.init_from_ply` (`threedgrut/model/model.py:714`) copies PLY
  x/y/z **verbatim** — no transform on load.
- `USDZExporter.export` (`threedgrut/export/usdz_exporter.py`): with `dataset=None`
  the normalizing transform stays identity (the log line "Applying normalizing transform"
  never fires — consistent with docs/full_pass.md).
- `serialize_nurec_usd` (`threedgrut/export/usd_util.py`) then does, **unconditionally**:

  ```python
  default_conv_tf = np.array([[-1,0,0,0],[0,0,-1,0],[0,-1,0,0],[0,0,0,1]])
  corrected_matrix = np.linalg.inv(normalizing_transform) @ default_conv_tf
  matrix_op = gauss_volume.AddTransformOp()
  matrix_op.Set(Gf.Matrix4d(*corrected_matrix.flatten()))
  ```

  and sets `omni:nurec:useProxyTransform = False`, which makes that transform apply to the
  rendered gaussians. In USD's row-vector convention `p' = p @ M`, this matrix maps
  `(x, y, z) -> (-x, -z, -y)`.

- Layer metadata: `initialize_usd_stage` writes `metersPerUnit = 1`, `upAxis = "Z"` into
  both `default.usda` and `gauss.usda`. So the *metadata* claim in the export script is
  true.

**Why the matrix exists upstream:** 3dgrut's normal path applies
`estimate_normalizing_transform` (`threedgrut/export/normalizing_transform.py`), which
aligns the average camera *down* vector with **+Y** — i.e. 3dgrut's normalized scene frame
is Y-down. `default_conv_tf` converts that Y-down frame to the declared Z-up stage frame
(down +Y -> -Z, i.e. up -> +Z). For a COLMAP-ish Y-down PLY that is a reasonable default.
For our PLY — already gravity-aligned Z-up metres in the MPS world frame — it is exactly
one frame conversion too many: up (+Z) lands on stage **-Y**.

### 1.2 What is actually inside the USDZ (pxr / zip inspection)

The USDZ is a stored zip: `default.usda` (806 B), `Outside_20260812_141244.nurec`
(174.8 MB), `gauss.usda` (2.3 KB). `gauss.usda` (before the fix) contained:

```
metersPerUnit = 1
upAxis = "Z"
...
def Volume "gauss"
{
    float3[] extent = [(-48.0622, -48.530827, -49.75611), (49.070374, 48.658726, 47.625977)]
    ...
    custom bool omni:nurec:useProxyTransform = 0
    matrix4d xformOp:transform = ( (-1, 0, 0, 0), (0, 0, -1, 0), (0, -1, 0, 0), (0, 0, 0, 1) )
    uniform token[] xformOpOrder = ["xformOp:transform"]
```

The `extent` equals the filtered PLY's own AABB **exactly** (see 1.3) — confirming the
`.nurec` payload keeps raw MPS-world coordinates and the only frame change is that
`xformOp:transform`.

### 1.3 The PLY's own geometry (numeric signature)

Computed from `point_cloud_filtered.ply` (1,481,184 gaussians) in the `ego_splats` env
with `plyfile` (opacity = sigmoid of the stored logit):

| Quantity | Value |
|---|---|
| PLY AABB min / max | (-48.0622, -48.5308, -49.7561) / (49.0704, 48.6587, 47.6260) |
| median centre | (-0.099, -0.346, -1.307) |
| opacity mass within 5 / 10 / 15 / 20 / 30 m of median | 32.8 % / **84.9 %** / 92.1 % / 95.2 % / 97.8 % |
| gaussians with opacity > 0.5 | 799,176 (54 %) |
| their z-histogram peak (floor) | z in [-1.554, -1.455], 68,737 gaussians in one 10 cm bin |
| MPS device z (155,800 closed-loop poses) | median -0.094, range -0.977 .. 0.686 |
| implied device height above floor | -0.094 - (-1.50) = **1.41 m** |

Two conclusions:

- **The PLY content is genuinely Z-up metric MPS-world**: a sharp floor plane at constant
  z = -1.50 m, sitting a plausible head height (1.41 m) below the walking trajectory.
  So the *content* frame claim was right; the exporter's rotation is what breaks it.
- **The "very small" appearance is framing**: the asset's AABB (~97 m across, set by the
  50 m outlier-filter radius that retains a sparse low-opacity shell) is ~5-10x larger
  than the visible content (85 % of opacity mass within a 20 m diameter). Pressing `F`
  frames the AABB, so the scene occupies a small fraction of the view. Units are correct.

### 1.4 Baseline Isaac Sim run (before the fix)

```bash
~/isaac-sim/python.sh scripts/isaacsim_load_splat.py \
    output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/isaacsim/Outside_20260812_141244.usdz \
    --report <scratch>/usd_report_before.json
```

(Read via `--report` JSON as instructed — Kit swallows stdout.) Result:

```json
{ "ok": true, "up_axis": "Z", "meters_per_unit": 1.0, "nurec_prim_count": 2,
  "world_aabb_min": [-49.0704, -47.6260, -48.6587],
  "world_aabb_max": [ 48.0622,  49.7561,  48.5308] }
```

- `meters_per_unit` **1.0 on the Isaac Sim stage** (Isaac Sim 5.1 defaults to metres) and
  1.0 in the asset -> hypothesis "cm stage vs m asset" rejected with direct evidence.
- The world AABB is the local `extent` pushed through `(x,y,z) -> (-x,-z,-y)`, matching
  **axis-for-axis, digit-for-digit**:
  - world x = -[local x]: [-49.0704, 48.0622] from local [-48.0622, 49.0704] ✓
  - world y = -[local z]: [-47.6260, 49.7561] from local [-49.7561, 47.6260] ✓
  - world z = -[local y]: [-48.6587, 48.5308] from local [-48.5308, 48.6587] ✓

  This proves the baked rotation is live in the composed stage the renderer sees — the
  quantitative smoking gun, no GUI needed. (Same numbers as the "verified result" in
  docs/full_pass.md §5, which had been read as success because the rotated bounds of a
  near-symmetric AABB still *look* plausible.)

## 2. The fix (deterministic, export-side)

The correct layer is the **asset**: fixing it there makes File->Open, drag-and-drop,
references from E6, and any other consumer correct, with no per-consumer counter-rotation.
`/home/sun/3dgrut` was deliberately left untouched (third-party repo; its default is
defensible for COLMAP-frame PLYs) — the fix lives downstream in this repo.

### 2.1 `scripts/fix_nurec_usdz_frame.py` (new)

Rewrites the NuRec `Volume`'s `xformOp:transform` to identity inside the USDZ
(unzip -> edit `gauss.usda` via `pxr.Sdf` -> re-zip stored, preserving entry order with the
root layer first, mirroring 3dgrut's own `write_to_usdz`). Deliberately conservative:

- transform == known 3dgrut conversion matrix -> set to identity;
- transform == identity already -> no-op (idempotent);
- anything else -> **refuse and exit 2** rather than guess.

In-place by default, `-o/--output` to write elsewhere. Runs under any Python with `pxr`
(the `3dgrut` env is used here; `ego_splats` has no `pxr`).

All three paths were exercised:

```
$ .../3dgrut/bin/python scripts/fix_nurec_usdz_frame.py <asset>.usdz
gauss.usda: /World/gauss xformOp:transform: 3dgrut conversion -> identity   # exit 0
$ ... (again)
gauss.usda: /World/gauss transform already identity -- nothing to do        # exit 0
$ ... (copy with a hand-edited 0.5-scale matrix)
ERROR: gauss.usda: /World/gauss has an unexpected transform, refusing ...   # exit 2
```

### 2.2 `scripts/bash_local/export_gen2_outside_usdz.sh` (updated)

- Runs `fix_nurec_usdz_frame.py` on the USDZ right after `ply_to_usd.py`, with a comment
  explaining why (pointing at this report).
- The false trailing claim is corrected: it now says the baked rotation has been stripped,
  and warns that `F`-framing on the +/-50 m filter AABB makes the content look small
  (framing, not units). Filter radius hoisted into a `FILTER_RADIUS` variable.

### 2.3 `scripts/isaacsim_load_splat.py` (updated)

Now finds the `Volume` prim, computes its composed local-to-world transform, prints it,
and adds `volume_local_to_world` + `volume_frame_is_identity` to the `--report` JSON —
so a regressed asset (e.g. re-exported without the fix step) is caught by the smoke test
instead of by someone rotating it by hand in the GUI.

## 3. Validation of the fixed asset

The shipped USDZ was fixed **in place** (original preserved at
`<scratchpad>/e1/Outside_original_backup.usdz`).

1. **Layer content:** `gauss.usda` now has `xformOp:transform = identity`; `extent`,
   crop bounds, `.nurec` payload byte-identical; zip order/storage preserved
   (`default.usda` first, all entries STORED).
2. **Pure pxr composed stage** (no Kit): `Usd.Stage.Open(usdz)` -> upAxis Z,
   metersPerUnit 1.0, world AABB min (-48.0622, -48.5308, -49.7561) /
   max (49.0704, 48.6587, 47.6260) — **equals the PLY AABB exactly**.
3. **Isaac Sim acceptance run** (same command as 1.4, updated loader):

   ```json
   { "ok": true, "up_axis": "Z", "meters_per_unit": 1.0, "nurec_prim_count": 2,
     "volume_frame_is_identity": true,
     "world_aabb_min": [-48.0622, -48.5308, -49.7561],
     "world_aabb_max": [ 49.0704,  48.6587,  47.6260] }
   ```

   Both `OmniNuRecFieldAsset` prims still resolve in the RTX renderer, the volume
   transform is identity, and the world bounds are now the MPS-world PLY bounds verbatim.
   The splat therefore stands upright (+Z up) at metric scale in the MPS world frame —
   a collision mesh in that frame (E2-E5) drops in with **no registration transform**,
   which is the invariant E6 depends on.

## 4. Problems hit and honest notes

- **Kit swallows stdout / lies with exit codes** — as pre-warned. All Isaac Sim evidence
  above comes from `--report` JSON files, never from printed output. Runs were done in
  the background (startup is minutes). GPU was checked idle before each run (597 MiB
  used, 0 % util; no training jobs started — the loader render itself is light).
- **`docs/full_pass.md` §4 ("Why no coordinate conversion is needed") is partially
  wrong** and should be amended in the final human report: the units half is correct, but
  "the splat drops in at the right ... orientation with no extra transform" was false
  until this fix — its §5 "verified result" only checked prim resolution and bounds
  magnitude, and a rotation of a near-symmetric AABB is invisible to that check. The
  loader's new `volume_frame_is_identity` field closes that verification gap.
- **What was *not* changed:** `/home/sun/3dgrut` (upstream repo untouched); the 50 m
  outlier-filter radius (the "small on F" framing effect remains, now documented in the
  export script's output; tightening the radius to ~20-30 m would shrink the AABB at the
  cost of dropping ~2-5 % of opacity mass — left as a knob, not a change);
  `point_cloud_filtered.ply` and the `.nurec` payload (untouched by the fix).
- A GUI eyeball check of the fixed asset was not performed in this stage (headless
  policy); the numeric evidence (identity composed transform + exact AABB match, with the
  renderer resolving both NuRec prims) is sufficient to conclude orientation. If desired:
  `~/isaac-sim/python.sh scripts/isaacsim_load_splat.py <usdz> --gui`.
- Nothing was committed to git.

## 5. Exact commands run (chronological, deduplicated)

```bash
# inspection
python (3dgrut env): zipfile listing + dump of default.usda / gauss.usda from the USDZ
python (ego_splats env): PLY stats (plyfile) on point_cloud_filtered.ply
python (ego_splats env): closed_loop_trajectory.csv z-range

# baseline (broken asset)
cp <usdz> <scratchpad>/e1/Outside_original_backup.usdz
~/isaac-sim/python.sh scripts/isaacsim_load_splat.py <usdz> --report .../usd_report_before.json

# fix + tests
/home/sun/miniforge3/envs/3dgrut/bin/python scripts/fix_nurec_usdz_frame.py <usdz>   # in place
# re-run (idempotency), -o mode on the backup, refusal test on a doctored copy

# acceptance (fixed asset, updated loader)
python (3dgrut env): pure-pxr Usd.Stage.Open bounds check
~/isaac-sim/python.sh scripts/isaacsim_load_splat.py <usdz> --report .../usd_report_after.json
```

Artifacts kept in the session scratchpad `e1/`: `usd_report_before.json`,
`usd_report_after.json`, `isaac_before.log`, `isaac_after.log`,
`Outside_original_backup.usdz`.
