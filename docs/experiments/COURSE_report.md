# COURSE-W — course.html update: run report

TASKID=COURSE-W. Date: 2026-08-26. Target: `docs/course.html` (untracked file on
`gen2-port`; nothing committed). Spec followed:
`<scratchpad>/course_update_spec.md`. All numbers below were verified against the
reports in `docs/experiments/` before being written into the course; every command
that appears in the new lab chapter was executed on this machine first.

## 1. What was added, where (line numbers of the final file, 4,930 lines total)

| Addition | Lines |
|---|---|
| TOC part "VIII · The bigger picture" + 3 anchors (`data-mins` 35/25/30) | 526–529 |
| Ch-0 lede: "thirteen hours" → "fourteen hours" (existing TOC summed ~740 min; +90 min ≈ 13.8 h) | 537 |
| Ch-0: "built in eight parts" → "nine parts" | 541 |
| Ch-0 parts table: new row for Part VIII | 552 |
| Ch-26 "Where to go next": one sentence pointing at Chapters 29–31 (spec's TOC-updates item) | 4155 |
| **Ch-29 "Follow one frame"** (capstone lab, `id="ch-29"`) | 4460–4605 |
| **Ch-30 "Designing any pipeline: the four questions"** (`id="ch-30"`) | 4606–4654 |
| **Ch-31 "The second track: a mesh for contact"** (`id="ch-31"`) | 4655–4714 |

New chapters reuse only existing classes (`article.chapter`, `ch-eyebrow`, `lede`,
`loc`/`loc-track`/`loc-step [here|ext]`/`loc-arrow`/`loc-meta`, `box def/warn/aside`,
`terms`, `tbl`, `pre`/`pre.out` with `span.k/.s/.c`, `details.quiz`/`div.ans`,
`chfoot`). No CSS added or changed. Existing chapter numbering untouched.

## 2. Corrections applied to existing chapters

### Ch-27 (data_factor / pcd_stride are inert) — 5 edits

Grep confirmed `data_factor`/`pcd_stride` occurred **only** in ch-27 (old lines 4289,
4299); "half resolution"/"1008×756" claims occurred only in ch-27 (old lines 4308,
4319). No other chapter repeated the story (ch-r4, ch-25, ch-24 checked clean).

1. Attempt-4 table row: `cap_max=1.5M + data_factor=2 + pcd_stride=2 → completed,
   ~4 GB peak` → `cap_max=1.5M (plus two flags later found inert — below) →
   completed at full resolution, ~4 GB peak`.
2. Box def "The shape of the constraint": was "Capping one alone is not enough;
   attempt 4 halves both … drop data_factor and raise cap_max" → now: only the cap
   was actually reduced (3 M → 1.5 M took >14 GB to ~4 GB at unchanged resolution);
   "The cap was the fix. On a 24 GB card, raise cap_max."
3. New box warn "Measured finding: two of attempt 4's flags did nothing" (line 4310):
   both flags exist in `conf/config.yaml`, no Python code reads either (grep over
   `scene/`, `model/`, `train_lightning.py` — per `fundamentals_draft.md` §3.7);
   confirmed empirically by the completed run's `cameras.json` / rendered test images
   at full 2016×1512, fx=1008.
4. "What trained": "MCMC capped at 1.5 M, 1008×756" → "…, full 2016×1512
   resolution"; "Two caveats … trained at half resolution, and depth supervision
   was off" → one caveat (depth off). Following aside retitled "Why that second
   caveat…" → "Why that caveat is the interesting one".
5. Attempt-3 quiz answer: previously concluded "the remaining pressure must scale
   with pixels"; rewritten — attempt 3 proves the 3 M allocation is real (not
   fragmentation), attempt 4 halved *only* the cap at unchanged resolution and
   peaked ~4 GB, pinning the driver to Gaussian count.

### Ch-28 (Isaac import / rotation) — 7 edits, per E1_report.md

Note: the chapter as found did not literally contain "press F, add a rotate"
advice; what it contained was the stronger false claim ("the splat drops in at the
right scale **and orientation** with no transform") plus the **before-fix**
verification JSON presented as a success. Both corrected:

1. Lede: "three more steps — and one of them is not optional" → "a few more steps —
   and two of them are not optional" (outlier filter + frame fix).
2. After the `ply_to_usd.py` block: added that the whole export (filter →
   conversion → frame fix) is wrapped in
   `scripts/bash_local/export_gen2_outside_usdz.sh` (verified on disk — the script
   runs `fix_nurec_usdz_frame.py` automatically).
3. Section retitled "Coordinate frames: the conversion you do not need" → "…: right
   units, one rotation too many" (line 4402). Units half kept (Z-up metric MPS
   frame, `upAxis=Z`, `metersPerUnit=1.0`, `dataset=None` skips normalising
   transform). New: 3dgrut unconditionally bakes the Y-down→Z-up rotation
   `(x,y,z) → (−x,−z,−y)` as `xformOp:transform`; on the already-Z-up PLY that is
   one conversion too many — up lands on stage −Y.
4. New box def "The fix, and the layer it lives at": `scripts/fix_nurec_usdz_frame.py`
   (identity-rewrite; refuses unknown transforms), run automatically by the export
   script; asset now loads upright at metric scale with no manual transform.
5. "Loading it": added the "very small" framing-artifact note (F frames the ±50 m
   outlier-filter AABB; ~85 % of opacity mass within 10 m — framing, not units; E1
   §1.3).
6. Verification JSON replaced with the **after-fix** report
   (`volume_frame_is_identity: true`, bounds `[-48.06,-48.53,-49.76] …
   [49.07,48.66,47.63]` = the PLY AABB axis-for-axis), and a new box warn "How a
   sideways asset once passed this check" (line 4433) keeps the diagnosis: the old
   bounds were the correct box pushed through the rotation, invisible to a
   magnitude check; the loader now reports the composed transform.
7. "The full chain" box: chain now `… .ply → filter → convert → frame fix →
   .usdz → Isaac Sim`; plus pointers from this chapter's closing aside and box to
   Chapter 31.

### Correction 3 of the spec (2DGS ignored `opt.mcmc_strategy.*`)

The existing course nowhere claimed the 2DGS path honored that config (its only
2DGS mention is the dispatch snippet in ch-18). Per the spec's "mention only if the
course touches it", the fix is mentioned in one sentence inside new ch-31's 2DGS
paragraph (bare `MCMCStrategy()` silently ignored the 1.5 M cap; now passes the
configured values — E2 reports, fix 1).

## 3. Numbers in the new chapters, and where each was verified

| Claim in course | Source |
|---|---|
| 3DGS mesh 0.501 / 19.94 m² / 3.19 cm | `E2_3dgs_report.md` scorecard (0.5013 / 19.94 / sd-median 0.0319 m); same values in the four-way table of `E3_report.md` §7 |
| 2DGS mesh 0.640 / 8.86 m² / 3.47 cm; largest hole in same region as 3DGS's; −56 % | `E2_2dgs_report.md` (0.6405; §"same central steep-grade region"; "-56 %") |
| Stereo 0.864 / 0.91 m² / 2.18 cm; B = 135 mm (13.5 cm); FoundationStereo; σ_Z = Z²σd/(f·B): ~1 cm @1 m, 30 cm @5 m, 1.2 m @10 m; 4 m depth cut | `E3_report.md` §3 (B = 134.961 mm; 1.2 cm / 30.3 cm / 1.21 m at σd = 0.5 px), §6–7 (d40 kept, 0.8638 / 0.91 / 0.0218) |
| Trajectory floor 0.978 / 0.48 m²; eye height 1.67 m calibrated by ray-casting onto a recovered mesh | four-way table in `E3_report.md` §7 / `E2_3dgs_report.md` (recalibrated floor, h = 1.6683 from the E2a calibration; `E6_candidates_report.md` §1) |
| Fusion into stereo → 0.970 / 0.40 m² | `E6_candidates_report.md` §5–6 (FUSED-STEREO / FUSED-BEST at h = 1.6683) |
| TSDF voxel 2 cm / trunc 8 cm; ~1,000 render poses; marching cubes | `E2_3dgs_report.md` (voxel 0.02, sdf_trunc 0.08, kept 1,031/4,675 frames) |
| Drive test: splat visuals + invisible collision mesh, robot along the real path; wedge on a fusion seam every per-cell metric passed | `E6_drive_report.md` §2–4 (visual check, wp-70 seam wedge, "why E6a could not see this") |
| Splat has no surface; 1.5 M Gaussians; filtered 1,481,184 | `E1_report.md`, `fundamentals_draft.md`, and re-measured directly (see §4) |
| Lab frame `camera-rgb_239420108589`: 7,194 depth points, median 5.40 m, pos (−0.12, 3.49, −0.32), readout 10.1 ms, exposure 3.986 ms, gain 1.011, 19 JSON keys | measured directly on `/home/sun/aria/processed/...` (see §4) |
| ~2 of 3 RGB sparse-depth files empty | sampled 94 files at stride 50: 32 non-empty (34 %), median non-empty count 6,732 — consistent with the 33.3 % match rate |
| 4,090 train frames → each frame steers ~7 times | 4,675 − 585 held-out (ch-27's split); 30,000 / 4,090 ≈ 7.3 |

## 4. Commands tested before inclusion in ch-29 (all run on this machine, all successful)

```bash
# frame picker (returns camera-rgb_239420108589.png, 7194 depth points)
cd /home/sun/aria/processed/Outside_20260812_141244/camera-rgb-rectified-1008-h1512
for f in $(ls images | tail -n +$(( $(ls images | wc -l) / 2 ))); do
  ts=${f//[!0-9]/}
  n=$(jq '.z | length' "sparse_depth/camera-rgb_${ts}.json" 2>/dev/null || echo 0)
  [ "$n" -gt 0 ] && echo "$f  ($n depth points)" && break
done

# single-entry extraction (19 keys), pose/readout derivation, depth side-file stats
jq ".frames[] | select(.image_path | endswith(\"camera-rgb_${TS}.png\"))" transforms_with_sparse_depth.json
jq ".frames[] | select(...) | {pos: [...[0][3],[1][3],[2][3]], readout_ms: ...}" ...   # → (−0.123, 3.488, −0.325), 10.1
jq '{points: (.u | length), median_z: (.z | sort | .[(. | length) / 2 | floor])}' sparse_depth/camera-rgb_${TS}.json  # → 7194 / 5.397

# code stops
sed -n '749,777p' scene/cameras.py                    # AriaCamera.__init__ signature
grep -n "class SceneInfo" -A 12 scene/dataset_readers.py   # line 50
grep -n "def training_step" model/vanilla_gsplat.py        # line 1027

# artifacts
/home/sun/miniforge3/envs/ego_splats/bin/python - <<'EOF'   # → 1500000 / 1481184
from plyfile import PlyData
root = "output/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/point_cloud/iteration_30000/"
for name in ["point_cloud.ply", "point_cloud_filtered.ply"]:
    print(name, PlyData.read(root + name).elements[0].count)
EOF
unzip -l output/.../isaacsim/Outside_20260812_141244.usdz   # 3 entries: default.usda / .nurec / gauss.usda
which xdg-open   # /usr/bin/xdg-open (viewer command in the chapter)
```

The `sed` line-number command in the chapter is flagged in-page as "line numbers as
of this writing". `export_gen2_outside_usdz.sh` itself was not re-run (needs the
3dgrut env + GPU compile); the lab inspects its existing outputs instead.

## 5. Spec-vs-report discrepancies (report wins in the course text)

1. **Stereo error at 1 m**: spec says "~1 cm at 1 m"; `E3_report.md` says 1.2 cm at
   σd = 0.5 px. Course says "roughly a centimetre at 1 m, 30 cm at 5 m, 1.2 m at
   10 m" — the report's values.
2. **Trajectory-floor 0.978 / 0.48 m²**: E5's original standalone floor scored
   0.9866 / 0.37 m² (h = 1.60 build, default harness). The spec's numbers are the
   **recalibrated** floor (rebuilt at h = 1.6683) under the default-h harness — the
   values in the four-way comparison table that also produced the other three rows
   (`E3_report.md` §7). Used as-is for table consistency; the fused numbers quoted
   (0.970 / 0.40) are E6a's calibrated-h scoring, which is that experiment's own
   like-for-like protocol.
3. **`E6_drive_report.md` is unfinished**: it contains `DRIVE-RESULTS-PLACEHOLDER`
   and `RANKING-PLACEHOLDER`. Ch-31 therefore describes the drive test's design and
   its one fully documented finding (the fusion-seam wedge and its census) and
   quotes **no** final fall-through counts, per-candidate outcomes, or winner.
4. **Spec's suggested quiz** "why can 2DGS never fill a floor hole 3DGS has?" is
   contradicted by the data as phrased (2DGS *did* shrink the holes: 8.86 vs
   19.94 m²). Reworded to the defensible version the reports support: why the
   largest hole sits in the same region for both.
5. **Spec's ch-28 description** ("press F, add a rotate" advice) did not match the
   chapter as found; the actual defects were the false "no transform needed for
   orientation" claim and the before-fix verification JSON shown as success. Both
   were corrected per `E1_report.md` (see §2).

## 6. Other notes

- Validated the full file with an HTML tag-balance parser after patching: the only
  imbalance is a **pre-existing** literal `<exp>` placeholder inside a `<code>` in
  ch-23's locator (line 3751), present before this task and left untouched
  (out of scope; browsers parse it leniently).
- All 19 patch operations asserted a unique anchor before replacing; the patch
  script is `<scratchpad>/patch_course.py`.
- Ch-0's part count/hours were updated (§1) because the new TOC part made the
  existing "eight parts / thirteen hours" text false; no other untouched chapter
  makes a claim invalidated by the additions.
- Nothing committed to git.

## Review

TASKID=COURSE-R. Date: 2026-08-26. Adversarial review of the COURSE-W update to
`docs/course.html`. Verdict: the update is sound; two small factual/phrasing fixes
were applied directly. Nothing committed to git.

### 1. Format fidelity — PASS

- Diff-read ch-29/30/31 against pre-existing ch-15 and ch-27/28: identical
  furniture (`article.chapter` + `ch-eyebrow` with `sep` spans, `h1`, `p.lede`,
  optional `div.loc` locator, `h2` sections, `box def/warn/aside` with `span.tag`,
  `<h2>Check yourself</h2>` + `details.quiz`/`div.ans`, closing `div.chfoot`).
  "Check yourself" h2 now appears 32×; the new chapters follow the majority
  convention (ch-27/28's quiz-without-heading is pre-existing variance).
- Every class used in the new chapters is defined in the `<style>` block
  (`loc*`, `box`/`def`/`warn`/`aside`, `tag`, `terms`/`row`, `tbl`, `td.m/th.m`,
  `pre .k/.s/.c`, `quiz`, `ans`, `chfoot`). One nuance: `pre.out` has no CSS rule —
  but that was already true of the ~40 pre-existing `pre class="out"` uses
  (first at line 818), so no new class was introduced. No new CSS.
- TOC: part "VIII · The bigger picture" present with three anchors,
  `data-mins` 35/25/30; all 36 TOC hrefs resolve to exactly-once article ids; no
  chapter missing from the TOC; total `data-mins` = 860 min ≈ 14.3 h, matching
  ch-0's updated "fourteen hours". No markdown syntax leaking in lines 4460–4714.

### 2. Facts — PASS after one fix

Every scoreboard number re-checked against the reports: 3DGS 0.501/19.94 m²/
3.19 cm (E2_3dgs scorecard + E3 §7 table), 2DGS 0.640/8.86/3.47 with −56 % and
same-central-region hole (E2_2dgs), stereo 0.864/0.91/2.18 with B=134.961 mm,
sigma table 1.2 cm@1 m / 30.3 cm@5 m / 1.21 m@10 m, 4 m depth cut, 512 px rig
(E3 §3–7), floor 0.978/0.48 (E3 §7 four-way table — the recalibrated h=1.6683
build under the default harness, consistent with the other three rows; E5's
standalone 0.9866/0.37 is a different protocol, correctly not mixed in), fused
0.970/0.40 (E6_candidates §5), h fitted = 1.6683 by ray-casting (E2_3dgs),
TSDF voxel 2 cm / trunc 8 cm / 1,031 of 4,675 poses (E2_3dgs), harness 10 cm
cells / ±15 cm (E0), wp-70 fusion-seam wedge with per-cell-pass trench
(E6_drive §4 — the one documented drive finding; no placeholder results were
quoted, correct). Ch-28's rewrite matches E1_report digit for digit: rotation
(x,y,z)→(−x,−z,−y), after-fix bounds [−48.06,−48.53,−49.76]…[49.07,48.66,47.63]
= PLY AABB, before-fix bounds = the same box rotated, ~85 % opacity mass within
10 m, `volume_frame_is_identity`, fix wired into `export_gen2_outside_usdz.sh`
(filter → ply_to_usd → fix_nurec order verified in the script itself). Ch-29's
per-frame numbers (7,194 pts, median 5.397 m, pos (−0.123, 3.488, −0.325),
readout 10.1 ms, exposure 3.986 ms, gain 1.011, 19 keys, fx/fy/cx/cy
1008/1008/1007.5/755.5 at 2016×1512) all re-measured directly. Split arithmetic
(4,675 → 585 held out at indices 0,8,16…, 4,090 train, ≈7 visits) matches
`dataset_readers.py`. PSNR 25.55 / 585 frames match ch-27.

**Fixed (factual error):** ch-29 said the pose is re-derived "at up to sixteen
sub-frame times". The code (`scene/cameras.py:746`,
`_max_rolling_shutter_sample = 8`) and Chapter 21's own text ("Between 1 and 8")
say eight. Changed to "up to eight".

**Fixed (contradiction risk):** ch-29 quiz called the scene "the 10 m walk" —
ch-27 states the walk is 100.2 m long; the ~10 m figure is the trajectory's X–Y
span (ch-28). Rephrased to "the walk's roughly 10 m X–Y extent (Chapter 28)".

Not touched: ch-28's pre-existing median-filter box says the median Gaussian
position was (−0.14, −0.39, −1.30) while E1_report says (−0.099, −0.346, −1.307).
Pre-existing text outside this update's scope, both ≈1.4 m from origin (the
quiz's claim), and plausibly from the filter run's own log rather than E1;
flagged here rather than silently changed.

### 3. Commands — PASS (all run verbatim on this machine)

Frame picker → `camera-rgb_239420108589.png  (7194 depth points)`; full-entry jq
(19 keys, all as listed); pos/readout jq → (−0.123, 3.488, −0.325) / 10.1;
depth-side-file jq → 7194 / 5.397; `sed -n '749,777p' scene/cameras.py` prints
the complete `AriaCamera.__init__` signature (class starts line 730 —
correct class); `grep "class SceneInfo"` → line 50; `grep "def training_step"` →
1027; the plyfile heredoc → 1500000 / 1481184; `unzip -l` → the three entries
with the exact sizes shown (806 / 174,794,913 / 2,259); `xdg-open` exists at
/usr/bin/xdg-open. `$PROC`/`$RECT` paths agree with ch-r3's exports
(`REC=/home/sun/aria`). No command failed; no fixes needed.

### 4. Corrections complete — PASS

Whole-file greps: `data_factor`/`pcd_stride` survive only inside the new
"Measured finding" warn box teaching that they are inert (line 4311); no "press
F"/"press f" anywhere; every rotation+Isaac passage teaches the baked-3dgrut-
rotation story with `fix_nurec_usdz_frame.py`; no "half resolution"/"1008×756"
claims about the run remain; attempt-4 row, constraint box, "What trained",
caveat aside, and attempt-3 quiz answer all consistently teach cap_max as the
fix at full resolution.

### 5. Integrity — PASS

`html.parser` sweep: the only imbalance is the pre-existing literal `<exp>`
inside `<code>` at line 3751 (ch-23 locator), untouched by COURSE-W and by this
review. 36 `article.chapter` elements = 33 pre-existing + 3 new; no duplicate
ids; every TOC href targets an existing id.

### 6. Voice spot-check — PASS

Ch-29 read end-to-end as the learner: every leaned-on concept is taught earlier
(133° lens → ch-3, centred principal point → ch-10, 33.3 % match → ch-15/27,
`expose_image` → ch-22, 7-1 split → ch-23, profile10 readout and PSNR → ch-27,
export chain → ch-28). Ch-30's single forward reference (Chapter 31) is an
explicit pointer, consistent with course practice. FoundationStereo, surfels,
TSDF, marching cubes are all defined at first use in ch-31. Second person,
em-dash cadence, "worth sitting with"-style asides match the house voice.
