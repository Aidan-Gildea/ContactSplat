# REPORT-W — final report assembly: run log

TASKID=REPORT-W. Date: 2026-08-26. Deliverable:
`docs/experiments/final_report_content.md` (content only; typesetting is a later
task). Spec followed: `<scratchpad>/final_report_spec.md`. Nothing committed to git.

## Sources used

Read in full before writing:

- `docs/experiments/`: `E0_report.md`, `E1_report.md`, `E2_3dgs_report.md`,
  `E2_2dgs_report.md`, `E3_report.md`, `E5_report.md`, `E6_candidates_report.md`,
  `E6_drive_report.md`, `fundamentals_draft.md`, `DOC_report.md`, `COURSE_report.md`.
- Context: `docs/full_pass.md`, `docs/mesh_experiments.md` (incl. the E4 brief, used
  for the photogrammetry design section).
- Numeric ground truth (all opened and cross-checked, JSON wins over prose):
  - `output/Outside_20260812_141244/collision_candidates/comparison_e6a_h16683.json`
    (all seven h=1.6683 scorecards) and the three per-fusion
    `*.ply.scorecard_h16683.json` files.
  - `.../collision_candidates/fused_{stereo,2dgs,best}/fusion_report.json` and
    `fused_best/prefuse_composition.json`.
  - `.../camera-rgb-rectified-1008-h1512/mesh_tsdf/*.scorecard.json` +
    `comparison_e2a.json`; `.../-2dgs/mesh_tsdf/*.scorecard.json` +
    `comparison_e2b.json`; `.../stereo_depth/*.scorecard*.json` +
    `comparison_e3.json`; `.../floor_from_trajectory/floor_mesh.ply.scorecard.json`.
  - Scratchpad `e6b/`: `drive_*.json` (all six runs), `bench_*.json`,
    `seam_hazard_profile.json`, `waypoint_raycast_profile.json`, `visual_check.json`.

`E6_drive_report.md` on disk is **finalized** — no placeholders found (the
COURSE_report note about placeholders described an earlier state). Its headline
numbers were nevertheless re-verified against the scratchpad drive JSONs:
wedge counts (11/12/13/7/0), stalls (44/52/58/33/0), completion fractions
(1.0/1.0/1.0/0.9143/0.2016), distances (90.64 / 82.88 / 18.28 m), fall-throughs
(0 everywhere), max roll/pitch, seam census (188/393 cells, max 0.215/0.251 m),
and bench outcomes (cd probe box rest z -1.3604 vs correct -1.6675). All match.

## Cross-checks and discrepancies found

Every number in the report was traced to a report table or a JSON. Discrepancies
found and how they were resolved (all footnoted in the report where they surface):

1. **`full_pass.md` vs code audit (`DOC_report.md`)** — full_pass credits
   `data_factor=2`/`pcd_stride=2` for the VRAM fix and describes training at
   1008x756. The code audit + the run's own `cameras.json`/rendered images show both
   flags are inert and the run trained at full 2016x1512; cap_max/MCMC was the fix.
   Code/artifact evidence wins -> footnote [^df] in the report.
2. **`full_pass.md` §4 "no coordinate conversion needed"** — true for units, false
   for orientation until the E1 fix (3dgrut's baked Volume-prim rotation). Reported
   per E1; the obsolete manual-rotation ritual is called out as obsolete.
3. **PSNR rounding/protocol** — stored 25.548/0.8457/0.3887 (RS-on eval) vs matched
   RS-off 25.415/0.8430/0.3895 vs 2DGS 23.689/0.8217/0.4529. The report quotes the
   stored value as the headline and the matched pair for the 3DGS-vs-2DGS comparison,
   with footnote [^psnr].
4. **Trajectory-floor protocol split** — 0.9780/0.48 m2 (h=1.6683-built floor under
   default-h harness; the four-way tables in E2/E3) vs 0.9866/0.37 m2 (same floor
   under the h=1.6683 harness, `comparison_e6a_h16683.json`; identical to the
   original h=1.60/1.60 numbers). Both quoted with protocol labels; footnote
   [^floorproto].
5. **Report-table rounding vs JSON** — E6a's prose table prints 0.478/0.987/2.9 cm
   etc.; the JSONs carry 0.4782/0.9866/0.0287. The scoreboard uses JSON precision;
   footnote [^scoring].
6. **Spec's executive-summary phrasing "only the fused-best candidate completes
   100%"** — the drive JSONs show FUSED-2DGS also completes 100% (with a worse
   census). The report states the accurate version: only the *fused* candidates
   finish; fused-best has the fewest rescues and zero tip-overs. Report data wins
   over spec phrasing per the spec's own rules.
7. **Spec's "floating-robot disqualification" for convexDecomposition** — the
   benchmark probe was a drop-box, not the robot (box rest 0.32 m above the true
   floor). Described accurately as the box.
8. **Minor**: E2a counts 1,028 trajectory-occupied cells while E5's calibration used
   856 of 1,035 walked cells — different cell definitions in different scripts; each
   number is quoted only in its own context, not reconciled.
9. **sigma_Z table** — recomputed Z^2·sigma_d/(f·B) at f·B = 41.28 m·px: values match
   E3's table (1.2 cm / 4.8 / 30.3 / 121 / 485 at 0.5 px).

## Omissions (deliberate, and why)

- **E1's z-histogram-implied device height (1.41 m)** — a rough estimate superseded
  by the E2a ray-cast calibration (1.6683 m); including both invites confusion and
  the E1 number was never used downstream.
- **E0 synthetic-mesh validation numbers** — summarized as "validated on synthetic
  known-good/known-bad meshes" only; the full table lives in `E0_report.md` and is
  not decision-relevant to the final recipe.
- **Latent code crashers not on the run path** (l1_grad typo details,
  `interpolate_aria_pose` edge case, stale `project()` import) — compressed into one
  line in §2.6; full detail remains in `DOC_report.md`.
- **COURSE_report.md content** — used only as a cross-check corpus (its verified
  numbers agree with the sources used here); the course itself is a separate
  deliverable and is not described in the final report.
- **Per-run stall position lists, bench sub-timings, waypoint-raycast per-candidate
  table** — the drive section quotes the headline table and the seam census; the
  full evidence set is cited by path (scratchpad `e6b/`).
- **Fused heightfield hybrid-collider option** (PhysX heightfield + mesh obstacles)
  — not in the final report's recipe because the drive test validated the
  static-trimesh route; the option remains documented in `E6_candidates_report.md`.

## Structure compliance

Spec structure followed exactly: exec summary; fundamentals (edited from
`fundamentals_draft.md`, file:line citations preserved; "a splat is not a surface";
gauge-freedom/fixed-poses argument; "what the code actually does" subsection);
uniform per-method template (run path -> differentiator -> results vs others ->
measured failure modes) in the spec's order, with the Isaac frame fix +
manual-rotation story inside the visual-splat section and a short eval-harness
preamble before the methods (the harness is an instrument, not a method producing a
USD product); photogrammetry as design+hypothesis only, zero fabricated results;
scoreboard at h=1.6683 with default-h raw values bracketed and footnoted;
limitations & next steps (seam feathering, scan-first capture protocol, 2DGS RS-off
note, Gen 2 eye-gaze MPS check, open photogrammetry slot, residual hazards);
reproduction appendix (env table, script inventory, end-to-end sequence). Plain
names only in headings/prose; one table maps plain names -> report files. All
commands shown are condensed from commands the reports actually ran.

## Review

Adversarial review of `final_report_content.md` (TASKID=REPORT-R, 2026-08-26).
Verdict: **pass with 5 small fixes applied directly** (listed at the end). No
invented, transposed, or unsupported numbers found after independent re-derivation.

### Gate 1 — numbers (every quantitative claim re-verified against JSONs/reports)

- **Scoreboard (§4)**: every main-column value re-checked against
  `collision_candidates/comparison_e6a_h16683.json` (params confirm
  `eye_height_m: 1.6683`): fused-best 0.9696 / 0.40 / 0.0287 / 26.7093 / 1
  non-manifold / 4,120,799 tris; fused-stereo 0.9696 / 0.40 / 0.0288 / 26.7093;
  fused-2DGS 0.9565 / 0.47 / 0.0382 / 85.8328; raw stereo 0.9029 / 1.5 / 0.0218 /
  27.2351; raw 2DGS 0.6163 / 7.21 / 0.0347 / 86.0394; raw 3DGS 0.4782 / 18.88 /
  0.0319 / 341.0382; trajectory floor 0.9866 / 0.37 / 0.4083 / 0.0. All match.
  Bracketed default-h values re-checked against the per-mesh `*.scorecard.json`
  files (all with `eye_height_m: 1.6`): 3DGS 0.5013/19.94, 2DGS 0.6405/8.86,
  stereo 0.8638/0.91, and stereo `scorecard_h16683` 0.9029/1.50. Footnote claim
  that sd-median/junk are h-independent confirmed value-by-value.
- **Trajectory-floor protocol footnote independently reproduced**: I re-ran
  `eval_mesh.py` on the on-disk `floor_from_trajectory/floor_mesh.ply` (the
  h=1.6683 rebuild) at default h=1.6 and got exactly 0.978 / 0.48 m² /
  sd-median 0.4083 — confirming both halves of [^floorproto] and the four-way
  tables in E2b/E3 (0.978/0.48 lines found at E3:257, E2_2dgs:190).
- **Drive-test table**: re-derived from scratchpad `e6b/drive_*.json`:
  fused-best completed 1.0, first wedge 17.53 m, 0 fall-throughs, 44 stalls,
  11 teleports, roll/pitch 42.8/39.0; v05 18.53/52/12/63.1/42.3; fused-2DGS
  6.76/58/13/55.2/32.5; raw stereo 0.9143 (82.88 m), 33/7, pitch 74.6, outcome
  tipped; floor 0.2016 (18.28 m), 0/0/0, roll 70.1/70.8 at both speeds. Report
  table matches (degree values follow E6_drive_report's half-up rounding).
- **Stereo error model**: B=134.961 mm, f=305.867 px, f·B=41.28 m·px and the
  five-row sigma_Z table match E3_report verbatim; crossings 1.28 m / 2.57 m
  support the "~1.3 m / ~2.6 m" prose. Sanity gate re-checked against
  `stereo_depth/sanity_stereo_vs_semidense.json`: pooled median_dz
  −0.00049 m (−0.5 mm), bin 1.5–2.5 m median |dz| 0.0517 m (the "5.2 cm at ~2 m").
- **PSNRs**: recomputed means from the raw per-image `test_logs.json`:
  3DGS stored 25.5476/0.8457/0.3887; 3DGS RS-off (`test_rs_off/summary.json`)
  25.41519/0.84299/0.38947; 2DGS 23.6889/0.8217/0.4529. The report's
  25.55/25.415/23.689, the ~0.13 dB RS-at-eval delta, and the ~1.73 dB honest
  gap all reproduce.
- **Fusion**: `fusion_report.json` files confirm pins +0.0093 m / −0.0337 m,
  the byte-identical 6,294/229/621 classification for fused-best and
  fused-stereo, threshold 0.1 m, strip band 0.40 m / 1-cell dilation.
  `seam_hazard_profile.json` confirms 188 cells / 0.2149 m max / 1.88 m²
  (fused-best) and 393 / 0.2513 m (fused-2DGS).
- **Bench**: `bench_*.json` confirm 7.574 s cold / 1.329 s warm reset,
  0.437–0.447 ms trimesh step, SDF res 256 with +1.13 s setup and equal step
  time. One discrepancy found and fixed (see below): convexDecomposition box
  rest z −1.3604 vs −1.6675 = **0.31 m**, not the 0.32 m in E6_drive_report's
  prose.
- **Fundamentals/visual-splat figures** (155,800 poses, 1,557 calib records,
  10.1 ms readout, 4,292,410 points, 938,113 filtered, 7.7–9.1 M observation
  rows, 33.3 % match rate, 4,719→4,675, 585 held-out frames, 18,816 dropped
  (1.25 %), 1,481,184 kept, ~120 km AABB, 372 MB PLY / 175 MB USDZ, post-fix
  AABB [−48.06,−48.53,−49.76]…[49.07,48.66,47.63], four training attempts,
  ~4 GB peak VRAM, h calibration 1.6683/1.7145/0.2754 over 856 cells,
  1,031/4,675 fused frames, 1,623/975 through-hit cells, 884,386 vs 904,972
  Gaussians, 3 h 47 min retrain, 1.98 GiB isect_tiles OOM, 222.6 s vs ~90 s
  fusion, 1,558 stride-3 frames / 842 fused / 11.0 % LR-masked / 17.5 min /
  3.3 GB) — each located in full_pass.md, E1–E6, E5, or the E2/E3 reports.
  The eval-harness constants (7,144 cells / 71.44 m², 10 cm grid, 0.75 m
  buffer, ±15 cm tol) were additionally confirmed live by my eval_mesh re-run.

### Gate 2 — structure

All eight method sections present in the spec's order, each opening with a
"How do I run this" block, then differentiator, comparative results, measured
failure modes (photogrammetry deviates exactly as the spec directs:
design/hypothesis/judgment, no results). Exec summary, fundamentals-from-the-
ground-up (splat-is-not-a-surface + gauge-freedom included), scoreboard with
the scoring h stated per column, limitations (all five spec items + residual
hazards), reproduction appendix (envs, script inventory, end-to-end) all
present. Plain names in all headings; one mapping table carries the E-codes.

### Gate 3 — honesty

Photogrammetry section contains zero result numbers (only design parameters).
Caveats verified present: stall/rescue framing ("completion = completable with
N rescues"; autonomous number = distance before first wedge), 2DGS PSNR cost,
RS-off forced-not-chosen with the matched re-eval, seam-step discovery with
census, the [^hole] harness-artifact footnote on 19.94 m², teleport-inflation,
powersave-governor and yaw-deadlock caveats. "Passes the drive test" claims are
scoped: both fused candidates complete; only fused-best is "recommended";
raw-stereo/floor failures stated with mechanisms. Exec-summary claim "only the
fused candidates finish the course" is consistent with the data.

### Gate 4 — commands (spot-runs)

Ran verbatim (read-only): `--help` on `eval_mesh.py`, `compare_meshes.py`,
`floor_from_trajectory.py`, `compose_near_far_mesh.py`, `extract_mesh_tsdf.py`,
`extract_mesh_stereo_tsdf.py`, `filter_splat_outliers.py`, `eval_splat_test.py`
(all flags quoted in the report exist), plus one full harness run:
`eval_mesh.py floor_from_trajectory/floor_mesh.ply` → reproduced 0.978/0.48.
GPU/Kit scripts verified by inspection instead of execution:
`isaacsim_build_env.py` (`--collision/--approximation/--splat/--bench/--report`
all present), `sim_drive_test.py` (defaults confirm the quoted protocol:
eye-height 1.6683, spacing 0.25, waypoint-tol 0.40, fall 0.5 m, stall 6 s/6 cm),
`isaacsim_load_splat.py` (usdz positional + `--report`). The quoted training
invocation matches `scripts/bash_local/train_gen2_outside.sh`; all four bash
wrappers exist.

### Gate 5 — citations (25+ spot-checked, all accurate)

Verified against the working tree, including: `extract_aria_vrs.py` 93-127,
96-124, 153-165, 171-172, 173-184, 374, 410, 417-421, 435-448, 510-513,
546-563, 721-724, 777, 952; `aria_utils.py` 100-140, 175-234, 347, 445-447;
`dataset_readers.py` 483, 565-566, 663-675, 687-703; `cameras.py` 496+,
910-918, 949+, 1029-1035; `vanilla_gsplat.py` 613-620, 965-983 (`RGB+ED`),
1101-1110 (hard-coded 0.2, `l1_grad_lamda` typo); `GS2D_gsplat.py` 62-64,
423+; `conf/config.yaml` 41/44/61; `simple_gsplat_30K.yaml` 135;
`filter_splat_outliers.py` (median-centred radius + activated-opacity filter);
`fix_nurec_usdz_frame.py` 48-54, 102-113 (refusal path); `isaacsim_load_splat.py`
26-31, 64+; `utils/point_utils.py` 17-33; `threedgrut/export/usdz_exporter.py`
70-79 (in `/home/sun/3dgrut`). Independently re-ran the inert-key grep:
`data_factor`/`pcd_stride` have zero reads under `scene/ model/ utils/
train_lightning.py` — the §2.6 claim stands on my own evidence, not just
DOC_report's.

### Gate 6 — readability

E-codes confined to the mapping table and file paths after fixes (two bare
usages found and reworded). Terms (VRS, MPS, TSDF context, coverage/hole/junk
definitions, wedge/stall/rescue) are defined before first use.

### Fixes applied directly (5)

1. §3.8 convexDecomposition probe: **0.32 m → 0.31 m** above the true floor —
   the bench JSONs give rest z −1.3604 vs −1.6675 (Δ = 0.307 m); per the
   report's own JSON-wins rule, with the E6_drive_report prose rounding noted
   inline.
2. §3.3: "The E2a strategy fix" → "The MCMC-config fix from the code audit
   (§2.6)" (bare E-code in prose).
3. §3.6: "joins the table on page 1 as the E4 report" → "joins the mapping
   table on page 1 (the reserved E4 slot)" (ties the code to the table).
4. §2.2: RGB sparse depth sentence conflated density with median depth; now
   reads "denser …, and reaches deeper (median depth ~5.1 m vs 2.8 m)"
   (full_pass.md:516-517 gives these as median depths).
5. §3.8: "The first fused-best attempt wedged … at waypoint 70" → "An early
   fused-best attempt (the first to survive spawn)" — E6_drive_report's
   chronology has attempt 1 dying at spawn and attempt 2 wedging at wp 70; the
   old sentence contradicted the report's own later spawn-tip caveat.

### Noted, deliberately not changed

- Degree values in the drive table (43/33/41/21°) follow E6_drive_report's
  half-up rounding of 42.8/32.5/40.5/20.5; ambiguous only at the 0.5 boundary,
  and consistency with the source report was preferred.
- "~1–3 cm effective clearance": per-run JSON `ground_clearance_m` spans
  0.0016–0.0341 m; the phrase is E6_drive_report's own summary and fair.
- The on-disk `floor_from_trajectory/floor_mesh.ply.scorecard.json`
  (0.9866/0.37 at h=1.6) predates the calibrated rebuild of the mesh it names;
  the report correctly relies on the comparison JSON and the four-way tables
  instead, and [^floorproto] already explains the protocol split.

Nothing committed to git.
