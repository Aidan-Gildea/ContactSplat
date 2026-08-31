# E6b — Isaac Sim collision setup + drive test: run report

Date: 2026-08-26. Isaac Sim 5.1.0 (`~/isaac-sim/python.sh`, PhysX 107.3.26), headless
throughout; every result below comes from `--report` JSON files, never from printed
output (Kit swallows stdout and force-exits, as pre-warned). GPU: RTX 4080 SUPER 16 GB,
verified idle before starting (690 MiB desktop use, 3 %); all Isaac runs serialized.
Nothing committed or pushed.

Task: brief E6's Isaac Sim half — build the splat + collision environment, benchmark the
physics approximations, write and run the drive-test acceptance harness over the E6a
candidate set, and pick THE collision asset.

## Deliverables

| File | What |
|---|---|
| `scripts/isaacsim_build_env.py` (new) | loads splat USDZ under `/World/AriaSplat` (visual) + collision mesh under `/World/AriaCollision` (invisible, `UsdPhysics.CollisionAPI`, approximation `none`\|`sdf`\|`convexDecomposition` via CLI); converts PLY→USD with atomic caching; `--bench` measures cooking + steady-state step time with a dynamic drop-box; reports AABBs + E1 frame checks to JSON |
| `scripts/sim_drive_test.py` (new) | the acceptance test: Nova Carter driven along the MPS closed-loop XY path (0.25 m waypoints), slope-tolerant follower, JSON metrics: fall-throughs, stalls/wedges on the walked path, distance before failure, path length |
| `<candidate>.collision.usd` next to each tested PLY | cached PLY→USD conversions (Z-up, metres, `sourcePly`/`sourceAabb*` provenance in customData) |
| scratchpad `e6b/` | all run reports/logs: `bench_*.json`, `visual_check.json`, `drive_*.json`, `waypoint_raycast_profile.json`, `seam_hazard_profile.json` |

Scratchpad = `/tmp/claude-1000/-home-sun-Desktop-aria-proj-egocentric-splats/83deeb14-482b-4b45-b1f1-27ad7da94749/scratchpad`.

## 1. Physics approximation benchmark (task 1)

All on FUSED-BEST (`collision_candidates/fused_best_stereoNear_2dgsFar_trajfloor.ply`,
2,267,606 verts / 4,120,799 tris). PLY→USD conversion: 1.6 s (0.11 s read + 1.5 s
write). Benchmark protocol: static collider (no RigidBodyAPI), friction material
0.8/0.7 bound to the mesh, one 0.3 m dynamic cube dropped at the trajectory start
(floor there = −1.83 m ⇒ correct rest z = −1.68), `world.reset()` timed as
setup/cooking, then 300 physics steps at 60 Hz timed after 20 warmup steps
(`world.step(render=False)`).

| approximation | setup (`world.reset`) | first step | steady step mean / p95 | box rest z | verdict |
|---|---|---|---|---|---|
| **none** (static trimesh) cold | **7.57 s** | 0.001 s | **0.437 / 0.536 ms** | **−1.6675** ✓ | **winner** |
| none, warm cook cache | 1.33 s | 0.001 s | 0.447 ms | −1.6675 ✓ | (2nd process run; PhysX cooked-data cache) |
| sdf (resolution 256) | 8.70 s | 0.001 s | 0.425 / 0.537 ms | −1.6675 ✓ | no benefit for a **static** collider |
| convexDecomposition | 2.77 s | 0.001 s | 0.414 / 0.515 ms | **−1.3604 ✗** (slid to y+0.45) | **disqualified — wrong geometry** |

Reads:

- **The brief's "do not use raw triangle-mesh collision" concern does not materialize
  for a *static* collider.** PhysX 5 cooks a 4.1 M-triangle static mesh in 7.6 s
  (1.3 s warm) and steps convex-vs-trimesh contacts in ~0.44 ms — the robot
  articulation itself dominates the drive-test step (11 ms, §3). Raw-trimesh cost is a
  dynamic-mesh problem; the environment is static.
- **SDF buys nothing here**: identical step time, +1.1 s setup, and the log shows no
  SDF-specific activity — SDF contact generation matters for *dynamic* SDF meshes;
  static-vs-convex pairs use the normal midphase either way. Both numbers recorded as
  the brief asked; `none` is the winner and is used for all drive tests.
- **convexDecomposition is not merely slow, it is wrong**: 32 convex hulls over a
  concave 25 m terrain bridge every hollow — the test box comes to rest 0.32 m above
  the true floor and slides. Any drive test on it would measure hull soup, not the
  scene. Disqualified on correctness.

## 2. Load-only visual check (task 4) — no units/axis regression

`isaacsim_build_env.py --splat default --collision <FUSED-BEST> --approximation none`
(`visual_check.json`):

| Check | Result |
|---|---|
| NuRec prims resolved under `/World/AriaSplat` | 2 ✓ |
| splat Volume composed transform | **identity** ✓ (E1 fix intact) |
| splat world AABB | (−48.062, −48.531, −49.756) … (49.070, 48.659, 47.626) — **byte-for-byte the E1-validated MPS-frame PLY bounds** |
| collision world AABB | (−12.811, −11.905, −2.861) … (12.408, 12.970, 4.959) — equals the FUSED-BEST scorecard bbox to float32 rounding |
| collision AABB ⊂ splat AABB | true ✓; center offset (−0.71, 0.47, 2.11) m, expected — the splat AABB is the ±50 m outlier-filter shell, its content median centre is (−0.10, −0.35, −1.31) per E1 |
| metric floor cross-check | bench box rests at z = −1.6675 on the collision mesh ⇒ floor at −1.82, matching the calibrated trajectory floor at that XY (−1.83) to ~1 cm ✓ |

Both assets share the MPS frame with **no registration transform anywhere** — the E1
invariant holds in the composed Isaac stage.

## 3. The drive test (tasks 2–3)

### Robot: Nova Carter (exactly as spawned)

- Asset: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd`
- Differential drive: wheel joints `joint_wheel_left`/`joint_wheel_right`, wheel radius
  0.14 m, wheel base 0.4132 m (values from Isaac Sim's own Carter tests), velocity
  control mode, `DifferentialController` limits v ≤ 0.8 m/s, ω ≤ 1.5 rad/s.
- **Clearance (measured in-sim at settle)**: the `chassis_link` subtree's AABB bottom
  sits 0.0137 m above the local floor expectation — i.e. essentially touching: Nova
  Carter contacts through its two 0.28 m drive wheels plus low caster assemblies, and
  has effectively **a few cm of obstacle clearance**. This number decides several
  results below.

### Path and follower

- Waypoints: `closed_loop_trajectory.csv` (155,800 poses) subsampled to ≥0.25 m XY
  spacing → **363 waypoints, 90.64 m**; per-waypoint expected floor = trajectory z −
  1.6683 (calibrated h), then a ±1 m rolling-median clean that replaced **4 posture
  outliers** (the wearer crouching produces up to 0.85 m dips in raw device z — found
  when the raw max segment grade read 313 %; cleaned max segment grade 0.72, p95 0.32).
- Follower: heading-error P-turn (ω = 2·err, capped), v = vmax·cos(err) gated off above
  80° error; waypoint reached at 0.40 m with 8-waypoint lookahead; stall = commanded
  v with <6 cm XY displacement over 6 s; 2 stalls → reverse-and-retry, 3rd stall at the
  same waypoint = **wedge**: recorded, then the robot is teleported 2 waypoints ahead
  so one wedge cannot blind the rest of the path (`distance_before_first_wedge_m`
  keeps the honest "distance before failure" number).
- Fall-through: base z (settle-calibrated) > 0.5 m below the expected local floor,
  nearest-waypoint within a ±30-index window so multi-pass grade doesn't alias.

### Run-by-run results

All runs: Nova Carter, `--approximation none`, identical follower settings
(vmax 0.8 m/s unless noted), same 363-waypoint / 90.64 m path, physics 60 Hz. A
"wedge" = 3 stalls at one waypoint with reverse-recovery failing twice → the robot is
teleported 2 waypoints ahead (recorded; a real robot would need rescue there).
`dist-1st-wedge` is the honest "distance completed before failure"; completion counts
the full census with recoveries. Full JSONs in scratchpad `e6b/drive_*.json`.

| candidate | outcome | completion | dist before 1st wedge | fall-throughs | stalls | wedges | max roll/pitch | z-dev p95 / max | step ms | sim s |
|---|---|---|---|---|---|---|---|---|---|---|
| **FUSED-BEST** | **completed** (11 recoveries) | **100 %** | 17.5 m (wp 70) | **0** | 44 | 11 | 43° / 39° | 0.139 / 0.278 m | 10.8 | 520 |
| FUSED-BEST @ vmax 0.5 | **completed** (12 recoveries) | **100 %** | 18.5 m (wp 74) | **0** | 52 | 12 | 63° / 42° | 0.172 / 0.233 m | 10.3 | 621 |
| FUSED-2DGS | completed (13 recoveries) | 100 % | **6.8 m (wp 27)** | **0** | 58 | 13 | 55° / 33° | 0.140 / 0.375 m | 11.4 | 692 |
| RAW-STEREO | **tipped** (pitch 75°) | 91.4 % (82.9 m) | 17.5 m (wp 70) | **0** | 33 | 7 | 41° / **75°** | 0.192 / 0.218 m | 15.1 | 431 |
| E5-FLOOR | **tipped** (roll 70°) | **20.2 % (18.3 m)** | — (0 stalls, then flip) | **0** | 0 | 0 | **70°** / 21° | 0.175 / 0.253 m | 1.4 | 26 |
| E5-FLOOR @ vmax 0.5 | **tipped** (roll 71°) | 20.2 % | — | **0** | 0 | 0 | 71° / 21° | 0.162 / 0.269 m | 1.4 | 39 |

Reads (each verified against the JSON evidence and the offline probes):

- **Zero fall-throughs on every candidate, at every speed.** The failure class E6a set
  out to eliminate stays eliminated under real physics: no run ever had the base drop
  more than 0.28 m below the expected local floor (threshold 0.5 m). The E1 frame
  invariant plus the fused floors deliver exactly what they promised.
- **Only the fused candidates finish the course.** FUSED-BEST completes at both speeds;
  its wedge census (11–12) is the best of the fused set and its wedge sites are
  geometric, not speed-induced (same sites at 0.5 m/s).
- **FUSED-2DGS also finishes but is the roughest ride**: first wedge at 6.8 m, 58
  stalls, 13 rescues — matching its 2× seam-hazard census (§4). Its spawn also slid
  0.6 m before settling (accepted at roll 3°). Strictly dominated by FUSED-BEST.
- **RAW-STEREO ends in an unrecoverable rollover** at 82.9 m (pitch 75° climbing smear
  near (0.85, −0.4)), after 7 wedges. Fewest stalls while it lasted — its floor has no
  fusion seams — but a flip is robot loss, strictly worse than a stall, and its floor
  sits off the calibrated height (fusion pin was +0.9 cm; z-dev p95 0.19 m here).
- **E5-FLOOR is catastrophically bad as a *collision* asset despite its perfect
  scorecard**: the robot cruises at near-vmax (step 1.4 ms) for 18 m and then **rolls
  over in the wp-70 depression at both 0.8 and 0.5 m/s** — the smooth prior floor
  renders the real ~15 cm gutter with clean steep walls and nothing else; one wheel
  drops in and nothing arrests the roll. 0 stalls then a flip. The floor-only
  heightfield remains useful as a *prior*, not as the collider.
- **The wp-70 area (x≈3.2, y≈3.6–4.2) is the scene's acid test** — every candidate
  fails there first, each in its own way: fused = seam trench wedge; raw stereo = wedge
  against a 1.8 m-tall, ~10 cm-wide pillar 0.2 m off-path at (3.37, 4.21) (real pole or
  stereo smear — present in all stereo-based meshes); E5 = rollover in the clean dip.
  The trajectory prior itself says the walked line dips ~15 cm there, so much of this
  is *real terrain* that a ~1–3 cm-clearance differential robot cannot take at the
  head-path line — human-walkable does not imply Carter-drivable.
- Step cost ranking matches mesh weight: E5 1.4 ms (14 k tris) < FUSED-BEST 10.8 <
  FUSED-2DGS 11.4 < RAW-STEREO 15.1 ms (junk components add contact work; note the
  robot articulation dominates — the empty-scene benchmark was 0.44 ms).


## 4. What the first FUSED-BEST attempt taught (failures + fixes)

Chronology of real failures hit and fixed (all reproducible from the logs):

1. **Three conversion-layer bugs** on first launches of `isaacsim_build_env.py`:
   `Gf.Vec3f` rejects numpy scalars; USD `customData` rejects Python lists (needs
   `Gf.Vec3d`); and a crashed conversion left a truncated `.collision.usd` that the
   next run took as a cache hit ("no Mesh prim found"). Fixed: float casts, `Gf.Vec3d`,
   and write-to-temp + atomic rename.
2. **Robot flipped at spawn (first drive attempt)**: dropped 0.30 m onto wp0 with an
   0.87 m-tall stereo-smear blob 0.4 m off-centre inside the footprint → tipped during
   settle, run dead at 0 m. Fixed: 0.10 m drop + settle-retry at +2/+4/+8 waypoints
   (tilt >20° or off-height rejects the spawn). The final runs all settle at attempt 1
   with roll −7.0°, pitch −4.7°.
3. **Raw trajectory z is not a floor** (§3): rolling-median clean added.
4. **Wedge at wp 70** (second attempt, run otherwise healthy): robot immovable at
   (3.22, 4.20) — 15 stalls at the identical position, reverse-recovery useless, run
   over at 18.3 m/20 %. Root-caused offline (see below) and answered with the
   teleport-recovery protocol so the census covers the full path.

### The wp-70 wedge is a *fusion seam*, not noise — the drive test's key finding

Fine raycast probing of FUSED-BEST around the stall point shows a **~25 cm-wide,
10–15 cm-deep slot** running exactly along the walked line (y 3.5→4.2 at x ≈ 3.15),
with flat kept-mesh floor at −1.83…−1.85 on both sides. The fused heightfield's
provenance array explains it: those cells are `source=3` (parent mesh **rejected** —
it sat >10 cm above the trajectory prior) so the fusion replaced them with a floor
patch at prior height (−1.95…−1.99), while the neighbouring cells stayed `source=1`
(parent mesh kept at −1.81…−1.85). The result is a curb-edged trench one
robot-track wide: drive wheels drop in, the ~1–3 cm-clearance chassis grounds on the
edges, and a differential robot is beached — forward and reverse both dead.

**Why E6a could not see this**: the E0 harness scores each cell's height against that
cell's own expectation (±15 cm) — every trench cell *and* every shelf cell passes
individually. Lateral discontinuity is invisible to the per-cell metric; it took the
drive test to expose it. This is precisely the "acceptance test decides" logic of the
brief working as intended.

Census of this hazard class over the fused heightfields (patch cells whose height
steps >8 cm against an adjacent kept-mesh cell — `seam_hazard_profile.json`):

| candidate | patch cells | seam-step cells >8 cm | max step | seam area |
|---|---|---|---|---|
| FUSED-STEREO / FUSED-BEST | 850 | **188** | 0.21 m | 1.88 m² |
| FUSED-2DGS | 3,032 | **393** | 0.25 m | 3.93 m² |

Raw meshes have no *fusion* seams (their own TSDF steps are what the fused floor was
built to fix); the E5 floor is seam-free by construction. The drive results bear this
prediction out.

### Offline waypoint raycast profile (context for the drive numbers)

Straight-down first hits at every waypoint centre (`waypoint_raycast_profile.json`),
after the posture clean, on the calibrated floor expectation:

| candidate | no-hit | first hit >0.30 m above floor (obstacle in corridor) | hit >0.5 m below | median dz | p95 dz |
|---|---|---|---|---|---|
| FUSED-BEST | 0 | 3 (wp 36, 264, 323) | 0 | +0.001 | +0.095 |
| FUSED-2DGS | 0 | 9 | 0 | −0.005 | +0.104 |
| RAW-STEREO | 0 | 4 | 0 | −0.006 | +0.128 |
| E5-FLOOR | 0 | **0** | 0 | −0.001 | +0.045 |

Zero no-hit and zero deep-low cells on every candidate — consistent with E6a's
"fall-through class structurally eliminated" for the fused meshes, and (at waypoint
centres) even for raw stereo.

## 5. Final ranking and recommendation (task 5)

Drive metrics first, scorecard (E6a, h = 1.6683) second:

| rank | candidate | drive verdict | scorecard (cover / hole / sd-med) | tris |
|---|---|---|---|---|
| **1** | **FUSED-BEST** (`collision_candidates/fused_best_stereoNear_2dgsFar_trajfloor.ply`) | only candidate: 100 % completion at both speeds, 0 fall-throughs, 0 tip-overs, fewest fused-set wedges (11) | best full-scene: 0.970 / 0.40 m² / 2.9 cm | 4.12 M |
| 2 | FUSED-STEREO (lean twin) | not driven separately — byte-identical floor classification and identical near-field geometry to FUSED-BEST inside the corridor (E6a §3); expected to behave the same minus far-field walls | 0.970 / 0.40 m² / 2.9 cm | 3.11 M |
| 3 | FUSED-2DGS | completes, but first wedge at 6.8 m, 58 stalls / 13 rescues | 0.957 / 0.47 m² / 3.8 cm | 6.30 M |
| 4 | RAW-STEREO | unrecoverable rollover at 91 % | 0.903 / 1.50 m² / 2.2 cm | 3.26 M |
| 5 | E5-FLOOR mesh | rollover at 20 % at both speeds; no obstacles at all | 0.987 / 0.37 m² / — (floor-only) | 14 k |
| 6 | raw 2DGS / 3DGS | not driven — strictly dominated on every scorecard metric and superseded by their fusions | 0.616 / 0.478 cover | 6.6 / 10.7 M |

### Recommendation

**THE collision asset: `output/Outside_20260812_141244/collision_candidates/fused_best_stereoNear_2dgsFar_trajfloor.ply`,
converted to USD (`.collision.usd` sibling, cached) and mounted as a static collider
with `physics:approximation = "none"` (static triangle mesh), friction ≈ 0.8/0.7,
invisible under `/World/AriaCollision`, next to the splat USDZ under `/World/AriaSplat`
— exactly what `scripts/isaacsim_build_env.py` builds.** Cooking is 7.6 s cold / 1.3 s
warm, steady physics cost 0.44 ms/step (environment alone), and the E1 frame invariant
was re-verified in the composed stage (§2). It is the only candidate that carried a
robot over the full demonstrated path, with the best scorecard of the full-scene set.

### Residual risks (ranked by evidence)

1. **Fusion seam steps** — the drive test's central discovery. Where the fusion rejects
   the parent floor (`source=3`) it patches at prior height, leaving curb-like 8–21 cm
   step edges one robot-track wide (188 cells / 1.88 m² on FUSED-BEST; 8 of its 11
   wedge sites). Per-cell scorecards are structurally blind to this. **Mitigation for a
   v2: feather the patch↔kept-mesh transition over ~3–5 cells in
   `floor_from_trajectory.py --fuse-with`, then re-gate and re-drive.**
2. **Thin near-field pillars/blobs** (E6a's deliberately-kept 26.7 m² junk class): at
   least one 1.8 m × ~10 cm pillar stands 0.2 m off the walked line at (3.37, 4.21) and
   wedges/tips robots. It may be a real pole — verify the handful of on-corridor
   pillars against the splat rendering before deleting anything; a blanket filter
   remains risky, targeted removal is not.
3. **Robot-envelope mismatch, not mesh error**: Nova Carter's measured effective
   clearance is ~1–3 cm (chassis-subtree AABB 1.4 cm above floor at settle) against a
   mesh whose honest cm-scale roughness and real ~15 cm gutter produce stalls a
   higher-clearance platform would shrug off. Clearpath Jackal and Dingo are on the
   same asset server (probed 200 OK) if the project wants a second-robot data point;
   likewise a path-planner with lateral offsets (instead of replaying the raw head
   path 0.2 m from obstacles) would remove a further share of the stalls.
4. **The wp-70 gutter is real**: even a perfect mesh will fail a low-clearance robot
   there. Any downstream demo should either route around (x≈3.2, y≈3.6–4.2) or accept
   that segment as a known hazard.
5. **Follower/metric caveats**: stall counting needed a yaw-deadlock trigger (added:
   no waypoint advance + no XY motion for 20 s); FUSED-BEST's headline run predates
   that trigger but never entered the state (verified: it advanced continuously or
   stalled with forward command). Teleport recovery means "completion 100 %" reads as
   "completable with N rescues", never as autonomous success; `dist-before-first-wedge`
   is the autonomous number. CPU governor was `powersave` — all step timings are
   conservative upper bounds.


## Commands run (chronological, deduplicated)

```bash
SCR=<scratchpad>/e6b
REPO=/home/sun/Desktop/aria_proj/egocentric_splats
CAND=$REPO/output/Outside_20260812_141244/collision_candidates
BEST=$CAND/fused_best_stereoNear_2dgsFar_trajfloor.ply

# benchmarks (task 1)
~/isaac-sim/python.sh scripts/isaacsim_build_env.py --collision $BEST \
    --approximation none --bench --report $SCR/bench_none.json          # + warm re-run
~/isaac-sim/python.sh scripts/isaacsim_build_env.py --collision $BEST \
    --approximation sdf --bench --report $SCR/bench_sdf.json
~/isaac-sim/python.sh scripts/isaacsim_build_env.py --collision $BEST \
    --approximation convexDecomposition --bench --report $SCR/bench_cd.json

# visual check (task 4)
~/isaac-sim/python.sh scripts/isaacsim_build_env.py --collision $BEST \
    --approximation none --splat default --report $SCR/visual_check.json

# drive tests (tasks 2-3), all with --approximation none
~/isaac-sim/python.sh scripts/sim_drive_test.py --collision $BEST \
    --report $SCR/drive_fused_best.json
~/isaac-sim/python.sh scripts/sim_drive_test.py --collision $CAND/fused_2dgs_op03_trajfloor.ply \
    --report $SCR/drive_fused_2dgs.json
~/isaac-sim/python.sh scripts/sim_drive_test.py --collision $CAND/raw_stereo_tsdf_d40.ply \
    --report $SCR/drive_raw_stereo.json
~/isaac-sim/python.sh scripts/sim_drive_test.py \
    --collision output/Outside_20260812_141244/floor_from_trajectory/floor_mesh.ply \
    --report $SCR/drive_e5_floor.json

# speed-sensitivity probes (--vmax 0.5): fused_best_v05, e5_floor_v05
```

All runs `timeout`-guarded and detached; JSON reports written continuously during the
drive so a wedged Kit leaves evidence. No run had to be killed.
