#!/usr/bin/env python3
"""E6 acceptance test: drive a wheeled robot along the human's walked path.

Spawns a wheeled robot from the Isaac Sim asset library (Nova Carter by
default) on a candidate collision mesh, drives the XY waypoint path taken
from the MPS closed-loop trajectory (subsampled to ~0.25 m spacing), and
logs per-run metrics to JSON:

- fall-throughs: robot base drops >0.5 m below the expected local floor
  (trajectory z minus calibrated eye height, evaluated at the nearest
  waypoint in a local window so multi-pass grade changes don't alias);
- spurious collisions/stalls on the walked path (the human walked it, so it
  is traversable by definition): commanded forward motion with no XY
  progress over a time window;
- distance completed before failure, and the total path length.

The follower is slope-tolerant by construction (velocity-controlled wheels +
a friction material bound to the mesh in isaacsim_build_env.setup_collision_prim);
the terrain has ~1.3 m of grade.

Run with Isaac Sim's own interpreter (headless), e.g.:
    ~/isaac-sim/python.sh scripts/sim_drive_test.py \
        --collision output/.../fused_best_stereoNear_2dgsFar_trajfloor.ply \
        --approximation none --report /path/drive_report.json

Kit swallows stdout and force-exits: the JSON report (written continuously
during the run) is the only reliable evidence.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DEFAULT_MPS = "/home/sun/aria/mps_Outside_20260812_141244_vrs/slam"

ROBOTS = {
    # name: (usd subpath on the assets root, wheel joint names, wheel radius m,
    #        wheel base m)
    "nova_carter": (
        "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        ["joint_wheel_left", "joint_wheel_right"], 0.14, 0.4132,
    ),
    "jetbot": (
        "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
        ["left_wheel_joint", "right_wheel_joint"], 0.0325, 0.1125,
    ),
    "create3": (
        "/Isaac/Robots/iRobot/Create3/create_3.usd",
        ["left_wheel_joint", "right_wheel_joint"], 0.03575, 0.233,
    ),
}


def load_waypoints(mps_dir, spacing, eye_height):
    """closed_loop_trajectory.csv -> waypoints (x, y, floor_z) at ~spacing.

    floor_z is trajectory z minus eye height, then cleaned with a rolling
    median (+-1 m of arc length): the wearer crouching/bending produces up to
    0.85 m dips in raw device z that are posture, not terrain (cf. E0/E5).
    """
    csv_path = os.path.join(mps_dir, "closed_loop_trajectory.csv")
    xyz = np.loadtxt(
        csv_path, delimiter=",", skiprows=1, usecols=(3, 4, 5), dtype=np.float64
    )
    pts = []
    last = None
    for p in xyz:
        if last is None or math.hypot(p[0] - last[0], p[1] - last[1]) >= spacing:
            pts.append([p[0], p[1], p[2] - eye_height])
            last = p
    wps = np.array(pts)
    raw_floor = wps[:, 2].copy()
    half = max(1, int(round(1.0 / spacing)))  # +-1 m of path
    smoothed = np.array(
        [
            np.median(raw_floor[max(0, i - half): i + half + 1])
            for i in range(len(raw_floor))
        ]
    )
    # keep raw where it agrees; replace posture outliers with the local median
    outlier = np.abs(raw_floor - smoothed) > 0.30
    wps[:, 2] = np.where(outlier, smoothed, raw_floor)
    seg = np.hypot(np.diff(wps[:, 0]), np.diff(wps[:, 1]))
    arclen = np.concatenate([[0.0], np.cumsum(seg)])
    return wps, arclen, int(outlier.sum())


def yaw_from_quat(q):
    """q scalar-first [w,x,y,z] -> yaw (Z-up)."""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_pitch_from_quat(q):
    w, x, y, z = q
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    return roll, pitch


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision", required=True, help="collision mesh (.ply/.usd)")
    parser.add_argument("--approximation", default="none",
                        choices=("none", "sdf", "convexDecomposition"))
    parser.add_argument("--sdf-resolution", type=int, default=256)
    parser.add_argument("--mps-dir", default=DEFAULT_MPS)
    parser.add_argument("--eye-height", type=float, default=1.6683)
    parser.add_argument("--spacing", type=float, default=0.25)
    parser.add_argument("--robot", default="nova_carter", choices=sorted(ROBOTS))
    parser.add_argument("--vmax", type=float, default=0.8, help="m/s")
    parser.add_argument("--wmax", type=float, default=1.5, help="rad/s")
    parser.add_argument("--waypoint-tol", type=float, default=0.40, help="m")
    parser.add_argument("--lookahead", type=int, default=8,
                        help="waypoints ahead that also count as 'reached'")
    parser.add_argument("--fall-threshold", type=float, default=0.5, help="m")
    parser.add_argument("--stall-window", type=float, default=6.0, help="s")
    parser.add_argument("--stall-dist", type=float, default=0.06, help="m")
    parser.add_argument("--deadlock-window", type=float, default=20.0,
                        help="s without waypoint advance AND without XY motion "
                        "that also counts as a stall (catches yaw-deadlock "
                        "wedges where commanded forward speed is ~0)")
    parser.add_argument("--max-sim-seconds", type=float, default=900.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    report = {
        "ok": False,
        "task": "sim_drive_test",
        "status": "starting",
        "collision_input": os.path.abspath(args.collision),
        "approximation": args.approximation,
        "robot": args.robot,
        "eye_height_m": args.eye_height,
        "waypoint_spacing_m": args.spacing,
        "argv": sys.argv[1:],
    }

    def write_report():
        tmp = args.report + ".tmp"
        with open(tmp, "w") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp, args.report)

    write_report()

    # ---- waypoints (pure numpy; before Kit so a bad CSV fails fast) ----
    wps, arclen, n_posture_outliers = load_waypoints(
        args.mps_dir, args.spacing, args.eye_height
    )
    n_wp = len(wps)
    seg = np.hypot(np.diff(wps[:, 0]), np.diff(wps[:, 1]))
    grade = np.abs(np.diff(wps[:, 2])) / np.maximum(seg, 1e-6)
    report["n_posture_outlier_waypoints"] = n_posture_outliers
    report["n_waypoints"] = int(n_wp)
    report["total_path_m"] = round(float(arclen[-1]), 2)
    report["floor_z_range_m"] = [round(float(wps[:, 2].min()), 3),
                                 round(float(wps[:, 2].max()), 3)]
    report["max_segment_grade"] = round(float(grade.max()), 3)
    report["p95_segment_grade"] = round(float(np.percentile(grade, 95)), 3)
    write_report()

    from isaacsim import SimulationApp  # noqa: E402

    t_app0 = time.perf_counter()
    simulation_app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    report["app_startup_s"] = round(time.perf_counter() - t_app0, 2)

    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )
        from isaacsim.storage.native import get_assets_root_path

        from isaacsim_build_env import (
            resolve_collision_usd,
            setup_collision_prim,
            compute_world_aabb,
        )

        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Isaac Sim assets root not found (no network?)")
        usd_sub, wheel_joints, wheel_radius, wheel_base = ROBOTS[args.robot]
        robot_usd = assets_root + usd_sub
        report["robot_usd"] = robot_usd
        report["wheel_radius_m"] = wheel_radius
        report["wheel_base_m"] = wheel_base

        # ---- stage ----
        usd_path, conv_info = resolve_collision_usd(args.collision)
        report["collision_usd"] = usd_path
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0,
                      rendering_dt=1.0 / 60.0)
        t0 = time.perf_counter()
        col_info = setup_collision_prim(
            stage, usd_path,
            approximation=args.approximation,
            sdf_resolution=args.sdf_resolution,
        )
        report["collision"] = col_info
        report["collision_world_aabb"] = compute_world_aabb(stage, "/World/AriaCollision")

        # spawn pose: first waypoint, facing the second, dropped from 0.1 m up
        yaw0 = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
        q0 = np.array([math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)])
        spawn = np.array([wps[0][0], wps[0][1], wps[0][2] + 0.10])
        robot = WheeledRobot(
            prim_path="/World/Robot",
            name="drive_robot",
            wheel_dof_names=wheel_joints,
            create_robot=True,
            usd_path=robot_usd,
            position=spawn,
            orientation=q0,
        )
        world.scene.add(robot)

        # wait for the (possibly remote) robot asset to finish loading
        t0 = time.perf_counter()
        for _ in range(36000):
            _, _, loading = omni.usd.get_context().get_stage_loading_status()
            if loading == 0:
                break
            simulation_app.update()
        report["asset_load_wait_s"] = round(time.perf_counter() - t0, 2)
        report["status"] = "assets_loaded"
        write_report()

        t0 = time.perf_counter()
        world.reset()  # physics init + collision cooking
        report["world_reset_s"] = round(time.perf_counter() - t0, 3)

        controller = DifferentialController(
            name="diff", wheel_radius=wheel_radius, wheel_base=wheel_base,
            max_linear_speed=args.vmax, max_angular_speed=args.wmax,
        )

        dt = 1.0 / 60.0

        # ---- settle, with spawn retry ----
        # The first spawn spot can carry parent-mesh junk inside the robot
        # footprint (observed: an 0.87 m blob 0.4 m from wp0 on the fused
        # stereo mesh flipped the robot on drop). If the robot ends the settle
        # phase tilted or grossly off-height, retry a little further along the
        # walked path before declaring anything.
        settle_steps = int(args.settle_seconds / dt)
        spawn_attempts = []
        start_wp = 0
        settled_ok = False
        for wp_off in (0, 2, 4, 8):
            i0 = wp_off
            if i0 + 1 >= n_wp:
                break
            yaw_i = math.atan2(
                wps[i0 + 1][1] - wps[i0][1], wps[i0 + 1][0] - wps[i0][0]
            )
            q_i = np.array([math.cos(yaw_i / 2), 0.0, 0.0, math.sin(yaw_i / 2)])
            robot.set_world_pose(
                position=np.array([wps[i0][0], wps[i0][1], wps[i0][2] + 0.10]),
                orientation=q_i,
            )
            try:
                robot._articulation_view.set_velocities(np.zeros((1, 6)))
            except Exception:  # noqa: BLE001
                pass
            for _ in range(settle_steps):
                robot.apply_wheel_actions(controller.forward([0.0, 0.0]))
                world.step(render=False)
            pos, quat = robot.get_world_pose()
            s_roll, s_pitch = roll_pitch_from_quat(quat)
            dz = float(pos[2]) - float(wps[i0][2])
            attempt = {
                "start_wp": i0,
                "settle_pos": [round(float(v), 4) for v in pos],
                "roll_deg": round(math.degrees(s_roll), 1),
                "pitch_deg": round(math.degrees(s_pitch), 1),
                "base_offset_m": round(dz, 4),
            }
            spawn_attempts.append(attempt)
            if abs(s_roll) < math.radians(20) and abs(s_pitch) < math.radians(20) \
                    and -args.fall_threshold < dz < 1.0:
                start_wp = i0
                settled_ok = True
                break
        report["spawn_attempts"] = spawn_attempts
        if not settled_ok:
            report["status"] = "finished"
            report["outcome"] = (
                "fell_through" if spawn_attempts and
                spawn_attempts[-1]["base_offset_m"] < -args.fall_threshold
                else "unstable_at_spawn"
            )
            report["n_fall_throughs"] = int(report["outcome"] == "fell_through")
            report["n_stalls"] = 0
            report["distance_completed_m"] = 0.0
            report["completion_frac"] = 0.0
            report["ok"] = True
            write_report()
            return
        pos, quat = robot.get_world_pose()
        base_offset = float(pos[2]) - float(wps[start_wp][2])
        report["start_wp"] = start_wp
        report["settled_base_offset_m"] = round(base_offset, 4)
        report["settle_pos"] = [round(float(v), 4) for v in pos]
        # ground clearance: chassis AABB bottom minus local floor
        for chassis_name in ("chassis_link", "base_link", "create_3"):
            aabb = compute_world_aabb(stage, f"/World/Robot/{chassis_name}")
            if aabb is not None:
                report["chassis_prim"] = chassis_name
                report["chassis_aabb_z"] = [round(aabb[0][2], 4), round(aabb[1][2], 4)]
                report["ground_clearance_m"] = round(
                    aabb[0][2] - float(wps[start_wp][2]), 4
                )
                break
        report["status"] = "driving"
        write_report()

        # ---- drive loop ----
        i_wp = start_wp + 1
        max_reached = start_wp
        sim_t = 0.0
        stalls = []            # {waypoint, sim_t, pos}
        teleports = []         # wedge recoveries: {from_wp, to_wp, sim_t, pos}
        first_wedge_wp = None
        skipped_wps = []
        fall_events = []
        max_abs_roll = 0.0
        max_abs_pitch = 0.0
        z_devs = []            # sampled base-z deviation from expected
        pos_hist = deque()     # (sim_t, x, y)
        cmd_hist = deque()     # (sim_t, v_cmd)
        per_wp_stalls = {}
        state = "NORMAL"       # or REVERSING
        state_until = 0.0
        reverse_turn = 1.0
        last_advance_t = 0.0   # sim_t of the last waypoint advance/teleport
        outcome = None
        last_partial = 0.0
        step_times = []

        def expected_floor(x, y, hint):
            lo = max(0, hint - 30)
            hi = min(n_wp, hint + 6)
            w = wps[lo:hi]
            d = np.hypot(w[:, 0] - x, w[:, 1] - y)
            j = int(np.argmin(d))
            return float(w[j][2]), float(d[j])

        while sim_t < args.max_sim_seconds:
            pos, quat = robot.get_world_pose()
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            yaw = yaw_from_quat(quat)
            roll, pitch = roll_pitch_from_quat(quat)
            max_abs_roll = max(max_abs_roll, abs(roll))
            max_abs_pitch = max(max_abs_pitch, abs(pitch))

            # fall-through / tip checks
            efloor, edist = expected_floor(x, y, i_wp)
            z_dev = (z - base_offset) - efloor
            if len(z_devs) < 200000 and int(sim_t / dt) % 6 == 0:
                z_devs.append(z_dev)
            if z_dev < -args.fall_threshold:
                fall_events.append(
                    {"sim_t": round(sim_t, 2), "pos": [round(x, 3), round(y, 3), round(z, 3)],
                     "expected_floor": round(efloor, 3), "waypoint": i_wp}
                )
                outcome = "fell_through"
                break
            if abs(roll) > math.radians(70) or abs(pitch) > math.radians(70):
                outcome = "tipped"
                break

            # waypoint advance (with limited lookahead for corner cutting)
            for j in range(i_wp, min(i_wp + args.lookahead, n_wp)):
                if math.hypot(wps[j][0] - x, wps[j][1] - y) <= args.waypoint_tol:
                    i_wp = j + 1
                    last_advance_t = sim_t
                    break
            max_reached = max(max_reached, i_wp - 1)
            if i_wp >= n_wp:
                outcome = "completed"
                break

            # controller
            tx, ty = wps[i_wp][0], wps[i_wp][1]
            err = wrap_angle(math.atan2(ty - y, tx - x) - yaw)
            if state == "REVERSING":
                v_cmd = -0.3
                w_cmd = 0.6 * reverse_turn
                if sim_t >= state_until:
                    state = "NORMAL"
                    pos_hist.clear()
                    cmd_hist.clear()
            else:
                w_cmd = max(-args.wmax, min(args.wmax, 2.0 * err))
                v_cmd = args.vmax * max(0.0, math.cos(err)) if abs(err) < math.radians(80) else 0.0
            robot.apply_wheel_actions(controller.forward([v_cmd, w_cmd]))

            t0 = time.perf_counter()
            world.step(render=False)
            step_times.append(time.perf_counter() - t0)
            sim_t += dt

            # stall detection (only while trying to move forward)
            pos_hist.append((sim_t, x, y))
            cmd_hist.append((sim_t, v_cmd))
            while pos_hist and pos_hist[0][0] < sim_t - args.stall_window:
                pos_hist.popleft()
                cmd_hist.popleft()
            if (
                state == "NORMAL"
                and pos_hist
                and sim_t - pos_hist[0][0] >= args.stall_window * 0.95
                and math.hypot(x - pos_hist[0][1], y - pos_hist[0][2]) < args.stall_dist
                and (
                    np.mean([c[1] for c in cmd_hist]) > 0.15
                    or sim_t - last_advance_t > args.deadlock_window
                )
            ):
                stalls.append(
                    {"sim_t": round(sim_t, 2), "waypoint": i_wp,
                     "pos": [round(x, 3), round(y, 3), round(z, 3)]}
                )
                per_wp_stalls[i_wp] = per_wp_stalls.get(i_wp, 0) + 1
                if per_wp_stalls[i_wp] >= 3:
                    # wedged (reverse recovery failed twice): teleport past the
                    # blockage so ONE wedge cannot blind the rest of the path.
                    # The wedge is still fully recorded (stalls + teleports);
                    # distance_before_first_wedge_m keeps the honest
                    # "how far before failure" number.
                    j = min(i_wp + 2, n_wp - 1)
                    skipped_wps.extend(range(i_wp, j + 1))
                    teleports.append(
                        {"sim_t": round(sim_t, 2), "from_wp": i_wp, "to_wp": j,
                         "pos": [round(x, 3), round(y, 3), round(z, 3)]}
                    )
                    if first_wedge_wp is None:
                        first_wedge_wp = i_wp
                    yaw_j = math.atan2(
                        wps[min(j + 1, n_wp - 1)][1] - wps[j][1],
                        wps[min(j + 1, n_wp - 1)][0] - wps[j][0],
                    )
                    robot.set_world_pose(
                        position=np.array(
                            [wps[j][0], wps[j][1], wps[j][2] + 0.10]
                        ),
                        orientation=np.array(
                            [math.cos(yaw_j / 2), 0.0, 0.0, math.sin(yaw_j / 2)]
                        ),
                    )
                    try:
                        robot._articulation_view.set_velocities(np.zeros((1, 6)))
                    except Exception:  # noqa: BLE001
                        pass
                    for _ in range(60):
                        robot.apply_wheel_actions(controller.forward([0.0, 0.0]))
                        world.step(render=False)
                        sim_t += dt
                    i_wp = min(j + 1, n_wp)
                    max_reached = max(max_reached, i_wp - 1)
                    last_advance_t = sim_t
                    state = "NORMAL"
                else:
                    state = "REVERSING"
                    state_until = sim_t + 2.0
                    reverse_turn = -reverse_turn
                pos_hist.clear()
                cmd_hist.clear()
                if len(stalls) >= 60 or len(teleports) >= 15:
                    outcome = "stalled_out"
                    break
                if i_wp >= n_wp:
                    outcome = "completed"
                    break

            if sim_t - last_partial > 10.0:
                last_partial = sim_t
                report["status"] = "driving"
                report["progress_waypoint"] = int(max_reached)
                report["progress_m"] = round(
                    float(arclen[min(max_reached, n_wp - 1)] - arclen[start_wp]), 2)
                report["sim_t"] = round(sim_t, 1)
                write_report()

        if outcome is None:
            outcome = "timeout"
        if outcome == "completed" and teleports:
            outcome = "completed_with_recoveries"

        z_devs = np.array(z_devs) if z_devs else np.zeros(1)
        st = np.array(step_times[10:]) * 1e3 if len(step_times) > 20 else np.array([0.0])
        completed_m = float(arclen[min(max_reached, n_wp - 1)] - arclen[start_wp])
        total_m = float(arclen[-1] - arclen[start_wp])
        report.update(
            {
                "ok": True,
                "status": "finished",
                "outcome": outcome,
                "sim_time_s": round(sim_t, 1),
                "wall_drive_s": round(float(np.sum(step_times)), 1),
                "step_ms_mean": round(float(st.mean()), 3),
                "step_ms_p95": round(float(np.percentile(st, 95)), 3),
                "n_fall_throughs": len(fall_events),
                "fall_events": fall_events,
                "n_stalls": len(stalls),
                "stalls": stalls,
                "n_wedge_teleports": len(teleports),
                "teleports": teleports,
                "first_wedge_wp": first_wedge_wp,
                "distance_before_first_wedge_m": round(
                    float(arclen[first_wedge_wp] - arclen[start_wp]), 2
                ) if first_wedge_wp is not None else None,
                "skipped_waypoints": skipped_wps,
                "n_waypoints_reached": int(max_reached),
                "distance_completed_m": round(completed_m, 2),
                "total_path_m": round(total_m, 2),
                "completion_frac": round(completed_m / total_m, 4),
                "max_abs_roll_deg": round(math.degrees(max_abs_roll), 1),
                "max_abs_pitch_deg": round(math.degrees(max_abs_pitch), 1),
                "z_dev_mean_m": round(float(np.mean(z_devs)), 4),
                "z_dev_p95_abs_m": round(float(np.percentile(np.abs(z_devs), 95)), 4),
                "z_dev_max_abs_m": round(float(np.max(np.abs(z_devs))), 4),
            }
        )
        write_report()
    except Exception as e:  # noqa: BLE001
        import traceback

        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        report["status"] = "error"
        write_report()
    finally:
        write_report()
        simulation_app.close()


if __name__ == "__main__":
    main()
