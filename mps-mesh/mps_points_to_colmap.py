#!/usr/bin/env python3
"""
Write the MPS semi-dense point cloud as a COLMAP dense workspace, so that
`colmap delaunay_mesher --input_type dense` meshes it exactly as it would mesh the
`stereo_fusion` output of patch-match MVS. Only the source of the points changes.

    <out_dir>/dense/fused.ply        confident semi-dense points (MPS world frame, metres)
    <out_dir>/dense/fused.ply.vis    for every point, the SLAM-camera views that observed it
    <out_dir>/dense/sparse/          cameras.txt / images.txt for those views, points3D.txt empty

Points are filtered with the same confidence thresholds as scripts/extract_aria_vrs.py and
photogrammetry/evaluate_mesh.py. A view is one (timestamp, SLAM camera) pair from
semidense_observations.csv.gz. Its pose is the closed-loop device pose at that timestamp
(nearest 1 kHz sample) composed with that camera's T_Device_Camera from the first record
of online_calibration.jsonl.

The mesher needs a camera model only to decide which nearby points to merge, so each SLAM
camera is written as a SIMPLE_PINHOLE with the fisheye's focal length and principal point.

The MPS files are found next to the semidense_points.csv.gz symlink that
extract_aria_vrs.py places in the rectified folder.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

INV_DIST_STD_THR = 0.005      # same semi-dense confidence filter as extract_aria_vrs.py
DIST_STD_THR = 0.01
SLAM_IMAGE_SIZE = 512         # Gen 2 SLAM cameras are 512 x 512


def quat_to_rotmat(qw, qx, qy, qz) -> np.ndarray:
    """(N,) quaternion components -> (N, 3, 3) rotation matrices."""
    qw, qx, qy, qz = (np.asarray(a, dtype=np.float64) for a in (qw, qx, qy, qz))
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.stack([
        np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], -1),
        np.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], -1),
        np.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)], -1),
    ], -2)


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 4) unit quaternions (qw, qx, qy, qz), COLMAP ordering, qw >= 0."""
    tr = np.trace(R, axis1=1, axis2=2)
    q = np.empty((len(R), 4))
    q[:, 0] = np.sqrt(np.maximum(0.0, 1 + tr)) / 2
    q[:, 1] = np.sqrt(np.maximum(0.0, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2])) / 2
    q[:, 2] = np.sqrt(np.maximum(0.0, 1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2])) / 2
    q[:, 3] = np.sqrt(np.maximum(0.0, 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2])) / 2
    q[:, 1] = np.copysign(q[:, 1], R[:, 2, 1] - R[:, 1, 2])
    q[:, 2] = np.copysign(q[:, 2], R[:, 0, 2] - R[:, 2, 0])
    q[:, 3] = np.copysign(q[:, 3], R[:, 1, 0] - R[:, 0, 1])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def load_camera_calibrations(path: Path) -> dict:
    """serial -> (label, T_device_camera 4x4, focal, cx, cy) from the first online record."""
    with open(path) as f:
        record = json.loads(f.readline())
    cams = {}
    for c in record["CameraCalibrations"]:
        if not c.get("SerialNumber") or not c["Label"].startswith("slam"):
            continue
        qw, (qx, qy, qz) = c["T_Device_Camera"]["UnitQuaternion"]
        T = np.eye(4)
        T[:3, :3] = quat_to_rotmat([qw], [qx], [qy], [qz])[0]
        T[:3, 3] = c["T_Device_Camera"]["Translation"]
        f_, cx, cy = c["Projection"]["Params"][:3]
        cams[c["SerialNumber"]] = (c["Label"], T, f_, cx, cy)
    return cams


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rectified_dir", required=True, type=Path,
                    help="camera-rgb-rectified-<focal>-h<height> folder from extract_aria_vrs.py")
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    slam_dir = (args.rectified_dir / "semidense_points.csv.gz").resolve().parent
    points_csv = slam_dir / "semidense_points.csv.gz"
    obs_csv = slam_dir / "semidense_observations.csv.gz"
    traj_csv = slam_dir / "closed_loop_trajectory.csv"
    calib_jsonl = slam_dir / "online_calibration.jsonl"
    for p in (points_csv, obs_csv, traj_csv, calib_jsonl):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    # --- points ------------------------------------------------------------------------
    pts = pd.read_csv(points_csv, usecols=["uid", "px_world", "py_world", "pz_world",
                                           "inv_dist_std", "dist_std"])
    n_all = len(pts)
    pts = pts[(pts["inv_dist_std"] < INV_DIST_STD_THR) & (pts["dist_std"] < DIST_STD_THR)]

    # --- observations -> views -----------------------------------------------------------
    obs = pd.read_csv(obs_csv, usecols=["uid", "frame_tracking_timestamp_us", "camera_serial"])
    n_obs_all = len(obs)
    obs = obs[obs["uid"].isin(pts["uid"])].drop_duplicates()
    cams = load_camera_calibrations(calib_jsonl)
    unknown = set(obs["camera_serial"].unique()) - set(cams)
    if unknown:
        raise SystemExit(f"observations reference cameras missing from online_calibration: {unknown}")

    views = (obs[["frame_tracking_timestamp_us", "camera_serial"]].drop_duplicates()
             .sort_values(["frame_tracking_timestamp_us", "camera_serial"]).reset_index(drop=True))
    views["view_idx"] = np.arange(len(views), dtype=np.uint32)
    obs = obs.merge(views, on=["frame_tracking_timestamp_us", "camera_serial"])

    # keep only points that some view saw, in a fixed order
    pts = pts[pts["uid"].isin(obs["uid"])].reset_index(drop=True)
    pts["point_idx"] = np.arange(len(pts), dtype=np.int64)
    obs = obs.merge(pts[["uid", "point_idx"]], on="uid").sort_values(["point_idx", "view_idx"])

    # --- view poses ----------------------------------------------------------------------
    traj = pd.read_csv(traj_csv, usecols=["tracking_timestamp_us", "tx_world_device", "ty_world_device",
                                          "tz_world_device", "qx_world_device", "qy_world_device",
                                          "qz_world_device", "qw_world_device"])
    traj = traj.sort_values("tracking_timestamp_us").reset_index(drop=True)
    t_traj = traj["tracking_timestamp_us"].to_numpy()
    t_view = views["frame_tracking_timestamp_us"].to_numpy()
    j = np.clip(np.searchsorted(t_traj, t_view), 1, len(t_traj) - 1)
    j = np.where(np.abs(t_traj[j - 1] - t_view) <= np.abs(t_traj[j] - t_view), j - 1, j)
    max_dt_us = int(np.abs(t_traj[j] - t_view).max())
    if max_dt_us > 5000:
        raise SystemExit(f"a view is {max_dt_us} us from the nearest trajectory sample")
    tr = traj.iloc[j]
    T_world_device = np.tile(np.eye(4), (len(views), 1, 1))
    T_world_device[:, :3, :3] = quat_to_rotmat(tr["qw_world_device"], tr["qx_world_device"],
                                               tr["qy_world_device"], tr["qz_world_device"])
    T_world_device[:, :3, 3] = tr[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()
    T_device_cam = np.stack([cams[s][1] for s in views["camera_serial"]])
    T_cam_world = np.linalg.inv(T_world_device @ T_device_cam)
    qvec = rotmat_to_qvec(T_cam_world[:, :3, :3])
    tvec = T_cam_world[:, :3, 3]

    # --- write the dense workspace -------------------------------------------------------
    dense = args.out_dir / "dense"
    sparse = dense / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    serials = sorted(views["camera_serial"].unique())
    cam_id = {s: i + 1 for i, s in enumerate(serials)}
    with open(sparse / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for s in serials:
            _, _, f_, cx, cy = cams[s]
            # fisheye pixel centres are at integer coordinates; COLMAP's at +0.5
            f.write(f"{cam_id[s]} SIMPLE_PINHOLE {SLAM_IMAGE_SIZE} {SLAM_IMAGE_SIZE} "
                    f"{f_} {cx + 0.5} {cy + 0.5}\n")

    # No rigs.txt / frames.txt: COLMAP then registers images in file order, so line k of
    # images.txt is index k in fused.ply.vis (DelaunayMeshingInput::ReadDenseReconstruction).
    with open(sparse / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for k, (t_us, s) in enumerate(zip(views["frame_tracking_timestamp_us"], views["camera_serial"])):
            q, t = qvec[k], tvec[k]
            f.write(f"{k + 1} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {cam_id[s]} {cams[s][0]}_{t_us}\n\n")
    (sparse / "points3D.txt").write_text("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

    xyz = pts[["px_world", "py_world", "pz_world"]].to_numpy(dtype=np.float32)
    vertex = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    vertex["red"] = vertex["green"] = vertex["blue"] = 128
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(dense / "fused.ply", "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())

    # fused.ply.vis: uint64 N, then per point uint32 count followed by count uint32 view indices
    point_idx = obs["point_idx"].to_numpy()
    view_idx = obs["view_idx"].to_numpy(dtype=np.uint32)
    counts = np.bincount(point_idx, minlength=len(pts))
    starts = np.arange(len(pts)) + np.concatenate([[0], np.cumsum(counts)[:-1]])
    vis = np.empty(len(pts) + len(obs), dtype="<u4")
    vis[starts] = counts
    rank = np.arange(len(obs)) - np.repeat(np.cumsum(counts) - counts, counts)
    vis[np.repeat(starts, counts) + 1 + rank] = view_idx
    with open(dense / "fused.ply.vis", "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        f.write(vis.tobytes())

    summary = {
        "mps_slam_dir": str(slam_dir),
        "confidence_filter": {"inv_dist_std_lt": INV_DIST_STD_THR, "dist_std_lt": DIST_STD_THR},
        "points_total": n_all,
        "points_written": int(len(pts)),
        "observations_total": n_obs_all,
        "observations_written": int(len(obs)),
        "observations_per_point": {q: float(np.percentile(counts, p))
                                   for q, p in (("p5", 5), ("median", 50), ("p95", 95))},
        "views": int(len(views)),
        "views_per_camera": {cams[s][0]: int((views["camera_serial"] == s).sum()) for s in serials},
        "max_view_to_trajectory_dt_us": max_dt_us,
    }
    with open(args.out_dir / "mps_points.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {dense}")


if __name__ == "__main__":
    main()
