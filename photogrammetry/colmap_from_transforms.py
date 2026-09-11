#!/usr/bin/env python3
"""
Build a COLMAP sparse model *skeleton* (cameras.txt / images.txt / points3D.txt) from the
posed, rectified dataset written by scripts/extract_aria_vrs.py.

The poses come straight from MPS (bundle-adjusted, gravity-aligned, metric, Z-up) and are
written as fixed COLMAP image poses. COLMAP's `mapper` is never run downstream, so every
product of this pipeline stays in the MPS world frame -- the same frame as the trained
Gaussian splat.

Frames are subsampled to keyframes by pose delta (translation OR rotation), optionally
after restricting to a time window [start, start + duration] measured in seconds from the
first posed frame.

Outputs (in --out_dir):
    sparse_in/cameras.txt   one PINHOLE camera shared by every keyframe
    sparse_in/images.txt    one line per keyframe: quaternion + translation of world->camera
    sparse_in/points3D.txt  empty (point_triangulator fills the points in)
    keyframes.txt           image file names, relative to <rectified_dir>/images
    keyframes.json          selection metadata (thresholds, window, timestamps)
"""

import argparse
import json
from pathlib import Path

import numpy as np


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (qw, qx, qy, qz), COLMAP ordering.

    Shepperd's method: pick the largest diagonal term so the division is stable.
    """
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R = R_a.T @ R_b
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def select_keyframes(c2w: np.ndarray, min_translation: float, min_rotation_deg: float):
    """Greedy pose-delta subsampling. Keeps frame i if it moved >= min_translation metres
    OR rotated >= min_rotation_deg degrees since the last kept frame."""
    keep = [0]
    last = c2w[0]
    for i in range(1, len(c2w)):
        T = c2w[i]
        dt = np.linalg.norm(T[:3, 3] - last[:3, 3])
        da = rotation_angle_deg(last[:3, :3], T[:3, :3])
        if dt >= min_translation or da >= min_rotation_deg:
            keep.append(i)
            last = T
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rectified_dir", required=True, type=Path,
                    help="camera-rgb-rectified-<focal>-h<height> folder from extract_aria_vrs.py")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--start", type=float, default=0.0,
                    help="window start, seconds after the first posed frame (default 0)")
    ap.add_argument("--duration", type=float, default=-1.0,
                    help="window length in seconds; <0 means to the end of the recording")
    ap.add_argument("--min_translation", type=float, default=0.10,
                    help="keyframe if the camera moved at least this far (metres)")
    ap.add_argument("--min_rotation_deg", type=float, default=10.0,
                    help="keyframe if the camera rotated at least this much (degrees)")
    ap.add_argument("--max_keyframes", type=int, default=-1,
                    help="if >0, thin the keyframe list uniformly down to this many")
    args = ap.parse_args()

    transforms_path = args.rectified_dir / "transforms_with_sparse_depth.json"
    if not transforms_path.exists():
        transforms_path = args.rectified_dir / "transforms.json"
    if not transforms_path.exists():
        raise SystemExit(f"no transforms.json in {args.rectified_dir}")

    with open(transforms_path) as f:
        transforms = json.load(f)
    if transforms.get("camera_model") != "linear":
        raise SystemExit(
            f"camera_model is {transforms.get('camera_model')!r}, expected 'linear' "
            "(rectified pinhole). Fisheye output cannot be fed to COLMAP as PINHOLE."
        )
    frames = transforms["frames"]
    if not frames:
        raise SystemExit("transforms.json holds no frames")

    # --- intrinsics: must be one shared pinhole camera --------------------------------
    intr = {(fr["fx"], fr["fy"], fr["cx"], fr["cy"], fr["w"], fr["h"]) for fr in frames}
    if len(intr) != 1:
        raise SystemExit(f"expected one set of intrinsics across frames, found {len(intr)}: {intr}")
    fx, fy, cx, cy, w, h = intr.pop()
    # projectaria_tools puts the centre of the top-left pixel at (0, 0); COLMAP puts it at
    # (0.5, 0.5). Shift the principal point by half a pixel so SIFT keypoints (COLMAP
    # convention) and the fixed MPS poses agree.
    cx_colmap, cy_colmap = cx + 0.5, cy + 0.5

    # --- time window --------------------------------------------------------------------
    frames.sort(key=lambda fr: fr["timestamp"])
    ts = np.array([fr["timestamp"] for fr in frames], dtype=np.float64)  # ns
    t0 = ts[0]
    rel_s = (ts - t0) / 1e9
    win_start = args.start
    win_end = np.inf if args.duration < 0 else args.start + args.duration
    in_window = np.where((rel_s >= win_start) & (rel_s <= win_end))[0]
    if len(in_window) == 0:
        raise SystemExit(f"no frames in window [{win_start}, {win_end}] s "
                         f"(recording spans 0 .. {rel_s[-1]:.1f} s)")

    c2w_all = np.array([frames[i]["transform_matrix"] for i in in_window], dtype=np.float64)
    keep_local = select_keyframes(c2w_all, args.min_translation, args.min_rotation_deg)
    if args.max_keyframes > 0 and len(keep_local) > args.max_keyframes:
        idx = np.linspace(0, len(keep_local) - 1, args.max_keyframes).round().astype(int)
        keep_local = [keep_local[i] for i in idx]
    keep = [int(in_window[i]) for i in keep_local]

    # --- write COLMAP text model --------------------------------------------------------
    sparse_in = args.out_dir / "sparse_in"
    sparse_in.mkdir(parents=True, exist_ok=True)

    with open(sparse_in / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {int(w)} {int(h)} {fx} {fy} {cx_colmap} {cy_colmap}\n")

    names = []
    with open(sparse_in / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, i in enumerate(keep, start=1):
            fr = frames[i]
            c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)
            w2c = np.linalg.inv(c2w)
            q = rotmat_to_qvec(w2c[:3, :3])
            t = w2c[:3, 3]
            name = Path(fr["image_path"]).name  # relative to <rectified_dir>/images
            names.append(name)
            f.write(f"{image_id} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} 1 {name}\n")
            f.write("\n")  # no 2D points yet

    with open(sparse_in / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

    with open(args.out_dir / "keyframes.txt", "w") as f:
        f.write("\n".join(names) + "\n")

    meta = {
        "rectified_dir": str(args.rectified_dir.resolve()),
        "transforms": str(transforms_path.resolve()),
        "images_dir": str((args.rectified_dir / "images").resolve()),
        "camera": {"model": "PINHOLE", "width": int(w), "height": int(h),
                   "fx": fx, "fy": fy, "cx": cx_colmap, "cy": cy_colmap,
                   "aria_cx": cx, "aria_cy": cy},
        "window": {"start_s": win_start,
                   "end_s": None if np.isinf(win_end) else win_end,
                   "recording_span_s": float(rel_s[-1]),
                   "first_timestamp_ns": int(t0)},
        "keyframe_selection": {"min_translation_m": args.min_translation,
                               "min_rotation_deg": args.min_rotation_deg,
                               "max_keyframes": args.max_keyframes},
        "num_frames_total": len(frames),
        "num_frames_in_window": int(len(in_window)),
        "num_keyframes": len(keep),
        "keyframe_timestamps_ns": [int(frames[i]["timestamp"]) for i in keep],
    }
    with open(args.out_dir / "keyframes.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"frames: {len(frames)} total, {len(in_window)} in window "
          f"[{win_start:.1f}, {'end' if np.isinf(win_end) else f'{win_end:.1f}'}] s, "
          f"{len(keep)} keyframes "
          f"(>= {args.min_translation} m or >= {args.min_rotation_deg} deg apart)")
    print(f"camera: PINHOLE {int(w)}x{int(h)} fx={fx} fy={fy} cx={cx_colmap} cy={cy_colmap}")
    print(f"wrote {sparse_in}")


if __name__ == "__main__":
    main()
