# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fuse Aria Gen 2 stereo depth maps into a TSDF mesh (task E3).

Consumes the output of Meta's ``projectaria_gen2_depth_from_stereo`` export
tool (https://github.com/facebookresearch/projectaria_gen2_depth_from_stereo):

    <export_dir>/
        depth/depth_XXXXXXXX.png          uint16 millimetres, 0 = invalid
        masks/mask_XXXXXXXX.png           LR-consistency, 255 = consistent
        rectified_images/image_XXXXXXXX.png   uint8 grayscale (used as color)
        pinhole_camera_parameters.json    per-frame intrinsics + T_world_camera

``T_world_camera`` in that JSON is derived from the MPS closed-loop
trajectory, so the fused mesh lives in the MPS world frame (gravity-aligned,
Z-up, metric metres) -- the same frame as the trained Gaussian splats and the
E2 meshes. No registration step is needed. The rectified camera convention is
X-right / Y-down / Z-forward, which is exactly Open3D's pinhole convention.

The fusion stage itself (``ScalableTSDFVolume``, per-frame ``integrate``,
marching cubes, pose-delta frame subsampling) is shared with
``scripts/extract_mesh_tsdf.py`` via ``scripts/tsdf_fusion.py`` so the two
routes are directly comparable at identical voxel/truncation settings.

Masking before integration:
- LR-consistency mask == 0 (occlusions, textureless mismatch)  [--no-mask off]
- depth == 0 (invalid) or outside [--depth-min, --depth-trunc]

Max fusion depth (--depth-trunc): stereo depth error grows quadratically,
sigma_Z = Z^2 * sigma_d / (f * B). This recording: f = 305.87 px rectified,
B = 0.13496 m -> f*B = 41.28 m*px. At sigma_d = 0.5 px, sigma_Z crosses the
2 cm TSDF voxel at Z = 1.28 m and the 8 cm truncation band at Z = 2.57 m; the
E3 sanity check against MPS semi-dense points measured median |dz| = 3.3 cm
below 1.5 m, 5.2 cm at 1.5-2.5 m, but ~31 cm at 2.5-4 m. Beyond ~4 m stereo
depth mostly smears the volume. Default 4.0 m keeps the well-measured band
plus a margin the multi-view averaging can still use; run with 2.5 for a
strict variant.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import open3d as o3d  # noqa: E402

import tsdf_fusion  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--export-dir", type=Path, required=True,
                    help="output dir of export_depth_from_stereo.py")
    ap.add_argument("--output", type=Path, required=True,
                    help="output mesh path (.ply)")
    # TSDF parameters -- defaults identical to extract_mesh_tsdf.py (E2)
    ap.add_argument("--voxel-length", type=float, default=0.02,
                    help="TSDF voxel size [m] (default 0.02, matches E2)")
    ap.add_argument("--sdf-trunc", type=float, default=0.08,
                    help="TSDF truncation distance [m] (default 0.08, matches E2)")
    ap.add_argument("--depth-min", type=float, default=0.3,
                    help="discard depth below this [m] (default 0.3)")
    ap.add_argument("--depth-trunc", type=float, default=4.0,
                    help="max fusion depth [m] (default 4.0; see module "
                         "docstring for the stereo error model rationale)")
    ap.add_argument("--no-mask", action="store_true",
                    help="debug: ignore the LR-consistency masks")
    # frame subsampling -- defaults identical to extract_mesh_tsdf.py (E2)
    ap.add_argument("--min-trans", type=float, default=0.10,
                    help="keep a frame after this much translation [m]")
    ap.add_argument("--min-rot-deg", type=float, default=10.0,
                    help="or after this much rotation [deg]")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="debug: stop after this many fused frames (0 = all)")
    return ap.parse_args()


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def frame_transforms(fr):
    """(K 3x3, c2w 4x4, w2c 4x4) from one JSON frame entry."""
    fx, fy, cx, cy = fr["camera"]["Parameters"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    R = quat_xyzw_to_R(fr["T_world_camera"]["QuaternionXYZW"])
    t = np.asarray(fr["T_world_camera"]["Translation"], dtype=np.float64)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return K, c2w, w2c


def main() -> int:
    args = parse_args()
    t0 = time.time()

    json_path = args.export_dir / "pinhole_camera_parameters.json"
    frames = json.loads(json_path.read_text())
    frames.sort(key=lambda fr: fr["frameTimestampNs"])
    print(f"[stereo_tsdf] {len(frames)} exported frames in {json_path}")

    transforms = [frame_transforms(fr) for fr in frames]
    kept_idx = tsdf_fusion.subsample_indices_by_pose_delta(
        [c2w for _, c2w, _ in transforms], args.min_trans, args.min_rot_deg
    )
    if args.max_frames > 0:
        kept_idx = kept_idx[: args.max_frames]
    print(f"[stereo_tsdf] pose-delta subsampling "
          f"(>= {args.min_trans} m or >= {args.min_rot_deg} deg): "
          f"kept {len(kept_idx)}/{len(frames)} frames")

    volume = tsdf_fusion.make_tsdf_volume(
        voxel_length=args.voxel_length, sdf_trunc=args.sdf_trunc
    )

    t1 = time.time()
    valid_px_total = 0
    total_px = 0
    masked_by_lr_total = 0
    for n_done, i in enumerate(kept_idx, start=1):
        fr = frames[i]
        idx = fr["index"]
        K, _, w2c = transforms[i]

        depth = np.asarray(
            Image.open(args.export_dir / "depth" / f"depth_{idx:08d}.png"),
            dtype=np.uint16,
        ).astype(np.float32) / 1000.0
        gray = np.asarray(
            Image.open(args.export_dir / "rectified_images" / f"image_{idx:08d}.png"),
            dtype=np.uint8,
        )
        color_u8 = np.repeat(gray[..., None], 3, axis=2)

        invalid = (depth <= 0) | (depth < args.depth_min) | (depth > args.depth_trunc)
        if not args.no_mask:
            mask_path = args.export_dir / "masks" / f"mask_{idx:08d}.png"
            lr_mask = np.asarray(Image.open(mask_path)) == 255
            masked_by_lr_total += int((~lr_mask & ~invalid).sum())
            invalid |= ~lr_mask
        depth[invalid] = 0.0

        valid_px_total += int((~invalid).sum())
        total_px += depth.size
        tsdf_fusion.integrate_frame(volume, color_u8, depth, K, w2c,
                                    args.depth_trunc)

        if n_done % 100 == 0 or n_done == len(kept_idx):
            rate = n_done / (time.time() - t1)
            print(f"  fused {n_done}/{len(kept_idx)} frames ({rate:.2f} fps)")

    t_fuse = time.time() - t1
    print(f"[stereo_tsdf] fusion took {t_fuse:.1f}s "
          f"({len(kept_idx) / max(t_fuse, 1e-9):.2f} fps), "
          f"mean valid px {valid_px_total / max(total_px, 1):.1%}")

    t2 = time.time()
    mesh = tsdf_fusion.extract_mesh(volume)
    t_mesh = time.time() - t2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.output), mesh)
    bbox = mesh.get_axis_aligned_bounding_box()
    print(f"[stereo_tsdf] mesh: {len(mesh.vertices)} vertices, "
          f"{len(mesh.triangles)} triangles (marching cubes {t_mesh:.1f}s)")
    print(f"[stereo_tsdf] AABB {np.round(bbox.min_bound, 2).tolist()} .. "
          f"{np.round(bbox.max_bound, 2).tolist()}")
    print(f"[stereo_tsdf] wrote {args.output}")

    meta = {
        "export_dir": str(args.export_dir),
        "voxel_length": args.voxel_length,
        "sdf_trunc": args.sdf_trunc,
        "depth_min": args.depth_min,
        "depth_trunc": args.depth_trunc,
        "lr_mask_used": not args.no_mask,
        "min_trans": args.min_trans,
        "min_rot_deg": args.min_rot_deg,
        "frames_total": len(frames),
        "frames_fused": len(kept_idx),
        "valid_px_mean_fraction": round(valid_px_total / max(total_px, 1), 4),
        "lr_masked_px_fraction": round(masked_by_lr_total / max(total_px, 1), 4),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "aabb_min": np.asarray(bbox.min_bound).round(4).tolist(),
        "aabb_max": np.asarray(bbox.max_bound).round(4).tolist(),
        "time_fuse_s": round(t_fuse, 1),
        "time_mesh_s": round(t_mesh, 1),
        "time_total_s": round(time.time() - t0, 1),
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[stereo_tsdf] metadata -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
