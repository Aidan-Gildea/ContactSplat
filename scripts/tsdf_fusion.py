# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared TSDF fusion stage (Open3D ScalableTSDFVolume).

Factored out of ``scripts/extract_mesh_tsdf.py`` (task E2) so that other
depth sources -- e.g. the Aria Gen 2 stereo depth maps fused by
``scripts/extract_mesh_stereo_tsdf.py`` (task E3) -- reuse the exact same
fusion/meshing code instead of duplicating it.

All poses are expected in the MPS world frame (gravity-aligned, Z-up,
metric metres); the camera convention on both sides is X-right / Y-down /
Z-forward, which is what Open3D's pinhole back-projection assumes.
"""

import numpy as np
import open3d as o3d


def make_tsdf_volume(voxel_length: float = 0.02, sdf_trunc: float = 0.08):
    """Create the ScalableTSDFVolume used by every mesh-extraction task."""
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def integrate_frame(volume, color_u8, depth_f32, K, w2c_44, depth_trunc):
    """Integrate one masked depth frame into the TSDF volume.

    Args:
        volume: ScalableTSDFVolume from :func:`make_tsdf_volume`.
        color_u8: (H, W, 3) uint8 color image.
        depth_f32: (H, W) float32 depth in metres; 0 = invalid pixel.
        K: (3, 3) pinhole intrinsics.
        w2c_44: (4, 4) world-to-camera extrinsic.
        depth_trunc: max depth [m] Open3D integrates.
    """
    h, w = depth_f32.shape
    intr = o3d.camera.PinholeCameraIntrinsic(
        w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color_u8)),
        o3d.geometry.Image(np.ascontiguousarray(depth_f32)),
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    volume.integrate(rgbd, intr, np.asarray(w2c_44, dtype=np.float64))


def extract_mesh(volume):
    """Marching-cubes mesh with vertex normals from the fused volume."""
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def subsample_indices_by_pose_delta(c2ws, min_trans: float, min_rot_deg: float):
    """Indices of frames kept after >= min_trans [m] or >= min_rot_deg [deg]
    of motion since the last kept frame. ``c2ws`` is a sequence of (4, 4)
    camera-to-world matrices in temporal order."""
    kept, last_c2w = [], None
    cos_tol = np.cos(np.deg2rad(min_rot_deg))
    for i, c2w in enumerate(c2ws):
        if last_c2w is None:
            kept.append(i)
            last_c2w = c2w
            continue
        dt = np.linalg.norm(c2w[:3, 3] - last_c2w[:3, 3])
        cos_angle = (np.trace(last_c2w[:3, :3].T @ c2w[:3, :3]) - 1.0) / 2.0
        if dt >= min_trans or cos_angle <= cos_tol:
            kept.append(i)
            last_c2w = c2w
    return kept
