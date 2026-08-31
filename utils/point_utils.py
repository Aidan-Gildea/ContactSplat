# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from plyfile import PlyData, PlyElement
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array


def fetchPly(path: str, stride: int = 1):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T

    if stride > 1:
        print(
            f"Subsampling point cloud of length {positions.shape[0]} with stride {stride}..."
        )
        positions = positions[::stride]
        colors = colors[::stride]
        normals = normals[::stride]
        print(f"New point cloud size: {positions.shape[0]}")

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path: str, xyz: np.ndarray, rgb: np.ndarray, normals: np.ndarray=None):
    """
    """

    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def project(point3d: np.ndarray, T_w2c: np.ndarray, calibK: np.ndarray, frame_h: int, frame_w: int):
    """Project world points into a pinhole camera, keeping only in-frustum hits.

    `point3d` is (3, N) for a batch, or (3,) for a single point. A batch returns
    (u, v, z) already filtered plus the boolean mask used to filter them, so
    callers can select their own per-point attributes with the same mask. A
    single point returns scalars, or four Nones if it is not visible.

    Two things this used to get wrong:

    * The visibility test was branched on `point3d.shape[-1] > 1`, so a batch
      that happened to contain exactly one point (or none) fell into the scalar
      branch and returned mask=None -- which every batched caller then used as
      an index. Rank is what distinguishes the two cases, not batch size.
    * The scalar test read `u < 0 or u >= w-1 or v < 0 or v >= h-1 and z > 0`.
      `and` binds tighter than `or`, so `z > 0` applied only to the last clause
      and points BEHIND the camera were never rejected. Such a point projects
      to a sign-flipped u,v that can land inside the image, so it was accepted
      and written into sparse depth with a negative depth.
    """
    single = point3d.ndim == 1
    pts = point3d.reshape(3, 1) if single else point3d

    rot = T_w2c[:3, :3]
    t = T_w2c[:3, 3:]
    point3d_cam = rot @ pts + t
    point3d_proj = calibK @ point3d_cam

    with np.errstate(divide="ignore", invalid="ignore"):
        u_proj = point3d_proj[0] / point3d_proj[2]
        v_proj = point3d_proj[1] / point3d_proj[2]
    z = point3d_proj[2]

    mask = (
        (u_proj > 0) & (u_proj < frame_w)
        & (v_proj > 0) & (v_proj < frame_h)
        & (z > 0)
    )

    if single:
        if not mask[0]:
            return None, None, None, None
        return u_proj[0], v_proj[0], z[0], None

    return u_proj[mask], v_proj[mask], z[mask], mask
