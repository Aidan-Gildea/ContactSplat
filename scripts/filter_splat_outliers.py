# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Drop far-field outlier Gaussians from a trained PLY.

MCMC densification injects noise into Gaussian positions, and a small number of
low-opacity Gaussians drift arbitrarily far from the scene. They are visually
negligible -- on a 2.5 minute Aria walk, the ~200 Gaussians beyond 5 km had a
median opacity of 0.011 -- but they set the asset's bounding box, and a 120 km
AABB around a 10 m scene ruins framing and depth precision once the splat is
imported into a USD stage.

Filtering by radius about the median Gaussian position, rather than about the
origin, keeps this correct for scenes whose world frame is not centred on the
capture (which is the normal case for Aria: the MPS world origin sits wherever
tracking initialised).

`--min-opacity` additionally drops Gaussians whose activated (sigmoid) opacity
falls below a threshold. Low-opacity floaters are visually negligible but they
still write into depth renders, punching spurious holes into a TSDF fusion --
so a depth-render copy of the asset should be filtered aggressively while the
visual copy is left untouched (0.0 disables the filter, the default).

Usage:
    python scripts/filter_splat_outliers.py in.ply --radius 50 --output out.ply
    python scripts/filter_splat_outliers.py in.ply --radius 50 --min-opacity 0.3 \
        --output depth_render_copy.ply
"""

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def filter_ply(
    input_path: Path, output_path: Path, radius: float, min_opacity: float = 0.0
) -> None:
    plydata = PlyData.read(str(input_path))
    vertex = plydata["vertex"]

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    centre = np.median(xyz, axis=0)
    distance = np.linalg.norm(xyz - centre, axis=1)
    keep = distance <= radius

    total = keep.size
    dropped_radius = total - int(keep.sum())
    print(f"centre (median position): {np.round(centre, 3).tolist()}")
    print(f"kept {int(keep.sum())}/{total} Gaussians, dropped {dropped_radius} "
          f"({100.0 * dropped_radius / total:.3f}%) beyond {radius} m")

    if min_opacity > 0.0:
        # PLY stores the pre-activation logit; compare in activated space.
        sigmoid_opacity_all = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"])))
        opacity_keep = sigmoid_opacity_all >= min_opacity
        dropped_opacity = int((keep & ~opacity_keep).sum())
        keep &= opacity_keep
        print(f"dropped {dropped_opacity} more ({100.0 * dropped_opacity / total:.3f}%) "
              f"with activated opacity < {min_opacity}; kept {int(keep.sum())}/{total}")

    dropped = total - int(keep.sum())

    if dropped:
        opacity = np.asarray(vertex["opacity"])[~keep]
        sigmoid_opacity = 1.0 / (1.0 + np.exp(-opacity))
        print(f"dropped Gaussians' median opacity: {np.median(sigmoid_opacity):.4f}")

    before = xyz.min(axis=0), xyz.max(axis=0)
    after = xyz[keep].min(axis=0), xyz[keep].max(axis=0)
    print(f"AABB before: {np.round(before[0], 1).tolist()} .. {np.round(before[1], 1).tolist()}")
    print(f"AABB after : {np.round(after[0], 1).tolist()} .. {np.round(after[1], 1).tolist()}")

    # Preserve every property and every extra element (the `metadata` element
    # carries colour format and SH degree, which the loader reads back).
    filtered = PlyElement.describe(vertex.data[keep], "vertex")
    others = [el for el in plydata.elements if el.name != "vertex"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([filtered, *others], text=False).write(str(output_path))
    print(f"wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="input PLY")
    parser.add_argument("--output", type=Path, default=None, help="output PLY")
    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="keep Gaussians within this distance (m) of the median position",
    )
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=0.0,
        help="drop Gaussians with activated (sigmoid) opacity below this "
        "threshold (0.0 disables; use e.g. 0.3-0.5 for a depth-render copy)",
    )
    args = parser.parse_args()

    output = args.output or args.input_file.with_name(
        args.input_file.stem + "_filtered.ply"
    )
    filter_ply(args.input_file, output, args.radius, args.min_opacity)


if __name__ == "__main__":
    main()
