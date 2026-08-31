#!/usr/bin/env python
"""Strip the baked frame-conversion rotation from a 3dgrut NuRec USDZ.

3dgrut's `threedgrut/export/scripts/ply_to_usd.py` (via
`threedgrut/export/usd_util.py:serialize_nurec_usd`) unconditionally authors

    matrix4d xformOp:transform = ( (-1,0,0,0), (0,0,-1,0), (0,-1,0,0), (0,0,0,1) )

on the NuRec Volume prim, with `omni:nurec:useProxyTransform = 0` so the
renderer applies it. In USD's row-vector convention that maps
(x, y, z) -> (-x, -z, -y): it converts 3dgrut's *normalized* frame (average
camera "down" aligned with +Y, see `estimate_normalizing_transform`) into the
declared Z-up stage frame.

A PLY exported from this repo is already in the MPS world frame -- gravity
aligned, Z-up, metres -- and `ply_to_usd.py` runs with `dataset=None`, so no
normalizing transform is applied and the payload keeps MPS coordinates
verbatim. The baked rotation is therefore wrong here: it sends +Z (up) to -Y
and lays the scene on its side in Isaac Sim.

This script rewrites the volume's `xformOp:transform` to identity inside the
USDZ, so the asset renders in the MPS world frame with no extra transform.
It is deliberately conservative: it only edits the transform if it exactly
matches the known 3dgrut conversion matrix (or is already identity, in which
case it is a no-op), and aborts on anything else rather than guess.

Run with any Python that has `pxr` (usd-core), e.g. the 3dgrut env:

    /home/sun/miniforge3/envs/3dgrut/bin/python \
        scripts/fix_nurec_usdz_frame.py <asset.usdz>          # fix in place
        scripts/fix_nurec_usdz_frame.py in.usdz -o out.usdz   # write elsewhere

Exit codes: 0 fixed or already-identity, 2 unexpected transform / no NuRec
volume found.
"""

import argparse
import os
import shutil
import sys
import tempfile
import zipfile

from pxr import Gf, Sdf

# The matrix serialize_nurec_usd bakes when no normalizing transform is given
# ("Default conversion matrix from 3DGRUT to USDZ").
THREEDGRUT_CONV_TF = Gf.Matrix4d(
    -1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -1.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)
IDENTITY = Gf.Matrix4d(1.0)


def find_nurec_volume_spec(layer):
    """Return the first prim spec in `layer` that is a NuRec volume."""
    stack = list(layer.rootPrims)
    while stack:
        spec = stack.pop(0)
        is_nurec = spec.properties.get("omni:nurec:isNuRecVolume") is not None
        if (spec.typeName == "Volume" or is_nurec) and spec.properties.get(
            "xformOp:transform"
        ):
            return spec
        stack.extend(spec.nameChildren)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("usdz", help="NuRec .usdz produced by 3dgrut's ply_to_usd.py")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="write the fixed asset here instead of replacing the input in place",
    )
    args = parser.parse_args()

    out_path = args.output or args.usdz

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(args.usdz) as zin:
            names = zin.namelist()  # preserve order: root layer must stay first
            zin.extractall(tmpdir)

        edited_name = None
        for name in names:
            if not name.lower().endswith((".usda", ".usd", ".usdc")):
                continue
            layer = Sdf.Layer.FindOrOpen(os.path.join(tmpdir, name))
            if layer is None:
                continue
            spec = find_nurec_volume_spec(layer)
            if spec is None:
                continue

            attr = spec.properties["xformOp:transform"]
            current = Gf.Matrix4d(attr.default)
            if current == IDENTITY:
                print(f"{name}: {spec.path} transform already identity -- nothing to do")
                return 0
            if current != THREEDGRUT_CONV_TF:
                print(
                    f"ERROR: {name}: {spec.path} has an unexpected transform, refusing "
                    f"to touch it:\n{current}",
                    file=sys.stderr,
                )
                return 2

            attr.default = IDENTITY
            layer.Save()
            edited_name = name
            print(f"{name}: {spec.path} xformOp:transform: 3dgrut conversion -> identity")
            break

        if edited_name is None:
            print("ERROR: no NuRec Volume prim with xformOp:transform found", file=sys.stderr)
            return 2

        # Rewrite the zip with the same entry order and storage (uncompressed),
        # mirroring 3dgrut's write_to_usdz.
        fd, tmpzip = tempfile.mkstemp(suffix=".usdz", dir=os.path.dirname(out_path) or ".")
        os.close(fd)
        try:
            with zipfile.ZipFile(tmpzip, "w", compression=zipfile.ZIP_STORED) as zout:
                for name in names:
                    zout.write(os.path.join(tmpdir, name), name)
            shutil.move(tmpzip, out_path)
        finally:
            if os.path.exists(tmpzip):
                os.unlink(tmpzip)

    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
