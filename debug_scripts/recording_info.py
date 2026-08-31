#!/usr/bin/env python3
"""Print the facts about a recording that you need before preprocessing.

Reports the device generation, the camera labels present, the RGB frame count
and resolution, and the rolling-shutter readout time of every camera. It ends
with the exact rectification flags to pass for a 90 degree horizontal field of
view at the sensor's native resolution.

    python debug_scripts/recording_info.py --vrs /path/to/my_recording.vrs

The MPS folder is derived from the VRS path unless you pass --mps. Readout
times come from MPS online calibration, because factory calibration does not
carry them on either generation.
"""

import argparse
import math
import sys
from pathlib import Path

from projectaria_tools.core import data_provider, mps

RGB_LABEL = "camera-rgb"


def derive_mps_folder(vrs_path: Path) -> Path:
    """MPS names its output folder after the VRS file, with dots as underscores."""
    return vrs_path.parent / f"mps_{vrs_path.stem}_vrs" / "slam"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vrs", required=True, type=Path, help="path to the .vrs recording")
    ap.add_argument("--mps", type=Path, default=None,
                    help="path to the MPS slam folder (default: derived from --vrs)")
    ap.add_argument("--fov", type=float, default=90,
                    help="target horizontal FOV for the suggested flags (default: 90)")
    args = ap.parse_args()

    vrs_path = args.vrs.expanduser()
    if not vrs_path.exists():
        sys.exit(f"cannot find {vrs_path}")

    mps_folder = (args.mps.expanduser() if args.mps else derive_mps_folder(vrs_path))

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    calib = provider.get_device_calibration()

    print(f"vrs            {vrs_path}")
    print(f"device         {calib.get_device_version().name}")
    print(f"cameras        {sorted(calib.get_camera_labels())}")
    print()

    # Resolution is read from a decoded frame, not from the calibration object.
    # extract_aria_vrs.py does the same, noting that the calibration image size
    # can be wrong.
    stream_id = provider.get_stream_id_from_label(RGB_LABEL)
    n_frames = provider.get_num_data(stream_id)
    image_data, _ = provider.get_sensor_data_by_index(
        stream_id, n_frames // 2).image_data_and_record()
    array = image_data.to_numpy_array()
    height, width = array.shape[0], array.shape[1]

    print(f"rgb frames     {n_frames}")
    print(f"rgb resolution {width} x {height}   (width x height)")
    print()

    if mps_folder.exists():
        calib_file = mps_folder / "online_calibration.jsonl"
        if calib_file.exists():
            online = mps.read_online_calibration(str(calib_file))
            for cam in online[0].camera_calibs:
                readout = cam.get_readout_time_sec()
                ms = 0.0 if readout is None else readout * 1e3
                note = "  (global shutter)" if readout is None else ""
                print(f"readout {cam.get_label():18s} {ms:6.2f} ms{note}")
        else:
            print(f"no online_calibration.jsonl in {mps_folder}")

        print()
        required = ["closed_loop_trajectory.csv", "online_calibration.jsonl",
                    "semidense_points.csv.gz", "semidense_observations.csv.gz"]
        for name in required:
            present = (mps_folder / name).exists()
            print(f"  {'found  ' if present else 'MISSING'} {name}")
    else:
        print(f"MPS folder not found at {mps_folder}")
        print("Pass --mps if it lives somewhere else.")

    focal = (width / 2) / math.tan(math.radians(args.fov) / 2)
    print()
    print(f"For {args.fov:.0f} degrees horizontal at native resolution:")
    print(f"    --rectified_rgb_focal {focal:.0f} --rectified_rgb_size {height}")
    print(f"    --rectified_monochrome_focal 180 --rectified_monochrome_height 512")
    print()
    print(f"Rectified folder will be named camera-rgb-rectified-{focal:.0f}-h{height}")


if __name__ == "__main__":
    main()
