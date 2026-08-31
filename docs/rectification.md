# Choosing the rectification values

Preprocessing resynthesises every fisheye frame as if an ideal pinhole camera had taken
it. Two flags control that synthetic camera, and both depend on the resolution your
recording actually used:

```
--rectified_rgb_focal    the focal length of the synthetic camera, in pixels
--rectified_rgb_size     the output HEIGHT in pixels, not the width
```

If you record with **profile10** you can use `1008` and `1512` and skip this page. Any
other sensor mode needs its own pair of numbers, and this page explains how to work them
out.

---

## Why the height and not the width

The width is derived from the source aspect ratio, in one line of
`scripts/aria_utils.py`:

```python
if rectified_w is None:
    rectified_w = int((rectified_h / frame.h) * frame.w)
```

So for a 4:3 sensor, a height of 1512 gives a width of 2016. You set the height, the
pipeline computes the width, and the focal length is then chosen against that computed
width.

Set the height to your sensor's **native** height. A larger height interpolates between
real pixels and cannot add detail that was never captured, so you pay proportionally more
disk, GPU memory and time per training iteration for a blurrier version of the same
information. A smaller height is a legitimate choice if you want a cheaper run, but you
must scale the focal by the same ratio or you will narrow the field of view as well as
shrink the image.

---

## Step 1: find your RGB resolution

Run the helper against your recording:

```bash
python debug_scripts/recording_info.py --vrs "/path/to/my_recording.vrs"
```

```
device         Gen2
cameras        ['camera-rgb', 'slam-front-left', 'slam-front-right', ...]

rgb frames     4719
rgb resolution 2016 x 1512   (width x height)

readout camera-rgb          10.10 ms
readout slam-front-left      0.00 ms  (global shutter)
```

`projectaria_tools` writes decoder logs to stderr, so add `2>/dev/null` if you want only
the report.

The script reads the resolution from a decoded frame in the middle of the recording rather
than from the calibration object. That is deliberate, and it copies what the pipeline
itself does. `to_frame_json` in `scripts/extract_aria_vrs.py` carries the comment "the
calibration image size might be incorrect due to API issue" and takes its dimensions from
the decoded array instead. Trust the pixels.

Zero readout on the SLAM cameras is correct rather than missing. Every Aria SLAM camera is
global shutter on both generations, so there is no rolling readout to report.

---

## Step 2: solve for the focal length

Focal length decides how much of the scene fits into the image. The relationship is:

```
FOV = 2 * atan((width / 2) / focal)
```

Rearranged for the focal length you need:

```
focal = (width / 2) / tan(FOV / 2)
```

There is one shortcut worth memorising. When the focal length equals half the width, the
horizontal field of view is exactly 90 degrees.

Run the helper:

```bash
python debug_scripts/solve_focal.py --width 2016 --height 1512
```

```
  FOV     focal
  -----   -----
   60      1746
   70      1440
   80      1201
   90      1008  <- target
  100       846
  110       706
  120       582

For 90 degrees at width 2016, focal is 1008.

Flags to pass:
    --rectified_rgb_focal 1008 --rectified_rgb_size 1512
```

To go the other way and check what a focal you already have produces:

```bash
python debug_scripts/solve_focal.py --width 2016 --focal 756
```

---

## The trade you are making

The pixel budget is fixed by the sensor. Focal length decides how you spend it across the
scene, and there is no setting that gives you both coverage and detail.

| Focal at width 2016 | Horizontal FOV | Result |
|---|---|---|
| `1512` | 67.4 degrees | Sharp but narrow. Discards a lot of what the lens saw. |
| `1008` | 90.0 degrees | Balanced. Matches what the Gen 1 defaults effectively produced. |
| `756` | 106.3 degrees | More of the room per frame, less detail per degree. |

Start at 90 degrees. It is a reasonable default for indoor and outdoor scenes alike, and it
gives you a baseline to compare against if you later decide to retune.

Gen 2 RGB covers roughly 133 degrees horizontally, which is more than a pinhole can
usefully represent. Some cropping is therefore unavoidable. If you push the target field
of view too wide, output pixels start asking for directions the lens never saw, and
`undistort_image` fills those with black. A small amount is harmless, since the rectified
`mask.png` covers it, but a large amount means you are training on invented black regions.
If you genuinely need the full 133 degrees, `--extract_fisheye` writes an equidistant
fisheye projection instead, which the rasterizer also supports.

---

## Worked examples

### profile10, 2016 x 1512 at 30 Hz

This is the recommended mode. Frame rate matters more than resolution for splat coverage,
because more viewpoints means more parallax and better constrained geometry.

```bash
--rectified_rgb_focal 1008 --rectified_rgb_size 1512
```

Width follows as 2016, and focal 1008 is half of that, so the horizontal field of view is
90 degrees at native sampling.

### profile8, 2560 x 1920 at 10 Hz

```bash
--rectified_rgb_focal 1280 --rectified_rgb_size 1920
```

Width follows as 2560, and 1280 is half of that, so again 90 degrees at native sampling.

### The mistake these two examples exist to prevent

`scripts/bash_local/run_vrs_preprocessing_gen2.sh` ships the profile8 pair. Using it on a
profile10 recording resamples every frame **up** to 2560 x 1920. That costs about 30 per
cent more disk, memory and training time and adds no information at all, because the extra
pixels are interpolated from ones that were already there. Nothing errors and nothing warns
you. The run simply takes longer and the result is no better.

---

## The SLAM cameras

Less thought is required here. Gen 2 SLAM is 512 x 512 with a 119 degree lens, and focal
180 at height 512 keeps about 110 degrees of it:

```bash
--rectified_monochrome_focal 180 --rectified_monochrome_height 512
```

Use those unless you have a specific reason not to. You will almost certainly not train on
the SLAM cameras, since their contribution is generating the sparse depth that supervises
the RGB view.

Note that both focal flags default to `-1`, which means skip that camera entirely. Omit
them and preprocessing finishes quickly having produced nothing.

---

## What the numbers become

The pair you choose is encoded into the output folder name:

```
{camera_label}-rectified-{int(focal)}-h{height}
```

So `1008` and `1512` produce `camera-rgb-rectified-1008-h1512`. That exact string is what
you later pass to training as `scene.scene_name`, so it changes whenever you retune, and
the training command has to change with it. A mismatch between the two is the most common
reason a training launch fails immediately.

The naming scheme has a useful consequence. Every parameter combination lands in its own
folder, so experiments never overwrite each other and the folder name is a complete record
of how its contents were made. You can preprocess at three focal lengths and train on all
three without editing a config file. Frame extraction is shared between them and gets
skipped on the second and third runs, so only the rectification work is repeated.
