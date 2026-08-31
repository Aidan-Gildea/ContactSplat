# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Extract a TSDF-fused triangle mesh from a trained Gaussian splat.

Gaussians are not a surface, so meshing is a separate extraction step:

1. Load the trained model the same way ``render_lightning.py`` does -- hydra
   config, ``initialize_eval_info`` for the scene/camera setup, dispatch on
   ``cfg.train_model`` (3dgs -> VanillaGSplat, 2dgs -> Gaussians2D), then
   ``module.load_ply``.
2. Render expected z-depth (gsplat ``render_mode="RGB+ED"``) at the training
   camera poses. Rolling-shutter pose arrays are disabled for this pass: each
   frame is rendered from its center-row pose. At walking speed the 10.1 ms
   RGB readout moves the camera ~1.5 cm, under the TSDF truncation band.
3. Mask unreliable depth: pixels outside the rectified valid mask
   (``mask.png``), pixels with low accumulated alpha (sky / unmodelled
   background, where expected depth is meaningless), and depth outside
   [--depth-min, --depth-trunc].
4. Integrate into an Open3D ``ScalableTSDFVolume`` with each camera's
   world-to-camera extrinsic and pinhole intrinsics, then marching-cubes.

The camera poses come from the MPS closed-loop trajectory, so the output mesh
lives in the MPS world frame (gravity-aligned, Z-up, metric metres) -- the
same frame as the trained splat. No registration step is needed.

Max fusion depth (--depth-trunc, default 6.0 m): the training config clips
rendered depth to [0.1, 7.0] (conf/config.yaml render.depth_min/depth_max)
and the RGB sparse depth used during preprocessing has median ~5.1 m, so the
splat's depth is best constrained inside that band. Splat expected-depth
error grows with range (weaker parallax, bigger Gaussians), and TSDF fusion
with sdf_trunc=0.08 assumes depth noise well under ~8 cm -- far-field depth
would smear the volume rather than add usable surface. The walk covers the
whole evaluated footprint, so every walkable cell is seen from ~2 m by some
camera and a 6 m cutoff costs no floor coverage.

~4,700 training frames at 30 fps are heavily redundant for fusion, so frames
are subsampled by pose delta (keep a frame when it moved --min-trans metres
or rotated --min-rot-deg degrees since the last kept frame); what is dropped
is logged and recorded in the metadata JSON.

Usage (ego_splats env, GPU required):

    python scripts/extract_mesh_tsdf.py \
        --ply <filtered point_cloud.ply> \
        --data-dir /home/sun/aria/processed/<scene>/<rectified folder> \
        --output <mesh.ply>

The input PLY should be the aggressively filtered depth-render copy produced
by ``scripts/filter_splat_outliers.py --min-opacity ...`` -- low-opacity
floaters punch spurious holes into the TSDF. Do not feed the visual asset.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import open3d as o3d  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

import tsdf_fusion  # noqa: E402  (shared fusion stage, also used by E3)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ply", type=Path, required=True,
                    help="trained (and outlier/opacity-filtered) splat PLY")
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="preprocessed rectified folder containing "
                         "transforms_with_sparse_depth.json")
    ap.add_argument("--output", type=Path, required=True,
                    help="output mesh path (.ply)")
    ap.add_argument("--train-model", default="3dgs", choices=["3dgs", "2dgs"],
                    help="model type the PLY was trained with (default 3dgs)")
    ap.add_argument("--opt", default="simple_gsplat_30K",
                    help="opt config group used at training time")
    ap.add_argument("--override", action="append", default=[],
                    help="extra hydra overrides (repeatable)")
    # TSDF parameters
    ap.add_argument("--voxel-length", type=float, default=0.02,
                    help="TSDF voxel size [m] (default 0.02)")
    ap.add_argument("--sdf-trunc", type=float, default=0.08,
                    help="TSDF truncation distance [m] (default 0.08)")
    ap.add_argument("--depth-min", type=float, default=0.1,
                    help="discard depth below this [m] (default 0.1, matches "
                         "conf render.depth_min)")
    ap.add_argument("--depth-trunc", type=float, default=6.0,
                    help="max fusion depth [m] (default 6.0; see module "
                         "docstring for the rationale)")
    ap.add_argument("--alpha-thresh", type=float, default=0.5,
                    help="discard pixels with accumulated alpha below this "
                         "(default 0.5; low alpha = unmodelled background)")
    # frame subsampling
    ap.add_argument("--min-trans", type=float, default=0.10,
                    help="keep a frame after this much translation [m] "
                         "(default 0.10)")
    ap.add_argument("--min-rot-deg", type=float, default=10.0,
                    help="or after this much rotation [deg] (default 10)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="debug: stop after this many fused frames (0 = all)")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def build_module(args):
    """Load config, scene and model the way render_lightning.py does."""
    overrides = [
        f"train_model={args.train_model}",
        f"opt={args.opt}",
        # single-pose rendering: no rolling-shutter motion arrays at fusion time
        "opt.handle_rolling_shutter=false",
        f"scene.load_ply={args.ply}",
        "render.render_only=true",
    ] + args.override

    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"),
                               version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    from scene import initialize_eval_info

    # render_lightning.py points scene.source_path at the folder holding the
    # transforms json; initialize_eval_info also disables the 3D smooth filter
    # and the trainer viewer.
    cfg.scene.source_path = str(args.data_dir)
    scene_info = initialize_eval_info(cfg)

    if "3dgs" in cfg.train_model:
        from model.vanilla_gsplat import VanillaGSplat
        module = VanillaGSplat(cfg=cfg, scene_info=scene_info)
    elif "2dgs" in cfg.train_model:
        from model.GS2D_gsplat import Gaussians2D
        module = Gaussians2D(cfg=cfg, scene_info=scene_info)
    else:
        raise RuntimeError(f"cannot recognize the train model {cfg.train_model}")

    assert Path(cfg.scene.load_ply).exists(), \
        f"Need a valid ply file to load from! {cfg.scene.load_ply} does not exist!"
    module.load_ply(cfg.scene.load_ply)
    module.to(args.device)
    module.eval()
    return cfg, scene_info, module


def subsample_by_pose_delta(cameras, min_trans: float, min_rot_deg: float):
    """Keep a frame when it moved/rotated enough since the last kept frame."""
    cameras = sorted(cameras, key=lambda c: c.time_s)
    kept_idx = tsdf_fusion.subsample_indices_by_pose_delta(
        [cam.c2w_44_np for cam in cameras], min_trans, min_rot_deg
    )
    return cameras, [cameras[i] for i in kept_idx]


def render_depth_frame(module, camera, alpha_thresh, depth_min, depth_trunc):
    """Render one frame; return (color_u8 HWC, depth_f32 HW) fusion-ready."""
    with torch.no_grad():
        pkg = module.render(camera)

    depth = pkg["depth"].squeeze().clone()          # (H, W), z-depth [m]
    alpha = pkg["alphas"].squeeze()                 # (H, W)

    invalid = alpha < alpha_thresh
    invalid |= depth < depth_min
    invalid |= depth > depth_trunc
    invalid |= ~torch.isfinite(depth)

    valid_mask = camera.valid_mask
    if valid_mask is not None:
        invalid |= (valid_mask.to(depth.device).squeeze() < 0.5)

    depth[invalid] = 0.0

    image = pkg["render"][0]                        # (3, H, W) linear irradiance
    image = camera.expose_image(irradiance=image, clamp=False)
    image = torch.clamp(image, 0.0, 1.0)
    color_u8 = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    depth_np = depth.cpu().numpy().astype(np.float32)
    n_valid = int((~invalid).sum().item())
    return np.ascontiguousarray(color_u8), depth_np, n_valid


def main() -> int:
    args = parse_args()
    t0 = time.time()

    cfg, scene_info, module = build_module(args)
    t_load = time.time() - t0
    print(f"[extract_mesh_tsdf] model + scene loaded in {t_load:.1f}s, "
          f"{module.total_points} Gaussians")

    all_cams, kept = subsample_by_pose_delta(
        scene_info.all_cameras, args.min_trans, args.min_rot_deg
    )
    if args.max_frames > 0:
        kept = kept[: args.max_frames]
    print(f"[extract_mesh_tsdf] pose-delta subsampling "
          f"(>= {args.min_trans} m or >= {args.min_rot_deg} deg): "
          f"kept {len(kept)}/{len(all_cams)} frames, "
          f"dropped {len(all_cams) - len(kept)} redundant ones")

    volume = tsdf_fusion.make_tsdf_volume(
        voxel_length=args.voxel_length, sdf_trunc=args.sdf_trunc
    )

    t1 = time.time()
    valid_px_total = 0
    total_px = 0
    for i, cam in enumerate(kept):
        color_u8, depth_np, n_valid = render_depth_frame(
            module, cam, args.alpha_thresh, args.depth_min, args.depth_trunc
        )
        valid_px_total += n_valid

        h, w = depth_np.shape
        total_px += h * w
        tsdf_fusion.integrate_frame(
            volume, color_u8, depth_np, cam.intrinsic_np,
            cam.w2c_44_np, args.depth_trunc,
        )

        if (i + 1) % 100 == 0 or (i + 1) == len(kept):
            rate = (i + 1) / (time.time() - t1)
            print(f"  fused {i + 1}/{len(kept)} frames "
                  f"({rate:.2f} fps, {n_valid / (h * w):.1%} valid px last)")

    t_fuse = time.time() - t1
    print(f"[extract_mesh_tsdf] fusion took {t_fuse:.1f}s "
          f"({len(kept) / max(t_fuse, 1e-9):.2f} fps)")

    t2 = time.time()
    mesh = tsdf_fusion.extract_mesh(volume)
    t_mesh = time.time() - t2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(args.output), mesh)
    bbox = mesh.get_axis_aligned_bounding_box()
    print(f"[extract_mesh_tsdf] mesh: {len(mesh.vertices)} vertices, "
          f"{len(mesh.triangles)} triangles (marching cubes {t_mesh:.1f}s)")
    print(f"[extract_mesh_tsdf] AABB {np.round(bbox.min_bound, 2).tolist()} .. "
          f"{np.round(bbox.max_bound, 2).tolist()}")
    print(f"[extract_mesh_tsdf] wrote {args.output}")

    meta = {
        "ply": str(args.ply),
        "data_dir": str(args.data_dir),
        "train_model": args.train_model,
        "gaussians": int(module.total_points),
        "voxel_length": args.voxel_length,
        "sdf_trunc": args.sdf_trunc,
        "depth_min": args.depth_min,
        "depth_trunc": args.depth_trunc,
        "alpha_thresh": args.alpha_thresh,
        "min_trans": args.min_trans,
        "min_rot_deg": args.min_rot_deg,
        "frames_total": len(all_cams),
        "frames_fused": len(kept),
        "valid_px_mean_fraction": round(valid_px_total / max(total_px, 1), 4),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "aabb_min": np.asarray(bbox.min_bound).round(4).tolist(),
        "aabb_max": np.asarray(bbox.max_bound).round(4).tolist(),
        "time_load_s": round(t_load, 1),
        "time_fuse_s": round(t_fuse, 1),
        "time_mesh_s": round(t_mesh, 1),
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[extract_mesh_tsdf] metadata -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
