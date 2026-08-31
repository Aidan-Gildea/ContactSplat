# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate a trained splat PLY on the held-out test split (matched protocol).

Runs the repo's own `test_step` path (same ImageLoss, same masking, same
exposure) over the standard 7-1 test split, but from a saved PLY and with a
caller-chosen rolling-shutter setting. Written for the E2 3DGS-vs-2DGS
comparison: the 2DGS model cannot render the rolling-shutter motion array on
16 GB (each eval render becomes a 4-8 camera batched rasterization call and
OOMs), so it trains and evaluates with RS off -- and the 3DGS baseline must
then be re-evaluated with RS off as well for the PSNR/SSIM/LPIPS comparison
to be like-for-like.

Results land in <output-dir>/test_logs.json (per-frame) and a printed
mean summary. The model's own directory is not touched.

Usage (ego_splats env, GPU):

    python scripts/eval_splat_test.py \
        --ply output/.../point_cloud/iteration_30000/point_cloud.ply \
        --train-model 3dgs \
        --output-dir output/.../test_rs_off
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ply", type=Path, required=True)
    ap.add_argument("--train-model", default="3dgs", choices=["3dgs", "2dgs"])
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="where test_logs.json is written (NOT the model dir)")
    ap.add_argument("--data-root", default="/home/sun/aria/processed")
    ap.add_argument("--scene-name",
                    default="Outside_20260812_141244/camera-rgb-rectified-1008-h1512")
    ap.add_argument("--opt", default="simple_gsplat_30K")
    ap.add_argument("--rolling-shutter", action="store_true",
                    help="evaluate WITH the rolling-shutter motion array "
                         "(default off; 2DGS OOMs with it on 16 GB)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rs = "true" if args.rolling_shutter else "false"
    overrides = [
        f"train_model={args.train_model}",
        f"opt={args.opt}",
        f"opt.handle_rolling_shutter={rs}",
        f"scene.data_root={args.data_root}",
        f"scene.scene_name={args.scene_name}",
        "scene.input_format=aria",
        f"scene.load_ply={args.ply}",
        "scene.save_ply=false",           # do not re-save the PLY on test end
        f"exp_name=_eval_scratch",        # model_path -> scratch, see below
        "output_root=./output",
        "viewer.use_trainer_viewer=false",
    ]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"),
                               version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # test_logs.json is written to cfg.scene.model_path in on_test_epoch_end;
    # point it at the requested output dir so nothing of a real run is clobbered.
    cfg.scene.model_path = str(args.output_dir)
    # test renders (image/depth/normal) go with the eval results too
    cfg.render.render_output = str(args.output_dir / "render")

    import lightning as L
    from scene import get_data_loader, initialize_scene_info

    scene_info = initialize_scene_info(cfg.scene)

    if "3dgs" in cfg.train_model:
        from model.vanilla_gsplat import VanillaGSplat
        module = VanillaGSplat(cfg=cfg, scene_info=scene_info)
    else:
        from model.GS2D_gsplat import Gaussians2D
        module = Gaussians2D(cfg=cfg, scene_info=scene_info)

    test_loader = get_data_loader(scene_info.test_cameras, subset="test",
                                  shuffle=False)

    trainer = L.Trainer(accelerator="gpu", devices=1, logger=False,
                        enable_checkpointing=False)
    trainer.test(model=module, dataloaders=test_loader)

    # summarize
    logs = json.loads((args.output_dir / "test_logs.json").read_text())
    summary = {}
    for k in ("psnr", "ssim", "lpips"):
        vals = [v for v in logs.get(k, {}).values() if isinstance(v, (int, float))]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "n": len(vals)}
    summary["rolling_shutter"] = args.rolling_shutter
    summary["ply"] = str(args.ply)
    summary["train_model"] = args.train_model
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
