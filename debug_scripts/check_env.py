#!/usr/bin/env python3
"""Check that the environment can run both halves of the pipeline.

Preprocessing needs projectaria_tools and OpenCV. Training additionally needs a
CUDA build of PyTorch. Run this after installing, before spending hours on a
run that will fail at import.

    python debug_scripts/check_env.py
"""

import importlib
import sys

PREPROCESS = ["projectaria_tools", "cv2", "numpy", "pandas", "PIL", "tqdm"]
TRAIN = ["torch", "gsplat", "lightning", "einops", "omegaconf", "plyfile"]


def probe(names):
    ok = True
    for name in names:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "")
            print(f"  found   {name:22s} {version}")
        except Exception as exc:
            print(f"  MISSING {name:22s} {type(exc).__name__}")
            ok = False
    return ok


def main() -> None:
    print(f"python  {sys.version.split()[0]}")
    print(f"prefix  {sys.prefix}")
    print()

    print("Preprocessing (Block 1)")
    preprocess_ok = probe(PREPROCESS)

    print()
    print("Training (Block 2)")
    train_ok = probe(TRAIN)

    print()
    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"torch            {torch.__version__}")
        print(f"cuda available   {cuda_ok}")
        if cuda_ok:
            print(f"device           {torch.cuda.get_device_name(0)}")
        else:
            print("  Training will not run. Reinstall torch from the CUDA index")
            print("  matching the version nvidia-smi reports.")
    except Exception:
        print("torch not importable, so training cannot run")

    print()
    if preprocess_ok and train_ok and cuda_ok:
        print("Ready for Block 1 and Block 2.")
    elif preprocess_ok:
        print("Ready for Block 1. Block 2 needs the items above resolved.")
    else:
        print("Block 1 cannot run yet. Install the missing packages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
