# ContactSplat
Walk through a place once with Aria Gen 2 glasses, then drive a robot through it in Isaac Sim

<div align="center">
  <img src="media/isaacsim_drive.gif" alt="A Nova Carter robot driving through a ContactSplat scene in Isaac Sim" style="max-width: 100%; height: auto;"/>
</div>

ContactSplat turns one Project Aria Gen 2 recording into a simulation scene for NVIDIA Isaac Sim. It trains a 3D Gaussian splat on the RGB video for <ins>appearance</ins>, builds a triangle mesh from the MPS SLAM point cloud for <ins>collisions</ins>, and packs both into <ins>one .usdz file</ins>. Both come out in the MPS world frame (metres, Z up), so they line up with no alignment step. The preprocessing and splat training are from Meta's [egocentric_splats](https://github.com/facebookresearch/egocentric_splats).

```mermaid
flowchart TD
    vrs["<b>Recording (.vrs)</b><br/>RGB video"]
    mps["<b>MPS slam/</b><br/>Trajectory, points"]
    frames["<b>Frames</b><br/>Rectified and posed"]
    splat["<b>Gaussian splat</b><br/>What you see"]
    mesh["<b>Collision mesh</b><br/>What the robot touches"]
    usdz["<b>Scene (.usdz)</b><br/>Splat + hidden mesh"]
    sim["<b>Isaac Sim</b><br/>Open, press Play"]

    vrs --> frames
    mps --> frames
    frames --> splat
    mps --> mesh
    splat --> usdz
    mesh --> usdz
    usdz --> sim

    classDef input fill:#EEEDFE,stroke:#534AB7,color:#3C3489
    classDef step fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef output fill:#E1F5EE,stroke:#0F6E56,color:#085041
    class vrs,mps input
    class frames,splat,mesh step
    class usdz,sim output
```

## User Guide

You need:
- **Linux with an NVIDIA GPU.** 16 GB was enough for the outdoor recordings; the indoor Room sample needed a 48 GB GPU.
- **A Python 3.10 environment with CUDA PyTorch 2.6 or newer**, plus the CUDA toolkit (`nvcc`) that matches it. gsplat compiles its CUDA code the first time it runs.
- **[COLMAP](https://colmap.github.io/install.html) on your `PATH`**, built with CGAL. Package-manager builds include it. CUDA is only needed for the optional multi-view stereo mesh.
- **Isaac Sim 5.1** to open the result.

**1. Install.**

```bash
git clone --branch mps-mesh https://github.com/Aidan-Gildea/ContactSplat.git
cd ContactSplat
pip install -r requirements.txt
```

**2. Get MPS SLAM output** for your recording. The pipeline expects it in the same folder as the `.vrs`, named `mps_<recording>_vrs/`, which is where `aria_mps` puts it. If it is somewhere else, set `MPS_FOLDER` to its `slam/` folder. Sample recordings with their MPS output are in the [ContactSplat Sample Trajectories](https://drive.google.com/drive/folders/1INGZFiCD6qYtHSzbXo4Qegmj-flwikv0) folder.

```text
~/recordings/
├── Outside_20260812_141244.vrs
└── mps_Outside_20260812_141244_vrs/
    └── slam/
```

`aria_mps` asks you to sign in with your Project Aria account, uploads the recording and downloads the result. Install it with `pipx`, because its pinned packages clash with the training packages.

```bash
pipx install projectaria-mps
aria_mps single -i ~/recordings/Outside_20260812_141244.vrs --features SLAM
```

The defaults assume Aria recording profile `profile10` (RGB 2016×1512 at 30 Hz). For any other profile, `python debug_scripts/recording_info.py --vrs <file>` prints the focal length and height to set as `RGB_FOCAL` and `RGB_HEIGHT`.

**3. Run the pipeline.** Pass the `.vrs`. A first run takes a few hours: extracting the frames and training the splat each take one to three hours, depending on the recording's length. The mesh takes about a minute. If you stop and rerun it, the frames, mesh and splat stages that already finished are skipped. A stopped training run starts over.

```bash
bash run_contactsplat.sh ~/recordings/Outside_20260812_141244.vrs
```

| Stage | Script | What it makes |
| --- | --- | --- |
| 1. Frames | `scripts/bash_local/run_gen2_outside.sh` | Undistorted RGB frames, each with its camera pose from MPS |
| 2. Mesh | `mps-mesh/run_mps_mesh.sh` | A mesh from the MPS points, plus a simplified copy used as the collider |
| 3. Splat | `scripts/bash_local/train_gen2_outside.sh` | The Gaussian splat, trained for 30,000 steps |
| 4. Scene | `isaacsim/splat_and_mesh_to_usdz.py` | One `.usdz` with the splat (visible) and the mesh (invisible collider) |

This is where everything is written:

```text
~/recordings/
└── processed/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/        1. frames
ContactSplat/output/
├── mps-mesh/Outside_20260812_141244/camera-rgb-rectified-1008-h1512/
│   ├── mesh_delaunay_decimated.ply                                           2. collider
│   └── report_decimated.md                                                   mesh score
└── Outside_20260812_141244/camera-rgb-rectified-1008-h1512/
    ├── point_cloud/iteration_30000/point_cloud.ply                           3. splat
    ├── test_logs.json                                                        splat score
    └── isaacsim/Outside_20260812_141244_with_collider.usdz                   4. scene
```

`report_decimated.md` scores the collider: how much of the walked floor it covers, its largest hole, and its floor height error. `test_logs.json` scores the splat on frames held out from training.

**4. Open it in Isaac Sim.** Use File > Open on the `.usdz` and press Play to turn on collisions. The splat is what you see. The mesh is invisible and only collides.

<div align="center">
  <img src="media/splat_and_collider.jpg" alt="The same view in Isaac Sim: the Gaussian splat on the left, the collision mesh on the right" width="600"/>
  <p><i>Left: the splat you see. Right: the hidden collision mesh the robot drives on.</i></p>
</div>

**5. Tune (optional).** Every setting is at the top of `run_contactsplat.sh`, with a note on what changing it does. Edit it there, or set it before the command, for example `ITERATIONS=15000 bash run_contactsplat.sh ...`.

| Setting | Default | Effect |
| --- | --- | --- |
| `ITERATIONS` | `30000` | Training steps. Training time grows about linearly with it |
| `CAP_MAX` | `1500000` | Most Gaussians training may grow to. Higher gives more detail, more GPU memory and slower steps |
| `ROLLING_SHUTTER`, `RS_START` | `true`, `10000` | Model the camera reading out row by row, from that step on. Sharper, but slower steps |
| `DEPTH_LOSS` | `false` | Also fit the MPS sparse depth |
| `TRAIN_SPLIT` | `7-1` | `7-1` holds out every 8th frame to score the splat. `all` trains on every frame |
| `MESH_SOURCE` | `mps` | `mvs` builds the collider with COLMAP multi-view stereo instead. That takes hours and fills outdoor floors only slightly better. Known bug: frames get the wrong poses when their timestamps change length partway through a recording |
| `DECIMATE` | `1` | `0` uses the full mesh as the collider |
| `EYE_HEIGHT` | `1.6683` | Height of the glasses above the floor in metres. Only used to score the mesh |
| `RGB_FOCAL`, `RGB_HEIGHT` | `1008`, `1512` | Rectified frame size. Change for recording profiles other than `profile10` |

A changed setting only affects stages that have not run yet. To redo the mesh, add `FORCE=1`. To retrain the splat, delete `output/<recording>/<rect>/point_cloud/`.

To look at the splat on its own in a browser (port 8080):

```bash
python launch_viewer.py model_root=output/Outside_20260812_141244
```

## Recording tips

- Walk through the scene. Turning in place gives no depth.
- Record for 2 to 5 minutes, and pass earlier spots again so MPS can close loops.
- Turn your head slowly and avoid dim places. Indoors the camera exposure is long, so fast motion blurs the frames.

## Repository layout

| Path | What it does |
| --- | --- |
| `run_contactsplat.sh` | Runs the whole pipeline and holds every setting |
| `scripts/extract_aria_vrs.py` | Stage 1: reads the `.vrs` and the MPS output, writes rectified, posed frames |
| `mps-mesh/` | Stage 2: collision mesh from the MPS points |
| `photogrammetry/` | Mesh scoring and simplification used by stage 2, and the optional multi-view stereo mesh |
| `train_lightning.py`, `model/`, `scene/`, `conf/` | Stage 3: splat training (from egocentric_splats). Defaults for every training option are in `conf/` |
| `isaacsim/splat_and_mesh_to_usdz.py` | Stage 4: the Isaac Sim scene |
| `debug_scripts/`, `docs/` | Checks, and notes on Aria Gen 2 and rectification |

## Citation

This work builds on egocentric_splats:

```
@inproceedings{lv2025egosplats,
    title={Photoreal Scene Reconstruction from an Egocentric Device},
    author={Lv, Zhaoyang and Monge, Maurizio and Chen, Ka and Zhu, Yufeng and Goesele, Michael and Engel, Jakob and Dong, Zhao and Newcombe, Richard},
    booktitle={ACM SIGGRAPH},
    year={2025}
}
```

## License

[CC BY-NC 4.0](LICENSE), inherited from [egocentric_splats](https://github.com/facebookresearch/egocentric_splats). Meshing uses [COLMAP](https://colmap.github.io/), splat training uses [gsplat](https://github.com/nerfstudio-project/gsplat), and the USDZ writer adapts code from [3dgrut](https://github.com/nv-tlabs/3dgrut) (Apache 2.0).
