# paper_evaluations: grading a collider against an iPad LiDAR scan

These scripts grade a ContactSplat collider against an independent reference: a same-day iPad LiDAR scan of the place, exported from RTAB-Map as a point cloud. They are evaluation tools for the paper and are not part of the pipeline in `run_contactsplat.sh`.

The scan is called *reference geometry*, not ground truth. Its own error is measured, and only thresholds of at least twice that error are trusted.

| Script | What it does | Writes |
|---|---|---|
| `eval_reference.py` | Aligns the scan to the MPS world frame: levels the scan's floor, runs a 1° yaw sweep with top-down map correlation, then point-to-plane ICP of the confident MPS points onto the scan (never the mesh being graded). Then checks the alignment: the score at the other three quarter turns, a free-scale fit, floor tilt against MPS "up", and the error budget σ = √(σ_scan² + σ_align²). | `report.json` (includes `T_scan_raw_to_mps`), `alignment_overlay.png`, `scan_in_mps_frame.ply` |
| `score_cropped.py` | Scores the decimated collider inside the scan's footprint and in the contact band (up to 1 m above the local ground): precision, recall and F at 2/5/10/20 cm, plus how much of the scanned ground has collider within 5 and 10 cm. `--sim3` first refits the scan with free scale against the MPS points. | `cropped_<mesh>.json`, `cropped_<mesh>.png` (error map) |

## Run

Use the same environment as the pipeline. The scripts also need `scipy`, which isn't in `requirements.txt`.

```bash
python paper_evaluations/eval_reference.py --ipad room.ply --slam ~/recordings/mps_Room_vrs/slam \
  --mesh output/mps-mesh/Room/camera-rgb-rectified-1008-h1512/mesh_delaunay_decimated.ply \
  --eye 1.673 --out eval/room
python paper_evaluations/score_cropped.py --scan room.ply --report eval/room/report.json \
  --mesh output/mps-mesh/Room/camera-rgb-rectified-1008-h1512/mesh_delaunay_decimated.ply \
  --slam ~/recordings/mps_Room_vrs/slam --out eval/room
```

`--eye` is the wearer's eye height in metres. It's only a fallback for finding the MPS floor.

## Notes

- Precision, recall and F-score follow Tanks and Temples (Knapitsch et al., 2017). `eval_reference.py` also prints whole-scan scores. For a partial outdoor scan use `score_cropped.py`, because mesh outside the scan's footprint would otherwise count as error.
- The alignment uses the same MPS points the mesh is built from, so the reference is not fully independent of the mesh.
- The paper's numbers were made with these exact files: md5 `2e23b3e1…` (`eval_reference.py`) and `1f6a2d78…` (`score_cropped.py`).
