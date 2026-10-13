#!/usr/bin/env python
"""Score a mesh against an aligned iPad scan inside the scan's footprint (for partial outdoor scans).

eval_reference.py crops the mesh to the scan's bounding box, which suits a closed room but not a partial
outdoor scan: most of the box is space the iPad never saw, so every mesh triangle there counts as error.
This scorer reuses eval_reference.py's alignment (report.json: T_scan_raw_to_mps) and crops instead to:

  footprint     10 cm top-down cells that hold reference points, grown by --dilate cells (square), from 1 m below
                the local ground up to 0.3 m above the highest reference point within +-0.5 m. Mesh below the
                floor counts as error; mesh above the scan's reach is left out (the iPad never saw there).
  contact band  the footprint, up to --band metres above the local ground
  local ground  per cell, the 10th-percentile height of the scan; then the lowest such value within +-0.5 m
                (robust to scan noise, and below furniture tops)

Precision uses the cropped mesh samples against all visible reference points; recall uses the reference
points in the same region against all mesh samples, so neither side is penalised at the crop edge.
"ground" recall: cells whose 10th-percentile height is within 10 cm of the local ground are ground cells;
their surface height is the median of the cell's low cluster, and scan points within 3 cm of it are the
ground surface. The reference is the scan at 1 cm voxels, as in eval_reference.py.

With --sim3 the scan is first refit with free scale against the MPS points (never the mesh), so a global
scale disagreement between iPad and MPS (0.6-1% outdoors) is reported on its own instead of leaking into
the local-shape scores.

    python score_cropped.py --scan scan.ply --report <eval_out>/report.json --mesh mesh.ply --slam <slam> --out <eval_out> [--sim3]
"""
import argparse, json, os
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation, minimum_filter, maximum_filter

TAUS = (0.02, 0.05, 0.10, 0.20)


def scores(acc, comp, sigma_gt):
    out = {}
    for tau in TAUS:
        P = float((acc < tau).mean()) if len(acc) else 0.0
        R = float((comp < tau).mean()) if len(comp) else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        out[f'{int(tau * 100)}cm'] = dict(precision=round(P, 3), recall=round(R, 3), F=round(F, 3),
                                          trustworthy=bool(tau >= 2 * sigma_gt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', required=True); ap.add_argument('--report', required=True)
    ap.add_argument('--mesh', required=True); ap.add_argument('--slam', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cell', type=float, default=0.10); ap.add_argument('--dilate', type=int, default=2)
    ap.add_argument('--band', type=float, default=1.0); ap.add_argument('--vis-range', type=float, default=4.0)
    ap.add_argument('--sim3', action='store_true', help='refit the scan with free scale against the MPS points first')
    a = ap.parse_args()

    o3d.utility.random.seed(0); np.random.seed(0)
    rep = json.load(open(a.report))
    T = np.array(rep['T_scan_raw_to_mps'])
    R = np.asarray(o3d.io.read_point_cloud(a.scan).voxel_down_sample(0.01).points) @ T[:3, :3].T + T[:3, 3]
    tr = pd.read_csv(os.path.join(a.slam, 'closed_loop_trajectory.csv'),
                     usecols=['tx_world_device', 'ty_world_device', 'tz_world_device']).to_numpy()[::100]
    scale_fit = None
    if a.sim3:
        # refit with free scale against the MPS points (as eval_reference.py's scale step, never the mesh),
        # so the mesh scores measure local shape; the global scale disagreement is reported on its own
        pts = pd.read_csv(os.path.join(a.slam, 'semidense_points.csv.gz'),
                          usecols=['px_world', 'py_world', 'pz_world', 'inv_dist_std', 'dist_std'])
        pts = pts[(pts.inv_dist_std < 0.005) & (pts.dist_std < 0.01)]
        M = pts[['px_world', 'py_world', 'pz_world']].to_numpy()
        M = M[np.all((M >= R.min(0) - 0.3) & (M <= R.max(0) + 0.3), axis=1)]
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(M))
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(R)).voxel_down_sample(0.02)
        X = o3d.pipelines.registration.registration_icp(
            src, tgt, 0.05, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)).transformation
        scale_fit = float(np.cbrt(np.linalg.det(X[:3, :3])))        # scan / MPS
        Xi = np.linalg.inv(X)
        R = R @ Xi[:3, :3].T + Xi[:3, 3]                             # scan, rescaled into the MPS frame
    Rv = R[cKDTree(tr).query(R, k=1)[0] < a.vis_range]           # same visible region as eval_reference.py

    # top-down grids of the visible reference: occupancy, local ground and local top
    lo = Rv[:, :2].min(0) - 1.0
    ij = np.floor((Rv[:, :2] - lo) / a.cell).astype(int); shape = tuple(ij.max(0) + 3)
    cid = ij[:, 0] * shape[1] + ij[:, 1]
    df = pd.DataFrame({'c': cid, 'z': Rv[:, 2]})
    zlow_c = df.groupby('c')['z'].quantile(0.10)                      # robust low height per cell
    zlow = np.full(shape, np.inf); zlow.flat[zlow_c.index.to_numpy()] = zlow_c.to_numpy()
    zmax = np.full(shape, -np.inf); np.maximum.at(zmax, (ij[:, 0], ij[:, 1]), Rv[:, 2])
    occ = np.isfinite(zlow)
    foot = binary_dilation(occ, structure=np.ones((3, 3), bool), iterations=a.dilate)
    k = 2 * max(1, int(round(0.5 / a.cell))) + 1                      # symmetric +-0.5 m window
    ground = minimum_filter(zlow, size=k, mode='constant', cval=np.inf)
    top = maximum_filter(zmax, size=k, mode='constant', cval=-np.inf)
    g_fallback = float(np.percentile(Rv[:, 2], 2))
    ground[~np.isfinite(ground)] = g_fallback; top[~np.isfinite(top)] = g_fallback
    # ground surface: cells at the local low level (not furniture tops); height = median of the cell's low cluster
    zl_pt = zlow.flat[cid]
    gcell_pt = zl_pt <= ground.flat[cid] + 0.10
    low = gcell_pt & (Rv[:, 2] <= zl_pt + 0.06)
    gs_c = pd.Series(Rv[low, 2]).groupby(cid[low]).median()
    gsurf = np.full(shape, np.nan); gsurf.flat[gs_c.index.to_numpy()] = gs_c.to_numpy()

    mesh = o3d.io.read_triangle_mesh(a.mesh)
    ns = int(min(4e6, max(1e5, mesh.get_surface_area() / 1e-4)))
    ms = np.asarray(mesh.sample_points_uniformly(ns).points)

    def cells(P):
        c = np.floor((P[:, :2] - lo) / a.cell).astype(int)
        inside = np.all((c >= 0) & (c < np.array(shape)), axis=1)
        c = np.clip(c, 0, np.array(shape) - 1)
        return c, inside

    cm, inm = cells(ms)
    g_m, t_m = ground[cm[:, 0], cm[:, 1]], top[cm[:, 0], cm[:, 1]]
    in_foot = inm & foot[cm[:, 0], cm[:, 1]] & (ms[:, 2] >= g_m - 1.0) & (ms[:, 2] <= t_m + 0.3)
    in_band = in_foot & (ms[:, 2] <= g_m + a.band)
    cr, _ = cells(Rv)
    g_r = ground[cr[:, 0], cr[:, 1]]
    r_band = Rv[:, 2] <= g_r + a.band
    gs_r = gsurf.flat[cid]
    r_ground = gcell_pt & np.isfinite(gs_r) & (np.abs(Rv[:, 2] - np.nan_to_num(gs_r, nan=1e9)) < 0.03)

    tree_R, tree_M = cKDTree(Rv), cKDTree(ms)
    acc_foot = tree_R.query(ms[in_foot], k=1)[0]; acc_band = tree_R.query(ms[in_band], k=1)[0]
    comp_all = tree_M.query(Rv, k=1)[0]
    sigma_gt = float(rep.get('sigma_gt_cm', 0)) / 100

    out = dict(
        mesh=a.mesh, crop=dict(cell_m=a.cell, dilate_cells=a.dilate, band_m=a.band, vis_range_m=a.vis_range),
        sim3=bool(a.sim3), scale_fit_scan_over_mps=None if scale_fit is None else round(scale_fit, 4),
        mesh_samples=int(len(ms)), samples_in_footprint=int(in_foot.sum()), samples_in_band=int(in_band.sum()),
        reference_points=int(len(Rv)), reference_in_band=int(r_band.sum()), reference_ground=int(r_ground.sum()),
        footprint_area_m2=round(float(foot.sum()) * a.cell ** 2, 1),
        footprint=dict(accuracy_median_cm=round(100 * float(np.median(acc_foot)), 2),
                       completeness_median_cm=round(100 * float(np.median(comp_all)), 2),
                       scores=scores(acc_foot, comp_all, sigma_gt)),
        contact_band=dict(accuracy_median_cm=round(100 * float(np.median(acc_band)), 2),
                          completeness_median_cm=round(100 * float(np.median(comp_all[r_band])), 2),
                          scores=scores(acc_band, comp_all[r_band], sigma_gt)),
        ground_covered_within_5cm=round(float((comp_all[r_ground] < 0.05).mean()), 3),
        ground_covered_within_10cm=round(float((comp_all[r_ground] < 0.10).mean()), 3),
        sigma_gt_cm=rep.get('sigma_gt_cm'))
    name = os.path.splitext(os.path.basename(a.mesh))[0] + ('_sim3' if a.sim3 else '')
    json.dump(out, open(os.path.join(a.out, f'cropped_{name}.json'), 'w'), indent=2)

    try:   # top-down check picture: footprint mesh samples coloured by distance to the scan
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        S = ms[in_foot][::10]; d = acc_foot[::10]
        col = np.where(d < 0.05, 'tab:green', np.where(d < 0.10, 'gold', 'tab:red'))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(Rv[::40, 0], Rv[::40, 1], s=0.2, c='lightgray', label='iPad scan')
        ax.scatter(S[:, 0], S[:, 1], s=0.3, c=col)
        ax.plot(tr[:, 0], tr[:, 1], 'k-', lw=0.6, label='glasses path')
        ax.set_aspect('equal'); ax.set_title(f'{name}: mesh in scan footprint (green <5 cm, yellow <10, red >=10)')
        lo2, hi2 = Rv.min(0) - 1, Rv.max(0) + 1; ax.set_xlim(lo2[0], hi2[0]); ax.set_ylim(lo2[1], hi2[1])
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f'cropped_{name}.png'), dpi=110)
    except Exception as e:
        out['png_error'] = str(e)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
