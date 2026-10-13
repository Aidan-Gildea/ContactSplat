#!/usr/bin/env python
"""Align an iPad LiDAR scan (RTAB-Map point cloud) to the Aria MPS world frame, report how well the
device-only frame holds up (claim C1), and optionally grade a collision mesh against the scan.

    python eval_reference.py --ipad room.ply --slam <mps_..._vrs/slam> --out <dir> [--mesh mesh.ply]

Steps (Handbook lesson 6.4, "From scan to score, in order"):
  1. Find up from the data: fit the scan's floor plane, rotate it to +z, put the floor at z = 0.
  2. Rough fit: sweep the turn about vertical in 1 degree steps; for each, cross-correlate a top-down
     map of the scan's walls and furniture with the same map of the MPS semi-dense points. The best
     peak gives the turn and the 2D shift. The best few candidates go on to step 3.
  3. Fine fit against the MPS semi-dense points (never the mesh being graded): point-to-plane ICP that
     slides the MPS points onto the scan's surfaces, robust to outliers. Then one free-scale fit.
  4. Report: tilt between the true floor and MPS "up", scale factor, alignment residual, the scan's own
     plane-fit noise, and the smallest threshold worth reporting (tau >= 2 sigma_GT).
  5. With --mesh: accuracy, completeness and F-score of the mesh against the scan, inside the region
     the glasses could see.
"""
import argparse, gzip, json, os
import numpy as np
import open3d as o3d
from scipy.signal import fftconvolve
from scipy.ndimage import binary_dilation

INV_DIST_STD, DIST_STD = 0.005, 0.01      # same confidence filter as mps-mesh/mps_points_to_colmap.py
GRID = 0.05                               # 5 cm cells for the top-down maps
BAND = (0.3, 2.0)                         # heights above the floor that count as walls and furniture


def rot_a_to_b(a, b):
    """Smallest rotation taking unit vector a onto unit vector b (Rodrigues)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def T(R=np.eye(3), t=np.zeros(3)):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M


def plane_rms(pc, n_guess=None, thresh=0.05, horizontal=None):
    """Largest plane (optionally horizontal/vertical) -> unit normal, offset, RMS of points within thresh."""
    rest = pc
    for _ in range(8):
        model, inl = rest.segment_plane(0.02, 3, 3000)
        n = np.array(model[:3]); n /= np.linalg.norm(n)
        ok = horizontal is None or (horizontal and abs(n[2]) > 0.9) or (not horizontal and abs(n[2]) < 0.15)
        if ok:
            P = np.asarray(rest.points)
            d = P @ n + model[3]
            sel = P[np.abs(d) < thresh]
            c = sel.mean(0); u, s, vt = np.linalg.svd(sel - c, full_matrices=False)
            n2 = vt[2] * np.sign(vt[2] @ n)
            r = (sel - c) @ n2
            return n2, c, float(np.sqrt(np.mean(r ** 2))), len(sel)
        rest = rest.select_by_index(inl, invert=True)
    raise RuntimeError('no plane of the requested kind found')


def top_map(xy, lo, shape):
    ij = np.floor((xy - lo) / GRID).astype(int)
    keep = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    M = np.zeros(shape, bool); M[ij[keep, 0], ij[keep, 1]] = True
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ipad', required=True); ap.add_argument('--slam', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--mesh')
    ap.add_argument('--eye', type=float, default=None, help='eye height (m); only a fallback for the MPS floor guess')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    o3d.utility.random.seed(0); np.random.seed(0)
    rep = {}

    # ---------- 1. find up from the scan's floor ----------
    raw = o3d.io.read_point_cloud(a.ipad)
    pc = raw.voxel_down_sample(0.01)
    coarse = pc.voxel_down_sample(0.03)
    rest, horiz = coarse, []
    for _ in range(6):                                   # horizontal planes; the floor is the lowest big one
        model, inl = rest.segment_plane(0.02, 3, 3000)
        n = np.array(model[:3]); n /= np.linalg.norm(n)
        if abs(n[2]) > 0.9:
            n = n * np.sign(n[2]); P = np.asarray(rest.select_by_index(inl).points)
            horiz.append((float((P @ n).mean()), n, len(inl)))
        rest = rest.select_by_index(inl, invert=True)
    big = [h for h in horiz if h[2] > 0.25 * max(x[2] for x in horiz)]
    floor_h, n_floor, _ = min(big, key=lambda h: h[0])
    R_level = rot_a_to_b(n_floor, np.array([0, 0, 1.0]))
    pc.rotate(R_level, center=(0, 0, 0))
    z_floor = np.percentile(np.asarray(pc.points)[:, 2], 1)
    Pz = np.asarray(pc.points)[:, 2]
    hist, edges = np.histogram(Pz[Pz < z_floor + 0.3], bins=60)
    z_floor = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2
    pc.translate((0, 0, -z_floor))
    T_level = T(t=np.array([0, 0, -z_floor])) @ T(R_level)
    rep['scan_points'] = len(raw.points)
    rep['scan_up_error_deg'] = float(np.degrees(np.arccos(min(1.0, n_floor[2]))))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    n_f, c_f, rms_floor, nf = plane_rms(pc, horizontal=True)
    n_w, c_w, rms_wall, nw = plane_rms(pc, horizontal=False)
    rep['scan_floor_plane_rms_cm'] = round(100 * rms_floor, 2)
    rep['scan_wall_plane_rms_cm'] = round(100 * rms_wall, 2)
    S = np.asarray(pc.points)

    # ---------- MPS semi-dense points and trajectory ----------
    import pandas as pd
    pts = pd.read_csv(os.path.join(a.slam, 'semidense_points.csv.gz'))
    pts = pts[(pts.inv_dist_std < INV_DIST_STD) & (pts.dist_std < DIST_STD)]
    M = pts[['px_world', 'py_world', 'pz_world']].to_numpy()
    tr = pd.read_csv(os.path.join(a.slam, 'closed_loop_trajectory.csv'), usecols=['tx_world_device', 'ty_world_device', 'tz_world_device'])
    TR = tr.to_numpy()[::100]
    rep['mps_points_confident'] = len(M)
    # MPS floor: densest 2 cm height bin between 2.2 m and 1.2 m below the median device height
    zt = np.median(TR[:, 2])
    zz = M[(M[:, 2] > zt - 2.2) & (M[:, 2] < zt - 1.2), 2]
    if len(zz) > 200:
        h, e = np.histogram(zz, bins=np.arange(zt - 2.2, zt - 1.2, 0.02)); mps_floor = e[np.argmax(h)] + 0.01
    else:
        mps_floor = zt - (a.eye or 1.6683)
    rep['mps_floor_guess_z'] = round(float(mps_floor), 3)

    # ---------- 2. rough fit: yaw sweep with 2D cross-correlation ----------
    Sb = S[(S[:, 2] > BAND[0]) & (S[:, 2] < BAND[1])]
    Mb = M[(M[:, 2] > mps_floor + BAND[0]) & (M[:, 2] < mps_floor + BAND[1])]
    # pad the MPS map by the scan's diagonal so any turned scan map fits inside it ('valid' correlation);
    # a partial outdoor scan can reach beyond the area the glasses' points cover
    pad = 1.0 + float(np.linalg.norm(np.ptp(Sb[:, :2], axis=0))) + 2 * GRID
    lo = Mb[:, :2].min(0) - pad; shape = tuple((np.ceil((Mb[:, :2].max(0) + pad - lo) / GRID)).astype(int))
    Mmap = binary_dilation(top_map(Mb[:, :2], lo, shape), iterations=1).astype(np.float32)
    cands = []
    sc = Sb[:, :2] - Sb[:, :2].mean(0)
    for deg in range(0, 360, 1):
        th = np.radians(deg); Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        q = sc @ Rz.T; qlo = q.min(0); kshape = tuple((np.ceil((q.max(0) - qlo) / GRID) + 1).astype(int))
        K = top_map(q, qlo, kshape).astype(np.float32)
        C = fftconvolve(Mmap, K[::-1, ::-1], mode='valid')
        i, j = np.unravel_index(np.argmax(C), C.shape)
        score = C[i, j] / K.sum()
        shift = lo + np.array([i, j]) * GRID - qlo           # where the kernel's corner lands
        cands.append((score, deg, shift, Sb[:, :2].mean(0)))
    cands.sort(key=lambda c: -c[0])
    rep['rough_fit_best_scores'] = [[round(float(c[0]), 3), c[1]] for c in cands[:5]]

    # ---------- 3. fine fit: point-to-plane ICP of MPS points onto the scan ----------
    tgt = pc.voxel_down_sample(0.02); tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    best = None
    seen = set()
    for score, deg, shift, cen in cands:
        if len(seen) >= 4: break
        key = (round(deg / 10), tuple(np.round(shift, 0)))
        if key in seen: continue
        seen.add(key)
        th = np.radians(deg); Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        # scan -> MPS:  x' = Rz (x - [cen,0]) + [shift,0] ; z: floor 0 -> mps_floor
        t0 = np.r_[shift, mps_floor] - Rz @ np.r_[cen, 0]
        T_s2m = T(Rz, t0)
        T_m2s0 = np.linalg.inv(T_s2m)
        lo_s, hi_s = S.min(0) - 0.3, S.max(0) + 0.3
        Ms = (T_m2s0[:3, :3] @ M.T).T + T_m2s0[:3, 3]
        keep = np.all((Ms >= lo_s) & (Ms <= hi_s), axis=1)
        if keep.sum() < 200: continue
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(M[keep]))
        Tcur = T_m2s0
        for dmax in (0.30, 0.15, 0.08, 0.05):
            loss = o3d.pipelines.registration.TukeyLoss(k=dmax)
            res = o3d.pipelines.registration.registration_icp(src, tgt, dmax, Tcur,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            Tcur = res.transformation
        ev = o3d.pipelines.registration.evaluate_registration(src, tgt, 0.05, Tcur)
        cand = dict(deg=deg, rough=float(score), fitness=ev.fitness, rmse=ev.inlier_rmse, T_m2s=Tcur, n=int(keep.sum()), keep=keep)
        if best is None or cand['fitness'] > best['fitness']:
            best = cand
    if best is None: raise RuntimeError('rough fit found no overlap between the scan and the MPS points')
    T_m2s = best['T_m2s']; T_s2m = np.linalg.inv(T_m2s)
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(M[best['keep']]))
    # residuals: signed point-to-plane distance of each MPS point to the scan
    srcT = o3d.geometry.PointCloud(src); srcT.transform(T_m2s)
    kd = o3d.geometry.KDTreeFlann(tgt); TP, TN = np.asarray(tgt.points), np.asarray(tgt.normals)
    d = []
    for p in np.asarray(srcT.points):
        k, idx, _ = kd.search_knn_vector_3d(p, 1); q = TP[idx[0]]; d.append(float((p - q) @ TN[idx[0]]))
    d = np.abs(np.array(d)); inl = d < 0.05
    rep['icp_candidate_turn_deg'] = best['deg']
    rep['icp_mps_points_used'] = best['n']
    rep['icp_inlier_fraction_5cm'] = round(float(inl.mean()), 3)
    rep['icp_inlier_rms_cm'] = round(100 * float(np.sqrt(np.mean(d[inl] ** 2))), 2)
    rep['icp_inlier_median_cm'] = round(100 * float(np.median(d[inl])), 2)
    # tilt: fit the floor plane of the aligned scan, and of the MPS points' own floor, both in the MPS frame
    S_m = (T_s2m[:3, :3] @ S.T).T + T_s2m[:3, 3]
    zf = np.percentile(S_m[:, 2], 2)
    def floor_normal(P):
        # RANSAC for the inlier set, then a least-squares (SVD) refit on points within 3 cm: stable to < 0.05 deg
        q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); m, inl = q.segment_plane(0.02, 3, 5000)
        n0 = np.array(m[:3]) / np.linalg.norm(m[:3]); sel = P[np.abs(P @ n0 + m[3] / np.linalg.norm(m[:3])) < 0.03]
        c = sel.mean(0); n = np.linalg.svd(sel - c, full_matrices=False)[2][2]
        return n * np.sign(n[2]), len(sel)
    n_sf, k_sf = floor_normal(S_m[np.abs(S_m[:, 2] - zf) < 0.06])
    Mf = M[np.abs(M[:, 2] - zf) < 0.06]
    rep['tilt_scan_floor_vs_mps_up_deg'] = round(float(np.degrees(np.arccos(min(1.0, abs(n_sf[2]))))), 3)
    if len(Mf) > 500:
        n_mf, k_mf = floor_normal(Mf)
        rep['tilt_mps_own_floor_vs_mps_up_deg'] = round(float(np.degrees(np.arccos(min(1.0, abs(n_mf[2]))))), 3)
        rep['angle_scan_floor_vs_mps_floor_deg'] = round(float(np.degrees(np.arccos(min(1.0, abs(n_sf @ n_mf))))), 3)
        rep['mps_floor_points'] = int(len(Mf))
    # uniqueness: turn the aligned scan by quarter turns about vertical, re-run ICP, compare fitness
    sc2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(S_m)).voxel_down_sample(0.02)
    sc2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    mp2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(M)).voxel_down_sample(0.02)
    cc = S_m.mean(0); fits = {}
    for deg in (0, 90, 180, 270):
        th = np.radians(deg); Rq = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        s2 = o3d.geometry.PointCloud(sc2); s2.transform(T(Rq, cc - Rq @ cc)); cur = np.eye(4)
        for dmax in (0.30, 0.15, 0.08, 0.05):
            cur = o3d.pipelines.registration.registration_icp(mp2, s2, dmax, cur, o3d.pipelines.registration.TransformationEstimationPointToPlane(o3d.pipelines.registration.TukeyLoss(k=dmax))).transformation
        fits[f'+{deg}deg'] = round(o3d.pipelines.registration.evaluate_registration(mp2, s2, 0.05, cur).fitness, 3)
    rep['uniqueness_fitness_by_quarter_turn'] = fits
    # free scale: point-to-point with scaling on close pairs only
    res_s = o3d.pipelines.registration.registration_icp(src, tgt, 0.05, T_m2s,
              o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    Rs = res_s.transformation[:3, :3]; scale_m2s = float(np.cbrt(np.linalg.det(Rs)))
    rep['scale_scan_over_mps'] = round(scale_m2s, 4)
    # error budget
    sigma_scan = max(rms_floor, rms_wall); sigma_align = float(np.sqrt(np.mean(d[inl] ** 2)))
    sigma_gt = float(np.hypot(sigma_scan, sigma_align))
    rep['sigma_scan_cm (plane noise only, single scan)'] = round(100 * sigma_scan, 2)
    rep['sigma_align_cm (ICP inlier RMS, includes MPS point noise)'] = round(100 * sigma_align, 2)
    rep['sigma_gt_cm'] = round(100 * sigma_gt, 2)
    rep['smallest_threshold_to_report_cm'] = round(200 * sigma_gt, 1)

    # save the scan in the MPS frame (1 cm) and the full transform raw-scan -> MPS
    T_total = T_s2m @ T_level
    rep['T_scan_raw_to_mps'] = np.round(T_total, 6).tolist()
    out_pc = o3d.geometry.PointCloud(pc); out_pc.transform(T_s2m)
    o3d.io.write_point_cloud(os.path.join(a.out, 'scan_in_mps_frame.ply'), out_pc)

    # overlay picture for a visual check
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        A = np.asarray(out_pc.points)[::20]; Mk = M[best['keep']]
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))
        ax[0].scatter(A[:, 0], A[:, 1], s=0.2, c='tab:blue', alpha=0.3, label='iPad scan (aligned)')
        ax[0].scatter(Mk[:, 0], Mk[:, 1], s=0.6, c='tab:red', alpha=0.6, label='MPS points')
        ax[0].plot(TR[:, 0], TR[:, 1], 'k-', lw=0.6, label='glasses path'); ax[0].set_aspect('equal'); ax[0].legend(markerscale=10, loc='upper right'); ax[0].set_title('Top view')
        ax[1].scatter(A[:, 0], A[:, 2], s=0.2, c='tab:blue', alpha=0.3); ax[1].scatter(Mk[:, 0], Mk[:, 2], s=0.6, c='tab:red', alpha=0.6)
        ax[1].set_aspect('equal'); ax[1].set_title('Side view (x, z)')
        lo2, hi2 = A.min(0) - 0.5, A.max(0) + 0.5
        ax[0].set_xlim(lo2[0], hi2[0]); ax[0].set_ylim(lo2[1], hi2[1]); ax[1].set_xlim(lo2[0], hi2[0]); ax[1].set_ylim(lo2[2], hi2[2])
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'alignment_overlay.png'), dpi=110)
    except Exception as e:
        rep['overlay_error'] = str(e)

    # ---------- 5. optional: grade a mesh against the aligned scan ----------
    if a.mesh:
        mesh = o3d.io.read_triangle_mesh(a.mesh)
        area = mesh.get_surface_area(); ns = int(min(4e6, max(1e5, area / 1e-4)))
        ms = np.asarray(mesh.sample_points_uniformly(ns).points)
        R = np.asarray(out_pc.points)
        lo3, hi3 = R.min(0) + [0.05, 0.05, -0.1], R.max(0) - [0.05, 0.05, 0.15]     # inside the scanned room, drop the ceiling
        ms = ms[np.all((ms >= lo3) & (ms <= hi3), axis=1)]
        # visible region: reference points within 4 m of the glasses' path
        from scipy.spatial import cKDTree
        near = cKDTree(TR).query(R, k=1)[0] < 4.0
        Rv = R[near & np.all((R >= lo3) & (R <= hi3), axis=1)]
        acc = cKDTree(Rv).query(ms, k=1)[0]; comp = cKDTree(ms).query(Rv, k=1)[0]
        scores = {}
        for tau in (0.02, 0.05, 0.10, 0.20):
            P_ = float((acc < tau).mean()); R_ = float((comp < tau).mean()); F_ = 2 * P_ * R_ / (P_ + R_) if P_ + R_ else 0.0
            scores[f'{int(tau * 100)}cm'] = dict(precision=round(P_, 3), recall=round(R_, 3), F=round(F_, 3), trustworthy=bool(tau >= 2 * sigma_gt))
        floor_ref = Rv[np.abs(Rv[:, 2] - np.percentile(Rv[:, 2], 2)) < 0.03]       # the scan's floor, now in the MPS frame
        fc = cKDTree(ms).query(floor_ref, k=1)[0]
        rep['mesh'] = dict(path=a.mesh, samples=int(len(ms)), reference_points=int(len(Rv)),
                           accuracy_median_cm=round(100 * float(np.median(acc)), 2), completeness_median_cm=round(100 * float(np.median(comp)), 2),
                           scores=scores, floor_reference_points=int(len(floor_ref)),
                           floor_covered_within_5cm=round(float((fc < 0.05).mean()), 3), floor_covered_within_10cm=round(float((fc < 0.10).mean()), 3))

    json.dump(rep, open(os.path.join(a.out, 'report.json'), 'w'), indent=2)
    for k, v in rep.items():
        if k != 'T_scan_raw_to_mps': print(f'{k}: {v}')


if __name__ == '__main__':
    main()
