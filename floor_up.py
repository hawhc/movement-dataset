# -*- coding: utf-8 -*-
"""全場共用地面法向（垂直軸）—— 按標定分組，用同組所有受測者嘅腳踝擬一個地面。

點解要：ingest 原本逐位受測者自己擬地面（_up_from_floor）。移動類冇問題（行過一大片地），
但原地類細路只踩住一細塊地，平面法向好唔穩；再加深蹲／踮腳，腳踝點更亂。
2026-09-24 實測：同一條片兩個細路嘅垂直軸差 15°；原地深蹲、下肢伸展、K3 持球深蹲跳
逐位 vs 全片夾角中位 12–15°、p90 23–30°，移動類全部 0°。垂直軸歪 15°，企直都讀成
「彎腰 35–60°」（下肢伸展 rec_20260918_153319#0 手腕離地讀到 168% 腿長）。

同一份 calibration.json（md5 相同）＝ 同一個世界座標，所以可以將成場所有細路嘅腳踝
一齊擬 —— 腳分佈得開，法向好穩。結果寫入每條片嘅 floor_up.json，ingest.measure
見到就用，唔再逐位擬。

用法：
    python floor_up.py --movement k1_squat
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LSH, RSH, LHIP, RHIP, LAN, RAN = 5, 6, 11, 12, 15, 16


def _ransac_plane(A, tol, iters=600, seed=0):
    rng = np.random.default_rng(seed)
    best, bn, bd = 0, None, None
    for _ in range(iters):
        s = A[rng.choice(len(A), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = n @ s[0]
        k = int((np.abs(A @ n - d) < tol).sum())
        if k > best:
            best, bn, bd = k, n, d
    inl = A[np.abs(A @ bn - bd) < tol]
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c)
    return vt[-1], inl


def group_up(dirs):
    """同一份標定嘅所有片 → (up, 點數, 內點比例)。座標係 raw 世界（未乘 scale）。"""
    ank, hips = [], []
    for d in dirs:
        z = np.load(os.path.join(d, 'pose3d.npz'))
        try:
            m = json.load(open(os.path.join(d, 'metrics.json')))
        except Exception:
            continue
        names = [str(x) for x in z['subjects']]
        for s in m['subjects']:
            if s.get('suspect') or s['subject_ref'] not in names:
                continue
            i = names.index(s['subject_ref'])
            k = s.get('scale_correction') or 1.0
            X = z['xyz'][i][z['valid'][i]].astype(float) / k
            for j in (LAN, RAN):
                p = X[:, j]
                ank.append(p[np.isfinite(p).all(1)])
            hh = X[:, [LHIP, RHIP]].mean(1)
            hips.append(hh[np.isfinite(hh).all(1)])
    if not ank:
        return None
    A, H = np.vstack(ank), np.vstack(hips)
    if len(A) < 50:
        return None
    # 容差：地面上嘅腳踝（企／行）大約同一高度；踮腳同抬腳係離群，由 RANSAC 剔走
    n, inl = _ransac_plane(A, tol=0.03)
    if np.median(H @ n) < np.median(inl @ n):
        n = -n
    return n, len(A), len(inl) / len(A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--movement', required=True)
    ap.add_argument('--data', default=os.path.join(HERE, 'data'))
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(a.data, a.movement, 'rec_*'))
                  if os.path.exists(os.path.join(d, 'pose3d.npz'))
                  and os.path.exists(os.path.join(d, 'calibration.json')))
    groups = {}
    for d in dirs:
        h = hashlib.md5(open(os.path.join(d, 'calibration.json'), 'rb').read()).hexdigest()[:8]
        groups.setdefault(h, []).append(d)
    for h, ds in groups.items():
        r = group_up(ds)
        if r is None:
            print(f'  ✗ 標定 {h}: 腳踝點唔夠（{len(ds)} 條片）')
            continue
        n, npts, frac = r
        for d in ds:
            json.dump({'up': [float(x) for x in n], 'calib_md5': h, 'n_clips': len(ds),
                       'n_ankle_points': int(npts), 'inlier_frac': round(frac, 3)},
                      open(os.path.join(d, 'floor_up.json'), 'w'), indent=1)
        print(f'  標定 {h}: {len(ds)} 條片、{npts} 個腳踝點、內點 {frac*100:.0f}% → up {np.round(n, 3).tolist()}')


if __name__ == '__main__':
    main()
