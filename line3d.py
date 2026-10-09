# -*- coding: utf-8 -*-
"""平衡線（地上白色膠紙）3D —— 畀「平衡線控制」(d_line) 用。

點解唔用物件偵測模型：條線係**靜止**、**貼地**、**高對比**（深灰地墊上嘅白膠紙），
逐台機攞中位幀（行過嘅細路自然消失）再做 top-hat 已經分得好乾淨
（2026-09-23 K1 平衡線前行 rec_20260916_142441 四台都清楚）。

做法（唔使跨機位配對點，所以冇鬼影問題）：
  1. 每台機：抽 15 幀取中位 → top-hat（亮過附近地面）+ 低飽和 = 白膠紙候選像素
  2. 每粒像素去畸變後射線投去**地面平面**（由受測者腳踝 RANSAC 擬合）
  3. 地面上按 3cm 格計「有幾多台機投到呢格」—— 真係喺地上嘅嘢四台都會疊埋；
     牆上、枱上嘅白色投落地面會亂飛，唔會疊
  4. 只留受測者腳步路徑 0.5m 之內、≥3 台機確認嘅格 → RANSAC 擬直線

地面高度：腳踝關節離地約 7cm，腳踝平面唔係地面。用錯高度，斜望嘅機投落地面
就會錯位（機高 ~2.5m、距離 ~5m → 7cm 高度差 ≈ 14cm 橫向偏移），四台就疊唔埋。
所以**掃高度**（腳踝平面以下 0–15cm），揀四台最疊得埋嗰個 —— 自己驗證自己。

輸出 <out_dir>/line3d.npz：p0, p1（線兩端，世界座標、未修正尺度）、n（地面法向）、
support（確認格數）、offset_m、cams_agree。ingest.py 用 _load_line 讀。

用法：
    python line3d.py --data data/k1_balance_line_walk --src <錄影資料夾>
"""
import argparse, glob, json, os
import numpy as np

CAMS = ['cam01_dev0', 'cam02_dev1', 'cam03_dev2', 'cam04_dev3']
CKB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                   'human-selfcalib', 'calibs', 'tapo_c120_checkerboard.json')
LAN, RAN = 15, 16


def tape_pixels(mp4, n=15, tophat=40, sat=60, stride=2):
    """中位幀嘅白膠紙候選像素 (N,2)。"""
    import cv2
    v = cv2.VideoCapture(mp4)
    N = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    fr = []
    for f in np.linspace(0, max(N - 1, 0), n).astype(int):
        v.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, x = v.read()
        if ok:
            fr.append(x)
    v.release()
    if not fr:
        return np.zeros((0, 2)), None
    med = np.median(np.stack(fr), 0).astype(np.uint8)
    g = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    m = (th > tophat) & (hsv[..., 1] < sat)
    m[:40] = False                                   # 時間戳
    ys, xs = np.nonzero(m[::stride, ::stride])
    return np.stack([xs * stride, ys * stride], 1).astype(np.float64), med


def _floor(xyz_raw, valid):
    """腳踝點 RANSAC 擬合平面 → (n, d)，n 指向上（n·X = d）。"""
    A = []
    for s in range(len(xyz_raw)):
        for j in (LAN, RAN):
            p = xyz_raw[s][valid[s], j]
            A.append(p[np.isfinite(p).all(1)])
    A = np.vstack(A)
    if len(A) < 30:
        return None
    rng = np.random.default_rng(0)
    best, bn, bd = 0, None, None
    for _ in range(400):
        s = A[rng.choice(len(A), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = n @ s[0]
        k = (np.abs(A @ n - d) < 0.04).sum()
        if k > best:
            best, bn, bd = k, n, d
    inl = A[np.abs(A @ bn - bd) < 0.04]
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c)
    n = vt[-1]
    # 法向指向上：肩喺腳踝上面
    return n, float(n @ c), inl


def _rays_to_plane(uv, K, R, T, n, d):
    """去畸變像素 → 世界座標射線 → 同平面 n·X=d 交點。"""
    Ki = np.linalg.inv(K)
    x = np.hstack([uv, np.ones((len(uv), 1))]) @ Ki.T       # 相機座標方向
    dirs = x @ R                                             # R^T x（行向量）
    C = -R.T @ T.reshape(3)
    den = dirs @ n
    lam = (d - n @ C) / np.where(np.abs(den) < 1e-9, np.nan, den)
    X = C + dirs * lam[:, None]
    return X[lam > 0]


def build(out_dir, rec_dir, verbose=True):
    import cv2
    cal = json.load(open(os.path.join(out_dir, 'calibration.json')))
    if not all(str(v) in cal for v in range(len(CAMS))):
        # 三機重建（例如 K1 平衡線 s28：cam01 郁咗）—— 條線抄同場四機嗰條
        print('  - %s: 標定只有 %d 機，跳過（用同場四機嘅線）' % (os.path.basename(out_dir), len(cal)))
        return None
    ckb = json.load(open(CKB))
    K_ckb = np.array(ckb['0']['K'], np.float64).reshape(3, 3)
    dist = np.array(ckb['_distCoeffs'], np.float64)
    z = np.load(os.path.join(out_dir, 'pose3d.npz'))
    met = json.load(open(os.path.join(out_dir, 'metrics.json')))
    subs = met['subjects']
    # ⚠ metrics.json 嘅 subjects 唔一定同 pose3d.npz 一一對應（幀數太少嘅軌跡
    # metrics 會略過）—— 2026-09-24 查到 237 條入面 68 條錯位。一定要用 subject_ref
    # 對返 pose3d 嘅 'subjects'，唔可以用位置。
    names = [str(x) for x in z['subjects']]
    idx = {n: i for i, n in enumerate(names)}
    keep, ks = [], []
    for s in subs:
        if s.get('suspect') or s['subject_ref'] not in idx:
            continue
        keep.append(idx[s['subject_ref']])
        ks.append(s.get('scale_correction') or 1.0)
    if not keep:
        print('  ✗ %s: 冇非 suspect 受測者' % os.path.basename(out_dir))
        return None
    ks = np.array(ks)
    xyz_raw = (z['xyz'][keep] / ks[:, None, None, None]).astype(np.float64)
    valid = z['valid'][keep]
    xyz_raw[~valid] = np.nan
    fl = _floor(xyz_raw, valid)
    if fl is None:
        print('  ✗ %s: 腳踝點唔夠擬地面' % os.path.basename(out_dir))
        return None
    n, d0, inl = fl
    hip = np.nanmean(xyz_raw[:, :, [11, 12]], 2).reshape(-1, 3)
    hip = hip[np.isfinite(hip).all(1)]
    if np.median(hip @ n) < d0:                              # 翻轉，令 n 向上
        n, d0 = -n, -d0
    # 腳步路徑（投影到地面平面）—— 只喺呢條路徑附近搵條線
    feet = inl - np.outer(inl @ n - d0, n)

    per_cam = []
    for v, cam in enumerate(CAMS):
        mp4 = os.path.join(rec_dir, cam + '.mp4')
        px, _ = tape_pixels(mp4)
        if not len(px):
            per_cam.append(None)
            continue
        uv = cv2.undistortPoints(px.reshape(-1, 1, 2), K_ckb, dist,
                                 P=np.array(cal[str(v)]['K']).reshape(3, 3)).reshape(-1, 2)
        per_cam.append((uv, np.array(cal[str(v)]['K']).reshape(3, 3),
                        np.array(cal[str(v)]['R']).reshape(3, 3),
                        np.array(cal[str(v)]['T']).reshape(3)))

    # 平面內 2D 座標系
    e1 = np.cross(n, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(n, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    fp = np.stack([feet @ e1, feet @ e2], 1)
    from scipy.spatial import cKDTree
    tree = cKDTree(fp)
    # 腳踝平面嘅尺度未修正：腳踝離地 ~7cm 真實 → raw 單位要除 k
    k_med = float(np.median(ks))
    best = None
    for off_true in np.arange(0.0, 0.151, 0.01):
        off = off_true / k_med
        d = d0 - off
        cell = 0.03 / k_med
        occ = {}
        for v, pc in enumerate(per_cam):
            if pc is None:
                continue
            X = _rays_to_plane(pc[0], pc[1], pc[2], pc[3], n, d)
            q = np.stack([X @ e1, X @ e2], 1)
            near = tree.query(q, distance_upper_bound=0.5 / k_med)[0] < np.inf
            for c in {tuple(x) for x in np.floor(q[near] / cell).astype(int)}:
                occ.setdefault(c, set()).add(v)
        agree = np.array([c for c, s in occ.items() if len(s) >= 3], float)
        score = len(agree)
        if best is None or score > best[0]:
            best = (score, off_true, d, agree * cell + cell / 2)
    score, off_true, d, pts = best
    if score < 10:
        print('  ✗ %s: ≥3 台確認嘅格得 %d 個' % (os.path.basename(out_dir), score))
        return None
    # RANSAC 直線
    rng = np.random.default_rng(0)
    tol = 0.03 / k_med
    bi, bl = 0, None
    for _ in range(500):
        a, b = pts[rng.choice(len(pts), 2, replace=False)]
        u = b - a
        L = np.linalg.norm(u)
        if L < 0.2 / k_med:
            continue
        u /= L
        r = np.abs((pts - a) @ np.array([-u[1], u[0]]))
        k = (r < tol).sum()
        if k > bi:
            bi, bl = k, (a, u)
    a, u = bl
    inl2 = pts[np.abs((pts - a) @ np.array([-u[1], u[0]])) < tol]
    c = inl2.mean(0)
    _, _, vt = np.linalg.svd(inl2 - c)
    u = vt[0]
    t = (inl2 - c) @ u
    lo, hi = np.percentile(t, [1, 99])
    to3 = lambda p: p[0] * e1 + p[1] * e2 + d * n
    p0, p1 = to3(c + lo * u), to3(c + hi * u)
    length_true = float((hi - lo) * k_med)
    res = np.abs((inl2 - c) @ np.array([-u[1], u[0]])) * k_med
    out = dict(p0=p0, p1=p1, n=n, d=d, support=len(inl2), cells=score,
               offset_true_m=off_true, length_true_m=length_true,
               resid_true_cm=float(np.median(res) * 100))
    np.savez(os.path.join(out_dir, 'line3d.npz'), **out)
    if verbose:
        print('  %s: 線長 %.2fm、%d 格確認（≥3 台）、地面 = 腳踝平面下 %.0fcm、'
              '殘差中位 %.1fcm' % (os.path.basename(out_dir), length_true, len(inl2),
                                  off_true * 100, out['resid_true_cm']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='data/<movement>')
    ap.add_argument('--src', required=True, help='錄影資料夾（含 rec_*）')
    ap.add_argument('--recs', nargs='*', default=None)
    a = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(a.data, 'rec_*')))
    if a.recs:
        dirs = [x for x in dirs if os.path.basename(x) in a.recs]
    for od in dirs:
        rd = os.path.join(a.src, os.path.basename(od))
        if os.path.exists(os.path.join(od, 'calibration.json')) and os.path.isdir(rd):
            build(od, rd)


if __name__ == '__main__':
    main()
