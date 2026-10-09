# -*- coding: utf-8 -*-
"""紅色小欄杆 3D —— 畀 K1 繞過小欄杆「與欄杆距離」用。

equipment 模型冇 hurdle 類，但欄杆係**靜止**、**紅色**、**腳貼地**，同 line3d.py 一樣可以
逐台機攞中位幀（行過嘅細路消失）再投地面：

  1. 同一份標定嘅幾條片一齊取中位幀（欄杆冇郁過；細路／老師喺唔同位 → 消失）
  2. HSV 紅色遮罩 → 去畸變射線投去地面平面（法向用 floor_up.json，高度由腳踝推、掃 0–15cm）
  3. 地面上 4cm 格，≥3 台機確認先留 —— 牆上嘅紅色裝飾、公仔投落地面會亂飛
     （09-18 s11 cam02 標定唔準，靠其餘三台過關）
  4. 相連格分群；只留離細路路徑 0.8m 之內嘅 —— 兩邊嘅紅色雪糕筒喺邊界，離路徑遠
  5. 欄杆係拱形：落地嘅只有兩隻腳，所以每個欄杆嘅地面位置 = 兩隻腳之間嘅線段

輸出 <rec>/hurdles3d.npz：points（確認嘅紅色地面點，raw 世界座標）、centers（群中心）、
sizes、offset_true_m、n_clips。ingest._hurdle_weave 用 points 計腳離欄杆距離。

用法：
    python hurdle3d.py --data data/k1_hurdle_weave --src <staging 資料夾>
"""
import argparse, glob, hashlib, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMS = ['cam01_dev0', 'cam02_dev1', 'cam03_dev2', 'cam04_dev3']
CKB = os.path.join(HERE, '..', 'human-selfcalib', 'calibs', 'tapo_c120_checkerboard.json')
LAN, RAN, LHIP, RHIP = 15, 16, 11, 12


def red_pixels(mp4s, n_per=6, stride=2):
    import cv2
    fr = []
    for p in mp4s:
        v = cv2.VideoCapture(p)
        N = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in np.linspace(0, max(N - 1, 0), n_per).astype(int):
            v.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, x = v.read()
            if ok:
                fr.append(x)
        v.release()
    if not fr:
        return np.zeros((0, 2))
    med = np.median(np.stack(fr), 0).astype(np.uint8)
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    m = (((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 70))
    m[:40] = False
    ys, xs = np.nonzero(m[::stride, ::stride])
    return np.stack([xs * stride, ys * stride], 1).astype(np.float64)


def _rays_to_plane(uv, K, R, T, n, d):
    Ki = np.linalg.inv(K)
    x = np.hstack([uv, np.ones((len(uv), 1))]) @ Ki.T
    dirs = x @ R
    C = -R.T @ T.reshape(3)
    den = dirs @ n
    lam = (d - n @ C) / np.where(np.abs(den) < 1e-9, np.nan, den)
    X = C + dirs * lam[:, None]
    return X[lam > 0]


def _subjects_raw(d):
    z = np.load(os.path.join(d, 'pose3d.npz'))
    m = json.load(open(os.path.join(d, 'metrics.json')))
    names = [str(x) for x in z['subjects']]
    out = []
    for s in m['subjects']:
        if s.get('suspect') or s['subject_ref'] not in names:
            continue
        i = names.index(s['subject_ref'])
        k = s.get('scale_correction') or 1.0
        X = z['xyz'][i].astype(float) / k
        X[~z['valid'][i]] = np.nan
        out.append((X, k))
    return out


def build_group(dirs, src, verbose=True):
    import cv2
    from scipy.spatial import cKDTree
    from scipy import ndimage
    cal = json.load(open(os.path.join(dirs[0], 'calibration.json')))
    ckb = json.load(open(CKB))
    K_ckb = np.array(ckb['0']['K'], float).reshape(3, 3)
    dist = np.array(ckb['_distCoeffs'], float)
    up = np.array(json.load(open(os.path.join(dirs[0], 'floor_up.json')))['up'], float)
    subs = [x for d in dirs for x in _subjects_raw(d)]
    if not subs:
        return None
    ank = np.vstack([X[:, j] for X, _ in subs for j in (LAN, RAN)])
    ank = ank[np.isfinite(ank).all(1)]
    hip = np.vstack([np.nanmean(X[:, [LHIP, RHIP]], 1) for X, _ in subs])
    hip = hip[np.isfinite(hip).all(1)]
    k_med = float(np.median([k for _, k in subs]))
    d_ank = float(np.percentile(ank @ up, 20))           # 企地腳踝高度（raw）
    e1 = np.cross(up, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    hp = np.stack([hip @ e1, hip @ e2], 1)
    tree = cKDTree(hp)

    per_cam, tape_cam = [], []
    for v, cam in enumerate(CAMS):
        mp4s = [os.path.join(src, os.path.basename(d), cam + '.mp4') for d in dirs[:4]]
        ex = [p for p in mp4s if os.path.exists(p)]
        px = red_pixels(ex)
        if not len(px) or str(v) not in cal:
            per_cam.append(None)
            tape_cam.append(None)
            continue
        c = cal[str(v)]
        Kc = np.array(c['K']).reshape(3, 3)
        uv = cv2.undistortPoints(px.reshape(-1, 1, 2), K_ckb, dist, P=Kc).reshape(-1, 2)
        per_cam.append((uv, Kc, np.array(c['R']).reshape(3, 3), np.array(c['T']).reshape(3)))
        from line3d import tape_pixels
        tp, _ = tape_pixels(ex[0])
        tape_cam.append(cv2.undistortPoints(tp.reshape(-1, 1, 2), K_ckb, dist, P=Kc).reshape(-1, 2)
                        if len(tp) else None)

    cell = 0.04 / k_med
    best = None
    for off_true in np.arange(0.0, 0.151, 0.01):
        dpl = d_ank - off_true / k_med
        occ = {}
        for v, pc in enumerate(per_cam):
            if pc is None:
                continue
            X = _rays_to_plane(pc[0], pc[1], pc[2], pc[3], up, dpl)
            q = np.stack([X @ e1, X @ e2], 1)
            near = tree.query(q, distance_upper_bound=0.8 / k_med)[0] < np.inf
            for cc in {tuple(x) for x in np.floor(q[near] / cell).astype(int)}:
                occ.setdefault(cc, set()).add(v)
        agree = [cc for cc, s in occ.items() if len(s) >= 3]
        if best is None or len(agree) > best[0]:
            best = (len(agree), off_true, dpl, agree, occ)
    n_ok, off_true, dpl, agree, occ = best
    # （試過線走廊內放寬到 ≥2 台：會出鬼影、將一個欄杆斬成幾個 —— 唔用）
    if n_ok < 5:
        if verbose:
            print('  ✗ ≥3 台確認嘅紅色地面格得 %d 個' % n_ok)
        return None
    A = np.array(agree)
    lo = A.min(0)
    grid = np.zeros(tuple(A.max(0) - lo + 1), bool)
    grid[tuple((A - lo).T)] = True
    lab, nl = ndimage.label(grid, structure=np.ones((3, 3)))
    cen, size, pts = [], [], []
    for i in range(1, nl + 1):
        cc = np.argwhere(lab == i) + lo
        if len(cc) < 2:
            continue
        p2 = (cc + 0.5) * cell
        P3 = p2[:, :1] * e1 + p2[:, 1:] * e2 + dpl * up
        pts.append(P3)
        cen.append(P3.mean(0))
        size.append(len(cc))
    if not pts:
        return None
    # 剔紅色雪糕筒：欄杆全部排喺同一條線（白膠紙）上，筒喺起點／終點或者旁邊。
    # 中心點 RANSAC 擬直線，只留離線 < 0.35m（真實）嘅群。
    C = np.array(cen)
    c2 = np.stack([C @ e1, C @ e2], 1)
    rng = np.random.default_rng(0)
    tol = 0.35 / k_med
    bi, bl = 0, None
    for _ in range(300):
        a, b = c2[rng.choice(len(c2), 2, replace=False)]
        u = b - a
        if np.linalg.norm(u) < 1e-9:
            continue
        u /= np.linalg.norm(u)
        r = np.abs((c2 - a) @ np.array([-u[1], u[0]]))
        if (r < tol).sum() > bi:
            bi, bl = int((r < tol).sum()), (a, u)
    a, u = bl
    nrm2 = np.array([-u[1], u[0]])
    side = (c2 - a) @ nrm2
    keep = np.abs(side) < tol
    dropped = int((~keep).sum())
    C, c2, side = C[keep], c2[keep], side[keep]
    pts = [p for p, k_ in zip(pts, keep) if k_]
    size = [z_ for z_, k_ in zip(size, keep) if k_]
    # 分欄杆：欄杆橫跨條線，所以沿線方向（t）相近嘅群＝同一個欄杆（腳可能只偵測到一隻、
    # 或者兩隻腳同橫杆黏埋一群）。t 相隔 > 0.2m（真實）就開新一個。
    # 每個欄杆 = 垂直條線、橫跨該組所有點側向範圍嘅線段；只得一邊嘅用 ±半闊補（中位闊）。
    t = (c2 - a) @ u
    order = np.argsort(t)
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if t[i] - t[cur[-1]] > 0.2 / k_med:
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    P_all = [np.stack([p @ e1, p @ e2], 1) for p in pts]
    spans = []
    for g in groups:
        q = np.vstack([P_all[i] for i in g])
        sd = (q - a) @ nrm2
        tm = float(np.mean((q - a) @ u))
        tq = (q - a) @ u
        spans.append((tm, float(sd.min()), float(sd.max()), len(q),
                      float(np.ptp(tq)), float(np.ptp(sd))))
    if os.environ.get('HURDLE_DEBUG'):
        for sp in spans:
            print('    t=%.2f  side %.2f..%.2f  cells %d  沿線 %.2f 橫 %.2f' % (sp[0] * k_med, sp[1] * k_med, sp[2] * k_med, sp[3], sp[4] * k_med, sp[5] * k_med))
    # 分類（2026-09-25 三場實測嘅形狀）：
    #   欄杆（兩隻腳）：橫跨條線，闊 ≥ 0.4m（實測 0.62–0.68m）
    #   欄杆（得一隻腳）：喺條線一邊、中心離線 ≥ 0.12m（腳喺 ±0.2–0.4m）
    #   雪糕筒（放喺條線上做起點／終點）：窄（< 0.4m）而且中心貼住條線（< 0.12m）
    #   雜訊：< 4 格
    hur, n_cone, n_noise = [], 0, 0
    for sp in spans:
        tm, lo, hi, n_ = sp[:4]
        w, mid = (hi - lo) * k_med, (hi + lo) / 2 * k_med
        if n_ < 4:
            n_noise += 1
        elif lo < 0 < hi and w >= 0.4:
            hur.append(sp)
        elif abs(mid) < 0.12:
            n_cone += 1
        elif sp[4] >= 1.3 * sp[5]:
            # 單邊：欄杆腳係一條同行進方向平行嘅底座 → 沿線長、橫向窄
            # （實測沿線 0.16–0.24m、橫 0.04–0.12m）。雪糕筒底係圓，唔會咁長。
            hur.append(sp)
        else:
            n_cone += 1
    dropped += n_cone + n_noise
    spans = sorted(hur)
    # 白膠紙範圍：欄杆全部喺膠紙上面；起點／終點雪糕筒喺膠紙兩端之外。
    # （試過用間距分：欄杆由近至遠間距 1.14→0.92m 漸變，同雪糕筒分唔開。）
    occ_t = {}
    for v, tp in enumerate(tape_cam):
        if tp is None or per_cam[v] is None:
            continue
        _, Kc, Rc, Tc = per_cam[v]
        X = _rays_to_plane(tp, Kc, Rc, Tc, up, dpl)
        q = np.stack([X @ e1, X @ e2], 1)
        on = np.abs((q - a) @ nrm2) < 0.1 / k_med
        for cc in {tuple(x) for x in np.floor(q[on] / cell).astype(int)}:
            occ_t.setdefault(cc, set()).add(v)
    tcells = np.array([cc for cc, s_ in occ_t.items() if len(s_) >= 3], float)
    tape_ext = None
    if len(tcells) >= 10:
        tt = ((tcells + 0.5) * cell - a) @ u
        tape_ext = (float(np.percentile(tt, 2)), float(np.percentile(tt, 98)))
        pad = 0.25 / k_med
        before = len(spans)
        spans = [sp for sp in spans if tape_ext[0] - pad <= sp[0] <= tape_ext[1] + pad]
        dropped += before - len(spans)
    # （試過用頭尾間距剔雪糕筒：欄杆間距由近至遠 1.14→0.92m 漸變，分唔開 —— 改用底座形狀）
    w_full = [sp[2] - sp[1] for sp in spans if sp[1] < 0 < sp[2]]
    wmed = float(np.median(w_full)) if w_full else 0.0
    low_conf = not w_full                      # 冇一個見齊兩隻腳：闊度靠估
    half = wmed / 2 if wmed > 0 else 0.3 / k_med
    segs = []
    for tm, lo, hi, *_ in spans:
        if not (lo < 0 < hi and (hi - lo) * k_med >= 0.4):   # 得一隻腳：以條線為中心補返
            lo, hi = -half, half
        p0 = a + tm * u + lo * nrm2
        p1 = a + tm * u + hi * nrm2
        segs.append((p0[0] * e1 + p0[1] * e2 + dpl * up, p1[0] * e1 + p1[1] * e2 + dpl * up))
    seg_pts = np.vstack([np.linspace(p0, p1, 12) for p0, p1 in segs])
    widths = [float(np.linalg.norm(p1 - p0) * k_med) for p0, p1 in segs]
    out = dict(points=np.vstack(pts + [seg_pts]), centers=C, sizes=np.array(size),
               segments=np.array(segs) if segs else np.zeros((0, 2, 3)),
               up=up, plane_d=dpl, offset_true_m=off_true, n_clips=len(dirs),
               low_confidence=low_conf,
               tape_extent_m=np.array(tape_ext if tape_ext else (np.nan, np.nan)) * k_med,
               cells=n_ok)
    if verbose:
        print('  %d 條片：%d 個 ≥3 台確認格、%d 群（剔走 %d：線外雪糕筒＋線上雪糕筒＋雜訊）→ %d 個欄杆、'
              '闊 %s cm、地面＝腳踝下 %.0fcm'
              % (len(dirs), n_ok, len(cen) + dropped, dropped, len(segs),
                 [round(w * 100) for w in widths], off_true * 100)
              + ('  ⚠ 冇一個欄杆見齊兩隻腳，闊度用預設' if low_conf else ''))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--src', required=True)
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(a.data, 'rec_*'))
                  if all(os.path.exists(os.path.join(d, f)) for f in
                         ('pose3d.npz', 'calibration.json', 'floor_up.json', 'metrics.json')))
    groups = {}
    for d in dirs:
        h = hashlib.md5(open(os.path.join(d, 'calibration.json'), 'rb').read()).hexdigest()[:8]
        groups.setdefault(h, []).append(d)
    for h, ds in groups.items():
        print('標定 %s:' % h)
        r = build_group(ds, a.src)
        if r is None:
            continue
        for d in ds:
            np.savez(os.path.join(d, 'hurdles3d.npz'), **r)


if __name__ == '__main__':
    main()
