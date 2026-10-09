# -*- coding: utf-8 -*-
"""雪糕筒 3D：逐台偵測 → 跨機位配對三角化 → 場地幾何 + 踩筒判定

同 ball3d.py 分別：筒係**靜止**，所以
  * 唔使逐幀跑 —— 抽十幾幀取共識就夠，快好多；
  * 唔使追蹤 —— 一次解出位置就成條片通用；
  * 可以用「多幀一致」做額外守門：真筒每幀都喺同一點，假陽性唔會。

用法：
  python cone3d.py detect --movement k3_zigzag_move
  python cone3d.py build  --movement k3_zigzag_move
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CKB = '/Users/bcm01032/game7/human-selfcalib/calibs/tapo_c120_checkerboard.json'
MODEL = '/Users/bcm01032/game7/best_4cls_bal.pt'
CAMS = ['cam01_dev0', 'cam02_dev1', 'cam03_dev2', 'cam04_dev3']
CONE_CLS = [2, 3]          # cone / sport_cone
CONF = 0.25
N_FRAMES = 12              # 抽幾多幀取共識
REPROJ_MAX = 25.0          # px
GHOST_PX = 20.0            # 精修時同實際偵測配對嘅容差（px）
MERGE = 0.20               # 幾多米內當同一個筒（未修正尺度）
LAN, RAN = 15, 16


def detect(rec_dir, cache, model=MODEL, conf=CONF, n=N_FRAMES, imgsz=1280):
    import cv2
    os.makedirs(cache, exist_ok=True)
    todo = [c for c in CAMS if not os.path.exists(os.path.join(cache, c + '.json'))]
    if not todo:
        return
    from ultralytics import YOLO
    m = YOLO(model)
    for cam in todo:
        vid = os.path.join(rec_dir, cam + '.mp4')
        if not os.path.exists(vid):
            continue
        cap = cv2.VideoCapture(vid)
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        W, H = int(cap.get(3)), int(cap.get(4))
        picks = np.linspace(nf * 0.1, nf * 0.9, n).astype(int)
        det = []
        for f in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, fr = cap.read()
            if not ok:
                continue
            r = m.predict(fr, imgsz=imgsz, conf=conf, classes=CONE_CLS,
                          verbose=False, device='mps')[0]
            if r.boxes is None or not len(r.boxes):
                continue
            for (cx, cy, w, h), c in zip(r.boxes.xywh.cpu().numpy(),
                                         r.boxes.conf.cpu().numpy()):
                # 筒係企喺地下 —— 用**框底中點**做接地點，唔用框中心
                det.append([float(cx), float(cy + h / 2), float(w), float(h),
                            float(c), int(f)])
        cap.release()
        # 多幀共識：喺影像空間聚埋一堆（靜止物件），出中位位置 + 出現率
        cones, used = [], [False] * len(det)
        for i, d in enumerate(det):
            if used[i]:
                continue
            grp = [d]
            used[i] = True
            for j in range(i + 1, len(det)):
                if used[j]:
                    continue
                if abs(det[j][0] - d[0]) < 0.6 * d[2] and abs(det[j][1] - d[1]) < 0.6 * d[3]:
                    grp.append(det[j])
                    used[j] = True
            fr_seen = len(set(g[5] for g in grp))
            a = np.array(grp)
            cones.append({'x': float(np.median(a[:, 0])), 'y': float(np.median(a[:, 1])),
                          'w': float(np.median(a[:, 2])), 'h': float(np.median(a[:, 3])),
                          'conf': float(np.median(a[:, 4])),
                          'seen': fr_seen, 'of': len(picks)})
        json.dump({'meta': {'W': W, 'H': H, 'frames_sampled': len(picks)},
                   'cones': cones}, open(os.path.join(cache, cam + '.json'), 'w'))
        stable = sum(1 for c in cones if c['seen'] >= 0.6 * len(picks))
        print('  %s: %d 個候選，其中 %d 個喺 ≥60%% 幀都見到' % (cam, len(cones), stable))


def _undist(pts, K_cal, K_ckb, dist):
    import cv2
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K_ckb, dist, P=K_cal).reshape(-1, 2)


def _tri(P1, x1, P2, x2):
    A = np.stack([x1[0] * P1[2] - P1[0], x1[1] * P1[2] - P1[1],
                  x2[0] * P2[2] - P2[0], x2[1] * P2[2] - P2[1]])
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else np.full(3, np.nan)


def _proj(P, X):
    x = P @ np.append(X, 1.0)
    return x[:2] / x[2] if x[2] > 1e-9 else np.full(2, np.nan)


def build(out_dir, min_seen=0.6, verbose=True):
    """跨機位配對 → 三角化 → 只留 ≥3 台支持、而且貼近地面嘅點。"""
    rid = os.path.basename(out_dir)
    work = os.path.join(os.path.dirname(os.path.dirname(out_dir)), '_work', rid)
    cache = os.path.join(work, 'cones')
    if not all(os.path.exists(os.path.join(cache, c + '.json')) for c in CAMS):
        print('  ✗ %s: 偵測未齊' % rid)
        return None
    cal = json.load(open(os.path.join(out_dir, 'calibration.json')))
    ckb = json.load(open(CKB))
    K_ckb = np.array(ckb['0']['K'], np.float64).reshape(3, 3)
    dist = np.array(ckb['_distCoeffs'], np.float64)
    cams = []
    for v, cam in enumerate(CAMS):
        d = json.load(open(os.path.join(cache, cam + '.json')))
        K = np.array(cal[str(v)]['K'], np.float64).reshape(3, 3)
        R = np.array(cal[str(v)]['R'], np.float64).reshape(3, 3)
        T = np.array(cal[str(v)]['T'], np.float64).reshape(3, 1)
        keep = [c for c in d['cones']
                if c['seen'] >= min_seen * d['meta']['frames_sampled']]
        uv = (_undist([[c['x'], c['y']] for c in keep], K, K_ckb, dist)
              if keep else np.zeros((0, 2)))
        cams.append({'P': K @ np.hstack([R, T]), 'uv': uv, 'raw': keep})

    hits = []
    for i in range(4):
        for j in range(i + 1, 4):
            for a, ua in enumerate(cams[i]['uv']):
                for b, ub in enumerate(cams[j]['uv']):  # noqa: E501
                    X = _tri(cams[i]['P'], ua, cams[j]['P'], ub)
                    if not np.isfinite(X).all():
                        continue
                    e = max(np.linalg.norm(_proj(cams[i]['P'], X) - ua),
                            np.linalg.norm(_proj(cams[j]['P'], X) - ub))
                    if np.isfinite(e) and e <= REPROJ_MAX:
                        # 筒底半徑：框闊 ÷ 2 × 深度 ÷ 焦距。同量個波直徑同一招 ——
                        # 用物件自己嘅大細定接觸半徑，好過憑空估一個門檻。
                        rr = []
                        for (v, bi) in ((i, a), (j, b)):
                            Kc = np.array(cal[str(v)]['K'], np.float64).reshape(3, 3)
                            Rc = np.array(cal[str(v)]['R'], np.float64).reshape(3, 3)
                            Tc = np.array(cal[str(v)]['T'], np.float64).reshape(3)
                            zc = (Rc @ X + Tc)[2]
                            if zc > 0:
                                rr.append(cams[v]['raw'][bi]['w'] / 2 * zc / Kc[0, 0])
                        hits.append((X, {i, j}, e,
                                     float(np.median(rr)) if rr else np.nan))
    # 聚類：幾多台機支持
    pts, used = [], [False] * len(hits)
    for i, (X, vs, e, rad) in enumerate(hits):
        if used[i]:
            continue
        grp = [(X, vs, e, rad)]
        used[i] = True
        for j in range(i + 1, len(hits)):
            if not used[j] and np.linalg.norm(hits[j][0] - X) < MERGE:
                grp.append(hits[j])
                used[j] = True
        views = set().union(*[g[1] for g in grp])
        rads = [g[3] for g in grp if np.isfinite(g[3])]
        pts.append({'xyz': np.mean([g[0] for g in grp], 0).tolist(),
                    'n_cams': len(views),
                    'err': float(np.mean([g[2] for g in grp])),
                    'radius': float(np.median(rads)) if rads else float('nan')})
    # ---- 全視圖 bundle 精修 ----
    # 之前係「逐對機位三角化再平均」，位置誤差大約 10cm —— 實測 rec_20260915_114426
    # f143 就係因為一個筒擺錯咗位，做出用戶核實為假嘅「踩筒」，而且假到連
    # 收緊接觸門檻都分唔開（真陽性反而比佢更「淺」）。
    # 呢度改為：逐個候選點揾返每台機對應嘅實際偵測，然後同時最小化所有
    # 確認機位嘅重投影誤差。位置準咗，接觸判定先企得穩。
    from scipy.optimize import least_squares

    def _match(X, thr):
        """每台機搵最近嘅實際偵測（像素）。回傳 [(view, uv, box_w)]。"""
        out = []
        for v in range(4):
            if not len(cams[v]['uv']):
                continue
            u = _proj(cams[v]['P'], X)
            if not np.isfinite(u).all():
                continue
            k = int(np.argmin(np.linalg.norm(cams[v]['uv'] - u, axis=1)))
            if np.linalg.norm(cams[v]['uv'][k] - u) <= thr:
                out.append((v, cams[v]['uv'][k], cams[v]['raw'][k]['w']))
        return out

    for p in pts:
        X = np.array(p['xyz'], float)
        m = _match(X, GHOST_PX)
        if len(m) >= 2:
            def resid(x, m=m):
                return np.concatenate([_proj(cams[v]['P'], x) - uv for v, uv, _ in m])
            try:
                X = least_squares(resid, X, method='lm', max_nfev=200).x
            except Exception:
                pass
            m = _match(X, GHOST_PX)          # 精修後再確認一次
        p['xyz'] = X.tolist()
        p['confirm'] = len(m)
        p['err'] = (float(np.median([np.linalg.norm(_proj(cams[v]['P'], X) - uv)
                                     for v, uv, _ in m])) if m else 9e9)
        # 半徑用確認咗嘅機位重新量（框闊 ÷ 2 × 深度 ÷ 焦距）
        rr = []
        for v, _, bw in m:
            Kc = np.array(cal[str(v)]['K'], np.float64).reshape(3, 3)
            Rc = np.array(cal[str(v)]['R'], np.float64).reshape(3, 3)
            Tc = np.array(cal[str(v)]['T'], np.float64).reshape(3)
            zc = (Rc @ X + Tc)[2]
            if zc > 0:
                rr.append(bw / 2 * zc / Kc[0, 0])
        p['radius'] = float(np.median(rr)) if rr else float('nan')

    # 精修後可能有幾個候選塌埋同一點 —— 合併，保留誤差最細嗰個
    pts.sort(key=lambda q: q['err'])
    uniq = []
    for q in pts:
        if all(np.linalg.norm(np.array(q['xyz']) - np.array(u['xyz'])) > MERGE
               for u in uniq):
            uniq.append(q)
    pts = uniq

    # 鬼影守門：兩台機各自嘅**唔同**筒,射線一樣會喺空中交到一點（幾何鬼影），
    # 而且鬼影傾向落喺走廊中間 —— 啱啱係細路行過嘅地方,會做出假「踩筒」。
    # 實測未加呢條時，最近距離中位 19% 腿長（約 6cm 真實）= 喺筒底範圍之內，
    # 但片入面明明冇踩到；核圖亦見到有洋紅圈落喺空地。
    # 真筒喺**每一台**睇到佢嘅機度都應該對得返一個實際偵測。
    # 四台全部確認太嚴（筒成日被人／被其他筒遮，實測只剩 0–2 個）。
    # 改為 ≥3 台確認 + 精修後重投影誤差要細 —— 鬼影精修唔埋，誤差會爆。
    ERR_MAX = 10.0
    good = [p for p in pts if p['confirm'] >= 3 and p['err'] <= ERR_MAX]
    np.savez_compressed(os.path.join(out_dir, 'cones3d.npz'),
                        xyz=np.array([p['xyz'] for p in good], np.float32),
                        n_cams=np.array([p['n_cams'] for p in good], np.int8),
                        err=np.array([p['err'] for p in good], np.float32),
                        radius=np.array([p['radius'] for p in good], np.float32),
                        confirm=np.array([p['confirm'] for p in good], np.int8))
    if verbose:
        rr = [p['radius'] for p in good if np.isfinite(p['radius'])]
        print('  %s: 候選 %s 個/台 → 三角化 %d 個（≥3 台），重投影中位 %.1fpx，'
              '筒底半徑中位 %.1fcm'
              % (rid, '/'.join(str(len(c['uv'])) for c in cams), len(good),
                 float(np.median([p['err'] for p in good])) if good else -1,
                 100 * float(np.median(rr)) if rr else -1))
    return {'rec': rid, 'n_cones': len(good)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['detect', 'build'])
    ap.add_argument('--movement', required=True)
    ap.add_argument('--data', default=os.path.join(HERE, 'data'))
    ap.add_argument('--src', required=False, default='')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    recs = sorted(glob.glob(os.path.join(a.data, a.movement, 'rec_*')))
    recs = [d for d in recs if os.path.exists(os.path.join(d, 'meta.json'))]
    if a.limit:
        recs = recs[:a.limit]
    if not recs:
        sys.exit('✗ 冇已入庫錄影')
    for i, d in enumerate(recs, 1):
        rid = os.path.basename(d)
        if a.step == 'detect':
            src = a.src or json.load(open(os.path.join(d, 'meta.json')))['source']
            print('[%d/%d] %s' % (i, len(recs), rid))
            detect(src if os.path.isdir(src) else os.path.join(a.src, rid),
                   os.path.join(a.data, '_work', rid, 'cones'))
        else:
            build(d)


if __name__ == '__main__':
    main()
