# -*- coding: utf-8 -*-
"""自動建議表演區 ROI —— 由「邊個真係喺度做緊動作」反推，唔靠人手估。

點解要：自標定要四台機睇住**同一批**表演者。`--moving N` 係逐台各自揀最郁嘅
N 條軌跡，四台會揀到唔同人（實測「之」字移動：cam01/03 剩 2 人、cam02/04 剩 1 人，
自標定直接失敗）。ROI 係影像空間嘅幾何條件，四台指住同一塊地，先夠一致。

判別（兩重，都係相對嘅，唔使知場地尺寸）：
  1. 有做動作 —— 軌跡嘅髖位移／髖高起伏夠大（--mode travel / squat）
  2. 唔係大人 —— 軀幹長 ≤ 1.3 × 該台機所有候選嘅中位（示範緊嘅教練體型明顯大）

出 rois/<name>.json（同框選工具同一個格式）+ 逐台疊圖畀人覆核。
**建議框未經人手確認，落場之前要開圖睇一眼。**

用法：
  python roi_propose.py --src <場次資料夾> --recs rec_a rec_b --mode travel \\
      --name yau_yat_chuen_zigzag_s1
"""
import argparse, glob, json, os
import numpy as np

LSH, RSH, LHIP, RHIP, LANK, RANK = 5, 6, 11, 12, 15, 16
CAMS = ['cam01_dev0', 'cam02_dev1', 'cam03_dev2', 'cam04_dev3']
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', '_work', 'roi_tracks')
YOLO_W = '/Users/bcm01032/game7/human-selfcalib/yolo11m-pose.pt'


def tracks(src, recs, cam, stride=3, imgsz=960):
    import cv2
    from ultralytics import YOLO
    m = YOLO(YOLO_W)
    out = []
    W = H = 0
    for rec in recs:
        p = os.path.join(src, rec, cam + '.mp4')
        if not os.path.exists(p):
            continue
        # 軌跡快取：調門檻重跑唔使再跑 YOLO（一場 10+ 分鐘）
        cp = os.path.join(CACHE, '%s_%s_s%d.pkl' % (rec, cam, stride))
        if os.path.exists(cp):
            import pickle
            W, H, got = pickle.load(open(cp, 'rb'))
            out += got
            continue
        cap = cv2.VideoCapture(p)
        W, H = int(cap.get(3)), int(cap.get(4))
        tr, fid = {}, -1
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fid += 1
            if fid % stride:
                continue
            r = m.track(fr, imgsz=imgsz, persist=True, verbose=False, device='mps')[0]
            if r.keypoints is None or r.boxes is None or r.boxes.id is None:
                continue
            for tid, k in zip(r.boxes.id.cpu().numpy(), r.keypoints.xy.cpu().numpy()):
                if (k[[LSH, RSH, LHIP, RHIP]] == 0).any():
                    continue
                hip = k[[LHIP, RHIP]].mean(0)
                sh = k[[LSH, RSH]].mean(0)
                t = hip[1] - sh[1]
                an = k[[LANK, RANK]]
                an = an[(an > 0).all(1)]
                st = (an[:, 1].mean() - hip[1]) / t if len(an) and t > 1 else np.nan
                if t > 1:
                    tr.setdefault(int(tid), []).append((hip[0], hip[1], t, st))
        cap.release()
        got = [np.array(v) for v in tr.values() if len(v) >= 20]
        import pickle
        os.makedirs(CACHE, exist_ok=True)
        pickle.dump((W, H, got), open(cp, 'wb'))
        out += got
    return out, W, H


def pick(trs, mode, min_travel=2.0, min_rise=0.5, adult=1.30, stand=0.9):
    """揀表演者。travel：髖橫向行程 ÷ 軀幹長；squat：髖高起伏 ÷ 軀幹長。

    另加企姿守門：(踝y − 髖y) ÷ 軀幹長 嘅 p75 要 ≥ stand。坐喺地下睇嘅細路
    郁下郁下都過到 squat 門檻（2026-09-23 跳繩 s4 實測每台揀咗 29–39 條，
    成排坐低嘅觀眾入晒框）；企 1.1–1.5、坐 0.07–0.47，分界好乾淨。
    用 p75 唔用中位：深蹲類有一半時間踎低。
    """
    cand = []
    for a in trs:
        torso = float(np.median(a[:, 2]))
        if torso <= 1:
            continue
        st = a[:, 3][np.isfinite(a[:, 3])]
        if len(st) < 10 or np.percentile(st, 75) < stand:
            continue
        if mode == 'travel':
            v = (np.percentile(a[:, 0], 97) - np.percentile(a[:, 0], 3)) / torso
            keep = v >= min_travel
        else:
            v = (np.percentile(a[:, 1], 95) - np.percentile(a[:, 1], 5)) / torso
            keep = v >= min_rise
        if keep:
            cand.append((a, torso, v))
    if not cand:
        return [], []
    med = float(np.median([c[1] for c in cand]))
    kid = [c for c in cand if c[1] <= adult * med]
    drop = [(round(c[1] / med, 2), round(c[2], 1)) for c in cand if c[1] > adult * med]
    return kid, drop


def dwell(a, torso):
    """軌跡停留最耐嘅位置：以半個軀幹長做格，攞點數最多嗰格入面嘅中位。"""
    xy = a[:, :2]
    q = np.floor(xy / max(torso * 0.5, 1)).astype(int)
    keys, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    return np.median(xy[inv.ravel() == cnt.argmax()], 0)


def shared_path(kid, share=0.3):
    """移動類：只留「好多條軌跡都行過」嘅格。

    同一場每個細路都行同一條線／路線，表演路徑係多條軌跡重疊嘅地方；後面
    行過嘅大人、坐喺度郁嚟郁去嘅觀眾都係零星一兩條（2026-09-23 平衡線 s9、
    前行擺腿 s36 實測：唔篩嘅話框拉到成個畫面闊）。
    格 = 半個中位軀幹長；一格要有 ≥ max(3, share × 軌跡數) 條唔同軌跡行過。
    """
    tor = float(np.median([c[1] for c in kid]))
    cell = max(tor * 0.5, 1.0)
    visits = {}
    for n, c in enumerate(kid):
        for q in {tuple(x) for x in np.floor(c[0][:, :2] / cell).astype(int)}:
            visits.setdefault(q, set()).add(n)
    need = max(3, int(np.ceil(share * len(kid))))
    keep = {q for q, v in visits.items() if len(v) >= need}
    P = np.vstack([c[0][:, :2] for c in kid])
    m = np.array([tuple(x) in keep for x in np.floor(P / cell).astype(int)])
    return P[m] if m.sum() >= 20 else P


def tape_segment(src, recs, cam):
    """地上白膠紙嘅最長線段（影像座標）。用 line3d.tape_pixels 同一個偵測。"""
    import cv2
    from line3d import tape_pixels
    for rec in recs:
        px, med = tape_pixels(os.path.join(src, rec, cam + '.mp4'))
        if med is None:
            continue
        m = np.zeros(med.shape[:2], np.uint8)
        m[px[:, 1].astype(int), px[:, 0].astype(int)] = 255
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        L = cv2.HoughLinesP(m, 1, np.pi / 360, 80, minLineLength=120, maxLineGap=25)
        if L is None:
            continue
        L = L.reshape(-1, 4).astype(float)
        return L[np.argmax(np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]))]
    return None


def on_line(kid, seg, frac=0.4):
    """平衡線類：留腳（≈ 髖x, 髖y + 企姿×軀幹）大部分時間喺條膠紙上嘅軌跡。

    2026-09-23 K1 平衡線前行：travel 模式揀咗跪低嘅老師同坐喺後排郁嚟郁去
    嘅細路（K1 細路細粒、場內人多），個框跌咗落觀眾度。條線本身就係表演區。
    """
    a, b = seg[:2], seg[2:]
    u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    out = []
    for c in kid:
        A, tor = c[0], c[1]
        st = np.where(np.isfinite(A[:, 3]), A[:, 3], np.nan)
        foot = np.stack([A[:, 0], A[:, 1] + st * A[:, 2]], 1)
        ok = np.isfinite(foot).all(1)
        if ok.sum() < 10:
            continue
        r = foot[ok] - a
        t = r @ u
        dperp = np.abs(r @ np.array([-u[1], u[0]]))
        inside = (t > -0.5 * tor) & (t < np.linalg.norm(b - a) + 0.5 * tor) \
            & (dperp < max(0.6 * tor, 25))
        if inside.mean() >= frac:
            out.append(c)
    return out


def line_poly(seg, kid, W, H, ext=0.8, below=0.5, above=2.3):
    """平衡線類嘅框直接由膠紙線段砌：兩端沿線外延 ext 個軀幹長，
    直向由腳下 below 到腳上 above 個軀幹長（判定點係髖中點，企直約 1.6 軀幹長高）。
    軀幹長隨遠近變 —— 用線附近細路嘅 (腳y → 軀幹長) 線性擬合，兩端各自估。
    """
    a, b = seg[:2].astype(float), seg[2:].astype(float)
    fy, ft = [], []
    for c in kid:
        A = c[0]
        ok = np.isfinite(A[:, 3])
        fy += list(A[ok, 1] + A[ok, 3] * A[ok, 2])
        ft += list(A[ok, 2])
    fy, ft = np.array(fy), np.array(ft)
    if len(fy) >= 20 and np.ptp(fy) > 30:
        m, c0 = np.polyfit(fy, ft, 1)
        tor = lambda y: float(np.clip(m * y + c0, np.percentile(ft, 5), np.percentile(ft, 95)))
    else:
        t0 = float(np.median(ft)) if len(ft) else 60.0
        tor = lambda y: t0
    u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    pts = []
    for e, sgn in ((a, -1), (b, 1)):
        t = tor(e[1])
        e2 = e + sgn * u * ext * t
        for dx in (-0.8 * t, 0.8 * t):
            pts.append([e2[0] + dx, e2[1] + below * t])
            pts.append([e2[0] + dx, e2[1] - above * t])
    import cv2
    hull = cv2.convexHull(np.array(pts, np.float32)).reshape(-1, 2)
    hull[:, 0] = np.clip(hull[:, 0], 0, W)
    hull[:, 1] = np.clip(hull[:, 1], 0, H)
    return hull.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--recs', nargs='+', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--mode', default='travel', choices=['travel', 'squat', 'line'])
    ap.add_argument('--pad', type=float, default=0.08)
    ap.add_argument('--dwell-pad-x', type=float, default=1.2,
                    help='squat 模式：停留點外加幾多個軀幹長（橫）')
    ap.add_argument('--dwell-pad-y', type=float, default=0.8,
                    help='squat 模式：停留點外加幾多個軀幹長（直）')
    ap.add_argument('--dwell-max', type=float, default=3.0,
                    help='squat 模式：離停留點中位超過幾多個軀幹長當離群')
    ap.add_argument('--share', type=float, default=0.3,
                    help='travel 模式：一格要有幾多成軌跡行過先算表演路徑')
    ap.add_argument('--hull', action='store_true',
                    help='travel 模式：用共同路徑點嘅凸包（斜行路徑用）而唔係長方形')
    ap.add_argument('--hull-pad', type=float, default=0.6,
                    help='凸包向外推幾多個軀幹長')
    ap.add_argument('--out-dir', default='rois')
    ap.add_argument('--img-dir', default='')
    a = ap.parse_args()
    import cv2
    regions, px, size = {}, {}, {}
    for cam in CAMS:
        trs, W, H = tracks(a.src, a.recs, cam)
        seg = None
        if a.mode == 'line':
            seg = tape_segment(a.src, a.recs, cam)
            cand, drop = pick(trs, 'travel', min_travel=0.8)
            kid = on_line(cand, seg, frac=0.25) if seg is not None else []
            # 條線搵到但冇軌跡落喺線上（s28 cam04 實測）：框照砌，軀幹長用全部候選估
            if seg is not None and not kid and cand:
                print('  %s: 冇軌跡喺線上，框純粹由膠紙線段砌' % cam)
                kid = cand
        else:
            kid, drop = pick(trs, a.mode)
        if not kid:
            print('  %s: ✗ 揀唔到表演者（%d 條軌跡）' % (cam, len(trs)))
            continue
        if a.mode == 'squat':
            # 原地類：框由每條軌跡「停留最耐嘅位置」砌，唔用全部點 —— 行入行出
            # 嘅路徑會將框拉到觀眾區（2026-09-23 跳繩 s4 實測）。
            D = np.array([dwell(c[0], c[1]) for c in kid])
            tor = float(np.median([c[1] for c in kid]))
            # 表演點唔止一個（兩個細路並排），但都喺幾個軀幹長之內；
            # 排隊／走位嘅離群停留點拉闊個框（實測 cam01 一點離群 4.5 軀幹長）
            far = np.hypot(*(D - np.median(D, 0)).T) > a.dwell_max * tor
            if far.any():
                print('  %s: 剔離群停留點 %d 個 %s' % (cam, far.sum(), D[far].astype(int).tolist()))
            D = D[~far]
            P = D
            lo = D.min(0) - [a.dwell_pad_x * tor, a.dwell_pad_y * tor]
            hi = D.max(0) + [a.dwell_pad_x * tor, a.dwell_pad_y * tor]
        elif a.mode == 'line':
            P = np.vstack([c[0][:, :2] for c in kid])
            poly = line_poly(seg, kid, W, H)
        else:
            P = shared_path(kid, a.share)
            lo, hi = np.percentile(P, [1, 99], axis=0)
            pad = a.pad * (hi - lo)
            lo, hi = lo - pad, hi + pad
        if a.mode == 'travel' and a.hull:
            # 斜行路徑用長方形一定包埋兩邊嘅觀眾（2026-09-23 前行擺腿 s36 cam02
            # 框到坐喺路邊嗰排細路、中位 8 人入框 → 自標定出錯、腳踝全部重建唔到）。
            # 改用共同路徑點嘅凸包，再向外推 hull_pad 個軀幹長。
            import cv2
            tor = float(np.median([c[1] for c in kid]))
            q = np.percentile(P, [1, 99], axis=0)
            Pq = P[((P >= q[0]) & (P <= q[1])).all(1)]
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ring = np.stack([np.cos(ang), np.sin(ang)], 1) * a.hull_pad * tor
            h0 = cv2.convexHull(Pq.astype(np.float32)).reshape(-1, 2)
            poly = cv2.convexHull((h0[:, None, :] + ring[None]).reshape(-1, 2)
                                  .astype(np.float32)).reshape(-1, 2)
            poly[:, 0] = np.clip(poly[:, 0], 0, W)
            poly[:, 1] = np.clip(poly[:, 1], 0, H)
            poly = poly.tolist()
            lo, hi = np.min(poly, 0), np.max(poly, 0)
        elif a.mode != 'line':
            lo, hi = np.maximum(lo, 0), np.minimum(hi, [W, H])
            poly = [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
        else:
            lo, hi = np.min(poly, 0), np.max(poly, 0)
        c4 = cam.split('_')[0]
        size[c4] = [W, H]
        px[c4] = [[int(x), int(y)] for x, y in poly]
        regions[c4] = [[round(x / W, 4), round(y / H, 4)] for x, y in poly]
        print('  %s: %d 條軌跡 → 表演者 %d、剔成人 %d %s → x[%.0f,%.0f] y[%.0f,%.0f]'
              % (cam, len(trs), len(kid), len(drop), drop, lo[0], hi[0], lo[1], hi[1]))
        img = a.img_dir or a.out_dir
        os.makedirs(img, exist_ok=True)
        cap = cv2.VideoCapture(os.path.join(a.src, a.recs[-1], cam + '.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
        ok, fr = cap.read()
        cap.release()
        if ok:
            for p in P:
                cv2.circle(fr, tuple(p.astype(int)), 3, (0, 140, 255), -1)
            cv2.polylines(fr, [np.array(poly, np.int32)], True, (0, 255, 255), 4)
            if seg is not None:
                cv2.line(fr, tuple(seg[:2].astype(int)), tuple(seg[2:].astype(int)), (255, 0, 255), 3)
            cv2.imwrite(os.path.join(img, '%s_%s.jpg' % (a.name, c4)),
                        cv2.resize(fr, (1280, 720)))
    os.makedirs(a.out_dir, exist_ok=True)
    dst = os.path.join(a.out_dir, a.name + '.json')
    json.dump({'_note': '自動建議（roi_propose.py，mode=%s）—— **未經人手確認**，'
                        '落場前開返 %s_cam0*.jpg 睇一眼。' % (a.mode, a.name),
               'test_point': 'hip_mid',
               'source': {c.split('_')[0]: c + '.mp4' for c in CAMS},
               'size': size, 'regions': regions, 'regions_px': px},
              open(dst, 'w'), ensure_ascii=False, indent=1)
    print('\n→ %s' % dst)


if __name__ == '__main__':
    main()
