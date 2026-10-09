"""球 3D：逐台跑球偵測 → 對極配對 + 手腕閘 → 三角化 → ball3d.npz

點解唔塞入 ComputeSession：球唔係人體關節，混入去會搞亂對極關聯同 re-id。
呢度係第二條偵測管線，借返同一場嘅 calibration.json 事後三角化。

用法：
  python ball3d.py detect  --movement k3_squat_jump_with_ball          # 第一步
  python ball3d.py build   --movement k3_squat_jump_with_ball          # 第二步
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.expanduser('~/Downloads/k3_squat_ball_all')
MODEL = '/Users/bcm01032/game7/best_4cls_bal.pt'
CAMS = ['cam01_dev0', 'cam02_dev1', 'cam03_dev2', 'cam04_dev3']
CONF = 0.10          # 放低收得盡；假陽性交畀幾何閘清，唔靠 conf 門檻


def detect(rec_dir, cache, model=MODEL, conf=CONF, imgsz=960, classes=(0,)):
    """四路全片跑球偵測，逐台存 JSON（有快取就跳過）。"""
    import cv2
    os.makedirs(cache, exist_ok=True)
    todo = [c for c in CAMS if not os.path.exists(os.path.join(cache, c + '.json'))]
    if not todo:
        print(f'  {os.path.basename(rec_dir)}: 用返球偵測快取')
        return
    from ultralytics import YOLO
    m = YOLO(model)
    for cam in todo:
        vid = os.path.join(rec_dir, cam + '.mp4')
        if not os.path.exists(vid):
            print(f'  ✗ 搵唔到 {vid}')
            continue
        cap = cv2.VideoCapture(vid)
        W, H = int(cap.get(3)), int(cap.get(4))
        out, fid = {}, 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            r = m.predict(fr, imgsz=imgsz, conf=conf, classes=list(classes),
                          verbose=False, device='mps')[0]
            if r.boxes is not None and len(r.boxes):
                xy = r.boxes.xywh.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                out[fid] = [[round(float(v), 2) for v in b] + [round(float(c), 3)]
                            for b, c in zip(xy, cf)]
            fid += 1
        cap.release()
        json.dump({'meta': {'model': os.path.basename(model), 'conf': conf,
                            'W': W, 'H': H, 'frames': fid}, 'boxes': out},
                  open(os.path.join(cache, cam + '.json'), 'w'))
        n = sum(len(v) for v in out.values())
        print(f'  {cam}: {fid} 幀、{n} 個框（{100*len(out)/max(fid,1):.0f}% 幀有）')



# ---------------------------------------------------------------- 時間對齊
def frame_offsets(work):
    """每台機嘅「session 幀號 → 影片幀號」常數偏移。

    ComputeSession 嘅 fidmap 當初冇存落硬碟，所以由現成中間檔反推：
    det_<v>.json 係 session 幀號索引、filtered/<cam>.json 係影片幀號索引，
    兩者裝住同一批關鍵點 —— 拎頭三個關節做指紋比中就知差幾多幀。
    實測係一個常數（rec_20260915_104407 四台都對得返）。
    """
    import collections
    offs, votes = {}, {}
    for v, cam in enumerate(CAMS):
        dp = os.path.join(work, f'det_{v}.json')
        fp = os.path.join(work, 'filtered', cam + '.json')
        if not (os.path.exists(dp) and os.path.exists(fp)):
            return None
        det = json.load(open(dp))['instance_info']
        filt = json.load(open(fp))['instance_info']

        def sig(fr):
            ins = fr['instances']
            return (None if not ins else
                    np.round(np.array(ins[0]['keypoints'])[:3], 2).tobytes())

        fmap = {}
        for fr in filt:
            g = sig(fr)
            if g:
                fmap.setdefault(g, fr['frame_id'])
        d = collections.Counter()
        for fr in det:
            g = sig(fr)
            if g and g in fmap:
                d[fmap[g] - fr['frame_id']] += 1
        if not d:
            return None
        off, n = d.most_common(1)[0]
        votes[cam] = d
        offs[cam] = int(off) if (n >= 20 and n >= 0.5 * sum(d.values())) else None
    if all(v is not None for v in offs.values()):
        return offs
    # 2026-09-25 rec_20260918_093840：四台機嘅偏移都唔係常數（8／5／7 幀混住，片中途
    # 跳幀），但四台跳法一樣 —— 相對偏移仍然係 0。用票最多嗰台嘅眾數套晒四台，
    # 即係「session 幀 = 影片幀」嘅相對關係唔變；絕對誤差 ≤3 幀（0.15s），氣球慢，得。
    tot = collections.Counter()
    for d in votes.values():
        tot.update(d)
    off, n = tot.most_common(1)[0]
    if n < 0.4 * sum(tot.values()):
        return None
    print(f'  ⚠ 幀偏移唔係常數，四台共用眾數 {off}（{n}/{sum(tot.values())} 票）')
    return {cam: int(off) for cam in CAMS}


# ---------------------------------------------------------------- 幾何
def _undistort(pts, K_cal, K_ckb, dist):
    import cv2
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K_ckb, dist, P=K_cal).reshape(-1, 2)


def _tri(P1, x1, P2, x2):
    """兩視圖 DLT。"""
    A = np.stack([x1[0] * P1[2] - P1[0], x1[1] * P1[2] - P1[1],
                  x2[0] * P2[2] - P2[0], x2[1] * P2[2] - P2[1]])
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else np.full(3, np.nan)


def _proj(P, X):
    x = P @ np.append(X, 1.0)
    return x[:2] / x[2] if x[2] > 1e-9 else np.full(2, np.nan)


CKB = '/Users/bcm01032/game7/human-selfcalib/calibs/tapo_c120_checkerboard.json'
LWR, RWR, LSH, RSH, LHIP, RHIP = 9, 10, 5, 6, 11, 12
# 關聯距離：球心離「人」最遠幾多米，仍然當係呢位受測者嗰個球。
# ⚠ 呢個係**關聯**閘，唔係「有冇揸住」嘅判斷 —— 球放咗喺地下／筒上一樣要追到，
# 「球喺唔喺胸前」係準則要量嘅指標，唔可以當成過濾條件剔走（實測收緊到
# 0.45m 覆蓋率得 44%，放到 1.5m 有 68%，差嗰批正正係放低咗嘅球）。
# 單位係**真實米**，入面會除返該位受測者嘅 scale 轉去未修正尺度嘅世界。
BODY_MAX = 1.2
REPROJ_MAX = 25.0     # 三角化重投影誤差上限（px）
MERGE = 0.15          # 兩個三角化結果差幾多米內當同一個球


def build(out_dir, verbose=True):
    """對極配對 + 手腕閘 + 三角化 → ball3d.npz。

    唔靠 conf 門檻分真假：牆上嘅畫、燈籠、卡通公仔喺單一機位同真球一樣圓，
    但係佢哋 (a) 喺另一台機嘅對極線上搵唔到對應、(b) 就算夾硬配到，三角化
    出嚟嘅點都唔會落喺表演者手腕附近。兩關一齊做，假陽性自然清走。
    """
    rid = os.path.basename(out_dir)
    work = os.path.join(os.path.dirname(os.path.dirname(out_dir)), '_work', rid)
    cal = json.load(open(os.path.join(out_dir, 'calibration.json')))
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    z = np.load(os.path.join(out_dir, 'pose3d.npz'), allow_pickle=True)
    xyz, valid, subs = z['xyz'], z['valid'], [str(x) for x in z['subjects']]
    scales = np.array([meta.get('scale_corrections', {}).get(r, 1.0) for r in subs])
    raw = xyz / scales[:, None, None, None]        # 三角化要喺未修正尺度嘅世界做

    offs = frame_offsets(work)
    if offs is None:
        print(f'  ✗ {rid}: 對唔到幀，跳過')
        return None
    json.dump(offs, open(os.path.join(out_dir, 'framemap.json'), 'w'))
    off0 = offs[CAMS[0]]

    ckb = json.load(open(CKB))
    K_ckb = np.array(ckb['0']['K'], np.float64).reshape(3, 3)
    dist = np.array(ckb['_distCoeffs'], np.float64)

    cams = []
    for v, cam in enumerate(CAMS):
        p = os.path.join(work, 'balls', cam + '.json')
        if not os.path.exists(p):
            print(f'  ✗ {rid}: 冇 {cam} 球偵測')
            return None
        d = json.load(open(p))
        K = np.array(cal[str(v)]['K'], np.float64).reshape(3, 3)
        R = np.array(cal[str(v)]['R'], np.float64).reshape(3, 3)
        T = np.array(cal[str(v)]['T'], np.float64).reshape(3, 1)
        cams.append({'boxes': {int(k): v2 for k, v2 in d['boxes'].items()},
                     'P': K @ np.hstack([R, T]), 'K': K, 'off': offs[cam]})

    S, nT = raw.shape[0], raw.shape[1]
    ball = np.full((S, nT, 3), np.nan, np.float32)
    ncam = np.zeros((S, nT), np.int8)
    stat = {'frames_any': 0, 'pairs': 0, 'rej_reproj': 0, 'rej_wrist': 0}

    for t in range(nT):
        sess = t - off0
        cand = []                                   # (view, undistorted xy, conf)
        for v, c in enumerate(cams):
            f = sess + c['off']
            for b in c['boxes'].get(f, []):
                u = _undistort([[b[0], b[1]]], c['K'], K_ckb, dist)[0]
                cand.append((v, u, b[4], (b[2] + b[3]) / 2))
        if len(cand) < 2:
            continue
        # 關聯用嘅身體參考點：手腕中點／肩中點／髖中點，就近者算
        wr = {}
        for s in range(S):
            if not valid[s, t]:
                continue
            pts = []
            for idx in ([LWR, RWR], [LSH, RSH], [LHIP, RHIP]):
                q = np.nanmean(raw[s, t, idx], 0)
                if np.isfinite(q).all():
                    pts.append(q)
            if pts:
                wr[s] = np.array(pts)
        if not wr:
            continue
        hits = []
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                if cand[i][0] == cand[j][0]:
                    continue
                X = _tri(cams[cand[i][0]]['P'], cand[i][1],
                         cams[cand[j][0]]['P'], cand[j][1])
                if not np.isfinite(X).all():
                    continue
                stat['pairs'] += 1
                e = max(np.linalg.norm(_proj(cams[cand[i][0]]['P'], X) - cand[i][1]),
                        np.linalg.norm(_proj(cams[cand[j][0]]['P'], X) - cand[j][1]))
                if not np.isfinite(e) or e > REPROJ_MAX:
                    stat['rej_reproj'] += 1
                    continue
                s_best, d_best = None, 1e9
                for s, w in wr.items():
                    d = float(np.min(np.linalg.norm(X - w, axis=1)) * scales[s])
                    if d < d_best:                    # 已轉返真實米
                        s_best, d_best = s, d
                if d_best > BODY_MAX:
                    stat['rej_wrist'] += 1
                    continue
                hits.append((s_best, X, e, d_best,
                             {cand[i][0], cand[j][0]}))
        if not hits:
            continue
        stat['frames_any'] += 1
        for s in set(h[0] for h in hits):
            hs = [h for h in hits if h[0] == s]
            # 揀支持機位最多嗰舊；平手就揀離手腕最近
            best, bn = None, -1
            for h in hs:
                grp = [g for g in hs if np.linalg.norm(g[1] - h[1]) < MERGE]
                vs = set().union(*[g[4] for g in grp])
                if len(vs) > bn or (len(vs) == bn and h[3] < best[3]):
                    best, bn = h, len(vs)
            ball[s, t] = best[1] * scales[s]         # 對返受測者嘅修正尺度
            ncam[s, t] = bn
    cov = float((ncam > 0).mean())
    np.savez_compressed(os.path.join(out_dir, 'ball3d.npz'),
                        ball=ball, n_cams=ncam, subjects=np.array(subs))
    if verbose:
        print(f'  {rid}: 覆蓋 {100*cov:.0f}%、'
              f'≥3 台 {100*float((ncam >= 3).mean()):.0f}%、'
              f'重投影剔 {stat["rej_reproj"]}、手腕閘剔 {stat["rej_wrist"]}')
    return {'rec': rid, 'coverage': round(cov, 3), **stat}


BALL_TRUE_CM = 18.1      # size 3 basketball（周長 56–57cm）—— 用戶 2026-09-17 提供


def world_scale(out_dir):
    """由個球反推世界尺度：個球係全場唯一已知尺寸嘅物件。

    d_world = 框短邊(px) × 深度 ÷ 焦距。用短邊因為手揸住嗰陣長邊會被拉闊。
    實測 14 條片、22 位、4 台機一致度 ±3%，喺手／唔喺手讀數差 2%（38.4 vs 39.1），
    即係量度本身穩陣。寫入 meta.json 嘅 ball_dia_raw_cm / world_scale_ball。

    ⚠ 呢個尺度同「肩寬錨點」對唔上（差約 1.4 倍，見 rubrics 檔 _scale_caveat）。
    兩者邊個啱要外部量度先定得到，所以呢度只係記錄，唔會自動覆蓋。
    """
    rid = os.path.basename(out_dir)
    work = os.path.join(os.path.dirname(os.path.dirname(out_dir)), '_work', rid)
    p = os.path.join(out_dir, 'ball3d.npz')
    if not os.path.exists(p):
        return None
    cal = json.load(open(os.path.join(out_dir, 'calibration.json')))
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    offs = json.load(open(os.path.join(out_dir, 'framemap.json')))
    off0 = offs[CAMS[0]]
    z = np.load(p, allow_pickle=True)
    boxes = {c: json.load(open(os.path.join(work, 'balls', c + '.json')))['boxes']
             for c in CAMS}
    D = []
    for si, ref in enumerate([str(x) for x in z['subjects']]):
        k = meta.get('scale_corrections', {}).get(ref, 1.0)
        for t in np.where(z['n_cams'][si] >= 3)[0]:
            X = z['ball'][si, t] / k
            if not np.isfinite(X).all():
                continue
            for v, cam in enumerate(CAMS):
                bb = boxes[cam].get(str(t - off0 + offs[cam]))
                if not bb:
                    continue
                K = np.array(cal[str(v)]['K']).reshape(3, 3)
                R = np.array(cal[str(v)]['R']).reshape(3, 3)
                T = np.array(cal[str(v)]['T']).reshape(3)
                xc = R @ X + T
                if xc[2] <= 0:
                    continue
                uv = (K @ (xc / xc[2]))[:2]
                bb = np.array(bb)
                j = int(np.argmin(np.linalg.norm(bb[:, :2] - uv, axis=1)))
                if np.linalg.norm(bb[j, :2] - uv) > 40:
                    continue
                D.append(min(bb[j, 2], bb[j, 3]) * xc[2] / K[0, 0] * 100)
    if len(D) < 100:
        return None
    dia = float(np.median(D))
    meta['ball_dia_raw_cm'] = round(dia, 2)
    meta['ball_true_cm'] = BALL_TRUE_CM
    meta['world_scale_ball'] = round(BALL_TRUE_CM / dia, 4)
    meta['ball_dia_n'] = len(D)
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'),
              ensure_ascii=False, indent=1)
    return meta['world_scale_ball'], dia, len(D)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['detect', 'build', 'scale'])
    ap.add_argument('--movement', required=True)
    ap.add_argument('--data', default=os.path.join(HERE, 'data'))
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--classes', default='0',
                    help='偵測類別（逗號分隔）：0=ball 1=balloon。K1 雙手拋接用氣球，'
                         '2026-09-24 實測四台機多數判做 balloon、少數 ball → 用 0,1')
    a = ap.parse_args()
    cls = tuple(int(x) for x in a.classes.split(','))
    recs = sorted(glob.glob(os.path.join(a.data, a.movement, 'rec_*')))
    if not recs:
        sys.exit(f'✗ {a.movement} 未有已入庫錄影')
    if a.step == 'scale':
        for d in recs:
            r = world_scale(d)
            print(f'  {os.path.basename(d)}: ' +
                  (f'球徑 {r[1]:.1f}cm（n={r[2]}）→ 世界尺度 ×{r[0]}'
                   if r else '量唔到'))
        return
    if a.step == 'build':
        rep = [build(d) for d in recs]
        rep = [r for r in rep if r]
        if rep:
            c = np.array([r['coverage'] for r in rep])
            print(f'\n✅ {len(rep)} 條完成，覆蓋率 中位 {100*np.median(c):.0f}%'
                  f'（最低 {100*c.min():.0f}%、最高 {100*c.max():.0f}%）')
        return
    if a.step == 'detect':
        for i, d in enumerate(recs, 1):
            rid = os.path.basename(d)
            print(f'[{i}/{len(recs)}] {rid}')
            detect(os.path.join(a.src, rid),
                   os.path.join(a.data, '_work', rid, 'balls'), a.model, classes=cls)
        print('\n✅ 球偵測完成')


if __name__ == '__main__':
    main()
