"""用 roi.json 嘅表演區框，喺自標定之前剔走框外嘅人。

輸入：一個 rec 資料夾（cam*.mp4）+ roi.json（正規化 0-1 多邊形）
輸出：一個新資料夾，mp4 symlink + 同名 cam*.json（VideoRig 會當預存 2D 檢測讀）

用法:
  python roi_filter.py --rec <rec 資料夾> --roi roi.json --out <輸出資料夾>
  # 已經有原始檢測快取就唔會再跑 YOLO：--dets <detect_raw 輸出資料夾>
"""
import argparse, glob, json, os
import numpy as np

LHIP, RHIP = 11, 12
LSH, RSH = 5, 6


def hip_mid(k, s, W, H):
    a, b = s[LHIP] > .3, s[RHIP] > .3
    if a and b: p = (np.array(k[LHIP]) + np.array(k[RHIP])) / 2
    elif a:     p = np.array(k[LHIP])
    elif b:     p = np.array(k[RHIP])
    else:       return None
    return p / [W, H]


def inside(pt, poly):
    if len(poly) < 3: return True          # 空/不完整多邊形 = 唔過濾
    x, y = pt; hit = False
    for i in range(len(poly)):
        xi, yi = poly[i]; xj, yj = poly[i - 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            hit = not hit
    return hit


def detect(rec, out):
    """冇現成檢測就跑 YOLO11m-pose，存返 detect_raw 同一個格式。"""
    import cv2
    from ultralytics import YOLO
    m = YOLO('/Users/bcm01032/game7/human-selfcalib/yolo11m-pose.pt')
    os.makedirs(out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(rec, 'cam*.mp4'))):
        name = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(out, name + '.json')
        if os.path.exists(dst):
            print(f'  {name}: 用返快取'); continue
        cap = cv2.VideoCapture(p); W, H = int(cap.get(3)), int(cap.get(4))
        info, fid = [], 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            r = m.predict(fr, imgsz=960, verbose=False, device='mps')[0]
            insts = []
            if r.keypoints is not None and r.keypoints.xy is not None:
                xy = r.keypoints.xy.cpu().numpy()
                cf = (r.keypoints.conf.cpu().numpy()
                      if r.keypoints.conf is not None else np.ones(xy.shape[:2]))
                insts = [{'keypoints': k.tolist(), 'keypoint_scores': s.tolist()}
                         for k, s in zip(xy, cf)]
            info.append({'frame_id': fid, 'instances': insts}); fid += 1
        cap.release()
        json.dump({'meta': {'model': 'yolo11m-pose', 'video': name, 'W': W, 'H': H},
                   'instance_info': info}, open(dst, 'w'))
        print(f'  {name}: {fid} 幀')
    return out


def _track(info, W):
    """逐視角 2D 追蹤（最近鄰配對）。moving_tracks 同 jumping_tracks 共用。"""
    tracks = []
    for fr in info:
        fid = fr['frame_id']
        its = [(np.array(i['keypoints'], float),
                np.array(i['keypoint_scores'], float)) for i in fr['instances']]
        live = [t for t in tracks if fid - t['f'] <= 8]
        pairs = []
        for i, (k, s) in enumerate(its):
            for j, t in enumerate(live):
                m = (s > .3) & (t['s'] > .3)
                if m.sum() < 4:
                    continue
                d = float(np.linalg.norm(k[m] - t['k'][m], axis=1).mean())
                if d <= 0.05 * W * min(max(1, fid - t['f']), 4):
                    pairs.append((d, i, j))
        ui, uj = set(), set()
        for d, i, j in sorted(pairs):
            if i in ui or j in uj:
                continue
            ui.add(i); uj.add(j)
            t = live[j]; k, s = its[i]
            t.update(k=k, s=s, f=fid)
            t['hip'].append((k[LHIP] + k[RHIP]) / 2)
            t['K'].append(k)
            t['own'].append((fid, i))
        for i, (k, s) in enumerate(its):
            if i not in ui:
                tracks.append(dict(k=k, s=s, f=fid, own=[(fid, i)], K=[k],
                                   hip=[(k[LHIP] + k[RHIP]) / 2]))
    return tracks


def _pick(scored, min_score, keep_n):
    scored.sort(key=lambda x: -x[0])
    keep = [t for v, t in scored[:keep_n] if v >= min_score]
    want = {}
    for t in keep:
        for fid, i in t['own']:
            want.setdefault(fid, set()).add(i)
    return want, [round(v, 2) for v, _ in scored[:4]]


def moving_tracks(info, W, min_travel, keep_n):
    """跑動型：只保留**水平位移**最大嗰幾條軌跡。

    跑動型動作（敏捷梯、回來跑、方形跑）用呢個好過畫框：跑手係全場
    唯一會郁嘅人，而排隊嗰班就企喺跑道旁邊，影像空間嘅框分唔開佢哋。
    實測 rec_20260915_092821（K3 敏捷梯）：跑手位移/畫面寬 0.47–0.69，
    第二名 0.00–0.17，差 2.9–7.7 倍。
    """
    scored = []
    for t in _track(info, W):
        if len(t['hip']) < 15:
            continue
        P = np.array(t['hip'])
        scored.append((float(np.linalg.norm(P.max(0) - P.min(0)) / W), t))
    return _pick(scored, min_travel, keep_n)


def jumping_tracks(info, W, min_rise, keep_n):
    """原地跳躍型：只保留**垂直**行程最大嗰幾條軌跡。

    深蹲跳、開合跳呢類原地動作，水平位移同旁觀者一樣近乎零，
    moving_tracks 完全分唔開。垂直位移就分得開 —— 但一定要
    **除返佢自己嘅軀幹長**，唔可以除畫面高：企遠嗰個畫面上細，
    郁極都係得幾多像素。實測 rec_20260915_103656（K3 持球深蹲跳）：

        原始像素 ÷ 畫面高     頂兩條 0.18–0.23，第三條 0.09–0.14（得 1.5–2 倍，分唔開）
        ÷ 自己軀幹長          頂兩條 1.43–1.85，第三條 0.54–1.10（1.7–2.8 倍，分得開）

    兩位表演者穩定讀到 1.4–1.85 個軀幹長（蹲落去約一個軀幹 + 跳起），
    物理上合理。用 p5–p95 而唔用全距：一幀關節飛咗就會令全距爆掉。
    """
    scored = []
    for t in _track(info, W):
        if len(t['hip']) < 15:
            continue
        K = np.array(t['K'])
        sh = (K[:, LSH] + K[:, RSH]) / 2
        hip = (K[:, LHIP] + K[:, RHIP]) / 2
        torso = float(np.median(np.linalg.norm(sh - hip, axis=1)))
        if torso < 5:                      # 太細 = 檢測垃圾，除落去會爆
            continue
        y = hip[:, 1]
        scored.append((float((np.percentile(y, 95) - np.percentile(y, 5)) / torso), t))
    return _pick(scored, min_rise, keep_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rec', required=True)
    ap.add_argument('--roi', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--dets', default='')
    ap.add_argument('--moving', type=int, default=0, metavar='N',
                    help='改用「郁得最多」過濾：每個視角只保留位移最大嘅 N 條軌跡。'
                         '跑動型動作（敏捷梯／回來跑／方形跑）用呢個，唔使畫框')
    ap.add_argument('--min-travel', type=float, default=0.25,
                    help='--moving 嘅最低位移（÷畫面寬），預設 0.25')
    a = ap.parse_args()

    roi = json.load(open(a.roi))['regions'] if a.roi else {}
    dets = a.dets or os.path.join(a.out, '_raw')
    if not glob.glob(os.path.join(dets, 'cam*.json')):
        print('▶ 跑 YOLO 檢測…'); detect(a.rec, dets)
    os.makedirs(a.out, exist_ok=True)

    for f in sorted(glob.glob(os.path.join(dets, 'cam*.json'))):
        d = json.load(open(f)); meta = d['meta']
        name, W, H = meta['video'], meta['W'], meta['H']
        poly = roi.get(name, [])
        new, kept, tot = [], 0, 0
        want, top = (moving_tracks(d['instance_info'], W, a.min_travel, a.moving)
                     if a.moving else (None, None))
        for fr in d['instance_info']:
            keep = []
            for idx, i in enumerate(fr['instances']):
                tot += 1
                if want is not None:
                    ok = idx in want.get(fr['frame_id'], set())
                else:
                    hm = hip_mid(i['keypoints'], i['keypoint_scores'], W, H)
                    ok = hm is not None and inside(hm, poly)
                if ok:
                    keep.append(i); kept += 1
            new.append({'frame_id': fr['frame_id'], 'instances': keep})
        if top:
            print(f'   位移最大四條軌跡（÷畫面寬）：{top}')
        json.dump({'meta': meta, 'instance_info': new},
                  open(os.path.join(a.out, name + '.json'), 'w'))
        src = os.path.abspath(os.path.join(a.rec, name + '.mp4'))
        dst = os.path.join(a.out, name + '.mp4')
        if os.path.islink(dst) or os.path.exists(dst): os.remove(dst)
        os.symlink(src, dst)
        n = [len(x['instances']) for x in new]
        print(f'{name}: 保留 {kept}/{tot} 個檢測 → 每幀中位 {int(np.median(n))} 人，'
              f'最多 {max(n)}，空幀 {sum(1 for x in n if x == 0)}/{len(n)}')
    print(f'\n✅ 完成 → {a.out}（VideoRig 直接讀得）')


if __name__ == '__main__':
    main()
