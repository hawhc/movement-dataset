#!/usr/bin/env python
"""睇下入咗庫嘅嘢啱唔啱 —— 邊條軌跡係真表演者、邊條係旁觀者。

自動守門（唔係企住／體型偏離／幀數太少）捉到大部分，但唔係萬能。
出事嗰陣要用肉眼核，呢支就係做呢件事：把 3D 骨架反投影返落四個機位，
逐個標住編號，你一眼就認到邊個係邊個。

用法:
  # 一覽：邊位入咗常模、邊位畀守門剔咗、點解
  python review.py --movement k1_straight_run

  # 某條錄影畫張四機標號圖出嚟（存喺該錄影嘅資料夾）
  python review.py --movement k1_straight_run --rec rec_20260915_100345 --image

  # 指定幀（預設揀同時見到最多人嗰幀）
  python review.py --movement ... --rec ... --image --frame 93
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]
COLORS = [(0, 215, 255), (255, 120, 0), (80, 255, 80), (200, 80, 255),
          (80, 80, 255), (255, 255, 0), (0, 160, 255), (160, 255, 160)]


def status(s):
    if s.get('suspect'):
        return '❌ 唔入常模', s.get('suspect_reason', '')
    if s.get('did_perform') is False:
        return '⚠ 冇做到動作', f"峰值 p90 {s['peak_lift_deg']}° < 閘 {s['min_lift_deg']}°（照計入常模）"
    return '✅ 入常模', ''


def overlay(out_dir, src_rec, frame=None):
    """把每位受測者嘅 3D 骨架反投影返四個機位，標住編號。"""
    import cv2
    cal = json.load(open(os.path.join(out_dir, 'calibration.json')))
    d = np.load(os.path.join(out_dir, 'pose3d.npz'), allow_pickle=True)
    X, V, S = d['xyz'], d['valid'], d['subjects']
    meta = json.load(open(os.path.join(out_dir, 'meta.json')))
    sc = meta.get('scale_corrections', {})
    if frame is None:
        frame = int(np.argmax(V.sum(0)))          # 同時見到最多人嗰幀
    vids = sorted(glob.glob(os.path.join(src_rec, 'cam*.mp4')))
    if not vids:
        return None, frame, '✗ 搵唔到原片，畫唔到圖（rec 資料夾搬咗？用 --src 指定）'
    tiles = []
    for ci, v in enumerate(vids):
        e = cal[str(ci)]
        K = np.array(e['K']).reshape(3, 3)
        R = np.array(e['R']).reshape(3, 3)
        T = np.array(e['T']).ravel()
        cap = cv2.VideoCapture(v)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        for i, ref in enumerate(S):
            if not V[i][frame]:
                continue
            P = X[i][frame] / sc.get(str(ref), 1.0)   # 反轉尺度修正，返標定單位
            cam = (R @ P.T).T + T
            uv = (K @ cam.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            good = np.isfinite(uv).all(1) & (cam[:, 2] > 0)
            c = COLORS[i % len(COLORS)]
            for a, b in EDGES:
                if good[a] and good[b]:
                    cv2.line(img, tuple(uv[a].astype(int)),
                             tuple(uv[b].astype(int)), c, 4)
            if good[11] and good[12]:
                p = ((uv[11] + uv[12]) / 2).astype(int)
                cv2.putText(img, '#' + str(ref).split('#')[-1],
                            (p[0] - 25, p[1] - 95), 0, 1.7, c, 6)
        tiles.append(cv2.resize(img, (760, 428)))
    if len(tiles) < 4:
        return None, frame, f'✗ 得 {len(tiles)} 個機位讀到畫面'
    dst = os.path.join(out_dir, f'review_f{frame}.jpg')
    cv2.imwrite(dst, np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    return dst, frame, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--movement', required=True)
    ap.add_argument('--rec', default='', help='淨係睇呢條錄影')
    ap.add_argument('--data', default=os.path.join(HERE, 'data'))
    ap.add_argument('--image', action='store_true', help='畫四機標號圖')
    ap.add_argument('--frame', type=int, default=None)
    ap.add_argument('--src', default='', help='原片資料夾（搬咗位置先要指定）')
    a = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(a.data, a.movement, 'rec_*')))
    if a.rec:
        dirs = [d for d in dirs if os.path.basename(d) == a.rec]
    if not dirs:
        raise SystemExit(f'✗ 搵唔到 {a.movement}' + (f' / {a.rec}' if a.rec else ''))

    tot = {'ok': 0, 'nodo': 0, 'susp': 0}
    for d in dirs:
        mp = os.path.join(d, 'meta.json')
        if not os.path.exists(mp):        # 自標定失敗會留低空資料夾
            print(f'⚠ {os.path.basename(d)}: 冇 meta.json（入庫失敗？）—— 跳過')
            continue
        meta = json.load(open(mp))
        mx = json.load(open(os.path.join(d, 'metrics.json')))
        print(f"\n═══ {meta['rec_id']}  {meta.get('movement_label', '')}  "
              f"{meta['grade']}  ({meta['frames']} 幀)")
        for s in mx['subjects']:
            st, why = status(s)
            tot['susp' if s.get('suspect') else
                ('nodo' if s.get('did_perform') is False else 'ok')] += 1
            hr = ('' if not s.get('assess_start_frame')
                  else f" 舉手f{s['hand_raise_frame']}→f{s['assess_start_frame']}")
            print(f"  {s['subject_ref']:34s} {st}  {s['frames']} 幀、"
                  f"{s['cycles']} 循環、企姿 {s.get('stance_ratio')}"
                  f"/p90 {s.get('stance_peak')}{hr}")
            if why:
                print(f"      └ {why}")
        if a.image:
            src = a.src or meta.get('source', '')
            path, fr, err = overlay(d, src, a.frame)
            print(f"  🖼  {err or f'第 {fr} 幀標號圖 → {path}'}")

    print(f"\n合計：{tot['ok']} 位入常模、{tot['nodo']} 位冇做到（照計）、"
          f"{tot['susp']} 位守門剔走")
    print('圖入面編號同上面對得返。如果有旁觀者冇被剔走，或者真表演者被錯剔，'
          '話我知，守門規則要調。')


if __name__ == '__main__':
    main()
