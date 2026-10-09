# -*- coding: utf-8 -*-
"""按機位穩定度自動分場 —— 唔好靠時鐘分。

點解要有呢個：`batch.py` 本來用 `日期`（後來 `日期+am/pm`）分場，決定邊幾條
可以共用一次自標定。兩次都唔夠：

  * 2026-09-15 持球深蹲跳：上午同下午同一日，機位平移 121–476px —— 靠日期分會撈埋。
  * 2026-09-15 「之」字移動：兩段都喺中午之前（09:4x / 11:3x），機位差 603px ——
    靠 am/pm 分一樣撈埋；而且 11:3x 嗰段**中途** cam02 郁咗（114540 跳 36px、
    114607 累積 101px），時鐘點分都分唔到。

所以直接由畫面判：逐條片同「本場第一條」比 ORB 特徵中位平移，超過門檻就開新一場。
共用標定用錯 = 三角化全錯，而且唔會報錯，所以寧可多開一場（多花 2–3 分鐘自標定）。

用法：
    python session_scan.py --src <資料夾> [--out sessions/<name>.json]
    python session_scan.py --src <資料夾> --max-shift 40 --frame 60

輸出 {rec_id: session_label}，batch.py 見到 sessions.json 就會用佢。
"""
import argparse, glob, json, os, sys
import numpy as np

MAX_SHIFT = 40.0      # px @1920x1080。實測：場內慢爬 12–27px 無害；
                      # 真係郁過嘅一跳就 36px 起，跨場 121–603px。
FRAME = 60


def _gray(path, frame):
    import cv2
    c = cv2.VideoCapture(path)
    c.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, fr = c.read()
    c.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if ok else None


def _shift(orb, bf, g0, g1):
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or not len(d0) or not len(d1):
        return float('nan')
    m = sorted(bf.match(d0, d1), key=lambda x: x.distance)[:300]
    if len(m) < 20:
        return float('nan')
    v = np.array([np.array(k1[x.trainIdx].pt) - np.array(k0[x.queryIdx].pt) for x in m])
    return float(np.hypot(*np.median(v, 0)))


def _shift_multi(orb, bf, a, b):
    """幾對幀入面取最細嘅平移 —— 只要有一對冇人遮，就量到真機位。"""
    if not a or not b:
        return float('nan')
    v = [_shift(orb, bf, x, y) for x, y in zip(a, b)
         if x is not None and y is not None]
    v = [x for x in v if np.isfinite(x)]
    return min(v) if v else float('nan')


def scan(src, max_shift=MAX_SHIFT, frame=FRAME, verbose=True):
    import cv2
    recs = sorted(d for d in glob.glob(os.path.join(src, 'rec_*')) if os.path.isdir(d))
    if not recs:
        sys.exit('✗ 搵唔到 rec_* ：%s' % src)
    frames = [frame, frame * 3, frame * 5]
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    out, anchor, sid, acams = {}, None, 0, []
    for d in recs:
        rid = os.path.basename(d)
        cams = sorted(os.path.basename(p)[:-4]
                      for p in glob.glob(os.path.join(d, 'cam0*.mp4')))
        # 取幾幀：近鏡頭有人企過會拖歪 ORB 中位（2026-09-23 跳轉身 09-16 實測
        # 老師企喺 cam03 前面報 55–291px 假郁機），真郁機係每幀都一樣咁移
        cur = {c: [_gray(os.path.join(d, c + '.mp4'), f) for f in frames]
               for c in cams}
        if anchor is None or set(cams) != set(acams):
            sid += 1
            anchor, acams = cur, cams
            out[rid] = sid
            if verbose:
                print('  %s  → 第 %d 場（新）' % (rid, sid))
            continue
        sh = {c: _shift_multi(orb, bf, anchor.get(c), cur.get(c)) for c in cams}
        mx = max([v for v in sh.values() if np.isfinite(v)] or [0.0])
        if mx > max_shift:
            sid += 1
            anchor, acams = cur, cams
            worst = max(sh, key=lambda c: (sh[c] if np.isfinite(sh[c]) else -1))
            if verbose:
                print('  %s  → 第 %d 場（%s 郁咗 %.0fpx）' % (rid, sid, worst, mx))
        else:
            if verbose:
                print('  %s     第 %d 場（最大 %.1fpx）' % (rid, sid, mx))
        out[rid] = sid
    # 每場用該場第一條片嘅日期（之前一律用全資料夾第一條 → 跨日嘅場標錯日期）
    first = {}
    for k in sorted(out):
        first.setdefault(out[k], k[4:12])
    return {k: '%s_s%d' % (first[v], v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--max-shift', type=float, default=MAX_SHIFT)
    ap.add_argument('--frame', type=int, default=FRAME)
    a = ap.parse_args()
    m = scan(a.src, a.max_shift, a.frame)
    n = len(set(m.values()))
    print('\n%d 條片 → %d 場' % (len(m), n))
    for s in sorted(set(m.values())):
        ks = sorted(k for k, v in m.items() if v == s)
        print('  %s: %d 條（%s … %s）' % (s, len(ks), ks[0], ks[-1]))
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'sessions', os.path.basename(a.src.rstrip('/')) + '.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(m, open(out, 'w'), ensure_ascii=False, indent=1)
    print('\n→ %s' % out)


if __name__ == '__main__':
    main()
