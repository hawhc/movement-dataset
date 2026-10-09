# -*- coding: utf-8 -*-
"""準則計分：分檔內按比例（用戶 2026-09-25 揀「option 2」）。

每列三檔（門檻見 rubric 嘅 bands）：
  滿分檔：到滿分門檻 → 滿分
  部分檔：喺滿分門檻同部分門檻之間 → 按比例落 docx 部分分數範圍（貼近滿分門檻 → 範圍頂）
  零分檔：差過部分門檻 → 由零分範圍頂按比例跌到 0，去到「底線」（floor）就 0 分

底線 = 全班 p10（越大越好）／p90（越細越好）；如果 p10／p90 未差過部分門檻，
用最差嗰位；再唔得就部分門檻再行一個（滿分−部分）距離。寫入 rubric 嘅 scoring.floor。

計數類（掉球、碰欄杆、跌倒）唔可以「半次」，用 count_points 逐次數定分：
  用戶：「不掉球 掉 3 次要差過 2 次」→ 每多一次都扣。

用法：
    python score_rubric.py --movement k1_toss_catch            # 出每位細路總分
    python score_rubric.py --movement k1_toss_catch --write    # 將 scoring 寫入 rubric
"""
import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# docx 分數範圍規律（25：16–22／0–12；20：12–17／0–10；15：9–13／0–7；10 按比例）
RANGES = {25: ((16, 22), 12), 20: ((12, 17), 10), 15: ((9, 13), 7), 10: ((6, 8), 4)}

# 計數類：次數 → 分（最後一個數 = 嗰個次數或以上）
COUNT_POINTS = {
    ('k1_toss_catch', 'drops'): [20, 15, 10, 6, 3, 0],
    ('k1_hurdle_weave', 'hurdle_contacts'): [25, 12, 6, 0],
    ('k1_hurdle_weave', 'falls'): [20, 5, 0],
}
# 與欄杆距離：docx 寫明「偏差大：15–22」
RANGE_OVERRIDE = {('k1_hurdle_weave', 'hurdle_contacts'): ((15, 22), 12)}


def subjects(movement, grade):
    """同 compare_rubric 一樣經 build_norms 讀：剔 suspect／exclude.json，補 cyc_* 逐循環中位。"""
    import build_norms as BN
    rows = BN.subject_rows(BN.load(os.path.join(HERE, 'data'), movement, grade))
    raw = {}
    for f in glob.glob(os.path.join(HERE, 'data', movement, 'rec_*', 'metrics.json')):
        for x in json.load(open(f))['subjects']:
            raw[x['subject_ref']] = x
    for r in rows:
        r['subject_ref'] = r['_ref']
        r['did_perform'] = raw.get(r['_ref'], {}).get('did_perform')     # build_norms SKIP 咗
    return sorted(rows, key=lambda r: r['_ref'])


def _floor(row, vals):
    b = row['bands']
    full, part = b['full'], b['partial']
    lower = row['direction'] == 'lower'
    if not len(vals):
        return part + (part - full)
    cand = [float(np.percentile(vals, 90 if lower else 10)), float(max(vals) if lower else min(vals))]
    for c in cand:
        if (c > part) if lower else (c < part):
            return round(c, 2)
    return round(part + (part - full), 2)


def build_scoring(movement, rub, subs):
    perf = [s for s in subs if s.get('did_perform') is not False]
    for row in rub['rows']:
        if not row.get('bands'):
            continue
        key = (movement, row['metric'])
        (plo, phi), zhi = RANGE_OVERRIDE.get(key, RANGES[row['points']])
        sc = {'full_points': row['points'], 'partial_range': [plo, phi], 'zero_range': [0, zhi]}
        if key in COUNT_POINTS:
            sc['count_points'] = COUNT_POINTS[key]
        else:
            vals = [s[row['metric']] for s in perf if s.get(row['metric']) is not None]
            sc['floor'] = _floor(row, vals)
        if row.get('partial_if'):
            m = row['partial_if']['metric']
            vals = [s[m] for s in perf if s.get(m) is not None]
            sc['partial_if_floor'] = round(float(max(vals)), 2) if vals else None
        if row.get('zero_if'):
            sc['zero_if_points'] = round(zhi / 2)
        row['scoring'] = sc
    rub['_scoring_method'] = ('2026-09-25 用戶揀分檔內按比例（option 2）：部分檔按貼近滿分門檻嘅程度落 docx '
                              '部分範圍；零分檔由範圍頂跌到底線 floor（全班 p10／p90）先 0 分。計數類'
                              '（掉球／碰欄杆／跌倒）用 count_points 逐次定分，每多一次都扣。')


def _lerp(v, a, b, lo, hi):
    """v 由 a（→lo）去到 b（→hi），夾喺 [lo, hi]。"""
    if b == a:
        return hi
    f = min(1.0, max(0.0, (v - a) / (b - a)))
    return lo + f * (hi - lo)


def score_row(row, s):
    sc = row.get('scoring')
    if not sc:
        return None
    v = s.get(row['metric'])
    full, part = row['bands']['full'], row['bands']['partial']
    (plo, phi), (_, zhi) = sc['partial_range'], sc['zero_range']
    if row.get('zero_if'):
        z = row['zero_if']
        x = s.get(z['metric'])
        if x is not None and x >= z.get('at_least', float('inf')):
            return sc['zero_if_points']
    if v is None:
        return 0 if row.get('if_missing') else None
    if 'count_points' in sc:
        cp = sc['count_points']
        pts = cp[min(int(v), len(cp) - 1)]
        if pts == sc['full_points'] and row.get('partial_if'):
            pi = row['partial_if']
            x = s.get(pi['metric'])
            if x is not None and x > pi['above']:
                return round(_lerp(x, sc['partial_if_floor'], pi['above'], plo, phi), 1)
        return pts
    lower = row['direction'] == 'lower'
    good = (v <= full) if lower else (v >= full)
    mid = (v <= part) if lower else (v >= part)
    if good:
        return sc['full_points']
    if mid:
        return round(_lerp(v, part, full, plo, phi), 1)
    return round(_lerp(v, sc['floor'], part, 0, zhi), 1)


def score_subject(rub, s):
    """總分 /100。量唔到（None，唔係表現造成）嘅列唔入分母，再按比例還原做 100 分制
    —— 同 App 端「未偵測唔入分母」一致（kinetic-window-coverage-refactor）。
    if_missing 嘅列（量唔到＝表現造成，例如一次都冇接住）照計 0 分，唔剔。"""
    dnp = rub.get('did_not_perform', {})
    if s.get('did_perform') is False:
        return dnp.get('score', 15), {}
    pts = {r['label']: score_row(r, s) for r in rub['rows']}
    avail = sum(r['points'] for r in rub['rows'] if pts[r['label']] is not None)
    got = sum(p for p in pts.values() if p is not None)
    return (round(100 * got / avail, 1) if avail else None), pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--movement', required=True)
    ap.add_argument('--grade', default='K1')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    p = os.path.join(HERE, 'rubrics', f'{a.movement}_{a.grade}.json')
    rub = json.load(open(p))
    subs = subjects(a.movement, a.grade)
    build_scoring(a.movement, rub, subs)
    if a.write:
        json.dump(rub, open(p, 'w'), ensure_ascii=False, indent=1)
    labels = [r['label'] for r in rub['rows']]
    print(f"{rub['movement_label']}（{len(subs)} 位）")
    print('kid'.ljust(22) + ''.join(l[:6].ljust(8) for l in labels) + '總分')
    tot = []
    for s in subs:
        t, pts = score_subject(rub, s)
        tot.append(t)
        ref = s['subject_ref'].replace('rec_', '')
        if not pts:
            print(ref.ljust(22) + '冇做到動作 → 鼓勵分'.ljust(8 * len(labels)) + f'{t}')
            continue
        miss = sum(1 for l in labels if pts[l] is None)
        print(ref.ljust(22) + ''.join(('－' if pts[l] is None else f'{pts[l]:g}').ljust(8) for l in labels)
              + f'{t:g}' + (f'  （{miss} 列量唔到，按比例還原）' if miss else ''))
    print(f'總分 中位 {np.median(tot):.0f}、最低 {min(tot):.0f}、最高 {max(tot):.0f}')


if __name__ == '__main__':
    main()
