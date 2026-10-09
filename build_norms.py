#!/usr/bin/env python
"""由 ingest.py 出嘅一批 dataset 砌常模（norm）。

兩段式彙總 —— 呢個係唔可以省嘅一步：
  1. 同一位受測者嘅多個循環 → 先取中位數，變成「一次表演一個代表值」
  2. 各次表演之間 → 先算百分位
做一段式（所有循環倒落一個池）會令做得多循環嗰個細路佔重幾倍，
常模就會偏向佢。

用法:
  python build_norms.py --movement k1_straight_run --grade K1
  python build_norms.py --all                 # 掃晒 data/ 入面每個動作
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 分佈類指標先做百分位；識別欄位／身型唔入常模
SKIP = {'track_id', 'subject_ref', 'frames', 'coverage', 'duration_s',
        'suspect', 'suspect_reason', 'hand_raise_frame', 'assess_start_frame',
        'start_trigger', 'trigger_found', 'did_perform', 'peak_lift_deg',
        'min_lift_deg', 'cycles_dropped_short',
        'scale_correction', 'raw_shoulder_cm', 'cycles'}
MIN_SUBJECTS = 20          # 少過呢個數就標「參考資料不足」，唔好攞去評學生
EXCLUDE = os.path.join(HERE, 'exclude.json')


def manual_excludes():
    """人手排除清單 —— 守門規則捉唔到、覆查之後確認唔可信嘅受測者。

    刻意做成一個檔而唔係去 data/ 度刪資料：刪咗就冇得覆查，而且下次
    有人重跑 batch.py 會靜靜雞入返去。清單入面每條要寫低理由。
    """
    out = {}
    if os.path.exists(EXCLUDE):
        out.update(json.load(open(EXCLUDE, encoding='utf-8')).get('subjects', {}))
    # auto_ingest.py（admin 頁評分錄影自動入庫）逐條片寫嘅 seat_match.json：
    # 冇對應評分席位嘅軌跡（旁觀者／老師）同同一席位嘅碎片軌跡唔入常模。
    # 放喺 rec 資料夾而唔係改 metrics.json —— ingest --remeasure 會重寫 metrics，
    # 標記會冇咗。
    for p in glob.glob(os.path.join(HERE, 'data', '*', 'rec_*', 'seat_match.json')):
        try:
            out.update(json.load(open(p, encoding='utf-8')).get('exclude', {}))
        except Exception:
            pass
    return out


def load(data_dir, movement, grade=None):
    recs = []
    for m in sorted(glob.glob(os.path.join(data_dir, movement or '*'))):
        if os.path.basename(m).startswith('_'):
            continue
        for d in sorted(glob.glob(os.path.join(m, '*'))):
            mp = os.path.join(d, 'meta.json')
            xp = os.path.join(d, 'metrics.json')
            if not (os.path.exists(mp) and os.path.exists(xp)):
                continue
            meta = json.load(open(mp))
            if grade and meta.get('grade') != grade:
                continue
            recs.append((meta, json.load(open(xp))))
    return recs


def subject_rows(recs, stats=None):
    """每位受測者一行：summary 指標 + 逐循環指標嘅中位數。

    只排除 suspect —— 佢哋根本唔係受測者（老師、坐喺地下睇嘅細路、碎片軌跡）。

    ⚠ did_perform=False 嘅學生**要照計**（用戶 2026-09-15 決定）。佢都係班入面
    嘅一分子；剔走佢就等於當佢唔存在，令「典型 K1 表現」被高估，之後攞個常模
    去比其他學生，個個都會顯得差過實際。佢哋嘅幅度類指標（膝抬、漂移）正正
    就係分佈嘅低端證據。
    節奏類指標（步頻／CV／停頓）佢哋係 null —— 冇跑就真係量唔到節奏，
    嗰幾條指標個 n 自然會細啲，呢個係誠實，唔係排除。
    """
    rows = []
    drop = manual_excludes()
    for meta, mx in recs:
        cyc = mx.get('cycles', {})
        for s in mx['subjects']:
            if s.get('suspect'):        # 混入嘅大人／碎片軌跡，唔入常模
                if stats is not None:
                    stats['suspect'] = stats.get('suspect', 0) + 1
                continue
            if s['subject_ref'] in drop:        # exclude.json 人手排除
                if stats is not None:
                    stats.setdefault('manual', []).append(s['subject_ref'])
                continue
            if s.get('did_perform') is False and stats is not None:
                stats['did_not_perform'] = stats.get('did_not_perform', 0) + 1
            r = {k: v for k, v in s.items() if k not in SKIP and v is not None}
            r['_ref'] = s['subject_ref']
            r['_rec'] = meta['rec_id']
            r['_grade'] = meta['grade']
            r['_movement'] = meta['movement_key']
            r['_session'] = meta.get('session')
            per = cyc.get(s['subject_ref'], [])
            keys = {k for c in per for k, v in c.items()
                    if isinstance(v, (int, float)) and k not in
                    ('cycle', 'start_frame', 'end_frame')}
            for k in keys:
                vals = [c[k] for c in per if c.get(k) is not None]
                if vals:
                    r['cyc_' + k] = float(np.median(vals))
            rows.append(r)
    return rows


def norms(rows):
    keys = sorted({k for r in rows for k in r if not k.startswith('_')})
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r], float)
        v = v[np.isfinite(v)]
        if len(v) < 2:          # 兩位就算得出，可唔可以用睇 sufficient 旗標
            continue
        p = np.percentile(v, [10, 25, 50, 75, 90])
        iqr = p[3] - p[1]
        out[k] = {
            'n_subjects': int(len(v)),
            'p10': round(float(p[0]), 2), 'p25': round(float(p[1]), 2),
            'p50': round(float(p[2]), 2), 'p75': round(float(p[3]), 2),
            'p90': round(float(p[4]), 2),
            'mean': round(float(v.mean()), 2),
            'sd': round(float(v.std(ddof=1)), 2) if len(v) > 1 else 0.0,
            # 穩健 sd：IQR/1.349，畀離群值污染嘅機會細好多
            'robust_sd': round(float(iqr / 1.349), 2),
            'min': round(float(v.min()), 2), 'max': round(float(v.max()), 2),
        }
    return out


def zscore(value, n):
    """一位學生對常模：百分位同穩健 z。"""
    sd = n['robust_sd'] or n['sd']
    z = (value - n['p50']) / sd if sd else 0.0
    pts = [n['p10'], n['p25'], n['p50'], n['p75'], n['p90']]
    pct = float(np.interp(value, pts, [10, 25, 50, 75, 90]))
    return round(z, 2), round(pct, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'data'))
    ap.add_argument('--movement', default='')
    ap.add_argument('--grade', default='')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'norms'))
    a = ap.parse_args()

    combos = []
    if a.all:
        for m in sorted(glob.glob(os.path.join(a.data, '*'))):
            if os.path.basename(m).startswith('_'):
                continue
            for meta, _ in load(a.data, os.path.basename(m)):
                combos.append((meta['movement_key'], meta['grade']))
        combos = sorted(set(combos))
    else:
        if not a.movement:
            ap.error('要 --movement 或 --all')
        combos = [(a.movement, a.grade or None)]

    os.makedirs(a.out, exist_ok=True)
    for movement, grade in combos:
        recs = load(a.data, movement, grade)
        stats = {}
        rows = subject_rows(recs, stats)
        if not rows:
            print(f'{movement} / {grade}: 冇資料')
            continue
        n = norms(rows)
        tag = f'{movement}_{grade or "ALL"}'
        doc = {'movement_key': movement, 'grade': grade,
               'n_recordings': len(recs), 'n_subjects': len(rows),
               'excluded_suspect': stats.get('suspect', 0),
               'excluded_manual': stats.get('manual', []),
               'n_did_not_perform': stats.get('did_not_perform', 0),
               'did_not_perform_rate': (round(stats.get('did_not_perform', 0)
                                              / len(rows), 3) if rows else None),
               'sufficient': len(rows) >= MIN_SUBJECTS,
               'min_subjects_for_use': MIN_SUBJECTS,
               'aggregation': '兩段式：受測者內取中位數 → 受測者之間取百分位',
               'metrics': n}
        json.dump(doc, open(os.path.join(a.out, tag + '.json'), 'w'),
                  ensure_ascii=False, indent=1)

        flag = '' if doc['sufficient'] else f'  ⚠ 少過 {MIN_SUBJECTS} 位，唔好攞去評學生'
        parts = []
        if stats.get('suspect'):
            parts.append(f"排除 {stats['suspect']} 位疑似大人/碎片")
        if stats.get('manual'):
            parts.append(f"人手排除 {len(stats['manual'])} 位"
                         f"（{'、'.join(stats['manual'])}）")
        if stats.get('did_not_perform'):
            parts.append(f"當中 {stats['did_not_perform']} 位冇做到動作（照計）")
        ex = ('，' + '、'.join(parts)) if parts else ''
        print(f'\n=== {movement} / {grade or "全部年級"} ── '
              f'{len(recs)} 段錄影、{len(rows)} 位受測者{ex}{flag}')
        print(f'  {"指標":<26}{"n":>4}{"p10":>9}{"p25":>9}{"p50":>9}'
              f'{"p75":>9}{"p90":>9}')
        for k, v in n.items():
            print(f'  {k:<26}{v["n_subjects"]:>4}{v["p10"]:>9.2f}{v["p25"]:>9.2f}'
                  f'{v["p50"]:>9.2f}{v["p75"]:>9.2f}{v["p90"]:>9.2f}')
    print(f'\n✅ 常模寫入 {a.out}/')


if __name__ == '__main__':
    main()
