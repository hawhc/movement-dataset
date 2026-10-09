#!/usr/bin/env python
"""把現行準則門檻同真細路常模並排，睇下每條門檻實際分唔分到人。

回答四條問題：
  1. 門檻擺得啱唔啱   —— 現行門檻之下，幾多 % 細路落喺滿分／部分／零分
  2. 呢一列有冇分辨力 —— p90−p10 相對中位數；差異太細即係送分，評唔到嘢
  3. 有冇撞天花板／地板 —— 超過 80% 人落同一格 = 呢 20 分白畀
  4. 兩列係咪重複     —— 受測者之間嘅相關；>0.85 即係同一件事計兩次

⚠ 呢支程式唔會自動改準則。「良好應該擺喺邊」係教學決定，唔係統計決定
（揀 p50 做良好，即係永遠一半人唔合格 —— 係咪想要，要老師拍板）。
佢只係把證據攤出嚟，同埋計埋「如果改用某個百分位，門檻會變幾多」。

用法:
  python compare_rubric.py --movement k1_straight_run --grade K1
  python compare_rubric.py --movement k1_straight_run --grade K1 \\
      --propose-full p75 --propose-partial p25 --write
"""
import argparse
import json
import os

import numpy as np

import build_norms as BN

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = 0.25          # (p90−p10)/|p50| 細過呢個 = 分辨力不足
LUMP = 0.80          # 超過呢個比例落同一格 = 撞天花板／地板
DUP = 0.85           # 相關超過呢個 = 兩列可能重複


def _cond(c, rec):
    x = rec.get(c['metric']) if rec else None
    if x is None:
        return False
    return ('above' in c and x > c['above']) or ('at_least' in c and x >= c['at_least'])


def band_of(v, row, rec=None):
    """rec = 成個受測者嘅指標：zero_if（例如跌倒）→ 直接零分檔；
    partial_if（例如兜大圈）→ 滿分降做部分。"""
    b = row.get('bands')
    if not b:
        return None
    if row.get('zero_if') and _cond(row['zero_if'], rec):
        return '零分'
    if row['direction'] == 'lower':
        g = '滿分' if v <= b['full'] else ('部分' if v <= b['partial'] else '零分')
    else:
        g = '滿分' if v >= b['full'] else ('部分' if v >= b['partial'] else '零分')
    if g == '滿分' and row.get('partial_if') and _cond(row['partial_if'], rec):
        g = '部分'
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--movement', required=True)
    ap.add_argument('--grade', required=True)
    ap.add_argument('--data', default=os.path.join(HERE, 'data'))
    ap.add_argument('--rubric', default='')
    ap.add_argument('--propose-full', default='p75',
                    help='用邊個百分位做「滿分」門檻（p25/p50/p75/p90）')
    ap.add_argument('--propose-partial', default='p25')
    ap.add_argument('--write', action='store_true',
                    help='把建議門檻寫入 rubrics/<動作>_<年級>.proposed.json')
    a = ap.parse_args()

    rp = a.rubric or os.path.join(HERE, 'rubrics',
                                  f'{a.movement}_{a.grade}.json')
    if not os.path.exists(rp):
        raise SystemExit(f'✗ 搵唔到準則設定：{rp}')
    rub = json.load(open(rp))

    recs = BN.load(a.data, a.movement, a.grade)
    rows = BN.subject_rows(recs)
    if not rows:
        raise SystemExit('✗ data/ 入面搵唔到呢個動作嘅資料')
    norm = BN.norms(rows)
    n_sub = len(rows)
    enough = n_sub >= BN.MIN_SUBJECTS

    print(f"\n{rub['movement_label']}（{rub['movement_key']}）/ {a.grade}")
    print(f"{len(recs)} 段錄影、{n_sub} 位受測者"
          + ('' if enough else
             f'   ⚠ 少過 {BN.MIN_SUBJECTS} 位，以下數字只當方向參考，唔好改準則'))

    # 冇做到動作嘅細路：唔入常模，但要報返，因為佢哋係「零分界線」嘅唯一證據
    nonperf = []
    for meta, mx in recs:
        for x in mx['subjects']:
            if x.get('suspect') or x.get('did_perform') is not False:
                continue
            nonperf.append((meta['rec_id'], x))
    dnp = rub.get('did_not_perform')
    if nonperf:
        print(f"\n── 冇做到動作（{len(nonperf)} 位，唔入常模）")
        if dnp:
            print(f"   底分 {dnp['score']}/{rub['total']}「{dnp['label']}」"
                  f"（{dnp.get('decided_by', '')}）")
        else:
            print('   ⚠ 準則設定未定底分 —— 現行邏輯會當 0 分')
        for rid, x in nonperf:
            if 'peak_lift_deg' in x:                   # 步態家族
                print(f"   {x['subject_ref']:32s} 峰值 p90 {x['peak_lift_deg']}° "
                      f"< 閘 {x['min_lift_deg']}°，節奏類指標唔出數")
            else:                                       # 其他家族各自嘅前置閘
                print(f"   {x['subject_ref']:32s} did_perform=false（{dnp.get('applies_when', '家族前置閘') if dnp else '家族前置閘'}）")

    proposals = []
    for row in rub['rows']:
        key = row['metric']
        n = norm.get(key)
        print(f"\n── {row['label']}（{row['points']} 分）· {key} · "
              f"{'越細越好' if row['direction'] == 'lower' else '越大越好'}")
        if not n:
            print('   ⚠ 常模冇呢個指標（ingest 未量／全部 null）')
            continue
        recs = [r for r in rows if key in r and np.isfinite(r[key])]
        vals = np.array([r[key] for r in recs], float)
        print(f"   常模  p10 {n['p10']}  p25 {n['p25']}  p50 {n['p50']}  "
              f"p75 {n['p75']}  p90 {n['p90']}   (n={n['n_subjects']})")

        # 1. 現行門檻之下嘅分佈
        if row.get('bands'):
            b = row['bands']
            cnt = {}
            for v, rec in zip(vals, recs):
                g = band_of(v, row, rec)
                cnt[g] = cnt.get(g, 0) + 1
            dist = '  '.join(f'{g} {cnt.get(g, 0)}人 {cnt.get(g, 0)/len(vals)*100:.0f}%'
                             for g in ('滿分', '部分', '零分'))
            print(f"   現行  滿分≤{b['full']} / 部分≤{b['partial']}"
                  if row['direction'] == 'lower' else
                  f"   現行  滿分≥{b['full']} / 部分≥{b['partial']}")
            print(f"   實際  {dist}")
            top = max(cnt.values()) / len(vals) if cnt else 0
            if top >= LUMP:
                g = max(cnt, key=cnt.get)
                print(f"   ⚠ {top*100:.0f}% 人落晒「{g}」—— 呢 {row['points']} 分"
                      f"分唔到人，門檻要調")
        else:
            print(f"   現行  冇數字門檻（{row['docx']}）—— 呢條正好由數據定")

        # 2. 分辨力
        spread = (n['p90'] - n['p10']) / max(abs(n['p50']), 1e-9)
        flag = '  ⚠ 差異太細，評唔到嘢' if spread < FLAT else ''
        print(f"   分辨力 (p90−p10)/p50 = {spread:.2f}{flag}")

        # 3. 建議門檻
        # ⚠ 方向：higher-better 嘅「好」喺分佈高位（p75），
        # lower-better 嘅「好」喺低位（p25）—— 唔調轉就會建議「越差越滿分」。
        if row['direction'] == 'lower':
            kf, kp = a.propose_partial, a.propose_full     # 對調
        else:
            kf, kp = a.propose_full, a.propose_partial
        pf, pp = n.get(kf), n.get(kp)
        if pf is not None:
            print(f'   建議  滿分={pf}（{kf}）  部分={pp}（{kp}）'
                  f"  → 約 {25 if row['direction']=='lower' else 25}% 滿分、"
                  f'50% 部分、25% 零分')
            if row.get('bands'):
                d = pf - row['bands']['full']
                print(f"         同現行差 {d:+.2f}"
                      + ('   ← 差好遠，要人手拍板' if abs(d) >
                         0.5 * abs(row['bands']['full']) else ''))
            proposals.append({**row, 'proposed': {'full': pf, 'partial': pp},
                              'proposed_from': [kf, kp],
                              'n_subjects': n['n_subjects']})

    # 4. 重複檢查
    keys = [r['metric'] for r in rub['rows'] if r['metric'] in norm]
    print('\n── 兩列係咪量緊同一樣嘢（受測者之間嘅相關）')
    found = False
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            pair = [(r[ka], r[kb]) for r in rows if ka in r and kb in r
                    and np.isfinite(r[ka]) and np.isfinite(r[kb])]
            if len(pair) < 4:
                continue
            c = float(np.corrcoef(np.array(pair).T)[0, 1])
            if abs(c) >= DUP:
                found = True
                print(f"   ⚠ {ka} ↔ {kb}  r={c:+.2f} —— 可能係同一件事計兩次分")
    if not found:
        print('   （樣本不足或者冇發現高度相關）' if n_sub < 4 else '   冇發現高度相關')

    if a.write and proposals:
        out = os.path.join(HERE, 'rubrics',
                           f'{a.movement}_{a.grade}.proposed.json')
        json.dump({'_note': '由細路常模導出嘅建議門檻，未經人手審批，'
                            '唔好直接覆蓋現行準則',
                   'movement_key': a.movement, 'grade': a.grade,
                   'n_subjects': n_sub, 'sufficient': enough,
                   'rows': proposals}, open(out, 'w'),
                  ensure_ascii=False, indent=1)
        print(f'\n📄 建議門檻寫入 {out}')


if __name__ == '__main__':
    main()
