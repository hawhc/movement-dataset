#!/usr/bin/env python
"""審批建議門檻：將 rubrics/<動作>_<年級>.proposed.json 揀咗嘅列寫入正式準則。

compare_rubric.py --write 只會出 *.proposed.json（「未經人手審批」）。呢個係**人手拍板**
嗰一步（admin 頁「待審批門檻」撳批准就行呢度）：
  * 只改揀咗嘅列嘅 bands；舊值連日期、來源記入 row['bands_history']，隨時可以追返
  * bands_source 寫明常模百分位同樣本數
  * 用 score_rubric.build_scoring 按新門檻重算分檔（零分底線 floor 等）
  * 頂層 _changelog 加一條

用法：
    python approve_rubric.py --movement k1_squat --grade K1 --metrics squat_knee_min_deg,knee_align_deg
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def approve(movement, grade, metrics, who='admin 頁審批'):
    import score_rubric as SR
    rp = os.path.join(HERE, 'rubrics', f'{movement}_{grade}.json')
    pp = os.path.join(HERE, 'rubrics', f'{movement}_{grade}.proposed.json')
    rub = json.load(open(rp, encoding='utf-8'))
    prop = json.load(open(pp, encoding='utf-8'))
    bym = {r['metric']: r for r in prop['rows']}
    today = time.strftime('%Y-%m-%d')
    changed = []
    for row in rub['rows']:
        p = bym.get(row['metric'])
        if row['metric'] not in metrics or not p or not p.get('proposed'):
            continue
        old = row.get('bands')
        new = {'full': p['proposed']['full'], 'partial': p['proposed']['partial']}
        if old == new:
            continue
        row.setdefault('bands_history', []).append(
            {'until': today, 'bands': old, 'source': row.get('bands_source')})
        row['bands'] = new
        kf, kp = p.get('proposed_from') or ['?', '?']
        row['bands_source'] = f"常模 {kf}/{kp} (n={p.get('n_subjects')})，{today} {who}"
        changed.append({'label': row['label'], 'metric': row['metric'], 'old': old, 'new': new})
    if not changed:
        return {'changed': [], 'file': rp}
    SR.build_scoring(movement, rub, SR.subjects(movement, grade))   # 按新門檻重算分檔
    rub.setdefault('_changelog', []).append(
        {'date': today, 'by': who, 'n_subjects': prop.get('n_subjects'),
         'rows': [c['metric'] for c in changed]})
    tmp = rp + '.tmp'
    json.dump(rub, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    os.replace(tmp, rp)
    return {'changed': changed, 'file': rp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--movement', required=True)
    ap.add_argument('--grade', required=True)
    ap.add_argument('--metrics', required=True, help='逗號分隔')
    ap.add_argument('--by', default='admin 頁審批')
    a = ap.parse_args()
    r = approve(a.movement, a.grade, set(a.metrics.split(',')), a.by)
    print(json.dumps(r, ensure_ascii=False))


if __name__ == '__main__':
    main()
