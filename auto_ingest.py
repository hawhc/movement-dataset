#!/usr/bin/env python
"""admin 頁「評分＋同步錄影」→ 自動入庫（一條錄影）。

人手入庫（batch.py）最麻煩嘅三步，即時系統評分嗰陣已經做咗：
  * 自標定     → 用即時 session 當刻嘅標定（calibration.json，朝向已經 bake 入 R,T，
                 固定標定重建 = 同一個世界座標）
  * ROI        → 即時「計算區域」（四部相機圍出嘅地面矩形）投返落每台機（roi.json）
  * 邊個係表演者 → 評分席位（seats.json：每個席位跟住嘅人嘅髖部世界座標＋時間）

所以呢度可以全自動：
  1. ingest.py --calib --roi         20fps 重建（即時只得 ~4fps，唔可以直接入常模）
  2. floor_up（同一份標定嘅片一齊擬地面）
  3. 器材：欄杆 hurdle3d／平衡線 line3d／球 ball3d（按動作）
  4. ingest.py --remeasure           用地面＋器材重算指標
  5. 席位配對：離線軌跡 ↔ 即時席位（同一世界座標，比髖部位置）
       配到 → 帶學員 id；配唔到（旁觀者、老師）／同席碎片 → 寫入 seat_match.json 嘅
       exclude，build_norms 唔計（唔改 metrics.json，remeasure 都唔會冇咗）
  6. 逐位按 rubrics/<動作>_<年級>.json 計分（同 score_rubric 一樣）
  7. build_norms + compare_rubric --write → 建議門檻寫入 *.proposed.json（**唔會**改正式準則）

輸出：每步印 `STEP <名>`，最後印 `RESULT <json>`（backend 讀）。結果亦寫入
data/<動作>/<rec>/seat_match.json。

用法：python auto_ingest.py <rec 資料夾>
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
PY = sys.executable
LHIP, RHIP = 11, 12
MATCH_MAX = 0.6        # 髖部水平距離中位（世界單位）上限；同一標定、同一人通常 <0.2
MATCH_MIN_PAIRS = 5    # 至少幾多個時間點對得上先算
TRIGGER = 'hand_raise'  # 同 batch.py 預設；NO_TRIGGER 嘅動作 ingest 自己會忽略


def step(name, **kw):
    print('STEP ' + name + (' ' + json.dumps(kw, ensure_ascii=False) if kw else ''), flush=True)


def run(cmd):
    """跑子程序，輸出照轉（backend 會記入 job log）；失敗就拋最後一行。"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    last = ''
    for ln in p.stdout:
        ln = ln.rstrip()
        if ln:
            last = ln
            print('  ' + ln, flush=True)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f'{os.path.basename(cmd[1])} 失敗：{last}')


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()[:8]


def same_calib_dirs(movement, h):
    out = []
    for d in sorted(glob.glob(os.path.join(DATA, movement, 'rec_*'))):
        c = os.path.join(d, 'calibration.json')
        if os.path.exists(c) and os.path.exists(os.path.join(d, 'pose3d.npz')) and md5(c) == h:
            out.append(d)
    return out


def floor_up(movement, h):
    import floor_up as FU
    ds = same_calib_dirs(movement, h)
    r = FU.group_up(ds)
    if r is None:
        print('  ⚠ 腳踝點唔夠擬地面，改用逐位擬（原地類垂直軸可能歪）', flush=True)
        return
    n, npts, frac = r
    for d in ds:
        json.dump({'up': [float(x) for x in n], 'calib_md5': h, 'n_clips': len(ds),
                   'n_ankle_points': int(npts), 'inlier_frac': round(frac, 3)},
                  open(os.path.join(d, 'floor_up.json'), 'w'), indent=1)
    print(f'  地面：{len(ds)} 條片、{npts} 個腳踝點 → up {np.round(n, 3).tolist()}', flush=True)


def equipment(movement, out, rec, h):
    fam = __import__('ingest').FAMILY.get(movement, 'gait')
    if fam == 'hurdle_weave':
        import hurdle3d
        ds = same_calib_dirs(movement, h)
        r = hurdle3d.build_group(ds, os.path.dirname(rec))
        if r is None:
            print('  ⚠ 搵唔到欄杆 —— 「與欄杆距離」會量唔到', flush=True)
        else:
            for d in ds:
                np.savez(os.path.join(d, 'hurdles3d.npz'), **r)
    elif fam == 'line_walk':
        import line3d
        line3d.build(out, rec)
    elif fam == 'toss_catch':
        import ball3d
        ball3d.detect(rec, os.path.join(DATA, '_work', os.path.basename(rec), 'balls'), classes=(0, 1))
        ball3d.build(out)
    else:
        print('  （呢個動作唔使器材）', flush=True)


def match_seats(out, rec, meta_rec):
    """離線軌跡 ↔ 即時席位。回 (matches, exclude)。"""
    seats = json.load(open(os.path.join(rec, 'seats.json')))
    t0, fps = float(seats['t0']), float(seats['fps'])
    samples = seats['samples']                     # [t, slot, tid, x, y, z]
    z = np.load(os.path.join(out, 'pose3d.npz'), allow_pickle=True)
    meta = json.load(open(os.path.join(out, 'meta.json')))
    refs = [str(x) for x in z['subjects']]
    xyz, valid = z['xyz'], z['valid']
    T = xyz.shape[1]
    students = {int(k): v for k, v in (meta_rec.get('auto_dataset', {}).get('seat_students') or {}).items()}

    by_slot = {}
    for t, slot, _tid, x, y, zz in samples:
        by_slot.setdefault(int(slot), []).append((float(t), np.array([x, y, zz], float)))

    cand = []                                      # (median dist, ref, slot, pairs)
    for i, ref in enumerate(refs):
        k = meta.get('scale_corrections', {}).get(ref, 1.0) or 1.0
        hip = np.nanmean(xyz[i][:, [LHIP, RHIP]], 1) / k    # 還原 raw 世界座標
        for slot, pts in by_slot.items():
            d = []
            for t, p in pts:
                f = int(round((t - t0) * fps))
                if 0 <= f < T and valid[i, f] and np.isfinite(hip[f]).all():
                    d.append(float(np.hypot(hip[f][0] - p[0], hip[f][2] - p[2])))   # 水平（Y 向上）
            if len(d) >= MATCH_MIN_PAIRS:
                cand.append((float(np.median(d)), ref, slot, len(d)))

    best = {}                                      # ref → (dist, slot, pairs)
    for dist, ref, slot, n in sorted(cand):
        if dist <= MATCH_MAX and ref not in best:
            best[ref] = (dist, slot, n)
    matches, exclude = {}, {}
    for slot in sorted({s for _, s, _ in best.values()}):
        mine = sorted(((n, ref, dist) for ref, (dist, s, n) in best.items() if s == slot), reverse=True)
        n, ref, dist = mine[0]                     # 對得上最多時間點嗰條 = 主軌跡
        stu = students.get(slot)
        matches[ref] = {'slot': slot, 'student_id': (stu or {}).get('id'),
                        'student_name': (stu or {}).get('name'),
                        'median_dist': round(dist, 3), 'pairs': n}
        for n2, ref2, _ in mine[1:]:
            exclude[ref2] = f'席位 {slot + 1} 嘅碎片軌跡（主軌跡 {ref}），避免同一位細路計兩次'
    for ref in refs:
        if ref not in matches and ref not in exclude:
            exclude[ref] = '唔喺任何評分席位（旁觀者／老師／區內其他人）—— auto_ingest 席位配對'
    return matches, exclude


def score(movement, grade, out, matches):
    import build_norms as BN
    import score_rubric as SR
    rub = json.load(open(os.path.join(HERE, 'rubrics', f'{movement}_{grade}.json'), encoding='utf-8'))
    meta = json.load(open(os.path.join(out, 'meta.json')))
    mx = json.load(open(os.path.join(out, 'metrics.json')))
    raw = {s['subject_ref']: s for s in mx['subjects']}
    # subject_rows 會按 manual_excludes 剔人 —— 呢度要逐位計分（包括被剔嘅都唔要緊，
    # 我哋只計 matches），所以直接用佢嘅 cyc_ 彙總邏輯
    rows = {r['_ref']: r for r in BN.subject_rows([(meta, mx)])}
    res = []
    for ref, m in matches.items():
        r = rows.get(ref)
        if r is None:                             # 被 suspect 守門剔咗（體型／企姿）
            res.append({**m, 'ref': ref, 'total': None,
                        'why': raw.get(ref, {}).get('suspect_reason') or '未通過入庫守門'})
            continue
        r['did_perform'] = raw.get(ref, {}).get('did_perform')
        total, pts = SR.score_subject(rub, r)
        crit = []
        for row in rub['rows']:
            p = pts.get(row['label']) if pts else None
            v = r.get(row['metric'])
            crit.append({'label': row['label'], 'max': row['points'],
                         'score': 0 if p is None else round(float(p), 1),
                         'value': None if v is None else round(float(v), 2),
                         'counted': p is not None,
                         'detail': ('未做到動作 → 鼓勵分' if not pts else
                                    '量唔到（唔入分母）' if p is None else
                                    f"{row['metric']} = {v:g}")})
        res.append({**m, 'ref': ref, 'total': total, 'criteria': crit,
                    'did_perform': r['did_perform']})
    return res


def main():
    rec = os.path.abspath(sys.argv[1])
    rid = os.path.basename(rec)
    meta_rec = json.load(open(os.path.join(rec, 'rec_meta.json'), encoding='utf-8'))
    ad = meta_rec.get('auto_dataset') or {}
    movement, grade = ad.get('movement'), ad.get('grade', 'K1')
    if not movement:
        raise RuntimeError('rec_meta.json 冇 auto_dataset.movement —— 唔係評分錄影')
    calib, roi = os.path.join(rec, 'calibration.json'), os.path.join(rec, 'roi.json')
    for p in (calib, roi, os.path.join(rec, 'seats.json')):
        if not os.path.exists(p):
            raise RuntimeError(f'缺少 {os.path.basename(p)}')
    fps = ad.get('fps') or 20.0
    h = md5(calib)
    out = os.path.join(DATA, movement, rid)

    step('ingest', fps=fps)
    run([PY, os.path.join(HERE, 'ingest.py'), '--rec', rec, '--roi', roi, '--calib', calib,
         '--movement', movement, '--grade', grade, '--out', DATA, '--fps', str(fps),
         '--session', f'live_{h}', '--start-trigger', TRIGGER, '--force'])
    if not os.path.exists(os.path.join(out, 'pose3d.npz')):
        raise RuntimeError('ingest 冇出到 3D 骨架')

    step('floor_up')
    sys.path.insert(0, HERE)
    floor_up(movement, h)
    step('equipment')
    equipment(movement, out, rec, h)
    step('remeasure')
    run([PY, os.path.join(HERE, 'ingest.py'), '--remeasure', '--rec', out, '--roi', 'x',
         '--movement', movement, '--grade', grade, '--out', DATA, '--fps', str(fps)])

    step('match')
    matches, exclude = match_seats(out, rec, meta_rec)
    print(f'  配到 {len(matches)} 位、排除 {len(exclude)} 條軌跡', flush=True)
    step('score')
    scores = score(movement, grade, out, matches)
    doc = {'rec': rid, 'movement': movement, 'grade': grade, 'matches': matches,
           'exclude': exclude, 'scores': scores}
    json.dump(doc, open(os.path.join(out, 'seat_match.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    step('norms')
    try:
        run([PY, os.path.join(HERE, 'build_norms.py'), '--movement', movement, '--grade', grade])
        run([PY, os.path.join(HERE, 'compare_rubric.py'), '--movement', movement,
             '--grade', grade, '--write'])
    except Exception as exc:                  # 常模出錯唔影響今次入庫／計分
        print(f'  ⚠ 常模／建議門檻未更新：{exc}', flush=True)
    print('RESULT ' + json.dumps({'ok': True, 'out': out, **doc}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print('RESULT ' + json.dumps({'ok': False, 'error': str(e)}, ensure_ascii=False), flush=True)
        sys.exit(1)
