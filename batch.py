#!/usr/bin/env python
"""成批入庫 —— 一條命令跑晒一疊錄影，自己處理標定重用同常模。

你只需要寫一張清單（每行：資料夾  動作key  年級），佢會自動：
  • 同一場次（rec_ 後面嗰個日期）只自標定一次，其餘全部重用 —— 快好多
  • 已經做過嘅自動跳過（要重做加 --force）
  • 邊條失敗就記低原因、繼續做下一條，唔會停晒
  • 全部完成之後自動砌常模 + 出準則對照

用法:
  python batch.py --list jobs.txt --roi rois/yau_yat_chuen_20260915.json

  # 或者唔寫清單，直接掃一個資料夾（入面每個 rec_* 都當同一個動作）
  python batch.py --scan ~/Downloads --movement k1_straight_run --grade K1 \\
      --roi rois/yau_yat_chuen_20260915.json

jobs.txt 格式（# 開頭 = 註解，空行跳過）:
  /Users/me/Downloads/rec_20260915_095915   k1_straight_run   K1
  /Users/me/Downloads/rec_20260915_095955   k1_straight_run   K1
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def session_of(rec_id):
    """rec_20260915_100054 → 20260915（同日同場，相機通常冇郁過）。"""
    # ⚠ 只計日期唔夠：同一日上午／下午分兩場拍，中間機位郁過（實測 2026-09-15
    # 上午 vs 下午，ORB 比對四台平移 121–476px，cam03 成個視角唔同），
    # 共用標定會整死成場。加半日標記，上下午各自自標定一次。
    # （場次內部穩定：上午最大累積爬移 12.7px、下午 ≤1.2px。）
    if rec_id in _SESSIONS:
        return _SESSIONS[rec_id]
    m = re.search(r'rec_(\d{8})_(\d{2})', rec_id)
    if m:
        return m.group(1) + ('am' if int(m.group(2)) < 12 else 'pm')
    m = re.search(r'rec_(\d{8})', rec_id)
    return m.group(1) if m else rec_id


_SESSIONS = {}


def load_sessions(path):
    """讀 session_scan.py 出嘅 {rec_id: session} —— 按機位穩定度分場。

    時鐘分唔到場：實測「之」字移動 39 條，兩段都喺中午前（09:4x / 11:3x）
    但機位差 603px；而 11:3x 嗰段**中途** cam02 郁咗（114540 跳 44px），
    時鐘點分都分唔到。共用錯標定 = 三角化全錯而且唔會報錯。
    """
    global _SESSIONS
    if path and os.path.exists(path):
        _SESSIONS = json.load(open(path))
        n = len(set(_SESSIONS.values()))
        print('▶ 用 %s：%d 條片 → %d 場\n' % (os.path.basename(path), len(_SESSIONS), n))


def existing_calib(data, movement, session):
    """搵返同一場次已經解出嘅標定。"""
    for d in sorted(glob.glob(os.path.join(data, movement, 'rec_*'))):
        cal = os.path.join(d, 'calibration.json')
        mp = os.path.join(d, 'meta.json')
        if not (os.path.exists(cal) and os.path.exists(mp)):
            continue
        try:
            if json.load(open(mp)).get('session') == session:
                return cal
        except Exception:
            pass
    return None


def read_jobs(a):
    jobs = []
    if a.list:
        for line in open(a.list, encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # 先試 tab 分隔 —— 錄影可能放喺外置硬碟，路徑有空格
            # （實測 /Volumes/BCM-00926/Game7 AI system Data/... 用空白分會斬爛條路徑）。
            parts = [p for p in line.split('\t') if p.strip()]
            if len(parts) < 3:
                parts = line.rsplit(None, 2)      # 由右邊斬：動作同年級冇空格
            if len(parts) < 3:
                print(f'⚠ 跳過（格式要「資料夾<TAB>動作<TAB>年級」）：{line}')
                continue
            jobs.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    elif a.scan:
        if not (a.movement and a.grade):
            sys.exit('✗ --scan 要同時畀 --movement 同 --grade')
        pat = f'rec_{a.date}_*' if a.date else 'rec_*'
        for d in sorted(glob.glob(os.path.join(os.path.expanduser(a.scan), pat))):
            if os.path.isdir(d) and glob.glob(os.path.join(d, 'cam*.mp4')):
                jobs.append((d, a.movement, a.grade))
        if not a.date:
            print('⚠ 冇寫 --date，會掃晒資料夾入面所有 rec_*（可能包住舊場次）\n')
    else:
        sys.exit('✗ 要 --list 或者 --scan')
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', default='', help='清單檔（資料夾 動作 年級）')
    ap.add_argument('--scan', default='', help='掃呢個資料夾入面嘅 rec_*')
    ap.add_argument('--date', default='',
                    help='只做呢日嘅錄影，例如 20260915。--scan 幾乎一定要用 —— '
                         '唔寫就會連舊場次（例如 8 月嗰批成人示範片）都掃埋落去')
    ap.add_argument('--movement', default='')
    ap.add_argument('--grade', default='')
    ap.add_argument('--roi', default='')
    ap.add_argument('--moving', type=int, default=0, metavar='N',
                    help='跑動型動作：每視角只保留郁得最多嘅 N 條軌跡，唔使框')
    ap.add_argument('--standing', type=float, default=0.0, metavar='R',
                    help='傳落 ingest：只留 2D 企姿 ≥ R 嘅人（可以同 --roi 一齊用或者單用）')
    ap.add_argument('--data', default=os.path.join(HERE, 'data'))
    ap.add_argument('--sessions', default='',
                    help='session_scan.py 出嘅 json；唔畀就用日期+am/pm')
    ap.add_argument('--start-trigger', default='hand_raise',
                    choices=['', 'hand_raise'],
                    help='預設 hand_raise（細路舉手先開始評測）')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--remeasure', action='store_true',
                    help='唔重跑影片，淨係用新門檻重算已入庫嘅指標（幾秒）')
    ap.add_argument('--min-lift', type=float, default=None,
                    help='「有冇做過動作」嘅膝抬門檻（預設 25°）')
    ap.add_argument('--no-norms', action='store_true')
    a = ap.parse_args()

    load_sessions(a.sessions)
    if a.remeasure:
        if not a.movement:
            sys.exit('✗ --remeasure 要畀 --movement')
        dirs = sorted(glob.glob(os.path.join(a.data, a.movement, 'rec_*')))
        if not dirs:
            sys.exit(f'✗ {a.movement} 未有任何已入庫嘅錄影')
        print(f'▶ 重算 {len(dirs)} 條已入庫錄影'
              + (f'（--min-lift {a.min_lift}）' if a.min_lift else '') + '\n')
        for d in dirs:
            mp = os.path.join(d, 'meta.json')
            if not os.path.exists(mp):    # 入庫失敗留低嘅空資料夾
                print(f'   ⚠ {os.path.basename(d)}: 冇 meta.json，跳過')
                continue
            meta = json.load(open(mp))
            cmd = [PY, os.path.join(HERE, 'ingest.py'), '--remeasure',
                   '--rec', d, '--roi', 'x', '--movement', a.movement,
                   '--grade', meta['grade'], '--out', a.data]
            if a.min_lift is not None:
                cmd += ['--min-lift', str(a.min_lift)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            for ln in r.stdout.splitlines():
                if '循環' in ln or '⚠' in ln or '✗' in ln:
                    print('   ' + ln.strip())
        print(f'\n{"="*60}')
        subprocess.run([PY, os.path.join(HERE, 'build_norms.py'),
                        '--movement', a.movement, '--data', a.data])
        rub = glob.glob(os.path.join(HERE, 'rubrics', f'{a.movement}_*.json'))
        rub = [x for x in rub if not x.endswith('.proposed.json')]
        if rub:
            g = os.path.basename(rub[0]).rsplit('_', 1)[1].split('.')[0]
            subprocess.run([PY, os.path.join(HERE, 'compare_rubric.py'),
                            '--movement', a.movement, '--grade', g,
                            '--data', a.data])
        return

    jobs = read_jobs(a)
    if not jobs:
        sys.exit('✗ 冇嘢做（清單空，或者資料夾入面冇 rec_*/cam*.mp4）')
    if not a.roi and not a.moving and not a.standing:
        sys.exit('✗ 入庫要 --roi（企定型動作）、--moving N（跑動型動作）或者 --standing R')

    print(f'▶ 一共 {len(jobs)} 條錄影')
    print('   每條大約：2D 檢測 1 分鐘 / 250 幀，加 3D 重建約影片長度 ×1.7，'
          '\n   該場第一條再加 0.5–1 分鐘自標定。已入庫嘅會即刻跳過。\n')
    ok, skipped, failed = [], [], []
    t0 = time.time()
    for i, (rec, movement, grade) in enumerate(jobs, 1):
        rec_id = os.path.basename(os.path.normpath(rec))
        out = os.path.join(a.data, movement, rec_id)
        head = f'[{i}/{len(jobs)}] {rec_id}  {movement}  {grade}'
        if os.path.exists(os.path.join(out, 'pose3d.npz')) and not a.force:
            print(f'{head}  —— 已經做過，跳過')
            skipped.append(rec_id)
            continue
        if not glob.glob(os.path.join(rec, 'cam*.mp4')):
            print(f'{head}  ✗ 搵唔到 cam*.mp4')
            failed.append((rec_id, '冇 cam*.mp4'))
            continue

        cal = existing_calib(a.data, movement, session_of(rec_id))
        print(f'{head}  ' + (f'（重用 {os.path.basename(os.path.dirname(cal))} 嘅標定）'
                             if cal else '（呢場第一條，要自標定）'))
        cmd = [PY, os.path.join(HERE, 'ingest.py'), '--rec', rec,
               '--movement', movement, '--grade', grade, '--out', a.data]
        if a.moving:
            cmd += ['--moving', str(a.moving)]
        elif a.roi:
            cmd += ['--roi', a.roi]
        if a.standing:
            cmd += ['--standing', str(a.standing)]
        if cal:
            cmd += ['--calib', cal]
        cmd += ['--session', session_of(rec_id)]
        if a.start_trigger:
            cmd += ['--start-trigger', a.start_trigger]
        if a.force:
            cmd += ['--force']
        # 邊跑邊出進度 —— 唔好用 capture_output，否則跑幾分鐘都一片空白，
        # 用嘅人分唔清係跑緊定係死咗機。
        t1 = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines, last = [], ''
        for ln in proc.stdout:
            ln = ln.rstrip()
            lines.append(ln)
            if ln.startswith('▶'):                       # 步驟
                last = ln.split('  ')[0]
                print(f'   {int(time.time()-t1):3d}s  {ln}', flush=True)
            elif ln.lstrip().startswith('cam') and ('幀' in ln or '快取' in ln):
                print(f'         {ln.strip()}', flush=True)
            elif '自標定完成' in ln or '標定計算中' in ln:
                print(f'   {int(time.time()-t1):3d}s  {ln.strip()}', flush=True)
            elif '幀、' in ln or '⚠' in ln or '⛔' in ln or ln.startswith('✗'):
                print(f'   {ln.strip()}', flush=True)
        proc.wait()
        took = time.time() - t1
        if proc.returncode == 0 and os.path.exists(os.path.join(out, 'pose3d.npz')):
            print(f'   ✓ {took/60:.1f} 分鐘')
            ok.append(rec_id)
        else:
            why = ([x for x in lines if x.strip()] or ['(冇輸出)'])[-1]
            print(f'   ✗ 失敗：{why}')
            failed.append((rec_id, why))

    mins = (time.time() - t0) / 60
    print(f'\n{"="*60}\n完成 {len(ok)} 條、跳過 {len(skipped)} 條、失敗 {len(failed)} 條'
          f'（{mins:.1f} 分鐘）')
    for rid, why in failed:
        print(f'  ✗ {rid}: {why}')

    if failed:
        print('\n最常見嘅失敗原因：自標定唔成功 —— 通常係表演區框太鬆（旁觀者入咗框）。'
              '用框選工具收窄個框再跑過。')

    if ok and not a.no_norms:
        combos = sorted({(m, g) for r, m, g in jobs
                         if os.path.basename(os.path.normpath(r)) in ok})
        for movement, grade in combos:
            print(f'\n{"="*60}')
            subprocess.run([PY, os.path.join(HERE, 'build_norms.py'),
                            '--movement', movement, '--grade', grade,
                            '--data', a.data])
            rub = os.path.join(HERE, 'rubrics', f'{movement}_{grade}.json')
            if os.path.exists(rub):
                subprocess.run([PY, os.path.join(HERE, 'compare_rubric.py'),
                                '--movement', movement, '--grade', grade,
                                '--data', a.data])


if __name__ == '__main__':
    main()
