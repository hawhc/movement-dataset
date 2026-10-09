#!/usr/bin/env python
"""一條命令由 4-Cam 錄影資料夾整出一筆 dataset：骨架 xyz + 逐項指標。

管線（每一步都有快取，可以中途停咗再跑）：
  cam*.mp4 → YOLO11m-pose 2D → ROI 過濾（剔走旁觀者）→ 自標定 K,R,T
  → 逐幀對極關聯 + 三角化 → 每位受測者嘅 17 點 3D 骨架 → 指標

輸出 data/<movement_key>/<rec_id>/：
  pose3d.npz    xyz (S,T,17,3) 世界座標公尺、xyz_norm（髖為原點／肩寬正規化）
  meta.json     場次、動作、年級、標定品質、尺度修正係數
  metrics.json  逐位受測者嘅指標 + 逐個步態循環嘅明細（畀 build_norms.py 用）
  calibration.json

用法:
  python ingest.py --rec <rec 資料夾> --roi rois/<場地>.json \\
      --movement k1_straight_run --grade K1
  # 同場第二條片起，重用第一條嘅標定（相機冇郁過）：
  python ingest.py --rec <rec2> --roi ... --movement ... --grade K1 \\
      --calib data/k1_straight_run/<rec1>/calibration.json
"""
import argparse
import json
import os
import shutil
import sys
import time

import warnings

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)   # 全 NaN 幀好正常

HERE = os.path.dirname(os.path.abspath(__file__))
SELFCALIB = '/Users/bcm01032/game7/human-selfcalib'
CHECKERBOARD = os.path.join(SELFCALIB, 'calibs', 'tapo_c120_checkerboard.json')

# COCO-17
NOSE, LSH, RSH, LEL, REL, LWR, RWR = 0, 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKN, RKN, LAN, RAN = 11, 12, 13, 14, 15, 16
# 各年級肩寬（cm）—— 同 movement_engine.SHOULDER_CM 一致。
# 管線嘅公制尺度係成人骨長先驗定出，對幼童會偏大（實測 K1 讀到 ~40cm），
# 所以所有距離類指標都要乘返 grade_cm / 實測肩寬。角度類唔受影響。
SHOULDER_CM = {'K1': 26.0, 'K2': 28.0, 'K3': 30.0}
# 動作家族 —— 決定用邊套指標。預設 'gait'（步態：膝抬／步頻／速度）。
# 每個動作嘅準則唔同，唔可以一套指標行天下：深蹲跳用膝抬做前置閘同切循環
# 會錯判（實測 14 條片 8 位真表演者被當成旁觀者）。
FAMILY = {
    'k3_squat_jump_with_ball': 'squat_jump',
    'k3_zigzag_move': 'zigzag',
    'k3_jump_rope_continuous': 'rope_jump',
    'k1_balance_line_walk': 'line_walk',
    'k3_walking_leg_swing_stretch': 'leg_swing',
    'k1_squat': 'squat',
    'k1_breath_lower_stretch': 'fold_stretch',
    'k1_hurdle_weave': 'hurdle_weave',
    'k1_command_audio': 'command',
    'k1_toss_catch': 'toss_catch',
}
# 起計訊號：跑步要等舉手，但深蹲跳係孖手捧球，捧球過頭會被當成舉手
# （實測 14 條入面 13 條「搵到舉手」，全部係假觸發）。
NO_TRIGGER = {'k3_squat_jump_with_ball', 'k3_zigzag_move', 'k3_jump_rope_continuous',
              'k3_walking_leg_swing_stretch', 'k1_balance_line_walk',
              'k3_balance_line_one_hand_dribble', 'k1_hurdle_weave', 'k1_squat',
              'k1_toss_catch', 'k1_command_audio', 'k1_breath_lower_stretch'}
# 企住嘅最低 (髖高 − 踝高) ÷ 軀幹長，**比嘅係 p90 唔係中位**（見 measure()）。
# 實測跑步真表演者中位 1.36~1.54；坐喺地下睇嘅細路腳踝重建唔到，讀 NaN。
STAND_MIN = 1.0
# 原地類家族：表演者全程喺場，碎片軌跡可以用幀數判
IN_PLACE = {'rope_jump', 'jump_turn'}


# ---------------------------------------------------------------- 2D 檢測
def detect(rec, cache, model='yolo11m-pose.pt', imgsz=960):
    """四路全片跑 YOLO11-pose，存成 detect JSON（有快取就跳過）。"""
    import glob
    import cv2
    os.makedirs(cache, exist_ok=True)
    vids = sorted(glob.glob(os.path.join(rec, 'cam*.mp4')))
    if not vids:
        sys.exit(f'✗ 資料夾入面搵唔到 cam*.mp4：{rec}')
    todo = [p for p in vids
            if not os.path.exists(os.path.join(
                cache, os.path.splitext(os.path.basename(p))[0] + '.json'))]
    if todo:
        from ultralytics import YOLO
        m = YOLO(os.path.join(SELFCALIB, model))
    for p in vids:
        name = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(cache, name + '.json')
        if os.path.exists(dst):
            print(f'  {name}: 用返檢測快取')
            continue
        cap = cv2.VideoCapture(p)
        W, H = int(cap.get(3)), int(cap.get(4))
        info, fid = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            r = m.predict(fr, imgsz=imgsz, verbose=False, device='mps')[0]
            insts = []
            if r.keypoints is not None and r.keypoints.xy is not None:
                xy = r.keypoints.xy.cpu().numpy()
                cf = (r.keypoints.conf.cpu().numpy()
                      if r.keypoints.conf is not None else np.ones(xy.shape[:2]))
                insts = [{'keypoints': k.tolist(), 'keypoint_scores': s.tolist()}
                         for k, s in zip(xy, cf)]
            info.append({'frame_id': fid, 'instances': insts})
            fid += 1
        cap.release()
        json.dump({'meta': {'model': model, 'video': name, 'W': W, 'H': H},
                   'instance_info': info}, open(dst, 'w'))
        print(f'  {name}: {fid} 幀')
    return cache


# ---------------------------------------------------------------- 3D 重建
def _session(rec_dir, out_dir, fixed=None, loop=True, speed=1.0, budget=180):
    """跑 human-selfcalib 嘅 ComputeSession（管線零改動，只換輸入來源）。"""
    if SELFCALIB not in sys.path:
        sys.path.insert(0, SELFCALIB)
    cwd = os.getcwd()
    os.chdir(SELFCALIB)                     # YOLO 權重相對路徑以 repo 根為準
    try:
        from recorder.capture import VideoRig
        from recorder.compute import ComputeSession, load_calib_file
        rig = VideoRig(rec_dir, loop=loop, speed=speed)
        intr = load_calib_file(CHECKERBOARD)
        sess = ComputeSession(
            rig.workers, out_dir, det_long=960, intr=intr,
            fixed_calib=(load_calib_file(fixed) if fixed else None),
            layout=True, same_model=True, reid=True,
            refine=fixed is None, aggressive=fixed is None, log=print)
        fidmap = {}
        orig = sess._process

        def patched(fid, frames, vfids, model):
            fidmap[fid] = [None if v is None else int(v) for v in vfids]
            return orig(fid, frames, vfids, model)
        sess._process = patched

        rig.start()
        sess.start()
        t0 = time.time()
        while time.time() - t0 < budget:
            time.sleep(3)
            if not loop and rig.finished:
                break
            if loop and sess.calib is not None:
                break                        # 標定階段：一拿到解就收工
        time.sleep(2)
        sess.stop()
        rig._stop.set()                      # VideoRig 冇 .stop()
        if sess._thread:
            sess._thread.join(180)           # 唔 join 就唔會寫 poses3d.json
        return sess.calib is not None, fidmap
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------- 指標
def _ang(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.nan if n < 1e-9 else np.degrees(
        np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


MIN_CYCLE_FRAMES = 7       # 一個循環至少要咁多幀先量得準（20fps 下 = 0.35s）


def _cycles(sig, fps, lo=0.15, hi=1.5, amp_frac=0.35):
    """由「膝抬高」訊號切步態循環：升穿中位線 = 一步嘅起點。

    回傳 (循環清單, 短過取樣極限而掉棄嘅數目)。

    ⚠ 一定要加幅度閘：片頭片尾企喺度嗰陣，訊號喺中位線附近抖，
    淨計過線會切出一大堆「假循環」（實測 16 秒切到 17 個，
    當中一半峰值只有 16°，而真正跑嗰陣係 45°+），逐循環嘅統計就會被溝淡。

    ⚠ 仲要有取樣下限（2026-09-16 加）。循環邊界靠正負號轉變定位，所以
    每個邊界本身就有 ±1 幀誤差；20fps 下一個 0.30s 嘅循環得 6 幀，單單
    量化誤差已經係 ±17%，`cycle_cv_pct` 就算跑得好穩都會讀到幾十 %。
    實測朝早五位報 150–200 步/分嘅細路，個個 cv 同時係 46–75% —— 步頻同
    變異度一齊爆高，正正係循環短過取樣極限嘅簽名，唔係真係跑得快。
    掉棄之後佢哋可能一個循環都唔剩，`cadence_spm` 出 None ——
    咁樣係啱嘅：20fps 真係量唔到，寧願唔出數都好過出個假數污染常模。
    要量到咁快就要提高拍攝幀率。
    """
    ok = np.isfinite(sig)
    if ok.sum() < 20:
        return [], 0
    lo = max(lo, MIN_CYCLE_FRAMES / fps)
    mid = np.nanmedian(sig)
    span = np.nanpercentile(sig, 95) - mid
    floor = mid + amp_frac * span                # 峰值至少要升到咁高先算一步
    cr = np.where(np.diff(np.sign(np.nan_to_num(sig - mid, nan=0.0))) > 0)[0]
    out, short = [], 0
    for a, b in zip(cr[:-1], cr[1:]):
        d = (b - a) / fps
        if np.nanmax(sig[a:b + 1]) < floor:
            continue
        if d < lo:
            short += 1                           # 有幅度但短過取樣極限
        elif d <= hi:
            out.append((int(a), int(b), float(d)))
    return out, short


def find_hand_raise(K, up, fps, hold=0.15, margin=0.2):
    """「細路舉手就開始評測」—— 搵手腕升過肩線嗰一刻。

    回傳 (舉手幀, 開始評測幀 = 舉手幀 + margin)。唔切走前面嗰段嘅話，
    行入場、老師調位嗰幾秒會污染每一條指標（尤其係「髖部離起點位移」
    —— 起點會變咗場邊而唔係定位點）。

    ⚠ 唔好等「手放低」先開始。實測 rec_20260915_095915 有個細路舉手舉足
    5.9 秒（117 幀），期間已經開始跑；等手放低就會切走成段動作。
    舉手本身就係開始訊號。

    搵唔到就回傳 (None, None)，由呼叫者決定用成條片。
    """
    def h(p):
        return (p * up).sum(-1)

    sh = h(np.nanmean(K[:, [LSH, RSH]], 1))
    wr = np.nanmax(np.stack([h(K[:, LWR]), h(K[:, RWR])]), axis=0)
    hi = np.nan_to_num(wr - sh, nan=-1.0) > 0.0          # 手腕高過肩
    need = max(2, int(round(hold * fps)))
    run = 0
    for i, x in enumerate(hi):
        run = run + 1 if x else 0
        if run >= need:
            raised = i - need + 1
            return int(raised), int(min(raised + round(margin * fps),
                                        len(hi) - 1))
    return None, None


def find_run_start(hip_h, up, fps, frac=0.30):
    """跑動型動作：由「真係開始跑」嗰刻起計。

    實測 rec_20260915_092821（K3 敏捷梯）：細路頭 7 秒慢慢行埋去起跑點，
    之後先衝。唔切走嗰段，「最長停頓」會讀到 5.85 秒 —— 量緊行埋去，
    唔係停頓。門檻用「自己峰值速度嘅 30%」，唔用絕對值，因為原地跑
    （髖速度 p50 0.07–0.14 m/s）同衝刺（1.23 / 峰值 5.03）差 10 倍以上。
    """
    v = np.linalg.norm(np.diff(hip_h, axis=0), axis=1) * fps
    if not np.isfinite(v).any():
        return None
    pk = float(np.nanpercentile(v, 95))
    if pk < 0.8:                       # 原地型動作，根本冇「起跑」可言
        return None
    hit = np.where(np.nan_to_num(v, nan=0.0) >= frac * pk)[0]
    return int(hit[0]) if len(hit) else None


def _load_ball(out_dir, ref, scale):
    """讀該位受測者嘅球 3D 軌跡（ball3d.py build 出嘅）。冇就回傳 None。

    ball3d.npz 存嘅已經係乘返受測者 scale 之後嘅座標，同 pose3d.npz 嘅 xyz
    同一個空間，可以直接相減。
    """
    p = os.path.join(out_dir, 'ball3d.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    subs = [str(x) for x in z['subjects']]
    if ref not in subs:
        return None
    i = subs.index(ref)
    return z['ball'][i], z['n_cams'][i]


def _up_from_floor(K, up0, thr=0.04, iters=300, seed=0):
    """由腳踝點擬合地面，法向量 = 真垂直。擬合唔到就回傳 None（用返軀幹方向）。

    點解要換：原本嘅 up 係「軀幹方向」（肩−髖嘅平均）。軀幹唔係垂直基準 ——
    細路前傾、蹲低、身體本身就歪，平均落嚟同真垂直可以差十幾度。一差咗，
    水平位移就會被投影成假高度：差 10°、行開 3 米 = 52cm 假「升幅」。
    實測 rec_20260915_104407#1 讀到髖高 166cm、踝高 69cm（真人企喺平地），
    連帶 ΔY_hip 報 37cm（幼童唔可能）。

    腳踝喺地面嘅時間佔絕大多數，跳起同坐低嗰啲係少數離群點 —— 正好用 RANSAC。
    """
    P = K[:, [LAN, RAN], :].reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    if len(P) < 60:
        return None
    rng = np.random.default_rng(seed)
    best_n, best_k, best_p = None, -1, None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        nv = np.cross(b - a, c - a)
        nn = np.linalg.norm(nv)
        if nn < 1e-9:
            continue
        nv = nv / nn
        k = int((np.abs((P - a) @ nv) < thr).sum())
        if k > best_k:
            best_n, best_k, best_p = nv, k, a
    if best_n is None or best_k < 0.4 * len(P):
        return None
    inl = P[np.abs((P - best_p) @ best_n) < thr]
    # 退化檢查：腳踝全部企喺同一點／一條線，個平面就靠估
    ev = np.linalg.svd(inl - inl.mean(0), compute_uv=False)
    if len(inl) < 50 or ev[1] < 0.15 * ev[0]:
        return None
    u, sv, vt = np.linalg.svd(inl - inl.mean(0))
    up = vt[2] / np.linalg.norm(vt[2])
    if np.dot(up, up0) < 0:                 # 指返向上
        up = -up
    if np.degrees(np.arccos(np.clip(np.dot(up, up0), -1, 1))) > 40:
        return None                          # 同軀幹方向差太遠 —— 唔信
    return up


def _ball_drops(bxyz, wrist, hip, h, torso, fps, off, idx, bn=None):
    """掉球判定 —— 用物理，唔用分佈。

    點解唔用百分位：本批 58 位**冇一個真係跌波**（準則「掉球 0–12 分」嗰層零樣本）。
    如果由呢批數據嘅百分位反推門檻，將來真係有細路跌波就會判錯 —— 佢會被一堆
    冇跌波嘅人拉住，照樣攞到中間分。所以「掉球」必須係絕對嘅事件判定。

    放低（做得唔標準，但唔係跌）同跌波（失去控制）嘅物理分別：
      放低 —— 球**跟住隻手落到近地面**先離手，球同手嘅下墜速度差唔多。
      跌波 —— 球**仲喺高處（高過髖）就脫手**，之後自己加速落，手追唔到。

    所以一段「唔喺手」嘅時間要同時符合：
      (a) 脫手嗰刻球仲高過髖 —— 排除「蹲低放喺地下／筒上」；
      (b) 之後 0.5 秒內球嘅下墜速度 > DROP_FALL 軀幹長/秒，而同一刻隻手落得
          慢過球嘅一半 —— 球自己跌，唔係跟住隻手落。

    門檻全部用軀幹長正規化，唔受未定案嘅公制錨點影響。追蹤斷咗嗰啲幀當「唔知」，
    唔當跌波。只計跳躍循環嘅幀 —— 循環之間放低個球去排隊唔關「持球」事。

    ⚠ 呢個判定係喺零正樣本之下寫嘅，靠物理同合成測試驗證（見 tests/）。
    第一次真係影到細路跌波，要攞返嗰條片核一次。
    """
    DROP_FALL, MIN_S, WIN_S, MAX_SPD = 3.0, 0.15, 0.5, 20.0
    if len(idx) < 5:
        return {'ball_drop_events': None, 'ball_max_away_torso': None}
    n = len(wrist)
    b = np.full((n, 3), np.nan)
    m = idx[(idx + off) < len(bxyz)]
    b[m] = bxyz[m + off]
    ok = np.isfinite(b).all(1) & np.isfinite(wrist).all(1)
    d = np.full(n, np.nan)
    d[ok] = np.linalg.norm(b[ok] - wrist[ok], axis=1) / torso
    held = d < 1.0
    hb, hw, hp = (np.full(n, np.nan) for _ in range(3))
    hb[ok], hw[ok] = h(b[ok]), h(wrist[ok])
    ohp = np.isfinite(hip).all(1)
    hp[ohp] = h(hip[ohp])
    vb, vw = np.full(n, np.nan), np.full(n, np.nan)
    vb[1:] = -(hb[1:] - hb[:-1]) * fps / torso        # 正 = 向下
    vw[1:] = -(hw[1:] - hw[:-1]) * fps / torso
    inrep = np.zeros(n, bool)
    inrep[idx[idx < n]] = True
    drops, away, i, gapped = 0, 0.0, 0, 0
    need, win = max(2, int(MIN_S * fps)), max(2, int(WIN_S * fps))
    while i < n:
        if held[i] or not np.isfinite(d[i]) or not inrep[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and np.isfinite(d[j + 1]) and not held[j + 1] and inrep[j + 1]:
            j += 1
        if (j - i + 1) >= need:
            away = max(away, float(np.nanmax(d[i:j + 1])))
            # 「脫手嗰陣仲喺高處」要睇**脫手之前**嘅高度，唔可以睇 i 嗰刻。
            # 合成測試捉到：自由落體要跌夠 1 個軀幹長先算「唔喺手」，
            # 而跌到嗰陣個球已經喺髖以下，用 hb[i] 判就永遠唔會響。
            k0 = max(0, i - max(1, int(0.3 * fps)))
            with np.errstate(invalid='ignore'):
                high = bool((np.nan_to_num(hb[k0:i + 1], nan=-9) >
                             np.nan_to_num(hp[k0:i + 1], nan=9)).any())
            w2 = slice(i, min(i + win, n))
            with np.errstate(invalid='ignore'):
                fall = bool(((np.nan_to_num(vb[w2], nan=0) > DROP_FALL) &
                             (np.nan_to_num(vb[w2], nan=0) >
                              2 * np.nan_to_num(vw[w2], nan=0))).any())
            # 追蹤連續性守門：整段（連 0.3s 回望）要幀幀有球、而且有 ≥2 台機支持。
            # 冇呢條就會把「追蹤跳格」當成跌波 —— 實測 rec_20260915_104407#0 f123：
            # 三角化嘅球一幀跳咗去細路腰部，前後兩幀係 nan，真球仲喺手邊，
            # 但個判定照響。零正樣本之下寧願漏報，唔好報假。
            w3 = slice(k0, min(i + win, n))
            cont = bool(np.isfinite(hb[w3]).all())
            # 物理速度守門：個球一幀之間飛唔到咁遠。關聯跳咗去另一個細路
            # 嗰陣速度會去到幾十個軀幹長/秒（實測 rec_20260915_103947#0 f256：
            # 追蹤到嘅球一幀由自己手上跳咗去前景另一位細路度）。
            # 真球最快都唔會過 MAX_SPD。
            bs = b[w3]
            if cont and len(bs) > 1:
                st = np.linalg.norm(np.diff(bs, axis=0), axis=1) * fps / torso
                cont = bool(np.nanmax(st) <= MAX_SPD)
            if bn is not None:
                seg_n = bn[(np.arange(w3.start, w3.stop) + off).clip(0, len(bn) - 1)]
                cont = cont and bool((seg_n >= 2).all())
            if high and fall and cont:
                drops += 1
            elif high and fall:
                gapped += 1
        i = j + 1
    return {'ball_drop_events': int(drops),
            'ball_drop_rejected_gap': int(gapped),
            'ball_max_away_torso': _r(float(away)) if away else 0.0}


def _load_cones(out_dir):
    """讀該場嘅雪糕筒 3D（cone3d.py build 出嘅），未修正尺度嘅世界座標。"""
    p = os.path.join(out_dir, 'cones3d.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    if not len(z['xyz']):
        return None
    r = z['radius'] if 'radius' in z.files else np.full(len(z['xyz']), np.nan)
    return z['xyz'], r


def _zigzag(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
            off, stance, stance_peak, raise_f, start_f, start_trigger,
            min_travel, cones=None, cone_scale=1.0):
    """K3「之」字移動（向前＋向後）—— 照《K3 優化版》第 2 條五條指標。

    | 指標           | 準則計算方式        | K3 良好標準      | 分 |
    |----------------|---------------------|------------------|----|
    | 面向目標       | θ_orient            | 多數時間面向目標 | 25 |
    | 不踩雪糕筒     | IsContact(Foot,Cone)| 全程無碰撞       | 25 |
    | 身體直線度     | ΔX_hip              | < 9 cm           | 20 |
    | 步伐穩定性     | 步幅差異率          | < 12%            | 15 |
    | 有方向的流暢度 | 面向目標下 t_pause  | < 0.25 s         | 15 |

    三個判讀決定（docx 冇寫死，要記住）：

    1. **「身體直線度」唔係「路徑貼唔貼近直線」**。之字路線本身就係彎嘅，
       量偏離直線冇意義。依據說明寫「減少左搖右擺」，所以量嘅係髖相對
       **自己平滑後路徑**嘅橫向抖動（去趨勢），唔係相對一條直線。
    2. **「面向目標」量一致性**。呢個動作係向前行完再向後行，身體應該
       一路面住同一邊。邊一邊係「目標」docx 冇講，所以用佢自己嘅
       中位朝向做基準，計有幾多幀維持喺該朝向 ±45° 內。
    3. **「不踩雪糕筒」要筒嘅 3D 位置** —— 未有就出 null，唔好靠估。

    尺度：所有長度類讀數都 ÷腿長（公制錨點未定案，見 rubrics 檔 _scale_caveat）。
    """
    shank = np.nanmedian(np.linalg.norm(K[:, LKN] - K[:, LAN], axis=1))
    leg = float(thigh + shank)
    hz = hip - np.outer(h(hip), up)                  # 髖喺水平面嘅位置
    ok = np.isfinite(hz).all(1)
    n = len(hz)

    # ---- 行程 + 前置閘 ----
    travel = np.nan
    if ok.sum() >= 5:
        P = hz[ok] - hz[ok].mean(0)
        _, _, vt = np.linalg.svd(P, full_matrices=False)
        axis = vt[0] / np.linalg.norm(vt[0])          # 之字走廊嘅主軸
        along = P @ axis
        travel = float(along.max() - along.min())
    else:
        axis = np.array([1.0, 0.0, 0.0])
        along = np.zeros(0)
    travel_ratio = travel / leg if leg > 1e-6 else np.nan
    did = bool(np.isfinite(travel_ratio) and travel_ratio >= min_travel)

    # ---- 面向目標 ----
    fwd = np.cross(np.broadcast_to(up, (n, 3)), K[:, RSH] - K[:, LSH])
    fwd = fwd - np.outer((fwd * up).sum(-1), up)
    nn = np.linalg.norm(fwd, axis=1, keepdims=True)
    fwd = np.divide(fwd, np.where(nn > 1e-9, nn, np.nan))
    okf = np.isfinite(fwd).all(1)
    facing_pct = orient_med = None
    ref = None
    if okf.sum() >= 10:
        ref = np.nanmedian(fwd[okf], 0)
        ref = ref / max(np.linalg.norm(ref), 1e-9)
        ang = np.degrees(np.arccos(np.clip((fwd[okf] * ref).sum(-1), -1, 1)))
        facing_pct = _r(100.0 * float((ang <= 45).mean()))
        # 朝向同走廊主軸嘅夾角（0° = 面住走廊盡頭）
        a2 = np.degrees(np.arccos(np.clip(abs(float(ref @ axis)), -1, 1)))
        orient_med = _r(float(np.median(ang)))
        facing_axis_deg = _r(a2)
    else:
        facing_axis_deg = None

    # ---- 身體直線度：去趨勢後嘅橫向抖動 ----
    sway = None
    if ok.sum() >= int(0.6 * fps):
        lat = P @ np.array([-axis[1], axis[0], 0.0] if abs(axis[2]) < 0.9
                           else [1.0, 0.0, 0.0])
        w = max(3, int(0.5 * fps) | 1)               # 0.5 秒平滑 = 路徑本身
        ker = np.ones(w) / w
        base = np.convolve(lat, ker, mode='same')
        m = slice(w, len(lat) - w) if len(lat) > 2 * w else slice(None)
        res = lat[m] - base[m]
        if len(res) > 5:
            sway = _r(100.0 * float(np.percentile(np.abs(res), 90)) / leg)

    # 「左搖右擺」嘅另一個讀法：軀幹左右傾角（同深蹲跳嗰條同一個量法）。
    # docx 寫 ΔX_hip 但依據說明寫「減少左搖右擺」，兩個讀法差好遠，
    # 兩個都出，畀人揀（見 rubrics 檔）。
    ml = K[:, RSH] - K[:, LSH]
    ml = ml - np.outer((ml * up).sum(-1), up)
    nml = np.linalg.norm(ml, axis=1, keepdims=True)
    ml = np.divide(ml, np.where(nml > 1e-9, nml, np.nan))
    tv = sh - hip
    tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    with np.errstate(invalid='ignore'):
        lat_deg = np.degrees(np.arctan2(np.abs((tv * ml).sum(-1)),
                                        (tv * up).sum(-1)))
    trunk_lat = (_r(float(np.nanpercentile(lat_deg, 90)))
                 if np.isfinite(lat_deg).any() else None)
    # 「左搖右擺」係**擺動**，唔淨係「幾傾」——所以另外出一個有正負號嘅擺幅
    # （左傾為正、右傾為負，p90−p10）。傾定一邊 ≠ 搖擺，兩個讀數分得開。
    with np.errstate(invalid='ignore'):
        lat_signed = lat_deg * np.sign((tv * ml).sum(-1))
    trunk_lat_range = (_r(float(np.nanpercentile(lat_signed, 90)
                                - np.nanpercentile(lat_signed, 10)))
                       if np.isfinite(lat_signed).any() else None)

    # ---- 步伐穩定性 ----
    # docx 要「步幅差異率 < 12%」，但全片計落 39 位**冇一個**做得到（p10 已經 56%）。
    # 原因唔係細路做得差：呢個動作有向前、向後、仲要繞筒轉向，轉向嗰兩三步
    # 天然就短，同直線跑嘅步幅冇得比。所以分開三個讀數：
    #   stride_cv_pct         全片（對照用，同 docx 直接比就係呢個）
    #   stride_cv_seg_pct     只計「同一方向連續行進」嘅段落 —— 最貼 docx 原意
    #   stride_time_cv_pct    步伐**時間**一致性（節奏），唔受轉向步幅短影響
    # 行進段落：沿走廊主軸嘅速度方向一致、而且夠快（>0.2 × p75）、持續 ≥0.5s。
    seg_id = np.full(n, -1, int)
    if ok.sum() >= int(0.6 * fps) and len(along):
        al = np.full(n, np.nan)
        al[np.where(ok)[0]] = along
        v = np.full(n, np.nan)
        v[1:] = np.diff(al) * fps
        sp = np.abs(v)
        thr = 0.20 * float(np.nanpercentile(sp, 75)) if np.isfinite(sp).any() else 0
        sg = np.where(np.isfinite(v) & (sp > thr), np.sign(v), 0).astype(int)
        i, k2 = 0, 0
        while i < n:
            if sg[i] == 0:
                i += 1
                continue
            j = i
            while j + 1 < n and sg[j + 1] == sg[i]:
                j += 1
            if (j - i + 1) >= int(0.5 * fps):
                seg_id[i:j + 1] = k2
                k2 += 1
            i = j + 1

    stride_cv = stride_cv_rob = stride_cv_seg = stride_time_cv = None
    strides, strides_seg, step_dt = [], [], []
    for an in (LAN, RAN):
        ah = h(K[:, an])
        pos = K[:, an] - np.outer(h(K[:, an]), up)
        last = None
        for i in range(2, n - 2):
            wnd = ah[i - 2:i + 3]
            if not np.isfinite(wnd).all() or ah[i] != wnd.min():
                continue
            if wnd.max() - wnd.min() < 0.02 * leg:
                continue
            if last is not None and np.isfinite(pos[i]).all() \
                    and np.isfinite(pos[last]).all():
                d = float(np.linalg.norm(pos[i] - pos[last]))
                if 0.05 * leg < d < 2.0 * leg:
                    strides.append(d)
                    # 兩次落點都喺同一個行進段落先計入「段內步幅」
                    if seg_id[i] >= 0 and seg_id[i] == seg_id[last]:
                        strides_seg.append(d)
                        step_dt.append((i - last) / fps)
            last = i
    if len(strides) >= 4:
        stride_cv = _r(100.0 * float(np.std(strides) / np.mean(strides)))
        # 穩健版：IQR/1.349 ÷ 中位。原版受轉向同停低嗰兩三步嘅極端值拉高 ——
        # 呢個動作有向前向後同繞筒轉向，步幅天然唔平均，docx 個 12%
        # 應該係量直線跑嗰種步態，兩邊唔同嘢（實測 39 位原版 p10 56%、p90 81%）。
        q1, q3 = np.percentile(strides, [25, 75])
        med = float(np.median(strides))
        if med > 1e-9:
            stride_cv_rob = _r(100.0 * float((q3 - q1) / 1.349) / med)
    if len(strides_seg) >= 4:
        stride_cv_seg = _r(100.0 * float(np.std(strides_seg) / np.mean(strides_seg)))
    if len(step_dt) >= 4:
        stride_time_cv = _r(100.0 * float(np.std(step_dt) / np.mean(step_dt)))

    # ---- 有方向的流暢度：**面向目標期間**嘅最長停頓 ----
    # docx 寫明係「面向目標下 t_pause」，唔係全程 t_pause。冇呢個限制嘅話，
    # 細路轉身背住目標時嗰陣慢落嚟都會計入停頓 —— 嗰個唔關「有方向的流暢度」事。
    pause = pause_all = None
    if did and ok.sum() >= 5:
        spd = np.full(n, np.nan)
        spd[1:] = np.linalg.norm(np.diff(hz, axis=0), axis=1) * fps / leg
        if len(strides) and np.isfinite(spd).any():
            thr = 0.20 * float(np.nanpercentile(spd, 75))
            slow = np.isfinite(spd) & (spd < thr)

            def _longest(mask):
                run = best = 0
                for x in mask:
                    run = run + 1 if x else 0
                    best = max(best, run)
                return best / fps

            pause_all = _r(_longest(slow))
            facing_mask = np.ones(n, bool)
            if okf.sum() >= 10:
                ang_all = np.full(n, np.nan)
                ang_all[okf] = np.degrees(np.arccos(np.clip(
                    (fwd[okf] * ref).sum(-1), -1, 1)))
                facing_mask = np.nan_to_num(ang_all, nan=180.0) <= 45
            pause = _r(_longest(slow & facing_mask))

    # ---- 不踩雪糕筒（準則 25 分）----
    # 筒係靜止 3D 點（cone3d.py）。兩件事要處理：
    #  1. **分「之字路線嘅筒」同「貼牆邊界嘅筒」** —— 課室四圍都有筒，唔可以全部當
    #     障礙。用表演者行經路徑反推：離路徑 1.5 × 腿長之內先算課程用嘅筒。
    #  2. 踩唔踩到 —— 腳踝水平距離細過 CONE_R × 腿長就當接觸，連續幀算一次。
    # 門檻用腿長正規化（公制錨點未定案）。⚠ 呢層好可能係零樣本，要用幾何事件判定，
    # 唔可以用百分位 —— 同 ball 掉球一樣，見 tests/。
    # 接觸半徑 = **筒自己嘅底半徑**（由偵測框闊 × 深度 ÷ 焦距量返，唔係憑空估）
    # + 腳掌容差。實測筒底半徑 16.0cm（重建世界）；乘返個波定出嚟嘅世界尺度
    # ×0.46 = 7.4cm 半徑 / 14.7cm 底闊，啱啱係細號訓練筒 —— 兩件唔同嘅已知物件
    # 喺唔同場次得出同一個尺度，互相印證。
    # 接觸判定要三樣同時成立（門檻由用戶核片嘅真值定出嚟，2026-09-22）：
    #   1. 水平距離插入筒底 —— 距離 < 筒半徑 × (1 + FOOT_K)
    #   2. **隻腳踩實地下** —— 踝離地 < PLANT × 筒半徑。隻腳喺半空唔可能踩到筒。
    #   3. **插入夠深** —— 深入 ≥ PEN × 筒半徑，擦邊唔算。
    # 真值：rec_20260915_094938（用戶確認**真係踩到**）唯一嗰次深入 8.0cm、踝離地 −1.1cm；
    # rec_20260915_114426（用戶確認**冇踩到**）四次全部係假陽性 ——
    # 兩次隻腳喺半空（+37.9cm、+36.0cm），兩次擦邊（深入 3.5cm、3.9cm）。
    # 容差用**筒半徑**（相機量返嘅）唔用腿長：腿長係骨架先驗出嚟，兩位受測者
    # 讀到 78cm vs 100cm，用佢做容差會令唔同人嘅門檻差好遠。
    # PLANT 由真值定：真踩到嗰次踝離地 5.2cm（筒半徑 12.9cm → 0.40 倍），
    # 兩次假陽性係 12.8cm 同 15.2cm（0.92 / 1.09 倍）。0.5 倍切喺中間，兩邊都有餘裕。
    FOOT_K, PLANT, PEN, NEAR_PATH, CONE_R_FALLBACK = 0.8, 0.5, 0.4, 1.5, 0.30
    # cone_hits 留 None = **判唔到**（冇偵測到課程筒），唔係「0 次碰撞」——
    # 報 0 等於白送呢 25 分。有課程筒先會出 0 或以上嘅數。
    cone_n = cone_hits = None
    cone_min = None
    if cones is not None and len(cones[0]) and ok.sum() >= 5:
        C = np.asarray(cones[0], np.float64) * cone_scale
        Crad = np.asarray(cones[1], np.float64) * cone_scale
        Ch = C - np.outer(h(C), up)
        # 只留貼近路徑嘅筒
        dmin = np.array([np.nanmin(np.linalg.norm(hz[ok] - c, axis=1)) for c in Ch])
        keep = dmin < NEAR_PATH * leg
        course, crad = Ch[keep], Crad[keep]
        crad = np.where(np.isfinite(crad), crad, CONE_R_FALLBACK * leg)
        # 半徑離群 = 鬼影。同一個場地啲筒係同款，半徑應該一致；幾何鬼影
        # （兩台機各自唔同筒嘅錯配交點）位置同大細都錯，讀到嘅半徑會離群。
        # 實測 rec_20260915_114426：真筒 12.6–14.5cm，鬼影讀到 16.6 / 19.8 /
        # 25.4 / 30.5 / 30.5cm，而其中 16.6cm 嗰個落喺空地但啱啱喺細路腳邊，
        # 做出用戶核實為假嘅「踩筒」。兩輪修剪 ±20% 就剔得乾淨。
        if len(crad) >= 3:
            med = float(np.nanmedian(crad))
            for _ in range(2):
                inl = np.abs(crad - med) <= 0.20 * med
                if inl.sum() >= 2:
                    med = float(np.nanmedian(crad[inl]))
            good = np.abs(crad - med) <= 0.20 * med
            course, crad = course[good], crad[good]
        cone_n = int(len(course))
        if cone_n:
            ank = np.stack([K[:, LAN], K[:, RAN]])          # (2,T,3)
            ankh = ank - np.einsum('stx,x->st', ank, up)[..., None] * up
            d = np.linalg.norm(ankh[:, :, None, :] - course[None, None, :, :], axis=-1)
            ankz = np.stack([h(K[:, LAN]), h(K[:, RAN])])
            floor = (float(np.nanpercentile(ankz, 5))
                     if np.isfinite(ankz).any() else np.nan)
            rr = crad * (1.0 + FOOT_K)
            with np.errstate(invalid='ignore'):
                near = np.nanmin(d, axis=(0, 2)) / leg      # 逐幀最近嘅筒（腿長倍數）
                pen = rr[None, None, :] - d                 # 插入深度（米）
                # 條件 2：隻腳要踩實地下先算數
                planted = (ankz - floor) < PLANT * np.nanmedian(crad)
                pen = np.where(planted[:, :, None], pen, -9.0)
                # 條件 3：插入要夠深
                deep = pen - PEN * crad[None, None, :]
                best = np.nanmax(deep, axis=(0, 2))
            cone_min = _r(float(np.nanmin(near))) if np.isfinite(near).any() else None
            touch = np.nan_to_num(best, nan=-9.0) > 0
            # 最短持續：單幀跳動係量度雜訊，唔算踩到。實測真陽性
            # （rec_20260915_094938 f826，用戶核實）持續 3 幀；
            # 而 rec_20260915_114426（用戶核實冇踩到）剩低嗰宗喺片頭 f16、得 1 幀。
            MIN_F = max(2, int(0.10 * fps))
            cone_hits, run = 0, 0
            for x in list(touch) + [False]:
                if x:
                    run += 1
                else:
                    if run >= MIN_F:
                        cone_hits += 1
                    run = 0

    summary = {
        'frames': int(valid.sum()), 'duration_s': _r(len(K) / fps),
        'hand_raise_frame': raise_f, 'assess_start_frame': start_f,
        'start_trigger': start_trigger or 'none',
        'trigger_found': None if not start_trigger else start_f is not None,
        'coverage': _r(valid.mean()),
        'shoulder_cm': _r(np.nanmedian(
            np.linalg.norm(K[:, LSH] - K[:, RSH], axis=1)) * 100),
        'leg_cm': _r(leg * 100), 'scale_correction': _r(scale),
        'stance_ratio': _r(stance), 'stance_peak': _r(stance_peak),
        'did_perform': did,
        'travel_pct_leg': _r(100 * travel_ratio) if np.isfinite(travel_ratio) else None,
        'min_travel_ratio': min_travel,
        'cycles': len(strides),                      # 落點數（review.py 顯示用）
        # ── 準則五條 ──
        'facing_ok_pct': facing_pct,                 # 面向目標 25
        'orient_dev_deg': orient_med,
        'facing_axis_deg': facing_axis_deg,
        'cone_contacts': cone_hits,                  # 不踩雪糕筒 25
        'cone_min_dist_pct_leg': _r(100 * cone_min) if cone_min is not None else None,
        'cones_course_n': cone_n,
        'cone_radius_cm': _r(float(np.nanmedian(Crad)) * 100)
        if cones is not None and len(cones[0]) else None,
        'sway_pct_leg': sway,                        # 身體直線度 20
        'stride_cv_pct': stride_cv,                  # 步伐穩定性 15
        'stride_cv_robust_pct': stride_cv_rob,
        'stride_cv_seg_pct': stride_cv_seg,
        'stride_time_cv_pct': stride_time_cv,
        'stride_seg_n': len(strides_seg),
        'stride_n': len(strides),
        'trunk_lateral_deg': trunk_lat,              # 身體直線度 20（讀法 a）
        'trunk_lateral_range_deg': trunk_lat_range,  # 讀法 a2：擺動幅度
        'longest_pause_s': pause,                    # 有方向的流暢度 15（面向目標期間）
        'longest_pause_all_s': pause_all,            # 全程（對照）
    }
    return summary, []


def _squat_jump(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                off, stance, stance_peak, raise_f, start_f, start_trigger,
                min_drop, ball=None, ws=None):
    """K3 持球原地深蹲跳 —— 照《K3_上學期下學期_體能動作評估準則_優化版》
    第 2 條度，唔沿用 k1_straight_run 嗰套步態指標。

    | 指標     | 準則計算方式           | K3 良好標準 | 分 |
    |----------|------------------------|-------------|----|
    | 深蹲深度 | θ_hip 最低點 (23-25-27)| 55°–110°    | 20 |
    | 持球穩定 | 15,16 + Ball           | 球保持胸前控制不掉 | 25 |
    | 跳躍離地 | ΔY_hip                 | 明顯離地    | 20 |
    | 落地穩定 | 落地後 1s 內 ΔX_hip    | < 9 cm      | 20 |
    | 軀幹控制 | θ_vert                 | < 15°       | 15 |

    ⚠ 兩件要記住：

    1. 「持球穩定」（25 分，準則入面最大嗰條）**呢度量唔到** —— pose3d.npz
       得人體 17 點，冇球。準則標題自己寫明「物件：Ball BB（含短暫丟失容許）」，
       即係要 4 路球偵測 + 三角化先做得。暫時出 null。
    2. docx 啲門檻係**正面 2D 投影年代**嘅數（表頭寫「拍攝角度：正面」）。
       3D 讀數唔同量級，尤其 θ_vert：正面機位睇到嘅其實淨係左右傾，
       深蹲本身嘅前傾佢量唔到。所以呢度同時出三個讀數
       （總傾角／左右傾／前後傾），邊個對得返 docx 要由數據判。
    """
    shank = np.nanmedian(np.linalg.norm(K[:, LKN] - K[:, LAN], axis=1))
    leg = float(thigh + shank)
    hh = h(hip)

    # θ_hip 最低點：docx 寫「髖關節最低點角度」但關鍵點係 23-25-27，
    # 即係膝角。同 K2 原地深蹲跳嗰張表一樣嘅讀法。
    knee = {}
    for side, (hp, kn, an) in (('L', (LHIP, LKN, LAN)), ('R', (RHIP, RKN, RAN))):
        knee[side] = np.array([_ang(K[i, hp], K[i, kn], K[i, an])
                               for i in range(len(K))])
    knee_m = np.nanmean(np.stack([knee['L'], knee['R']]), 0)

    # 企直基線：膝角 > 160° 嗰啲幀嘅髖高。用「企直」而唔係「全片中位」，
    # 因為蹲住嘅時間佔成段片一大截，中位會把基線拉低。
    #
    # ⚠ 淨計膝角唔夠：細路做完坐喺地下、對腳伸直，膝角一樣 >160°，但髖高好低。
    # 實測 rec_20260915_104407#1 —— 佢坐低嗰段被當成「企直」，基線塌下去，
    # 連帶 ΔY_hip 讀到 20cm（假）、最低點膝角讀到 16°（其實係坐緊）。
    # 所以要加多一條「髖remains高過踝」：企直時 (髖−踝)/軀幹 實測 1.57–1.80，
    # 坐低時遠低於 1，用 1.2 做界線有闊落嘅安全邊際。
    ank_h = np.nanmean(np.stack([h(K[:, LAN]), h(K[:, RAN])]), 0)
    torso_len = max(float(np.nanmedian(h(sh) - h(hip))), 1e-6)
    st_series = (hh - ank_h) / torso_len
    standing = (np.isfinite(knee_m) & (knee_m > 160)
                & np.isfinite(st_series) & (st_series > 1.2))
    if standing.sum() >= 5:
        base = float(np.nanmedian(hh[standing]))
        ank_base = float(np.nanmedian(ank_h[standing]))
    else:                                   # 成段片都冇企直過 —— 唔好靠估
        base = ank_base = np.nan
    # 坐低／跪低嘅幀唔應該當成深蹲。呢個比率同膝角幾何上綁死
    # （蹲得深，髖自然近踝），所以唔可以單靠佢分坐同蹲 —— 真正嘅分界
    # 係跳唔跳得起（見下面嘅滯空判別），呢個只係出嚟畀人覆查。
    sit_frac = (float(np.nanmean(np.nan_to_num(st_series, nan=9) < 0.45))
                if np.isfinite(st_series).any() else np.nan)

    # 軀幹三個讀數：總傾角（已計好嘅 trunk）、左右傾、前後傾。
    # 左右／前後軸由受測者自己嘅肩線定，唔使知場地朝向。
    ml = sh * 0
    v_sh = K[:, RSH] - K[:, LSH]
    ml = v_sh - np.outer((v_sh * up).sum(-1), up)
    ml = ml / np.maximum(np.linalg.norm(ml, axis=1, keepdims=True), 1e-9)
    ap = np.cross(np.broadcast_to(up, ml.shape), ml)
    tv = sh - hip
    tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    lat = np.degrees(np.arctan2(np.abs((tv * ml).sum(-1)), (tv * up).sum(-1)))
    fwd = np.degrees(np.arctan2(np.abs((tv * ap).sum(-1)), (tv * up).sum(-1)))

    # ---- 前置閘：有冇真係蹲落去 ----
    # 唔用膝抬（嗰個係跑步訊號）。用髖下沉量 ÷ 腿長：企定唔郁嘅人讀近 0。
    drop = base - hh
    peak_drop = (float(np.nanpercentile(drop, 95))
                 if np.isfinite(drop).any() else np.nan)
    peak_drop_ratio = peak_drop / leg if leg > 1e-6 else np.nan
    did = bool(np.isfinite(peak_drop_ratio) and peak_drop_ratio >= min_drop)

    # ---- 切循環：髖高訊號，唔係膝抬訊號 ----
    deep = np.nan_to_num(drop, nan=-1.0) >= min_drop * leg
    reps, i = [], 0
    while i < len(deep):
        if not deep[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(deep) and deep[j + 1]:
            j += 1
        # 太短嘅當雜訊；太長嘅唔係一個深蹲跳 —— 實測真循環 0.2–1.3s，
        # 而做完坐喺地下嗰段會切出 4–5 秒嘅「循環」（膝角 16°、髖沉 80cm），
        # 佢喺 1 秒窗口內起返身就會做出假滯空。時長上限直接斬走呢類。
        if max(2, int(0.15 * fps)) <= (j - i + 1) <= int(2.0 * fps):
            reps.append((i, j))
        i = j + 1

    rows, depth, rise, sway, tk, tk_lat, tk_fwd, flight = [], [], [], [], [], [], [], []
    sway_lat = []
    win = max(1, int(round(1.0 * fps)))
    for n, (a, b) in enumerate(reps):
        seg = slice(a, b + 1)
        d_deg = float(np.nanmin(knee_m[seg])) if np.isfinite(knee_m[seg]).any() else np.nan
        # 跳躍：落底之後 1 秒內髖高最高點相對企直基線
        post = slice(b, min(b + win + 1, len(hh)))
        apex_i = (b + int(np.nanargmax(hh[post]))
                  if np.isfinite(hh[post]).any() else None)
        r_cm = ((hh[apex_i] - base) * 100 if apex_i is not None
                and np.isfinite(hh[apex_i]) else np.nan)
        # 滯空：**髖**高過企直基線，而且兩踝離開企直時嘅踝高。
        # 只計腳踝唔得 —— 坐喺地下嗰陣對腳抬起，腳踝一樣高過基線，
        # 會做出假滯空（實測 rec_20260915_104407#1 坐低嗰段全部中招）。
        # 真跳躍一定連髖一齊升過企直高度，坐低點都做唔到。
        fl = 0
        if np.isfinite(ank_base) and np.isfinite(base):
            seg2 = np.arange(b, min(b + win + 1, len(ank_h)))
            fl = int(np.nansum((ank_h[seg2] - ank_base > 0.03)
                               & (hh[seg2] - base > 0.02 * leg)))
        # 落地：頂點之後髖高回到基線附近嗰一幀
        land_i = None
        if apex_i is not None:
            for t in range(apex_i + 1, min(apex_i + 2 * win, len(hh))):
                if np.isfinite(hh[t]) and hh[t] <= base + 0.02 * leg:
                    land_i = t
                    break
        s_cm = s_lat = np.nan
        if land_i is not None:
            w = slice(land_i, min(land_i + win + 1, len(hip)))
            ho = hip[w] - np.outer(h(hip[w]), up)         # 水平面投影
            ok = np.isfinite(ho).all(1)
            if ok.sum() >= 3:
                c = ho[ok] - np.nanmedian(ho[ok], 0)
                s_cm = float(np.nanmax(np.linalg.norm(c, axis=1)) * 100)
                # docx 寫 ΔX_hip、拍攝角度正面 —— 正面機位量到嘅係左右分量，
                # 唔係水平面上嘅總位移。兩個都出，門檻先對得返。
                mlv = np.nanmedian(ml[w][ok], axis=0)
                if np.isfinite(mlv).all():
                    s_lat = float((np.nanmax(c @ mlv) - np.nanmin(c @ mlv)) * 100)
        t_deg = _r(np.nanpercentile(trunk[seg], 90)) if np.isfinite(trunk[seg]).any() else None
        rows.append({
            'cycle': n, 'start_frame': a + off, 'end_frame': b + off,
            'bottom_frame': int(a + off + np.nanargmin(knee_m[seg]))
            if np.isfinite(knee_m[seg]).any() else None,
            'squat_depth_deg': _r(d_deg),
            'hip_drop_cm': _r(float(np.nanmax(drop[seg])) * 100),
            'jump_rise_cm': _r(r_cm),
            'flight_frames': fl,
            'landing_sway_cm': _r(s_cm),
            'landing_sway_lat_cm': _r(s_lat),
            'trunk_lean_deg': t_deg,
        })
        # 「跳」先算一個循環。實測 255 個由髖高切出嚟嘅循環入面，42 個
        # （16%）滯空 0 幀、ΔY_hip 中位 −3.3cm —— 嗰啲係彎腰執地下個球／
        # 淨係蹲唔跳，唔應該混入「深蹲跳」嘅深度同落地讀數。
        rows[-1]['is_jump'] = bool(fl >= 2)
        if fl < 2:
            continue
        for lst, v in ((depth, d_deg), (rise, r_cm), (sway, s_cm),
                       (sway_lat, s_lat)):
            if np.isfinite(v):
                lst.append(float(v))
        flight.append(fl)
        for lst, arr in ((tk, trunk), (tk_lat, lat), (tk_fwd, fwd)):
            if np.isfinite(arr[seg]).any():
                lst.append(float(np.nanpercentile(arr[seg], 90)))

    # ---- 持球穩定（準則 25 分）----
    # 準則：「球是否保持胸前控制」「球保持控制不掉」（物件 Ball BB，含短暫丟失容許）。
    #
    # 量法係實測分佈逼出嚟嘅，唔係估：球心↔手腕中點呈明顯雙峰 —— 10–30cm 一大舊
    # （揸住），40–60cm 係穀底，100–120cm 第二舊（放咗喺地下／筒上）。
    # 用穀底做界線切開之後，兩批嘅球高度乾淨到分家：
    #   揸住  → 球高（相對髖 ÷ 軀幹長）中位 +0.80，即係髖同肩之間 = 胸前 ✅
    #   放開  → 中位 −0.83，遠低過髖 = 喺地下 ❌
    # 用戶確認過：捧住球蹲落去、把球放喺地下個筒度再起身跳 = **做得唔標準**，
    # 正正就係呢 25 分要扣嘅嘢。
    #
    # 門檻用軀幹長正規化（1.0 × 軀幹長 ≈ 45cm）而唔用絕對 cm —— 公制錨點
    # SHOULDER_CM 未核實（見 rubrics 檔 _scale_caveat），正規化就唔受影響。
    # 只計跳躍循環嘅幀：循環之間細路排隊、放低個球，嗰啲唔關「持球」事。
    bm = {'ball_seen_pct': None, 'ball_held_pct': None, 'ball_height_ratio': None,
          'ball_release_events': None, 'ball_wrist_cm': None,
          'ball_drop_events': None, 'ball_drop_rejected_gap': None,
          'ball_max_away_torso': None}
    if ball is not None:
        bxyz, bn = ball
        torso = max(float(np.nanmedian(h(sh) - h(hip))), 1e-6)
        wrist = np.nanmean(K[:, [LWR, RWR]], 1)
        idx = (np.concatenate([np.arange(a, b + 1) for r2, (a, b) in
                               zip(rows, reps) if r2.get('is_jump')])
               if any(r2.get('is_jump') for r2 in rows) else np.array([], int))
        idx = idx[(idx + off) < len(bxyz)]
        if len(idx) >= 10:
            Bp = bxyz[idx + off]
            seen = np.isfinite(Bp).all(1) & np.isfinite(wrist[idx]).all(1)
            bm['ball_seen_pct'] = _r(100.0 * seen.mean())
            if seen.sum() >= 5:
                dw = np.linalg.norm(Bp[seen] - wrist[idx][seen], axis=1)
                held = dw < 1.0 * torso
                bm['ball_wrist_cm'] = _r(float(np.nanmedian(dw)) * 100)
                bm['ball_held_pct'] = _r(100.0 * float(held.mean()))
                hr = (h(Bp[seen]) - h(hip[idx][seen])) / torso
                if held.any():
                    bm['ball_height_ratio'] = _r(float(np.nanmedian(hr[held])))
                # 放手事件：連續 ≥0.25s 唔喺手 —— 短暫丟失唔計（準則容許）
                need, run, ev = max(2, int(0.25 * fps)), 0, 0
                for x in ~held:
                    run = run + 1 if x else 0
                    if run == need:
                        ev += 1
                bm['ball_release_events'] = int(ev)
                bm.update(_ball_drops(bxyz, wrist, hip, h, torso, fps, off, idx, bn))

    def med(v):
        return _r(float(np.median(v))) if v else None

    summary = {
        'frames': int(valid.sum()), 'duration_s': _r(len(K) / fps),
        'hand_raise_frame': raise_f, 'assess_start_frame': start_f,
        'start_trigger': start_trigger or 'none',
        'trigger_found': None if not start_trigger else start_f is not None,
        'coverage': _r(valid.mean()),
        'shoulder_cm': _r(np.nanmedian(
            np.linalg.norm(K[:, LSH] - K[:, RSH], axis=1)) * 100),
        'leg_cm': _r(leg * 100),
        'scale_correction': _r(scale),
        'stance_ratio': _r(stance),
        'stance_peak': _r(stance_peak),
        # 前置閘（深蹲跳自己嘅，唔係跑步嗰個 min_lift）
        'did_perform': did,
        'peak_hip_drop_cm': _r(peak_drop * 100),
        'peak_hip_drop_ratio': _r(peak_drop_ratio),
        'min_drop_ratio': min_drop,
        'sit_fraction': _r(sit_frac),
        'standing_frames': int(standing.sum()),
        'cycles': int(sum(1 for r in rows if r.get('is_jump'))),
        'reps_total': len(reps),
        'reps_no_jump': int(sum(1 for r in rows if not r.get('is_jump'))),
        'jumped': bool(any(r.get('is_jump') for r in rows)),
        # ── 準則五條 ──
        'squat_depth_deg': med(depth),              # 深蹲深度 20 分
        **bm,                                       # 持球穩定 25 分
        'jump_rise_cm': med(rise),                  # 跳躍離地 20 分
        'flight_frames_med': med(flight),
        'landing_sway_cm': med(sway),               # 落地穩定 20 分（水平總位移）
        'landing_sway_lat_cm': med(sway_lat),       # 正面機位睇到嘅 ΔX
        # ── 正規化版：除返腿長，唔受公制錨點影響 ──
        # 準則對照用呢啲，唔用 cm。原因：肩寬錨點同球錨點爭成 1.4 倍（見下），
        # 邊個啱未定案之前，任何 cm 門檻都企唔穩；比率就冇呢個問題。
        'jump_rise_pct_leg': _r(100 * med(rise) / (leg * 100))
        if med(rise) is not None and leg > 1e-6 else None,
        'landing_sway_lat_pct_leg': _r(100 * med(sway_lat) / (leg * 100))
        if med(sway_lat) is not None and leg > 1e-6 else None,
        'hip_drop_pct_leg': _r(100 * peak_drop / leg) if leg > 1e-6 else None,
        # ── 球錨點換算出嚟嘅「真實」cm（個球係場內唯一已知尺寸物件）──
        'world_scale_ball': _r(ws, 4),
        'cm_true_factor': _r(ws / scale, 4) if ws else None,
        'jump_rise_cm_true': _r(med(rise) * ws / scale)
        if (ws and med(rise) is not None) else None,
        'landing_sway_lat_cm_true': _r(med(sway_lat) * ws / scale)
        if (ws and med(sway_lat) is not None) else None,
        'peak_hip_drop_cm_true': _r(peak_drop * 100 * ws / scale) if ws else None,
        'trunk_lean_deg': med(tk),                  # 軀幹控制 15 分（3D 總傾角）
        'trunk_lateral_deg': med(tk_lat),           # 正面機位睇到嘅
        'trunk_forward_deg': med(tk_fwd),
    }
    return summary, rows


def _body_axes(K, up):
    """受測者自己嘅左右軸（肩線去垂直分量）同前後軸。唔使知場地朝向。"""
    v = K[:, RSH] - K[:, LSH]
    ml = v - np.outer((v * up).sum(-1), up)
    ml = ml / np.maximum(np.linalg.norm(ml, axis=1, keepdims=True), 1e-9)
    ap = np.cross(np.broadcast_to(up, ml.shape), ml)
    return ml, ap


def _rope_jump(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
               off, stance, stance_peak, raise_f, start_f, start_trigger):
    """K3 跳繩完整動作（連續）—— 照《K3_Physical_Assessment_Criteria_Optimized.md》
    下學期 手腳協調 第 3 條（「不偵測繩子；只用身體關鍵點判斷連續跳」）。

    | 指標       | 準則計算方式        | K3 良好標準 | 分 |
    |------------|---------------------|-------------|----|
    | 連續跳次數 | N_jump（明顯離地）  | ≥ 4–5       | 25 |
    | 跳躍節奏   | 跳躍週期差率        | < 12%       | 20 |
    | 落地穩定   | 落地後 ΔX_hip       | < 9 cm      | 20 |
    | 手臂擺動   | 手腕軌跡穩定性      | 大致穩定    | 20 |
    | 軀幹穩定   | ΔX_torso            | < 8 cm      | 15 |

    ⚠ 上學期都有一張同名表（差率 < 15%）；資料夾寫「（連續）」＝下學期。

    跳 = **兩踝同時**離開企直踝高 **而且** 髖高過企直基線 —— 同深蹲跳一樣。
    2026-09-23 rec_20260915_101628 逐幀對過片：K3 細路好多係「跨繩」
    （一隻腳抬到 20–40cm、另一隻踩實地、膝屈 90–130°），淨睇腳踝會當成跳；
    要兩踝都離地先係跳。真跳之間仲有細路會加一下細跳（雙拍），呢啲照計。
    """
    shank = np.nanmedian(np.linalg.norm(K[:, LKN] - K[:, LAN], axis=1))
    leg = float(thigh + shank)
    hh = h(hip)
    aL, aR = h(K[:, LAN]), h(K[:, RAN])
    alow = np.fmin(aL, aR)                       # 較低嗰隻腳：佢都離地 = 兩腳離地
    knee_m = np.nanmean(np.stack([
        np.array([_ang(K[i, LHIP], K[i, LKN], K[i, LAN]) for i in range(len(K))]),
        np.array([_ang(K[i, RHIP], K[i, RKN], K[i, RAN]) for i in range(len(K))])]), 0)
    torso = max(float(np.nanmedian(h(sh) - hh)), 1e-6)
    st = (hh - alow) / torso
    standing = np.isfinite(knee_m) & (knee_m > 160) & np.isfinite(st) & (st > 1.2)
    if standing.sum() >= 5:
        hb = float(np.nanmedian(hh[standing]))
        ab = float(np.nanmedian(alow[standing]))
    else:
        hb = ab = np.nan

    # ---- 跳躍事件 ----
    air = (np.nan_to_num(alow - ab, nan=-1) > 0.04) & \
          (np.nan_to_num(hh - hb, nan=-1) > 0.03 * leg)
    ev, i = [], 0
    while i < len(air):
        if not air[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(air) and (air[j + 1] or (j + 2 < len(air) and air[j + 2])):
            j += 1                               # 容許中間斷一幀
        d = j - i + 1
        rise = float(np.nanmax(hh[i:j + 1]) - hb) / leg if leg > 1e-6 else np.nan
        # 2 幀（0.1s）以下當雜訊；0.6s 以上唔係一下跳（實測跨繩／行位）
        if 2 <= d <= int(0.6 * fps) and np.isfinite(rise) and rise >= 0.05:
            ev.append((i, j, rise))
        i = j + 1

    # ---- 連續段：兩下起跳相隔 ≤ 1.5s 當連續（一轉繩 + 雙拍都喺呢個範圍）----
    gap_max = 1.5 * fps
    streaks, cur = [], []
    for e in ev:
        if cur and e[0] - cur[-1][0] > gap_max:
            streaks.append(cur)
            cur = []
        cur.append(e)
    if cur:
        streaks.append(cur)
    best = max(streaks, key=len) if streaks else []

    # 節奏：所有 ≥3 下嘅連續段入面嘅起跳間距
    iv = [(b[0] - a[0]) / fps for sk in streaks if len(sk) >= 3
          for a, b in zip(sk, sk[1:])]
    # 至少 3 個間距先計差率：2 個間距 20fps 量化後好易啱啱相等，讀 0%（實測 101737#1）
    cv = (float(np.std(iv) / np.mean(iv) * 100) if len(iv) >= 3 else np.nan)

    ml, _ap = _body_axes(K, up)
    rows, sway = [], []
    for n, (a, b, r) in enumerate(ev):
        nxt = ev[n + 1][0] if n + 1 < len(ev) else len(hh)
        w = slice(b + 1, min(b + 1 + int(0.5 * fps), nxt, len(hh)))  # 落地到下一跳之前
        s_lat = np.nan
        if w.stop - w.start >= 3:
            ho = hip[w] - np.outer(h(hip[w]), up)
            ok = np.isfinite(ho).all(1)
            mlv = np.nanmedian(ml[w][ok], 0) if ok.sum() >= 3 else None
            if mlv is not None and np.isfinite(mlv).all():
                c = (ho[ok] - np.median(ho[ok], 0)) @ mlv
                s_lat = float((c.max() - c.min()) / leg * 100)
                sway.append(s_lat)
        rows.append({'cycle': n, 'start_frame': a + off, 'end_frame': b + off,
                     'flight_frames': b - a + 1, 'rise_pct_leg': _r(r * 100),
                     'landing_sway_lat_pct_leg': _r(s_lat)})

    # ---- 連續段期間嘅手臂同軀幹 ----
    # 手臂／軀幹：用**所有跳躍**嘅幀（每下 ±5 幀）而唔係淨係最長嗰段 ——
    # 最長連續段得 1–3 下嘅細路幀數唔夠，實測 31 位入面 5 位因此出 None，
    # 但佢哋明明跳咗 4–13 下，數據係有嘅。
    if ev:
        seg = np.unique(np.concatenate([
            np.arange(max(0, a - 5), min(len(hh), b + 6)) for a, b, _ in ev]))
    else:
        seg = np.array([], int)
    wr_spread = trunk_rng = torso_lat = np.nan
    if len(seg) >= 10:
        wr = np.nanmean(K[seg][:, [LWR, RWR]], 1) - hip[seg]      # 手腕相對髖
        ok = np.isfinite(wr).all(1)
        if ok.sum() >= 10:
            # 手腕軌跡穩定性：相對髖嘅位置散佈（p90−p10 距中位）÷ 軀幹長。
            # 穩定搖繩手喺身側細圈；亂揮就散。
            dv = np.linalg.norm(wr[ok] - np.median(wr[ok], 0), axis=1)
            wr_spread = float(np.percentile(dv, 90) / torso)
        tv = sh[seg] - hip[seg]
        tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
        sl = np.degrees(np.arctan2((tv * ml[seg]).sum(-1), (tv * up).sum(-1)))
        if np.isfinite(sl).sum() >= 10:
            trunk_rng = float(np.nanpercentile(sl, 90) - np.nanpercentile(sl, 10))
        # ΔX_torso：肩中點**相對髖中點**嘅左右擺幅 ÷ 腿長。唔可以用肩髖中點
        # 嘅絕對位置 —— 細路跳跳下會成個人橫移，實測讀到 268%腿長（= 1.9m 行位）。
        mlv = np.nanmedian(ml[seg], 0)
        x = (sh[seg] - hip[seg]) @ mlv
        if np.isfinite(x).sum() >= 10:
            torso_lat = float((np.nanpercentile(x, 90) - np.nanpercentile(x, 10)) / leg * 100)

    did = len(ev) >= 2
    summary = {
        'frames': int(valid.sum()), 'duration_s': _r(len(K) / fps),
        'hand_raise_frame': raise_f, 'assess_start_frame': start_f,
        'start_trigger': start_trigger or 'none',
        'trigger_found': None if not start_trigger else start_f is not None,
        'coverage': _r(valid.mean()),
        'shoulder_cm': _r(np.nanmedian(np.linalg.norm(K[:, LSH] - K[:, RSH], axis=1)) * 100),
        'leg_cm': _r(leg * 100), 'scale_correction': _r(scale),
        'stance_ratio': _r(stance), 'stance_peak': _r(stance_peak),
        'standing_frames': int(standing.sum()),
        'did_perform': bool(did),
        'cycles': len(ev),
        # ── 準則五條 ──
        'jumps_total': len(ev),
        'jump_streak_max': len(best),                         # 連續跳次數 25
        'jump_streaks': len(streaks),
        'jump_interval_s': _r(float(np.median(iv))) if iv else None,
        'jump_interval_cv_pct': _r(cv),                       # 跳躍節奏 20
        'jump_rise_pct_leg': _r(float(np.median([e[2] for e in ev])) * 100) if ev else None,
        'landing_sway_lat_pct_leg': _r(float(np.median(sway))) if sway else None,  # 落地穩定 20
        'wrist_spread_torso': _r(wr_spread, 3),               # 手臂擺動 20
        'torso_lat_range_pct_leg': _r(torso_lat),             # 軀幹穩定 15（ΔX_torso）
        'trunk_lateral_range_deg': _r(trunk_rng),
    }
    return summary, rows


def _load_hurdles(out_dir):
    """hurdle3d.py 出嘅欄杆地面線段（raw 世界座標，N×2×3）；冇就 None。"""
    p = os.path.join(out_dir, 'hurdles3d.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z['segments'] if len(z['segments']) else None


def _load_floor_up(out_dir):
    """floor_up.py 寫嘅全場垂直軸；冇就 None（退返逐位擬地面）。"""
    p = os.path.join(out_dir, 'floor_up.json')
    return json.load(open(p))['up'] if os.path.exists(p) else None


def _load_line(out_dir):
    """讀該條片嘅平衡線 3D（line3d.py 出嘅），未修正尺度嘅世界座標。"""
    p = os.path.join(out_dir, 'line3d.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z['p0'], z['p1']


def _line_walk(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
               off, stance, stance_peak, raise_f, start_f, start_trigger,
               line=None):
    """K1 平衡線前行 —— 照《K1_Physical_Assessment_Criteria_Optimized.md》
    上學期 平衡能力 第 2 條（拍攝角度：正面 / 45°｜物件：Line BB）。

    | 指標           | 準則計算方式                   | K1 良好標準    | 分 |
    |----------------|--------------------------------|----------------|----|
    | 平衡線控制     | d_line（允許輕微偏離）         | 多數時間接近線上 | 25 |
    | 身體直線度     | ΔX_hip                         | < 15 cm        | 25 |
    | 步伐規律性     | 髖水平位移週期差 + 膝蓋起伏規律 | 規律性尚可     | 20 |
    | 姿勢控制       | θ_vert                         | < 20°          | 15 |
    | 有方向的流暢度 | 方向正確下 t_pause             | < 0.6 s        | 15 |

    條線由 line3d.py 重建（地上白膠紙，四台機投地面疊埋）。**評估窗口 = 細路
    髖部投影落喺線段範圍之內嗰段**，行入行出唔計。冇條線（line3d 失敗）就
    平衡線控制／身體直線度出 None（判唔到），其餘照計、窗口改用全段。
    """
    shank = np.nanmedian(np.linalg.norm(K[:, LKN] - K[:, LAN], axis=1))
    leg = float(thigh + shank)
    T = len(K)
    hz = hip - np.outer(h(hip), up)                            # 髖水平投影
    ml, _ap = _body_axes(K, up)

    have_line = line is not None
    if have_line:
        p0, p1 = (np.asarray(x, np.float64) * scale for x in line)
        p0 = p0 - (p0 @ up) * up
        p1 = p1 - (p1 @ up) * up
        L = float(np.linalg.norm(p1 - p0))
        u = (p1 - p0) / max(L, 1e-9)
        nrm = np.cross(up, u)                                  # 地面上垂直條線
        t_hip = (hz - p0) @ u
        on = np.isfinite(t_hip) & (t_hip >= 0) & (t_hip <= L)
    else:
        on = np.isfinite(hz).all(1)
    idx = np.where(on)[0]
    win = slice(idx[0], idx[-1] + 1) if len(idx) >= 10 else slice(0, 0)
    W = np.arange(T)[win]
    # 覆蓋守門：髖部喺線方向實際行過幾多成條線。追蹤斷咗（例如 s28 rec_20260916_142900
    # 重投影核過：只喺起點有腳踝點，上咗線之後 cam04 被跪低嘅大人遮住）就唔夠一半，
    # 嗰種讀數係「量唔到」唔係「做得差」—— 平衡線兩列出 None。
    progress = np.nan
    if have_line and len(idx):
        tt = t_hip[idx]
        progress = float((np.nanmax(tt) - np.nanmin(tt)) / max(L, 1e-9) * 100)
    covered = bool(have_line and np.isfinite(progress) and progress >= 50)

    # ---- 平衡線控制：支撐腳（踩地嗰隻）離條線幾遠 ----
    foot_med = foot_on = hip_rng = hip_off = None
    if covered and len(W) >= 10:
        aL, aR = h(K[:, LAN]), h(K[:, RAN])
        ab = np.nanpercentile(np.fmin(aL, aR)[W], 20)
        ds = []
        for an, ah in ((LAN, aL), (RAN, aR)):
            planted = np.nan_to_num(ah[W] - ab, nan=9) < 0.03    # 腳踩實地
            fz = K[W, an] - np.outer(h(K[W, an]), up)
            d = np.abs((fz - p0) @ nrm)[planted]
            ds.append(d[np.isfinite(d)])
        ds = np.concatenate(ds) if ds else np.array([])
        if len(ds) >= 10:
            foot_med = float(np.median(ds) / leg * 100)
            # 「接近線上」：15% 腿長 ≈ K1 細路一隻半腳闊
            foot_on = float((ds < 0.15 * leg).mean() * 100)
        dh = ((hz[W] - p0) @ nrm)
        dh = dh[np.isfinite(dh)]
        if len(dh) >= 10:
            hip_rng = float((np.percentile(dh, 90) - np.percentile(dh, 10)) / leg * 100)
            hip_off = float(np.median(np.abs(dh)) / leg * 100)

    # ---- 步伐規律性：膝抬週期 ----
    lift = np.degrees(np.arccos(np.clip((h(hip) - h(K[:, LKN])) / max(thigh, 1e-6), -1, 1)))
    cyc, n_short = _cycles(lift[win], fps) if len(W) >= 10 else ([], 0)
    per = [c[2] for c in cyc]

    # ---- 姿勢控制：軀幹左右傾（正面機位睇到嘅分量）同總傾角 ----
    tv = sh - hip
    tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    lat = np.degrees(np.arctan2(np.abs((tv * ml).sum(-1)), (tv * up).sum(-1)))
    lat_p90 = float(np.nanpercentile(lat[W], 90)) if len(W) and np.isfinite(lat[W]).any() else np.nan
    tr_p90 = float(np.nanpercentile(trunk[W], 90)) if len(W) and np.isfinite(trunk[W]).any() else np.nan

    # ---- 有方向的流暢度：沿線方向速度嘅最長停頓 ----
    longest = stall = np.nan
    if len(W) >= 10:
        dirv = u if have_line else None
        if dirv is None:                                       # 冇線：用成段位移方向
            a0, a1 = hz[W][np.isfinite(hz[W]).all(1)][[0, -1]]
            dirv = (a1 - a0) / max(np.linalg.norm(a1 - a0), 1e-9)
        prog = hz[W] @ dirv
        v = np.abs(np.diff(prog)) * fps
        if len(v) >= 5:
            sm = v.copy()
            for i in range(2, len(v) - 2):
                sm[i] = np.nanmedian(v[i - 2:i + 3])
            ok = np.isfinite(sm)
            if ok.sum() >= 5:
                slow = ok & (sm < 0.20 * np.nanpercentile(sm, 75))
                stall = float(slow.sum() / ok.sum())
                run = best = 0
                for x, o in zip(slow, ok):
                    run = run + 1 if (x and o) else 0
                    best = max(best, run)
                longest = best / fps

    did = bool(len(W) >= int(1.0 * fps))
    summary = {
        'frames': int(valid.sum()), 'duration_s': _r(T / fps),
        'hand_raise_frame': raise_f, 'assess_start_frame': start_f,
        'start_trigger': start_trigger or 'none',
        'trigger_found': None if not start_trigger else start_f is not None,
        'coverage': _r(valid.mean()),
        'shoulder_cm': _r(np.nanmedian(np.linalg.norm(K[:, LSH] - K[:, RSH], axis=1)) * 100),
        'leg_cm': _r(leg * 100), 'scale_correction': _r(scale),
        'stance_ratio': _r(stance), 'stance_peak': _r(stance_peak),
        'did_perform': did,
        'line_found': have_line,
        'line_length_m': _r(L) if have_line else None,
        'on_line_frames': int(len(W)),
        'line_progress_pct': _r(progress),
        'line_covered': covered,
        'on_line_s': _r(len(W) / fps),
        'cycles': len(cyc), 'cycles_dropped_short': int(n_short),
        # ── 準則五條 ──
        'foot_line_dist_pct_leg': _r(foot_med),              # 平衡線控制 25
        'foot_on_line_pct': _r(foot_on),
        'hip_line_range_pct_leg': _r(hip_rng),               # 身體直線度 25（ΔX_hip）
        'hip_line_offset_pct_leg': _r(hip_off),
        'cycle_cv_pct': _r(np.std(per) / np.mean(per) * 100) if len(per) > 2 else None,  # 步伐規律性 20
        'cadence_spm': _r(60.0 / np.median(per)) if per else None,
        'trunk_lateral_p90_deg': _r(lat_p90),                # 姿勢控制 15（正面 θ_vert）
        'trunk_lean_p90_deg': _r(tr_p90),
        'longest_pause_s': _r(longest),                      # 有方向的流暢度 15
        'stall_fraction': _r(stall),
    }
    rows = [{'cycle': i, 'start_frame': int(a + off + win.start),
             'end_frame': int(b + off + win.start),
             'duration_s': round(float(d), 3)} for i, (a, b, d) in enumerate(cyc)]
    return summary, rows


def _leg_swing(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
               off, stance, stance_peak, raise_f, start_f, start_trigger):
    """K3 前行擺腿伸展 —— 照《K3_Physical_Assessment_Criteria_Optimized.md》
    下學期 伸展 第 3 條（拍攝角度：正面 / 側面）。

    | 指標         | 準則計算方式             | K3 良好標準 | 分 |
    |--------------|--------------------------|-------------|----|
    | 擺腿高度     | θ_swing 最高點           | 擺腿明顯    | 25 |
    | 支撐腳穩定   | ΔD（29–32）              | < 8 cm      | 20 |
    | 前進方向控制 | ΔX_hip                   | < 9 cm      | 20 |
    | 左右對稱     | 左右 θ_swing 差異率      | < 12%       | 20 |
    | 動作連貫性   | t_pause                  | < 0.25 s    | 15 |

    θ_swing = 成隻腳（髖→踝）離「吊直向下」嘅角度：0° 吊直、90° 打橫。
    2026-09-23 對 rec_20260915_111609：擺腿峰值 60–90°、左右交替，普通行路 <20°。
    一次擺腿 = 該腳角度 >35° 嘅一段。
    """
    shank = np.nanmedian(np.linalg.norm(K[:, LKN] - K[:, LAN], axis=1))
    leg = float(thigh + shank)
    T = len(K)
    ang = {}
    for side, (hp, an) in (('L', (LHIP, LAN)), ('R', (RHIP, RAN))):
        v = K[:, an] - K[:, hp]
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        ang[side] = np.degrees(np.arccos(np.clip(-(v * up).sum(-1), -1, 1)))

    ev = []                                          # (side, a, b, peak)
    for side in ('L', 'R'):
        hi = np.nan_to_num(ang[side], nan=0) > 35
        i = 0
        while i < T:
            if not hi[i]:
                i += 1
                continue
            j = i
            while j + 1 < T and (hi[j + 1] or (j + 2 < T and hi[j + 2])):
                j += 1
            if j - i + 1 >= 2:
                ev.append((side, i, j, float(np.nanmax(ang[side][i:j + 1]))))
            i = j + 1
    ev.sort(key=lambda e: e[1])
    pk = {sd: [e[3] for e in ev if e[0] == sd] for sd in ('L', 'R')}
    mL = float(np.median(pk['L'])) if pk['L'] else np.nan
    mR = float(np.median(pk['R'])) if pk['R'] else np.nan

    # 支撐腳穩定：擺腿期間另一隻腳嘅水平滑動（max 距離離起點）÷ 腿長
    slide = []
    for sd, a, b, _ in ev:
        an = RAN if sd == 'L' else LAN
        f = K[a:b + 1, an]
        f = f - np.outer(h(f), up)
        f = f[np.isfinite(f).all(1)]
        if len(f) >= 2:
            slide.append(float(np.max(np.linalg.norm(f - f[0], axis=1)) / leg * 100))

    # 評估窗口：第一下擺腿前 0.5s 到最後一下後 0.5s
    if ev:
        w0 = max(0, ev[0][1] - int(0.5 * fps))
        w1 = min(T, ev[-1][2] + int(0.5 * fps) + 1)
    else:
        w0 = w1 = 0
    W = np.arange(w0, w1)
    hz = hip - np.outer(h(hip), up)
    dev = longest = stall = np.nan
    if len(W) >= 10:
        P = hz[W]
        ok = np.isfinite(P).all(1)
        if ok.sum() >= 10:
            c = P[ok].mean(0)
            _, _, vt = np.linalg.svd(P[ok] - c)
            u = vt[0]
            nrm = np.cross(up, u)
            lat = (P[ok] - c) @ nrm
            dev = float((np.percentile(lat, 90) - np.percentile(lat, 10)) / leg * 100)
            prog = P @ u
            v = np.abs(np.diff(prog)) * fps
            if len(v) >= 5:
                sm = v.copy()
                for i in range(2, len(v) - 2):
                    sm[i] = np.nanmedian(v[i - 2:i + 3])
                okv = np.isfinite(sm)
                if okv.sum() >= 5:
                    slow = okv & (sm < 0.20 * np.nanpercentile(sm, 75))
                    stall = float(slow.sum() / okv.sum())
                    run = best = 0
                    for x, o in zip(slow, okv):
                        run = run + 1 if (x and o) else 0
                        best = max(best, run)
                    longest = best / fps

    did = len(ev) >= 2
    worse = np.nanmin([mL, mR]) if np.isfinite([mL, mR]).any() else np.nan
    summary = {
        'frames': int(valid.sum()), 'duration_s': _r(T / fps),
        'hand_raise_frame': raise_f, 'assess_start_frame': start_f,
        'start_trigger': start_trigger or 'none',
        'trigger_found': None if not start_trigger else start_f is not None,
        'coverage': _r(valid.mean()),
        'shoulder_cm': _r(np.nanmedian(np.linalg.norm(K[:, LSH] - K[:, RSH], axis=1)) * 100),
        'leg_cm': _r(leg * 100), 'scale_correction': _r(scale),
        'stance_ratio': _r(stance), 'stance_peak': _r(stance_peak),
        'did_perform': bool(did),
        'cycles': len(ev),
        'swings_L': len(pk['L']), 'swings_R': len(pk['R']),
        # ── 準則五條 ──
        'swing_peak_deg': _r(float(np.nanmedian(pk['L'] + pk['R']))) if ev else None,  # 擺腿高度 25
        'swing_peak_L_deg': _r(mL), 'swing_peak_R_deg': _r(mR),
        'swing_peak_worse_deg': _r(worse),
        'support_slide_pct_leg': _r(float(np.median(slide))) if slide else None,       # 支撐腳穩定 20
        'path_lateral_range_pct_leg': _r(dev),                                           # 前進方向控制 20
        'swing_asym_pct': _r(_asym(mL, mR)) if np.isfinite([mL, mR]).all() else None,  # 左右對稱 20
        'longest_pause_s': _r(longest),                                                  # 動作連貫性 15
        'stall_fraction': _r(stall),
    }
    rows = [{'cycle': n, 'side': sd, 'start_frame': int(a + off), 'end_frame': int(b + off),
             'peak_deg': _r(p)} for n, (sd, a, b, p) in enumerate(ev)]
    return summary, rows


def _squat(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
           off, stance, stance_peak, raise_f, start_f, start_trigger):
    """K1 原地深蹲 —— 照《K1_Physical_Assessment_Criteria_Optimized.md》上學期
    大肌肉發展 第 2 條（拍攝角度：正面）。量測幾何照
    game7-4cam-3d/backend/app/rubrics/k1_squat.py（COCO-17 3D，同一套定義）。

    | 指標     | 準則計算方式          | K1 良好標準   | 分 |
    |----------|-----------------------|---------------|----|
    | 深蹲深度 | θ_hip 最低點 23-25-27 | 越低越好      | 25 |
    | 膝蓋對齊 | θ_align               | < 18°         | 25 |
    | 腳跟落地 | 腳跟離地情況          | 雙腳跟大致着地 | 20 |
    | 軀幹控制 | θ_vert                | < 25°         | 15 |
    | 整體穩定 | 深蹲過程 ΔX_hip       | < 14 cm       | 15 |

    ⚠ 深蹲深度：md 寫 θ_hip 但關鍵點係 23-25-27（髖-膝-踝＝膝角），同 K3 深蹲跳
    一樣跟關鍵點；App 嘅 squat_rubric.py 用真髖角（肩-髖-膝）。兩個都出。
    COCO-17 冇腳跟：腳跟落地用腳踝升高 > 0.09 肩寬（k1_squat.py 實測掃出嚟嘅門檻）。
    一次 = 髖角跌穿 150° → 最低 < 140° → 升返過 158°（同 k1_squat.py 狀態機）。
    """
    T = len(K)
    shank = np.nanmedian(np.linalg.norm(K[:, LKN] - K[:, LAN], axis=1))
    leg = float(thigh + shank)
    sw = np.linalg.norm(K[:, LSH] - K[:, RSH], axis=1)
    swm = float(np.nanmedian(sw))
    ml, _ap = _body_axes(K, up)
    hipang = np.nanmean(np.stack([
        np.array([_ang(K[i, LSH], K[i, LHIP], K[i, LKN]) for i in range(T)]),
        np.array([_ang(K[i, RSH], K[i, RHIP], K[i, RKN]) for i in range(T)])]), 0)
    kneeang = np.nanmean(np.stack([
        np.array([_ang(K[i, LHIP], K[i, LKN], K[i, LAN]) for i in range(T)]),
        np.array([_ang(K[i, RHIP], K[i, RKN], K[i, RAN]) for i in range(T)])]), 0)
    # 膝蓋對齊：膝→踝 喺額狀面（受測者左右軸）相對垂直嘅傾角，取較差一邊
    shin = np.full(T, np.nan)
    for kn, an in ((LKN, LAN), (RKN, RAN)):
        v = K[:, an] - K[:, kn]
        hz_ = np.abs((v * ml).sum(-1))
        vt = np.abs((v * up).sum(-1))
        shin = np.fmax(shin, np.degrees(np.arctan2(hz_, np.maximum(vt, 1e-6))))
    tv = sh - hip
    tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    # 左右傾用 asin（見 _fold_stretch）：深蹲前傾 ~45°，arctan2 會放大左右分量
    lat = np.degrees(np.arcsin(np.clip(np.abs((tv * ml).sum(-1)), 0, 1)))

    # 分次
    reps, down, lo, mn = [], False, 0, 999.0
    for i in range(T):
        a = hipang[i]
        if not np.isfinite(a):
            continue
        if not down and a < 150:
            down, lo, mn = True, max(0, i - 3), a
        elif down:
            mn = min(mn, a)
            if a > 158:
                if mn < 140 and i - lo >= 6:
                    reps.append((lo, i))
                down = False
    # 企直基線（腳踝高）：髖角 > 160° 嘅幀
    aL, aR = h(K[:, LAN]), h(K[:, RAN])
    stand = np.nan_to_num(hipang, nan=0) > 160
    bL = np.nanmedian(aL[stand]) if stand.sum() >= 5 else np.nan
    bR = np.nanmedian(aR[stand]) if stand.sum() >= 5 else np.nan

    rows, dK, dH, al, heel, tk, tkl, sway = [], [], [], [], [], [], [], []
    for n, (a, b) in enumerate(reps):
        seg = np.arange(a, b + 1)
        hk = kneeang[seg]
        if not np.isfinite(hk).any():
            continue
        dk = float(np.nanmin(hk))
        dh = float(np.nanmin(hipang[seg]))
        # 最深 30% 幀評膝蓋對齊
        ha = hipang[seg]
        ok = np.isfinite(ha)
        deep = seg[ok][np.argsort(ha[ok])[:max(1, int(0.3 * ok.sum()))]]
        s_al = float(np.nanmedian(shin[deep])) if np.isfinite(shin[deep]).any() else np.nan
        # 腳跟：蹲低期間任何一邊腳踝升高 > 0.09 肩寬 嘅幀比例
        lift = np.fmax(aL[seg] - bL, aR[seg] - bR) / max(swm, 1e-6)
        hf = float(np.nanmean(lift > 0.09)) * 100 if np.isfinite(lift).any() else np.nan
        t90 = float(np.nanpercentile(trunk[seg], 90)) if np.isfinite(trunk[seg]).any() else np.nan
        tl90 = float(np.nanpercentile(lat[seg], 90)) if np.isfinite(lat[seg]).any() else np.nan
        x = hip[seg] @ np.nanmedian(ml[seg], 0)
        sw_ = (float((np.nanmax(x) - np.nanmin(x)) / leg * 100)
               if np.isfinite(x).sum() >= 3 else np.nan)
        for lst, v in ((dK, dk), (dH, dh), (al, s_al), (heel, hf), (tk, t90),
                       (tkl, tl90), (sway, sw_)):
            if np.isfinite(v):
                lst.append(v)
        rows.append({'cycle': n, 'start_frame': int(a + off), 'end_frame': int(b + off),
                     'knee_min_deg': _r(dk), 'hip_min_deg': _r(dh), 'shin_align_deg': _r(s_al),
                     'heel_lift_pct': _r(hf), 'trunk_p90_deg': _r(t90),
                     'trunk_lat_p90_deg': _r(tl90), 'hip_sway_lat_pct_leg': _r(sw_)})

    def med(v):
        return _r(float(np.median(v))) if v else None
    summary = {
        'frames': int(valid.sum()), 'duration_s': _r(T / fps),
        'hand_raise_frame': raise_f, 'assess_start_frame': start_f,
        'start_trigger': start_trigger or 'none',
        'trigger_found': None if not start_trigger else start_f is not None,
        'coverage': _r(valid.mean()),
        'shoulder_cm': _r(swm * 100), 'leg_cm': _r(leg * 100), 'scale_correction': _r(scale),
        'stance_ratio': _r(stance), 'stance_peak': _r(stance_peak),
        'did_perform': bool(len(reps) >= 1),
        'cycles': len(reps),
        # ── 準則五條 ──
        'squat_knee_min_deg': med(dK),             # 深蹲深度 25（23-25-27 膝角，主）
        'squat_hip_min_deg': med(dH),              # 同一列嘅真髖角（App 定義）
        'knee_align_deg': med(al),                 # 膝蓋對齊 25
        'heel_lift_pct': med(heel),                # 腳跟落地 20
        'trunk_p90_deg': med(tk),                  # 軀幹控制 15（3D 總傾角）
        'trunk_lat_p90_deg': med(tkl),             # 正面機位見到嘅左右分量
        'hip_sway_lat_pct_leg': med(sway),         # 整體穩定 15（ΔX_hip）
    }
    return summary, rows


def _longest_pause(v, fps, frac=0.20):
    """速度序列（已濾波）入面最長一段「慢過自己 p75 × frac」嘅時長（秒）同比例。"""
    ok = np.isfinite(v)
    if ok.sum() < 5:
        return np.nan, np.nan
    slow = ok & (v < frac * np.nanpercentile(v, 75))
    run = best = 0
    for x, o in zip(slow, ok):
        run = run + 1 if (x and o) else 0
        best = max(best, run)
    return best / fps, float(slow.sum() / ok.sum())


def _smooth_speed(P, fps):
    v = np.linalg.norm(np.diff(P, axis=0), axis=1) * fps
    sm = v.copy()
    for i in range(2, len(v) - 2):
        sm[i] = np.nanmedian(v[i - 2:i + 3])
    return sm


BEND_ENTER = 45      # 下肢伸展：軀幹離垂直 > 呢個先算開始彎
BEND_EXIT = 25       # < 呢個先算返上嚟（遲滯，防 35° 附近抖動斬碎）


def _fold_stretch(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                  off, stance, stance_peak, raise_f, start_f, start_trigger):
    """K1 下肢伸展 —— 《K1_Physical_Assessment_Criteria_Optimized.md》上學期 伸展 第 3 條。

    | 指標         | 準則計算方式      | K1 良好標準 | 分 |
    |--------------|-------------------|-------------|----|
    | 腿部伸展幅度 | 腿部伸直角度      | 伸展明顯    | 25 |
    | 支撐穩定     | 支撐側 ΔX_hip     | < 12 cm     | 20 |
    | 軀幹控制     | θ_vert            | < 25°       | 20 |
    | 左右對稱     | 左右伸展差異率    | < 20%       | 20 |
    | 動作連貫性   | t_pause           | < 0.6 s     | 15 |

    2026-09-24 睇片（rec_20260918_152944）：細路跟老師**雙手舉高 → 向前彎腰摸腳**，
    重複。所以一次 = 軀幹離垂直 > 50° 嘅一段（向前彎）。
      * 腿部伸展幅度：彎到最低時膝角（直腳 = 180°，越直越好）＋手離腳踝幾遠
      * 軀幹控制：正面機位嘅 θ_vert 只見到左右傾（同 K3 深蹲跳），彎腰本身唔扣分
      * 左右對稱：彎到最低時左右膝角差異率
    """
    T = len(K)
    shank = np.nanmedian(np.linalg.norm(K[:, LKN] - K[:, LAN], axis=1))
    leg = float(thigh + shank)
    ml, _ap = _body_axes(K, up)
    kL = np.array([_ang(K[i, LHIP], K[i, LKN], K[i, LAN]) for i in range(T)])
    kR = np.array([_ang(K[i, RHIP], K[i, RKN], K[i, RAN]) for i in range(T)])
    tv = sh - hip
    tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    # 左右傾 = 軀幹偏離「前後＋垂直」平面嘅角度（asin |左右分量|）。
    # 唔可以用 arctan2(左右, 垂直)：彎腰時垂直分量趨近 0，角度爆大
    # （2026-09-24 實測中位 28°、p90 64°，全部係彎腰假象）。
    lat = np.degrees(np.arcsin(np.clip(np.abs((tv * ml).sum(-1)), 0, 1)))
    # 分次用遲滯：軀幹 > 45° 先算開始彎，< 25° 先算返上嚟完成一次。
    # 單一門檻（> 35°）會喺 35° 附近抖動時將一次彎腰斬成幾段 —— 用戶示範 153319#1
    # 讀到 10 次、而且一半「冇返上嚟」，其實係碎片。
    reps, down, lo, done = [], False, 0, []
    for i in range(T):
        t = trunk[i]
        if not np.isfinite(t):
            continue
        if not down and t > BEND_ENTER:
            down, lo = True, i
        elif down and t < BEND_EXIT:
            if i - lo >= max(3, int(0.3 * fps)):
                reps.append((lo, i - 1))
                done.append(True)
            down = False
    if down and T - lo >= max(3, int(0.3 * fps)):
        reps.append((lo, T - 1))                 # 片尾仲彎住 = 未返上嚟
        done.append(False)
    wr = np.nanmean(K[:, [LWR, RWR]], 1)
    an = np.nanmean(K[:, [LAN, RAN]], 1)
    kn, asym, reach, sway, tl, flex, ret, feet = [], [], [], [], [], [], [], []
    rows = []
    for n, (a, b) in enumerate(reps):
        seg = np.arange(a, b + 1)
        # 彎腰深度（軀幹離垂直最大角度）：手腕貼地時成日被身體／腳遮住，三角化唔到
        # （2026-09-24 用戶示範嘅女仔 153319#1 十次彎腰得一次有手腕讀數），所以主指標用
        # 軀幹屈曲，手腕離地只做旁證。
        fx = float(np.nanmax(trunk[seg])) if np.isfinite(trunk[seg]).any() else np.nan
        if np.isfinite(fx):
            flex.append(fx)
        ret.append(done[n])                           # 「彎落去再返上嚟」（遲滯分次完成）
        t = trunk[seg]
        deep = seg[np.argsort(np.nan_to_num(-t, nan=0))[:max(1, len(seg) // 3)]]
        l, r = np.nanmedian(kL[deep]), np.nanmedian(kR[deep])
        k_ = np.nanmean([l, r])
        # 摸地程度 = 手腕離地高度 ÷ 腿長（0 = 摸到地）。2026-09-24 用戶示範：153319
        # 老師左手邊嘅女仔「彎得好好、摸到地、同時企得穩」。舊定義（手腕離腳踝距離）
        # 會將手放喺小腿嘅細路都當成「好近」。地面 = 企直時腳踝高度。
        floor = np.nanmedian(np.fmin(h(K[:, LAN]), h(K[:, RAN]))[np.nan_to_num(trunk, nan=90) < 20]) \
            if (np.nan_to_num(trunk, nan=90) < 20).sum() >= 5 else np.nanmedian(np.fmin(h(K[:, LAN]), h(K[:, RAN])))
        rch = float(np.nanmin(h(K[deep][:, [LWR, RWR]]).min(1) - floor)) / leg * 100
        # 平衡只計「彎到最低」嗰段（最深三分一幀）：用戶講「摸到地同時企得穩」。
        # 計成次（連返上嚟、舉高手）會混入正常嘅起身動作 —— 示範女仔 153319#1 排第 28 百分位。
        x = hip[deep] @ np.nanmedian(ml[deep], 0)
        sw_ = (float((np.nanpercentile(x, 90) - np.nanpercentile(x, 10)) / leg * 100)
               if np.isfinite(x).sum() >= 3 else np.nan)
        t90 = float(np.nanpercentile(lat[deep], 90)) if np.isfinite(lat[deep]).any() else np.nan
        # 腳企唔企得穩：成次彎腰期間兩隻腳踝水平位移嘅最大值 ÷ 腿長（踏步／移腳＝失平衡）。
        # md 支撐穩定列嘅關鍵點係 23,24＋29/30（髖＋腳跟），唔淨係髖。
        fm = []
        for aj in (LAN, RAN):
            f_ = K[seg, aj] - np.outer(h(K[seg, aj]), up)
            f_ = f_[np.isfinite(f_).all(1)]
            if len(f_) >= 3:
                fm.append(float(np.max(np.linalg.norm(f_ - np.median(f_, 0), axis=1)) / leg * 100))
        if fm:
            feet.append(max(fm))
        for lst, v in ((kn, k_), (reach, rch), (sway, sw_), (tl, t90)):
            if np.isfinite(v):
                lst.append(float(v))
        if np.isfinite(l) and np.isfinite(r):
            asym.append(_asym(180 - l + 1, 180 - r + 1))
        rows.append({'cycle': n, 'start_frame': int(a + off), 'end_frame': int(b + off),
                     'knee_deep_deg': _r(k_), 'knee_L_deg': _r(l), 'knee_R_deg': _r(r),
                     'reach_pct_leg': _r(rch), 'hip_sway_lat_pct_leg': _r(sw_),
                     'trunk_lat_p90_deg': _r(t90)})
    longest = stall = np.nan
    if len(reps) >= 2:
        w = slice(reps[0][0], reps[-1][1] + 1)
        # 郁動速度：手腕＋髖（伸展主要係上身同手）
        P = np.concatenate([wr[w], hip[w]], 1)
        longest, stall = _longest_pause(_smooth_speed(P, fps), fps)

    def med(v):
        return _r(float(np.median(v))) if v else None
    summary = {
        'frames': int(valid.sum()), 'duration_s': _r(T / fps),
        'hand_raise_frame': raise_f, 'assess_start_frame': start_f,
        'start_trigger': start_trigger or 'none',
        'trigger_found': None if not start_trigger else start_f is not None,
        'coverage': _r(valid.mean()),
        'shoulder_cm': _r(np.nanmedian(np.linalg.norm(K[:, LSH] - K[:, RSH], axis=1)) * 100),
        'leg_cm': _r(leg * 100), 'scale_correction': _r(scale),
        'stance_ratio': _r(stance), 'stance_peak': _r(stance_peak),
        'did_perform': bool(len(reps) >= 1),
        'cycles': len(reps),
        # ── 準則五條 ──
        'fold_depth_deg': med(flex),                # 腿部伸展幅度 25（彎腰深度，越大越好）
        'fold_knee_deg': med(kn),                   # 膝直程度（旁證）
        'fold_return_pct': _r(100 * float(np.mean(ret))) if ret else None,  # 動作連貫性 15（彎落再返上）
        'fold_reach_pct_leg': med(reach),           # 手腕離地 ÷ 腿長（越細＝摸得越落，0＝摸地）
        'hip_sway_lat_pct_leg': med(sway),          # 支撐穩定（髖，彎到最低時）
        'feet_move_pct_leg': med(feet),             # 支撐穩定 20（腳踝移位）
        'trunk_lat_p90_deg': med(tl),               # 軀幹控制 20（左右傾）
        'knee_asym_pct': med(asym),                 # 左右對稱 20（彎曲量差異率）
        'longest_pause_s': _r(longest),             # 動作連貫性 15
        'stall_fraction': _r(stall),
    }
    return summary, rows


def _hurdle_weave(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                  off, stance, stance_peak, raise_f, start_f, start_trigger, hurdles=None):
    """K1 繞過小欄杆 —— 《K1_Physical_Assessment_Criteria_Optimized.md》上學期 靈活移動 第 3 條
    （身體面向前方橫移繞過；步伐規律以髖部＋膝蓋判斷）。

    | 指標           | 準則計算方式                | K1 良好標準     | 分 |
    |----------------|-----------------------------|-----------------|----|
    | 轉彎流暢度     | 轉彎時 t_pause + 髖部旋轉   | 無明顯長時間停頓 | 25 |
    | 與欄杆距離     | D_body-hurdle + IsContact   | 適當距離冇碰撞   | 25 |
    | 身體穩定       | 轉彎後 ΔX_hip               | < 15 cm         | 20 |
    | 步伐規律性     | 橫移時髖部位移規律          | 規律性尚可       | 15 |
    | 有方向的流暢度 | 整體移動 t_pause            | < 0.6 s         | 15 |

    轉彎 = 髖部垂直於前進方向嘅左右位移轉向（繞欄杆＝左右左右）。
    ⚠ 與欄杆距離要欄杆 3D（equipment 模型冇 hurdle 類）—— 暫時出 None。
    """
    T = len(K)
    shank = np.nanmedian(np.linalg.norm(K[:, LKN] - K[:, LAN], axis=1))
    leg = float(thigh + shank)
    hz = hip - np.outer(h(hip), up)
    ok = np.isfinite(hz).all(1)
    longest = stall = turn_pause = post_sway = np.nan
    n_turns = 0
    cyc, n_short = [], 0
    travel = np.nan
    falls, fall_s = 0, 0.0
    if ok.sum() >= 20:
        P = hz[ok]
        c = P.mean(0)
        _, _, vt = np.linalg.svd(P - c)
        u = vt[0]
        nrm = np.cross(up, u)
        prog = (hz - c) @ u
        side = (hz - c) @ nrm
        # 評估窗口：沿前進方向由 5% 行到 95%
        pr = prog.copy()
        lo, hi = np.nanpercentile(pr, 5), np.nanpercentile(pr, 95)
        travel = float((hi - lo) / leg * 100)
        idx = np.where(np.isfinite(pr) & (pr >= lo) & (pr <= hi))[0]
        if len(idx) >= 20:
            W = np.arange(idx[0], idx[-1] + 1)
            sm = side[W].copy()
            for i in range(3, len(sm) - 3):
                sm[i] = np.nanmedian(side[W][i - 3:i + 4])
            d = np.sign(np.diff(sm))
            amp = 0.05 * leg
            turns, last, ext = [], None, sm[0]
            for i in range(1, len(sm)):
                if not np.isfinite(sm[i]):
                    continue
                if last is None:
                    last = 1 if sm[i] > ext else -1 if sm[i] < ext else None
                    ext = sm[i]
                    continue
                if (sm[i] - ext) * last >= 0:
                    ext = sm[i] if (sm[i] - ext) * last > 0 else ext
                elif abs(sm[i] - ext) >= amp:        # 轉咗向，而且幅度夠
                    turns.append(W[i])
                    last, ext = -last, sm[i]
            n_turns = len(turns)
            v = _smooth_speed(hz[W], fps)
            longest, stall = _longest_pause(v, fps)
            # 2026-09-24 用「最佳示範」rec_20260918_144607 核對後改定義：
            #  * 轉彎流暢度：細路轉彎唔會完全停，係減速 —— 用「轉彎 ±0.25s 平均速度
            #    ÷ 全程中位速度」（1 = 冇減速，越高越順）。舊定義（轉彎時停頓秒數）
            #    31 位 p90 都得 0.05s，分唔到人。
            #  * 身體穩定（轉彎後 ΔX_hip）：髖部左右位置減去平滑曲線嘅高頻抖動。舊定義量
            #    軀幹左右擺幅 —— 繞得快嘅細路自然向彎內傾，等於懲罰做得好嘅（最佳示範
            #    排第 87 百分位）。
            vmed = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
            sm2 = side[W].copy()
            for i in range(4, len(sm2) - 4):
                sm2[i] = np.nanmedian(side[W][i - 4:i + 5])
            jit = side[W] - sm2
            tp, ps = [], []
            for t in turns:
                a, b = max(W[0], t - int(0.25 * fps)), min(W[-1], t + int(0.25 * fps))
                seg = v[a - W[0]:b - W[0]]
                if np.isfinite(seg).any() and vmed > 1e-6:
                    tp.append(float(np.nanmean(seg) / vmed))
                w2 = np.arange(t - W[0], min(len(W), t - W[0] + int(0.5 * fps)))
                x = jit[w2]
                if np.isfinite(x).sum() >= 3:
                    ps.append(float((np.nanpercentile(x, 90) - np.nanpercentile(x, 10)) / leg * 100))
            turn_pause = float(np.median(tp)) if tp else np.nan
            post_sway = float(np.median(ps)) if ps else np.nan
            lift = np.degrees(np.arccos(np.clip((h(hip) - h(K[:, LKN])) / max(thigh, 1e-6), -1, 1)))
            cyc, n_short = _cycles(lift[W], fps)
            # 身體穩定（2026-09-25 用戶定義：轉彎時能否保持平衡，例如有細路轉彎跌落地）：
            # 跌倒 = 髖高（離地）< 企直髖高 60% 持續 ≥ 0.5s。睇片核：144136#0 跌落地
            # 四腳爬 1.6s；其餘 30 位最長 0.3s（3D 抖動，片中全部企直）。
            floor = np.nanpercentile(np.fmin(h(K[:, LAN]), h(K[:, RAN])), 10)
            hh = h(hip)[W] - floor
            stand_h = np.nanpercentile(hh, 75)
            low = np.nan_to_num(hh / stand_h, nan=1.0) < 0.6
            run = 0
            for x in list(low) + [False]:
                if x:
                    run += 1
                    continue
                if run >= int(0.5 * fps):
                    falls += 1
                    fall_s = max(fall_s, run / fps)
                run = 0
    # ---- 與欄杆距離（hurdle3d.py：紅色欄杆地面線段）----
    # 每個欄杆：細路喺佢沿線 ±0.5m 範圍入面時，腳踝（水平）離欄杆線段最近距離；取中位。
    # 碰撞 = 任一腳踝水平距離 < 15% 腿長（腳踝關節離腳掌前端約一個腳掌長）持續 ≥ 2 幀。
    clear = contacts = n_h = None
    if hurdles is not None and len(hurdles):
        H = np.asarray(hurdles, float) * scale
        H = H - np.einsum('nij,j->ni', H, up)[..., None] * up          # 投落地面
        A0, A1 = H[:, 0], H[:, 1]
        ank = [K[:, j] - np.outer(h(K[:, j]), up) for j in (LAN, RAN)]
        def seg_dist(P):                                              # (T,3) → (T,N)
            d = A1 - A0
            tt = np.clip(((P[:, None, :] - A0) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-9), 0, 1)
            return np.linalg.norm(P[:, None, :] - (A0 + tt[..., None] * d), axis=-1)
        D = np.fmin(seg_dist(ank[0]), seg_dist(ank[1]))               # (T,N)
        mids = (A0 + A1) / 2
        hz2 = hip - np.outer(h(hip), up)
        near = np.linalg.norm(hz2[:, None, :] - mids, axis=-1) < 0.5 * (leg / 0.55)
        per_h = []
        for j in range(len(H)):
            m_ = near[:, j] & np.isfinite(D[:, j])
            if m_.sum() >= 3:
                per_h.append(float(np.nanmin(D[m_, j])))
        n_h = len(per_h)
        if per_h:
            clear = float(np.median(per_h) / leg * 100)
        hit = np.nan_to_num(D.min(1), nan=9) < 0.15 * leg
        contacts, run = 0, 0
        for x in hit:
            run = run + 1 if x else 0
            if run == 2:
                contacts += 1
    per = [x[2] for x in cyc]
    summary = {
        'frames': int(valid.sum()), 'duration_s': _r(T / fps),
        'hand_raise_frame': raise_f, 'assess_start_frame': start_f,
        'start_trigger': start_trigger or 'none',
        'trigger_found': None if not start_trigger else start_f is not None,
        'coverage': _r(valid.mean()),
        'shoulder_cm': _r(np.nanmedian(np.linalg.norm(K[:, LSH] - K[:, RSH], axis=1)) * 100),
        'leg_cm': _r(leg * 100), 'scale_correction': _r(scale),
        'stance_ratio': _r(stance), 'stance_peak': _r(stance_peak),
        'travel_pct_leg': _r(travel),
        'did_perform': bool(np.isfinite(travel) and travel >= 150 and n_turns >= 2),
        'cycles': len(cyc), 'cycles_dropped_short': int(n_short),
        'turns': int(n_turns),
        # ── 準則五條 ──
        'turn_speed_ratio': _r(turn_pause, 3),            # 轉彎流暢度 25（越高越順）
        'hurdle_clearance_pct_leg': _r(clear),            # 與欄杆距離 25（最近距離中位）
        'hurdle_contacts': contacts,                      # 碰欄杆次數
        'hurdles_passed': n_h,
        'falls': int(falls),                              # 身體穩定 20（跌倒次數）
        'fall_longest_s': _r(fall_s),
        'post_turn_jitter_pct_leg': _r(post_sway),        # 舊定義，唔計分
        'cycle_cv_pct': _r(np.std(per) / np.mean(per) * 100) if len(per) > 2 else None,  # 步伐規律性 15
        'longest_pause_s': _r(longest),                   # 有方向的流暢度 15
        'stall_fraction': _r(stall),
    }
    return summary, [{'cycle': i, 'start_frame': int(a + off), 'end_frame': int(b + off),
                      'duration_s': round(float(d), 3)} for i, (a, b, d) in enumerate(cyc)]


def _command(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
             off, stance, stance_peak, raise_f, start_f, start_trigger):
    """K1 指令反應動作（聲音）—— 《K1_Physical_Assessment_Criteria_Optimized.md》上學期
    快速反應 第 1 條。

    | 指標       | 準則計算方式     | K1 良好標準   | 分 |
    |------------|------------------|---------------|----|
    | 反應時間   | t_react          | < 1.4 s       | 25 |
    | 動作準確度 | 是否正確執行     | 完全／大致正確 | 25 |
    | 身體穩定   | 動作時 ΔX_hip    | < 15 cm       | 20 |
    | 軀幹控制   | θ_vert           | < 22°         | 15 |
    | 過渡連貫性 | t_pause          | < 0.6 s       | 15 |

    ⚠ 反應時間、動作準確度量唔到：Tapo 錄影冇音軌，唔知指令幾時出、叫咗咩動作。
    2026-09-24 試過用同一條片三個細路嘅「活動起點」互相對齊做代理，但靜止段
    只係大致同步，信唔過，所以唔出分。兩列出 None。
    身體穩定／軀幹控制只計「郁緊」嘅幀（全身關節平均速度 > 0.4 m/s）。
    過渡連貫性 = 郁動段之間靜止段嘅中位時長（跟指令之間嘅停頓）。
    """
    T = len(K)
    shank = np.nanmedian(np.linalg.norm(K[:, LKN] - K[:, LAN], axis=1))
    leg = float(thigh + shank)
    ml, _ap = _body_axes(K, up)
    sp = np.nanmean(np.linalg.norm(np.diff(K, axis=0), axis=2), 1) * fps
    sp = np.concatenate([[np.nan], sp])
    act = np.nan_to_num(sp, nan=0) > 0.4
    tv = sh - hip
    tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    lat = np.degrees(np.arctan2(np.abs((tv * ml).sum(-1)), (tv * up).sum(-1)))
    sway = tl = np.nan
    if act.sum() >= 10:
        x = hip[act] @ np.nanmedian(ml[act], 0)
        if np.isfinite(x).sum() >= 10:
            sway = float((np.nanpercentile(x, 90) - np.nanpercentile(x, 10)) / leg * 100)
        if np.isfinite(lat[act]).any():
            tl = float(np.nanpercentile(lat[act], 90))
    # 靜止段（≥ 0.2s），只計第一同最後一段郁動之間
    rests, i = [], 0
    ia = np.where(act)[0]
    if len(ia):
        a0, a1 = ia[0], ia[-1]
        i = a0
        while i <= a1:
            if act[i]:
                i += 1
                continue
            j = i
            while j + 1 <= a1 and not act[j + 1]:
                j += 1
            if j - i + 1 >= int(0.2 * fps):
                rests.append((j - i + 1) / fps)
            i = j + 1
    summary = {
        'frames': int(valid.sum()), 'duration_s': _r(T / fps),
        'hand_raise_frame': raise_f, 'assess_start_frame': start_f,
        'start_trigger': start_trigger or 'none',
        'trigger_found': None if not start_trigger else start_f is not None,
        'coverage': _r(valid.mean()),
        'shoulder_cm': _r(np.nanmedian(np.linalg.norm(K[:, LSH] - K[:, RSH], axis=1)) * 100),
        'leg_cm': _r(leg * 100), 'scale_correction': _r(scale),
        'stance_ratio': _r(stance), 'stance_peak': _r(stance_peak),
        'did_perform': bool(act.sum() >= int(2 * fps)),
        'cycles': len(rests),
        'active_s': _r(act.sum() / fps),
        # ── 準則五條 ──
        'react_time_s': None,                          # 反應時間 25（冇音軌）
        'action_correct_pct': None,                    # 動作準確度 25（唔知指令）
        'hip_sway_lat_pct_leg': _r(sway),              # 身體穩定 20
        'trunk_lat_p90_deg': _r(tl),                   # 軀幹控制 15
        'rest_between_s': _r(float(np.median(rests))) if rests else None,  # 過渡連貫性 15
        'rest_count': len(rests),
    }
    return summary, []


def _toss_catch(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                off, stance, stance_peak, raise_f, start_f, start_trigger, ball=None):
    """K1 雙手向上拋接 —— 《K1_Physical_Assessment_Criteria_Optimized.md》上學期 手腳協調
    第 1 條（物件：Ball BB，含短暫丟失容許；K1 用氣球）。

    | 指標       | 準則計算方式                       | K1 良好標準 | 分 |
    |------------|------------------------------------|-------------|----|
    | 拋球控制   | 拋球高度與方向穩定性               | 大致穩定    | 25 |
    | 接球成功   | 成功接住比例                       | 成功率較高  | 25 |
    | 不掉球     | 球落地後連續彈地≥2次未接回＝掉球   | 掉球次數少  | 20 |
    | 身體穩定   | ΔX_hip                             | < 15 cm     | 15 |
    | 動作連貫性 | t_pause                            | < 0.6 s     | 15 |

    球 = ball3d.py（--classes 0,1：氣球多數判 balloon、少數 ball）。事件由實測訊號定：
    揸住 = 球心離兩手腕中點 < 0.6 軀幹長；拋 = 揸住 → 離手，之後 1.5s 內升過肩 +0.3 軀幹長；
    飛行結束 = 返到手（接住）／跌到膝以下 ≥0.5s（掉）／4s 仲未返（失去）。
    2026-09-24 rec_20260918_094203#0 睇到清楚嘅拋－接循環；#1 大部分時間個氣球喺地下。
    """
    T = len(K)
    shank = np.nanmedian(np.linalg.norm(K[:, LKN] - K[:, LAN], axis=1))
    leg = float(thigh + shank)
    torso = max(float(np.nanmedian(h(sh) - h(hip))), 1e-6)
    ml, _ap = _body_axes(K, up)
    empty = {'ball_seen_pct': None, 'tosses': 0, 'catches': 0, 'drops': None,
             'catch_pct': None, 'toss_height_torso': None, 'toss_height_cv_pct': None,
             'toss_drift_pct_leg': None, 'hip_sway_lat_pct_leg': None,
             'catch_to_toss_s': None}
    base = {
        'frames': int(valid.sum()), 'duration_s': _r(T / fps),
        'hand_raise_frame': raise_f, 'assess_start_frame': start_f,
        'start_trigger': start_trigger or 'none',
        'trigger_found': None if not start_trigger else start_f is not None,
        'coverage': _r(valid.mean()),
        'shoulder_cm': _r(np.nanmedian(np.linalg.norm(K[:, LSH] - K[:, RSH], axis=1)) * 100),
        'leg_cm': _r(leg * 100), 'scale_correction': _r(scale),
        'stance_ratio': _r(stance), 'stance_peak': _r(stance_peak),
    }
    if ball is None:
        return {**base, 'did_perform': False, 'cycles': 0, **empty}, []
    B = np.asarray(ball[0], float)[off:off + T]
    if len(B) < T:
        B = np.vstack([B, np.full((T - len(B), 3), np.nan)])
    seen = np.isfinite(B).all(1)
    # 揸住氣球時手腕成日被個波遮住（2026-09-25：23 位中位只有 62% 幀有手腕，
    # 094318#1 右腕 0%），舊做法「兩腕平均 < 0.6 軀幹」—— 手腕冇讀數就當冇揸住，
    # 接球漏計。改：每隻手有手腕用手腕；手腕缺就用手肘，容許多 0.3 軀幹長（前臂）。
    # 試過而唔用：由手肘沿上臂外推一個前臂 —— 胸前抱波時前臂向前、上臂向下，
    # 外推落到髖位（094015#1 睇片接住嘅 3 次全部變 lost）。
    dh = []
    for e_, w_ in ((LEL, LWR), (REL, RWR)):
        dwr = np.linalg.norm(B - K[:, w_], axis=1) / torso
        del_ = np.linalg.norm(B - K[:, e_], axis=1) / torso - 0.3
        dh.append(np.where(np.isfinite(dwr), dwr, del_))
    dw = np.nanmin(np.stack(dh), 0)
    hgt = (h(B) - h(sh)) / torso                      # 相對肩高 ÷ 軀幹長
    # 揸住半徑 0.9 軀幹（2026-09-25）：大氣球抱喺胸前，手腕喺波兩側，離波心 0.7–0.9
    # 軀幹 —— 0.6 令 093718#3 由頭到尾唔算揸住（睇片佢拋咗兩次）。放寬半徑會將細拋
    # 黐埋做一段長揸住（094203#0 12→6），所以加高度閘：波高過肩 +0.3 軀幹就唔算揸住。
    held_raw = seen & (np.nan_to_num(dw, nan=9) < 0.9) & (np.nan_to_num(hgt, nan=9) <= 0.3)
    # 遲滯：飛行中手掠過個波會閃一兩幀「揸住」—— 實測 catches 中位 6、最多 18，
    # 接→再拋只隔 0.2s。先補 ≤3 幀嘅洞（偵測斷續），再剔走 <0.3s 嘅揸住段。
    held = held_raw.copy()
    i = 0
    while i < T:
        if not held[i]:
            j = i
            while j < T and not held[j]:
                j += 1
            if 0 < i and j < T and j - i <= 3:
                held[i:j] = True
            i = j
        else:
            i += 1
    i = 0
    while i < T:
        if held[i]:
            j = i
            while j < T and held[j]:
                j += 1
            if j - i < int(0.3 * fps):
                held[i:j] = False
            i = j
        else:
            i += 1
    low = seen & (np.nan_to_num(hgt, nan=9) < -1.5)    # 約膝以下
    rise_win, drop_need, lost_s = int(1.5 * fps), int(0.5 * fps), int(4 * fps)

    events, i = [], 1
    while i < T:
        if held[i - 1] and not held[i]:
            j_end = min(T, i + rise_win)
            # 拋 = 升過肩 +0.6 軀幹（約頭頂）。0.3 只到頭嘅高度，093257#1 後段 3.D 波位
            # 誤落佢個頭度（背向機）被當 3 次拋；睇片佢只拋 3 次、接 3 次，0.6 啱晒。
            up_ok = np.nan_to_num(hgt[i:j_end], nan=-9) > 0.6
            if up_ok.any():
                rel = i - 1
                k, outcome, lowrun = i, 'lost', 0
                while k < min(T, i + lost_s):
                    if held[k]:
                        outcome = 'catch'
                        break
                    lowrun = lowrun + 1 if low[k] else 0
                    if lowrun >= drop_need:
                        outcome = 'drop'
                        break
                    k += 1
                fl = np.arange(i, k)
                # 呢一程飛行入面要真係升過肩 +0.3 軀幹長先算一次拋 —— 之前用
                # 「離手後 1.5s 內」判，會將下一次真拋嘅高度記落前面細郁動度
                # （實測 094203#0 報 14 次，一半最高點低過肩；改咗之後 8 次，睇片約 7 次）
                if not len(fl) or np.nanmax(np.nan_to_num(hgt[fl], nan=-9)) <= 0.6:
                    i = max(k, i + 1)
                    continue
                apex = fl[np.nanargmax(np.nan_to_num(hgt[fl], nan=-9))]
                d = B[apex] - B[rel] if np.isfinite(B[[apex, rel]]).all() else np.full(3, np.nan)
                drift = float(np.linalg.norm(d - (d @ up) * up) / leg * 100) if np.isfinite(d).all() else np.nan
                events.append({'release': rel, 'end': k, 'outcome': outcome,
                               'height': float(np.nanmax(hgt[fl])) if len(fl) else np.nan,
                               'drift': drift})
                i = max(k, i + 1)
                continue
        i += 1
    # 掉球：任何時候（唔止拋完）球喺膝以下 ≥0.5s 而且冇揸住
    drops, run = 0, 0
    for x, hd in zip(low, held):
        run = run + 1 if (x and not hd) else 0
        if run == drop_need:
            drops += 1
    tosses = len(events)
    catches = sum(e['outcome'] == 'catch' for e in events)
    hs = [e['height'] for e in events if np.isfinite(e['height'])]
    dr = [e['drift'] for e in events if np.isfinite(e['drift'])]
    gaps = [(b['release'] - a['end']) / fps for a, b in zip(events, events[1:])
            if a['outcome'] == 'catch' and b['release'] > a['end']]
    sway = np.nan
    if events:
        w = slice(events[0]['release'], events[-1]['end'] + 1)
        x = hip[w] @ np.nanmedian(ml[w], 0)
        if np.isfinite(x).sum() >= 10:
            sway = float((np.nanpercentile(x, 90) - np.nanpercentile(x, 10)) / leg * 100)
    summary = {**base,
        'did_perform': bool(tosses >= 1), 'cycles': tosses,
        'ball_seen_pct': _r(100 * seen.mean()),
        # ── 準則五條 ──
        'toss_drift_pct_leg': _r(float(np.median(dr))) if dr else None,   # 拋球控制 25（方向）
        'toss_height_torso': _r(float(np.median(hs))) if hs else None,     # 拋球控制（高度）
        # 高度一致性用標準差（軀幹長）唔用 CV：高度係相對肩，平均近 0，CV 會爆（實測中位 127%）
        'toss_height_sd_torso': _r(float(np.std(hs)), 3) if len(hs) >= 3 else None,
        'tosses': tosses, 'catches': int(catches),
        'catch_pct': _r(100 * catches / tosses) if tosses else None,      # 接球成功 25
        'drops': int(drops),                                               # 不掉球 20
        'hip_sway_lat_pct_leg': _r(sway),                                  # 身體穩定 15
        'catch_to_toss_s': _r(float(np.median(gaps))) if gaps else None,   # 動作連貫性 15
    }
    rows = [{'cycle': n, 'start_frame': int(e['release'] + off), 'end_frame': int(e['end'] + off),
             'outcome': e['outcome'], 'height_torso': _r(e['height']), 'drift_pct_leg': _r(e['drift'])}
            for n, e in enumerate(events)]
    return summary, rows


def measure(xyz, valid, fps, scale, start_trigger='', min_lift=25.0,
            family='gait', min_drop=0.10, ball=None, ws=None,
            min_travel=2.0, up_fixed=None):
    """一位受測者嘅指標。xyz 已經係修正後嘅公尺。回傳 (summary, cycles)。

    up_fixed：floor_up.py 用全場（同一份標定）所有細路腳踝擬出嘅垂直軸。有就用佢，
    唔再逐位擬 —— 原地類逐位擬會歪 12–15°（見 floor_up.py）。
    """
    K_all = xyz.copy()
    K_all[~valid] = np.nan
    hip_all = np.nanmean(K_all[:, [LHIP, RHIP]], 1)
    sh_all = np.nanmean(K_all[:, [LSH, RSH]], 1)
    up = np.nanmean(sh_all - hip_all, 0)
    up /= np.linalg.norm(up)
    # 垂直基準改用地面擬合（軀幹方向只做初值同守門）—— 見 _up_from_floor
    if up_fixed is not None:
        up = np.asarray(up_fixed, float) / np.linalg.norm(up_fixed)
    else:
        _fu = _up_from_floor(K_all, up)
        if _fu is not None:
            up = _fu

    raise_f = start_f = None
    if start_trigger == 'hand_raise':
        raise_f, start_f = find_hand_raise(K_all, up, fps)
    elif start_trigger == 'run_start':
        def _h(p):
            return (p * up).sum(-1)
        horiz_all = hip_all - np.outer(_h(hip_all), up)
        start_f = find_run_start(horiz_all, up, fps)
    off = start_f or 0
    K = K_all[off:]
    valid = valid[off:]

    hip = np.nanmean(K[:, [LHIP, RHIP]], 1)
    sh = np.nanmean(K[:, [LSH, RSH]], 1)
    thigh = np.nanmedian(np.linalg.norm(K[:, LHIP] - K[:, LKN], axis=1))

    def h(p):
        return (p * up).sum(-1)

    # 原地穩定度：髖部喺水平面上離起點幾遠
    horiz = hip - np.outer(h(hip), up)
    first = np.where(np.isfinite(horiz).all(1))[0]
    drift = (np.linalg.norm(horiz - horiz[first[0]], axis=1)
             if len(first) else np.full(len(hip), np.nan))

    # 膝抬高：膝低過髖幾多（除大腿長）→ acos，0°=吊直、90°=大腿水平。
    # 同 LimbLiftTracker 同一個約定，所以準則門檻讀數可以直接比。
    lift = {}
    for side, kn in (('L', LKN), ('R', RKN)):
        lift[side] = np.degrees(np.arccos(
            np.clip((h(hip) - h(K[:, kn])) / max(thigh, 1e-6), -1.0, 1.0)))

    elbow = {}
    for side, (a, b, c) in (('L', (LSH, LEL, LWR)), ('R', (RSH, REL, RWR))):
        elbow[side] = np.array([_ang(K[i, a], K[i, b], K[i, c])
                                for i in range(len(K))])

    # 企定定坐低：(髖高 − 踝高) ÷ 軀幹長。實測 rec_20260915_100345 —— 真表演者
    # 1.36~1.54，三個坐喺地下睇嘅細路全部 NaN（坐埋一堆，腳踝成段重建唔到）。
    # 體型檢查捉唔到佢哋（同樣係幼稚園細路，肩寬一樣），要靠呢條。
    #
    # 兩個讀數：中位（成段片典型姿勢）同 p90（佢有冇試過企直）。守門用 p90 ——
    # 中位喺深蹲類動作會錯剔真表演者：實測 k3_squat_jump_with_ball 14 條片,
    # 29 位入面 8 位中位讀 0.34~0.98 被剔，但佢哋 p90 全部 1.34~1.81（企得直),
    # 當中仲有做咗 6、7 個循環嘅。跑步類唔受影響：k1_straight_run 數值門檻
    # 一位都冇捉到（9 位全部係 NaN），敏捷梯只得 1 位（0 循環）由剔變入。
    ank_h = np.nanmean(np.stack([h(K[:, LAN]), h(K[:, RAN])]), 0)
    _st = (h(hip) - ank_h) / max(np.nanmedian(h(sh) - h(hip)), 1e-6)
    stance = float(np.nanmedian(_st))
    stance_peak = float(np.nanpercentile(_st, 90)) if np.isfinite(_st).any() else np.nan

    trunk = np.degrees(np.arccos(np.clip(
        ((sh - hip) / np.linalg.norm(sh - hip, axis=1, keepdims=True)
         * up).sum(-1), -1.0, 1.0)))

    if family == 'zigzag':
        return _zigzag(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                       off, stance, stance_peak, raise_f, start_f,
                       start_trigger, min_travel, ball, scale)
    if family == 'toss_catch':
        return _toss_catch(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                           off, stance, stance_peak, raise_f, start_f, start_trigger, ball)
    if family == 'command':
        return _command(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                        off, stance, stance_peak, raise_f, start_f, start_trigger)
    if family == 'fold_stretch':
        return _fold_stretch(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                             off, stance, stance_peak, raise_f, start_f, start_trigger)
    if family == 'hurdle_weave':
        return _hurdle_weave(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                             off, stance, stance_peak, raise_f, start_f, start_trigger, ball)
    if family == 'squat':
        return _squat(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                      off, stance, stance_peak, raise_f, start_f, start_trigger)
    if family == 'leg_swing':
        return _leg_swing(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                          off, stance, stance_peak, raise_f, start_f, start_trigger)
    if family == 'line_walk':
        return _line_walk(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                          off, stance, stance_peak, raise_f, start_f, start_trigger,
                          ball)
    if family == 'rope_jump':
        return _rope_jump(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                          off, stance, stance_peak, raise_f, start_f, start_trigger)
    if family == 'squat_jump':
        return _squat_jump(K, valid, fps, scale, up, h, hip, sh, trunk, thigh,
                           off, stance, stance_peak, raise_f, start_f,
                           start_trigger, min_drop, ball, ws)

    # ---- 前置閘：佢到底有冇做過個動作 ----
    # _cycles 嘅幅度閘係「相對佢自己」，所以一個完全冇抬腿嘅細路，
    # 佢自己嗰啲細擺動照樣切到「循環」。實測 rec_20260915_100241（用戶標明
    # 左邊嗰個冇跑）：佢切到 7 個循環、報 85.7 步/分，而且因為企定唔郁，
    # 停頓 0.25s、stall 0.07 —— 兩條都靚過真係跑嗰個（0.60s / 0.19），
    # 按準則「動作連貫性」嗰 15 分佢反而攞滿分。
    # 所以要一個**絕對**門檻：膝抬峰值過唔到就當冇做，唔好出節奏類指標。
    # 用 p90 而唔係單幀最大值：實測標明「冇跑」嗰個細路單幀最大值去到 27.95°
    # （一幀好彩就過關），但 p90 只得 21.0°，而最低嘅真跑者 p90 都有 27.9°。
    # p90 亦係準則「膝蓋抬高」嗰列用嘅同一個統計量。
    with np.errstate(all='ignore'):
        pk = [np.nanpercentile(lift[s2], 90) for s2 in ('L', 'R')]
    peak = float(np.nanmax(pk)) if np.isfinite(pk).any() else float('nan')
    did = bool(np.isfinite(peak) and peak >= min_lift)

    cyc, n_short = _cycles(lift['L'], fps) if did else ([], 0)
    per = [c[2] for c in cyc]
    rows = []
    for i, (a, b, d) in enumerate(cyc):
        rows.append({
            'cycle': i, 'start_frame': a + off, 'end_frame': b + off,
            'duration_s': round(d, 3),
            'knee_lift_L_deg': _r(np.nanmax(lift['L'][a:b + 1])),
            'knee_lift_R_deg': _r(np.nanmax(lift['R'][a:b + 1])),
            'trunk_lean_deg': _r(np.nanmedian(trunk[a:b + 1])),
        })

    # 原地穩定度只計「跑緊」嗰段：第一個到最後一個膝抬循環。
    # 原本計到片尾 —— 2026-09-24 K1 原地跑實測 16 位入面 5 位讀 >100cm，
    # 係跑完行返去坐（或者行入位）嗰段，唔係原地跑時嘅飄移。
    if cyc:
        a0, b1 = cyc[0][0], cyc[-1][1] + 1
        seg = horiz[a0:b1]
        okh = np.isfinite(seg).all(1)
        drift_run = (float(np.nanmax(np.linalg.norm(seg[okh] - seg[okh][0], axis=1)))
                     if okh.sum() >= 2 else np.nan)
    else:
        drift_run = np.nan

    # 髖部水平速度（真 m/s —— 尺度已修正）
    spd = np.linalg.norm(np.diff(horiz, axis=0), axis=1) * fps
    if len(spd) >= 5:                     # 5 幀中位數濾波：逐幀差分喺 20fps
        sm = spd.copy()                   # 會放大關節抖動，令峰值虛高
        for i in range(2, len(spd) - 2):
            sm[i] = np.nanmedian(spd[i - 2:i + 3])
        spd = sm
    peak_spd = float(np.nanpercentile(spd, 95)) if np.isfinite(spd).any() else np.nan
    accel_s = np.nan
    if np.isfinite(peak_spd) and peak_spd > 0.8:      # 有跑先計得加速
        f20 = np.where(np.nan_to_num(spd, nan=0.0) >= 0.2 * peak_spd)[0]
        f80 = np.where(np.nan_to_num(spd, nan=0.0) >= 0.8 * peak_spd)[0]
        if len(f20) and len(f80) and f80[0] > f20[0]:
            accel_s = float((f80[0] - f20[0]) / fps)

    # 停頓：用「膝抬變率」而唔係髖部速度。原地跑髖部本來就唔郁
    # （p50 0.07–0.14 m/s），用髖速度會把成個原地動作當成停頓；
    # 膝抬變率對兩種動作都係「肢體有冇喺度郁」嘅直接訊號。
    # 門檻用佢自己 p75 嘅 20%，唔用絕對值（K3 衝刺 95°/s vs K1 原地 10–38°/s）。
    rate = np.abs(np.diff(np.nanmax(np.stack([lift['L'], lift['R']]), 0))) * fps
    stall = np.nan
    longest = np.nan
    if did and np.isfinite(rate).any():   # 冇做過就唔好出節奏類指標
        # ⚠ 缺幀（重建唔到）係「唔知」，唔係「停咗」。唔分開就會把
        # 追蹤斷開嗰段當成停頓 —— 實測 K3 敏捷梯 222 幀得 105 幀有骨架，
        # 用 nan→0 會讀出 5.85 秒「停頓」，其實係冇資料。
        ok = np.isfinite(rate)
        slow = ok & (rate < 0.20 * np.nanpercentile(rate, 75))
        stall = float(slow.sum() / max(ok.sum(), 1))
        run = best = 0
        for x, o in zip(slow, ok):
            run = run + 1 if (x and o) else 0
            best = max(best, run)
        longest = best / fps

    # 落地緩衝：觸地瞬間嘅膝角（K3 準則要 >140°）。觸地 = 腳踝最低點。
    land = []
    for kn, an in ((LKN, LAN), (RKN, RAN)):
        ah = h(K[:, an])
        for i in range(2, len(ah) - 2):
            w = ah[i - 2:i + 3]
            if np.isfinite(w).all() and ah[i] == w.min() and w.max() - w.min() > 0.02:
                a = _ang(K[i, LHIP if kn == LKN else RHIP], K[i, kn], K[i, an])
                if np.isfinite(a):
                    land.append(a)

    summary = {
        'frames': int(valid.sum()), 'duration_s': _r(len(K) / fps),
        'hand_raise_frame': raise_f, 'assess_start_frame': start_f,
        'start_trigger': start_trigger or 'none',
        'trigger_found': None if not start_trigger else start_f is not None,
        'coverage': _r(valid.mean()),
        'shoulder_cm': _r(np.nanmedian(
            np.linalg.norm(K[:, LSH] - K[:, RSH], axis=1)) * 100),
        'leg_cm': _r((thigh + np.nanmedian(
            np.linalg.norm(K[:, LKN] - K[:, LAN], axis=1))) * 100),
        'scale_correction': _r(scale),
        'stance_ratio': _r(stance),
        'stance_peak': _r(stance_peak),
        'did_perform': bool(did), 'peak_lift_deg': _r(peak),
        'min_lift_deg': min_lift,
        'drift_from_start_cm': _r(drift_run * 100),
        'drift_whole_cm': _r(np.nanmax(drift) * 100),
        'knee_lift_L_deg': _r(np.nanpercentile(lift['L'], 90)),
        'knee_lift_R_deg': _r(np.nanpercentile(lift['R'], 90)),
        'knee_lift_asym_pct': _r(_asym(np.nanpercentile(lift['L'], 90),
                                       np.nanpercentile(lift['R'], 90))),
        'elbow_rom_L_deg': _r(np.nanpercentile(elbow['L'], 90)
                              - np.nanpercentile(elbow['L'], 10)),
        'elbow_rom_R_deg': _r(np.nanpercentile(elbow['R'], 90)
                              - np.nanpercentile(elbow['R'], 10)),
        'elbow_asym_deg': _r(abs(np.nanmedian(elbow['L']) - np.nanmedian(elbow['R']))),
        'trunk_lean_deg': _r(np.nanmedian(trunk)),
        'cycles': len(cyc),
        # 有幅度、但短過 MIN_CYCLE_FRAMES 而掉棄嘅數目。大過 0 即係呢位嘅
        # 節奏讀數受取樣率限制，睇常模嗰陣要當佢係「量唔到」而唔係「跑得慢」。
        'cycles_dropped_short': n_short,
        'cadence_spm': _r(60.0 / np.median(per)) if per else None,
        'cycle_cv_pct': _r(np.std(per) / np.mean(per) * 100) if len(per) > 2 else None,
        'stall_fraction': _r(stall),
        'longest_pause_s': _r(longest),
        'peak_speed_mps': _r(peak_spd),
        'accel_time_s': _r(accel_s),
        'landing_knee_deg': _r(np.median(land)) if land else None,
    }
    return summary, rows


RUBRIC_MD = '/Users/bcm01032/game7/GAME7_K1-K3_評分準則總表.md'


def movement_label(key):
    """由評分準則總表查中文名。

    ⚠ 唔少 key 係舊名：例如 `k1_straight_run` 個動作 2026-08-18 已經由
    「直線跑」改成「原地跑」，但 key 冇跟住改。淨靠 key 認動作會誤導，
    所以 meta.json 一定要順手存埋總表嗰個標籤。
    """
    try:
        for line in open(RUBRIC_MD, encoding='utf-8'):
            if f'`{key}`' in line and line.startswith('|'):
                cells = [c.strip() for c in line.split('|')]
                for i, c in enumerate(cells):
                    if c == f'`{key}`':
                        return cells[i - 1]
    except OSError:
        pass
    return ''


def _r(v, n=2):
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def _asym(a, b):
    d = max(abs(a), abs(b))
    return abs(a - b) / d * 100 if d > 1e-9 else np.nan


def _standing_2d(k, sc, ratio, conf=0.3):
    """2D 企姿守門：(踝y − 髖y) ÷ (髖y − 肩y) ≥ ratio；睇唔到腳踝當唔合格。

    點解：場內坐滿觀眾（一排二十個細路）係自標定失敗嘅主因 —— 11 人全部失敗、
    2 人 5 秒成功。多活動共用場地（2026-09-18 K1 s11：繞欄杆／平衡線／指令反應／
    下肢伸展同一條線上）冇一個 ROI 框得乾淨，但坐低嘅人喺 2D 已經分得開：
    企 1.1–1.5、坐 0.07–0.47。
    """
    k = np.asarray(k, float)
    sc = np.asarray(sc, float)
    if sc[LSH] < conf or sc[RSH] < conf or sc[LHIP] < conf or sc[RHIP] < conf:
        return False
    an = [k[j] for j in (LAN, RAN) if sc[j] >= conf]
    if not an:
        return False
    sh_y = (k[LSH][1] + k[RSH][1]) / 2
    hip_y = (k[LHIP][1] + k[RHIP][1]) / 2
    t = hip_y - sh_y
    if t <= 1:
        return False
    return (max(p[1] for p in an) - hip_y) / t >= ratio


def _roi_for(roi, name, idx):
    """由 roi.json 揾返呢一路嘅多邊形。

    兩種 key 都要食得：框選工具出嘅係按排序位置嘅 cam01…cam04，
    早期人手寫嘅係影片全名 cam01_dev0。對唔上就當「唔過濾」——
    咁樣一個 typo 會靜靜咁令成條片唔過濾，所以搵唔到要嘈。
    """
    for k in (name, f'cam{idx + 1:02d}', f'cam{idx:02d}'):
        if k in roi:
            return roi[k]
    print(f'  ⚠ roi.json 入面搵唔到 {name}（試過 cam{idx + 1:02d}）—— 呢一路唔會過濾')
    return []


def _mark_suspect(subs, nT, family='gait'):
    """混入嘅大人／碎片軌跡 —— 標記，唔刪。

    ROI 框係影像空間，老師行入框就會照收。體型係最硬淨嘅分界：實測同一條片
    兩個 K1 細路讀到 42.7 / 42.1cm，老師 56.2cm（高 32%）。唔直接刪係因為
    刪咗就冇得覆查；build_norms 會排除。
    """
    # 肩寬重建唔到（None）嘅唔入中位 —— 實測前行擺腿 rec_20260916_151350 一條碎片冇肩寬，
    # 直接 median 會 TypeError 令成條片入庫失敗
    # 參考只由「夠長」嘅軌跡計：碎片肩寬好唔穩（實測平衡線 rec_20260918_151432 一條
    # 40 幀碎片讀 22.6cm，把 p25 拉到 32.9，令真細路 43.3cm 被當成體型偏離）
    shs = [x['raw_shoulder_cm'] for x in subs if x.get('raw_shoulder_cm') is not None
           and x['frames'] >= 0.2 * nT]
    if len(shs) < 2:
        shs = [x['raw_shoulder_cm'] for x in subs if x.get('raw_shoulder_cm') is not None]
    # 參考體型用 p25 唔用中位：大人一定係大嗰個。實測 K1 繞欄杆 rec_20260918_144715
    # 得兩個人（大人 62.1cm、細路 35.7cm），中位 48.9 啱啱喺中間，兩個都偏離 >25%，
    # 真細路一齊被剔。p25 喺「好多細路＋一個大人」同「一大一細」都落喺細路體型。
    med = float(np.percentile(shs, 25)) if len(shs) >= 2 else None
    for x in subs:
        why = []
        # 1. 唔係企住 —— 坐喺地下睇嘅細路（腳踝成段重建唔到 → NaN）。
        #    用 p90 唔用中位：蹲類動作嘅真表演者中位好低，但一定企直過。
        #    舊資料冇 stance_peak，回退用中位（行為同以前一樣）。
        _sp = x.get('stance_peak', x.get('stance_ratio'))
        if _sp is None or _sp < STAND_MIN:
            why.append(f"唔係企住（髖−踝/軀幹 p90 = {_sp}）")
        # 2. 體型偏離 —— 混入嘅大人
        if x.get('raw_shoulder_cm') is None:
            why.append('肩寬重建唔到')
        elif med is not None and abs(x['raw_shoulder_cm'] - med) / max(med, 1e-6) > 0.25:
            why.append(f'體型偏離同場參考（p25）{med:.1f}cm')
        if x['frames'] < 0.2 * nT:
            why.append(f"只得 {x['frames']}/{nT} 幀")
        # 3. 原地類嘅碎片：表演者由頭到尾都喺度，唔夠一半幀 = 跨機位認錯人
        #    斷開嘅碎片。實測跳繩 rec_20260915_102255：黃衫細路企得遠、喺 cam01
        #    同綠衫重疊，斷成 3 段（428/431/216 幀，當中兩段同時存在），
        #    腿長 63cm（同場其他 72cm）、企姿 2.0（細路 1.6–1.8）—— 讀數全部唔可信。
        #    移動類唔用：細路本來就行入行出。
        elif family in IN_PLACE and x['frames'] < 0.5 * nT:
            why.append(f"原地動作只得 {x['frames']}/{nT} 幀（碎片）")
        x['suspect'] = bool(why)
        x['suspect_reason'] = '、'.join(why)
        if why:
            print(f"  ⚠ {x['subject_ref']} 標為 suspect（{x['suspect_reason']}）"
                  f' —— 常模會排除')


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rec', required=True, help='含 cam*.mp4 嘅錄影資料夾')
    ap.add_argument('--roi', default='', help='表演區框（框選工具輸出）')
    ap.add_argument('--moving', type=int, default=0, metavar='N',
                    help='跑動型動作用呢個代替畫框：每視角只保留位移最大嘅 N 條'
                         '軌跡。排隊嗰班企喺跑道旁邊，影像框分唔開，但位移分得開'
                         '（實測跑手 0.47–0.69 畫面寬，第二名 0.00–0.17）')
    ap.add_argument('--jumping', type=int, default=0, metavar='N',
                    help='原地跳躍型動作（深蹲跳、開合跳）用呢個：每視角只保留'
                         '**垂直**行程最大嘅 N 條軌跡，行程除返佢自己軀幹長。'
                         '水平位移同旁觀者一樣近乎零，--moving 分唔開；'
                         '垂直行程實測表演者 1.43–1.85 個軀幹長，旁觀者 0.34–1.10')
    ap.add_argument('--standing', type=float, default=0.0, metavar='R',
                    help='只留 2D 企姿 (踝−髖)/(髖−肩) ≥ R 嘅人（剔坐低觀眾；可以唔畀 --roi）')
    ap.add_argument('--movement', required=True, help='例如 k1_straight_run')
    ap.add_argument('--grade', required=True, choices=list(SHOULDER_CM))
    ap.add_argument('--session', default='', help='場次代號，預設由 rec 名嘅日期')
    ap.add_argument('--calib', default='', help='重用同場之前解出嘅 calibration.json')
    ap.add_argument('--out', default=os.path.join(HERE, 'data'))
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--expect-cams', type=int, default=4,
                    help='預期幾多條 cam*.mp4（預設 4）；數目唔啱就唔跑')
    ap.add_argument('--min-drop', type=float, default=0.10,
                    help='深蹲跳前置閘：髖下沉量 ÷ 腿長（預設 0.10）')
    ap.add_argument('--min-lift', type=float, default=25.0,
                    help='膝抬峰值低過呢個角度就當「冇做過個動作」，'
                         '節奏類指標唔出數（預設 25°）')
    ap.add_argument('--remeasure', action='store_true',
                    help='唔重跑 3D，淨係由已有嘅 pose3d.npz 重算指標 —— '
                         '調門檻嗰陣用，幾秒搞掂')
    ap.add_argument('--start-trigger', default='', choices=['', 'hand_raise', 'run_start'],
                    help='hand_raise = 細路舉手先開始評測；'
                         'run_start = 由真正起跑嗰刻先計（跑動型動作）')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    rec_id = os.path.basename(os.path.normpath(a.rec))
    if a.remeasure:
        out = os.path.abspath(os.path.join(a.out, a.movement, rec_id))
        npz = os.path.join(out, 'pose3d.npz')
        if not os.path.exists(npz):
            sys.exit(f'✗ 未入過庫，冇得 --remeasure：{npz}')
        d = np.load(npz, allow_pickle=True)
        corr, valid, refs = d['xyz'], d['valid'], [str(x) for x in d['subjects']]
        meta = json.load(open(os.path.join(out, 'meta.json')))
        subs, cyc_all = [], {}
        fam = FAMILY.get(a.movement, 'gait')
        for i, ref in enumerate(refs):
            if valid[i].sum() < 30:
                continue
            k = meta.get('scale_corrections', {}).get(ref, 1.0)
            trg = ('' if a.movement in NO_TRIGGER else
                   a.start_trigger or meta.get('start_trigger', '').replace('none', ''))
            bl = (_load_ball(out, ref, k) if fam in ('squat_jump', 'toss_catch')
                  else _load_cones(out) if fam == 'zigzag'
                  else _load_line(out) if fam == 'line_walk'
                  else _load_hurdles(out) if fam == 'hurdle_weave' else None)
            ws = meta.get('world_scale_ball')
            summ, rows = measure(corr[i], valid[i], a.fps, k, trg,
                                 a.min_lift, fam, a.min_drop, bl, ws,
                                 up_fixed=_load_floor_up(out))
            summ['track_id'] = ref.split('#')[-1]
            summ['subject_ref'] = ref
            summ['raw_shoulder_cm'] = _r(SHOULDER_CM[meta['grade']] / k) if k else None
            subs.append(summ)
            cyc_all[ref] = rows
        _mark_suspect(subs, corr.shape[1], fam)
        for x in subs:
            gate = ''
            if not x['did_perform']:
                if fam == 'squat_jump':
                    gate = (f"  ⛔ 冇做過（髖下沉 {x.get('peak_hip_drop_ratio')} "
                            f"< {a.min_drop} × 腿長）")
                elif fam == 'zigzag':
                    gate = (f"  ⛔ 冇做過（行程 {x.get('travel_pct_leg')}% "
                            f"< {100 * x.get('min_travel_ratio', 0):.0f}% 腿長）")
                else:
                    gate = (f"  ⛔ 冇做過（峰值 {x.get('peak_lift_deg')}° "
                            f"< {a.min_lift}°）")
            print(f"  {x['subject_ref']}: {x['frames']} 幀、{x['cycles']} 個循環{gate}"
                  + ('  [suspect]' if x.get('suspect') else ''))
        # 舊 metrics.json 可能係寫到一半失敗嘅（例如 int64 唔 serializable），
        # 重算只靠 pose3d.npz + meta.json，唔需要佢
        json.dump({'subjects': subs, 'cycles': cyc_all},
                  open(os.path.join(out, 'metrics.json'), 'w'),
                  ensure_ascii=False, indent=1)
        meta['min_lift_deg'] = a.min_lift
        if a.start_trigger:
            meta['start_trigger'] = a.start_trigger
        json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'),
                  ensure_ascii=False, indent=1)
        print(f'✅ 重算完成（未重跑 3D）→ {out}')
        return


    out = os.path.abspath(os.path.join(a.out, a.movement, rec_id))
    if os.path.exists(os.path.join(out, 'pose3d.npz')) and not a.force:
        sys.exit(f'✓ 已經處理過（加 --force 重做）：{out}')
    os.makedirs(out, exist_ok=True)
    work = os.path.abspath(os.path.join(a.out, '_work', rec_id))
    os.makedirs(work, exist_ok=True)

    # ---- 相機數必須齊 ----
    # 少咗一條唔會報錯，但後果好嚴重：重用嘅 calibration.json 係按相機次序
    # 0–3 對應嘅，少一條就成個機位對應錯晒，而三角化照樣出「睇落合理」嘅 3D。
    # （之前試過 Google Drive 匯出空資料夾／只落到部分檔案。）
    import glob as _g
    cams = sorted(_g.glob(os.path.join(a.rec, 'cam*.mp4')))
    if len(cams) != a.expect_cams:
        names = '、'.join(os.path.basename(c) for c in cams) or '(冇)'
        sys.exit(f'✗ 應該有 {a.expect_cams} 條 cam*.mp4，實際得 {len(cams)} 條：{names}\n'
                 f'   下載未齊嘅話補返齊先；真係得少過四機就用 --expect-cams 明確指定，'
                 f'但唔可以重用四機嘅標定。')
    print(f'▶ 1/5 2D 檢測  {rec_id}（{len(cams)} 機）')
    cache = detect(a.rec, os.path.join(work, 'dets'))

    print('▶ 2/5 表演區過濾')
    filt = os.path.abspath(os.path.join(work, "filtered"))
    sys.path.insert(0, HERE)
    import roi_filter as RF
    import glob
    if not a.roi and not a.moving and not a.jumping and not a.standing:
        sys.exit('✗ 要 --roi（畫框）、--moving N（跑動型）、--jumping N（原地跳躍型）'
                 '或者 --standing R（只留企住嘅人）')
    if a.moving and a.jumping:
        sys.exit('✗ --moving 同 --jumping 只可以揀一個')
    roi = json.load(open(a.roi))['regions'] if a.roi else {}
    os.makedirs(filt, exist_ok=True)
    kept_stats = {}
    for idx, f in enumerate(sorted(glob.glob(os.path.join(cache, 'cam*.json')))):
        d = json.load(open(f))
        meta = d['meta']
        name, W, H = meta['video'], meta['W'], meta['H']
        poly = _roi_for(roi, name, idx) if a.roi else []
        if a.moving:
            want, top = RF.moving_tracks(d['instance_info'], W, 0.25, a.moving)
        elif a.jumping:
            want, top = RF.jumping_tracks(d['instance_info'], W, 1.2, a.jumping)
        else:
            want, top = None, None
        new, kept, tot = [], 0, 0
        for fr in d['instance_info']:
            keep = []
            for j, i in enumerate(fr['instances']):
                tot += 1
                if want is not None:
                    ok = j in want.get(fr['frame_id'], set())
                else:
                    hm = RF.hip_mid(i['keypoints'], i['keypoint_scores'], W, H)
                    ok = hm is not None and RF.inside(hm, poly)
                if ok and a.standing:
                    ok = _standing_2d(i['keypoints'], i['keypoint_scores'], a.standing)
                if ok:
                    keep.append(i)
                    kept += 1
            new.append({'frame_id': fr['frame_id'], 'instances': keep})
        if top:
            lbl = ('垂直行程最大四條（÷自己軀幹長）' if a.jumping
                   else '位移最大四條（÷畫面寬）')
            print(f'    {lbl}：{top}')
        json.dump({'meta': meta, 'instance_info': new},
                  open(os.path.join(filt, name + '.json'), 'w'))
        dst = os.path.join(filt, name + '.mp4')
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(os.path.join(a.rec, name + '.mp4')), dst)
        n = [len(x['instances']) for x in new]
        kept_stats[name] = {'kept': kept, 'total': tot,
                            'median_people': int(np.median(n)), 'max_people': max(n)}
        print(f'  {name}: 保留 {kept}/{tot} → 每幀中位 {int(np.median(n))} 人')

    calib = os.path.abspath(a.calib) if a.calib else ''
    if calib:
        print(f'▶ 3/5 重用標定  {os.path.basename(calib)}')
    else:
        print('▶ 3/5 自標定（循環播片直到解出機位）')
        cal_dir = os.path.join(work, 'calib')
        os.makedirs(cal_dir, exist_ok=True)
        ok, _ = _session(filt, cal_dir, fixed=None, loop=True, speed=1.0)
        calib = os.path.join(cal_dir, 'calibration.json')
        if not ok or not os.path.exists(calib):
            sys.exit('✗ 自標定失敗 —— 通常係表演區框太鬆（旁觀者入咗框）'
                     '或者機位重疊。收窄個框再試，或者用同場另一條片嘅 --calib。')
    dst_cal = os.path.join(out, 'calibration.json')
    if os.path.abspath(calib) != os.path.abspath(dst_cal):
        shutil.copy(calib, dst_cal)

    print('▶ 4/5 單次播完，逐幀 3D')
    ok, fidmap = _session(filt, work, fixed=calib, loop=False, speed=0.6, budget=900)
    pj = os.path.join(work, 'poses3d.json')
    if not os.path.exists(pj):
        sys.exit('✗ 冇出到 3D 骨架')
    poses = json.load(open(pj))

    # -------- 砌成陣列 --------
    vf = {int(k): v[0] for k, v in fidmap.items() if v and v[0] is not None}
    tids = sorted({t for p in poses.values() for t in p})
    nT = max(vf.values()) + 1 if vf else 0
    S = len(tids)
    xyz = np.full((S, nT, 17, 3), np.nan, np.float32)
    for fid, per in poses.items():
        f = vf.get(int(fid))
        if f is None:
            continue
        for tid, J in per.items():
            s = tids.index(tid)
            for j, p in enumerate(J):
                if p is not None:
                    xyz[s, f, j] = p
    valid = np.isfinite(xyz).all(-1).any(-1)

    print('▶ 5/5 尺度修正 + 指標')
    grade_cm = SHOULDER_CM[a.grade]
    subs, cyc_all = [], {}
    scales = np.ones(S, np.float64)          # 逐位受測者一個尺度修正
    fam = FAMILY.get(a.movement, 'gait')
    # 呢啲動作唔用起計訊號（見 NO_TRIGGER）——  即使命令列畀咗都唔理
    trg = '' if a.movement in NO_TRIGGER else a.start_trigger
    if a.start_trigger and not trg:
        print(f'  ⓘ {a.movement} 唔用起計訊號，已忽略 --start-trigger '
              f'{a.start_trigger}')
    for s, tid in enumerate(tids):
        if valid[s].sum() < 30:
            continue
        raw_sh = np.nanmedian(np.linalg.norm(
            xyz[s, valid[s], LSH] - xyz[s, valid[s], RSH], axis=1)) * 100
        k = grade_cm / raw_sh if raw_sh > 1e-6 else 1.0
        scales[s] = k
        bl = (_load_ball(out, f'{rec_id}#{tid}', k) if fam in ('squat_jump', 'toss_catch')
              else _load_cones(out) if fam == 'zigzag'
              else _load_line(out) if fam == 'line_walk'
              else _load_hurdles(out) if fam == 'hurdle_weave' else None)
        summ, rows = measure(xyz[s] * k, valid[s], a.fps, k, trg,
                             a.min_lift, fam, a.min_drop, bl, None,
                             up_fixed=_load_floor_up(out))
        summ['track_id'] = tid
        summ['subject_ref'] = f'{rec_id}#{tid}'
        summ['raw_shoulder_cm'] = _r(raw_sh)
        subs.append(summ)
        cyc_all[summ['subject_ref']] = rows
        note = ''
        if trg:
            note = (f"、舉手 f{summ['hand_raise_frame']} → 由 f{summ['assess_start_frame']} 開始"
                    if summ['trigger_found'] else '、⚠ 搵唔到舉手，用咗成條片')
        print(f"  {summ['subject_ref']}: {summ['frames']} 幀、"
              f"肩寬 {raw_sh:.1f}→{grade_cm}cm（×{k:.3f}）、"
              f"{summ['cycles']} 個循環{note}")

    _mark_suspect(subs, nT, fam)

    # 修正後嘅世界座標 + 髖為原點／肩寬正規化（跨身高可比，畀 model 學用）
    corr = xyz * scales[:, None, None, None].astype(np.float32)
    hip = np.nanmean(corr[:, :, [LHIP, RHIP]], 2)
    shw = np.nanmedian(np.linalg.norm(corr[:, :, LSH] - corr[:, :, RSH], axis=-1),
                       axis=1, keepdims=True)[..., None, None]
    norm = (corr - hip[:, :, None, :]) / np.where(shw > 1e-6, shw, np.nan)

    np.savez_compressed(
        os.path.join(out, 'pose3d.npz'),
        xyz=corr.astype(np.float32), xyz_norm=norm.astype(np.float32),
        valid=valid, subjects=np.array([f'{rec_id}#{t}' for t in tids]),
        frames=np.arange(nT), fps=a.fps, layout=np.array('COCO-17'))

    meta = {
        'rec_id': rec_id, 'source': os.path.abspath(a.rec),
        'session': a.session or rec_id.split('_')[1],
        'movement_key': a.movement, 'movement_label': movement_label(a.movement),
        'grade': a.grade,
        'fps': a.fps, 'frames': int(nT), 'n_subjects': len(subs),
        'roi': os.path.abspath(a.roi) if a.roi else None,
        'subject_filter': (f'moving_top{a.moving}' if a.moving else
                           f'jumping_top{a.jumping}' if a.jumping else 'roi_polygon'),
        'roi_kept': kept_stats,
        'calibration': 'calibration.json',
        'calibration_reused': bool(a.calib),
        'grade_shoulder_cm': grade_cm,
        'min_lift_deg': a.min_lift,
        'start_trigger': a.start_trigger or 'none',
        'scale_corrections': {f'{rec_id}#{t}': round(float(k), 4)
                              for t, k in zip(tids, scales)},
        'pipeline': 'human-selfcalib ComputeSession (YOLO11m-pose + 對極關聯 + 三角化)',
        'ingested_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'),
              ensure_ascii=False, indent=1)
    json.dump({'subjects': subs, 'cycles': cyc_all},
              open(os.path.join(out, 'metrics.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f'\n✅ 完成 → {out}')
    print(f'   pose3d.npz  xyz {corr.shape}（公尺）+ xyz_norm（肩寬正規化）')
    print(f'   metrics.json  {len(subs)} 位受測者、'
          f'{sum(len(v) for v in cyc_all.values())} 個循環')


if __name__ == '__main__':
    main()
