# -*- coding: utf-8 -*-
"""重解單一台機嘅外參（R, T）—— 用其他機三角化出嚟嘅 3D 骨架 ＋ 呢台機嘅 2D 偵測做 PnP。

點解要：2026-09-18 s11 自標定（calibration md5 b4b15c4b）cam02 解錯咗位，骨架投返
cam02 差 100–600px（其餘三台 3–10px），50 條片共用呢份標定。3D 由三台好機決定，
所以指標冇壞，但 cam02 疊圖錯晒，亦等於白白冇用第四台機。

做法：同一份標定（md5）嘅所有片，每幀 3D 受測者 × 2D 偵測人全部配對做候選對應，
solvePnPRansac 自己剔走配錯人嘅點（唔靠舊外參配對，因為舊外參本身就錯）。
鏡頭內參用棋盤格 K＋畸變（同 ball3d／review_video 一致）。

用法：
    python fix_cam_pose.py --md5 b4b15c4b --cam 1            # 睇結果
    python fix_cam_pose.py --md5 b4b15c4b --cam 1 --write    # 寫返所有片嘅 calibration.json
"""
import argparse, glob, hashlib, json, os
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CAMS = ['cam01_dev0', 'cam02_dev1', 'cam03_dev2', 'cam04_dev3']
CKB = '/Users/bcm01032/game7/human-selfcalib/calibs/tapo_c120_checkerboard.json'
BODY = [5, 6, 11, 12, 13, 14, 15, 16]      # 肩、髖、膝、踝：穩陣過手腕／面


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()[:8]


def clips(md5_):
    out = []
    for c in sorted(glob.glob(os.path.join(HERE, 'data', 'k*', 'rec_*', 'calibration.json'))):
        if md5(c) == md5_:
            out.append(os.path.dirname(c))
    return out


def reproj(D, v, R, T, Kc, dist, step=5):
    rec = os.path.basename(D)
    W = os.path.join(HERE, 'data', '_work', rec, 'dets', CAMS[v] + '.json')
    det = {fr['frame_id']: fr['instances'] for fr in json.load(open(W))['instance_info']}
    p = np.load(os.path.join(D, 'pose3d.npz'), allow_pickle=True)
    meta = json.load(open(os.path.join(D, 'meta.json')))
    sc = [meta.get('scale_corrections', {}).get(str(s), 1.0) for s in p['subjects']]
    rv = cv2.Rodrigues(R)[0]
    errs, pairs3, pairs2 = [], [], []
    for t in range(0, p['xyz'].shape[1], step):
        ins = [np.array(i['keypoints'])[:, :2] for i in det.get(t, [])]
        if not ins:
            continue
        for i in range(len(sc)):
            if not p['valid'][i][t]:
                continue
            X = p['xyz'][i][t] / sc[i]
            ok = np.isfinite(X).all(1)
            if ok.sum() < 8:
                continue
            q = cv2.projectPoints(X[ok], rv, T, Kc, dist)[0].reshape(-1, 2)
            e = [np.median(np.linalg.norm(k[ok] - q, axis=1)) for k in ins]
            errs.append(min(e))
    return float(np.median(errs)) if errs else np.nan


def gather(Ds, v, max_people=4, step=3):
    # 用 filtered/（表演區框過濾後、同影片幀號）—— dets/ 連觀眾每幀 8–12 人，
    # 全配對候選入面啱嘅得一成，RANSAC 撈唔返。
    P3, P2 = [], []
    for D in Ds:
        rec = os.path.basename(D)
        W = os.path.join(HERE, 'data', '_work', rec, 'filtered', CAMS[v] + '.json')
        if not os.path.exists(W):
            continue
        det = {fr['frame_id']: fr['instances'] for fr in json.load(open(W))['instance_info']}
        p = np.load(os.path.join(D, 'pose3d.npz'), allow_pickle=True)
        meta = json.load(open(os.path.join(D, 'meta.json')))
        sc = [meta.get('scale_corrections', {}).get(str(s), 1.0) for s in p['subjects']]
        for t in range(0, p['xyz'].shape[1], step):
            ins = det.get(t, [])
            if not ins or len(ins) > max_people:
                continue
            for i in range(len(sc)):
                if not p['valid'][i][t]:
                    continue
                X = p['xyz'][i][t] / sc[i]
                for k in ins:
                    kp = np.array(k['keypoints'])
                    for j in BODY:
                        if np.isfinite(X[j]).all():
                            P3.append(X[j])
                            P2.append(kp[j, :2])
    return np.array(P3, np.float64), np.array(P2, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md5', required=True)
    ap.add_argument('--cam', type=int, required=True, help='0-based 機號（cam02 = 1）')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    ckb = json.load(open(CKB))
    Kc = np.array(ckb['0']['K'], np.float64).reshape(3, 3)
    dist = np.array(ckb['_distCoeffs'], np.float64)
    Ds = clips(a.md5)
    print(f'{len(Ds)} 條片用緊標定 {a.md5}')
    P3, P2 = gather(Ds, a.cam)
    print(f'候選對應 {len(P3)}')
    ok, rv, T, inl = cv2.solvePnPRansac(P3, P2, Kc, dist, reprojectionError=12.0,
                                        iterationsCount=5000, confidence=0.999,
                                        flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise SystemExit('✗ PnP 失敗')
    inl = inl.ravel()
    rv, T = cv2.solvePnPRefineLM(P3[inl], P2[inl], Kc, dist, rv, T)
    R = cv2.Rodrigues(rv)[0]
    T = T.ravel()
    print(f'內點 {len(inl)}（{100 * len(inl) / len(P3):.0f}%）')
    cal0 = json.load(open(os.path.join(Ds[0], 'calibration.json')))
    R0 = np.array(cal0[str(a.cam)]['R']).reshape(3, 3)
    T0 = np.array(cal0[str(a.cam)]['T'])
    C0, C1 = -R0.T @ T0, -R.T @ T
    ang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1)))
    print(f'機位 舊 {np.round(C0, 2)} → 新 {np.round(C1, 2)}；轉角差 {ang:.1f}°')
    print('逐條片重投影中位（舊 → 新 px）：')
    for D in Ds:
        e0 = reproj(D, a.cam, R0, T0, Kc, dist)
        e1 = reproj(D, a.cam, R, T, Kc, dist)
        print(f'  {os.path.basename(os.path.dirname(D))[:24]:24s} {os.path.basename(D)}  {e0:6.0f} → {e1:5.1f}')
    if a.write:
        for D in Ds:
            p = os.path.join(D, 'calibration.json')
            cal = json.load(open(p))
            bak = p.replace('.json', f'.before_fix_cam{a.cam}.json')
            if not os.path.exists(bak):
                json.dump(cal, open(bak, 'w'), indent=1)
            cal[str(a.cam)]['R'] = [float(x) for x in R.ravel()]
            cal[str(a.cam)]['T'] = [float(x) for x in T]
            cal[str(a.cam)]['_fixed'] = (f'fix_cam_pose.py：PnP 用其他三台機嘅 3D 骨架重解外參'
                                         f'（原標定 md5 {a.md5}）')
            json.dump(cal, open(p, 'w'), indent=1)
        print(f'✅ 寫入 {len(Ds)} 份 calibration.json（舊版存做 *.before_fix_cam{a.cam}.json）')


if __name__ == '__main__':
    main()
