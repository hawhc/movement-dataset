# -*- coding: utf-8 -*-
"""踩雪糕筒判定嘅合成測試。

同 test_ball_drop.py 同一個理由：「不踩雪糕筒」嘅低分層（碰撞）好可能係零樣本
（39 位全部繞得開），零正樣本之下判定啱唔啱冇得靠數據驗。所以造幾個物理上
真確嘅情境，睇個判定響唔響、同埋收唔收得住聲。

四個情境（一個細路沿直線行，筒擺喺唔同位置）：
  A 踩正個筒      —— 腳踝路徑正中筒 → 要響
  B 繞開（0.6 腿長）—— 要收聲
  C 貼牆嘅筒      —— 離路徑好遠，唔應該當成課程障礙（cones_course_n 要剔走佢）
  D 擦身而過      —— 啱啱喺門檻邊（0.30 腿長）附近，記錄最近距離
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ingest as I  # noqa: E402

FPS = 20.0
LEG = 0.70          # 大腿+小腿，同實測 leg_cm ≈ 70 一致
TORSO = 0.44
UP = np.array([0.0, 0.0, 1.0])
CONE_RADIUS = 0.10        # 筒底半徑（米）——實作會用偵測量返嘅真半徑


def _skel(n=120, step=0.05):
    """沿 +x 行嘅骨架。回傳 (K, valid)。踝喺地面，髖喺 LEG 高。"""
    K = np.zeros((n, 17, 3))
    x = np.arange(n) * step
    for t in range(n):
        K[t, I.LSH] = [x[t], -0.11, LEG + TORSO]
        K[t, I.RSH] = [x[t], 0.11, LEG + TORSO]
        K[t, I.LHIP] = [x[t], -0.07, LEG]
        K[t, I.RHIP] = [x[t], 0.07, LEG]
        K[t, I.LKN] = [x[t], -0.07, LEG * 0.55]
        K[t, I.RKN] = [x[t], 0.07, LEG * 0.55]
        K[t, I.LAN] = [x[t], -0.07, 0.02]
        K[t, I.RAN] = [x[t], 0.07, 0.02]
    return K, np.ones(n, bool)


def run(cones):
    K, valid = _skel()
    hip = np.nanmean(K[:, [I.LHIP, I.RHIP]], 1)
    sh = np.nanmean(K[:, [I.LSH, I.RSH]], 1)

    def h(p):
        return (p * UP).sum(-1)

    trunk = np.zeros(len(K))
    summ, _ = I._zigzag(K, valid, FPS, 1.0, UP, h, hip, sh, trunk,
                        LEG * 0.55, 0, 1.5, 1.7, None, None, '', 2.0,
                        (np.array(cones, float),
                         np.full(len(cones), CONE_RADIUS)), 1.0)
    return summ


if __name__ == '__main__':
    mid = 60 * 0.05                       # 路徑中點嘅 x
    cases = {
        'A': ([[mid, -0.07, 0.0]], 1, u'踩正個筒（腳踝路徑正中）'),
        'B': ([[mid, -0.07 - 0.6 * LEG, 0.0]], 0, u'繞開 0.6 腿長'),
        # 冇課程筒 = **判唔到**，唔係「0 次碰撞」—— 報 0 等於白送滿分。
        'C': ([[mid, 6.0, 0.0]], None, u'貼牆嘅筒（離路徑 6m）'),
        'D': ([[mid, -0.07 - 0.45 * LEG, 0.0]], 0, u'擦身而過 0.45 腿長'),
    }
    bad = 0
    for k in 'ABCD':
        cones, want, desc = cases[k]
        s = run(cones)
        got = s['cone_contacts']
        n = s['cones_course_n']
        ok = (got == want)
        if k == 'C':
            ok = (got is None) and n == 0   # 剔出課程筒 + 報「判唔到」
        bad += 0 if ok else 1
        print(u'  %s  %-24s 期望碰撞 %s、實際 %s  課程筒 %s  最近 %s%%腿長  %s'
              % (k, desc, want, got, n, s['cone_min_dist_pct_leg'],
                 u'✅' if ok else u'❌'))
    print(u'\n%s' % (u'全部通過' if not bad else u'✗ %d 個唔通過' % bad))
    sys.exit(1 if bad else 0)
