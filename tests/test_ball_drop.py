# -*- coding: utf-8 -*-
"""掉球判定嘅合成測試。

點解要有呢個檔：本批 58 位受測者**冇一個真係跌波**，準則「掉球 0–12 分」嗰層
零正樣本。零正樣本之下，判定啱唔啱冇得靠數據驗 —— 唯一方法係造一個物理上
真確嘅跌波軌跡，睇個判定響唔響；同時造一個「正常放低」，睇佢收唔收得住聲。

三個情境（全部用軀幹長正規化，fps 20）：
  A 跌波   —— 球喺胸口高度脫手，之後自由落體（g=9.8），手留喺原位
  B 放低   —— 球同手一齊慢慢落到地面先離手（做得唔標準，但唔係跌）
  C 全程揸住 —— 球一直喺手
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest import _ball_drops  # noqa: E402

FPS, TORSO, G = 20.0, 0.44, 9.8
UP = np.array([0.0, 0.0, 1.0])


def h(p):
    return (p * UP).sum(-1)


def _case(kind, n=60):
    """回傳 (ball, wrist, hip)。地面 z=0，髖高 0.5m，胸口 0.85m。"""
    ball = np.zeros((n, 3))
    wrist = np.zeros((n, 3))
    hip = np.zeros((n, 3))
    hip[:, 2] = 0.50
    t0 = 20                                  # 脫手／開始落嘅幀
    for i in range(n):
        if kind == 'C' or i < t0:
            wrist[i] = [0, 0, 0.85]
            ball[i] = [0, 0.10, 0.85]        # 離手腕 0.10m ≈ 0.23 軀幹長
            continue
        dt = (i - t0) / FPS
        if kind == 'A':                      # 自由落體，手唔郁
            wrist[i] = [0, 0, 0.85]
            z = max(0.09, 0.85 - 0.5 * G * dt * dt)
            ball[i] = [0, 0.10 + 0.6 * dt, z]        # 落埋碌走
        elif kind == 'B':                    # 球同手一齊落（放低）
            z = max(0.09, 0.85 - 0.75 * dt)          # 0.75 m/s，手控住
            wrist[i] = [0, 0, z + 0.02]
            ball[i] = [0, 0.10, z]
            if dt > 1.0:                     # 放低咗之後起返身，手離開個球
                wrist[i] = [0, 0, 0.50 + 0.35 * min(dt - 1.0, 1.0)]
    return ball, wrist, hip


def run(kind):
    ball, wrist, hip = _case(kind)
    idx = np.arange(len(ball))
    return _ball_drops(ball, wrist, hip, h, TORSO, FPS, 0, idx)


if __name__ == '__main__':
    exp = {'A': (1, '跌波：球喺胸口脫手後自由落體'),
           'B': (0, '放低：球同手一齊落到地面'),
           'C': (0, '全程揸住')}
    bad = 0
    for k in 'ABC':
        r = run(k)
        want, desc = exp[k]
        got = r['ball_drop_events']
        ok = (got == want)
        bad += 0 if ok else 1
        print(u'  %s  %-28s 期望 %d、實際 %s  離手最遠 %.2f 軀幹長  %s'
              % (k, desc, want, got, r['ball_max_away_torso'] or 0,
                 u'✅' if ok else u'❌'))
    print(u'\n%s' % (u'全部通過' if not bad else u'✗ %d 個唔通過' % bad))
    sys.exit(1 if bad else 0)
