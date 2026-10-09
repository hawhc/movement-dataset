# movement-dataset — 幼童動作 3D 資料集 + 常模

由 4-Cam 課室錄影整出可以計常模、又可以畀 model 學嘅資料集。
計算部分係引用 `human-selfcalib` 嘅 `ComputeSession`（同 admin 後台同一個引擎，
管線零改動），只係輸入由直播相機換成錄影檔。

## 流程

```
cam*.mp4 ──YOLO11m-pose──▶ 2D 檢測 ──表演區框──▶ 只剩表演者
        ──自標定 K,R,T──▶ 對極關聯 + 三角化 ──▶ 每人 17 點 3D 骨架 ──▶ 指標
```

### 1. 框表演區（每個場地一次）

框選工具：https://claude.ai/artifact/PsykC9XWniv673k5EERhsw
畫好之後存做 `rois/<場地>_<日期>.json`。

**呢步唔係可有可無。** 課室入面有廿幾個人（旁觀嘅細路、老師），自標定靠
「認人」反推機位，人多又冇單人幀就一定認錯 —— 實測未過濾時 20 次標定
全部失敗（解出兩台對角相機相距 0.45m）；過濾到剩返表演者，5 秒成功。

### 2. 入庫

```bash
# 一場嘅第一條片：自標定
python ingest.py --rec <rec 資料夾> --roi rois/<場地>.json \
    --movement k1_straight_run --grade K1

# 同場其餘嘅片：重用標定（相機冇郁過）
python ingest.py --rec <rec2> --roi rois/<場地>.json \
    --movement k1_straight_run --grade K1 \
    --calib data/k1_straight_run/<rec1>/calibration.json
```

輸出 `data/<movement_key>/<rec_id>/`：

| 檔案 | 內容 |
|---|---|
| `pose3d.npz` | `xyz` (S,T,17,3) 公尺、`xyz_norm`（髖為原點／肩寬正規化，跨身高可比）、`valid`、`subjects` |
| `metrics.json` | 逐位受測者嘅指標 + 逐個步態循環嘅明細 |
| `meta.json` | 場次、動作、年級、ROI 保留率、尺度修正係數 |
| `calibration.json` | 該場嘅機位解 |

2D 檢測有快取（`data/_work/<rec_id>/dets`），改演算法重跑唔使再等 YOLO。

### 3. 砌常模

```bash
python build_norms.py --movement k1_straight_run --grade K1
python build_norms.py --all
```

**兩段式彙總**：同一位受測者嘅多個循環先取中位數 → 各次表演之間先算百分位。
做一段式會令做得多循環嗰個細路佔重幾倍。輸出 `norms/<動作>_<年級>.json`，
包含 p10/25/50/75/90、mean、sd、robust_sd（IQR/1.349）同 `n_subjects`。
少過 20 位會標 `sufficient: false` —— 唔好攞去評學生。

`build_norms.zscore(value, norm)` 出「百分位 + 穩健 z」，就係攞一位學生
對常模嘅方法。呢個同準則分數係兩回事：準則分數係「達唔達標」，
百分位係「同年紀嘅細路之中排第幾」。

## 兩個要記住嘅陷阱

**公制尺度對幼童偏大約 65%。** 管線嘅尺度係成人骨長先驗定出，實測 K1 細路
讀到肩寬 43cm（實際約 26cm）。`ingest.py` 會逐位受測者乘返
`SHOULDER_CM[grade] ÷ 實測肩寬`，原始值留喺 `raw_shoulder_cm` 同
`scale_correction`。**角度類指標唔受影響**（尺度不變）。

**步態循環要有幅度閘。** 片頭片尾企喺度嗰陣訊號喺中位線附近抖，淨計過線會
切出一堆假循環（實測 16 秒切到 17 個，一半峰值得 16°，真正跑嗰陣係 45°+）。
`_cycles()` 要求峰值升到 `中位 + 0.35 × 幅度` 先算一步。

## 身份

冇學生身份。一條片 = 一次表演，受測者標記係 `<rec_id>#<track_id>`，
純粹用嚟分「邊幾個循環係同一個身體」，做兩段式彙總嗰陣要用。
同一個細路做兩次 = 兩筆，統計上成立。

## 4. 用數據改善準則

```bash
python compare_rubric.py --movement k1_straight_run --grade K1
python compare_rubric.py --movement ... --grade ... --propose-full p75 --write
```

`rubrics/<動作>_<年級>.json` 係現行準則門檻（抄自總表）對應到指標欄位。
`compare_rubric.py` 把佢同常模並排，報四樣嘢：

1. **現行門檻之下實際分佈** —— 幾多 % 落滿分／部分／零分
2. **分辨力** `(p90−p10)/p50` —— 太細即係送分，評唔到嘢
3. **撞天花板／地板** —— 超過 80% 落同一格就標出嚟
4. **兩列重唔重複** —— 受測者之間相關 >0.85 就標出嚟

`--write` 出 `.proposed.json`（建議門檻，**未經人手審批，唔會覆蓋現行準則**）。

⚠ 「良好應該擺喺邊」係教學決定，唔係統計決定。揀 p50 做良好，
即係永遠一半人唔合格 —— 係咪想要，要老師拍板。程式只攤證據，唔拍板。

## 點交資料

每場拍完，我需要三樣嘢：

1. **原始 rec 資料夾**（`cam01_dev0.mp4`…`cam04_dev3.mp4`，唔好經 WhatsApp）
2. **一張清單**：邊個資料夾 = 邊個動作 = 邊個年級。例如

   ```
   rec_20260915_100054   k1_straight_run   K1
   rec_20260915_101230   k1_straight_run   K1
   rec_20260915_103045   k3_sprint_agility_ladder_high_knees   K3
   ```

3. **場地／機位有冇變過** —— 冇變就成場共用一個 ROI 框同一份標定；
   郁過就要重新畫框 + 重新標定。

跟住逐條跑：

```bash
# 該場第一條：自標定（順便產生成場共用嘅 calibration.json）
python ingest.py --rec <rec1> --roi rois/<場地>.json --movement <key> --grade <K?>

# 其餘：重用標定
python ingest.py --rec <recN> --roi rois/<場地>.json --movement <key> --grade <K?> \
    --calib data/<key>/<rec1>/calibration.json

# 全部入完
python build_norms.py --all
python compare_rubric.py --movement <key> --grade <K?>
```

### 兩件影響「點拍」、事後補唔返嘅事

- **重複兩次嘅樣本**：揀約 10 個細路做兩次，兩次用同一個 `subject_ref`
  （唔使真名，`場次#7a` / `場次#7b` 就得）。冇呢批就分唔開
  「細路之間真係有分別」同「量度雜訊」，門檻可能定咗喺雜訊上面。
- **做得差嘅樣本**：全部都係正常表現，就只學到中間段，定唔到
  「進步中／基礎階段」嘅界線。叫幾個細路做「唔抬膝」「原地行嚟行去」
  嘅版本，並且標明係示範差嘅。

---

# 自己跑（唔使搵人）

## 最簡單：雙擊 `RUN.command`

Finder 度雙擊，佢會問你三樣嘢（資料夾／動作／年級），然後自己跑晒。
資料夾直接由 Finder 拖入 Terminal 個窗就得。

## 或者一條命令

```bash
cd /Users/bcm01032/game7/movement-dataset
PY=/Users/bcm01032/game7/venv/bin/python

# 掃一個資料夾入面某一日嘅錄影
$PY batch.py --scan ~/Downloads --date 20260915 \
    --movement k1_straight_run --grade K1 \
    --roi rois/yau_yat_chuen_20260915.json
```

⚠ **`--date` 幾乎一定要寫**。唔寫就會連舊場次都掃埋（Downloads 入面仲有
8 月嗰批成人示範片）。

## 一次過做幾個唔同動作：寫張清單

`jobs.txt`（每行：資料夾　動作key　年級，`#` 開頭係註解）：

```
/Users/bcm01032/Downloads/rec_20260915_095915   k1_straight_run   K1
/Users/bcm01032/Downloads/rec_20260915_095955   k1_straight_run   K1
/Users/bcm01032/Downloads/rec_20260915_103045   k3_sprint_agility_ladder_high_knees   K3
```

```bash
$PY batch.py --list jobs.txt --roi rois/yau_yat_chuen_20260915.json
```

## batch.py 自己會處理嘅嘢

- **標定重用**：同一場次（`rec_` 後面嗰個日期）只自標定一次，其餘全部重用，快好多
- **跳過做過嘅**（要重做加 `--force`）
- **失敗唔會停晒**：記低原因、繼續做下一條，最後一次過報你聽
- **跑完自動**砌常模 + 出準則對照

## 睇結果

```bash
$PY build_norms.py --all                                  # 常模
$PY compare_rubric.py --movement k1_straight_run --grade K1   # 準則對照
```

## 出事嘅時候

| 訊息 | 點解決 |
|---|---|
| `自標定失敗` | 表演區框太鬆，旁觀者入咗框 → 用框選工具收窄再跑 |
| `搵唔到 cam*.mp4` | 資料夾入面唔係 `cam01_dev0.mp4` 咁嘅命名 |
| `⚠ … 標為 suspect` | 老師／碎片軌跡，已自動排除，唔使理 |
| `⚠ 搵唔到舉手` | 嗰位受測者冇偵測到舉手，用咗成條片 —— 睇返條片核實 |
| 常模表空白／`n` 好細 | 樣本未夠，繼續入多幾場 |

## 自己核對 + 自己調門檻

### 睇下守門啱唔啱（`review.py`）

```bash
$PY review.py --movement k1_straight_run                    # 一覽
$PY review.py --movement k1_straight_run --rec rec_20260915_100345 --image
```

`--image` 會把每位受測者嘅 3D 骨架**反投影返四個機位**、逐個標住編號，
存成 `data/<動作>/<rec_id>/review_f<幀>.jpg`。開嚟睇，一眼就認到邊個係
真表演者、邊個係坐喺地下睇。圖入面嘅編號同一覽表對得返。

守門三條規則（任何一條中就唔入常模，原因會列出嚟）：

| 規則 | 捉乜 |
|---|---|
| `(髖高−踝高)÷軀幹長 < 1.0`（或 NaN） | 坐喺地下睇嘅細路 —— 坐埋一堆腳踝重建唔到 |
| 體型偏離同場中位 25% | 混入嘅老師 |
| 幀數少過全片 20% | 碎片軌跡 |

實測：真表演者企姿讀數 1.36–1.54，旁觀者全部 NaN。

### 守門捉唔到，但覆查發現唔可信（`exclude.json`）

三條規則係體型／姿勢／幀數，捉唔到「軌跡本身出錯」嗰種。例如
`rec_20260915_092744#0`：企姿 1.30、體型 39cm、83 幀全齊，三條規則全部過，
但三個循環係 0.35s / 0.35s / 1.15s、`peak_speed` 讀到 5.02 m/s（成人短跑速度）。
呢類個案寫入 `exclude.json`：

```json
{"subjects": {"rec_20260915_092744#0": "點解排除 —— 一定要寫"}}
```

**刻意做成一個檔而唔係去 `data/` 度刪資料**：刪咗就冇得覆查，而且下次有人
重跑 `batch.py` 會靜靜雞入返去。`build_norms.py` 會讀佢，常模嘅
`excluded_manual` 會列返邊幾位被人手排除。

### 改咗門檻想重算（唔使重跑影片）

```bash
$PY batch.py --remeasure --movement k1_straight_run                 # 用現行門檻
$PY batch.py --remeasure --movement k1_straight_run --min-lift 28   # 試新門檻
```

由已有嘅 `pose3d.npz` 直接重算全部已入庫錄影，**幾秒搞掂**（原本要每條 4 分鐘），
跟住自動出新常模同準則對照。試完唔啱就再跑過。

### 「冇做到動作」點處理

`--min-lift`（預設 25°，用膝抬 p90）過唔到就標 `did_perform=false`：

- **節奏類指標唔出數**（步頻／CV／停頓全部 null）—— 唔好畀企定嘅人攞到「冇停頓」滿分
- **但佢照計入常模**（用戶 2026-09-15 決定）—— 佢係班入面一分子，
  剔走會令「典型 K1 表現」被高估
- 評分底分喺 `rubrics/<動作>_<年級>.json` 嘅 `did_not_perform.score`（K1 原地跑 = 30）

---

# 加一個新動作：由零到準則對照

由 K3 持球原地深蹲跳（2026-09-17/18）整套跑完之後歸納出嚟。**照順序做**，
因為前面一步錯，後面所有數都係假嘅而且唔會報錯。

## 0. 先釘實準則（最緊要，最易做錯）

攞 `~/Downloads/K?_上學期下學期_體能動作評估準則_優化版.docx` —— **呢份先係現行**。

⚠ **唔好信 `GAME7_K1-K3_評分準則總表.md` 嘅 `<details>` 細節表。** 實測第 83 條
嘅**標題分數用新版、細節表用舊版**（新版 5 列／深度 55–110°／軀幹 θ_vert<15°；
舊版 6 列／深度 65–110°／軀幹前傾<18°），撈埋一齊。app 嘅 class 註釋亦抄咗舊版。

抄落一張表：指標｜分數｜MediaPipe 關鍵點｜計算方式｜良好標準。然後答三條：

1. **邊條要物件偵測**（球／筒／梯）？→ 要做第 6 步，而且個物件通常係全場唯一
   有已知尺寸嘅嘢，順便解尺度。
2. **docx 個門檻係 2D 投影數**（表頭「拍攝角度：正面」）。3D 同名讀數唔同量級 ——
   θ_vert 同 ΔX_hip 喺正面機位其實係**左右分量**（實測左右 7.5° vs 3D 總傾角 40.6°；
   左右 9.4cm vs 水平總位移 15.2cm）。逐條要判 docx 講嘅係邊個分量。
3. **有冇「零樣本」嘅層級**（例如「掉球 0–12 分」）？→ 第 10 步。

## 1. 收片（事後補唔返）

除咗 `rec_` 資料夾同清單（資料夾＝動作＝年級）之外，一定要有：

- **做得差／失敗嘅樣本**，並且標明。全部正常表現就只學到中間段，定唔到
  「進步中／基礎階段」嘅界線，而百分位門檻會把做得差當成正常。
- **重複兩次嘅樣本**（約 10 個細路，同一個 `subject_ref`），否則分唔開
  「細路之間真有分別」同「量度雜訊」。
- **機位有冇郁過**。同一日上午下午分兩場拍就一定當兩場（實測跨場平移
  121–476px，cam03 成個視角唔同；場內最大爬移只有 12.7px）。

## 2. 框表演區

每個場地 × 每個時段一次。框選工具見上文。⚠ 貼 JSON 過嚟好易中途截斷 ——
`regions_px` 完整就由佢除返 size 還原 `regions`（已中過兩次）。

## 3. 入庫

```bash
$PY batch.py --list jobs.txt --roi rois/<場地>_<時段>.json
```

`session_of()` 會自動按 `日期+am/pm` 分場，上下午各自自標定一次。

## 4. 核守門

```bash
$PY review.py --movement <key>            # 一覽
$PY review.py --movement <key> --rec <rec> --image   # 反投影核圖
```

守門規則係**跨動作共用**，改之前一定要驗其他動作唔受影響。實測例子：
`stance_ratio` 守門用中位數，喺深蹲類動作錯剔 8/29 位真表演者（佢哋成段片
一大半時間蹲住）；改用 p90 就啱，而跑步動作完全唔受影響（嗰邊數值門檻
本來一位都冇捉到，全靠 NaN 捉旁觀者）。

## 5. 加動作家族（唔可以沿用其他動作嘅指標）

`ingest.py` 嘅 `FAMILY` 加一個 key，寫一個 `_<family>()` 分支。**跑步嗰套
（膝抬／步頻／峰值速度）唔可以搬去其他動作**：前置閘、切循環嘅訊號、
起計訊號全部要換成動作本身嘅。實測深蹲跳要換嘅嘢：

| 沿用跑步會出事嘅位 | 換成 |
|---|---|
| 前置閘用膝抬 p90 ≥ 25° | 髖下沉量 ÷ 腿長 |
| 切循環用膝抬訊號 | 髖高訊號 + 時長上限 |
| `--start-trigger hand_raise` 預設 | `NO_TRIGGER`（捧球過頭會假觸發，14 條有 13 條中招）|

**三個必踩嘅量度陷阱**（已喺 `ingest.py` 修好，新家族照用就得）：

1. 垂直基準用 `_up_from_floor()`（腳踝點 RANSAC 擬合地面），**唔好用軀幹方向** ——
   軀幹唔垂直，人一走位水平位移會投影成假高度（實測讀到髖高 166cm、ΔY_hip 37cm）。
2. 企直基線唔可以淨靠膝角 >160° —— 細路坐喺地下對腳伸直一樣過關，基線會塌。
   要加 `(髖−踝)/軀幹 > 1.2`。
3. 「離地」唔可以淨計腳踝高度 —— 坐低時腳踝一樣高過基線。要求髖高過企直基線。

## 6. 要物件偵測就用 `ball3d.py` 嗰套

```bash
$PY ball3d.py detect --movement <key>    # 四路跑物件模型（~15 分鐘 / 14 條）
$PY ball3d.py build  --movement <key>    # 對極配對 + 身體關聯 + 三角化
$PY ball3d.py scale  --movement <key>    # 由已知尺寸物件解世界尺度
```

- **唔靠 conf 分真假**：牆畫／燈籠／卡通公仔喺單一機位同真球一樣圓，但係
  另一台機嘅對極線上搵唔到對應，就算夾硬配到都唔會落喺人身邊。兩關清走。
- **去畸變一定要做**：棋盤格 k1 = −0.37。`calibration.json` 冇 dist，要由
  `human-selfcalib/calibs/tapo_c120_checkerboard.json` 攞。
- **時間對齊**：`fidmap` 冇存落硬碟，用 `det_<v>.json`（session 幀號）同
  `filtered/<cam>.json`（影片幀號）嘅關鍵點指紋反推常數偏移，寫 `framemap.json`。
- ⚠ **關聯閘唔可以當成過濾**：最初用「球離手腕 <0.45m」剔走咗放喺地下嘅球，
  覆蓋率由 69% 跌到 44% —— 但「球喺唔喺胸前」正正係要量嘅指標。

## 7. 重算 → 常模 → 準則對照

```bash
$PY batch.py --remeasure --movement <key>     # 由 pose3d.npz 重算，幾秒
$PY build_norms.py --movement <key> --grade K?
$PY compare_rubric.py --movement <key> --grade K?
```

改指標定義唔使重跑影片。n < 20 會標 `sufficient: false`。

## 8. 寫 `rubrics/<key>_<grade>.json`

- `bands` 一律 `null` —— 唔好由其他動作或者其他年級搬數過嚟。
- **cm 類門檻企唔穩**：公制錨點未定案（肩寬錨點同球錨點爭 1.4 倍）。
  所有評分列用**角度**或**÷腿長／÷軀幹長嘅比率**，cm 只做參考。
- 逐行寫 `origin`（呢個數點嚟）同 `app_cross_check`（同 Dart scorer 嘅分歧）。
- 量唔到嘅寫 `not_measurable`，連「阻塞係乜」一齊寫。

## 9. 同 app 嘅 Dart scorer 對數

`lib/features/pose_trainer/logic/k?_gross_motor_scorers.dart`。核三樣：
列數同分數、量緊邊個量（同名唔代表同義）、門檻落喺常模邊個百分位。

實測四個結果：深度下界 app 已經喺 2026-09-11 拆走（我本來以為未決定）；
持球 0.90/0.60 啱啱好（0.90 = p50，49% 攞滿分）；跳躍 0.20 撞天花板
（90% 攞滿分、0% 跌落最低層，應該定喺 p75 ≈ 0.58）；軀幹 app 量前傾、
離線量左右傾 —— 兩樣唔同嘅嘢。

## 10. 零樣本嘅層級：事件判定 + 合成測試

冇正樣本嘅層（例如「掉球」）**唔可以用百分位**，否則將來真係出現嗰個個案
會被一堆正常樣本拉住、照樣攞中間分。要用物理／幾何寫成絕對事件旗標，
再用合成軌跡驗（見 `tests/test_ball_drop.py`：A 要響／B C 唔可以響）。

寫合成測試唔係形式 —— 實測即刻捉到一個令判定**永遠失效**嘅 bug。
另外真數據上要逐宗核圖，加守門（追蹤連續性、物理速度）直到假陽性歸零；
被守門剔走嘅宗數本身就係追蹤質素嘅訊號，模型改善之後要重驗。
