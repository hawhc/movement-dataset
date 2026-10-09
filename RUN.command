#!/bin/bash
# 雙擊執行：入庫 + 出常模。
# 拖單一個 rec_* 資料夾，或者拖裝住一堆 rec_* 嘅上層資料夾，兩樣都得。
cd "$(dirname "$0")"
PY=/Users/bcm01032/game7/venv/bin/python

echo "════════════════════════════════════════════"
echo "  GAME7 動作資料集 — 入庫"
echo "════════════════════════════════════════════"
echo
echo "把資料夾拖入嚟，撳 Enter"
echo "（單一個 rec_* 資料夾，或者裝住一堆 rec_* 嘅上層資料夾，都得）："
read -r FOLDER
FOLDER="${FOLDER%\"}"; FOLDER="${FOLDER#\"}"; FOLDER="${FOLDER/#\~/$HOME}"
FOLDER="$(echo "$FOLDER" | sed 's/\\ / /g; s/[[:space:]]*$//')"

if [ ! -d "$FOLDER" ]; then
  echo "✗ 搵唔到資料夾：$FOLDER"; echo "撳 Enter 關閉。"; read -r; exit 1
fi

# 資料夾本身有冇 cam*.mp4 → 係單一條錄影；否則當上層資料夾掃
SINGLE=0
if ls "$FOLDER"/cam*.mp4 >/dev/null 2>&1; then
  SINGLE=1
  NCAM=$(ls "$FOLDER"/cam*.mp4 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NCAM" != "4" ]; then
    echo "✗ 呢條錄影得 $NCAM 條 cam*.mp4，應該有 4 條："
    ls "$FOLDER"/cam*.mp4 2>/dev/null | sed 's|.*/|     |'
    echo "  下載未齊嘅話補返齊先 —— 少一條機位對應會錯晒，但唔會報錯。"
    echo "撳 Enter 關閉。"; read -r; exit 1
  fi
  echo "   ✓ 單一條錄影：$(basename "$FOLDER")（4 機齊）"
else
  N=$(ls -d "$FOLDER"/rec_* 2>/dev/null | wc -l | tr -d ' ')
  if [ "$N" = "0" ]; then
    echo "✗ 呢個資料夾入面搵唔到 cam*.mp4，亦冇 rec_* 子資料夾。"
    echo "  應該拖：裝住 cam01_dev0.mp4 嗰個資料夾，或者佢嘅上一層。"
    echo "撳 Enter 關閉。"; read -r; exit 1
  fi
  echo "   ✓ 搵到 $N 個 rec_* 資料夾"
fi
echo

echo "動作 key（例如 k1_straight_run）："
read -r MOVE
echo "年級（K1 / K2 / K3）："
read -r GRADE
echo

# ---- 點樣認出「要評分嗰個細路」----
# 課室入面成十一個人，要話畀電腦知計邊個。兩個方法，睇動作性質揀。
echo "呢個動作，細路係企定，定係跑嚟跑去？"
echo "   1) 企定唔郁   —— 原地跑、深蹲、平衡、平板支撐 …（用表演區框）"
echo "   2) 跑嚟跑去   —— 敏捷梯、回來跑、方形跑 …（揀最郁嗰個，唔使框）"
echo "揀 1 定 2（唔打就 1）："
read -r KIND
[ -z "$KIND" ] && KIND=1

FILT=""
if [ "$KIND" = "2" ]; then
  echo
  echo "同一時間有幾多個細路做緊？（唔打就 1 個）："
  read -r NMOVE
  [ -z "$NMOVE" ] && NMOVE=1
  FILT="--moving $NMOVE"
  echo "   ✓ 揀每個機位郁得最多嘅 $NMOVE 個 —— 唔使框"
else
  echo
  echo "表演區框："
  ls rois/*.json 2>/dev/null | sed 's|^|   |'
  echo "輸入檔名（唔打就用第一個）："
  read -r ROI
  [ -z "$ROI" ] && ROI=$(ls rois/*.json | head -1)
  [ -f "$ROI" ] || ROI="rois/$ROI"
  if [ ! -f "$ROI" ]; then
    echo "✗ 搵唔到個框：$ROI"; echo "撳 Enter 關閉。"; read -r; exit 1
  fi
  FILT="--roi $ROI"
  echo "   ✓ 用框 $ROI"
fi

echo
echo "▶ 開始：$MOVE | $GRADE | $FILT"
echo

if [ "$SINGLE" = "1" ]; then
  # 單一條：寫張臨時清單畀 batch.py，咁佢一樣會自動重用同場標定
  TMP=$(mktemp /tmp/g7jobs.XXXXXX)
  echo "$FOLDER	$MOVE	$GRADE" > "$TMP"
  "$PY" batch.py --list "$TMP" $FILT
  rm -f "$TMP"
else
  # 一堆：用資料夾入面最新嗰個日期，避免掃埋舊場次
  DATE=$(ls -d "$FOLDER"/rec_* 2>/dev/null | sed -E 's|.*/rec_([0-9]{8})_.*|\1|' \
         | sort -u | tail -1)
  echo "   只做 $DATE 嗰日嘅錄影（唔想要就用 batch.py --date 自己指定）"
  echo
  "$PY" batch.py --scan "$FOLDER" --date "$DATE" \
      --movement "$MOVE" --grade "$GRADE" $FILT
fi

echo
echo "完成。撳 Enter 關閉。"
read -r
