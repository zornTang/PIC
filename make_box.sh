#!/bin/bash
set -e

# 修改为你的文件名（你截图是 plbd1_min.pdb_predictions.csv）
P2R_DIR="$(dirname "$0")/P2R_PLBD1"
P2R_PRED="$P2R_DIR/plbd1_min.pdb_predictions.csv"

FPOCKET_DIR="$(dirname "$0")/../01_structure/plbd1_min_out/pockets"  # fpocket 的 pockets 目录
BOX_OUT="$(dirname "$0")/box.txt"

echo "[INFO] Try p2rank -> $P2R_PRED"
if [ -f "$P2R_PRED" ]; then
  # 读取表头，找出 center_x/center_y/center_z 与 rank 的列号
  header=$(head -n1 "$P2R_PRED")
  IFS=',' read -ra cols <<< "$header"
  cx_i=-1; cy_i=-1; cz_i=-1; rk_i=-1; sc_i=-1
  for i in "${!cols[@]}"; do
    col=$(echo "${cols[$i]}" | tr -d ' \r')
    case "$col" in
      center_x|x) cx_i=$((i+1));;
      center_y|y) cy_i=$((i+1));;
      center_z|z) cz_i=$((i+1));;
      rank)       rk_i=$((i+1));;
      pocket_score|score) sc_i=$((i+1));;
    esac
  done

  if [ $cx_i -gt 0 ] && [ $cy_i -gt 0 ] && [ $cz_i -gt 0 ]; then
    # 以 rank 升序优先，若没有 rank 列，则按 score 降序
    if [ $rk_i -gt 0 ]; then
      line=$(tail -n +2 "$P2R_PRED" | sort -t, -k${rk_i},${rk_i}n | head -1)
      cx=$(echo "$line" | awk -F, -v i=$cx_i '{print $i}')
      cy=$(echo "$line" | awk -F, -v i=$cy_i '{print $i}')
      cz=$(echo "$line" | awk -F, -v i=$cz_i '{print $i}')
      rk=$(echo "$line" | awk -F, -v i=$rk_i '{print $i}')
      src="p2rank ($P2R_PRED, rank=${rk})"
    else
      line=$(tail -n +2 "$P2R_PRED" | awk -F, -v s=$sc_i 'NF>0{print $0}' | sort -t, -k${sc_i},${sc_i}nr | head -1)
      cx=$(echo "$line" | awk -F, -v i=$cx_i '{print $i}')
      cy=$(echo "$line" | awk -F, -v i=$cy_i '{print $i}')
      cz=$(echo "$line" | awk -F, -v i=$cz_i '{print $i}')
      sc=$(echo "$line" | awk -F, -v i=$sc_i '{print $i}')
      src="p2rank ($P2R_PRED, top_score=${sc})"
    fi

    # 缺省盒子大小给 24Å 立方，短肽/AMC 一般 22–28 之间可微调
    sx=24; sy=24; sz=24

    {
      echo "center:  $cx $cy $cz"
      echo "size:    $sx $sy $sz"
      echo "source:  $src"
    } > "$BOX_OUT"

    echo "[OK] Wrote $BOX_OUT from p2rank"
    exit 0
  fi
fi

echo "[WARN] p2rank 未可用/未识别，回退到 fpocket..."

# --------- 回退方案：fpocket pocket*_info.txt 中抓取得分最高的口袋 ----------
# fpocket 的 info 文件含 "Score"（或 "Druggability Score"）和 "Pocket center :  x y z"
# 选分数最高的那个
if [ -d "$FPOCKET_DIR" ]; then
  best_file=""
  best_score="-1e9"

  for info in "$FPOCKET_DIR"/pocket*_info.txt; do
    [ -f "$info" ] || continue
    # 兼容不同键名：拿到最后一个含 Score 的数
    sc=$(grep -E "Score|Druggability" "$info" | grep -Eo "[-0-9.]+$" | tail -n1)
    if [ -z "$sc" ]; then sc="-1e9"; fi
    awk "BEGIN{exit !($sc > $best_score)}" || true
    if awk "BEGIN{exit !($sc > $best_score)}"; then
      best_score="$sc"
      best_file="$info"
    fi
  done

  if [ -n "$best_file" ]; then
    center_line=$(grep -E "Pocket center" "$best_file")
    # 中心行通常形如: "Pocket center :   12.3  45.6  -8.9"
    read _ _ _ cx cy cz <<< "$(echo "$center_line" | awk -F: '{print $2}')"
    sx=24; sy=24; sz=24

    {
      echo "center:  $cx $cy $cz"
      echo "size:    $sx $sy $sz"
      echo "source:  fpocket ($best_file, score=${best_score})"
    } > "$BOX_OUT"

    echo "[OK] Wrote $BOX_OUT from fpocket"
    exit 0
  fi
fi

echo "[ERR] 既没有有效的 p2rank，也没有有效的 fpocket 结果，无法生成 box.txt"
exit 1
