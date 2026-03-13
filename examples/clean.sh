#!/bin/bash
# =========================================================
# 脚本名称: clean.sh
# 功能描述: 清理 IR_artifacts 下的旧实验数据与日志
# 使用方法:
#   ./clean.sh              — 清理所有模型的所有 runs 和日志
#   ./clean.sh Transformer  — 只清理指定模型的 runs
#   ./clean.sh --keep 5     — 每个模型只保留最近 5 次 run，清理其余
#   ./clean.sh --dry-run    — 预览将被删除的目录，不实际删除
# =========================================================

set -e
set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$CURRENT_DIR/.." && pwd)
ARTIFACTS_DIR="$PROJECT_ROOT/IR_artifacts"
LOGS_DIR="$ARTIFACTS_DIR/logs"

DRY_RUN=0
KEEP=0
MODEL_FILTER=""

# ── 参数解析 ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)  DRY_RUN=1; shift ;;
        --keep)     KEEP="$2"; shift 2 ;;
        -*)
            echo -e "${RED}[ERROR]${NC} 未知选项: $1"
            echo "用法: $0 [模型名] [--keep N] [--dry-run]"
            exit 1
            ;;
        *)  MODEL_FILTER="$1"; shift ;;
    esac
done

[[ $DRY_RUN -eq 1 ]] && echo -e "${YELLOW}[DRY-RUN] 仅预览，不实际删除${NC}"

# ── 辅助函数 ──────────────────────────────────────────────
do_remove() {
    local target="$1"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo -e "  ${YELLOW}[DRY]${NC} 将删除: $target"
    else
        rm -rf "$target"
        echo -e "  ${GREEN}[DEL]${NC} 已删除: $target"
    fi
}

# ── 清理 runs 目录 ────────────────────────────────────────
if [[ ! -d "$ARTIFACTS_DIR" ]]; then
    echo -e "${YELLOW}[WARN]${NC} IR_artifacts 目录不存在，跳过。"
else
    # 遍历每个模型目录
    for model_dir in "$ARTIFACTS_DIR"/*/; do
        model_name=$(basename "$model_dir")
        [[ "$model_name" == "logs" ]] && continue  # 跳过 logs 目录

        if [[ -n "$MODEL_FILTER" && "$model_name" != "$MODEL_FILTER" ]]; then
            continue
        fi

        runs_dir="$model_dir/runs"
        if [[ ! -d "$runs_dir" ]]; then
            continue
        fi

        # 按修改时间从新到旧排序所有 run 目录
        mapfile -t all_runs < <(ls -1t "$runs_dir" 2>/dev/null)
        total=${#all_runs[@]}

        if [[ $KEEP -gt 0 && $total -le $KEEP ]]; then
            echo "[$model_name] 共 $total 次 run，保留全部（≤ keep=$KEEP）。"
            continue
        fi

        echo "[$model_name] 共 $total 次 run，keep=$KEEP，将删除 $((total - KEEP)) 个旧 run："
        for ((i=KEEP; i<total; i++)); do
            do_remove "$runs_dir/${all_runs[$i]}"
        done
    done
fi

# ── 清理日志文件 ──────────────────────────────────────────
if [[ -d "$LOGS_DIR" ]]; then
    if [[ -n "$MODEL_FILTER" ]]; then
        safe_filter=$(echo "$MODEL_FILTER" | tr '/' '_')
        log_pattern="run_${safe_filter}_*.log"
    else
        log_pattern="*.log"
    fi

    mapfile -t logs < <(find "$LOGS_DIR" -maxdepth 1 -name "$log_pattern" 2>/dev/null | sort -t_ -k4 -r)
    log_total=${#logs[@]}

    if [[ $log_total -eq 0 ]]; then
        echo "[logs] 没有匹配的日志文件。"
    elif [[ $KEEP -gt 0 && $log_total -le $KEEP ]]; then
        echo "[logs] 共 $log_total 个日志，保留全部（≤ keep=$KEEP）。"
    else
        echo "[logs] 共 $log_total 个日志，keep=$KEEP，将删除 $((log_total - KEEP)) 个旧日志："
        for ((i=KEEP; i<log_total; i++)); do
            do_remove "${logs[$i]}"
        done
    fi
fi

echo -e "${GREEN}[DONE]${NC} 清理完成。"
