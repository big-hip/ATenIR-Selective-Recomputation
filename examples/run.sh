#!/bin/bash
# =========================================================
# run.sh — 运行指定 example 目录内的 main.py 并注入重计算策略
#
# 用法: ./run.sh <模型目录> [策略(0-7)] [策略参数]
# 示例:
#   ./run.sh transformer 0         # baseline（不重计算）
#   ./run.sh transformer 6         # 自动廉价 depth=0
#   ./run.sh transformer 6 2       # 自动廉价 depth=2
#   ./run.sh transformer 7 0.5     # min-cut budget=0.5
#   ./run.sh transformer           # 默认策略 6
# =========================================================

set -euo pipefail

# ── 样式 ──────────────────────────────────────────────────
RED='\033[0;31m'  GREEN='\033[0;32m'
BLUE='\033[0;34m' YELLOW='\033[1;33m' NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_succ() { echo -e "${GREEN}[OK]${NC}   $1"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $1"; }

# ── 路径解析 ──────────────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
export PROJECT_ROOT

# ── 参数校验 ──────────────────────────────────────────────
if [[ -z "${1:-}" ]]; then
    echo -e "${YELLOW}用法:${NC} $0 <模型目录> [策略(0-7)] [策略参数]"
    echo
    echo "  策略 0: 不重计算（baseline）"
    echo "  策略 1: 全部重计算"
    echo "  策略 2: 按名称关键字（默认 relu,dropout）"
    echo "  策略 3: 按步幅选层（默认 start=0, stride=2）"
    echo "  策略 4: 按比例选前 N%（默认 50%）"
    echo "  策略 5: 按 Op Type 重计算（默认 relu.default）"
    echo "  策略 6: 自动廉价（默认 depth=0）"
    echo "  策略 7: min-cut 最优重计算（默认 budget=1.0）"
    echo
    echo -e "${YELLOW}可用模型:${NC}"
    for d in "$SCRIPT_DIR"/*/; do
        [[ -f "$d/main.py" ]] && echo "  $(basename "$d")"
    done
    exit 1
fi

MODEL_DIR_NAME="$1"
STRATEGY="${2:-6}"
MODEL_DIR="$SCRIPT_DIR/$MODEL_DIR_NAME"
TARGET="$MODEL_DIR/main.py"

if [[ ! -f "$TARGET" ]]; then
    log_err "找不到 $TARGET"
    echo -e "可用模型:"
    for d in "$SCRIPT_DIR"/*/; do
        [[ -f "$d/main.py" ]] && echo "  $(basename "$d")"
    done
    exit 1
fi

export MODEL_NAME="$MODEL_DIR_NAME"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export RECOMPUTE_LOG_LEVEL="${RECOMPUTE_LOG_LEVEL:-INFO}"

# ── 策略配置 ──────────────────────────────────────────────
case $STRATEGY in
    0) DESC="不重计算（baseline）";        export RECOMPUTE='{"0": null}'                   ;;
    1) DESC="全部重计算";                  export RECOMPUTE='{"1": null}'                   ;;
    2) DESC="按名称关键字";                export RECOMPUTE='{"2": ["relu", "dropout"]}'     ;;
    3) DESC="步幅选层 (0, 2)";            export RECOMPUTE='{"3": [0, 2]}'                 ;;
    4) DESC="按比例选前 50%";             export RECOMPUTE='{"4": 0.5}'                    ;;
    5) DESC="按 Op Type";                 export RECOMPUTE='{"5": ["relu.default"]}'       ;;
    6) DEPTH="${3:-0}"
       DESC="自动廉价（depth≤$DEPTH）";   export RECOMPUTE="{\"6\": $DEPTH}"               ;;
    7) BUDGET="${3:-1.0}"
       DESC="min-cut（budget=$BUDGET）";  export RECOMPUTE="{\"7\": $BUDGET}"              ;;
    *) log_err "未知策略: $STRATEGY (支持 0-7)"; exit 1                                     ;;
esac

# ── RUN_ID & 日志 ─────────────────────────────────────────
RUN_ID=$(date +%Y%m%d_%H%M%S)
export RUN_ID

LOG_DIR="$PROJECT_ROOT/IR_artifacts/logs"
mkdir -p "$LOG_DIR"
SAFE_NAME=$(echo "$MODEL_DIR_NAME" | tr '/' '_')
LOG_FILE="$LOG_DIR/run_${SAFE_NAME}_s${STRATEGY}_${RUN_ID}.log"
export RECOMPUTE_LOG_FILE="aten_${SAFE_NAME}_s${STRATEGY}_${RUN_ID}.log"

# ── Banner ────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo -e "  模型:   ${YELLOW}${MODEL_DIR_NAME}${NC}"
echo -e "  策略:   ${GREEN}${STRATEGY} — ${DESC}${NC}"
echo -e "  RUN_ID: ${RUN_ID}"
echo -e "  日志:   ${LOG_FILE}"
echo "═══════════════════════════════════════════════════════════════"

# ── 加载模型特定环境（如果有）──────────────────────────────
[[ -f "$MODEL_DIR/envs.sh" ]] && { log_info "加载 $MODEL_DIR/envs.sh"; source "$MODEL_DIR/envs.sh"; }

# ── 执行 ──────────────────────────────────────────────────
cd "$MODEL_DIR"

echo "─────────────────── [PYTHON OUTPUT] ───────────────────"
python3 main.py 2>&1 | tee "$LOG_FILE"
PY_EXIT=${PIPESTATUS[0]}
echo "───────────────────────────────────────────────────────"

cd "$PROJECT_ROOT"

if [[ $PY_EXIT -eq 0 ]]; then
    log_succ "完成！日志: $LOG_FILE"
else
    log_err "退出码: $PY_EXIT"
    exit $PY_EXIT
fi
