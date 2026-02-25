#!/bin/bash
# =========================================================
# 脚本名称: run.sh
# 功能描述: 自动配置环境并运行指定 example 文件夹内的 main.py，并注入重计算策略
# 使用方法: ./run.sh <模型样例目录> [重计算策略(1-5)]
# 示例:     ./run.sh Transformer 1
#          ./run.sh Gpt2/gpt2_local 4
# =========================================================

# 开启严格模式 (遇到错误立即退出，遇到未定义变量报错)
set -e
set -u

# =========================================================
# 1. 样式定义
# =========================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_succ()  { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $1"; }

# =========================================================
# 2. 参数校验与路径解析
# =========================================================
# 检查至少一个参数（模型目录名称）
if [ -z "${1:-}" ]; then
    echo -e "${YELLOW}用法:${NC} $0 <模型样例目录> [重计算策略(1-5)]"
    echo -e "${YELLOW}示例 1:${NC} $0 Transformer 1"
    echo -e "${YELLOW}示例 2:${NC} $0 Gpt2/gpt2_local 4"
    exit 1
fi

MODEL_DIR_NAME="$1"
STRATEGY="${2:-1}" # 如果未指定策略，默认为 1

# 路径解析
# CURRENT_DIR 是 example 目录，项目根目录在其上一级
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$CURRENT_DIR/.." && pwd)
export PROJECT_ROOT
MODEL_DIR="$PROJECT_ROOT/example/$MODEL_DIR_NAME"
export MODEL_NAME="$MODEL_DIR_NAME"

# 为本次运行生成唯一 RUN_ID，方便区分不同运行生成的 IR 与日志
RUN_ID=$(date +%Y%m%d_%H%M%S)
export RUN_ID
TARGET_SCRIPT_NAME="main.py"
TARGET_SCRIPT_PATH="$MODEL_DIR/$TARGET_SCRIPT_NAME"

# =========================================================
# 3. 环境变量与重计算策略配置
# =========================================================
# 防止生成 __pycache__ (解决你之前提出的问题)
export PYTHONDONTWRITEBYTECODE=1
# 确保项目根目录在 PYTHONPATH 中，让 aten_recompute 包可被找到
# 使用 ${PYTHONPATH:-} 避免在未定义时触发 set -u 错误
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export RECOMPUTE_LOG_LEVEL="DEBUG"

case $STRATEGY in
    1)
        STRATEGY_DESC="策略 1: 重计算所有激活值"
        export RECOMPUTE='{"1": null}'
        ;;
    2)
        STRATEGY_DESC="策略 2: 根据算子名称重计算 (例如 relu)"
        export RECOMPUTE='{"2": ["relu", "dropout"]}'
        ;;
    3)
        STRATEGY_DESC="策略 3: 步长重计算 (Start:0, Stride:2)"
        export RECOMPUTE='{"3": [0, 2]}'
        ;;
    4)
        STRATEGY_DESC="策略 4: 按比例重计算 (前 50%)"
        export RECOMPUTE='{"4": 0.5}'
        ;;
    5)
        STRATEGY_DESC="策略 5: 按 Op Type 重计算"
        export RECOMPUTE='{"5": ["relu.default"]}'
        ;;
    *)
        log_warn "未知策略: $STRATEGY. 回退到策略 1."
        STRATEGY_DESC="策略 1 (默认): 重计算所有激活值"
        export RECOMPUTE='{"1": null}'
        ;;
esac

# =========================================================
# 4. 打印任务信息
# =========================================================
echo "========================================================="
echo -e "任务启动: 模型样例 [${YELLOW}$MODEL_DIR_NAME${NC}]"
echo -e "重计算策略: ${GREEN}$STRATEGY_DESC${NC}"
echo -e "项目根目录: $PROJECT_ROOT"
echo -e "执行脚本:   $TARGET_SCRIPT_PATH"
echo "========================================================="

# =========================================================
# 5. 校验环境与文件
# =========================================================
if [ ! -d "$MODEL_DIR" ]; then
    log_err "找不到模型目录: '$MODEL_DIR'"
    exit 1
fi

if [ ! -f "$TARGET_SCRIPT_PATH" ]; then
    log_err "在 '$MODEL_DIR' 下找不到目标脚本 '$TARGET_SCRIPT_NAME'!"
    exit 1
fi

# 如果模型目录下有 envs.sh，则加载它
if [ -f "$MODEL_DIR/envs.sh" ]; then
    log_info "正在加载环境变量 $MODEL_DIR/envs.sh..."
    source "$MODEL_DIR/envs.sh"
fi

# =========================================================
# 6. 运行主程序
# =========================================================
# 切换到模型目录执行，确保相对路径读取文件（如 config.json）正常
cd "$MODEL_DIR" || exit 1

# 日志文件保存在项目根目录的 IR_artifacts/logs 文件夹下
mkdir -p "$PROJECT_ROOT/IR_artifacts/logs"
# 将目录名中的 / 替换为 _ 以防止路径错误
SAFE_MODEL_NAME=$(echo "$MODEL_DIR_NAME" | tr '/' '_')

# 主脚本日志文件：带上 RUN_ID，避免覆盖
LOG_FILE="$PROJECT_ROOT/IR_artifacts/logs/run_${SAFE_MODEL_NAME}_strategy${STRATEGY}_${RUN_ID}.log"

# aten_recompute 包内部使用的日志文件名，也带上 RUN_ID，保证与本次运行对应
export RECOMPUTE_LOG_FILE="aten_recompute_${SAFE_MODEL_NAME}_strategy${STRATEGY}_${RUN_ID}.log"

log_info "开始执行 Python 脚本..."
log_info "详细运行日志将保存至: $LOG_FILE"

echo "------------------- [PYTHON OUTPUT] -------------------"
# 运行 Python 脚本并将输出同时打到屏幕和日志文件
python3 "$TARGET_SCRIPT_NAME" 2>&1 | tee "$LOG_FILE"
# 捕获 Python 脚本的退出码 (忽略 tee 的退出码)
PY_EXIT_CODE=${PIPESTATUS[0]}
echo "-------------------------------------------------------"

# 切回根目录
cd "$PROJECT_ROOT"

if [ $PY_EXIT_CODE -eq 0 ]; then
    log_succ "任务执行成功！(Exit Code: 0)"
else
    log_err "任务执行失败！(Exit Code: $PY_EXIT_CODE)"
    exit $PY_EXIT_CODE
fi