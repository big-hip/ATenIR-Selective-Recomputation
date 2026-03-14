#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run.sh — ATenIR Transformer 示例统一入口
#
#  用法:
#    ./run.sh                      # 交互式菜单
#    ./run.sh benchmark            # 策略对比（静态 + 运行时 profiling）
#    ./run.sh train                # 完整训练对比（Eager vs ATenIR）
#    ./run.sh translate            # 交互式翻译
#    ./run.sh all                  # 依次跑 benchmark + train
#
#  环境变量:
#    RECOMPUTE='{"6": 0}'          # 重计算策略配置
#    MODEL_NAME=Transformer        # 模型名（影响输出路径）
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认值
RECOMPUTE="${RECOMPUTE:-'{\"6\": 0}'}"
MODEL_NAME="${MODEL_NAME:-Transformer}"

LINE="────────────────────────────────────────────────────────────"
BOLD="════════════════════════════════════════════════════════════"

run_benchmark() {
    echo ""
    echo "$BOLD"
    echo "  [benchmark] 策略对比: 编译 + 静态估算 + 运行时 profiling"
    echo "$BOLD"
    RECOMPUTE="$RECOMPUTE" MODEL_NAME="$MODEL_NAME" python benchmark.py
}

run_train() {
    echo ""
    echo "$BOLD"
    echo "  [train] 完整训练对比: Eager vs ATenIR (收敛 + BLEU)"
    echo "$BOLD"
    RECOMPUTE="$RECOMPUTE" MODEL_NAME="$MODEL_NAME" python train.py
}

run_translate() {
    echo ""
    echo "$BOLD"
    echo "  [translate] EN→DE 交互式翻译"
    echo "$BOLD"
    if [ ! -f checkpoints/transformer_best.pt ]; then
        echo "  错误: 找不到模型文件 checkpoints/transformer_best.pt"
        echo "  请先运行: ./run.sh train"
        exit 1
    fi
    python translate.py "$@"
}

show_menu() {
    echo ""
    echo "$BOLD"
    echo "  ATenIR Transformer 示例"
    echo "$BOLD"
    echo ""
    echo "  当前配置:"
    echo "    RECOMPUTE  = $RECOMPUTE"
    echo "    MODEL_NAME = $MODEL_NAME"
    echo ""
    echo "$LINE"
    echo "  选择要运行的任务:"
    echo ""
    echo "    1) benchmark   — 策略对比 (静态估算 + 运行时 profiling)"
    echo "    2) train       — 完整训练 (Eager vs ATenIR, 收敛 + BLEU + 保存模型)"
    echo "    3) translate   — 交互式翻译 (需要先跑过 train)"
    echo "    4) all         — 依次跑 benchmark + train"
    echo "    q) 退出"
    echo ""
    echo "$LINE"
    read -rp "  请选择 [1-4/q]: " choice
    case "$choice" in
        1|benchmark)  run_benchmark ;;
        2|train)      run_train ;;
        3|translate)  run_translate ;;
        4|all)        run_benchmark; run_train ;;
        q|Q|quit)     echo "  再见！"; exit 0 ;;
        *)            echo "  无效选择: $choice"; exit 1 ;;
    esac
}

# 主入口：支持命令行参数或交互菜单
case "${1:-}" in
    benchmark)  run_benchmark ;;
    train)      run_train ;;
    translate)  shift; run_translate "$@" ;;
    all)        run_benchmark; run_train ;;
    "")         show_menu ;;
    *)
        echo "未知命令: $1"
        echo "用法: ./run.sh [benchmark|train|translate|all]"
        exit 1
        ;;
esac
