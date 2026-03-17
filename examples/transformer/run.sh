#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  run.sh — ATenIR Transformer 示例统一入口
#
#  用法:
#    ./run.sh                      # 交互式菜单
#    ./run.sh benchmark            # 策略对比（静态 + 运行时 profiling）
#    ./run.sh train                # 完整训练对比（Eager vs ATenIR）
#    ./run.sh capture              # 仅捕获 IR 计算图
#    ./run.sh custom-train         # 单策略自定义训练（无 eager 对比）
#    ./run.sh translate            # 交互式翻译
#    ./run.sh all                  # 依次跑 benchmark + train
#    ./run.sh --strategy           # 先选策略，再选任务
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

# ── 策略选择 ─────────────────────────────────────────────────────────────────
select_strategy() {
    echo ""
    echo "$BOLD"
    echo "  选择重计算策略"
    echo "$BOLD"
    echo ""
    echo "    0) 不重计算                    （保留全部激活）"
    echo "    1) 全部重计算                  （丢弃全部激活）"
    echo "    2) 按节点名称关键字            （参数: 逗号分隔关键字）"
    echo "    3) 按层步长选择                （参数: start,stride）"
    echo "    4) 按比例选择前 N% 层          （参数: 0.0~1.0）"
    echo "    5) 按 ATen 算子类型            （参数: 逗号分隔算子名）"
    echo "    6) 自动廉价重计算（链深度）    （参数: 最大深度, 默认 0）"
    echo "    7) min-cut 最优重计算          （参数: memory_budget, 默认 0.5）"
    echo ""
    echo "$LINE"
    read -rp "  请选择策略 [0-7]: " strat

    case "$strat" in
        0)
            RECOMPUTE='{"0": null}'
            ;;
        1)
            RECOMPUTE='{"1": null}'
            ;;
        2)
            read -rp "  请输入关键字（逗号分隔，如 softmax,dropout）: " kw
            kw="${kw:-softmax}"
            # 转为 JSON 数组: "softmax,dropout" -> ["softmax","dropout"]
            json_arr=$(echo "$kw" | awk -F',' '{
                printf "["
                for(i=1;i<=NF;i++){
                    gsub(/^ +| +$/,"",$i)
                    printf "\"%s\"", $i
                    if(i<NF) printf ","
                }
                printf "]"
            }')
            RECOMPUTE="{\"2\": $json_arr}"
            ;;
        3)
            read -rp "  请输入 start,stride（如 0,2）: " ss
            start="${ss%%,*}"
            stride="${ss##*,}"
            start="${start:-0}"
            stride="${stride:-2}"
            RECOMPUTE="{\"3\": [$start, $stride]}"
            ;;
        4)
            read -rp "  请输入比例（0.0~1.0, 默认 0.5）: " ratio
            ratio="${ratio:-0.5}"
            RECOMPUTE="{\"4\": $ratio}"
            ;;
        5)
            read -rp "  请输入算子类型（逗号分隔，如 mm,bmm,addmm）: " ops
            ops="${ops:-mm}"
            json_arr=$(echo "$ops" | awk -F',' '{
                printf "["
                for(i=1;i<=NF;i++){
                    gsub(/^ +| +$/,"",$i)
                    printf "\"%s\"", $i
                    if(i<NF) printf ","
                }
                printf "]"
            }')
            RECOMPUTE="{\"5\": $json_arr}"
            ;;
        6)
            read -rp "  请输入最大链深度（默认 0）: " depth
            depth="${depth:-0}"
            RECOMPUTE="{\"6\": $depth}"
            ;;
        7)
            read -rp "  请输入 memory_budget（0.0~1.0, 默认 0.5）: " budget
            budget="${budget:-0.5}"
            RECOMPUTE="{\"7\": $budget}"
            ;;
        *)
            echo "  无效策略: $strat，使用默认策略 6"
            RECOMPUTE='{"6": 0}'
            ;;
    esac
    echo ""
    echo "  已选择: RECOMPUTE=$RECOMPUTE"
    echo "$LINE"
}

run_benchmark() {
    echo ""
    echo "$BOLD"
    echo "  [benchmark] 策略对比: 编译 + 静态估算 + 运行时 profiling"
    echo "$BOLD"
    RECOMPUTE="$RECOMPUTE" MODEL_NAME="$MODEL_NAME" python benchmark.py
}

run_capture() {
    echo ""
    echo "$BOLD"
    echo "  [capture] 捕获 IR 计算图 (编译 + 保存前向/后向 FX 图)"
    echo "$BOLD"
    RECOMPUTE="$RECOMPUTE" MODEL_NAME="$MODEL_NAME" python capture.py
}

run_custom_train() {
    echo ""
    echo "$BOLD"
    echo "  [custom-train] 单策略自定义训练 (RECOMPUTE=$RECOMPUTE)"
    echo "$BOLD"
    RECOMPUTE="$RECOMPUTE" MODEL_NAME="$MODEL_NAME" python custom_train.py "${@}"
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
    echo "    1) benchmark    — 策略对比 (静态估算 + 运行时 profiling)"
    echo "    2) train        — 完整训练 (Eager vs ATenIR, 收敛 + BLEU + 保存模型)"
    echo "    3) capture      — 仅捕获 IR 计算图 (快速查看 FX 图结构)"
    echo "    4) custom-train — 单策略自定义训练 (无 eager 对比, 直接训练)"
    echo "    5) translate    — 交互式翻译 (需要先跑过 train)"
    echo "    6) all          — 依次跑 benchmark + train"
    echo "    s) strategy     — 选择重计算策略（当前: $RECOMPUTE）"
    echo "    q) 退出"
    echo ""
    echo "$LINE"
    read -rp "  请选择 [1-6/s/q]: " choice
    case "$choice" in
        1|benchmark)     run_benchmark ;;
        2|train)         run_train ;;
        3|capture)       run_capture ;;
        4|custom-train)  run_custom_train ;;
        5|translate)     run_translate ;;
        6|all)           run_benchmark; run_train ;;
        s|S|strategy)    select_strategy; show_menu ;;
        q|Q|quit)        echo "  再见！"; exit 0 ;;
        *)               echo "  无效选择: $choice"; exit 1 ;;
    esac
}

# 主入口：支持命令行参数或交互菜单
case "${1:-}" in
    benchmark)     run_benchmark ;;
    train)         run_train ;;
    capture)       run_capture ;;
    custom-train)  shift; run_custom_train "$@" ;;
    translate)     shift; run_translate "$@" ;;
    all)           run_benchmark; run_train ;;
    --strategy|-s) select_strategy; show_menu ;;
    "")            show_menu ;;
    *)
        echo "未知命令: $1"
        echo "用法: ./run.sh [benchmark|train|capture|custom-train|translate|all|--strategy]"
        exit 1
        ;;
esac
