#!/usr/bin/env bash
# run.sh — 快速选择策略运行 SimpleMLP IR 捕获
#
# 用法:
#   ./run.sh              # 交互式选择策略
#   ./run.sh 0            # 直接指定策略编号
#   ./run.sh 6 0          # 策略 6，参数 0（max_depth=0）
#   ./run.sh 7 0.5        # 策略 7，memory_budget=0.5

set -euo pipefail
cd "$(dirname "$0")"

show_help() {
    cat <<'EOF'
┌─────────────────────────────────────────────────────┐
│         SimpleMLP IR Capture — 策略选择              │
├──────┬──────────────────────────────────────────────┤
│  0   │ 不重计算（保留全部激活）                       │
│  1   │ 全部重计算                                    │
│  2   │ 按名称关键字（参数: 逗号分隔关键字）           │
│  3   │ 按步幅选层（参数: start,stride）              │
│  4   │ 按比例选前 N% 层（参数: 0.0~1.0）             │
│  5   │ 按算子类型（参数: 逗号分隔算子名）             │
│  6   │ 自动廉价重计算（参数: max_depth，默认 0）      │
│  7   │ min-cut 最优重计算（参数: memory_budget）      │
│ sac  │ PyTorch 内置 SAC（min_cut + CheckpointPolicy）│
└──────┴──────────────────────────────────────────────┘
EOF
}

# ── 解析参数 ──────────────────────────────────────────
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    show_help
    exit 0
fi

STRATEGY="${1:-}"
PARAM="${2:-}"

# 交互式选择
if [[ -z "$STRATEGY" ]]; then
    show_help
    echo ""
    read -rp "请选择策略编号 [0-7 / sac]（默认 6）: " STRATEGY
    STRATEGY="${STRATEGY:-6}"

    # 根据策略提示参数输入
    case "$STRATEGY" in
        0|1|sac) ;;
        2) read -rp "关键字（逗号分隔，如 relu,mm）: " PARAM ;;
        3) read -rp "start,stride（如 0,2）: " PARAM ;;
        4) read -rp "比例 0.0~1.0（如 0.5）: " PARAM ;;
        5) read -rp "算子类型（逗号分隔，如 mm,addmm）: " PARAM ;;
        6) read -rp "max_depth（默认 0）: " PARAM ;;
        7) read -rp "memory_budget（如 0.5）: " PARAM ;;
        *) echo "错误: 未知策略 $STRATEGY"; exit 1 ;;
    esac
fi

# ── 构造 RECOMPUTE JSON ──────────────────────────────
if [[ "$STRATEGY" == "sac" ]]; then
    RECOMPUTE_JSON='{"sac": null}'
elif [[ -z "$PARAM" ]]; then
    RECOMPUTE_JSON="{\"${STRATEGY}\": null}"
else
    # 判断参数类型：纯数字/小数 → 数值，否则 → 字符串
    if [[ "$PARAM" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        RECOMPUTE_JSON="{\"${STRATEGY}\": ${PARAM}}"
    elif [[ "$PARAM" =~ ^[0-9]+,[0-9]+$ ]]; then
        # start,stride → [start, stride]
        IFS=',' read -r start stride <<< "$PARAM"
        RECOMPUTE_JSON="{\"${STRATEGY}\": [${start}, ${stride}]}"
    else
        RECOMPUTE_JSON="{\"${STRATEGY}\": \"${PARAM}\"}"
    fi
fi

echo ""
echo "━━━ RECOMPUTE=$RECOMPUTE_JSON ━━━"
echo ""

export RECOMPUTE="$RECOMPUTE_JSON"
python capture.py
