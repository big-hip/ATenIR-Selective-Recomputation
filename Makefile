.PHONY: help setup-local test test-fast run-quick run-demo run-sim run-phase run-gen run-horizontal run-figs

help: ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── 本地开发 ────────────────────────────────────────────────────────────────

setup-local: ## 本地安装依赖（需先激活 conda/venv）
	pip install -r requirements.txt

test: ## 运行全部测试（95 tests）
	PYTHONPATH=$(CURDIR) python -m pytest tests/ -x -q

test-fast: ## 快速测试（跳过 Inductor/Triton 慢测试）
	PYTHONPATH=$(CURDIR) python -m pytest tests/ -x -q -m "not inductor"

# ── 实验脚本 ────────────────────────────────────────────────────────────────

run-quick: ## 快速 Demo: IR 对比 + 仿真验证（GPT-2, <20s）
	PYTHONPATH=$(CURDIR) python toolkit_examples/ex_quick_demo.py

run-demo: ## Demo: L1/L2 三模型捕获（~30s）
	PYTHONPATH=$(CURDIR) python toolkit_examples/ex1_multi_model_capture.py

run-sim: ## 实验 1: 12 策略仿真精度验证（~10min, 需 GPU ≥40GB）
	PYTHONPATH=$(CURDIR) python toolkit_examples/ex_sim_accuracy.py

run-phase: ## 实验 2: batch×optimizer 峰值阶段分析（~15min）
	PYTHONPATH=$(CURDIR) python toolkit_examples/ex_peak_phase.py

run-gen: ## 实验 3: 多模型通用性验证（~10min）
	PYTHONPATH=$(CURDIR) python toolkit_examples/ex_model_generalization.py

run-horizontal: ## 实验 4: 横向仿真方法对比（quick/medium: ARGS=--quick 或 --medium）
	PYTHONPATH=$(CURDIR) python toolkit_examples/ex_horizontal_comparison.py $(ARGS)

run-figs: ## 生成论文图表 F0-F9（需先跑实验 1-4）
	PYTHONPATH=$(CURDIR) python toolkit_examples/generate_paper_figures.py
