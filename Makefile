.PHONY: build run shell clean test help

IMAGE_NAME := atenir-selective-recomputation

help: ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Docker ──────────────────────────────────────────────────────────────────

build: ## 构建 Docker 镜像
	docker compose build

run: build ## 一键运行 Demo（L1/L2 多模型对比）
	docker compose up --rm

shell: build ## 进入容器交互式终端
	docker compose run --rm atenir bash

clean: ## 清理 Docker 镜像
	docker compose down --rmi local 2>/dev/null || true

# ── 本地开发 ────────────────────────────────────────────────────────────────

setup-local: ## 本地安装依赖（需先激活 conda/venv）
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
	pip install -r requirements.txt

test: ## 运行全部测试（95 tests）
	PYTHONPATH=$(CURDIR) python -m pytest tests/ -x -q

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
