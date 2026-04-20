.PHONY: build run shell clean test help

IMAGE_NAME := atenir-selective-recomputation

help: ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 构建 Docker 镜像
	docker compose build

run: build ## 一键运行 L1/L2 多模型对比
	docker compose up --rm

shell: build ## 进入容器交互式终端
	docker compose run --rm atenir bash

clean: ## 清理 Docker 镜像
	docker compose down --rmi local 2>/dev/null || true

# ── 本地开发 ────────────────────────────────────────────────────────────────

setup-local: ## 本地安装依赖（需先激活 conda/venv）
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
	pip install -r requirements.txt

test: ## 运行全部测试
	PYTHONPATH=$(CURDIR) python -m pytest tests/ -x -q

run-ex1: ## L1/L2 对比（~30s）
	PYTHONPATH=$(CURDIR) python toolkit_examples/ex1_multi_model_capture.py

run-ex2: ## 仿真精度验证（~50s）
	PYTHONPATH=$(CURDIR) python toolkit_examples/ex2_strategy_comparison.py

run-ex3: ## 12 策略全对比（~3min）
	PYTHONPATH=$(CURDIR) python toolkit_examples/ex3_simulation_accuracy.py
