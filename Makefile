.PHONY: build run run-strategy shell clean help

IMAGE_NAME  := atenir-selective-recomputation
STRATEGY    ?= {"6": 0}

help: ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build: ## 构建 Docker 镜像
	docker compose build

run: build ## 一键运行 Transformer 示例（默认策略 6）
	docker compose up --rm

run-strategy: build ## 指定策略运行，用法: make run-strategy STRATEGY='{"1": null}'
	docker compose run --rm -e RECOMPUTE='$(STRATEGY)' atenir

shell: build ## 进入容器交互式终端
	docker compose run --rm atenir bash

clean: ## 清理 Docker 镜像和 IR 产物
	docker compose down --rmi local 2>/dev/null || true
	rm -rf IR_artifacts/

# ── 非 Docker 环境 ──────────────────────────────────────────────────────────

setup-local: ## 本地安装依赖（需先激活 conda/venv）
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
	pip install -r requirements.txt

run-local: ## 本地运行 Transformer 示例
	cd examples/transformer && \
	RECOMPUTE='$(STRATEGY)' RECOMPUTE_LOG_LEVEL=INFO \
	PROJECT_ROOT=$(CURDIR) PYTHONPATH=$(CURDIR) \
	python main.py
