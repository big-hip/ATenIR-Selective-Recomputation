# ── 基础镜像：PyTorch 2.6 + CUDA 12.4 ──────────────────────────────────────
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# ── 系统依赖 ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        graphviz \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依赖 ─────────────────────────────────────────────────────────────
WORKDIR /workspace/ATenIR-Selective-Recomputation

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 拷贝项目代码 ─────────────────────────────────────────────────────────────
COPY . .

# ── 默认环境变量 ─────────────────────────────────────────────────────────────
ENV RECOMPUTE='{"6": 0}' \
    RECOMPUTE_LOG_LEVEL=INFO \
    MODEL_NAME=Transformer \
    PROJECT_ROOT=/workspace/ATenIR-Selective-Recomputation \
    PYTHONPATH=/workspace/ATenIR-Selective-Recomputation

# ── 默认入口：运行 Transformer 示例 ──────────────────────────────────────────
CMD ["python", "examples/transformer/main.py"]
