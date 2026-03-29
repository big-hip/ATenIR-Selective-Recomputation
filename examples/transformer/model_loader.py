"""模型构建辅助：从 model config dict 严格构建 Transformer 并移动到指定设备。"""
from typing import Dict, Optional
import torch

from model import Transformer, device as _default_device


def _require_key(m: Dict, key: str):
    if key not in m:
        raise ValueError(f"model config missing required key: {key}")
    return m[key]


def _int(m: Dict, key: str) -> int:
    return int(_require_key(m, key))


def _float(m: Dict, key: str) -> float:
    return float(_require_key(m, key))


def build_transformer_from_map(
    model_map: Optional[Dict] = None,
    seq_len: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> tuple[Transformer, torch.device]:
    """构建 Transformer 实例并将其移动到 `device`。

    Args:
      model_map: 解析后的 model 配置映射（通常来自 model_config.yaml 或主 config 的 model 段）。
      seq_len: 可选的序列长度覆盖（用于设置 positional encoding 的长度）。
      device: 目标设备；若 None 则使用 model.py 中的默认 `device`。

    Returns:
      (transformer, device)
    """
    mm = model_map or {}
    src_vocab_size = _int(mm, "src_vocab_size")
    tgt_vocab_size = _int(mm, "tgt_vocab_size")
    d_model = _int(mm, "d_model")
    num_heads = _int(mm, "num_heads")
    num_layers = _int(mm, "num_layers")
    d_ff = _int(mm, "d_ff")
    max_seq_length = int(seq_len) if seq_len is not None else _int(mm, "max_seq_length")
    dropout = _float(mm, "dropout")
    padding_idx = _int(mm, "padding_idx")

    dev = device or _default_device

    transformer = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
        padding_idx,
    )
    transformer.to(dev)
    return transformer, dev
