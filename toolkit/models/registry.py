from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    PretrainedConfig,
)


@dataclass(frozen=True)
class ModelSpec:
    config_cls: type[PretrainedConfig]
    model_cls: type[nn.Module]
    block_class_name: str
    loss_fn: Callable
    small_preset: dict


SMALL_PRESETS = {
    "gpt2": dict(
        config_cls=GPT2Config,
        model_cls=GPT2LMHeadModel,
        block_class_name="GPT2Block",
        config_kwargs=dict(
            vocab_size=50257,
            n_embd=128,
            n_layer=2,
            n_head=2,
            n_positions=512,
            n_inner=512,
        ),
    ),
    "llama": dict(
        config_cls=LlamaConfig,
        model_cls=LlamaForCausalLM,
        block_class_name="LlamaDecoderLayer",
        config_kwargs=dict(
            vocab_size=32000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=512,
        ),
    ),
    "mistral": dict(
        config_cls=MistralConfig,
        model_cls=MistralForCausalLM,
        block_class_name="MistralDecoderLayer",
        config_kwargs=dict(
            vocab_size=32000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=512,
            max_position_embeddings=512,
        ),
    ),
}


class ModelRegistry:
    MODELS = {
        name: ModelSpec(
            config_cls=spec["config_cls"],
            model_cls=spec["model_cls"],
            block_class_name=spec["block_class_name"],
            loss_fn=lambda out: out.loss,
            small_preset=spec["config_kwargs"],
        )
        for name, spec in SMALL_PRESETS.items()
    }

    def create_model(self, name: str, **overrides) -> nn.Module:
        config = self.get_config(name, **overrides)
        spec = self._get_spec(name)
        return spec.model_cls(config)

    def get_config(self, name: str, **overrides) -> PretrainedConfig:
        spec = self._get_spec(name)
        kwargs = {**spec.small_preset, **overrides}
        return spec.config_cls(**kwargs)

    def get_block_class_name(self, name: str) -> str:
        return self._get_spec(name).block_class_name

    def default_loss_fn(self, name: str) -> Callable:
        return self._get_spec(name).loss_fn

    def list_models(self) -> list[str]:
        return list(self.MODELS.keys())

    def _get_spec(self, name: str) -> ModelSpec:
        try:
            return self.MODELS[name]
        except KeyError as exc:
            raise KeyError(f"Unknown model: {name}. Available: {self.list_models()}") from exc
