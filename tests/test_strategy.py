import pytest

from toolkit.models import ModelRegistry
from toolkit.strategy import (
    SAC_POLICIES,
    clear_memory_budget,
    get_partition_fn,
    set_memory_budget,
    unwrap_checkpoint,
    wrap_with_checkpoint,
    wrap_with_sac,
)


def _find_blocks(model, block_class_name: str):
    return [module for module in model.modules() if module.__class__.__name__ == block_class_name]


def _forward_impl(forward):
    return getattr(forward, "__func__", forward)


def test_wrap_with_checkpoint_and_unwrap():
    reg = ModelRegistry()
    model = reg.create_model("gpt2")
    block_class_name = reg.get_block_class_name("gpt2")
    blocks = _find_blocks(model, block_class_name)
    original_forwards = [_forward_impl(block.forward) for block in blocks]

    wrap_with_checkpoint(model, block_class_name)

    wrapped_blocks = _find_blocks(model, block_class_name)
    assert wrapped_blocks
    assert all(hasattr(block, "_original_forward") for block in wrapped_blocks)
    assert any(_forward_impl(block.forward) is not orig for block, orig in zip(wrapped_blocks, original_forwards))

    unwrap_checkpoint(model)
    assert all(not hasattr(block, "_original_forward") for block in wrapped_blocks)
    assert all(_forward_impl(block.forward) is orig for block, orig in zip(wrapped_blocks, original_forwards))


def test_wrap_with_sac():
    reg = ModelRegistry()
    model = reg.create_model("gpt2")
    block_class_name = reg.get_block_class_name("gpt2")

    wrap_with_sac(model, block_class_name, policy_name="save_matmuls")
    blocks = _find_blocks(model, block_class_name)

    assert blocks
    assert all(hasattr(block, "_original_forward") for block in blocks)

    unwrap_checkpoint(model)


def test_sac_policy_names_are_available():
    assert {"save_matmuls", "save_attention", "recompute_all"}.issubset(SAC_POLICIES)


def test_get_partition_fn():
    assert callable(get_partition_fn("default"))
    assert callable(get_partition_fn("min_cut"))
    with pytest.raises(ValueError):
        get_partition_fn("unknown")


def test_memory_budget_round_trip():
    import torch._functorch.config as cfg

    original = cfg.activation_memory_budget
    success = set_memory_budget(0.5)
    assert success is True
    assert cfg.activation_memory_budget == 0.5

    clear_memory_budget()
    assert cfg.activation_memory_budget == original
