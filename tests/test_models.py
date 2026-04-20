import torch

from toolkit.models import (
    ModelRegistry,
    get_hidden,
    get_intermediate,
    get_num_heads,
    get_num_kv_heads,
    get_num_layers,
    get_vocab_size,
    has_position_embedding,
)
from toolkit.utils import count_unique_params


BATCH = 2
SEQ = 64


def test_registry_lists_expected_models():
    reg = ModelRegistry()
    assert reg.list_models() == ["gpt2", "llama", "mistral"]


def test_create_model_and_cpu_forward():
    reg = ModelRegistry()
    for name in reg.list_models():
        model = reg.create_model(name)
        config = model.config
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ))
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out[0]
        assert logits.shape == (BATCH, SEQ, config.vocab_size)


def test_adapters_match_expected_configs():
    reg = ModelRegistry()

    gpt2 = reg.get_config("gpt2")
    assert get_hidden(gpt2) == 128
    assert get_num_layers(gpt2) == 2
    assert get_num_heads(gpt2) == 2
    assert get_intermediate(gpt2) == 512
    assert get_vocab_size(gpt2) == 50257
    assert has_position_embedding(gpt2) is True
    assert get_num_kv_heads(gpt2) == 2

    llama = reg.get_config("llama")
    assert get_hidden(llama) == 128
    assert get_num_layers(llama) == 2
    assert get_num_heads(llama) == 2
    assert get_intermediate(llama) == 512
    assert get_vocab_size(llama) == 32000
    assert has_position_embedding(llama) is False
    assert get_num_kv_heads(llama) == 2

    mistral = reg.get_config("mistral")
    assert get_hidden(mistral) == 128
    assert get_num_layers(mistral) == 2
    assert get_num_heads(mistral) == 2
    assert get_intermediate(mistral) == 512
    assert get_vocab_size(mistral) == 32000
    assert has_position_embedding(mistral) is False
    assert get_num_kv_heads(mistral) == 1


def test_count_unique_params_matches_deduped_named_parameters_for_gpt2():
    reg = ModelRegistry()
    model = reg.create_model("gpt2")

    unique_bytes = count_unique_params(model)
    named_param_bytes = sum(
        param.numel() * param.element_size()
        for _, param in model.named_parameters()
    )
    total_module_param_bytes = sum(
        param.numel() * param.element_size()
        for module in model.modules()
        for param in module._parameters.values()
        if param is not None
    )

    assert unique_bytes == named_param_bytes
    assert unique_bytes < total_module_param_bytes


def test_default_loss_fn_extracts_scalar_loss():
    reg = ModelRegistry()
    for name in reg.list_models():
        model = reg.create_model(name)
        model.train()
        input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ))
        out = model(input_ids, labels=input_ids)
        loss = reg.default_loss_fn(name)(out)
        assert loss is not None
        assert loss.ndim == 0
