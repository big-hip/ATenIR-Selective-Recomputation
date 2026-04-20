import gc

import pytest
import torch
import torch.fx as fx
import torch.nn as nn

from toolkit.capture import analyze_graph, capture_graphs, count_fw_output_bytes, count_fw_outputs, graph_stats
from toolkit.models import ModelRegistry


CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
BATCH = 2
SEQ = 64


class CEModelWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, labels: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.labels = labels

    def forward(self, input_ids):
        return self.base_model(input_ids, labels=self.labels)


@pytest.fixture
def registry():
    return ModelRegistry()


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_capture_graphs_requires_training_mode():
    model = nn.Linear(4, 4).eval()
    sample = torch.randn(2, 4)

    with pytest.raises(ValueError):
        capture_graphs(model, sample, lambda out: out.sum())


def test_capture_graphs_passes_dynamic_true(monkeypatch):
    import toolkit.capture.aot_capture as aot_capture_module

    recorded = {}

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    def fake_aot_module_simplified(gm, example_inputs, fw_compiler, bw_compiler, partition_fn):
        fw_compiler(gm, example_inputs)
        bw_compiler(gm, example_inputs)
        return gm

    def fake_compile(model, backend, dynamic):
        recorded["dynamic"] = dynamic

        def compiled(x):
            gm = fx.symbolic_trace(model)
            backend(gm, [x])
            return model(x)

        return compiled

    monkeypatch.setattr(aot_capture_module, "aot_module_simplified", fake_aot_module_simplified)
    monkeypatch.setattr(aot_capture_module.torch, "compile", fake_compile)
    monkeypatch.setattr(aot_capture_module.torch._dynamo, "reset", lambda: recorded.setdefault("reset", True))

    model = TinyModel().train()
    sample = torch.randn(2, 4)
    fw_gm, bw_gm = capture_graphs(model, sample, lambda out: out.sum())

    assert recorded["dynamic"] is True
    assert recorded["reset"] is True
    assert fw_gm is not None
    assert bw_gm is not None
    assert model.training is True


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires GPU")
@pytest.mark.parametrize("name", ["gpt2", "llama", "mistral"])
def test_capture_all_models(registry, name):
    model = registry.create_model(name).to(DEVICE).train()
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ), device=DEVICE)

    fw_gm, bw_gm = capture_graphs(model, input_ids, lambda out: out.logits.sum())

    fw_stats = graph_stats(fw_gm)
    bw_stats = graph_stats(bw_gm)

    assert fw_stats["n_total"] > 0
    assert bw_stats["n_total"] > 0
    assert fw_stats["n_alloc"] > 0
    assert bw_stats["n_placeholder"] > 0
    assert count_fw_outputs(fw_gm) > 0
    assert count_fw_output_bytes(fw_gm) > 0
    assert analyze_graph(fw_gm)["num_nodes"] == fw_stats["n_total"]
    assert model.training is True


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires GPU")
def test_wrapped_ce_loss_and_sum_loss_produce_distinct_graphs(registry):
    input_ids = torch.randint(0, registry.get_config("gpt2").vocab_size, (BATCH, SEQ), device=DEVICE)

    sum_model = registry.create_model("gpt2").to(DEVICE).train()
    sum_fw, sum_bw = capture_graphs(sum_model, input_ids, lambda out: out.logits.sum())

    ce_base_model = registry.create_model("gpt2").to(DEVICE).train()
    wrapped_ce_model = CEModelWrapper(ce_base_model, input_ids).train()
    ce_fw, ce_bw = capture_graphs(wrapped_ce_model, input_ids, registry.default_loss_fn("gpt2"))

    sum_bw_stats = graph_stats(sum_bw)
    ce_bw_stats = graph_stats(ce_bw)

    assert count_fw_outputs(sum_fw) > 0
    assert count_fw_outputs(ce_fw) > 0
    assert (
        sum_bw_stats["n_total"] != ce_bw_stats["n_total"]
        or sum_bw_stats["n_placeholder"] != ce_bw_stats["n_placeholder"]
        or count_fw_output_bytes(sum_fw) != count_fw_output_bytes(ce_fw)
    )
