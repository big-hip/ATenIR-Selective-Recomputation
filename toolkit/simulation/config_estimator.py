import torch
from transformers import GPT2Config

from toolkit.models import (
    get_hidden,
    get_intermediate,
    get_num_heads,
    get_num_kv_heads,
    get_num_layers,
    get_vocab_size,
    has_position_embedding,
)


def estimate_from_config(config, batch, seq, dtype=torch.float32, optimizer="adam", fused_optimizer=False) -> dict:
    elem = torch.finfo(dtype).bits // 8

    # Normalize optimizer: accept both string and class
    if isinstance(optimizer, type):
        _cls_map = {torch.optim.Adam: "adam", torch.optim.AdamW: "adamw", torch.optim.SGD: "sgd"}
        optimizer = _cls_map.get(optimizer, "adam")

    hidden = get_hidden(config)
    n_layers = get_num_layers(config)
    n_heads = get_num_heads(config)
    inter = get_intermediate(config)
    vocab = get_vocab_size(config)
    n_kv_heads = get_num_kv_heads(config)
    is_gpt2 = isinstance(config, GPT2Config)
    head_dim = hidden // n_heads if n_heads > 0 else hidden

    # --- Parameter estimation ---
    embed_params = vocab * hidden
    if has_position_embedding(config) and hasattr(config, "n_positions"):
        embed_params += config.n_positions * hidden
    # lm_head: separate matrix when embeddings are not tied
    tied = getattr(config, "tie_word_embeddings", True)
    lm_head_params = 0 if tied else vocab * hidden

    kv_proj_scale = n_kv_heads / n_heads if n_heads > 0 else 1
    attn_params = int((2 + 2 * kv_proj_scale) * hidden * hidden)
    # GPT-2: fc(H→I) + proj(I→H) = 2*H*I
    # LLaMA/Mistral: gate(H→I) + up(H→I) + down(I→H) = 3*H*I
    has_gate_proj = not is_gpt2
    mlp_params = 3 * hidden * inter if has_gate_proj else 2 * hidden * inter
    # GPT-2 LayerNorm: weight+bias per LN × 2 = 4*H; LLaMA/Mistral RMSNorm: weight only × 2 = 2*H
    ln_params = 4 * hidden if is_gpt2 else 2 * hidden
    per_layer_params = attn_params + mlp_params + ln_params
    # final norm: 2*H (GPT-2 LN weight+bias) or H (RMSNorm weight)
    final_norm_params = 2 * hidden if is_gpt2 else hidden
    total_params = embed_params + lm_head_params + per_layer_params * n_layers + final_norm_params

    param_bytes = total_params * elem
    grad_bytes = param_bytes
    optim_mul = {"adam": 2, "adamw": 2, "sgd": 0}.get(optimizer, 2)
    optim_bytes = param_bytes * optim_mul

    # --- Activation estimation (tensors saved for backward) ---
    bsh = batch * seq * hidden * elem
    # attention: Q proj output + attention weight matrix + K,V + context output
    q_proj_act = bsh
    attn_weights_act = batch * n_heads * seq * seq * elem
    kv_act = batch * n_kv_heads * seq * head_dim * elem * 2
    attn_out_act = bsh
    # MLP intermediate (gate*up for LLaMA, fc output for GPT-2)
    mlp_act = batch * seq * inter * elem
    if has_gate_proj:
        mlp_act *= 2  # gate_proj + up_proj both saved
    # LayerNorm/RMSNorm: input saved for backward (2 per layer)
    ln_act = 2 * bsh
    # Residual connections: input saved for add backward (2 per layer)
    residual_act = 2 * bsh

    per_layer_act = (q_proj_act + attn_weights_act + kv_act + attn_out_act
                     + mlp_act + ln_act + residual_act)
    total_act = per_layer_act * n_layers

    embed_act = bsh
    logits_act = batch * seq * vocab * elem
    activation_bytes = total_act + embed_act + logits_act

    # --- Absolute peaks (consistent with runtime measure_phased semantics) ---
    static_base = param_bytes + optim_bytes  # after zero_grad(set_to_none=True)

    fw_act = activation_bytes
    # BW activations: all saved activations + gradients being produced.
    # Gradients are NOT added separately — they are part of the BW live set,
    # consistent with L2 where grad tensors are BW-graph output nodes
    # already captured in bw_graph_peak via pin_output_inputs=True.
    bw_act = activation_bytes + grad_bytes

    # Optimizer temporary memory:
    #   foreach Adam (PyTorch default): _foreach_sqrt over all params → param_bytes
    #   fused Adam: single CUDA kernel → 0
    #   SGD: no momentum states → 0
    if optimizer in ("adam", "adamw") and not fused_optimizer:
        opt_temp = param_bytes
    else:
        opt_temp = 0

    fw_peak = static_base + fw_act
    bw_peak = static_base + bw_act
    opt_peak = static_base + grad_bytes + opt_temp
    fwbw_peak = max(fw_peak, bw_peak)
    true_peak = max(fw_peak, bw_peak, opt_peak)

    if true_peak == fw_peak:
        peak_phase = "FW"
    elif true_peak == bw_peak:
        peak_phase = "BW"
    else:
        peak_phase = "OPT"

    # Timeline 7 sample points (for phase_timeline_chart)
    after_fw = static_base + activation_bytes
    after_bw = static_base + grad_bytes
    after_opt = static_base + grad_bytes

    return {
        "tag": "config_L1",
        "param_bytes": param_bytes,
        "grad_bytes": grad_bytes,
        "optimizer_bytes": optim_bytes,
        "activation_bytes": activation_bytes,
        # absolute peaks
        "fw_peak": fw_peak,
        "bw_peak": bw_peak,
        "opt_peak": opt_peak,
        "fwbw_peak": fwbw_peak,
        "true_peak": true_peak,
        "estimated_peak": true_peak,  # backward-compatible alias
        "peak_phase": peak_phase,
        "opt_temp": opt_temp,
        # timeline sample points
        "base": static_base,
        "after_fw": after_fw,
        "after_bw": after_bw,
        "after_opt": after_opt,
    }
