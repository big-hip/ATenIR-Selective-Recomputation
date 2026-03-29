import copy


def run_train_validation(
    model_map: dict,
    bcfg: dict,
    strategy_config: dict,
    model_name: str,
    line: str,
    bold: str,
):
    import torch
    import torch.nn as nn

    from model import device as default_device
    from model_loader import build_transformer_from_map
    from aten_recompute.core import CompilerBackend, describe_strategy
    from meta_pipeline import inject_transformer_layer_tags

    src_vocab_size = int(model_map["src_vocab_size"])
    tgt_vocab_size = int(model_map["tgt_vocab_size"])
    d_model = int(model_map["d_model"])
    num_heads = int(model_map["num_heads"])
    num_layers = int(model_map["num_layers"])
    d_ff = int(model_map["d_ff"])
    max_seq_length = int(model_map["max_seq_length"])
    dropout = float(model_map["dropout"])
    padding_idx = int(model_map["padding_idx"])
    batch_size = int(bcfg["batch_size"])
    n_steps = int(bcfg["n_steps"])

    transformer, device = build_transformer_from_map(
        model_map=model_map, seq_len=max_seq_length, device=default_device
    )

    print(f"\n{bold}")
    print("  ATenIR Selective Recomputation")
    print(bold)
    print(f"  模型:         {model_name} ({num_layers} layers, d_model={d_model})")
    print(f"  设备:         {device}")
    print(f"  重计算策略:   {describe_strategy(strategy_config)}")
    print(f"  批次大小:     {batch_size}")
    print(f"  序列长度:     {max_seq_length}")
    print(f"  训练步数:     {n_steps}")
    print(line)

    clean_model = copy.deepcopy(transformer)

    inject_transformer_layer_tags(transformer)

    src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length)).to(device)
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    transformer.train()

    print(f"\n[阶段 1/5] 编译模型")
    print(line)
    backend = CompilerBackend(strategy_config=strategy_config, save_ir=True)
    compiled_transformer = torch.compile(transformer, backend=backend, dynamic=True)

    print(f"\n[阶段 2/5] 训练验证 ({n_steps} 步)")
    print(line)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9
    )

    for step in range(n_steps):
        optimizer.zero_grad()
        output = compiled_transformer(src_data, tgt_data[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()

        if step % 2 == 0 or step == n_steps - 1:
            print(f"  step {step:3d} | loss = {loss.item():.4f}")

    print(f"{line}")
    print("  训练验证完成")

    return {
        "torch": torch,
        "transformer": transformer,
        "compiled_transformer": compiled_transformer,
        "backend": backend,
        "criterion": criterion,
        "src_data": src_data,
        "tgt_data": tgt_data,
        "clean_model": clean_model,
        "device": device,
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "max_seq_length": max_seq_length,
    }
