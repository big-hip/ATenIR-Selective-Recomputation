import time


def execute_capture(args, model_map: dict, strategy_config: dict, run_device):
    import torch
    import torch.nn as nn

    from model_loader import build_transformer_from_map
    from aten_recompute.core import CompilerBackend
    from meta_pipeline import inject_transformer_layer_tags, run_train_step

    src_vocab_size = int(model_map["src_vocab_size"])
    tgt_vocab_size = int(model_map["tgt_vocab_size"])
    padding_idx = int(model_map["padding_idx"])

    transformer, _ = build_transformer_from_map(
        model_map=model_map, seq_len=args.seq_len, device=run_device
    )
    inject_transformer_layer_tags(transformer)

    backend = CompilerBackend(
        strategy_config=strategy_config,
        save_ir=True,
        mode=args.mode,
        use_meta=(args.mode == "static"),
        use_decomp=True,
    )
    compiled = torch.compile(transformer, backend=backend, dynamic=args.dynamic)

    torch.manual_seed(0)
    src_data = torch.randint(1, src_vocab_size, (args.batch_size, args.seq_len)).to(run_device)
    tgt_data = torch.randint(1, tgt_vocab_size, (args.batch_size, args.seq_len)).to(run_device)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    if args.mode == "static" and args.static_profile == "fast":
        transformer.eval()
    else:
        transformer.train()

    t0 = time.time()
    loss_value = None
    static_exec_error = None
    fallback_used = False
    try:
        loss_value = run_train_step(compiled, src_data, tgt_data, tgt_vocab_size, criterion)
    except Exception as exc:
        if args.mode != "static":
            raise

        static_exec_error = str(exc).splitlines()[0]
        if (backend.fw_gm is None or backend.bw_gm is None) and args.static_profile == "high":
            from torch import _dynamo

            _dynamo.reset()
            transformer.eval()
            backend = CompilerBackend(
                strategy_config=strategy_config,
                save_ir=True,
                mode="static",
                use_meta=True,
                use_decomp=True,
            )
            compiled = torch.compile(transformer, backend=backend, dynamic=args.dynamic)
            loss_value = run_train_step(compiled, src_data, tgt_data, tgt_vocab_size, criterion)
            fallback_used = True
        elif backend.fw_gm is None or backend.bw_gm is None:
            raise

    elapsed = time.time() - t0

    compile_debug = getattr(backend, "last_compile_debug", None)
    if compile_debug:
        print("  [debug] compile path summary:")
        print(
            "    mode={mode} | use_meta_flag={use_meta_flag} | use_decomp={use_decomp} | fake_mode={fake_mode_type}".format(
                **compile_debug
            )
        )
        for item in compile_debug.get("sample_inputs", []):
            if "shape" in item:
                print(
                    "    input[{index}] type={type} device={device} dtype={dtype} shape={shape} is_meta={is_meta}".format(
                        **item
                    )
                )
            else:
                print("    input[{index}] type={type}".format(**item))

    return {
        "transformer": transformer,
        "backend": backend,
        "compiled": compiled,
        "criterion": criterion,
        "src_data": src_data,
        "tgt_data": tgt_data,
        "tgt_vocab_size": tgt_vocab_size,
        "loss_value": loss_value,
        "elapsed": elapsed,
        "static_exec_error": static_exec_error,
        "fallback_used": fallback_used,
    }
