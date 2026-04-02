def run_runtime_signature_compare(args, ctx: dict, model_map: dict, strategy_config: dict):
    import torch

    from model import device as default_device
    from model_loader import build_transformer_from_map
    from aten_recompute.core import CompilerBackend
    from meta_pipeline import (
        compare_capture_semantics,
        inject_transformer_layer_tags,
        print_capture_semantics_report,
        run_train_step,
    )

    if not (args.mode == "static" and args.compare_runtime and torch.cuda.is_available()):
        return

    backend = ctx["backend"]
    tgt_vocab_size = ctx["tgt_vocab_size"]
    criterion = ctx["criterion"]
    src_data = ctx["src_data"]
    tgt_data = ctx["tgt_data"]

    print("=" * 68)
    print("  static 与 runtime 图签名对照")
    print("=" * 68)

    rt_model, _ = build_transformer_from_map(
        model_map=model_map, seq_len=args.seq_len, device=default_device
    )
    inject_transformer_layer_tags(rt_model)
    rt_backend = CompilerBackend(
        strategy_config=strategy_config,
        save_ir=False,
        mode="runtime",
        use_meta=False,
        use_decomp=True,
    )
    rt_compiled = torch.compile(rt_model, backend=rt_backend, dynamic=args.dynamic)

    rt_src = src_data.to(default_device)
    rt_tgt = tgt_data.to(default_device)
    rt_model.train()
    run_train_step(rt_compiled, rt_src, rt_tgt, tgt_vocab_size, criterion)

    report = compare_capture_semantics(backend, rt_backend)
    print_capture_semantics_report(report)
    ctx["meta_runtime_compare"] = report
    return report
