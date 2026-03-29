def run_runtime_signature_compare(args, ctx: dict, model_map: dict, strategy_config: dict):
    import torch

    from model import device as default_device
    from model_loader import build_transformer_from_map
    from aten_recompute.core import CompilerBackend
    from meta_pipeline import compare_graph_signatures, inject_transformer_layer_tags, run_train_step

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

    fw_cmp = compare_graph_signatures(backend.fw_gm, rt_backend.fw_gm)
    bw_cmp = compare_graph_signatures(backend.bw_gm, rt_backend.bw_gm)
    print(
        f"  [FW] meta/runtime 节点数: {fw_cmp['meta_nodes']}/{fw_cmp['runtime_nodes']} "
        f"(ratio={fw_cmp['ratio']:.3f})"
    )
    print(f"  [FW] 算子集合重叠: {fw_cmp['overlap']:.1%}")
    if fw_cmp["top_diffs"]:
        print("  [FW] Top-5 算子计数差异 (meta - runtime):")
        for item in fw_cmp["top_diffs"]:
            print(f"    {item['op']}: {item['delta']:+d}")

    print(
        f"  [BW] meta/runtime 节点数: {bw_cmp['meta_nodes']}/{bw_cmp['runtime_nodes']} "
        f"(ratio={bw_cmp['ratio']:.3f})"
    )
    print(f"  [BW] 算子集合重叠: {bw_cmp['overlap']:.1%}")
    if bw_cmp["top_diffs"]:
        print("  [BW] Top-5 算子计数差异 (meta - runtime):")
        for item in bw_cmp["top_diffs"]:
            print(f"    {item['op']}: {item['delta']:+d}")
