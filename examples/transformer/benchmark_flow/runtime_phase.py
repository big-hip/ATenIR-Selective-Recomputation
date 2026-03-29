import copy


def run_runtime_analysis(
    train_ctx: dict,
    static_ctx: dict,
    strat_tag: str,
    model_name: str,
    line: str,
    bold: str,
):
    torch = train_ctx["torch"]
    transformer = train_ctx["transformer"]
    compiled_transformer = train_ctx["compiled_transformer"]
    criterion = train_ctx["criterion"]
    src_data = train_ctx["src_data"]
    tgt_data = train_ctx["tgt_data"]
    clean_model = train_ctx["clean_model"]
    tgt_vocab_size = train_ctx["tgt_vocab_size"]

    from aten_recompute.analysis import MemoryProfiler, print_method_comparison
    from aten_recompute.core import CompilerBackend
    from aten_recompute.utils import apply_activation_checkpoint

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name != "cuda":
        print(f"\n[阶段 4/5] 跳过显存分析（非 GPU 环境）")
        print(bold)
        return None

    print(f"\n[阶段 4/5] 运行时峰值显存 & 耗时对比")
    print(line)
    analyzer = MemoryProfiler(device=device_name)
    analyzer.estimate_parameter_memory(transformer)

    def gpu_cleanup():
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_name)

    eager_model = copy.deepcopy(clean_model)
    eager_model.to(train_ctx["device"])
    eager_model.train()
    eager_opt = torch.optim.Adam(eager_model.parameters(), lr=1e-4)

    def eager_forward():
        out = eager_model(src_data, tgt_data[:, :-1])
        return criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )

    analyzer.profile_step("eager_baseline", forward_fn=eager_forward, optimizer=eager_opt)
    del eager_model, eager_opt
    gpu_cleanup()

    compiled_nr_model = copy.deepcopy(clean_model)
    compiled_nr_model.to(train_ctx["device"])
    compiled_nr_model.train()
    nr_backend = CompilerBackend(strategy_config={"0": None}, save_ir=False)
    compiled_nr = torch.compile(compiled_nr_model, backend=nr_backend, dynamic=True)
    nr_opt = torch.optim.Adam(compiled_nr_model.parameters(), lr=1e-4)

    def compiled_nr_forward():
        out = compiled_nr(src_data, tgt_data[:, :-1])
        return criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )

    analyzer.profile_step("compiled_no_recompute", forward_fn=compiled_nr_forward, optimizer=nr_opt)
    del compiled_nr_model, compiled_nr, nr_backend, nr_opt
    gpu_cleanup()

    ckpt_model = copy.deepcopy(clean_model)
    ckpt_model.to(train_ctx["device"])
    ckpt_model.train()
    apply_activation_checkpoint(
        ckpt_model,
        module_lists=[ckpt_model.encoder_layers, ckpt_model.decoder_layers],
        use_reentrant=False,
    )
    ckpt_opt = torch.optim.Adam(ckpt_model.parameters(), lr=1e-4)

    def ckpt_forward():
        out = ckpt_model(src_data, tgt_data[:, :-1])
        return criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )

    analyzer.profile_step("pytorch_checkpoint", forward_fn=ckpt_forward, optimizer=ckpt_opt)
    del ckpt_model, ckpt_opt
    gpu_cleanup()

    try:
        import functools
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            checkpoint as torch_checkpoint,
            create_selective_checkpoint_contexts,
        )

        compute_ops = {
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
        }
        for attn_op in (
            "torch.ops.aten._scaled_dot_product_flash_attention.default",
            "torch.ops.aten._scaled_dot_product_efficient_attention.default",
        ):
            try:
                compute_ops.add(eval(attn_op))
            except (AttributeError, Exception):
                pass

        def sac_policy(ctx, op, *args, **kwargs):
            if op in compute_ops:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        sac_context_fn = functools.partial(create_selective_checkpoint_contexts, sac_policy)

        sac_model = copy.deepcopy(clean_model)
        sac_model.to(train_ctx["device"])
        sac_model.train()

        for module_list in [sac_model.encoder_layers, sac_model.decoder_layers]:
            for layer in module_list:
                orig_forward = layer.forward

                def make_sac_forward(orig_fwd, ctx_fn):
                    def sac_forward(*args, **kwargs):
                        return torch_checkpoint(
                            orig_fwd,
                            *args,
                            use_reentrant=False,
                            context_fn=ctx_fn,
                            **kwargs,
                        )

                    return sac_forward

                layer.forward = make_sac_forward(orig_forward, sac_context_fn)

        compiled_sac = torch.compile(sac_model, dynamic=True)
        sac_opt = torch.optim.Adam(sac_model.parameters(), lr=1e-4)

        def sac_forward():
            out = compiled_sac(src_data, tgt_data[:, :-1])
            return criterion(
                out.contiguous().view(-1, tgt_vocab_size),
                tgt_data[:, 1:].contiguous().view(-1),
            )

        analyzer.profile_step("pytorch_SAC", forward_fn=sac_forward, optimizer=sac_opt)
        del sac_model, compiled_sac, sac_opt
        gpu_cleanup()

    except (ImportError, Exception) as sac_err:
        print(f"  [pytorch_SAC] 跳过：当前 PyTorch 版本不支持 SAC ({sac_err})")

    prof_opt = torch.optim.Adam(transformer.parameters(), lr=1e-4)

    def recomputed_forward():
        out = compiled_transformer(src_data, tgt_data[:, :-1])
        return criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )

    analyzer.profile_step("ATenIR_recompute", forward_fn=recomputed_forward, optimizer=prof_opt)

    captured_graphs = static_ctx["captured_graphs"]
    if "no_recompute" in captured_graphs and strat_tag in captured_graphs:
        nr_fw, nr_bw = captured_graphs["no_recompute"]
        rc_fw, rc_bw = captured_graphs[strat_tag]
        analyzer.estimate(
            fw_before=nr_fw,
            bw_before=nr_bw,
            fw_after=rc_fw,
            bw_after=rc_bw,
        )

    analyzer.compare_with_static(static_ctx["static_estimator"])

    analyzer.report()
    report_path = analyzer.save_report(model_name=model_name)

    print_method_comparison()

    print(bold)
    print(f"  完成。报告已保存至: {report_path}")
    print(f"{bold}\n")

    return report_path
