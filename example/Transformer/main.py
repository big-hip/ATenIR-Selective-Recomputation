import os
import sys

# 将项目根目录加入 sys.path，保证可以找到 aten_recompute 包
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    # ──────────────────────────────────────────────────────────────────────────
    # 0. 全局配置
    # ──────────────────────────────────────────────────────────────────────────
    os.environ["RECOMPUTE_LOG_LEVEL"] = "DEBUG"

    import copy
    import torch
    import torch.nn as nn

    from Transformer import Transformer, device
    from aten_recompute.core.Recom_pass import RecomputePass
    from aten_recompute.get_Aten_IR.Graph_compile_capture import GraphCapture
    from aten_recompute.utils import save_ir_and_dot, MemoryAnalyzer
    from aten_recompute.core.Tag import inject_layer_tags
    from aten_recompute.core.train_recomputed import (
        run_training, make_recomputed_fn, _collect_params_in_fw_order,
    )

    model_name = os.getenv("MODEL_NAME", "Transformer")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. 初始化模型与数据
    # ──────────────────────────────────────────────────────────────────────────
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model        = 512
    num_heads      = 8
    num_layers     = 6
    d_ff           = 2048
    max_seq_length = 100
    dropout        = 0.1

    transformer = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, num_heads,
        num_layers, d_ff, max_seq_length, dropout,
    )
    transformer.to(device)

    # 在捕获图之前注入层信息标签
    _enc_layers = [(layer, i) for i, layer in enumerate(transformer.encoder_layers)]
    _dec_layers = [
        (layer, len(transformer.encoder_layers) + i)
        for i, layer in enumerate(transformer.decoder_layers)
    ]
    _tag_handles = inject_layer_tags(_enc_layers + _dec_layers)  # 保存 handles 以备移除

    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length)).to(device)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    transformer.train()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. 捕获 FW/BW 图
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/4] 捕获 AOT Autograd FW/BW 图...")
    graph_capture = GraphCapture(transformer, src_data, tgt_data[:, :-1])
    compiled_transformer = graph_capture.compile()

    output = compiled_transformer(src_data, tgt_data[:, :-1])
    loss = criterion(
        output.contiguous().view(-1, tgt_vocab_size),
        tgt_data[:, 1:].contiguous().view(-1),
    )
    loss.backward()
    print("图捕获完成！fw_gm 和 bw_gm 已就绪。")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. 执行重计算 Pass
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/4] 执行重计算 Pass 优化计算图...")

    # 在 Pass 运行前拍快照：
    # _modify_forward_graph 原地修改 fw_gm，估算"之前"状态必须用快照。
    _fw_before = copy.deepcopy(graph_capture.fw_gm)
    _bw_before = copy.deepcopy(graph_capture.bw_gm)
    # 额外保留一份，供 eager 模式对比用（不能复用 _fw_before，它要给静态分析用）
    _fw_for_eager = copy.deepcopy(graph_capture.fw_gm)
    _bw_for_eager = copy.deepcopy(graph_capture.bw_gm)

    # JIT pass（默认）
    recompute_pass = RecomputePass({"FW": graph_capture.fw_gm, "BW": graph_capture.bw_gm})
    fw_gm_opt, bw_gm_opt = recompute_pass.run()

    # Eager pass（在独立拷贝上运行，供运行时对比）
    recompute_pass_eager = RecomputePass(
        {"FW": _fw_for_eager, "BW": _bw_for_eager}, mode='eager'
    )
    fw_gm_eager, bw_gm_eager = recompute_pass_eager.run()

    # ──────────────────────────────────────────────────────────────────────────
    # 3.5 静态激活内存估算（cleanup 之前，meta['val'] 仍完整）
    # ──────────────────────────────────────────────────────────────────────────
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    analyzer = MemoryAnalyzer(device=_device)
    analyzer.estimate_parameter_memory(transformer)
    analyzer.estimate(
        fw_before=_fw_before, bw_before=_bw_before,
        fw_after=fw_gm_opt,   bw_after=bw_gm_opt,
    )
    analyzer.save_report(model_name=model_name)   # 静态结果立即落盘

    # 清理图中分析阶段残留（mark_layer / layer_Rank）
    fw_gm_opt   = GraphCapture.cleanup_graph(fw_gm_opt)
    bw_gm_opt   = GraphCapture.cleanup_graph(bw_gm_opt)
    fw_gm_eager = GraphCapture.cleanup_graph(fw_gm_eager)
    bw_gm_eager = GraphCapture.cleanup_graph(bw_gm_eager)
    print("重计算 Pass 运行完毕！")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. 保存优化后的图
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/4] 保存优化后的图到文件...")
    save_ir_and_dot(fw_gm_opt, model_name=model_name, subfolder="recompute", graph_name="FW_recomputed")
    save_ir_and_dot(bw_gm_opt, model_name=model_name, subfolder="recompute", graph_name="BW_recomputed")
    print("全部完成！请在 IR_artifacts 目录下查看该模型对应的重计算图。")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. 使用优化后的图进行训练验证
    # ──────────────────────────────────────────────────────────────────────────
    if getattr(graph_capture, 'fw_params_flat', None):
        _params_flat = graph_capture.fw_params_flat
    else:
        _params_flat = _collect_params_in_fw_order(transformer)

    _recomputed_forward = make_recomputed_fn(fw_gm_opt, bw_gm_opt, _params_flat)   # JIT 模式
    _optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    def _recomputed_step():
        _optimizer.zero_grad()
        out = _recomputed_forward(src_data, tgt_data[:, :-1])
        _loss = criterion(
            out.contiguous().view(-1, tgt_vocab_size),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        _loss.backward()
        _optimizer.step()
        return _loss

    run_training(fw_gm_opt, bw_gm_opt, transformer, step_fn=_recomputed_step, n_steps=10)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. 运行时峰值显存 & 耗时对比
    #
    # 方法论说明：
    #   四路对比（全部使用同一引擎 Python FX，除 ④ 外）：
    #   ① no_recompute      : Pass 前 FX 图，纯 Python FX 执行（基准）
    #                         save-everything，无 min-cut（default_partition 无
    #                         must_recompute 标签时回落"全部保存"模式）
    #   ② recomputed_jit    : JIT 延迟共享插入，链在第一个 consumer 前实体化
    #   ③ recomputed_eager  : Eager 插入（旧行为），所有链在 BW 开头同时实体化
    #   ④ inductor_baseline : torch.compile Inductor 全优化路径（不同引擎，仅作参考）
    #
    #   ① → ② 差异：重计算 Pass 的纯效果（选对目标时应为负，即节省内存）。
    #   ② → ③ 差异：JIT vs Eager 插入策略的显存差异，体现 JIT 延迟共享的价值。
    #   使用独立的 profiling 专用优化器，与阶段 5 的参数状态相互隔离。
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[4/4] 开始运行时峰值显存 & 耗时对比...")

    # ── 构建 no-recompute baseline（Pass 前 FX 图，同引擎） ──────────────────
    # cleanup_graph 返回 rename_map：{mark_layer节点名: 替换节点名}
    # 若 mark_layer 节点曾出现在 FW 输出中（作为 FW→BW 边界张量），
    # BW 图中对应的 placeholder 名字需同步修正，否则 _detect_fw_calling_convention 会出错。
    _ml_rename: dict = {}
    _fw_no_rc = GraphCapture.cleanup_graph(copy.deepcopy(_fw_before), _ml_rename)
    _bw_no_rc = copy.deepcopy(_bw_before)
    for _node in _bw_no_rc.graph.nodes:
        if _node.op == 'placeholder' and _node.name in _ml_rename:
            _node.name = _ml_rename[_node.name]
        if 'layer_Rank' in _node.kwargs:
            _node.kwargs = {k: v for k, v in _node.kwargs.items() if k != 'layer_Rank'}
    _bw_no_rc.graph.lint()
    _bw_no_rc.recompile()

    _no_recompute_forward = make_recomputed_fn(_fw_no_rc, _bw_no_rc, _params_flat)

    # ── 构建 eager forward（fw_gm_eager / bw_gm_eager 已在阶段 3 中准备好） ──
    _eager_forward = make_recomputed_fn(fw_gm_eager, bw_gm_eager, _params_flat)

    # ── 四个独立优化器，避免各路梯度状态互相污染 ─────────────────────────────
    _prof_opt_no_rc    = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    _prof_opt_jit      = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    _prof_opt_eager    = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    _prof_opt_inductor = torch.optim.Adam(transformer.parameters(), lr=1e-4)

    def _no_recompute_step():
        _prof_opt_no_rc.zero_grad()
        out = _no_recompute_forward(src_data, tgt_data[:, :-1])
        l = criterion(out.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        l.backward()
        _prof_opt_no_rc.step()

    def _recomp_jit_step():
        _prof_opt_jit.zero_grad()
        out = _recomputed_forward(src_data, tgt_data[:, :-1])
        l = criterion(out.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        l.backward()
        _prof_opt_jit.step()

    def _recomp_eager_step():
        _prof_opt_eager.zero_grad()
        out = _eager_forward(src_data, tgt_data[:, :-1])
        l = criterion(out.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        l.backward()
        _prof_opt_eager.step()

    def _inductor_step():
        _prof_opt_inductor.zero_grad()
        out = compiled_transformer(src_data, tgt_data[:, :-1])
        l = criterion(out.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        l.backward()
        _prof_opt_inductor.step()

    # no_recompute 为基准，report() 将自动显示各路与它的差异
    analyzer.profile_step("no_recompute",      fn=_no_recompute_step,  warmup=2, steps=5)
    analyzer.profile_step("recomputed_jit",    fn=_recomp_jit_step,    warmup=2, steps=5)
    analyzer.profile_step("recomputed_eager",  fn=_recomp_eager_step,  warmup=2, steps=5)
    analyzer.profile_step("inductor_baseline", fn=_inductor_step,      warmup=2, steps=5)
    analyzer.report()

    # 覆盖写入完整报告（含运行时数据）
    analyzer.save_report(model_name=model_name)
    print("内存分析报告已更新。")


if __name__ == "__main__":
    main()
