"""
Graph_compile_capture.py

GraphCapture：通过 torch.compile 的自定义 backend 捕获 FW/BW GraphModule，
并提供 cleanup_graph() 用于训练前清理图中的分析阶段残留。
"""
import copy
import os
from collections import deque

import torch
import torch.fx as fx
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified

from ..utils.logger import get_logger
from ..utils.save_ir import save_fx_module_code_and_graph, save_ir_and_dot

logger = get_logger(__name__)


class GraphCapture:
    """
    捕获模型的 AOT Autograd FW/BW 图。

    工作流::

        gc = GraphCapture(model, *inputs)
        compiled = gc.compile()        # 注册 backend 回调
        compiled(*inputs)              # warm-up，触发真正的图捕获
        loss.backward()
        # gc.fw_gm / gc.bw_gm 现已可用

    Attributes
    ----------
    fw_gm        : 捕获到的前向 GraphModule（warm-up 后填充）
    bw_gm        : 捕获到的反向 GraphModule（backward 后填充）
    fw_params_flat : 按前向访问顺序排列的模型参数/buffer 列表
    """

    def __init__(self, model: torch.nn.Module, *inputs):
        self.fw_gm = fx.GraphModule(torch.nn.Module(), fx.Graph())
        self.bw_gm = fx.GraphModule(torch.nn.Module(), fx.Graph())
        self.fw_params_flat: list = []
        self.model = model
        self.inputs = inputs

    # ── backend 回调 ──────────────────────────────────────────────────────────

    def inspect_backend(self, gm: fx.GraphModule, sample_inputs: list):
        """
        torch.compile backend：捕获 FW/BW 图并提取参数顺序。

        sample_inputs 是 Dynamo 传入的真实张量（含模型参数），
        按前向访问顺序排列，通过 data_ptr() 过滤出参数/buffer。
        """
        # 提取 FW 图 primal 的参数顺序（data_ptr 匹配）
        all_params: dict = {}
        for _, p in self.model.named_parameters(remove_duplicate=False):
            all_params[p.data_ptr()] = p
        for _, b in self.model.named_buffers():
            all_params[b.data_ptr()] = b

        seen: set = set()
        fw_params: list = []
        for t in sample_inputs:
            if isinstance(t, torch.Tensor):
                ptr = t.data_ptr()
                if ptr in all_params and ptr not in seen:
                    seen.add(ptr)
                    fw_params.append(all_params[ptr])
        self.fw_params_flat = fw_params

        def fw(gm, _sample_inputs):
            self.fw_gm = copy.deepcopy(gm)
            return make_boxed_func(gm.forward)

        def bw(gm, _sample_inputs):
            self.bw_gm = copy.deepcopy(gm)
            return make_boxed_func(gm.forward)

        return aot_module_simplified(gm, sample_inputs, fw_compiler=fw, bw_compiler=bw)

    # ── 编译入口 ──────────────────────────────────────────────────────────────

    def compile(self) -> torch.nn.Module:
        """
        导出静态 FX 图并注册 inspect_backend，返回可调用的编译模型。
        首次 forward + backward 调用时触发真正的图捕获。
        """
        import torch._dynamo as dynamo

        fx_mod, _guards = dynamo.export(self.model, aten_graph=False)(*self.inputs)
        fx_mod.graph.lint()
        fx_mod.recompile()

        model_name = os.getenv("MODEL_NAME", "default_model")
        save_fx_module_code_and_graph(fx_mod, model_name=model_name, subfolder="capture")
        save_ir_and_dot(fx_mod, model_name=model_name, subfolder="capture", graph_name="FW_capture")

        return torch.compile(fx_mod, backend=self.inspect_backend, dynamic=True)

    # ── 图清理 ────────────────────────────────────────────────────────────────

    @staticmethod
    def cleanup_graph(gm: fx.GraphModule) -> fx.GraphModule:
        """
        在图交给训练执行之前，清理分析阶段注入的残留。

        Pass 1：将所有 mark_layer 节点替换为其输入张量（mark_layer 是恒等映射）。
        Pass 2：清除所有节点 kwargs 中的 layer_Rank 残留键。

        Parameters
        ----------
        gm : 待清理的 GraphModule（原地修改）。

        Returns
        -------
        清理并重编译后的同一个 GraphModule。

        Raises
        ------
        AssertionError : 若 Pass 1 未能彻底移除所有 mark_layer 节点。
        """
        graph = gm.graph

        # Pass 1：替换 mark_layer 节点
        for node in list(graph.nodes):
            if node.op == 'call_function' and 'mark_layer' in str(node.target):
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)

        remaining = [
            n for n in graph.nodes
            if n.op == 'call_function' and 'mark_layer' in str(n.target)
        ]
        assert not remaining, (
            f"cleanup_graph: Pass 1 未能移除 {len(remaining)} 个 mark_layer 节点: "
            f"{[n.name for n in remaining]}"
        )

        # Pass 2：清除所有节点残留的 layer_Rank kwarg
        for node in graph.nodes:
            if 'layer_Rank' in node.kwargs:
                node.kwargs = {k: v for k, v in node.kwargs.items() if k != 'layer_Rank'}

        graph.lint()
        gm.recompile()
        return gm
