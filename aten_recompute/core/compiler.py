"""
compiler.py

CompilerBackend：torch.compile 自定义后端，注入选择性重计算 partition_fn。
切分后的 FW/BW 图直接交给 fw_compiler / bw_compiler 执行，无需手动包装。
"""
import copy
import os
from typing import Optional

import torch
import torch.fx as fx
from torch._functorch.aot_autograd import aot_module_simplified
from torch._guards import detect_fake_mode
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.virtualized import V

from .partition import make_selective_partition_fn
from ..utils.logger import get_logger
from ..utils.save_ir import save_ir_and_dot

logger = get_logger(__name__)


class CompilerBackend:
    """
    torch.compile 自定义后端：注入选择性重计算 partition_fn。

    partition_fn 在 AOT Autograd 的 joint graph 切分阶段运行，
    控制哪些中间激活保存、哪些在反向中重计算。切分后的 FW/BW 图
    直接通过 fw_compiler / bw_compiler 执行。

    用法::

        backend = CompilerBackend(strategy_config={"6": 0})
        compiled_model = torch.compile(model, backend=backend)

        # 训练直接使用 compiled_model，无需额外包装
        output = compiled_model(*inputs)
        loss.backward()

    Parameters
    ----------
    strategy_config : dict, optional
        重计算策略配置，格式与 RECOMPUTE 环境变量相同。
        若为 None，则从 RECOMPUTE 环境变量读取。
    save_ir : bool
        是否保存切分后的 FW/BW 图 IR 产物。
    """

    def __init__(self, strategy_config: Optional[dict] = None, save_ir: bool = False):
        self.strategy_config = strategy_config
        self.save_ir = save_ir
        self.fw_gm: Optional[fx.GraphModule] = None
        self.bw_gm: Optional[fx.GraphModule] = None

    def __call__(self, gm: fx.GraphModule, sample_inputs: list):
        """torch.compile backend 回调。"""
        partition_fn = make_selective_partition_fn(self.strategy_config)

        def fw_compiler(gm, _sample_inputs):
            self.fw_gm = copy.deepcopy(gm)
            if self.save_ir:
                model_name = os.getenv("MODEL_NAME", "default_model")
                save_ir_and_dot(gm, model_name=model_name,
                                subfolder="partition", graph_name="FW_partitioned")
            return compile_fx_inner(gm, _sample_inputs)

        def bw_compiler(gm, _sample_inputs):
            self.bw_gm = copy.deepcopy(gm)
            if self.save_ir:
                model_name = os.getenv("MODEL_NAME", "default_model")
                save_ir_and_dot(gm, model_name=model_name,
                                subfolder="partition", graph_name="BW_partitioned")
            return compile_fx_inner(gm, _sample_inputs, is_backward=True)

        fake_mode = detect_fake_mode(sample_inputs)
        if not fake_mode:
            fake_mode = torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)

        with V.set_fake_mode(fake_mode):
            return aot_module_simplified(
                gm, sample_inputs,
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                partition_fn=partition_fn,
                decompositions=select_decomp_table(),
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  旧 API 已删除。若需要旧 GraphCapture，请查看 git 历史。
# ═══════════════════════════════════════════════════════════════════════════════
