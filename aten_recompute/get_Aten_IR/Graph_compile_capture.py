import torch
import torch.fx as fx
from torch._functorch.aot_autograd import aot_module_simplified
from functorch.compile import make_boxed_func
import copy
import os
import torch._dynamo as dynamo
from aten_recompute import logger
from aten_recompute.utils import save_fx_module_code_and_graph, save_ir_and_dot
class GraphCapture:
    def __init__(self, model, *inputs):
        self.FW_gm = fx.GraphModule(torch.nn.Module(), fx.Graph())
        self.BW_gm = fx.GraphModule(torch.nn.Module(), fx.Graph())
        self.model = model
        self.inputs = inputs
    
    def inspect_backend(self, gm, sample_inputs):
        def fw(gm, _sample_inputs):
            self.FW_gm = copy.deepcopy(gm)
            return make_boxed_func(gm.forward)
        
        def bw(gm, _sample_inputs):
            self.BW_gm = copy.deepcopy(gm)
            return make_boxed_func(gm.forward)
            
        return aot_module_simplified(gm, sample_inputs, fw_compiler=fw, bw_compiler=bw)

    def compile(self):
        # 1) 导出一个纯 Python‐API 级别的 FX GraphModule
        fx_mod, _guards = dynamo.export(
            self.model,
            aten_graph=False,
        )(*self.inputs)

        for node in fx_mod.graph.nodes:
            if node.meta.get('stack_trace',None):
                node.name = node.name 
                logger.info(node.name)
                node.meta['stack_trace'] =  node.name + str(node.target) + node.meta['stack_trace']
                # print(node.meta)
                logger.info("%s", node.meta)

        fx_mod.graph.lint()
        fx_mod.recompile()

        # 2) 保存 GraphModule 的 code 与 graph/Torch IR
        # 使用环境变量 MODEL_NAME / PROJECT_ROOT 自动放入:
        #   IR_artifacts/<model_name>/capture/
        model_name = os.getenv("MODEL_NAME", "default_model")
        save_fx_module_code_and_graph(
            fx_mod,
            model_name=model_name,
            subfolder="capture",
        )
        # 同时保存 DOT（以及一份可读 IR / code），便于可视化对比
        # 输出路径:
        #   IR_artifacts/<model_name>/runs/<RUN_ID>/capture/FW_capture.dot
        save_ir_and_dot(
            fx_mod,
            model_name=model_name,
            subfolder="capture",
            graph_name="FW_capture",
        )

        # 4) 把这个 GraphModule 编译一次
        compiled_model  = torch.compile(
            fx_mod,                        # 传入 GraphModule
            backend= self.inspect_backend ,
            dynamic= True                   # 如果你希望支持动态形状
        )
        return compiled_model
    