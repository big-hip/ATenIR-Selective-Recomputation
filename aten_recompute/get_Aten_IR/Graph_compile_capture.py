import torch
import torch.fx as fx
from torch._functorch.aot_autograd import aot_module_simplified
from functorch.compile import make_boxed_func
import copy
import sys, io
import torch._dynamo as dynamo
from .. import logger
class GraphCapture:
    def __init__(self, model, *input):
        self.FW_gm = fx.GraphModule(torch.nn.Module(), fx.Graph())
        self.BW_gm = fx.GraphModule(torch.nn.Module(), fx.Graph())
        self.model = model
        self.input = input
    
    def inspect_backend(self, gm, sample_inputs):
        def fw(gm, sample_inputs):
            self.FW_gm = copy.deepcopy(gm)
            '''
            for node in self.FW_gm.graph.nodes:
                print(node.name,': ',node.meta.get('stack_trace'))

            graph = self.FW_gm.graph.__deepcopy__()
            verbose_python_code = graph.python_code(
            root_module="self",
            verbose=True,
        )
            module_code = verbose_python_code.src
            print(module_code)
            self.print_readable(self.FW_gm,self.FW_gm._get_name())
            '''
            return make_boxed_func(gm.forward)
        
        def bw(gm, sample_inputs):
            self.BW_gm = copy.deepcopy(gm)
            '''
            for node in self.BW_gm.graph.nodes:
                print(node.name,': ',node.meta.get('stack_trace'))
            '''
            return make_boxed_func(gm.forward)
            
        return aot_module_simplified(gm, sample_inputs, fw_compiler=fw, bw_compiler=bw)

    def compile(self):
        # 1) 导出一个纯 Python‐API 级别的 FX GraphModule
        fx_mod, guards = dynamo.export(
            self.model,
            aten_graph=False,
        )(*self.input)

        for node in fx_mod.graph.nodes:
            if node.meta.get('stack_trace',None):
                node.name = node.name 
                logger.info(node.name)
                node.meta['stack_trace'] =  node.name + str(node.target) + node.meta['stack_trace']
                # print(node.meta)
                logger.info(f'{node.meta}')

        fx_mod.graph.lint()
        fx_mod.recompile()
        
        # 2) 保存GraphModule的code
        with io.StringIO() as buf:
            original_stdout = sys.stdout
            sys.stdout = buf
            print(fx_mod.code)
            sys.stdout = original_stdout
            output = buf.getvalue()
        with open('Fw_torch_Code.md', 'w') as f:
            # 写入捕获的输出
            f.write(output)
                
        # 3) 保存GraphModule的graph/Torch IR
        with io.StringIO() as buf:
            original_stdout = sys.stdout
            sys.stdout = buf
            print(fx_mod.graph)
            sys.stdout = original_stdout
            output = buf.getvalue()
        with open('Fw_torch_IR.md', 'w') as f:
            # 写入捕获的输出
            f.write(output)

        # 4) 把这个 GraphModule 编译一次
        compiled_model  = torch.compile(
            fx_mod,                        # 传入 GraphModule
            backend= self.inspect_backend ,
            dynamic= True                   # 如果你希望支持动态形状
        )
        return compiled_model
    