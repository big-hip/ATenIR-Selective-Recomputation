import os
from typing import Optional, Tuple

import torch.fx as fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

__all__ = [
    "save_fx_module_code_and_graph",
    "save_graphmodule_readable",
    "save_ir_and_dot",
]


def _default_ir_dir(model_name: str, subfolder: Optional[str] = None) -> str:
    """
    根据项目根目录、模型名和可选的 RUN_ID 构造统一的 IR 输出目录:
        <PROJECT_ROOT>/IR_artifacts/<model_name>[/runs/<RUN_ID>][/subfolder]
    """
    project_root = os.getenv("PROJECT_ROOT", os.getcwd())
    base_dir = os.path.join(project_root, "IR_artifacts", model_name)

    # 通过 RUN_ID 隔离不同批次的实验数据
    run_id = os.getenv("RUN_ID")
    if run_id:
        base_dir = os.path.join(base_dir, "runs", run_id)

    if subfolder:
        base_dir = os.path.join(base_dir, subfolder)

    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_fx_module_code_and_graph(
    fx_mod: fx.GraphModule,
    code_md_path: Optional[str] = None,
    ir_md_path: Optional[str] = None,
    model_name: Optional[str] = None,
    subfolder: Optional[str] = "capture",
) -> Tuple[str, str]:
    """
    保存 FX GraphModule 的底层 Python code 与纯文本 graph 到文件。
    """
    if code_md_path is None or ir_md_path is None:
        model = model_name or os.getenv("MODEL_NAME", "default_model")
        out_dir = _default_ir_dir(model, subfolder=subfolder or "capture")
        code_md_path = code_md_path or os.path.join(out_dir, "Fw_torch_Code.md")
        ir_md_path = ir_md_path or os.path.join(out_dir, "Fw_torch_IR.md")

    code_str = fx_mod.code
    with open(code_md_path, "w", encoding="utf-8") as f:
        f.write(code_str)

    graph_str = str(fx_mod.graph)
    with open(ir_md_path, "w", encoding="utf-8") as f:
        f.write(graph_str)

    return code_str, graph_str


def save_graphmodule_readable(
    gm: fx.GraphModule,
    md_path: Optional[str] = None,
    model_name: Optional[str] = None,
    subfolder: Optional[str] = "recompute",
) -> str:
    """
    将 GraphModule 格式化为人类可读的表格状 IR 并保存。
    """
    if md_path is None:
        model = model_name or os.getenv("MODEL_NAME", "default_model")
        out_dir = _default_ir_dir(model, subfolder=subfolder or "recompute")
        md_path = os.path.join(out_dir, "graph_readable.md")

    # print_output=False 是 PyTorch 原生支持直接返回字符串的优雅写法
    text = gm.print_readable(print_output=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def save_ir_and_dot(
    fx_mod: fx.GraphModule,
    model_name: Optional[str] = None,
    subfolder: str = "capture",
    graph_name: str = "FxGraph",
) -> Tuple[str, str]:
    """
    一键式存档：同时保存 可读IR、DOT可视化结构图、底层Code。
    在重计算 Pass 的前后调用此函数，可完美对比优化结果。
    """
    model = model_name or os.getenv("MODEL_NAME", "default_model")
    out_dir = _default_ir_dir(model, subfolder=subfolder)

    ir_md_path = os.path.join(out_dir, f"{graph_name}_IR.md")
    dot_path = os.path.join(out_dir, f"{graph_name}.dot")
    code_path = os.path.join(out_dir, f"{graph_name}_Code.py")

    # 1. 保存高可读性的 IR 文本
    ir_str = fx_mod.print_readable(print_output=False)
    with open(ir_md_path, "w", encoding="utf-8") as f:
        f.write(ir_str)

    # 2. 保存 DOT 文件，用于网络拓扑可视化
    try:
        drawer = FxGraphDrawer(fx_mod, graph_name)
        dot_str = str(drawer.get_dot_graph())
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write(dot_str)
    except Exception as e:
        print(f"[警告] 生成 DOT 文件失败，可能是节点存在不兼容类型: {e}")

    # 3. 保存底层生成的 Python Forward 代码
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(fx_mod.code)

    return ir_md_path, dot_path