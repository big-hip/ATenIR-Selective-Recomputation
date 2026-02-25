import io
import os
import sys
from typing import Tuple

import torch.fx as fx


def _capture_stdout_to_string(fn) -> str:
    """
    执行给定函数，将其标准输出捕获为字符串返回。
    """
    with io.StringIO() as buf:
        original_stdout = sys.stdout
        try:
            sys.stdout = buf
            fn()
        finally:
            sys.stdout = original_stdout
        return buf.getvalue()


def _default_ir_dir(model_name: str, subfolder: str | None = None) -> str:
    """
    根据项目根目录和模型名，构造统一的 IR 输出目录:
        <PROJECT_ROOT>/IR_artifacts/<model_name>[/runs/<RUN_ID>][/subfolder]
    PROJECT_ROOT 从环境变量 PROJECT_ROOT 读取，若不存在则使用当前工作目录。
    """
    project_root = os.getenv("PROJECT_ROOT", os.getcwd())
    base_dir = os.path.join(project_root, "IR_artifacts", model_name)

    # 如设置了 RUN_ID，则在 model 目录下再分 run 子目录，避免不同运行互相覆盖
    run_id = os.getenv("RUN_ID")
    if run_id:
        base_dir = os.path.join(base_dir, "runs", run_id)
    if subfolder:
        base_dir = os.path.join(base_dir, subfolder)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_fx_module_code_and_graph(
    fx_mod: fx.GraphModule,
    code_md_path: str | None = None,
    ir_md_path: str | None = None,
    model_name: str | None = None,
    subfolder: str | None = "capture",
) -> Tuple[str, str]:
    """
    保存 FX GraphModule 的 Python code 与 graph(IR) 到指定的 .md 文件。

    - 若显式提供 code_md_path / ir_md_path，则直接使用该路径；
    - 否则根据模型名与可选子目录自动生成：
          IR_artifacts/<model_name>/<subfolder>/Fw_torch_Code.md
          IR_artifacts/<model_name>/<subfolder>/Fw_torch_IR.md

    返回写入的字符串内容，便于上层调试或单元测试。
    """
    if code_md_path is None or ir_md_path is None:
        model = model_name or os.getenv("MODEL_NAME", "default_model")
        out_dir = _default_ir_dir(model, subfolder=subfolder)
        code_md_path = code_md_path or os.path.join(out_dir, "Fw_torch_Code.md")
        ir_md_path = ir_md_path or os.path.join(out_dir, "Fw_torch_IR.md")

    code_str = _capture_stdout_to_string(lambda: print(fx_mod.code))
    with open(code_md_path, "w") as f:
        f.write(code_str)

    graph_str = _capture_stdout_to_string(lambda: print(fx_mod.graph))
    with open(ir_md_path, "w") as f:
        f.write(graph_str)

    return code_str, graph_str


def save_graphmodule_readable(
    gm: fx.GraphModule,
    md_path: str | None = None,
    model_name: str | None = None,
    subfolder: str | None = "recompute",
) -> str:
    """
    调用 GraphModule.print_readable() 并将结果保存到指定 .md 文件。

    - 若显式提供 md_path，则直接使用；
    - 否则根据模型名与可选子目录自动生成：
          IR_artifacts/<model_name>/<subfolder>/<默认文件名>.md
      默认文件名由调用方语义决定，这里统一使用：
          FW_recomputed.md / BW_recomputed.md 等等，需由上层传入 md_path。

    返回写入的字符串内容。
    """
    if md_path is None:
        model = model_name or os.getenv("MODEL_NAME", "default_model")
        out_dir = _default_ir_dir(model, subfolder=subfolder)
        md_path = os.path.join(out_dir, "graph.md")

    text = _capture_stdout_to_string(lambda: gm.print_readable())
    with open(md_path, "w") as f:
        f.write(text)
    return text


__all__ = ["save_fx_module_code_and_graph", "save_graphmodule_readable"]

