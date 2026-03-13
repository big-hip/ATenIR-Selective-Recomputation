"""
捕获 AOT Autograd 联合图（Joint Graph）示例。
联合图是前向+反向在分区（partition）前的合并 FX 图。
通过替换 partition_fn 来拦截并保存联合图。
"""

import torch
from torch._functorch.aot_autograd import aot_function
from torch._functorch.partitioners import default_partition


captured_joint_graph = None


def capturing_partition_fn(joint_graph, joint_inputs, *args, **kwargs):
    global captured_joint_graph
    captured_joint_graph = joint_graph
    print("\n========== Joint Graph (前向 + 反向联合图) ==========")
    joint_graph.print_readable()
    print("======================================================\n")
    return default_partition(joint_graph, joint_inputs, *args, **kwargs)


def simple_mlp(x, w1, w2):
    h = torch.nn.functional.linear(x, w1)
    h = torch.relu(h)
    out = torch.nn.functional.linear(h, w2)
    return out


def main():
    torch.manual_seed(42)

    x  = torch.randn(4, 8,  requires_grad=False)
    w1 = torch.randn(16, 8, requires_grad=True)
    w2 = torch.randn(4, 16, requires_grad=True)

    compiled = aot_function(
        simple_mlp,
        fw_compiler=lambda gm, _: gm.forward,
        bw_compiler=lambda gm, _: gm.forward,
        partition_fn=capturing_partition_fn,
    )

    out = compiled(x, w1, w2)
    loss = out.sum()
    loss.backward()

    print("前向输出 shape:", out.shape)
    print("w1.grad shape :", w1.grad.shape)
    print("w2.grad shape :", w2.grad.shape)

    if captured_joint_graph is not None:
        print("\n联合图节点列表：")
        for node in captured_joint_graph.graph.nodes:
            print(f"  {node.op:15s}  {str(node.target):50s}  {node.name}")


if __name__ == "__main__":
    main()
