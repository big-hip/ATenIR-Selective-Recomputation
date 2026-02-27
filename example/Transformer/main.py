import os
import sys

# 将项目根目录加入 sys.path，保证可以找到 aten_recompute 包
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def cleanup_for_training(gm):
    """
    在训练前清理图中与分析阶段相关、但运行时不可执行的节点/kwargs：
    1. 将所有 mark_layer 节点替换为其输入张量（mark_layer 是 identity op）。
    2. 从所有节点的 kwargs 中删除 Dynamo 注入的 layer_Rank 残留键。
    """
    graph = gm.graph

    # Pass 1：替换 mark_layer 节点
    for node in list(graph.nodes):
        if node.op == 'call_function' and 'mark_layer' in str(node.target):
            # args[0] 是输入张量，mark_layer 是恒等映射
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)

    # Pass 2：清除所有节点上残留的 layer_Rank kwarg
    for node in graph.nodes:
        if 'layer_Rank' in node.kwargs:
            node.kwargs = {k: v for k, v in node.kwargs.items() if k != 'layer_Rank'}

    graph.lint()
    gm.recompile()
    return gm

# 1. 设置重计算策略和日志级别
os.environ["RECOMPUTE_LOG_LEVEL"] = "DEBUG"

from Transformer import *
from aten_recompute.core.Recom_pass import RecomputePass
from aten_recompute.get_Aten_IR.Graph_compile_capture import GraphCapture
from aten_recompute.utils import save_ir_and_dot
from aten_recompute.core.Tag import inject_layer_tags

# ==========================================
# 初始化模型与数据 (保持你的原逻辑)
# ==========================================
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 1
d_ff = 2048 
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.to(device)

# 在捕获图之前注入层信息标签
inject_layer_tags(transformer)

src_data = torch.randint(1, src_vocab_size, (64, max_seq_length)).to(device)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length)).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
transformer.train()

# ==========================================
# 阶段一：注册捕获回调 (Compile)
# 使用 aten_recompute/get_Aten_IR/Graph_compile_capture.py 中的 GraphCapture
# ==========================================
graph_capture = GraphCapture(transformer, src_data, tgt_data[:, :-1])
compiled_transformer = graph_capture.compile()

# ==========================================
# 阶段二：预热执行 (Warm-up) -> 触发真正的图捕获
# ==========================================
print("\n[1/3] 执行预热 (Warm-up)，触发 AOT Autograd 捕获前向与反向图...")

# 前向传播 (触发 fw_compiler，捕获 FW_gm)
output = compiled_transformer(src_data, tgt_data[:, :-1])
loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))

# 反向传播 (触发 bw_compiler，捕获 BW_gm)
loss.backward()

print("图捕获完成！FW_gm 和 BW_gm 已就绪。")

# ==========================================
# 阶段三：执行重计算 Pass (图变换)
# ==========================================
print("\n[2/3] 执行重计算 Pass 优化计算图...")
dist_ir_dict = {
    "FW": graph_capture.FW_gm, 
    "BW": graph_capture.BW_gm
}

# 实例化并运行重计算 Pass
recompute_pass = RecomputePass(dist_ir_dict)
recompute_pass.run() 

# 提取优化后的图，并清除训练时不可执行的 mark_layer / layer_Rank 残留
fw_gm_opt = cleanup_for_training(recompute_pass.fw_gm)
bw_gm_opt = cleanup_for_training(recompute_pass.bw_gm)

print("重计算 Pass 运行完毕！")

# ==========================================
# 阶段四：保存优化后的图进行验证
# ==========================================
print("\n[3/3] 保存优化后的图到文件...")

model_name = os.getenv("MODEL_NAME", "Transformer")

# 保存到:
#   IR_artifacts/<model_name>/runs/<RUN_ID>/recompute/FW_recomputed.dot
#   IR_artifacts/<model_name>/runs/<RUN_ID>/recompute/BW_recomputed.dot
save_ir_and_dot(
    fw_gm_opt,
    model_name=model_name,
    subfolder="recompute",
    graph_name="FW_recomputed",
)
save_ir_and_dot(
    bw_gm_opt,
    model_name=model_name,
    subfolder="recompute",
    graph_name="BW_recomputed",
)

print("全部完成！请在 IR_artifacts 目录下查看该模型对应的重计算图。")

# ==========================================
# 阶段五：使用优化后的图进行真正的训练
# ==========================================
from aten_recompute.core.train_recomputed import run_training

run_training(
    fw_gm=fw_gm_opt,
    bw_gm=bw_gm_opt,
    model=transformer,
    src_data=src_data,
    tgt_data=tgt_data,
    tgt_vocab_size=tgt_vocab_size,
    criterion=criterion,
    graph_capture=graph_capture,
)