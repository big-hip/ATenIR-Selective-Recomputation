import os
import io
import sys

# 将项目根目录加入 sys.path，保证可以找到 aten_recompute 包
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 1. 设置重计算策略和日志级别
# 策略1: 对所有候选的激活值进行重计算
os.environ["RECOMPUTE"] = '{"1": null}'          
os.environ["RECOMPUTE_LOG_LEVEL"] = "DEBUG"      

from Transformer import *
from aten_recompute.core.Recom_pass import RecomputePass
from aten_recompute.get_Aten_IR.Graph_compile_capture import GraphCapture

# ==========================================
# 初始化模型与数据 (保持你的原逻辑)
# ==========================================
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6 
d_ff = 2048 
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.to(device)

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

# 提取优化后的图
fw_gm_opt = recompute_pass.fw_gm
bw_gm_opt = recompute_pass.bw_gm

print("重计算 Pass 运行完毕！")

# ==========================================
# 阶段四：保存优化后的图进行验证
# ==========================================
print("\n[3/3] 保存优化后的图到文件...")

with open('aten_module_FW_recomputed.md', 'w') as f:
    with io.StringIO() as buf:
        original_stdout = sys.stdout
        sys.stdout = buf
        fw_gm_opt.print_readable()
        sys.stdout = original_stdout
        f.write(buf.getvalue())

with open('aten_module_BW_recomputed.md', 'w') as f:
    with io.StringIO() as buf:
        original_stdout = sys.stdout
        sys.stdout = buf
        bw_gm_opt.print_readable()
        sys.stdout = original_stdout
        f.write(buf.getvalue())

print("全部完成！请检查 aten_module_FW_recomputed.md 和 aten_module_BW_recomputed.md")