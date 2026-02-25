from .Transformer import *
#from IR_transform import aten_compile_capture
from torch.fx import symbolic_trace
from torch.nn.attention import sdpa_kernel, SDPBackend


src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6 #原6  代表encoder层或decoder层中Transformer block的堆叠次数
d_ff = 2048 #FFN隐藏层维度 4*d_model
max_seq_length = 100
dropout = 0.1



transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.to(device)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))# (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
src_data = src_data.to(device)
tgt_data = tgt_data.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()
import torch._dynamo as dynamo


graph_capture = Dist_IR.GraphCapture(transformer,src_data, tgt_data[:, :-1])
transformer = graph_capture.compile()


for epoch in range(1):
    # optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    #optimizer.step()
# #保存graph Module
# import io,sys
# output1 = io.StringIO()
# with io.StringIO() as buf:
#     original_stdout1 = sys.stdout
#     sys.stdout = buf
#     graph_capture.FW_gm.print_readable()
#     sys.stdout = original_stdout1
#     output1 = buf.getvalue()
# with open( 'aten_module_FW.md', 'w') as f:
#     # 写入捕获的输出
#     f.write(output1)
# # graph_capture.FW_gm.print_readable(colored = True)
# for node in graph_capture.FW_gm.graph.nodes:
#     if node.op == 'call_function':
#         print(node.meta)

# output1 = io.StringIO()
# with io.StringIO() as buf:
#     original_stdout1 = sys.stdout
#     sys.stdout = buf
#     graph_capture.BW_gm.print_readable()
#     sys.stdout = original_stdout1
#     output1 = buf.getvalue()
# with open( 'aten_module_BW.md', 'w') as f:
#     # 写入捕获的输出
#     f.write(output1)


# 调用 pass
"""用户参考 start"""
Dist_IR.Hybrid_Parallel_pass(graph_capture.FW_gm, graph_capture.BW_gm,optim_graph_capture.OPT_gm,1000)
"""用户参考 end"""