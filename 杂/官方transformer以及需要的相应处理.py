import torch
import torch.nn as nn
import math

class MyNativeTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512):
        super().__init__()
        # 1. 自己补充 Embedding 层
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        
        # 2. 调用原生 Transformer 骨架
        self.transformer = nn.Transformer(
            d_model=d_model, 
            batch_first=True
        )
        
        # 3. 输出分类头
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def forward(self, src_indices, tgt_indices):
        # src_indices/tgt_indices 是你之前的 torch.randint 生成的整数 (Batch, Seq)
        
        # 转换为连续向量并放大 (常规 Transformer 技巧)
        src = self.src_emb(src_indices) * math.sqrt(self.d_model)
        tgt = self.tgt_emb(tgt_indices) * math.sqrt(self.d_model)
        
        # 生成防作弊掩码
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        
        # 核心前向计算
        outs = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        # 映射回词表概率分布
        return self.fc_out(outs)

# --- 运行验证 ---
model = MyNativeTransformer(src_vocab_size=10000, tgt_vocab_size=10000)
src_data = torch.randint(1, 10000, (32, 15)) # Batch:32, Src_len:15
tgt_data = torch.randint(1, 10000, (32, 20)) # Batch:32, Tgt_len:20

logits = model(src_data, tgt_data)
print(f"原生 Transformer 包装完毕！最终输出维度: {logits.shape}") 
# 预期输出: [32, 20, 10000] (Batch, Tgt_len, Vocab_size)