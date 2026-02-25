import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

import torch._dynamo
from torch.fx.passes.graph_drawer import FxGraphDrawer
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified
device = torch.device("cuda") if torch.cuda.is_available() else "cpu" 
# print(f"正在使用的设备是: {device}")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        #只能均分多头
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    
    #在逻辑维度将 d_model:512 拆为num_head*head_dim 8*64 但是没有将8个head分到不同GPU上
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

#MLP
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        #改为使用pytorch框架MHA
        self.self_attn = nn.MultiheadAttention(d_model,num_heads,dropout=dropout,batch_first = True)
        #self.self_attn = CustomMHAFromGraph(d_model, num_heads, dropout=dropout)
        #在encoder中使用定义的多头
        #self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        #attn_output = self.self_attn(x, x, x, mask)
        if mask is not None:
            # 将 mask 从 (B, 1, 1, S) -> (B, S)
            key_padding_mask = mask.squeeze(1).squeeze(1)
        else:
            key_padding_mask = None

        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff, dropout):
#         super(DecoderLayer, self).__init__()
#         self.self_attn = MultiHeadAttention(d_model, num_heads)
#         self.cross_attn = MultiHeadAttention(d_model, num_heads)
#         self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, enc_output, src_mask, tgt_mask):
#         attn_output = self.self_attn(x, x, x, tgt_mask)
#         x = self.norm1(x + self.dropout(attn_output))
#         attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
#         x = self.norm2(x + self.dropout(attn_output))
#         ff_output = self.feed_forward(x)
#         x = self.norm3(x + self.dropout(ff_output))
#         return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        
        # 1. (与Encoder类似) 将自定义MHA替换为官方标准模块
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    # 2. 修改 forward 方法以接受和使用适配新模块的掩码
    # 注意：这里的掩码参数是适配 nn.MultiheadAttention 的2D掩码
    def forward(self, x, enc_output, causal_mask, src_padding_mask, tgt_padding_mask):
        
        # --- Decoder Self-Attention ---
        # 自注意力需要两种掩码：
        #   attn_mask: 因果掩码，防止看到未来的词元 (形状 [T, T])
        #   key_padding_mask: 目标序列自身的填充掩码 (形状 [B, T])
        attn_output, _ = self.self_attn(
            x, x, x, 
            attn_mask=causal_mask,
            key_padding_mask=tgt_padding_mask,
            need_weights=False  # (与Encoder类似) 如果不需要权重图，可以设为False以提高效率
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # --- Decoder Cross-Attention ---
        # 交叉注意力的 Query 来自解码器，Key 和 Value 来自编码器。
        # 它只需要一种掩码：
        #   key_padding_mask: 源序列(编码器输出)的填充掩码 (形状 [B, S])
        attn_output, _ = self.cross_attn(
            x, enc_output, enc_output,
            key_padding_mask=src_padding_mask,
            need_weights=False
        )
        x = self.norm2(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,padding_idx = 0):
        super(Transformer, self).__init__()
        # 将 padding_idx 保存为模型属性
        self.padding_idx = padding_idx
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding1 = PositionalEncoding(d_model, max_seq_length)
        self.positional_encoding2 = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.fc.weight = self.decoder_embedding.weight #decoder输入embedding 和输出 linear 层共享一个词表权重（weight tying）
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.stack = torch.stack

    def generate_mask(self, src, tgt): #生成掩码
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        nopeak_mask = nopeak_mask.to(device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def create_masks(self, src, tgt):
        device = src.device
        
        # --- 修改这里：使用 torch.eq() 替代 == ---
        # 原来的代码: src_padding_mask = (src == self.padding_idx)
        src_padding_mask = torch.eq(src, self.padding_idx)
        
        # 原来的代码: tgt_padding_mask = (tgt == self.padding_idx)
        tgt_padding_mask = torch.eq(tgt, self.padding_idx)

        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
        
        return src_padding_mask, tgt_padding_mask, causal_mask

    def forward(self, src, tgt):
        #生成掩码
        #src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_padding_mask, tgt_padding_mask, causal_mask = self.create_masks(src, tgt)
        #转为词向量+位置编码+dropout防止过拟合
        src_embedded = self.dropout1(self.positional_encoding1(self.encoder_embedding(src)))
        tgt_embedded = self.dropout2(self.positional_encoding2(self.decoder_embedding(tgt)))

        #将源序列的嵌入表示传入一层一层的编码器层中，每层都会使用src_mask来做注意力掩码
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_padding_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, causal_mask, src_padding_mask, tgt_padding_mask)

        #将output embedding预测的下一个词的词向量，去字典中查找对应的词后进行输出
        output = self.fc(dec_output)
        return output