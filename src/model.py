import torch
import torch.nn as nn
from components import MultiHeadAttention, scaled_dot_product_attention 

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        # self.pe: [max_seq_len, d_model] -> [1, max_seq_len, d_model]
        # 取前seq_len步，并广播到batch
        return x + self.pe[:seq_len, :].transpose(0, 1)

# Encoder块
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 自注意力
        attn_out = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        
        # 前馈网络
        ff_out = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)
        return src

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 自注意力
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 交叉注意力
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 前馈网络
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, vocab_size=5000, 
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # 词嵌入和位置编码
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Encoder和Decoder层
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        # 准备mask（简化版本）
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal=1).bool()
        tgt_mask = tgt_mask.to(src.device)
        
        # 词嵌入和位置编码
        src = self.dropout(self.pos_encoder(self.embedding(src) * torch.sqrt(torch.tensor(self.d_model))))
        tgt = self.dropout(self.pos_encoder(self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model))))
        
        # Encoder
        enc_output = src
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)
        
        # Decoder
        dec_output = tgt
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask)
        
        # 输出层
        output = self.output_layer(dec_output)
        return output
