import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 使用sin和cos函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为buffer（不参与训练）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: 输入张量 [batch_size, seq_length, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    # 使用更稳定的缩放方法
    scaling_factor = torch.sqrt(torch.tensor(d_k, dtype=q.dtype, device=q.device))
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scaling_factor
    
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)  # 使用更安全的值代替float('-inf')
    
    # 使用更稳定的 softmax
    attn_scores_max, _ = torch.max(attn_scores, dim=-1, keepdim=True)
    exp_scores = torch.exp(attn_scores - attn_scores_max)
    if mask is not None:
        exp_scores = exp_scores * mask
    attn_probs = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + 1e-9)
    
    if dropout is not None:
        attn_probs = dropout(attn_probs)
    
    output = torch.matmul(attn_probs, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # QKV投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 添加 dropout 和 layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 初始化参数
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        residual = q  # 保存残差连接
        
        # 投影并拆分多头
        q = self.split_heads(self.q_proj(q))
        k = self.split_heads(self.k_proj(k))
        v = self.split_heads(self.v_proj(v))
        
        # 计算注意力
        attn_out = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        
        # 合并多头
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, -1, self.d_model)
        
        # 输出投影并应用dropout
        output = self.dropout(self.out_proj(attn_out))
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        return output

class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        参数:
            x: 输入张量 [batch_size, seq_length, d_model]
        """
        residual = x
        
        # 两层前馈网络
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        
        # 残差连接和层归一化
        output = self.layer_norm(x + residual)
        return output

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        """
        参数:
            x: 输入张量 [batch_size, seq_length, d_model]
            mask: 注意力掩码
        """
        # 多头自注意力
        x = self.self_attn(x, x, x, mask)
        # 前馈网络
        output = self.feed_forward(x)
        return output

class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        参数:
            x: 解码器输入 [batch_size, tgt_length, d_model]
            enc_output: 编码器输出 [batch_size, src_length, d_model]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（防止看到未来信息）
        """
        # 自注意力
        x = self.self_attn(x, x, x, tgt_mask)
        # 交叉注意力
        x = self.cross_attn(x, enc_output, enc_output, src_mask)
        # 前馈网络
        output = self.feed_forward(x)
        return output

def generate_square_subsequent_mask(size):
    """生成方形的后续掩码（用于解码器的自注意力）"""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
