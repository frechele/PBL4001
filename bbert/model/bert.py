from bbert.data.instruction import Vocabulary

import torch
import torch.nn as nn

import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=64):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad_(False)

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class BERTEmbedding(nn.Module):
    def __init__(self, vmap: Vocabulary, embed_size: int, dropout: float=0.1):
        super(BERTEmbedding, self).__init__()

        self.token = nn.Embedding(num_embeddings=vmap.size, embedding_dim=embed_size, padding_idx=vmap.get_index('[PAD]'))
        self.position = PositionalEncoding(embed_size)

        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token(x) + self.position(x)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float=1e-6):
        super(LayerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout:float =0.1):
        super(PositionwiseFeedForward, self).__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = torch.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float):
        super(MultiHeadAttention, self).__init__()

        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, attn_heads: int, feed_forward_hidden: int, dropout: float):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)

        return self.dropout(x)


class BERT(nn.Module):
    def __init__(self, vmap: Vocabulary, hidden: int=768, n_layers: int=12, attn_heads: int=12, dropout: float=0.1):
        super(BERT, self).__init__()

        self.hidden = hidden

        self.pad_id = vmap.get_index('[PAD]')

        self.embedding = BERTEmbedding(vmap, embed_size=hidden)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, attn_heads, hidden * 4, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x != self.pad_id).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x)

        for block in self.transformer_blocks:
            x = block(x, mask)

        return x
