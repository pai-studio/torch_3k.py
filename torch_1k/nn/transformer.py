import math
from .module import Module
from .linear import Linear
from .activation import ReLU
from .normalization import LayerNorm
from torch_1k import functional as F


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, batch_first=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def _split_heads(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key=None, value=None, need_weights=False):
        if key is None:
            key = query
        if value is None:
            value = key
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        scores = F.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, axis=-1)
        context = F.matmul(attn, v)
        b, _, t, _ = context.shape
        out = context.transpose(1, 2).reshape(b, t, self.embed_dim)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)
        if need_weights:
            return out, attn.mean(axis=1)
        return out


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, batch_first=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.activation = ReLU()

    def forward(self, src):
        attn = self.self_attn(src)
        src = self.norm1(src + attn)
        ff = self.linear2(self.activation(self.linear1(src)))
        return self.norm2(src + ff)
