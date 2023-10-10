import math

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np


# 可学习相对位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        encodings = self.get_positional_encoding(d_model, max_len)
        self.register_buffer('positional_encodings', encodings, False)

    @staticmethod
    def get_positional_encoding(d_model: int, max_len: int):
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
        div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
        encodings = torch.zeros(max_len, d_model)
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        return encodings.unsqueeze(0).requires_grad_(False)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[1]].detach().requires_grad_(False)
        return self.dropout(x + pe)

import matplotlib.pyplot as plt

seq_len = 2048
model_dim = 4096
plt.figure(figsize=(15, 5))
pe = PositionalEncoding.get_positional_encoding(model_dim, seq_len)
print(pe.shape)
see_position = 2000
d = pe.matmul(pe.transpose(2, 1))
plt.plot(np.arange(seq_len), d[0, see_position, :].numpy())
plt.title("Positional encoding")
plt.show()



# rotary
# 像RoPE算是外推能力较好的位置编码，也只能外推10%到20%左右的长度而保持效果不变差，再长效果就会骤降。 - 苏剑林 https://kexue.fm/archives/9431/comment-page-1
# 这篇文章提出：没训练过的就没法保证能处理好，这是DL中很现实的现象，哪怕是Sinusoidal或RoPE这种函数式位置编码也是如此。关于第2点，可能读者会有些迷惑，Attention理论上
# 不就是可以处理任意长度的序列吗？训练和预测长度不一致影响什么呢？答案是熵，我们在《从熵不变性看Attention的Scale操作》也已经分析过这个问题，越多的token去平均注意力，
# 意味着最后的分布相对来说越“均匀”（熵更大），即注意力越分散；而训练长度短，则意味着注意力的熵更低，注意力越集中，这也是一种训练和预测的差异性，也会影响效果。
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    print(cos.shape)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed, q, k

seq_len = 2048
head_dim = 128
model_dim = 4096
n_heads = model_dim // head_dim
rotary = LlamaRotaryEmbedding(head_dim, max_position_embeddings=seq_len, base=10000, device=None)
cos, sin = rotary(key, seq_len=seq_len)
position_ids = torch.arange(seq_len).unsqueeze(0)

query_states = torch.ones(1, n_heads, seq_len,  head_dim)
key_states = torch.ones(1, n_heads, seq_len, head_dim)
query_states, key_states, q, k = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

plt.figure(figsize=(15, 5))
see_position = 2047
see_head = 0
plt.plot(np.arange(seq_len), attn_weights[0, see_head, see_position, :].numpy())

# plt.legend(f"see_position:{see_position},  see_head: {see_head}", 'medium')
plt.title("Rotary Positional encoding")
plt.show()