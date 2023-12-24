from transformers import AutoModelForCausalLM, AutoTokenizer


# !pip install simple_parsing
# pip install transformers==4.34


device = "cpu" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


device = "cpu" # the device to load the model onto
model_inputs = tokenizer('你好', return_tensors='pt')
model.to(device)
generated_ids = model.generate(**model_inputs, max_new_tokens=10, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])


# 测试moe模型实现
import dataclasses
from typing import List, Optional

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int
        
        
@dataclasses.dataclass
class ModelArgs(Serializable):
    dim: int
    n_layers: int
    hidden_dim: int

        
class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

    
class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        """每个token都有一个weight"""
        # inputs: (bsz * len) * dim
        # (bsz * len) * num_experts
        gate_logits = self.gate(inputs)
        # weights, selected_experts: (bsz * len) * num_experts_per_tok
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        # results: (bsz * len) * dim
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            # batch_idx shape: num_tokens; nth_expert shape: num_tokens
            batch_idx, nth_expert = torch.where(selected_experts == i)
            # weights[batch_idx, nth_expert] shape: num_tokens;  inputs[batch_idx] shape: num_tokens * dim
            # results[batch_idx] shape: num_tokens * dim
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results


n_layers = 32
bsz = 2
length = 100
dim = 2048
hidden_dim = 4096
num_experts = 8
num_experts_per_tok = 2

inputs = torch.randn(bsz, length, dim).view(-1, dim)
args = ModelArgs(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers)
moe = MoeArgs(num_experts, num_experts_per_tok)
feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(num_experts)],
                gate=nn.Linear(dim, num_experts, bias=False),
                moe_args=moe,
            )
result = feed_forward(inputs)



# Rotary 分析
import torch
from typing import Tuple

def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # abs=1, out=abs⋅cos(angle)+abs⋅sin(angle)⋅j
    # => freqs.cos() + freqs.sin()* j
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 分割为2分，分别作为实部和虚部
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    # xq_ * freqs_cis => (xq_0 + xq_1 * j) * (freqs.cos() + freqs.sin()* j)
    # = xq_0 * freqs.cos() + xq_1 * j * freqs.cos() + xq_0 * freqs.sin()* j + xq_1 * j *  freqs.sin()* j
    # = xq_0 * freqs.cos() + (xq_1 * freqs.cos() + xq_0 * freqs.sin()) * j - xq_1 * freqs.sin()
    # => (xq_0 * freqs.cos() - xq_1 * freqs.sin(), (xq_1 * freqs.cos() + xq_0 * freqs.sin()) * j)
    # => q0 * cos - q1 * sin,   q0 * sin + q1 * cos

    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # = (q0 * cos, q1 * cos) + (-q1 * sin, q0 * sin)
    # = (q0 * cos - q1 * sin, q1 * cos + q0 * sin)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# model.embed_tokens.weight torch.Size([32000, 4096])
# model.layers.0.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.0.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.0.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.0.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.0.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.0.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.0.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.0.input_layernorm.weight torch.Size([4096])
# model.layers.0.post_attention_layernorm.weight torch.Size([4096])
# model.layers.1.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.1.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.1.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.1.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.1.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.1.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.1.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.1.input_layernorm.weight torch.Size([4096])
# model.layers.1.post_attention_layernorm.weight torch.Size([4096])
# model.layers.2.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.2.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.2.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.2.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.2.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.2.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.2.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.2.input_layernorm.weight torch.Size([4096])
# model.layers.2.post_attention_layernorm.weight torch.Size([4096])
# model.layers.3.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.3.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.3.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.3.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.3.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.3.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.3.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.3.input_layernorm.weight torch.Size([4096])
# model.layers.3.post_attention_layernorm.weight torch.Size([4096])
# model.layers.4.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.4.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.4.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.4.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.4.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.4.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.4.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.4.input_layernorm.weight torch.Size([4096])
# model.layers.4.post_attention_layernorm.weight torch.Size([4096])
# model.layers.5.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.5.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.5.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.5.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.5.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.5.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.5.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.5.input_layernorm.weight torch.Size([4096])
# model.layers.5.post_attention_layernorm.weight torch.Size([4096])
# model.layers.6.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.6.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.6.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.6.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.6.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.6.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.6.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.6.input_layernorm.weight torch.Size([4096])
# model.layers.6.post_attention_layernorm.weight torch.Size([4096])
# model.layers.7.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.7.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.7.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.7.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.7.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.7.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.7.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.7.input_layernorm.weight torch.Size([4096])
# model.layers.7.post_attention_layernorm.weight torch.Size([4096])
# model.layers.8.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.8.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.8.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.8.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.8.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.8.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.8.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.8.input_layernorm.weight torch.Size([4096])
# model.layers.8.post_attention_layernorm.weight torch.Size([4096])
# model.layers.9.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.9.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.9.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.9.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.9.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.9.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.9.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.9.input_layernorm.weight torch.Size([4096])
# model.layers.9.post_attention_layernorm.weight torch.Size([4096])
# model.layers.10.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.10.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.10.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.10.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.10.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.10.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.10.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.10.input_layernorm.weight torch.Size([4096])
# model.layers.10.post_attention_layernorm.weight torch.Size([4096])
# model.layers.11.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.11.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.11.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.11.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.11.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.11.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.11.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.11.input_layernorm.weight torch.Size([4096])
# model.layers.11.post_attention_layernorm.weight torch.Size([4096])
# model.layers.12.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.12.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.12.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.12.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.12.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.12.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.12.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.12.input_layernorm.weight torch.Size([4096])
# model.layers.12.post_attention_layernorm.weight torch.Size([4096])
# model.layers.13.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.13.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.13.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.13.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.13.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.13.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.13.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.13.input_layernorm.weight torch.Size([4096])
# model.layers.13.post_attention_layernorm.weight torch.Size([4096])
# model.layers.14.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.14.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.14.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.14.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.14.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.14.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.14.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.14.input_layernorm.weight torch.Size([4096])
# model.layers.14.post_attention_layernorm.weight torch.Size([4096])
# model.layers.15.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.15.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.15.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.15.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.15.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.15.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.15.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.15.input_layernorm.weight torch.Size([4096])
# model.layers.15.post_attention_layernorm.weight torch.Size([4096])
# model.layers.16.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.16.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.16.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.16.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.16.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.16.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.16.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.16.input_layernorm.weight torch.Size([4096])
# model.layers.16.post_attention_layernorm.weight torch.Size([4096])
# model.layers.17.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.17.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.17.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.17.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.17.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.17.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.17.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.17.input_layernorm.weight torch.Size([4096])
# model.layers.17.post_attention_layernorm.weight torch.Size([4096])
# model.layers.18.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.18.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.18.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.18.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.18.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.18.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.18.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.18.input_layernorm.weight torch.Size([4096])
# model.layers.18.post_attention_layernorm.weight torch.Size([4096])
# model.layers.19.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.19.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.19.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.19.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.19.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.19.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.19.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.19.input_layernorm.weight torch.Size([4096])
# model.layers.19.post_attention_layernorm.weight torch.Size([4096])
# model.layers.20.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.20.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.20.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.20.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.20.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.20.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.20.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.20.input_layernorm.weight torch.Size([4096])
# model.layers.20.post_attention_layernorm.weight torch.Size([4096])
# model.layers.21.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.21.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.21.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.21.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.21.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.21.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.21.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.21.input_layernorm.weight torch.Size([4096])
# model.layers.21.post_attention_layernorm.weight torch.Size([4096])
# model.layers.22.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.22.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.22.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.22.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.22.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.22.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.22.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.22.input_layernorm.weight torch.Size([4096])
# model.layers.22.post_attention_layernorm.weight torch.Size([4096])
# model.layers.23.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.23.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.23.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.23.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.23.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.23.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.23.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.23.input_layernorm.weight torch.Size([4096])
# model.layers.23.post_attention_layernorm.weight torch.Size([4096])
# model.layers.24.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.24.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.24.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.24.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.24.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.24.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.24.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.24.input_layernorm.weight torch.Size([4096])
# model.layers.24.post_attention_layernorm.weight torch.Size([4096])
# model.layers.25.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.25.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.25.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.25.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.25.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.25.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.25.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.25.input_layernorm.weight torch.Size([4096])
# model.layers.25.post_attention_layernorm.weight torch.Size([4096])
# model.layers.26.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.26.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.26.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.26.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.26.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.26.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.26.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.26.input_layernorm.weight torch.Size([4096])
# model.layers.26.post_attention_layernorm.weight torch.Size([4096])
# model.layers.27.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.27.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.27.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.27.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.27.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.27.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.27.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.27.input_layernorm.weight torch.Size([4096])
# model.layers.27.post_attention_layernorm.weight torch.Size([4096])
# model.layers.28.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.28.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.28.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.28.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.28.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.28.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.28.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.28.input_layernorm.weight torch.Size([4096])
# model.layers.28.post_attention_layernorm.weight torch.Size([4096])
# model.layers.29.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.29.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.29.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.29.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.29.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.29.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.29.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.29.input_layernorm.weight torch.Size([4096])
# model.layers.29.post_attention_layernorm.weight torch.Size([4096])
# model.layers.30.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.30.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.30.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.30.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.30.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.30.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.30.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.30.input_layernorm.weight torch.Size([4096])
# model.layers.30.post_attention_layernorm.weight torch.Size([4096])
# model.layers.31.self_attn.q_proj.weight torch.Size([4096, 4096])
# model.layers.31.self_attn.k_proj.weight torch.Size([1024, 4096])
# model.layers.31.self_attn.v_proj.weight torch.Size([1024, 4096])
# model.layers.31.self_attn.o_proj.weight torch.Size([4096, 4096])
# model.layers.31.mlp.gate_proj.weight torch.Size([14336, 4096])
# model.layers.31.mlp.up_proj.weight torch.Size([14336, 4096])
# model.layers.31.mlp.down_proj.weight torch.Size([4096, 14336])
# model.layers.31.input_layernorm.weight torch.Size([4096])
# model.layers.31.post_attention_layernorm.weight torch.Size([4096])
# model.norm.weight torch.Size([4096])
# lm_head.weight torch.Size([32000, 4096])