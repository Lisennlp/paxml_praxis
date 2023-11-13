#!/usr/bin/env python
# coding: utf-8
import sys
import time
import os
import gc
import json
import argparse
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import flax
import numpy as np
import jax.numpy as jnp
import orbax
import orbax.checkpoint
from optax import MaskedNode
from etils import epath
from praxis import base_hyperparams
from praxis import pax_fiddle
from praxis import py_utils
from paxml import checkpoints
from paxml import checkpoint_managers
from paxml import train_states
from paxml import trainer_lib
from flax.traverse_util import flatten_dict, unflatten_dict

try:
    import torch
except:
    command = (
        "pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1 --index-url"
        " https://download.pytorch.org/whl/cpu"
    )
    subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    import torch



def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )


def apply_rotary_pos_emb_torch(
    q, k, cos, sin, offset: int = 0
):
    cos, sin = (
        cos[offset : q.shape[0] + offset, ...],
        sin[offset : q.shape[0] + offset, ...],
    )
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        print(f'x: {x.shape}')
        print(f'seq_dim: {seq_dim}')
        
        print(f'seq_len: {seq_len}')
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            print(f't: {t.shape}')
            print(f'self.inv_freq: {self.inv_freq.shape}')
            
            
            print(f'freqs: {freqs.shape}')
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            print(f'emb: {emb.shape}')
            
            if self.precision == torch.bfloat16:
                emb = emb.float()
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached, self.sin_cached
    
    
seq_len = 2048
num_heads = 16
rotary_pct = 0.25
dim = int(1024 // 16 * rotary_pct)
params_dtype = torch.bfloat16
rotary_emb = RotaryEmbedding(
                dim, base=10000, precision=params_dtype
            )
rotary_ndims = dim


query_layer = x.reshape(2, seq_len, num_heads, -1)
key_layer,  value_layer = query_layer, query_layer

query_rot, query_pass = (
    query_layer[..., :rotary_ndims],
    query_layer[...,rotary_ndims :],
)
key_rot, key_pass = (
    key_layer[..., :rotary_ndims],
    key_layer[...,rotary_ndims :],
)
apply_rotary_fn = apply_rotary_pos_emb_torch

seq_len = key_layer.shape[1]
offset = 0

cos, sin =rotary_emb(value_layer, seq_len=seq_len)


query_layer, key_layer = apply_rotary_fn(
    query_rot, key_rot, cos, sin, offset=offset
)

query_layer = torch.cat((query_layer, query_pass), dim=-1)
key_layer = torch.cat((key_layer, key_pass), dim=-1)


from typing import Optional, Union
from praxis import pytypes
JTensor = pytypes.JTensor


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return jnp.concatenate(
        (-x2, x1), axis=x1.ndim - 1
    )


def apply_rotary_pos_emb(inputs, cos, sin, offset: int = 0):
    cos, sin = (
        cos[offset : inputs.shape[0] + offset, ...],
        sin[offset : inputs.shape[0] + offset, ...],
    )
    return (inputs * cos) + (rotate_half(inputs) * sin)


class RotaryPositionalEmbedding():
    cast_as_fprop_dtype: bool = True
    rotary_pct: float = 1

    def setup(self) -> None:
        if self.embedding_dims % 2:
            raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")
        self.freqs_cis = precompute_freqs_cis(self.embedding_dims, 2048)  # XD
        super().setup()
        
    def __call__(
        self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
        inputs: JTensor,
        position: Optional[JTensor] = None,
        offset: int = 0
    ) -> JTensor:
        if len(inputs.shape) != 4:
            raise ValueError(
                "Input is assumed to be a rank 4 tensor of shape[batch, sequence, heads, dims]."
            )
        if self.embedding_dims != inputs.shape[3]:
            raise ValueError(
                "The embedding dims of the rotary position embedding"
                "must match the hidden dimension of the inputs."
            )
        
        rotary_ndims = int(self.embedding_dims * self.rotary_pct)
        fraction = jnp.arange(0, rotary_ndims, 2).astype(jnp.float32) / rotary_ndims
        timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        
        if position is None:
            seq_length = inputs.shape[1]
            # 调换了length维度顺序
            position = jnp.arange(seq_length, dtype=jnp.float32)[:, jnp.newaxis]
        else:
            if len(position.shape) == 2:
                assert position.shape[1] == inputs.shape[1]
                position = position.permute(1, 0)
            elif len(position.shape) == 1:
                position = position[:, jnp.newaxis]
            else:
                raise ValueError(f'Rotary position shape is error!!!!')
        position = position[:, :, jnp.newaxis, jnp.newaxis]
        timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
        sinusoid_inp = position / timescale
        sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
        
        if self.fprop_dtype == jnp.bfloat16:
            sinusoid_inp = sinusoid_inp.astype(jnp.float32)
            
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)
        
        inputs_rot, inputs_pass = (inputs[..., :rotary_ndims], inputs[..., rotary_ndims :], )
        
        inputs_layer = apply_rotary_pos_emb(inputs_rot, cos, sin, offset=offset)
        inputs_layer = jnp.concatenate((inputs_layer, inputs_pass), axis=-1)

        if self.cast_as_fprop_dtype:
            inputs_layer = inputs_layer.astype(self.fprop_dtype)
            
        return inputs_layer
    
rotary_jax = RotaryPositionalEmbedding()
rotary_jax.rotary_pct = 0.25
rotary_jax.embedding_dims = 64
rotary_jax.min_timescale = 1.0
rotary_jax.max_timescale = 10000
rotary_jax.fprop_dtype = jnp.bfloat16
res_jax = rotary_jax(x.numpy())
