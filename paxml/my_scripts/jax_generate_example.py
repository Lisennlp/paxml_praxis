import os
import time
import argparse
import socket
from typing import Dict
from functools import partial

# os.environ["JAX_PLATFORMS"] = "cpu"

import torch
import numpy as np
import flax
import orbax
import orbax.checkpoint
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from jax.sharding import Mesh



class MyModel(nn.Module):
    vocab: int = 500
    embed_dim: int = 128
    hidden_dim: int = 128
    def setup(self):
        kernel_init_shard = nn.with_logical_partitioning(jax.nn.initializers.normal(0.004), ('mp', 'fsdp'))
        self.embedding = self.param('embedding', kernel_init_shard, (self.vocab, self.embed_dim), jnp.bfloat16)

        kernel_init_shard = nn.with_logical_partitioning(jax.nn.initializers.normal(0.004), ('fsdp', 'mp'))
        self.pre_proj = self.param('pre_proj', kernel_init_shard, (self.embed_dim, self.hidden_dim), jnp.bfloat16)
        self.activation = nn.relu
        
        kernel_init_shard = nn.with_logical_partitioning(jax.nn.initializers.normal(0.004), ('fsdp', 'mp'))
        self.post_proj = self.param('post_proj', kernel_init_shard, (self.hidden_dim, self.embed_dim), jnp.bfloat16)

    @nn.compact
    def update_kv_cache(self, query, key):
        # 在用fake数据进行第一次前传编译的时候，初始化cache为0
        is_initialized = self.has_variable("cache", "cached_key")
        is_index = self.has_variable("cache", "cache_index")
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))
        if is_initialized:
            cur_index = cache_index.value
            indices =  (0, cur_index, 0)
            cached_key.value = jax.lax.dynamic_update_slice(cached_key.value, key, indices)
            cache_index.value += query.shape[1]

    def __call__(self, input_ids, mode='train', init_cache=False):
        key = self.embedding[input_ids]
        query = key
        if self.has_variable("cache", "cached_key") or init_cache:
            print('has cache')
            self.update_kv_cache(query, key)
            
        x = jnp.einsum('ble,eh->blh', key, self.pre_proj)
        x = self.activation(x)
        x = jnp.einsum('blh,he->blh',x, self.post_proj)
        return x
    

@flax.struct.dataclass
class GreedyState:
    cur_len: jnp.ndarray
    sequences: jnp.ndarray
    running_token: jnp.ndarray
    is_sent_finished: jnp.ndarray
    params: Dict[str, jnp.ndarray]


def prepare_params_for_generation(input_ids, max_length, attention_mask = None):
    batch_size, seq_length = input_ids.shape
    pngkey = jax.random.key(0)
    fake_ids = jax.random.randint(pngkey, minval=1, maxval=vocab, shape=[batch_size, max_length])
    params = model.init(pngkey, fake_ids, init_cache=True)
    return params


batch_size = 2
max_length = 100
vocab = 500
embed_dim = 128
hidden_dim = 256
seed = 0
pngkey = jax.random.key(seed)
model = MyModel(vocab, embed_dim, hidden_dim)

input_len = 10
input_ids = jax.random.randint(pngkey, minval=1, maxval=vocab, shape=[batch_size, input_len])
params = prepare_params_for_generation(input_ids, max_length)

dims = [1, 8, 1]
dim_names = ['dp', 'fsdp', 'mp']
mesh = Mesh(np.array(jax.devices()).reshape(dims), dim_names)
# 类似'post_proj': LogicallyPartitioned(value=ShapeDtypeStruct(shape=(256, 128), dtype=bfloat16), names=('fsdp', 'mp'), mesh=No
abstract_state = jax.eval_shape(lambda x: x, params)
# 根据名字获取spec文本, 类似'post_proj': PartitionSpec('fsdp', 'mp'),
state_logical_annotations = nn.get_partition_spec(abstract_state) # to PartitionSpec
# 给每个参数添加mesh信息, post_proj': NamedSharding(mesh=Mesh('dp': 1, 'fsdp': 8, 'mp': 1), spec=PartitionSpec('fsdp', 'mp')),
params_shard = jax.tree_map(lambda x: jax.sharding.NamedSharding(mesh, x), state_logical_annotations)


@partial(
    pjit,
    in_shardings=(PS(), params_shard),
    out_shardings=(PS())
)
def generate(input_ids, params):
    
    pad_token_id = 0
    eos_token_id = 499
    batch_size, cur_len = input_ids.shape
    eos_token_id = jnp.array(eos_token_id, dtype=jnp.int32 if eos_token_id is not None else None)
    pad_token_id = jnp.array(pad_token_id, dtype=jnp.int32)
    cur_len = jnp.array(cur_len)
    is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
    
    sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
    sequences = jax.lax.dynamic_update_slice(sequences, input_ids, (0, 0))
    state = GreedyState(
        cur_len=cur_len,
        sequences=sequences,
        running_token=input_ids,
        is_sent_finished=is_sent_finished,
        params=params,
    )
    
    def greedy_search_cond_fn(state):
        has_reached_max_length = state.cur_len == max_length
        all_sequence_finished = jnp.all(state.is_sent_finished)
        # 所有句子遇到eos或者达到最大生成长度则结束
        finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
        return ~finish_generation
        
    def greedy_search_body_fn(state):
        params = state.params
        logits, cache = model.apply(params, state.running_token, mutable=['cache'])
        params['cache'] = cache['cache']
        logits = logits[:, -1]
        next_token = jnp.argmax(logits, axis=-1)
        next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
        next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
        next_token = next_token[:, None]
        next_sequences = jax.lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
        return GreedyState(
            cur_len=state.cur_len + 1,
            sequences=next_sequences,
            running_token=next_token,
            is_sent_finished=next_is_sent_finished,
            params=params,
        )
    
    if input_ids.shape[1] > 1:
        state = greedy_search_body_fn(state)
    state = jax.lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)
    sequences=state.sequences
    return sequences



input_len = 20
input_ids = jax.random.randint(pngkey, minval=1, maxval=vocab, shape=[batch_size, input_len])

with mesh:
    output = generate(input_ids, params)
    output = jax.device_get(output)