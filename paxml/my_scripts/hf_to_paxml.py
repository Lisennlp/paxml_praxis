#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import numpy as np
import flax
import jax
import orbax
import orbax.checkpoint
import sys
import time
import os
import gc
import json

import os
import time
from pathlib import Path
import argparse


import jax
import numpy as np
import jax.numpy as jnp
import orbax
from optax import MaskedNode
from etils import epath
import torch

from praxis import base_hyperparams
from praxis import pax_fiddle
from praxis import py_utils
from paxml import checkpoints  # mapped to internal
from paxml import checkpoint_managers
from paxml import train_states
from paxml import trainer_lib
from flax.traverse_util import flatten_dict, unflatten_dict

os.environ["JAX_PLATFORMS"] = "cpu"

TrainState = train_states.TrainState
CheckpointType = checkpoints.CheckpointType
Checkpointer = checkpoints.Checkpointer
PaxCheckpointHandler = checkpoints.PaxCheckpointHandler
NestedMap = py_utils.NestedMap
checkpoint_type = CheckpointType.GDA
SAVE_INTERVAL_STEPS = 1



LLAMA_STANDARD_CONFIGS = {
    '7b': {
        'dim': 4096,
        'intermediate_size': 11008,
        'n_layers': 32,
        'n_heads': 32,
        'norm_eps': 1e-6,
    },
    '13b': {
        'dim': 5120,
        'intermediate_size': 13824,
        'n_layers': 40,
        'n_heads': 40,
        'norm_eps': 1e-6,
    },
    '30b': {
        'dim': 6656,
        'intermediate_size': 17920,
        'n_layers': 60,
        'n_heads': 52,
        'norm_eps': 1e-6,
    },
    '65b': {
        'dim': 8192,
        'intermediate_size': 22016,
        'n_layers': 80,
        'n_heads': 64,
        'norm_eps': 1e-5,
    },
}
params = LLAMA_STANDARD_CONFIGS['7b']
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]

read_dir = '/home/lishengping/baichuan-7b-hf'
ckpt_paths = sorted(Path(read_dir).glob("*.bin"))
ckpt = {}
for i, ckpt_path in enumerate(ckpt_paths):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    for k, v in checkpoint.items():
        if k.startswith('model.'):
            k = k[6:]
        ckpt[k] = v


save_dir = 'gs://jax_llm_logs/alsp_debug/0822/hf_to_paxml_Oproj_NoTranspose/checkpoints'
options = checkpoint_managers.CheckpointManagerOptions(
      max_to_keep=10,
      save_interval_steps=SAVE_INTERVAL_STEPS,
      cleanup_tmp_directories=True,
  )
checkpointer = Checkpointer(
          PaxCheckpointHandler(
              enforce_restore_shape_check=False,
              use_ocdbt=False,
          )
      )
save_dir = epath.Path(save_dir)
checkpoint_manager = checkpoint_managers.OrbaxCheckpointManager(
      save_dir,
      checkpointer,
      train_input_checkpointer=False,
      options=options,
      checkpoint_type=checkpoint_type,
      tensorstore_use_ocdbt=False,
  )


model_size = '7b'
vocab_size = 64000
intermediate_size = params['intermediate_size']
x_times = 32
n_heads = params['n_heads']
head_dim = dim // n_heads
n_layers = 32
step = 40

paxml_to_hf_key_and_shape = {
 'params.lm.embedding_lookup.emb_var': {'shape': (vocab_size, dim), 'map_to_hf': 'embed_tokens'},
 'params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w': {'shape': (dim, intermediate_size), 'map_to_hf': 'up_proj'},
 'params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1_gate.linear.w': {'shape': (dim, intermediate_size), 'map_to_hf': 'gate_proj'},
 'params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w': {'shape': (intermediate_size, dim), 'map_to_hf': 'down_proj'},
 'params.lm.transformer.repeat.sub.x_layers_0.self_attention.query.w': {'shape': (dim,n_heads,head_dim), 'map_to_hf': 'q_proj'},
 'params.lm.transformer.repeat.sub.x_layers_0.self_attention.key.w': {'shape': (dim, n_heads, head_dim), 'map_to_hf': 'k_proj'},
 'params.lm.transformer.repeat.sub.x_layers_0.self_attention.value.w': {'shape': (dim, n_heads, head_dim), 'map_to_hf': 'v_proj'},
 'params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w': {'shape': (dim, n_heads, head_dim), 'map_to_hf': 'o_proj'},
 'params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale': {'shape': (dim,), 'map_to_hf': 'input_layernorm'},
 'params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale': {'shape': (dim,), 'map_to_hf': 'post_attention_layernorm'},
 'params.lm.final_ln.scale': {'shape': (dim,), 'map_to_hf': 'norm.weight'},
 'params.lm.softmax.logits_ffn.linear.w': {'shape': (dim, vocab_size), 'map_to_hf': 'lm_head'}
}

hf_to_paxml_format = {v['map_to_hf']: k for k, v in paxml_to_hf_key_and_shape.items()}
padded_global_shapes = {}
for k, v in paxml_to_hf_key_and_shape.items():
    k = tuple(k.split('.'))
    if 'repeat' in k:
        padded_global_shapes[k] = jax.ShapeDtypeStruct(shape=(x_times, ) + v['shape'], dtype=jnp.float16)
    else:
        padded_global_shapes[k] = jax.ShapeDtypeStruct(shape=v['shape'], dtype=jnp.float16)
        
padded_global_shapes = TrainState(step=jnp.array(step), mdl_vars=unflatten_dict(padded_global_shapes), opt_states=None)
print(f'Padded_global_shapes bulid finished!!!')

gold_w = ckpt


split_qkv = {}
for k, v in gold_w.items():
    v = v.to(torch.float16)
    # o_proj不进行transpose，是个坑
    if len(v.shape) == 2 and 'embed_tokens' not in k and 'o_proj' not in k:
        v = v.transpose(1, 0)
    else:
        print(f'No transpose k: {k}')
    if 'W_pack' in k:
        qq = k.replace('W_pack', 'q_proj')
        kk = k.replace('W_pack', 'k_proj')
        vv = k.replace('W_pack', 'v_proj')
        split_qkv[qq] = v[:, :dim].numpy()
        split_qkv[kk] = v[:, dim:dim*2].numpy()
        split_qkv[vv] = v[:, -dim:].numpy()
    else:
        split_qkv[k] = v.numpy()
        
for k, v in split_qkv.items():
    print(k, v.shape)


trans_result = {}
flag = 0
with jax.default_device(jax.devices("cpu")[0]):
    for k, v in paxml_to_hf_key_and_shape.items():
        v = v['map_to_hf']
        k = tuple(k.split('.'))
        values = []
        for gold_key, glod_values in split_qkv.items():
            flag = 0
            if (v in gold_key and v != 'norm.weight') or v == gold_key == 'norm.weight':
                flag = 1
                if v in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    glod_values = glod_values.reshape(dim, n_heads, head_dim)
                try:
                    layer_index = int(re.findall('\d+', gold_key)[0])
                except:
                    layer_index = 0
                values.append([layer_index, glod_values])
        values = sorted(values, key=lambda x: x[0])
        if len(values) > 1:
            stack_values = np.stack(list(zip(*values))[1])
        else:
            stack_values = values[0][1]
        trans_result[k] = stack_values
    opt_state_mv = jax.tree_map(lambda x: jnp.zeros_like(x), trans_result)

step =40
print(f'Please simple check model shape and dtype...')
for k, v in trans_result.items():
    print(k, v.shape, v.dtype)

if step is None:
    latest_step =  checkpoint_manager.latest_step()
    if save_dir == read_dir:
        step = latest_step + SAVE_INTERVAL_STEPS if latest_step is not None else SAVE_INTERVAL_STEPS
    else:
        step = latest_step

print(f'Model save step is {step}')
start = time.time()
temp_no_prefix, temp_other = {}, {}
for key_tuple, param in opt_state_mv.items():
    if 'repeat' in key_tuple:
        temp_no_prefix[key_tuple] = MaskedNode()
        temp_other[key_tuple] = param
    else:
        temp_no_prefix[key_tuple] = param
        temp_other[key_tuple] = MaskedNode()

temp_no_prefix = unflatten_dict(temp_no_prefix)
temp_other = unflatten_dict(temp_other)
    
no_prefix = {'count': jnp.array(step), 'm': temp_no_prefix, 'v': temp_no_prefix}
other = {'count': jnp.array([step] * n_layers), 'm': temp_other, 'v': temp_other}
trans_opt_states = {
    'no_prefix': [{'count': jnp.array(step)}] * 2 + [no_prefix, {'count': jnp.array(step)}], 
    f'p#{n_layers}#i-1': [{'count': jnp.array([step] * n_layers)}] * 2 + [other, {'count': jnp.array([step] * n_layers)}], 
}
trans_opt_states = [trans_opt_states]
new_trainstate = TrainState(
                            step=jnp.array(step), 
                            mdl_vars=unflatten_dict(trans_result),
                            opt_states=trans_opt_states
)
padded_global_shapes = jax.tree_map(lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype) 
                                    if hasattr(x, 'shape') else x , new_trainstate)
checkpoint_manager.save(step, new_trainstate, padded_global_shapes, train_input_pipeline=None, force=False)
print(f'Saved model finished. take time: {time.time() - start}s...')



