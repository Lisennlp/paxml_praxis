import json
import random
import os
import time

os.environ["JAX_PLATFORMS"] = "cpu"

from etils import epath
import tensorflow as tf


path = 'gs://common_datasets/pythia_model_test/flan_test/flan_mini_filtered_v2.jsonl'
path = epath.Path(path)

lines = []
with path.open('r') as f:
    for line in f:
        line = json.loads(line)
        lines.append(line)
        

random.seed(1234)
random.shuffle(lines)

def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


start = time.time()
wp = 'gs://common_datasets/pythia_model_test/flan_test/flan_mini_filtered_v2_len2049.tfrecord'

N = len(lines)
max_len = 2049
with tf.io.TFRecordWriter(wp) as writer:
     for index, line in enumerate(lines):
        example = line
        labels = [-100] +  example['labels'][:-1]
        if len(labels) > max_len:
            print(f'exceed max length: {len(labels)}, index: {index}')
            continue
        assert len(line['input_ids']) == len(labels)
        
        if index % 100 == 0:
            print(f'processed: {index}/{N} take: {time.time() - start}s')
        feature = {
            "input_ids": _int64_feature(example['input_ids']),
            "labels": _int64_feature(labels),
            
                  }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

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

TrainState = train_states.TrainState
CheckpointType = checkpoints.CheckpointType
Checkpointer = checkpoints.Checkpointer
PaxCheckpointHandler = checkpoints.PaxCheckpointHandler
NestedMap = py_utils.NestedMap
checkpoint_type = CheckpointType.GDA
SAVE_INTERVAL_STEPS = 1

LLAMA_STANDARD_CONFIGS = {
    "7b": {
        "dim": 4096,
        "intermediate_size": 22016,
        "n_layers": 32,
        "n_heads": 32,
        "norm_eps": 1e-6,
        "vocab_size": 151936
    },
}
model_size = '7b'
params = LLAMA_STANDARD_CONFIGS[model_size]
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
intermediate_size = params["intermediate_size"]
head_dim = dim // n_heads

ckpt = {}
for k, v in model.named_parameters():
    ckpt[k] = v
assert len(ckpt) > 0, print(f"ckpt is empty, please model path whether right or error.....")


save_dir = f'gs://llm_base_models/qwen/{model_size}/paxml/checkpoints/'
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

for k, v in model.named_parameters():
    print(k, v.shape)

step = 0

model_size = "7b"

params = LLAMA_STANDARD_CONFIGS[model_size]
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
vocab_size = params['vocab_size']
intermediate_size = params["intermediate_size"]
head_dim = dim // n_heads


paxml_to_hf_key_and_shape = {
    "params.lm.embedding_lookup.emb_var": {
        "shape": (vocab_size, dim),
        "map_to_hf": "wte.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w": {
        "shape": (dim, intermediate_size),
        "map_to_hf": "w1.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1_gate.linear.w": {
        "shape": (dim, intermediate_size),
        "map_to_hf": "w2.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w": {
        "shape": (intermediate_size, dim),
        "map_to_hf": "mlp.c_proj.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.self_attention.query.w": {
        "shape": (dim, n_heads, head_dim),
        "map_to_hf": "q_proj.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.self_attention.query.b": {
        "shape": (dim, n_heads, head_dim),
        "map_to_hf": "q_proj.bias",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.self_attention.key.w": {
        "shape": (dim, n_heads, head_dim),
        "map_to_hf": "k_proj.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.self_attention.key.b": {
        "shape": (dim, n_heads, head_dim),
        "map_to_hf": "k_proj.bias",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.self_attention.value.w": {
        "shape": (dim, n_heads, head_dim),
        "map_to_hf": "v_proj.weight",
    },
     "params.lm.transformer.repeat.sub.x_layers_0.self_attention.value.b": {
        "shape": (dim, n_heads, head_dim),
        "map_to_hf": "v_proj.bias",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w": {
        "shape": (dim, n_heads, head_dim),
        "map_to_hf": "attn.c_proj.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale": {
        "shape": (dim,),
        "map_to_hf": "ln_1.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale": {
        "shape": (dim,),
        "map_to_hf": "ln_2.weight",
    },
    "params.lm.final_ln.scale": {"shape": (dim,), "map_to_hf": "ln_f.weight"},
    "params.lm.softmax.logits_ffn.linear.w": {
        "shape": (dim, vocab_size),
        "map_to_hf": "lm_head",
    },
}

gold_w = ckpt
split_qkv = {}
for k, v in gold_w.items():
    if v.dtype == torch.float32:
        pass
    else:
        v = v.to(torch.float32)
    # o_proj不进行transpose，是个坑
    if len(v.shape) == 2 and "wte.weight" not in k and "attn.c_proj.weight" not in k:
        v = v.transpose(1, 0)
    else:
        print(f"No transpose k: {k}")
        
    if "c_attn" in k:
        qq = k.replace("c_attn", "q_proj")
        kk = k.replace("c_attn", "k_proj")
        vv = k.replace("c_attn", "v_proj")
        print(f'v.shape')
        if len(v.shape) == 1:
#             v = v.reshape(n_heads, 3 * head_dim)
            split_qkv[qq] = v[..., :dim].detach().numpy().reshape(-1)
            split_qkv[kk] = v[..., dim: 2 * dim].detach().numpy().reshape(-1)
            split_qkv[vv] = v[..., 2 * dim: ].detach().numpy().reshape(-1)
        elif len(v.shape) == 2:
#             v = v.reshape(dim, n_heads, 3 * head_dim)
            split_qkv[qq] = v[..., :dim].detach().numpy().reshape(dim, -1)
            split_qkv[kk] = v[..., dim: 2 * dim].detach().numpy().reshape(dim, -1)
            split_qkv[vv] = v[..., 2 * dim: ].detach().numpy().reshape(dim, -1)
        else:
            raise ValueError(f'qkv shape is error!!!')
    else:
        split_qkv[k] = v.detach().numpy()

for k, v in split_qkv.items():
    print(k, v.shape)

import re


trans_result = {}
flag = 0
a = 0
with jax.default_device(jax.devices("cpu")[0]):
    for k, v in paxml_to_hf_key_and_shape.items():
        v = v["map_to_hf"]
        k = tuple(k.split("."))
        values = []
        for gold_key, glod_values in split_qkv.items():
            flag = 0
            if v in gold_key:
#                 print(v, gold_key, "====")
                flag = 1
                match_res = re.findall("q_proj|k_proj|v_proj|attn.c_proj", v)
                if match_res:
                    if len(glod_values.shape) > 1:
                        glod_values = glod_values.reshape(dim, n_heads, head_dim)
                    else:
#                         if "attn.c_proj.bias" not in v:
                        glod_values = glod_values.reshape(n_heads, head_dim)
                try:
                    layer_index = int(re.findall("\d+", gold_key)[0])
                except:
                    layer_index = 0
                values.append([layer_index, glod_values])
                print(f"match_res: {match_res}|| {len(values)}")
                
        values = sorted(values, key=lambda x: x[0])
        if len(values) > 1:
            stack_values = np.stack(list(zip(*values))[1])
        else:
            stack_values = values[0][1]
        trans_result[k] = stack_values
        
print(f"Please simple check model shape and dtype...")
for k, v in trans_result.items():
    k = '.'.join(k)
    print(k, v.shape, v.dtype)

if step is None:
    latest_step = checkpoint_manager.latest_step()
    if save_dir == read_dir:
        step = latest_step + SAVE_INTERVAL_STEPS if latest_step is not None else SAVE_INTERVAL_STEPS
    else:
        step = latest_step

print(f"Model save step is {step}")
start = time.time()

save_opt = True
if save_opt:
    with jax.default_device(jax.devices("cpu")[0]):
    
        opt_state_mv = jax.tree_map(lambda x: jnp.zeros_like(x), trans_result)
        temp_no_prefix, temp_other = {}, {}
        for key_tuple, param in opt_state_mv.items():
            if "repeat" in key_tuple:
                temp_no_prefix[key_tuple] = MaskedNode()
                temp_other[key_tuple] = param
            else:
                temp_no_prefix[key_tuple] = param
                temp_other[key_tuple] = MaskedNode()

        temp_no_prefix = unflatten_dict(temp_no_prefix)
        temp_other = unflatten_dict(temp_other)

        no_prefix = {"count": jnp.array(step), "m": temp_no_prefix, "v": temp_no_prefix}
        other = {"count": jnp.array([step] * n_layers), "m": temp_other, "v": temp_other}
        trans_opt_states = {
            "no_prefix": [{"count": jnp.array(step)}] * 2 + [no_prefix, {"count": jnp.array(step)}],
            f"p#{n_layers}#i-1": [{"count": jnp.array([step] * n_layers)}] * 2
            + [other, {"count": jnp.array([step] * n_layers)}],
        }
        trans_opt_states = [trans_opt_states]
else:
    trans_opt_states = []

new_trainstate = TrainState(
    step=jnp.array(step),
    mdl_vars=unflatten_dict(trans_result),
    opt_states=trans_opt_states,
)
padded_global_shapes = jax.tree_map(
    lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype) if hasattr(x, "shape") else x,
    new_trainstate,
)
print(f"padded_global_shapes: {padded_global_shapes}")
checkpoint_manager.save(
    step, new_trainstate, padded_global_shapes, train_input_pipeline=None, force=False
)
print(f"Saved model finished. take time: {time.time() - start}s...")
