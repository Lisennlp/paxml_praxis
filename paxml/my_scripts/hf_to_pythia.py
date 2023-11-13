#!/usr/bin/env python
# coding: utf-8
import sys
import time
import os
import gc
import re
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
from praxis import py_utils
from paxml import checkpoints
from paxml import checkpoint_managers
from paxml import train_states
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


# Usage:
# python hf_to_paxml.py --read_dir /path/to/hf_model_dir/  --save_dir /path/to/paxml_model_dir/ --step 0 --version v1

LLAMA_STANDARD_CONFIGS = {
    "410m": {
        "dim": 1024,
        "intermediate_size": 4096,
        "n_layers": 24,
        "n_heads": 16,
        "norm_eps": 1e-5,
        "vocab_size": 50304
    },
    "6.9b": {
        "dim": 4096,
        "intermediate_size": 16384,
        "n_layers": 32,
        "n_heads": 32,
        "norm_eps": 1e-5,
        "vocab_size": 50432
    },
}

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

save_dir = "gs://llm_base_models/pythia/test1109/checkpoints/"

save_dir = epath.Path(save_dir)
checkpoint_manager = checkpoint_managers.OrbaxCheckpointManager(
    save_dir,
    checkpointer,
    train_input_checkpointer=False,
    options=options,
    checkpoint_type=checkpoint_type,
    tensorstore_use_ocdbt=False,
)


step = 140
model_size = "410m"
params = LLAMA_STANDARD_CONFIGS[model_size]
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
vocab_size = params['vocab_size']
intermediate_size = params["intermediate_size"]
head_dim = dim // n_heads



from transformers import GPTNeoXForCausalLM, AutoTokenizer


model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-410m-deduped",
  revision="step3000",
  cache_dir="./pythia-410m-deduped/step3000",
)
tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-410m-deduped",
  revision="step3000",
  cache_dir="./pythia-410m-deduped/step3000",
)
model = model.to(torch.bfloat16)
model.eval()

ckpt = {}
for k, v in model.named_parameters():
    if k.startswith("gpt_neox."):
        k = k[9:]
    ckpt[k] = v

paxml_to_hf_key_and_shape = {
    "params.lm.embedding_lookup.emb_var": {
        "shape": (vocab_size, dim),
        "map_to_hf": "embed_in",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w": {
        "shape": (dim, intermediate_size),
        "map_to_hf": "dense_h_to_4h.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b": {
        "shape": (dim, intermediate_size),
        "map_to_hf": "dense_h_to_4h.bias",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w": {
        "shape": (intermediate_size, dim),
        "map_to_hf": "dense_4h_to_h.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b": {
        "shape": (intermediate_size, dim),
        "map_to_hf": "dense_4h_to_h.bias",
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
        "map_to_hf": "dense.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.b": {
        "shape": (dim, n_heads, head_dim),
        "map_to_hf": "dense.bias",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale": {
        "shape": (dim,),
        "map_to_hf": "input_layernorm.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.layer_norm.bias": {
        "shape": (dim,),
        "map_to_hf": "input_layernorm.bias",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale": {
        "shape": (dim,),
        "map_to_hf": "post_attention_layernorm.weight",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias": {
        "shape": (dim,),
        "map_to_hf": "post_attention_layernorm.bias",
    },
    "params.lm.final_ln.scale": {"shape": (dim,), "map_to_hf": "final_layer_norm.weight"},
    "params.lm.final_ln.bias": {"shape": (dim,), "map_to_hf": "final_layer_norm.bias"},
    "params.lm.softmax.logits_ffn.linear.w": {
        "shape": (dim, vocab_size),
        "map_to_hf": "embed_out.weight",
    },
}

gold_w = ckpt
split_qkv = {}
for k, v in gold_w.items():
    if v.dtype != torch.float32:
        v = v.to(torch.float32)
    # o_proj不进行transpose，是个坑
    if len(v.shape) == 2 and "embed_in" not in k and "dense." not in k:
        v = v.transpose(1, 0)
    else:
        print(f"No transpose k: {k}")

    if "query_key_value" in k:
        qq = k.replace("query_key_value", "q_proj")
        kk = k.replace("query_key_value", "k_proj")
        vv = k.replace("query_key_value", "v_proj")
        if len(v.shape) == 1:
            v = v.reshape(n_heads, 3 * head_dim)
            split_qkv[qq] = v[..., :head_dim].detach().numpy().reshape(-1)
            split_qkv[kk] = v[..., head_dim: 2 * head_dim].detach().numpy().reshape(-1)
            split_qkv[vv] = v[..., 2 * head_dim: ].detach().numpy().reshape(-1)
        elif len(v.shape) == 2:
            v = v.reshape(dim, n_heads, 3 * head_dim)
            split_qkv[qq] = v[..., :head_dim].detach().numpy().reshape(dim, -1)
            split_qkv[kk] = v[..., head_dim: 2 * head_dim].detach().numpy().reshape(dim, -1)
            split_qkv[vv] = v[..., 2 * head_dim: ].detach().numpy().reshape(dim, -1)
        else:
            raise ValueError(f'qkv shape is error!!!')
    else:
        split_qkv[k] = v.detach().numpy()

for k, v in split_qkv.items():
    print(k, v.shape)

# 构造模型参数
trans_result = {}
flag = 0
with jax.default_device(jax.devices("cpu")[0]):
    for k, v in paxml_to_hf_key_and_shape.items():
        v = v["map_to_hf"]
        k = tuple(k.split("."))
        values = []
        for gold_key, glod_values in split_qkv.items():
            flag = 0
            if (v in gold_key and v != "norm.weight") or v == gold_key == "norm.weight":
                print(v, gold_key, "====")
                flag = 1
                match_res = re.findall("q_proj|k_proj|v_proj|dense\.", v)
                if match_res:
                    if len(glod_values.shape) > 1:
                        glod_values = glod_values.reshape(dim, n_heads, head_dim)
                    else:
                        if "dense.bias" not in v:
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
    opt_state_mv = jax.tree_map(lambda x: jnp.zeros_like(x), trans_result)

print(f"Please simple check model shape and dtype...")
for k, v in trans_result.items():
    print(k, v.shape, v.dtype)

if step is None:
    latest_step = checkpoint_manager.latest_step()
    step = 0 if latest_step is None else latest_step + SAVE_INTERVAL_STEPS

# 构造optimizer参数
print(f"Model save step is {step}")
start = time.time()
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
checkpoint_manager.save(step, new_trainstate, padded_global_shapes, train_input_pipeline=None, force=False)
print(f"Saved model finished. take time: {time.time() - start}s...")
