import os
import time
import json
import argparse
import subprocess

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
from flax.traverse_util import flatten_dict, unflatten_dict
from paxml import train_states
from paxml import checkpoint_managers
from paxml import checkpoints  # mapped to internal

try:
    import torch
except Exception as e:
    print(f"Error: {e}")
    command = 'conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch'
    subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    import torch

TrainState = train_states.TrainState
CheckpointType = checkpoints.CheckpointType
Checkpointer = checkpoints.Checkpointer
PaxCheckpointHandler = checkpoints.PaxCheckpointHandler
checkpoint_type = CheckpointType.GDA
SAVE_INTERVAL_STEPS = 1

LLAMA_STANDARD_CONFIGS = {
    "7B": {
        "dim": 4096,
        "intermediate_size": 11008,
        "n_layers": 32,
        "n_heads": 32,
        "norm_eps": 1e-6,
        "vocab_size": 151936
    },
    "14B": {
        "dim": 5120,
        "intermediate_size": 13696,
        "n_layers": 40,
        "n_heads": 40,
        "norm_eps": 1e-6,
        "vocab_size": 152064
    },
}


step = 5200
model_size = '14B'
params = LLAMA_STANDARD_CONFIGS[model_size]
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
intermediate_size = params["intermediate_size"]
head_dim = dim // n_heads
save_opt = False

read_dir = 'gs://llm_base_models/qwen/14B/paxml_c8200/checkpoints'
read_dir = 'gs://llm_base_models_us-east5/qwen/14B/sft_base_bookstart_step9000_0313/checkpoints'
read_dir = 'gs://llm_base_models_us-east5/qwen/14B/sft_bookstart_step9000_with_continue_data'
# read_dir = 'gs://llm_base_models_us-east5/qwen/14B/sft_bookstart_step9000_without_continue_data_but_flag'

save_dir = f'paxml_to_hf{step}'

os.makedirs(save_dir, exist_ok=True)
params = LLAMA_STANDARD_CONFIGS[model_size]
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
x_times = n_layers
vocab_size = params['vocab_size']
intermediate_size = params["intermediate_size"]
head_dim = dim // n_heads

options = checkpoint_managers.CheckpointManagerOptions(
    max_to_keep=10,
    save_interval_steps=SAVE_INTERVAL_STEPS,
    step_prefix="checkpoint",
    step_format_fixed_length=8,
)
checkpointer = Checkpointer(
    PaxCheckpointHandler(
        enforce_restore_shape_check=False,
        use_ocdbt=False,
    )
)
job_log_dir = epath.Path(read_dir)
checkpoint_manager = checkpoint_managers.OrbaxCheckpointManager(
    job_log_dir,
    checkpointer,
    train_input_checkpointer=False,
    options=options,
    checkpoint_type=checkpoint_type,
    tensorstore_use_ocdbt=False,
)

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


hf_to_paxml_format = {v["map_to_hf"]: k for k, v in paxml_to_hf_key_and_shape.items()}
padded_global_shapes = {}
for k, v in paxml_to_hf_key_and_shape.items():
    k = tuple(k.split("."))
    if "repeat" in k:
        padded_global_shapes[k] = jax.ShapeDtypeStruct(
            shape=(x_times,) + v["shape"], dtype=jnp.float32
        )
    else:
        padded_global_shapes[k] = jax.ShapeDtypeStruct(shape=v["shape"], dtype=jnp.float32)

padded_global_shapes = TrainState(
    step=jnp.array(step), mdl_vars=unflatten_dict(padded_global_shapes), opt_states=None
)
print("Padded_global_shapes bulid finished!!!")

if step is None:
    step = checkpoint_manager.latest_step()

restore_kwargs = {"state": {"version": 1.1}}
items = {"state": padded_global_shapes}
restored_model = checkpoint_manager._manager.restore(
    step, items=items, restore_kwargs=restore_kwargs
)

def save_model(state_dict, save_path, mode="torch"):
    save_path = os.path.join(save_dir, filename)
    if mode != "torch":
        np.save(open(save_path, "wb"), state_dict)
    else:
        torch.save(
            {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in state_dict.items()},
            save_path,
        )


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f, indent=2)



flated_paxml_w = flatten_dict(restored_model["state"].mdl_vars)
loaded = {".".join(k): v for k, v in flated_paxml_w.items()}

index_dict = {"weight_map": {}}
param_count = 0
start = time.time()
for layer_i in range(n_layers):
    filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
    print(f"layer_i: {layer_i} || filename: {filename} take time: {time.time() - start}s")
    q = loaded[hf_to_paxml_format["q_proj.weight"]][layer_i].reshape(dim, -1).transpose(1, 0)
    k = loaded[hf_to_paxml_format["k_proj.weight"]][layer_i].reshape(dim, -1).transpose(1, 0)
    v = loaded[hf_to_paxml_format["v_proj.weight"]][layer_i].reshape(dim, -1).transpose(1, 0)
    qb = loaded[hf_to_paxml_format["q_proj.bias"]][layer_i].reshape(dim, )
    kb = loaded[hf_to_paxml_format["k_proj.bias"]][layer_i].reshape(dim, )
    vb = loaded[hf_to_paxml_format["v_proj.bias"]][layer_i].reshape(dim, )
    repeat_state_dict = {
        f"transformer.h.{layer_i}.attn.c_attn.weight": np.concatenate([q, k, v], axis=0),
        f"transformer.h.{layer_i}.attn.c_attn.bias": np.concatenate([qb, kb, vb], axis=0),
        # no transpose
        f"transformer.h.{layer_i}.attn.c_proj.weight": loaded[hf_to_paxml_format["attn.c_proj.weight"]][layer_i].reshape(dim, -1),
        f"transformer.h.{layer_i}.mlp.w2.weight": loaded[hf_to_paxml_format["w2.weight"]][layer_i].transpose(1, 0),
        f"transformer.h.{layer_i}.mlp.w1.weight": loaded[hf_to_paxml_format["w1.weight"]][layer_i].transpose(1, 0),
        f"transformer.h.{layer_i}.mlp.c_proj.weight": loaded[hf_to_paxml_format["mlp.c_proj.weight"]][layer_i].transpose(1, 0),
        f"transformer.h.{layer_i}.ln_1.weight": loaded[hf_to_paxml_format["ln_1.weight"]][layer_i],
        f"transformer.h.{layer_i}.ln_2.weight": loaded[hf_to_paxml_format["ln_2.weight"]][layer_i],
    }
    for k, v in repeat_state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.size
    save_path = os.path.join(save_dir, filename)
    save_model(repeat_state_dict, save_path, mode="torch")

no_repeat_state_dict = {
    # no transpose
    "transformer.wte.weight": loaded[hf_to_paxml_format["wte.weight"]],
    "transformer.ln_f.weight": loaded[hf_to_paxml_format["ln_f.weight"]],  # no transpose
    "lm_head.weight": loaded[hf_to_paxml_format["lm_head"]].transpose(1, 0),
}
filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
for k, v in no_repeat_state_dict.items():
    index_dict["weight_map"][k] = filename
    param_count += v.size
# save no repeat params
save_model(no_repeat_state_dict, os.path.join(save_dir, filename), mode="torch")
# save configs
index_dict["metadata"] = {"total_size": param_count * 2}
# cp py and config
for f in ['py', 'json', 'tiktoken']:
    command = ["gsutil", "cp", f"gs://llm_base_models_us-east5/qwen/{model_size}/hf/*.{f}", save_dir]
    result = subprocess.run(command, capture_output=True, text=True)

write_json(index_dict, os.path.join(save_dir, "model.safetensors.index.json"))
print(f"Convert finished, take time: {time.time() - start}s...")

# usage:
# !pip install transformers_stream_generator && pip install accelerate && pip install tiktoken
#  python paxml_to_hf.py --read_dir gs://llm_base_models/baichuan_models/13b/2/paxml_1011/checkpoints --save_dir ./bc2_13b_step7040/ --version v2 --model_size 13b --step 7040