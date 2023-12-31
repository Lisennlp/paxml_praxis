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
from praxis import py_utils

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
NestedMap = py_utils.NestedMap
checkpoint_type = CheckpointType.GDA
SAVE_INTERVAL_STEPS = 1


LLAMA_STANDARD_CONFIGS = {
    "1b": {
        "dim": 4096,
        "intermediate_size": 11008,
        "n_layers": 2,
        "n_heads": 32,
        "norm_eps": 1e-6,
    },
    "7b": {
        "dim": 4096,
        "intermediate_size": 11008,
        "n_layers": 32,
        "n_heads": 32,
        "norm_eps": 1e-6,
    },
    "13b": {
        "dim": 5120,
        "intermediate_size": 13696,
        "n_layers": 40,
        "n_heads": 40,
        "norm_eps": 1e-6,
    },
    "30b": {
        "dim": 6656,
        "intermediate_size": 17920,
        "n_layers": 60,
        "n_heads": 52,
        "norm_eps": 1e-6,
    },
    "65b": {
        "dim": 8192,
        "intermediate_size": 22016,
        "n_layers": 80,
        "n_heads": 64,
        "norm_eps": 1e-5,
    },
}

parser = argparse.ArgumentParser(description="Mesh-orbax to paxml-orbax format script")
parser.add_argument(
    "--read_dir",
    type=str,
    help=(
        "Need to be converted model weight dir. it is a dir, stong recomand use local dir instead"
        " of cloud bucket."
    ),
)
parser.add_argument(
    "--save_dir",
    type=str,
    help="Save model weight file path, it is a local dir not bucket dir.",
)
parser.add_argument(
    "--model_size",
    type=str,
    default="7b",
    choices=["7b", "13b", "30b", "65b"],
    help="model size",
)
parser.add_argument("--step", type=int, default=None, help="Load checkpoint step")
parser.add_argument(
    "--check",
    action="store_true",
    default=False,
    help="whether to check model is saved successful",
)
parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2"], help="Model version")

args = parser.parse_args()

model_size = args.model_size
read_dir = args.read_dir
save_dir = args.save_dir
step = args.step
version = args.version

os.makedirs(save_dir, exist_ok=True)

params = LLAMA_STANDARD_CONFIGS[model_size]
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
x_times = n_layers

if version == "v1":
    vocab_size = 64000
elif version == "v2":
    vocab_size = 125696
else:
    raise

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

paxml_to_mesh_key_and_shape = {
    "params.lm.embedding_lookup.emb_var": {
        "shape": (vocab_size, dim),
        "map_to_mesh": "wte",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w": {
        "shape": (dim, intermediate_size),
        "map_to_mesh": "w3",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1_gate.linear.w": {
        "shape": (dim, intermediate_size),
        "map_to_mesh": "w1",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w": {
        "shape": (intermediate_size, dim),
        "map_to_mesh": "w2",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.self_attention.query.w": {
        "shape": (dim, n_heads, head_dim),
        "map_to_mesh": "wq",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.self_attention.key.w": {
        "shape": (dim, n_heads, head_dim),
        "map_to_mesh": "wk",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.self_attention.value.w": {
        "shape": (dim, n_heads, head_dim),
        "map_to_mesh": "wv",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w": {
        "shape": (dim, n_heads, head_dim),
        "map_to_mesh": "wo",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale": {
        "shape": (dim,),
        "map_to_mesh": "attention_norm",
    },
    "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale": {
        "shape": (dim,),
        "map_to_mesh": "ffn_norm",
    },
    "params.lm.final_ln.scale": {"shape": (dim,), "map_to_mesh": "ln_f"},
    "params.lm.softmax.logits_ffn.linear.w": {
        "shape": (dim, vocab_size),
        "map_to_mesh": "lm_head",
    },
}

mesh_to_paxml_format = {v["map_to_mesh"]: k for k, v in paxml_to_mesh_key_and_shape.items()}
padded_global_shapes = {}
for k, v in paxml_to_mesh_key_and_shape.items():
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


def permute(w):
    return w
    # # torch ， view和reshape的区别：view要求连续内存，reshape随意
    # if isinstance(w, torch.Tensor):
    #     res = w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)
    # else:
    #     res = w.reshape(n_heads, dim // n_heads // 2, 2, dim).transpose(0, 2, 1, 3).reshape(dim, dim)
    # return res


# ===================================== =====================================
# transpose原因：pytorch的linear（input * w^T）和flax的linear（input * w）的实现不一致导致的
# jax中的linear实现有多重写法，而pytorch基本固定写法。所以需要看jax的model中的linear实现方式，以此
# 决定pytorch转jax model是否需要transpose
# 此外，除了linear的实现不同之外，rotary的实现方式也会导致qk的不同。需要注意。
# ===================================== =====================================


flated_paxml_w = flatten_dict(restored_model["state"].mdl_vars)
loaded = {".".join(k): v for k, v in flated_paxml_w.items()}
index_dict = {"weight_map": {}}
param_count = 0
start = time.time()
for layer_i in range(n_layers):
    filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
    print(f"layer_i: {layer_i} || filename: {filename} take time: {time.time() - start}s")
    q = permute(loaded[mesh_to_paxml_format["wq"]][layer_i].reshape(dim, -1).transpose(1, 0))
    k = permute(loaded[mesh_to_paxml_format["wk"]][layer_i].reshape(dim, -1).transpose(1, 0))
    v = loaded[mesh_to_paxml_format["wv"]][layer_i].reshape(dim, -1).transpose(1, 0)
    repeat_state_dict = {
        f"model.layers.{layer_i}.self_attn.W_pack.weight": np.concatenate([q, k, v], axis=0),
        # no transpose
        f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[mesh_to_paxml_format["wo"]][
            layer_i
        ].reshape(dim, -1),
        f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[mesh_to_paxml_format["w1"]][
            layer_i
        ].transpose(1, 0),
        f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[mesh_to_paxml_format["w2"]][
            layer_i
        ].transpose(1, 0),
        f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[mesh_to_paxml_format["w3"]][
            layer_i
        ].transpose(1, 0),
        f"model.layers.{layer_i}.input_layernorm.weight": loaded[
            mesh_to_paxml_format["attention_norm"]
        ][layer_i],
        f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
            mesh_to_paxml_format["ffn_norm"]
        ][layer_i],
    }
    for k, v in repeat_state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.size
    save_path = os.path.join(save_dir, filename)
    save_model(repeat_state_dict, save_path, mode="torch")

no_repeat_state_dict = {
    # no transpose
    "model.embed_tokens.weight": loaded[mesh_to_paxml_format["wte"]],
    "model.norm.weight": loaded[mesh_to_paxml_format["ln_f"]],  # no transpose
    "lm_head.weight": loaded[mesh_to_paxml_format["lm_head"]].transpose(1, 0),
}
filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
for k, v in no_repeat_state_dict.items():
    index_dict["weight_map"][k] = filename
    param_count += v.size
# save no repeat params
save_model(no_repeat_state_dict, os.path.join(save_dir, filename), mode="torch")
# save configs
index_dict["metadata"] = {"total_size": param_count * 2}


command = f'gsutil -m cp \
  "gs://llm_base_models/baichuan/v2/13B/hf/config.json" \
  "gs://llm_base_models/baichuan/v2/13B/hf/configuration_baichuan.py" \
  "gs://llm_base_models/baichuan/v2/13B/hf/generation_utils.py" \
  "gs://llm_base_models/baichuan/v2/13B/hf/modeling_baichuan.py" \
  "gs://llm_base_models/baichuan/v2/13B/hf/quantizer.py" \
  "gs://llm_base_models/baichuan/v2/13B/hf/special_tokens_map.json" \
  "gs://llm_base_models/baichuan/v2/13B/hf/tokenization_baichuan.py" \
  "gs://llm_base_models/baichuan/v2/13B/hf/tokenizer.model" \
  "gs://llm_base_models/baichuan/v2/13B/hf/tokenizer_config.json" \
  {save_dir}'
subprocess.run(command, stdout=subprocess.PIPE, shell=True)

write_json(index_dict, os.path.join(save_dir, "pytorch_model.bin.index.json"))

print(f"Convert finished, take time: {time.time() - start}s...")

# usage:
# STEP=9800
# python bc.py --read_dir gs://llm_base_models/baichuan/v2/13B/paxml1214/checkpoints --save_dir ./bc2_13b_step$STEP/ --version v2 --model_size 13b --step $STEP