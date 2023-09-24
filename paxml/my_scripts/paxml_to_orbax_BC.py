import os
import time
import json
import argparse
from collections import defaultdict

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
import smart_open
from praxis import py_utils
import orbax
import orbax.checkpoint
import flax
from flax.traverse_util import flatten_dict, unflatten_dict


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
    help="Need to be converted model weight dir. it is a dir, stong recomand use local dir instead of cloud bucket.",
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

# if not bucket dir, local dir in first indice must be ‘/’, such as /path/......
if 'gs:' not in read_dir:
    assert os.path.exists(read_dir)

if "gs" not in save_dir:
    os.makedirs(save_dir, exist_ok=True)

params = LLAMA_STANDARD_CONFIGS[model_size]
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
x_times = n_layers
head_dim = dim // n_heads
intermediate_size = params["intermediate_size"]

if version == "v1":
    vocab_size = 64000
elif version == "v2":
    vocab_size = 125696
else:
    raise

# checkpoint manager
step_prefix = "checkpoint"
step_format_fixed_length = 8

options = orbax.checkpoint.CheckpointManagerOptions(
    step_prefix=step_prefix, step_format_fixed_length=step_format_fixed_length
)
item = {"state": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())}
mngr = orbax.checkpoint.CheckpointManager(read_dir, item, options)

if step is None:
    step = mngr.latest_step()

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


def norm(x):
    wnorm = jnp.linalg.norm(x, ord=2.0, axis=0, keepdims=True)
    x = x / wnorm.clip(1e-12)
    return x, wnorm


def inverse_permute2(w):
    reshaped_w = w.reshape(dim, n_heads, 2, dim // n_heads // 2)
    transposed_w = reshaped_w.transpose(0, 1, 3, 2)
    inverted_w = transposed_w.reshape(dim, dim)
    return inverted_w


def get_metafile():
    unflat_unpadded_global_shapes = None
    checkpoint_name = f"{step_prefix}_" + str(step).zfill(step_format_fixed_length)
    metadata_path = os.path.join(read_dir, checkpoint_name, "metadata/metadata")
    print(f"metadata_path: {metadata_path}")
    try:
        with smart_open.open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as error:
        print(f"Error: {error}")
        metadata = None

    if metadata is None:
        return None

    flat_metadata = flatten_dict(metadata["train_state_metadata"])
    unpadded_global_shapes = defaultdict(dict)
    for k, v in flat_metadata.items():
        param_key, shape_dtype = k[:-1], k[-1]
        if shape_dtype in ["unpadded_shape", "dtype"]:
            unpadded_global_shapes[param_key][shape_dtype] = v
        shape_dtype = unpadded_global_shapes[param_key]
        if len(shape_dtype) == 2:
            shape_dtype = jax.ShapeDtypeStruct(shape=shape_dtype["unpadded_shape"], dtype=shape_dtype["dtype"])
            unpadded_global_shapes.update({param_key: shape_dtype})
    unflat_unpadded_global_shapes = unflatten_dict(unpadded_global_shapes)
    print("unpadded_global_shapes bulid finished!!!")
    assert unflat_unpadded_global_shapes is not None

    return unflat_unpadded_global_shapes


def build_global_shape():
    metafile = get_metafile()
    if metafile is not None:
        return metafile
    unpadded_global_shapes = {}
    for k, v in paxml_to_mesh_key_and_shape.items():
        k = tuple(k.split("."))
        if "repeat" in k:
            unpadded_global_shapes[k] = jax.ShapeDtypeStruct(shape=(x_times,) + v["shape"], dtype=jnp.float32)
        else:
            unpadded_global_shapes[k] = jax.ShapeDtypeStruct(shape=v["shape"], dtype=jnp.float32)
    return {"mdl_vars": unflatten_dict(unpadded_global_shapes)}
# build global shape
unpadded_global_shapes = build_global_shape()

items = {"state": unpadded_global_shapes}
restored_model = mngr.restore(step, items=items)
if isinstance(restored_model["state"], dict):
    loaded = {".".join(k): v for k, v in flatten_dict(restored_model["state"]["mdl_vars"]).items()}
else:
    loaded = {".".join(k): v for k, v in flatten_dict(restored_model["state"].mdl_vars).items()}
print("Model load finished!!!")

jax_weights = {
    "transformer": {
        "wte": {"embedding": loaded[mesh_to_paxml_format["wte"]]},
        "ln_f": {"kernel": loaded[mesh_to_paxml_format["ln_f"]]},
        "h": {
            "%d"
            % (layer_i): {
                "attention": {
                    "wq": {"kernel": inverse_permute2(loaded[mesh_to_paxml_format["wq"]][layer_i].reshape(dim, -1))},
                    "wk": {"kernel": inverse_permute2(loaded[mesh_to_paxml_format["wk"]][layer_i].reshape(dim, -1))},
                    "wv": {"kernel": loaded[mesh_to_paxml_format["wv"]][layer_i].reshape(dim, -1)},
                    "wo": {"kernel": loaded[mesh_to_paxml_format["wo"]][layer_i].reshape(dim, -1).transpose(1, 0)},
                },
                "feed_forward": {
                    "w1": {"kernel": loaded[mesh_to_paxml_format["w1"]][layer_i]},
                    "w2": {"kernel": loaded[mesh_to_paxml_format["w2"]][layer_i]},
                    "w3": {"kernel": loaded[mesh_to_paxml_format["w3"]][layer_i]},
                },
                "attention_norm": {"kernel": loaded[mesh_to_paxml_format["attention_norm"]][layer_i]},
                "ffn_norm": {"kernel": loaded[mesh_to_paxml_format["ffn_norm"]][layer_i]},
            }
            for layer_i in range(params["n_layers"])
        },
    },
    "lm_head": {"kernel": loaded[mesh_to_paxml_format["lm_head"]]},
}

with jax.default_device(jax.devices("cpu")[0]):
    item = {
        "params": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        "step": orbax.checkpoint.Checkpointer(orbax.checkpoint.ArrayCheckpointHandler()),
    }
    mngr = orbax.checkpoint.CheckpointManager(save_dir, item)
    start = time.time()
    mngr.save(step, {"params": {"params": flax.core.frozen_dict.freeze(jax_weights)}, "step": jax.numpy.array([step])})

print(f"Save orbax format finished, take time: {time.time() - start}")

# usage:
# python paxml_to_hf_BC.py --read_dir gs://llm_base_models/baichuan_models/13b/2/paxml/checkpoints --save_dir gs://llm_base_models/baichuan_models/13b/2/paxml/orbax/xm_model_step8000/ --step 8000 --model_size 13b --version v2
