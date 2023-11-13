import json
import os
from collections import defaultdict

os.environ["JAX_PLATFORMS"] = "cpu"

import mlxu
from flax.traverse_util import flatten_dict, unflatten_dict
import orbax.checkpoint
import orbax
import jax
import torch


# read metadata and make shape and dtype like checkpoint struct

read_dir = "gs://llm_projects_us-central2/log/C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiagv4/checkpoints"
step_prefix = "checkpoint"
step_format_fixed_length = 8
load_step = 60000

options = orbax.checkpoint.CheckpointManagerOptions(
    step_prefix=step_prefix, step_format_fixed_length=step_format_fixed_length
)
item = {
    "state": orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
}
mngr = orbax.checkpoint.CheckpointManager(read_dir, item, options)

if load_step is None:
    load_step = mngr.latest_step()

checkpoint_name = f"{step_prefix}_" + str(load_step).zfill(step_format_fixed_length)

print(f"checkpoint_name: {checkpoint_name}")
metadata_path = os.path.join(read_dir, checkpoint_name, "metadata/metadata")
print(f"metadata_path: {metadata_path}")

with mlxu.open_file(metadata_path, "r") as f:
    metadata = json.load(f)

flat_metadata = flatten_dict(metadata["train_state_metadata"])
unpadded_global_shapes = defaultdict(dict)
for k, v in flat_metadata.items():
    param_key, shape_dtype = k[:-1], k[-1]
    if shape_dtype in ["unpadded_shape", "dtype"]:
        unpadded_global_shapes[param_key][shape_dtype] = v
    shape_dtype = unpadded_global_shapes[param_key]
    if len(shape_dtype) == 2:
        shape_dtype = jax.ShapeDtypeStruct(
            shape=shape_dtype["unpadded_shape"], dtype=shape_dtype["dtype"]
        )
        unpadded_global_shapes.update({param_key: shape_dtype})


# load model
unflat_unpadded_global_shapes = unflatten_dict(unpadded_global_shapes)
with jax.default_device(jax.devices("cpu")[0]):
    weights = mngr.restore(load_step, items={"state": unflat_unpadded_global_shapes})


# save torch model
flat_weights = {".".join(k): v for k, v in flatten_dict(weights).items()}
for k, v in flat_weights.items():
    print(k, v.shape)
torch.save(flat_weights, f"{checkpoint_name}.torch.bin")

# test load torch model
# load_w = torch.load('checkpoint_00060000.torch.bin')
