import sys
import time
import os
import re
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import orbax
from optax import MaskedNode
from flax.traverse_util import flatten_dict, unflatten_dict
from etils import epath

from praxis import base_hyperparams
from praxis import pax_fiddle
from praxis import py_utils
from paxml import checkpoints
from paxml import checkpoint_managers
from paxml import train_states
from paxml import trainer_lib


os.environ["JAX_PLATFORMS"] = "cpu"

TrainState = train_states.TrainState
instantiate = base_hyperparams.instantiate
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
        'vocab_size': 64000,
    },
    '13b': {
        'dim': 5120,
        'intermediate_size': 13696, # baichuan
        'n_layers': 40,
        'n_heads': 40,
        'norm_eps': 1e-6,
        'vocab_size': 64000, # baichuan
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

parser = argparse.ArgumentParser(description='Mesh-orbax to paxml-orbax format script')
parser.add_argument('--read_dir', type=str, help='Need to be converted model weight dir. it is a dir, stong recomand use local dir instead of cloud bucket.')
parser.add_argument('--save_dir', type=str,  help='Save model weight file path, it is a dir.')
parser.add_argument('--model_size', type=str, default='7b', choices=['7b', '13b', '30b', '65b'], help='model size')
parser.add_argument('--step', type=int, default=None, help='Save checkpoint step')
parser.add_argument('--check', action='store_true', default=False, help='whether to check model is saved successful')

args = parser.parse_args()

model_size = args.model_size
read_dir = args.read_dir
save_dir = args.save_dir
step = args.step

params = LLAMA_STANDARD_CONFIGS[model_size.lower()]
n_layers = params['n_layers']
dim = params['dim']
vocab_size = params['vocab_size']
intermediate_size = params['intermediate_size']
x_times = 32
n_heads = params['n_heads']
head_dim = dim // n_heads

print(f'read_dir: {args.read_dir}')
print(f'save_dir: {args.save_dir}')
print(f'model_size: {args.model_size}')

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

start = time.time()
print(f'Start load pretrained model params....')
read_dir = epath.Path(read_dir)
gold_item = {'params': orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())}
gold_mngr = orbax.checkpoint.CheckpointManager(read_dir, gold_item)
with jax.default_device(jax.devices("cpu")[0]):
    gold_w = gold_mngr.restore(gold_mngr.latest_step())
print(f'Load pretrained model params finished, take time: {time.time() - start}s.')

paxml_to_mesh_key_and_shape = {
    'params.lm.embedding_lookup.emb_var': {'shape': (vocab_size, dim), 'map_to_mesh': 'wte'},
    'params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w': {'shape': (dim, intermediate_size), 'map_to_mesh': 'w3'},
    'params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1_gate.linear.w': {'shape': (dim, intermediate_size), 'map_to_mesh': 'w1'},
    'params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w': {'shape': (intermediate_size, dim), 'map_to_mesh': 'w2'},
    'params.lm.transformer.repeat.sub.x_layers_0.self_attention.query.w': {'shape': (dim, n_heads, head_dim), 'map_to_mesh': 'wq'},
    'params.lm.transformer.repeat.sub.x_layers_0.self_attention.key.w': {'shape': (dim, n_heads, head_dim), 'map_to_mesh': 'wk'},
    'params.lm.transformer.repeat.sub.x_layers_0.self_attention.value.w': {'shape': (dim, n_heads, head_dim), 'map_to_mesh': 'wv'},
    'params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w': {'shape': (dim, n_heads, head_dim), 'map_to_mesh': 'wo'},
    'params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale': {'shape': (dim,), 'map_to_mesh': 'attention_norm'},
    'params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale': {'shape': (dim, ), 'map_to_mesh': 'ffn_norm'},
    'params.lm.final_ln.scale': {'shape': (dim, ), 'map_to_mesh': 'ln_f'},
    'params.lm.softmax.logits_ffn.linear.w': {'shape': (dim, vocab_size), 'map_to_mesh': 'lm_head'}
    }

trans_result = {}
with jax.default_device(jax.devices("cpu")[0]):
    for k, v in paxml_to_mesh_key_and_shape.items():
        v = v['map_to_mesh']
        k = tuple(k.split('.'))
        values = []
        for gold_key, glod_values in flatten_dict(gold_w['params']).items():
            if v in gold_key:
                if v in 'wqwkwvwo':
                    glod_values = glod_values.reshape(dim, n_heads, head_dim)
                re_str = '#'.join(gold_key)
                try:
                    layer_index = int(re.findall('\d+', re_str)[0])
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

args.check = args.check
if args.check:
    start = time.time()
    print(f'Args check is {args.check}, start to check model whether saved successful...')
    mesh_to_paxml_format = {v['map_to_mesh']: k for k, v in paxml_to_mesh_key_and_shape.items()}
    padded_global_shapes = {}
    for k, v in paxml_to_mesh_key_and_shape.items():
        k = tuple(k.split('.'))
        if 'repeat' in k:
            padded_global_shapes[k] = jax.ShapeDtypeStruct(shape=(x_times, ) + v['shape'], dtype=jnp.float32)
        else:
            padded_global_shapes[k] = jax.ShapeDtypeStruct(shape=v['shape'], dtype=jnp.float32)
    padded_global_shapes = TrainState(step=jnp.array(step), mdl_vars=unflatten_dict(padded_global_shapes), opt_states=None)
    print(f'Padded_global_shapes bulid finished!!!')
    print(f'Start load model to check whether saved model is True or False...')
    restore_kwargs = {'state': {'version': 1.1}}
    items = {'state': padded_global_shapes}
    restored_model = checkpoint_manager._manager.restore(step, items=items, restore_kwargs=restore_kwargs)
    print(f'Check model finished. model is  saved successfully. take time: {time.time() - start}s...')