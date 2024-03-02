
from typing import Any, Callable, Iterable, Sequence, Tuple, Union, Optional

from absl import logging

import flax.linen as nn
import jax
import numpy as np


import functools
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.flax import aqt_flax
from dataclasses import dataclass


# chex==0.1.85
# optax==0.1.9
# flax==0.8
# jax==0.4.23

Config = Any

@dataclass
class AqtQuantization:
  """ Configures AQT quantization github.com/google/aqt. """
  quant_dg: aqt_config.DotGeneral
  quant_mode: aqt_flax.QuantMode = aqt_flax.QuantMode.TRAIN

  def dot_general_cls(self):
    """ Returns dot_general configured with aqt params. """
    aqt_dg_cls = functools.partial(
      aqt_flax.AqtDotGeneral,
      self.quant_dg,
      rhs_quant_mode=self.quant_mode
      )
    return aqt_dg_cls

  def einsum(self):
    """ Returns einsum configured with aqt params """
    aqt_einsum = functools.partial(aqt_flax.AqtEinsum(
      cfg=self.quant_dg,
      lhs_quant_mode=self.quant_mode
      )
    )
    return aqt_einsum

def _get_quant_config(config):
  if not config.quantization or config.quantization == '':
    return None
  elif config.quantization == "int8":
    if config.quantization_local_shard_count == 0:
      drhs_bits = None
      drhs_accumulator_dtype = None
      drhs_local_aqt=None
    else:
      drhs_bits = 8
      drhs_accumulator_dtype = jnp.int32
      drhs_local_aqt = aqt_config.LocalAqt(config.quantization_local_shard_count)
    return aqt_config.config_v3(
      fwd_bits=8,
      dlhs_bits=8,
      drhs_bits=drhs_bits,
      rng_type='jax.uniform',
      dlhs_local_aqt=None,
      drhs_local_aqt=drhs_local_aqt,
      fwd_accumulator_dtype=jnp.int32,
      dlhs_accumulator_dtype=jnp.int32,
      drhs_accumulator_dtype=drhs_accumulator_dtype,
    )
  else:
    raise ValueError(f'Invalid value configured for quantization {config.quantization}.')


def get_quant_mode(quant_mode_str: str = 'train'):
  """ Set quant mode."""
  if quant_mode_str == 'train':
    return aqt_flax.QuantMode.TRAIN
  elif quant_mode_str == 'serve':
    return aqt_flax.QuantMode.SERVE
  elif quant_mode_str == 'convert':
    return aqt_flax.QuantMode.CONVERT
  else:
    raise ValueError(f'Invalid quantization mode {quant_mode_str}.')
  return None


def configure_quantization(config: Config, quant_mode_str: str = 'train'):
  """ Configure quantization based on user config and quant mode."""
  quant_cfg = _get_quant_config(config)
  if quant_cfg:
    quant_mode = get_quant_mode(quant_mode_str)
    return AqtQuantization(quant_dg=quant_cfg, quant_mode=quant_mode)
  return None


@dataclass
class AqtCfg:
    quantization:str ="int8"
    quantization_local_shard_count: int = 1


# def aqt_dot_general(inputs, kernel, dimension_numbers):
#     config = AqtCfg()
#     aqt_config = configure_quantization(config)
#     dot_general = aqt_flax.AqtDotGeneral(aqt_config, rhs_quant_mode=aqt_flax.QuantMode.TRAIN)
#     # lsp: inputs and kernel dtype is bf16 or fp32
#     # AqtDotGeneral
#     # (head_nums, head_dim)
#     # axis = _canonicalize_tuple(axis) # -1 -> (-1, )
#     # # bsz * length * head_nums * head_dim
#     # axis = _normalize_axes(axis, inputs.ndim) # (-1, ) -> (3, )
#     # contract_ind = tuple(range(0, len(axis)))
#     output = dot_general(inputs, kernel, (dimension_numbers, ((), ())), precision=None)


