import string
from typing import Any, Callable, Iterable, Sequence, Tuple, Union, Optional

from absl import logging
import flax.linen as nn
import jax
import numpy as np
import jax.numpy as jnp
import functools
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.flax import aqt_flax
from dataclasses import dataclass


def einsum_eqn_to_dimension_numbers(
    eqn: str,
):
  if '.' in eqn:
    raise NotImplementedError(
        f"Dynamic batch dims ('...') are not supported. Input eqn: {eqn}. "
    )
  inputs, out_names = eqn.split('->')
  num_commas = inputs.count(',')
  if num_commas != 1:
    raise ValueError(
        f'einsum equation ({eqn}) expected two arguments, '
        f'but {num_commas+1} found.'
    )
  lhs_names, rhs_names = inputs.split(',')
  if len(lhs_names) != len(set(lhs_names)):
    raise ValueError(f'Repeated dim names are not supported, got {lhs_names=}')
  if len(rhs_names) != len(set(rhs_names)):
    raise ValueError(f'Repeated dim names are not supported, got {rhs_names=}')

  # Order batch_names by their appearance in out_names to minimize the need for
  # permutation.
  batch_names = [
      name for name in out_names if name in lhs_names and name in rhs_names
  ]
  lhs_batch_dims = [lhs_names.index(name) for name in batch_names]
  rhs_batch_dims = [rhs_names.index(name) for name in batch_names]

  # dot_general only supports contraction dims which exist in both arguments.
  lhs_contraction_names = set(lhs_names) - set(out_names)
  rhs_contraction_names = set(rhs_names) - set(out_names)
  if lhs_contraction_names != rhs_contraction_names:
    raise ValueError(
        'Contraction dims must be present in both lhs and rhs, but got '
        f'{lhs_contraction_names} and {rhs_contraction_names}'
    )
  # The order of the contraction dims does not matter to dot_general so long as
  # it is the same for both arguments, but ensuring that the order is
  # deterministic allows for easier validation of exported model artifacts.
  contraction_names = sorted(lhs_contraction_names)

  lhs_contraction_dims = [lhs_names.index(name) for name in contraction_names]
  rhs_contraction_dims = [rhs_names.index(name) for name in contraction_names]
  dimension_numbers = (
      (tuple(lhs_contraction_dims), tuple(rhs_contraction_dims)),
      (tuple(lhs_batch_dims), tuple(rhs_batch_dims)),
  )

  # Compute the order of the dot_general output names.
  shared_names = set(batch_names).union(contraction_names)
  lhs_unshared_names = [name for name in lhs_names if name not in shared_names]
  rhs_unshared_names = [name for name in rhs_names if name not in shared_names]
  dot_general_out_names = batch_names + lhs_unshared_names + rhs_unshared_names

  # Permute the dot_general output to match the einsum output if necessary.
  assert set(dot_general_out_names) == set(out_names), dot_general_out_names
  perm = None
  if dot_general_out_names != list(out_names):
    perm = tuple(dot_general_out_names.index(name) for name in out_names)
  return dimension_numbers, perm


# chex==0.1.85
# optax==0.1.9
# flax==0.8.1
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


def get_dimension(eqn, ndim):
    # eqn='BNTS,BSNH->BTNH'
    # eqn='BD,DH->BH'
    if '.' in eqn:
        # Replace the ellipsis with arbitrary symbols.
        eqn_sym = ''.join(sorted(set(string.ascii_uppercase) - set('yz')))
        batch_eqn = eqn_sym[:(ndim - 1)] if ndim else '...'
        eqn_edited = f'{batch_eqn}y,yz->{batch_eqn}z'
        dimension_numbers, _ = einsum_eqn_to_dimension_numbers(eqn_edited)
    else:
        dimension_numbers, _ = einsum_eqn_to_dimension_numbers(eqn)
    return  dimension_numbers
   

class DenseGeneral(nn.Module):
  quant: Optional[Any] = None

  @nn.compact
  def __call__(self, eqn, inputs, kernel, dimensions=None):

    assert self.quant is not None

    def compute_dot_general(inputs, kernel, dimensions):
        # AqtDotGeneral
        dot_general_cls = self.quant.dot_general_cls()
        dot_general = dot_general_cls()
        # dimensions = (((3, ), (3, )), ((0, ), (0, )))， dimensions[0]表示两个向量计算的维度, 
        # dimensions[1]表示两个向量的batch维度，可以是2，比如在计算qkscore的时候
        # 例如qk：inputs: (32, 2048, 32, 128) kernel: (32, 2048, 32, 128) dimensions: (((3,), (3,)), ((0, 2), (0, 2)))
        return dot_general(inputs, kernel, dimensions, precision=None)
    logging.info(f'inputs: {inputs.shape} kernel: {kernel.shape}')

    if dimensions is None:
      dimensions = get_dimension(eqn, ndim=inputs.ndim)

    logging.info(f'dimensions: {dimensions}')
    # return inputs
    output = compute_dot_general(inputs, kernel, dimensions=dimensions)
    return output

