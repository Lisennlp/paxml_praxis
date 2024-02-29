# coding=utf-8
# Copyright 2022 The Pax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantized Linear Layers."""

import copy
from typing import Any

from jax import numpy as jnp
from praxis import base_layer
from praxis import pytypes
from praxis.layers import linears
from praxis.layers.quantization import operations
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantizer
from praxis.layers.quantization import utils

QuantizationMode = quantization_hparams.QuantizationMode
QuantizationType = quantization_hparams.QuantizationType
QuantizationParams = quantization_hparams.QuantizationParams
WeightHParams = base_layer.WeightHParams
instance_field = base_layer.instance_field
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
WeightInit = base_layer.WeightInit



from typing import Optional
from jax import vmap
from praxis import pax_fiddle
from praxis import py_utils
from praxis.layers import activations
from praxis.layers import base_ops

NestedMap = py_utils.NestedMap
template_field = base_layer.template_field
LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]


def _rms(x):
    # Note: under pmap .mean() will produce a local mean, not across all hosts.
    return (x**2.0).mean().astype(jnp.float32) ** 0.5

    
class Linear(linears.Linear, quantizer.QuantizationLayer):  # pytype: disable=signature-mismatch
  """Quantized Linear layer without bias.

  Attributes:
    quantization: Information related to the quantization applied to this layer,
      such as the mode for the quantization.
  """
  _PACK_4BIT_DIM = 0

  def create_tensor_quantizers(self):
    weight_params = (
        self.quantization.weight_params if self.quantization else None
    )
    act_params = self.quantization.act_params if self.quantization else None
    self.create_child(
        'act_quantizer',
        quantizer.create_tensor_quantizer('act_quantizer', act_params),
    )
    self.create_child(
        'weight_quantizer',
        quantizer.create_tensor_quantizer('weight_quantizer', weight_params),
    )

  def _do_static_activation_quantization(self) -> bool:
    act_params = self.quantization.act_params if self.quantization else None
    return act_params is not None and act_params.stats_config is not None

  def setup(self) -> None:
    wp = self.weight_split_dims_mapping
    pc = WeightHParams(
        shape=[self.input_dims, self.output_dims],
        mesh_shape=self.mesh_shape,
        tensor_split_dims_mapping=wp.wt,
    )
    self.set_up_weights(
        weight_name='w',
        weight_params=pc,
        scale_shape=[self.output_dims],
        pack_dim=self._PACK_4BIT_DIM,
    )

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """

    ap = self.activation_split_dims_mapping
    eqn = '...y,yz->...z'

    out = self.quantized_einsum(
        eqn=eqn,
        x=inputs,
        w=self.theta.w,
        pack_dim=self._PACK_4BIT_DIM,
        reshape=[],
    )
    # Adjust sharding annotation during decoding.
    # TODO(pax): This logic should likely be lifted somewhere else.
    ap_out = ap.out
    if ap_out is not None and len(ap_out) == 3 and out.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    out = base_layer.maybe_shard(out, ap_out, self.mesh_axis_names)
    return out

  def quantized_partition_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      a map from names to partition spec.
    """
    assert self.quantization is not None, (
        'quantized_partition_specs is called during serving for quantized'
        ' model, please set quantized config for the model.'
    )
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    weight_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        self._weight_hparams['w'], self.mesh_axis_names
    )
    wp = self.weight_split_dims_mapping
    scale_split_dims_mapping = [wp.wt[1]]
    # scale_weight_hparam is unmaterialized so shape is irrelevant.
    scale_weight_hparam = WeightHParams(
        shape=(), tensor_split_dims_mapping=scale_split_dims_mapping)
    scale_pspec = base_layer._weight_hparam_to_pspec(  # pylint: disable=protected-access
        scale_weight_hparam, self.mesh_axis_names
    )
    partitionspec = {'w': weight_pspec, scale_name: scale_pspec}

    if not self.quantization.weight_params.use_symmetric:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      partitionspec[zp_name] = copy.deepcopy(scale_pspec)

    # Activation variable partitioning is only needed for static quantization.
    if self._do_static_activation_quantization():
      raise NotImplementedError(
          'Static activation quantization is not supported yet.')

    return {base_layer.PARAMS: partitionspec}

  def quantize_weight(self) -> NestedJTensor:
    """Get quantized weight.

    Returns:
      a map from names to quantized weights.
    """
    assert self.quantization is not None, (
        'quantize_weight is called during serving for quantized model, please'
        ' set quantized config for the model.'
    )
    theta = self.theta
    scale_name = 'w' + base_layer.QUANTIZED_SCALE_NAME_POSTFIX
    eqn = 'xy,yz->xz'
    if self.quantization.quantization_type in [
        QuantizationType.PTQ,
        QuantizationType.FQ,
        QuantizationType.FQ_VN,
    ]:
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
      else:
        q_w, q_s, zp = operations.reduce_einsum_weight_precision(
            eqn,
            theta.w,
            calculation_type=self.dtype,
            bits=self.quantization.weight_params.precision,
            percentile=self.quantization.weight_params.clipping_coeff,
            use_symmetric=self.quantization.weight_params.use_symmetric,
        )
    elif self.quantization.quantization_type == QuantizationType.AQT:
      if self._do_static_activation_quantization():
        raise NotImplementedError(
            'Static activation quantization is not supported yet.'
        )
      else:
        q_w, q_s, zp = self.weight_quantizer.quantize(
            self.theta.w,
            [0],
            squeeze_scale=True,
            quantized_dtype=self.quantization.weight_params.dtype,
        )

    if (
        self.quantization.weight_params.precision == 4
        and self.quantization.weight_params.use_int4_packed_weights
    ):
      q_w = utils.pack_4bit(
          q_w,
          self._PACK_4BIT_DIM,
          self.quantization.weight_params.int4_packed_weights_container_dtype,
      )

    if self.quantization.weight_params.use_symmetric:
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
    else:
      zp_name = 'w' + base_layer.QUANTIZED_ZP_NAME_POSTFIX
      return {base_layer.PARAMS: {'w': q_w, scale_name: q_s, zp_name: zp}}



class Bias(base_layer.BaseLayer):
    """Bias layer.

    Attributes:
      dims: Depth of the input.
      bias_init: Init scale (constant) of bias terms.
    """

    dims: int = 0
    bias_init: Optional[float] = 0.0

    def setup(self) -> None:
        wp = self.weight_split_dims_mapping
        self.create_variable(
            "b",
            WeightHParams(
                shape=[self.dims],
                init=WeightInit.Constant(self.bias_init),
                mesh_shape=self.mesh_shape,
                tensor_split_dims_mapping=wp.wt,
            ),
        )

    def __call__(self, inputs: JTensor) -> JTensor:
        """Adds bias to inputs.

        Args:
          inputs: The inputs JTensor.  Shaped [..., dims].

        Returns:
          Inputs plus bias.
        """
        return inputs + self.theta.b

        
class FeedForward(base_layer.BaseLayer):
    """Feedforward layer with activation.

    Attributes:
      input_dims: Depth of the input.
      output_dims: Depth of the output.
      has_bias: Adds bias weights or not.
      linear_tpl: Linear layer params.
      activation_tpl: Activation layer params.
      bias_init: Init scale (constant) of bias terms.
    """

    input_dims: int = 0
    output_dims: int = 0
    has_bias: bool = True
    linear_tpl: LayerTpl = template_field(Linear)
    bias_tpl: LayerTpl = template_field(Bias)
    activation_tpl: pax_fiddle.Config[activations.BaseActivation] = template_field(activations.ReLU)
    weight_init: Optional[WeightInit] = None
    bias_init: Optional[float] = 0.0

    def setup(self) -> None:
        wp = self.weight_split_dims_mapping
        ap = self.activation_split_dims_mapping
        linear_layer_p = self.linear_tpl.clone()
        linear_layer_p.set(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            weight_init=self.weight_init,
            weight_split_dims_mapping=wp.clone(),
            activation_split_dims_mapping=ap.clone(),
        )
        # Provide type hint.
        self.linear: Linear
        self.create_child("linear", linear_layer_p)
        if self.has_bias:
            bias_layer_p = self.bias_tpl.clone()
            bias_layer_p.set(dims=self.output_dims, bias_init=self.bias_init)
            if self.mesh_shape is not None and ap.out is not None:
                wp_bias = [ap.out[-1]]
                bias_layer_p.weight_split_dims_mapping.wt = wp_bias
            # Provide type hint.
            self.bias: Bias
            self.create_child("bias", bias_layer_p)
        # Provide type hints
        self.activation: activations.BaseActivation
        self.create_child("activation", self.activation_tpl.clone())

    def __call__(self, inputs: JTensor) -> JTensor:
        chunk_size = inputs.shape[1]
        n = inputs.shape[1] // chunk_size
        # output = jnp.empty(inputs.shape, dtype=inputs.dtype)
        output = []
        for i in range(n):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            inputs_chunk = inputs[:, start: end]
            projected_inputs = self.linear(inputs_chunk)
            if self.has_bias:
                projected_inputs = self.bias(projected_inputs)
            output_chunk = self.activation(projected_inputs)
            output.append(output_chunk)
        output = jnp.concatenate(output, axis=1)  
            # output = output.at[:, start: end].set(output_chunk)
        return output