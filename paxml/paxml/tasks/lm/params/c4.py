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

"""Language Model configurations on the T5/C4 dataset."""

import functools
import math
from typing import Dict, List, Optional

from absl import logging
import fiddle as fdl
import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import experiment_registry
from paxml import seqio_input
from paxml import tasks_lib
from paxml import trainer_lib
from paxml.tasks.lm import model_params
from paxml.tasks.lm.params import lm_cloud
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis.layers import normalizations  # XD
from praxis.layers import transformers
import seqio
import t5.data
from t5.data import preprocessors as t5_preprocessors
import tensorflow as tf
import numpy as np
import jax.numpy as jnp
from praxis import py_utils


NestedMap = py_utils.NestedMap

WeightInit = base_layer.WeightInit

GPT_EOS_ID = 1

class MyDatasets(seqio.SeqIOInput):

  def __init__(self, path, is_training=True):
    # valid_path = "gs://jax_llm_data/data-baichuan/dreamily_translation_general.test.tfrecords"
    # trainpath = "gs://jax_llm_data/data-baichuan/dreamily_translation_general.train.tfrecords"
    self.num_infeed_hosts =  jax.process_count()
    self.seq_len = 2048
    eopch_num = 1
    self.batch_size = 8
    self.dataset = self.load_tfrecord_dataset(
                                              index_fname=path, 
                                              batch_size=self.batch_size, 
                                              seq_len=self.seq_len, 
                                              repeat=eopch_num
                                              )
    self.is_training = is_training
    
  def peek_padded(self):
    return self.get_next_padded()

  def get_next_padded(self):
    return next(self.dataset)

  def get_global_batch_size(self):
    return self.batch_size * self.num_infeed_hosts

  def _parse_function(self, example_proto):
    feature_desc = {"input_ids": tf.io.VarLenFeature(tf.int64), "labels": tf.io.VarLenFeature(tf.int64)}
    example = tf.io.parse_single_example(example_proto, feature_desc)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64: t = tf.cast(t, dtype=tf.int32)
        example[name] = tf.sparse.to_dense(t, default_value=0)
    return example

  def format(self, data):
    data = jax.tree_map(lambda x: x.numpy(), data)
    model_needed_inputs = NestedMap()
    model_needed_inputs.ids = data['input_ids'][:, :-1]
    model_needed_inputs.labels = data['input_ids'][:, 1:]
    weights = data['labels'] > 0
    model_needed_inputs.weights = weights[:, 1:]
    model_needed_inputs.paddings = 1 - weights[:, 1:]
    model_needed_inputs.segment_ids = jnp.ones_like(model_needed_inputs.ids)
    model_needed_inputs.segment_pos = jnp.broadcast_to(jnp.arange(self.seq_len - 1), model_needed_inputs.ids.shape)
    return model_needed_inputs

  def load_tfrecord_dataset(self, index_fname, batch_size, seq_len, restore_state=None, repeat=3):
    tf.random.set_seed(42)
    fnames = [index_fname] if index_fname.endswith('.tfrecords') else open(index_fname).read().splitlines()
    ds = tf.data.Dataset.from_tensor_slices(fnames)
    ds = ds.apply(tf.data.TFRecordDataset)
    # shard host data
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=10000) # 从文件中取buffer_size数据，然后打乱
    ds = ds.padded_batch(batch_size=np.prod(batch_size), 
                        padded_shapes={'input_ids': [seq_len], 'labels': [seq_len]},
                        padding_values={'input_ids': 0, 'labels': 0}, 
                        drop_remainder=True)
    ds = ds.prefetch(10)
    ds = ds.repeat(repeat)
    return map(lambda x: self.format(x), iter(ds))


class C4UnsupervisedDataset(base_experiment.BaseExperiment):
  """Used for training Baseline ULM."""
  PERCORE_BATCH_SIZE = 1
  PERCORE_EVAL_BATCH_SIZE = None
  MAX_SEQ_LEN = 1024
  TRAINING_SEED = 9876
  TRAINING_NUM_BATCHES_TO_SKIP = None
  # TRAIN_FILE = "gs://jax_llm_data/data-baichuan/dreamily_translation_general.train.tfrecords"
  # VALID_FILE = "gs://jax_llm_data/data-baichuan/dreamily_translation_general.test.tfrecords"

  def _dataset_common(
      self, is_training
  ) -> pax_fiddle.Config[base_input.BaseInput]:
    if is_training:
      percore_batch_size = self.PERCORE_BATCH_SIZE
    else:
      if self.PERCORE_EVAL_BATCH_SIZE is not None:
        percore_batch_size = self.PERCORE_EVAL_BATCH_SIZE
      else:
        percore_batch_size = self.PERCORE_BATCH_SIZE

    num_local_devices = jax.local_device_count() # 8
    # lsp: global_batch_size: percore_batch_size * 8 * N
    global_batch_size = int(
        percore_batch_size * num_local_devices * jax.process_count() + 1e-6
    )
    if percore_batch_size >= 1:
      assert global_batch_size % num_local_devices == 0
      batch_size_per_process = int(
          math.ceil(percore_batch_size) * num_local_devices + 1e-6
      )
      num_infeed_hosts = global_batch_size // batch_size_per_process
    else:
      if jax.process_count() > 1:
        # assert global_batch_size % num_local_devices == 0  # XD: bug?
        # batch_size_per_process = num_local_devices  # XD: bug?
        batch_size_per_process = int(
            percore_batch_size * num_local_devices + 1e-6
        )
        # N hosts
        num_infeed_hosts = global_batch_size // batch_size_per_process
      else:
        batch_size_per_process = int(
            percore_batch_size * num_local_devices + 1e-6
        )
        num_infeed_hosts = 1
    # batch_size_per_process, num_infeed_hosts = 4, 2  # XD
    seed = None
    if is_training:
      seed = self.TRAINING_SEED
      # TODO(sgpyc): enable sync of seeds across hosts, currently the
      # following failed because of "sync_global_devices name mismatch"
      # seed = jnp.int32(multihost_utils.broadcast_one_to_all(seed))
      logging.info('Train input seed: %s',
                   'None' if seed is None else seed)
    p = pax_fiddle.Config(
        seqio_input.SeqIOInput,
        name='C4Train' if is_training else 'C4Validation',
        mixture_name='c4_lm_v301_gpt'
        if is_training
        else 'c4_lm_v301_gpt_eval_tokenized',
        # split_name='train2' if is_training else 'validation_tokenized_5662seqs',
        split_name='train' if is_training else 'validation',  # XD for 3.0.1
        # mixture_name='rotten_tomatoes_lm_gpt', split_name='train',  # XD
        task_feature_lengths={'targets': self.MAX_SEQ_LEN},
        use_cached=False,
        repeat=True if is_training else False,
        feature_converter=seqio_input.LanguageModelFeatures(
            pack=True if is_training else False,
            use_custom_packing_ops=False,
            bos_id=0,
            reverse_bos_padding=True,
            eos_id=GPT_EOS_ID,
        ),
        is_training=is_training,
        input_random_seed=(seed if is_training else 4321),
        batch_size=batch_size_per_process,
        drop_remainder=True if is_training else False,
        num_batches_to_skip=self.TRAINING_NUM_BATCHES_TO_SKIP,
        num_infeed_hosts=num_infeed_hosts,
        # reset_for_eval=False if is_training else True, # eval的时候为True
        reset_for_eval=False, # eval的时候为True -> False
        annotate_padding_fields=True,
    )
    return p
    
  # lsp: 数据
  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns a list of dataset parameters."""
    return [
        self._dataset_common(is_training=True),
        self._dataset_common(is_training=False)
    ]

def set_adam_and_learning_rate_schedule(
    cls,
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
) -> pax_fiddle.Config[tasks_lib.SingleTask]:
  """Sets the Adam optimizer and the learning rate schedule."""
  lp = task_p.train.learner
  lp.loss_name = 'total_loss'
  lp.optimizer = pax_fiddle.Config(
      optimizers.Adam,
      beta1=cls.ADAM_BETA1 if cls.ADAM_BETA1 else 0.9,
      beta2=cls.ADAM_BETA2 if cls.ADAM_BETA2 else 0.999,
      # weight_decay=cls.WEIGHT_DECAY if cls.WEIGHT_DECAY else 0.0, # lsp: DEPRECATION
      l2_regularizer_weight=cls.WEIGHT_DECAY if cls.WEIGHT_DECAY else 0.0,
      epsilon=cls.ADAM_EPSILON if cls.ADAM_EPSILON else 1e-6,
      epsilon_root=cls.ADAM_EPSILON_ROOT if cls.ADAM_EPSILON_ROOT else 0.0,
      clip_gradient_norm_to_value=cls.CLIP_GRADIENT_NORM_TO_VALUE
      if cls.CLIP_GRADIENT_NORM_TO_VALUE
      else 5.0,
      clip_threshold=cls.ADAM_CLIP_THRESHOLD
      if cls.ADAM_CLIP_THRESHOLD
      else 1.0,
  )

  if hasattr(cls, 'PERCORE_BATCH_SIZE'):
    global_batch_size = int(cls.PERCORE_BATCH_SIZE * jax.device_count() + 1e-6)
    if global_batch_size == 0:
      logging.warning(
          (
              'Found global_batch_size = 0: cls.PERCORE_BATCH_SIZE=%s,'
              ' jax.device_count()=%s'
          ),
          cls.PERCORE_BATCH_SIZE,
          jax.device_count(),
      )
    assert global_batch_size <= 8192
  else:
    global_batch_size = None

  if cls.LEARNING_RATE is not None:
    lp.optimizer.learning_rate = cls.LEARNING_RATE
  else:
    assert global_batch_size is not None
    if global_batch_size <= 3584:
      lp.optimizer.learning_rate = 2e-5
    else:
      lp.optimizer.learning_rate = 3e-5

  if cls.LR_SCHEDULE == 'linear_rampup_exponential_decay':
    lp.optimizer.lr_schedule = pax_fiddle.Config(
        schedules.LinearRampupExponentialDecay,
        warmup_steps=cls.LR_LRED_WARMUP,
        decay_start=cls.LR_LRED_DECAY_START,
        decay_end=cls.LR_LRED_DECAY_END,
        min_ratio=cls.LR_LRED_MIN_RATIO,
        max=cls.LR_LRED_MAX,
    )
  elif cls.LR_SCHEDULE == 'linear_rampup_cosine_decay':
    if cls.LR_COS_WARMUP is not None:
      warmup_steps = cls.LR_COS_WARMUP
    else:
      assert global_batch_size is not None
      warmup_steps = math.ceil(265.0 * 1536 / global_batch_size - 1e-6)
      assert warmup_steps > 0

    if cls.LR_COS_DECAY_START is not None:
      decay_start_step = cls.LR_COS_DECAY_START
    else:
      decay_start_step = warmup_steps + 1

    if cls.LR_COS_DECAY_END is not None:
      decay_end_step = cls.LR_COS_DECAY_END
    else:
      assert global_batch_size is not None
      decay_end_step = math.ceil(108600.0 * 1536 / global_batch_size - 1e-6)
      assert decay_end_step > 0

    lp.optimizer.lr_schedule = pax_fiddle.Config(
        schedules.LinearRampupCosineDecay,
        warmup_steps=warmup_steps,
        decay_start=decay_start_step,
        decay_end=decay_end_step,
        min_ratio=cls.LR_COS_MIN_RATIO,
        max=cls.LR_COS_MAX,
    )
  else:
    raise NotImplementedError(
        f'Learning rate schedule {cls.LR_SCHEDULE} is not supported.'
    )

  return task_p


class TransformerLmSpmdAdam(model_params.TransformerLmSpmdAdafactor):
  """Base SPMD Transformer LM configuration using Adam.

  Only things different from TransformerLmSpmdAdafactor are listed.
  """
  # architecture related
  NUM_LAYERS = 32
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.float32
  PACKED_INPUT = True
  USE_BIAS = False
  EMBEDDING_LOOKUP_STYLE = 'matmul'

  # optimizer related
  LEARNING_RATE = 2e-4  # XD: 1e-3
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95  # XD: 0.99
  ADAM_CLIP_THRESHOLD = 1.0
  ADAM_EPSILON = 1e-8  # XD: -6
  ADAM_EPSILON_ROOT = 0.0

  # Learning rate schedule
  LR_SCHEDULE = 'linear_rampup_cosine_decay'  # XD: exponential
  LR_LRED_WARMUP = 4000
  LR_LRED_DECAY_START = 4001
  LR_LRED_DECAY_END = 300000
  LR_LRED_MIN_RATIO = 0.1
  LR_LRED_MAX = 1.0

  LR_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0
  # XD
  LR_COS_WARMUP = 2000
  LR_COS_DECAY_START = 2001
  LR_COS_DECAY_END = 200000
  # LR_COS_WARMUP = 4000
  # LR_COS_DECAY_START = 4001
  # LR_COS_DECAY_END = 300000

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT  # pytype: disable=attribute-error  # enable-nested-classes
    # stacked_p: StackedTransformerRepeated
    stacked_p = model_p.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
    if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if self.USE_REPEATED_LAYER:
      # stacked_p.block: StackedTransformer
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p


class TransformerLmSpmdPipelineAdam(
    model_params.TransformerLmSpmdPipelineAdafactor
):
  """Base pipelined SPMD Transformer LM configuration using Adam.

  Only things different from TransformerLmSpmdPipelineAdafactor are listed.
  """
  # architecture related
  NUM_LAYERS = 32
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.float32
  PACKED_INPUT = True
  USE_BIAS = False
  EMBEDDING_LOOKUP_STYLE = 'matmul'

  # optimizer related
  LEARNING_RATE = 1e-3
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.99
  ADAM_CLIP_THRESHOLD = 1.0
  ADAM_EPSILON = 1e-6
  ADAM_EPSILON_ROOT = 0.0

  # Learning rate schedule
  LR_SCHEDULE = 'linear_rampup_exponential_decay'
  LR_LRED_WARMUP = 4000
  LR_LRED_DECAY_START = 4001
  LR_LRED_DECAY_END = 300000
  LR_LRED_MIN_RATIO = 0.1
  LR_LRED_MAX = 1.0

  LR_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0
  LR_COS_WARMUP = 4000
  LR_COS_DECAY_START = 4001
  LR_COS_DECAY_END = 300000

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT  # pytype: disable=attribute-error  # enable-nested-classes

    stacked_p = model_p.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
    if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if self.USE_REPEATED_LAYER:
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p


@experiment_registry.register
class LmCloudSpmdAdam(TransformerLmSpmdAdam, lm_cloud.SyntheticDataset):
  """Base config for an SPMD model."""
  NUM_LAYERS = 2
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 2]


@experiment_registry.register
class LmCloudSpmdAdamLimitSteps(LmCloudSpmdAdam):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 4000
    return task_p


class EarlyStoppingFn(base_hyperparams.FiddleBaseParameterizable):
  r"""Early stopping function to log eval log_pplx and stop when reaching target.

  Attributes:
    target_log_pplx: target log pplx value to stop training when eval log pplx
      reaches this value.
  """

  target_log_pplx: Optional[float] = None

  def __call__(
      self,
      metrics: Dict[str, float],
      running_mode: trainer_lib.RunningMode,
      global_step: int,
      is_last_ckpt: bool,
  ) -> bool:
    """Returns True if run should be stopped early."""
    if 'eval_test_C4Validation/metrics/log_pplx' not in metrics.keys():
      return False
    log_pplx = metrics['eval_test_C4Validation/metrics/log_pplx']

    if log_pplx <= self.target_log_pplx:
      return True
    return False


def configure_gpt3_task(
    cls,
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
) -> pax_fiddle.Config[tasks_lib.SingleTask]:
  """Returns task with gpt3 related configs."""
  # model: LauguageModel
  model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes

  model_p.decoder_tpl.eos_id = (
      GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
  )
  model_p.decoder_tpl.seqlen = cls.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes
  #lsp: 每个参数都是WeightHParams类，该类有个init函数，调用params_init。没有指定的默认WeightInit.Xavier(_DEFAULT_XAVIER_INIT)， _DEFAULT_XAVIER_INIT： 1.000001
  model_p.params_init = WeightInit.Gaussian(0.006)

  softmax_init = WeightInit.Gaussian(0.006)
  #lsp: lm_tpl: TransformerLM
  model_p.lm_tpl.softmax_tpl.params_init = softmax_init
  model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False # 
  model_p.lm_tpl.softmax_tpl.soft_cap_logits = None
  # lsp: True
  if cls.SEPARATE_EMBEDDING:
    # scale_sqrt_depth is true时，会对embedding进行scale
    model_p.lm_tpl.separate_embedding_tpl.scale_sqrt_depth = False
    # matmul
    model_p.lm_tpl.separate_embedding_tpl.lookup_style = (cls.EMBEDDING_LOOKUP_STYLE) 
  else:
    model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = False
    model_p.lm_tpl.softmax_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE # matmul
  if cls.TRAINABLE_POSITION_EMB:
    model_p.lm_tpl.position_emb_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE

  # lsp: 设置transformer的属性
  stacked_p = model_p.lm_tpl.stacked_transformer_tpl
  if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
    stacked_p = stacked_p.pipeline_stage
  # issubclass： 判断stacked_p是否继承了
  # fdl.get_callable(stacked_p)其实就是stacked_p未被pax_fiddle.Config(stacked_p)之前的类
  # lsp: stacked_p 继承 StackedTransformerRepeated为True
  if issubclass(
      fdl.get_callable(stacked_p), transformers.StackedTransformerRepeated
  ):
   # 如果stacked_p继承了StackedTransformerRepeated，那么stacked_p.block应该是StackedTransformerRepeated
   # 然后去设置block的属性
    stacked_p = stacked_p.block
  transformer_layer_p = stacked_p.transformer_layer_params_tpl
  # lsp: layer_norm
  transformer_layer_p.ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
  transformer_layer_p.tr_fflayer_tpl.ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
  model_p.lm_tpl.final_ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
  if cls.NORMALIZATION_CLS == normalizations.RmsNorm:  # XD
    transformer_layer_p.ln_tpl.intermediate_dtype = jnp.float32
    transformer_layer_p.tr_fflayer_tpl.ln_tpl.intermediate_dtype = jnp.float32
    model_p.lm_tpl.final_ln_tpl.intermediate_dtype = jnp.float32
  if True or cls.NORMALIZATION_CLS == normalizations.LayerNorm:  # XD
    transformer_layer_p.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
    transformer_layer_p.tr_fflayer_tpl.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
    model_p.lm_tpl.final_ln_tpl.epsilon = cls.LAYERNORM_EPSILON
  # lsp: tr_atten_tpl有可能为None，这样设置有问题把？
  transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
  transformer_layer_p.tr_atten_tpl.use_bias = cls.USE_BIAS  # XD: True

  transformer_layer_p.tr_fflayer_tpl.has_bias = not cls.USE_GATED_ACTIVATION or cls.USE_BIAS  # XD add
  if cls.ACTIVATION_CLS == layers.GELU: transformer_layer_p.tr_fflayer_tpl.activation_tpl.approximate = True  # XD: add if

  for atten_p in (
      transformer_layer_p.tr_atten_tpl,
      transformer_layer_p.cross_atten_tpl,
  ):
    if atten_p is None:
      continue
    atten_wp = atten_p.weight_split_dims_mapping
    atten_wp.proj = ['data', 'mdl', None]

  if task_p.early_stopping_fn is None:
    # lsp: EarlyStoppingFn: 当ppl变大的时候则停止
    task_p.early_stopping_fn = pax_fiddle.Config(EarlyStoppingFn)
    task_p.early_stopping_fn.target_log_pplx = cls.TARGET_LOG_PPLX

  return task_p


@experiment_registry.register
class C4SpmdAdam(TransformerLmSpmdAdam,
                 C4UnsupervisedDataset):
  r"""Base config for a decoder only transformer."""
  # VOCAB_SIZE = 50320  # XD: GPT2Tokenizer.vocab_size = 50257
  NUM_LAYERS = 24
  NUM_HEADS = 32
  MODEL_DIMS = 2048
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 32128
  ACTIVATION_CLS = layers.SiLU  # XD: GELU, SiLU
  USE_GATED_ACTIVATION = True  # XD: False

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B
  CHECKPOINT_EVERY_N_STEPS = 1000

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 2]

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.eos_id = GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.seqlen = self.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)
    return task_p


class C4SpmdGpt3AdamOrgHP(C4SpmdAdam):
  r"""GPT-3 config with original HPs.

  From the paper & after convergence matching with
  NVIDIA's Megatron-LM framework.
  """
  MAX_SEQ_LEN = 2048

  NUM_LAYERS = 96
  NUM_HEADS = 96
  MODEL_DIMS = 12288
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 50257
  USE_REPEATED_LAYER = True

  # Model configs
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  ATTEN_LOGIT_CAP = -1.0  # Disable logits cap in atten

  # HPs
  LEARNING_RATE = 6e-5
  WEIGHT_DECAY = 0.1
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_CLIP_THRESHOLD = -1.0  # Disable Adam clip_threshold
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  LAYERNORM_EPSILON = 1e-5

  # In units of steps for BS1.5k
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 265
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 108600
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  # Training target
  TARGET_LOG_PPLX = 2.69

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p = configure_gpt3_task(self, task_p)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamOrgHPBS1p5k1536Replicas(C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config in fp32 for 1536 replicas with 1536 global batch size."""
  # Padded to TPU friendly size
  VOCAB_SIZE = 51200

  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 64, 24]
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 100
  EVAL_INTERVAL_STEPS = 25
  SUMMARY_INTERVAL_STEPS = 1

@experiment_registry.register
class C4SpmdGpt3SmallRoPE(C4SpmdGpt3AdamOrgHP):  # XD
  r"""small GPT-3 config with RoPE.
  """
  VOCAB_SIZE = 32000  # XD
  NUM_LAYERS = 12
  MODEL_DIMS = 768
  ACTIVATION_CLS = layers.SiLU  # layers.SiLU/GELU  # XD
  USE_GATED_ACTIVATION = True  # XD
  HIDDEN_DIMS = MODEL_DIMS * 4 # 2048  # XD: MODEL_DIMS * 4
  NUM_HEADS = 12
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  USE_BIAS = False # XD add
  # NORMALIZATION_CLS = normalizations.LayerNorm  # XD add RmsNorm
  NORMALIZATION_CLS = normalizations.RmsNorm  # XD add RmsNorm

  LEARNING_RATE = 2e-4  # XD
  PERCORE_BATCH_SIZE = 4
  FPROP_DTYPE = jnp.bfloat16

  ICI_MESH_SHAPE = [1, 8, 4]

  SEPARATE_EMBEDDING = True  # XD
  USE_ROTARY_POSITION_EMB = False
  TRAINABLE_PE_MAX_SEQ_LEN = 2048  # XD add
  # lsp: 调用这个task
  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    # lsp: 位置向量在这里设置为None了
    if True or self.USE_ROTARY_POSITION_EMB: task_p.model.lm_tpl.position_emb_tpl = None  # XD: add if
    return task_p

@experiment_registry.register
class C4SpmdGpt37BRoPE(C4SpmdGpt3SmallRoPE):  # XD
  NUM_LAYERS = 2
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 32
  # DIMS_PER_HEAD = 128
  COMBINE_QKV = False # False 占用显存小于 True 1G+
  NUM_GROUPS = -1
  
  PERCORE_BATCH_SIZE = 2
  # ICI_MESH_SHAPE = [4, 1, 8]  # bs=2*8, 0.146, combine_qkv 0.1514 
  # ICI_MESH_SHAPE = [1, 8, 4]  # bs=8*8, 0.176, combine_qkv 0.180
  # ICI_MESH_SHAPE = [1, 16, 1] # 16 * 1 * 16 * 1 oom: 30M, combine_qkv: False
  # ICI_MESH_SHAPE = [1, 16, 1] # 8 * 1 * 16 * 1 combine_qkv: True, 0.138 * 2
  # ICI_MESH_SHAPE = [1, 16, 1] # 16 * 1 * 16 * 1 combine_qkv: True, 
  ICI_MESH_SHAPE = [1, 8, 1] # v5最后一个维度不能为2？，必须为1？

  VOCAB_SIZE = 64000
  CHECKPOINT_EVERY_N_STEPS = 20
  TRAIN_FILE = "gs://jax_llm_data/data-baichuan/dreamily_translation_general.train.tfrecords"
  VALID_FILE = "gs://jax_llm_data/data-baichuan/dreamily_translation_general.test.tfrecords"
    # lsp
  def _dataset_common(
      self, is_training
  ) -> pax_fiddle.Config[base_input.BaseInput]:
    path = self.TRAIN_FILE if is_training  else self.VALID_FILE
    p = pax_fiddle.Config(
        MyDatasets, 
        path=path,
        is_training=is_training,
    )
    return p

@experiment_registry.register
class C4SpmdGpt3MediumRoPE(C4SpmdGpt3SmallRoPE):  # XD
  NUM_LAYERS = 24
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4  # 11008  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 16
  
  # LEARNING_RATE = 6e-5
  PERCORE_BATCH_SIZE = 4
  # ICI_MESH_SHAPE = [1, 8*4, 4//4]
  ICI_MESH_SHAPE = [32, 1, 1]

@experiment_registry.register
class C4SpmdGpt3XLRoPE(C4SpmdGpt3SmallRoPE):  # XD
  NUM_LAYERS = 24
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4  # 11008  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 16
  # DIMS_PER_HEAD = 128
  
  PERCORE_BATCH_SIZE = 4
  ICI_MESH_SHAPE = [1, 8, 4]

@experiment_registry.register
class C4SpmdPipelineAdam(TransformerLmSpmdPipelineAdam, C4UnsupervisedDataset):
  r"""Base config for a decoder only transformer with pipeline."""
  NUM_LAYERS = 24
  NUM_HEADS = 32
  MODEL_DIMS = 2048
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 32128
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B
  CHECKPOINT_EVERY_N_STEPS = 1000

  # Sub-class has to specify a mesh.
  MICROBATCH_SIZE = 2
  ICI_MESH_SHAPE = [2, 1, 2, 2]
  NUM_STAGES = 2
  EMB_W_DATA_DIMS = ('replica', 'data')

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.eos_id = (
        GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
    )
    model_p.decoder_tpl.seqlen = self.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p


class C4SpmdPipelineGpt3AdamOrgHP(C4SpmdPipelineAdam):
  r"""GPT-3 config with original HPs.

  From the paper & after convergence matching with
  NVIDIA's Megatron-LM framework.
  """
  MAX_SEQ_LEN = 2048

  NUM_LAYERS = 96
  NUM_HEADS = 96
  MODEL_DIMS = 12288
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 50257
  USE_REPEATED_LAYER = False

  # Model configs
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  ATTEN_LOGIT_CAP = -1.0  # Disable logits cap in atten

  # HPs
  LEARNING_RATE = 6e-5
  WEIGHT_DECAY = 0.1
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_CLIP_THRESHOLD = -1.0  # Disable Adam clip_threshold
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  LAYERNORM_EPSILON = 1e-5

  # In units of steps for BS1.5k
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 265
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 108600
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  # Training target
  TARGET_LOG_PPLX = 2.69

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = super().task()
    task_p = configure_gpt3_task(self, task_p)
    return task_p


class C4SpmdPipelineGpt3AdamMLPerfHP(C4SpmdPipelineGpt3AdamOrgHP):
  r"""GPT-3 config for MLPerf reference."""
  # Padded to TPU friendly size
  VOCAB_SIZE = 51200
  FPROP_DTYPE = jnp.float32
  SUMMARY_INTERVAL_STEPS = 1
  # subclass must set the eval and the checkpoint intervals
  EVAL_INTERVAL_STEPS = None
  CHECKPOINT_EVERY_N_STEPS = None
  CHECKPOINT_MAX_TO_KEEP = 100

  # Let set_adam_and_learning_rate_schedule calculate the following HPs
  # based on global batch size
  LEARNING_RATE = None
  LR_COS_WARMUP = None
  LR_COS_DECAY_START = None
  LR_COS_DECAY_END = None


@experiment_registry.register
class C4SpmdPipelineGpt3AdamOrgHPBS1p5k768Replicas(C4SpmdPipelineGpt3AdamOrgHP):
  r"""GPT-3 config in fp32 for 768 replicas with 1536 global batch size.

  Using the orininal HP set.
  """
  PERCORE_BATCH_SIZE = 2
  VOCAB_SIZE = 51200
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 12]
  # NUM_MICROBATCHS = 192
  MICROBATCH_SIAZE = 8
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 100
  EVAL_INTERVAL_STEPS = 25
  SUMMARY_INTERVAL_STEPS = 1
  CHECKPOINT_EVERY_N_STEPS = 50
  STREAM_IO = False


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS1p5k768Replicas(
    C4SpmdPipelineGpt3AdamMLPerfHP
):
  r"""GPT-3 config in fp32 for 768 replicas with 1536 global batch size.

  Following MLPerf training benchmarking HP requirements.
  """
  PERCORE_BATCH_SIZE = 2
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 12]
  # NUM_MICROBATCHS = 192
  MICROBATCH_SIZE = 8
  EVAL_INTERVAL_STEPS = 16
  CHECKPOINT_EVERY_N_STEPS = EVAL_INTERVAL_STEPS * 2
  STREAM_IO = False


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS2k512Replicas(
    C4SpmdPipelineGpt3AdamMLPerfHP
):
  r"""GPT-3 config in fp32 for 512 replicas with 2k global batch size.

  Following MLPerf training benchmarking HP requirements.
  """
  PERCORE_BATCH_SIZE = 4
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 8]
  # NUM_MICROBATCHS = 256
  MICROBATCH_SIZE = 8
  EVAL_INTERVAL_STEPS = 12
  CHECKPOINT_EVERY_N_STEPS = EVAL_INTERVAL_STEPS * 2
  STREAM_IO = True


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS3k768Replicas(
    C4SpmdPipelineGpt3AdamMLPerfHP
):
  r"""GPT-3 config in fp32 for 768 replicas with 3072 global batch size.

  Following MLPerf benchmarking HP requirements.
  """
  PERCORE_BATCH_SIZE = 4
  NUM_STAGES = 4
  ICI_MESH_SHAPE = [4, 1, 16, 12]
  # NUM_MICROBATCHS = 192
  MICROBATCH_SIZE = 16
  EVAL_INTERVAL_STEPS = 8
  CHECKPOINT_EVERY_N_STEPS = EVAL_INTERVAL_STEPS * 2
  STREAM_IO = True


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS4k1024Replicas(
    C4SpmdPipelineGpt3AdamMLPerfHP
):
  r"""GPT-3 config in fp32 for 1024 replicas with 4096 global batch size.

  Following MLPerf benchmarking HP requirements.
  """
  PERCORE_BATCH_SIZE = 4
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 16]
  # NUM_MICROBATCHS = 512
  MICROBATCH_SIZE = 8
  EVAL_INTERVAL_STEPS = 6
  CHECKPOINT_EVERY_N_STEPS = EVAL_INTERVAL_STEPS * 2
  STREAM_IO = True


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS8k1024Replicas(
    C4SpmdPipelineGpt3AdamMLPerfHP
):
  r"""GPT-3 config in fp32 for 1024 replicas with 8192 global batch size.

  Following MLPerf benchmarking HP requirements.
  """
  PERCORE_BATCH_SIZE = 8
  NUM_STAGES = 4
  ICI_MESH_SHAPE = [4, 1, 16, 16]
  # NUM_MICROBATCHS = 512
  MICROBATCH_SIZE = 16
  EVAL_INTERVAL_STEPS = 3
  CHECKPOINT_EVERY_N_STEPS = EVAL_INTERVAL_STEPS * 2
  STREAM_IO = True


@experiment_registry.register
class C4Spmd1BAdam4Replicas(C4SpmdAdam):
  r"""GPT-3 config with 1B params.

  Model Parameters:  Global batch size = 1 * 4 * 1 * 32 = 128
  """
  NUM_LAYERS = 13
  MODEL_DIMS = 2560
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 20
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 32
  MAX_SEQ_LEN = 1024
  # VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 4, 1]


@experiment_registry.register
class C4Spmd1BAdam4ReplicasLimitSteps(C4Spmd1BAdam4Replicas):

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    task_p.train.num_train_steps = 15000
    return task_p


@experiment_registry.register
class C4Spmd2BAdam4Replicas(C4SpmdAdam):
  r"""GPT-3 config with 2B params.

  Model Parameters: Global batch size = 1 * 4 * 1 * 32 = 128.
  """
  NUM_LAYERS = 18
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 24
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 32
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 4, 1]

@experiment_registry.register
class C4Spmd2BAdam32Replicas(C4SpmdAdam):  # XD
  r"""
  Model Parameters: Global batch size = 1 * 8 * 4 * 8 = 256.
  """
  NUM_LAYERS = 18
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 24
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 8
  MAX_SEQ_LEN = 1024 * 2  # XD
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 8, 4]
  
@experiment_registry.register
class C4Spmd2BAdam32x2Replicas(C4SpmdAdam):  # XD
  r"""
  Model Parameters: Global batch size = 1 * 16 * 2 * 16 = 512.
  """
  NUM_LAYERS = 18
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 24
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 16
  MAX_SEQ_LEN = 1024 * 2  # XD
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 2]

@experiment_registry.register
class C4SpmdLLaMA7BAdam32Replicas(C4SpmdAdam):  # XD
  r"""
  Model Parameters: Global batch size = 4 * 1 * 8 * 1 / 8 = 4.
  """
  NUM_LAYERS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 32
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 1  # 4
  MAX_SEQ_LEN = 2048  # XD
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  # ICI_MESH_SHAPE = [1, 8, 4]
  ICI_MESH_SHAPE = [4, 1, 8]

  # def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
  #   task_p = super().task()
  #   task_p.train.num_train_steps = 30
  #   return task_p

@experiment_registry.register
class C4SpmdLLaMA1BAdam32Replicas(C4SpmdLLaMA7BAdam32Replicas):  # XD
  r"""
  Model Parameters: Global batch size = 4 * 1 * 8 * 1 / 8 = 4.
  """
  NUM_LAYERS = 24
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4  # 5504  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 16
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 8  # 4
  COMBINE_QKV = False

  # ICI_MESH_SHAPE = [1, 8, 4]
  ICI_MESH_SHAPE = [16, 1, 2]

@experiment_registry.register
class C4Spmd16BAdam32Replicas(C4SpmdAdam):
  r"""GPT-3 config with 16B params.

  Model Parameters: Global batch size = 1 * 2 * 16 * 16 = 512.
  """
  NUM_LAYERS = 36
  MODEL_DIMS = 6144
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 48
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 16 #// 8 # XD: v4->v3
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 2]


@experiment_registry.register
class C4Spmd32BAdam64Replicas(C4SpmdAdam):
  r"""GPT-3 config with 32B params.

  Model Parameters: Global batch size = 1 * 16 * 4 * 8 = 512.
  """
  NUM_LAYERS = 40
  MODEL_DIMS = 8192
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 64
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 8
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000  # XD
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 4]


@experiment_registry.register
class C4SpmdGpt3L16AdamOrgHP(C4SpmdGpt3AdamOrgHP):
  r"""Small GPT-3 config in bf16 for 64 replicas with 192 global batch size."""
  NUM_LAYERS = 16
  FPROP_DTYPE = jnp.bfloat16
  PERCORE_BATCH_SIZE = 3
  EVAL_INTERVAL_STEPS = 25000
  ICI_MESH_SHAPE = [1, 16, 4]


@experiment_registry.register
class C4SpmdPipelineGpt3SmallAdam8Replicas(C4SpmdPipelineGpt3AdamOrgHP):
  """Small GPT-3 config in bf16 for 8 replicas with 512 global batch size.

  This was called GPT-3 XL in the GPT-3 paper, with 1.3B parameters.
  """

  NUM_STAGES = 2
  NUM_LAYERS = 24
  NUM_HEADS = 24
  MODEL_DIMS = 3072
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = 128
  VOCAB_SIZE = 51200

  PERCORE_BATCH_SIZE = 64
  MICROBATCH_SIZE = 8
  FPROP_DTYPE = jnp.bfloat16
  LEARNING_RATE = 2.0e-4
  ICI_MESH_SHAPE = [2, 1, 2, 2]

  CHECKPOINT_MAX_TO_KEEP = 1000
  EVAL_INTERVAL_STEPS = 10
  SUMMARY_INTERVAL_STEPS = 5
  CHECKPOINT_EVERY_N_STEPS = 200
