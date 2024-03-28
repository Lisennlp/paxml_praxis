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
import numpy as np
import operator  # XD

from absl import logging
import fiddle as fdl
import tensorflow as tf
import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import experiment_registry
from paxml import seqio_input
from paxml import tasks_lib
from paxml import trainer_lib
from paxml.tasks.lm import model_params
# from paxml.tasks.lm.params import lm_cloud  # XD
from praxis import base_hyperparams
from praxis import base_input
from praxis import base_layer
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis import py_utils
from praxis.layers import normalizations  # XD
from praxis.layers import activations  # XD
from praxis.layers import transformers
import seqio
import t5.data
from t5.data import preprocessors as t5_preprocessors
from paxml.tasks.lm.params import global_cfg  # XD
from paxml.utils import c4_registry, tfids_registry, extract_pythia_datapath, extract_train_skip_step # XD fix

WeightInit = base_layer.WeightInit
NestedMap = py_utils.NestedMap

GPT_SPM_PATH = global_cfg.GPT_SPM_PATH 
GPT_EOS_ID = 1
GPT_VOCABULARY = t5.data.SentencePieceVocabulary(GPT_SPM_PATH)
PASS_THROUGH_VOCABULARY = t5.data.PassThroughVocabulary(size=50257)

C4_GPT_TRAIN_FEATURES_LM = {
    'targets': t5.data.Feature(vocabulary=GPT_VOCABULARY, add_eos=False)
}
C4_GPT_EVAL_FEATURES_LM = {
    'targets': t5.data.Feature(
        vocabulary=PASS_THROUGH_VOCABULARY, add_eos=False
    )
}
C4_TRAIN_DATADIR = global_cfg.C4_TRAIN_DATADIR # XD 'gs://common_datasets'  # XD: 'gs://mlperf-llm-public2'
C4_EVAL_DATADIR = global_cfg.C4_EVAL_DATADIR # XD 'gs://common_datasets' # XD: 'gs://mlperf-llm-public2'

# XD
# import tensorflow as tf
# RT_DATAPATH = 'gs://llm_projects/data/rotten_tomatoes_train_8530.tfrecords'
# feature_desc = {"input_ids": tf.io.VarLenFeature(tf.int64)}
# RT_GPT_FEATURES_LM = {'targets': seqio.Feature(vocabulary=PASS_THROUGH_VOCABULARY, dtype=tf.int32, rank=1)}

# @seqio.map_over_dataset
# def convert_datatype(ex):
#   return {k: tf.cast(tf.sparse.to_dense(v, default_value=0), dtype=tf.int32) for k, v in ex.items()}

# seqio.TaskRegistry.add('rotten_tomatoes_lm_gpt', 
#     seqio.TFExampleDataSource(split_to_filepattern={'train': RT_DATAPATH}, feature_description=feature_desc),
#     preprocessors=[
#         convert_datatype,
#         functools.partial(t5_preprocessors.rekey, key_map={'targets': 'input_ids'}),
#     ],
#     output_features=RT_GPT_FEATURES_LM,
# )

class TaskRegistry(t5.data.TaskRegistry):
  """Task registry with extra tracking."""

  TASK_NAMES = []

  @classmethod
  def add_versioned_tfds_task(cls,
                              name: str,
                              *,
                              versions: List[str],
                              pinned_version: Optional[str] = None,
                              tfds_name: str,
                              tfds_data_dir: Optional[str] = None,
                              **kwargs) -> List[seqio.Task]:
    tasks = []
    for version in versions:
      tasks.append(
          cls.add(
              f'{name}_{version}',
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    if pinned_version is not None:
      tasks.append(
          cls.add(
              name,
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{pinned_version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    return tasks


# C4 corpus for language model pretraining
TaskRegistry.add_versioned_tfds_task(
    'c4_lm_v301_gpt',
    versions=['3.0.1'],  # XD: 3.0.4 -> 3.0.1
    pinned_version='3.0.1',  # XD: 3.0.4 -> 3.0.1
    tfds_name='c4/en',
    tfds_data_dir=C4_TRAIN_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                'inputs': None,
                'targets': 'text',
            },
        ),
        seqio.preprocessors.tokenize,
        functools.partial(
            t5_preprocessors.reduce_concat_tokens,
            batch_size=4096,
        ),
        t5_preprocessors.split_tokens_to_targets_length,
    ],
    output_features=C4_GPT_TRAIN_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=10000,
)

TaskRegistry.add_versioned_tfds_task(
    'c4_lm_v301_gpt_eval_tokenized',
    versions=['3.0.1'],  # XD: 3.0.5 -> 3.0.1
    pinned_version='3.0.1',  # XD: 3.0.5 -> 3.0.1
    tfds_name='c4/en',
    tfds_data_dir=C4_EVAL_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                'inputs': None,
                'targets': 'ids',
            },
        ),
        seqio.preprocessors.tokenize,
    ],
    output_features=C4_GPT_EVAL_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=None,
)

# lsp
@experiment_registry.register
class DataParams():
    LOAD_SEQIO_ID = False
    LOAD_SEQIO_TEXT = True
    TRAINING_NUM_BATCHES_TO_SKIP = None
    TEST_BATCH_RATIO = 0.02
    SHUFFLE = {"train": True, "test": False}
    SHUFFLE_SIZE = 10000
    MAX_SEQ_LEN = 2048
    # default
    KEY_MAP = {"inputs": None, "targets": "text"}
    VOCAB_FILE = 'gs://common_datasets/vocab/c4_en_301_5Mexp_spm.model'
    VOCABULARY = t5.data.SentencePieceVocabulary(VOCAB_FILE)
    DATA_PATH = {
                'train': 'gs://common_datasets/', 
                'test':  'gs://common_datasets/', 
                }
    TASK_NAME = 'DataParams'
    DATA_FUNC = c4_registry

class PileDataParams(DataParams):
  EVAL_LOOP_NUM_BATCHES = 50
  MAX_SEQ_LEN = 2049
  VOCAB_SIZE = 50432
  LOAD_SEQIO_TEXT = False
  LOAD_SEQIO_ID = False
  SHUFFLE = {'train': False, 'test': False}
  VOCABULARY = t5.data.PassThroughVocabulary(size=VOCAB_SIZE)
  KEY_MAP = {"targets": "input_ids", "masks": "input_ids"}
  DATA_PATH = {
              'train': 'gs://common_datasets/pythia_pile_idxmaps_tfrecord',
              'test':  'gs://common_datasets/pythia_pile_idxmaps_tfrecord',
              }
  DATA_FUNC = extract_pythia_datapath

class SeqioPileDataParams(PileDataParams):
  LOAD_SEQIO_ID = False
  TASK_NAME = 'pythia'
  DATA_FUNC = tfids_registry

class C4UnsupervisedDataset(base_experiment.BaseExperiment):
  """Used for training Baseline ULM."""
  PERCORE_BATCH_SIZE = 1
  PERCORE_EVAL_BATCH_SIZE = None
  MAX_SEQ_LEN = 1024
  TRAINING_SEED = 9876
  TRAINING_NUM_BATCHES_TO_SKIP = None

  def _dataset_common(
      self, is_training,
      job_log_dir=None
  ) -> pax_fiddle.Config[base_input.BaseInput]:
    LOAD_SEQIO_ID = getattr(self, 'LOAD_SEQIO_ID', False)  # XD fix
    LOAD_SEQIO_TEXT = getattr(self, 'LOAD_SEQIO_TEXT', True)  # XD fix
    if not LOAD_SEQIO_ID and not LOAD_SEQIO_TEXT:  # XD fix
      meta_dict = extract_train_skip_step(job_log_dir=job_log_dir, step=self.TRAINING_NUM_BATCHES_TO_SKIP, only_eval=getattr(self, 'ONLY_EVAL', False))
      num_batches_to_skip = meta_dict.get('checkpoint_step', self.TRAINING_NUM_BATCHES_TO_SKIP)

    if is_training:
      percore_batch_size = self.PERCORE_BATCH_SIZE
    else:
      if self.PERCORE_EVAL_BATCH_SIZE is not None:
        percore_batch_size = self.PERCORE_EVAL_BATCH_SIZE
      else:
        percore_batch_size = self.PERCORE_BATCH_SIZE

    num_local_devices = jax.local_device_count()
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
        batch_size_per_process = int(
            percore_batch_size * num_local_devices + 1e-6
        )
        num_infeed_hosts = global_batch_size // batch_size_per_process
      else:
        batch_size_per_process = int(
            percore_batch_size * num_local_devices + 1e-6
        )
        num_infeed_hosts = 1
    seed = None
    if is_training:
      seed = self.TRAINING_SEED
      logging.info('Train input seed: %s',
                   'None' if seed is None else seed)
    # lsp
    if LOAD_SEQIO_ID:
        logging.info(f"Load seqio id data......")
        DataFeature = seqio_input.MyLanguageModelFeatures
        DataFeature.MAX_SEQ_LEN = self.MAX_SEQ_LEN
        # train test shuffle flag
        shuffle = self.SHUFFLE["train"] if is_training else self.SHUFFLE["test"]
        mixture_name = f"{self.TASK_NAME}.train" if is_training else f"{self.TASK_NAME}.test"
        name = "sft_train" if is_training else "sft_test"
        split_name = "train" if is_training else "test"
        task_feature_lengths = {
            "targets": self.MAX_SEQ_LEN,
            "masks": self.MAX_SEQ_LEN,
        }
        print(f"is_training: {is_training} shuffle: {shuffle}")
    elif LOAD_SEQIO_TEXT:
        logging.info(f"Load seqio text data......")
        DataFeature = seqio_input.LanguageModelFeatures
        mixture_name = "c4.train" if is_training else "c4.test"  # XD fix
        # mixture_name='c4_lm_v301_gpt'
        name = "C4Train" if is_training else "C4Validation"
        split_name = "train" if is_training else "validation"
        task_feature_lengths = {"targets": self.MAX_SEQ_LEN}
        shuffle = None
    else:
        logging.info(f"Load mesh id data......")
        shuffle_key = "train" if is_training else "test"
        shuffle_buffer_size = self.SHUFFLE_SIZE if self.SHUFFLE[shuffle_key] else None

    # print(f'is_training: {is_training}')  # XD fix
    mode='train' if is_training else 'test'
    DATA_PATH = self.DATA_FUNC(mode=mode) \
      if getattr(self, 'DATA_FUNC', None) is not None else None
    if LOAD_SEQIO_ID or LOAD_SEQIO_TEXT:
        # lsp
        num_batches_to_skip = getattr(self, 'TRAINING_NUM_BATCHES_TO_SKIP', None)  # XD fix
        assert DATA_PATH is None, print(f'Please check data params: “LOAD_SEQIO_TEXT“ or “LOAD_SEQIO_ID ” set...')
        p = pax_fiddle.Config(
            seqio_input.SeqIOInput,
            name=name,
            mixture_name=mixture_name,
            split_name=split_name,
            task_feature_lengths=task_feature_lengths,
            use_cached=False,
            shuffle=shuffle,
            repeat=True if is_training else False,
            feature_converter=DataFeature(
                pack=True if is_training else False,
                use_custom_packing_ops=False,
                bos_id=0,
                reverse_bos_padding=True,
                eos_id=GPT_EOS_ID,
            ),
            is_training=is_training,
            input_random_seed=(seed if is_training else 4321),
            # batch_size=int(batch_size_per_process),  # lsp
            batch_size=batch_size_per_process,  # XD fix
            drop_remainder=True if is_training else False,
            num_batches_to_skip=num_batches_to_skip,  # lsp: add skip batch step
            num_infeed_hosts=num_infeed_hosts,
            reset_for_eval=getattr(self, 'RESET_FOR_EVAL', False),  # eval的时候为True -> False  # XD fix ???
            annotate_padding_fields=True,
            eval_loop_num_batches=getattr(self, 'EVAL_LOOP_NUM_BATCHES', 1),  # XD fix
        )
    else:
        assert isinstance(DATA_PATH, dict)
        p = pax_fiddle.Config(
            MyDatasets,
            name=f"{self.TASK_NAME}.train" if is_training else f"{self.TASK_NAME}.test",
            path=DATA_PATH["train"] if is_training else DATA_PATH["test"],
            is_training=is_training,
            meta_dict=meta_dict,
            batch_size=int(self.PERCORE_BATCH_SIZE * num_local_devices),
            seq_len=self.MAX_SEQ_LEN,
            reset_for_eval=getattr(self, 'RESET_FOR_EVAL', False),
            repeat=1,
            eval_loop_num_batches=self.EVAL_LOOP_NUM_BATCHES,
            train_seed=self.TRAINING_SEED,
            task_features=list(self.KEY_MAP.values()),
            shuffle_buffer_size=shuffle_buffer_size,
            num_batches_to_skip=num_batches_to_skip,
            only_eval=getattr(self, 'ONLY_EVAL', False),
            zero_loss=getattr(self, 'ZERO_LOSS', True),
        )
    return p

  def datasets(self, job_log_dir=None) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns a list of dataset parameters."""
    return [
        self._dataset_common(is_training=True, job_log_dir=job_log_dir),
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
      weight_decay=cls.WEIGHT_DECAY if cls.WEIGHT_DECAY else 0.0,
      # decoupled_weight_decay=cls.WEIGHT_DECAY if cls.WEIGHT_DECAY else 0.0,  # XD: add decoupled_
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

    stacked_p = model_p.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
    if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if self.USE_REPEATED_LAYER:
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS
    if hasattr(self, 'USE_QK_BIAS'): transformer_layer_p.tr_atten_tpl.use_qk_bias = self.USE_QK_BIAS

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
class LmCloudSpmdAdam(TransformerLmSpmdAdam): # XD, lm_cloud.SyntheticDataset):
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
  model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
  if hasattr(model_p, 'data_full_shard'): model_p.data_full_shard = getattr(cls, 'DATA_FULL_SHARD', True)

  model_p.decoder_tpl.eos_id = (
      GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
  )
  model_p.decoder_tpl.seqlen = cls.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

  if getattr(cls, 'EXPLICIT_INIT', True):  # XD
    init_method = getattr(cls, 'INIT_METHOD', None)
    if isinstance(init_method, float): std = cls.INIT_METHOD
    elif init_method == 'small_init': std = math.sqrt(2 / (5 * cls.MODEL_DIMS))
    else: std = 0.006
    model_p.params_init = WeightInit.Gaussian(std)
    softmax_init = WeightInit.Gaussian(std)
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init
  else:  # XD: same as TransformerLmSpmdAdafactor
    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(cls.MODEL_DIMS))
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init
    assert getattr(cls, 'SCALE_EMBEDDING', False)
    if cls.SEPARATE_EMBEDDING:
      model_p.lm_tpl.separate_embedding_tpl.params_init = softmax_init
  model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False
  model_p.lm_tpl.softmax_tpl.soft_cap_logits = None
  if hasattr(cls, "LOSS_BATCH_MEAN"): model_p.lm_tpl.softmax_tpl.loss_batch_mean = cls.LOSS_BATCH_MEAN  # lsp

  if cls.SEPARATE_EMBEDDING:
    model_p.lm_tpl.separate_embedding_tpl.scale_sqrt_depth = getattr(cls, 'SCALE_EMBEDDING', False)  # XD: False
    model_p.lm_tpl.separate_embedding_tpl.lookup_style = (
        cls.EMBEDDING_LOOKUP_STYLE
    )
  else:
    model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = getattr(cls, 'SCALE_EMBEDDING', False)  # XD: False
    model_p.lm_tpl.softmax_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE
  if cls.TRAINABLE_POSITION_EMB:
    model_p.lm_tpl.position_emb_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE

  for prefix in (['early_'] if getattr(cls, 'NUM_EARLY_LAYERS', 0) else []) + ['']:  # XD
    # stacked_p = model_p.lm_tpl.stacked_transformer_tpl
    stacked_p = getattr(model_p.lm_tpl, prefix + 'stacked_transformer_tpl')
    if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if issubclass(
        fdl.get_callable(stacked_p), transformers.StackedTransformerRepeated
    ):
      stacked_p = stacked_p.block
    if hasattr(stacked_p, 'pre_compute_atten_mask'):
      stacked_p.pre_compute_atten_mask = getattr(cls, 'PRE_COMPUTE_ATTEN_MASK', True)
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
    transformer_layer_p.tr_fflayer_tpl.ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
    model_p.lm_tpl.final_ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
    if cls.NORMALIZATION_CLS == normalizations.RmsNorm:  # XD
      transformer_layer_p.ln_tpl.skip_weight_decay = cls.SKIP_RMSNORM_WD
      transformer_layer_p.tr_fflayer_tpl.ln_tpl.skip_weight_decay = cls.SKIP_RMSNORM_WD
      model_p.lm_tpl.final_ln_tpl.skip_weight_decay = cls.SKIP_RMSNORM_WD
    if cls.NORMALIZATION_CLS == normalizations.LayerNorm:  # XD
      transformer_layer_p.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
      transformer_layer_p.tr_fflayer_tpl.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
      model_p.lm_tpl.final_ln_tpl.epsilon = cls.LAYERNORM_EPSILON
    transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
    transformer_layer_p.tr_atten_tpl.use_bias = cls.USE_BIAS  # XD: True
    transformer_layer_p.tr_atten_tpl.num_kv_heads = getattr(cls, 'NUM_KV_HEADS', None)  # XD
    transformer_layer_p.tr_atten_tpl.interleave_kv_heads = getattr(cls, 'INTERLEAVE_KV_HEADS', True)  # XD
    if hasattr(cls, 'MERGE_DW_PROJ'): transformer_layer_p.tr_atten_tpl.merge_dw_proj = cls.MERGE_DW_PROJ  # XD
    if getattr(cls, 'OUTPUT_LAYER_INIT_METHOD', None) is not None:  # XD
      output_layer_std = 2 / (cls.NUM_LAYERS * math.sqrt(cls.MODEL_DIMS)) \
        if cls.OUTPUT_LAYER_INIT_METHOD == 'wang_init' else cls.OUTPUT_LAYER_INIT_METHOD
      transformer_layer_p.tr_atten_tpl.output_layer_std = output_layer_std
      transformer_layer_p.tr_fflayer_tpl.output_layer_std = output_layer_std
    if hasattr(cls, 'SEGMENT_SIZE'):  # XD
      transformer_layer_p.tr_fflayer_tpl.segment_size = cls.SEGMENT_SIZE

    if hasattr(cls, 'RESIDUAL_CROSS_ACT_PROJ'):  # XD
      transformer_layer_p.tr_fflayer_tpl.residual_cross_act_proj = cls.RESIDUAL_CROSS_ACT_PROJ

#================================================================================================
    if hasattr(cls, 'MGATE'):  # lsp
      transformer_layer_p.tr_fflayer_tpl.mgate = cls.MGATE
    if hasattr(cls, 'DSM'):  # lsp
      transformer_layer_p.tr_fflayer_tpl.dsm = cls.DSM
    if hasattr(cls, 'FFN_CHUKN_SIZE'):  # lsp
      transformer_layer_p.tr_fflayer_tpl.chunk_size = cls.FFN_CHUKN_SIZE
    if hasattr(cls, 'NUM_EXPERTS'):  # lsp
      stacked_p.num_experts = cls.NUM_EXPERTS
    if hasattr(cls, 'CAPACITY_FACTOR'):  # lsp
      stacked_p.unadjusted_expert_capacity_factor = cls.CAPACITY_FACTOR
    if hasattr(cls, 'MOE_LAYERS'):  # lsp
      stacked_p.moe_layers = cls.MOE_LAYERS
    if hasattr(cls, 'MOE_GATED_ACTIVATION'):  # lsp
      stacked_p.moe_gated_activation = cls.MOE_GATED_ACTIVATION
    if hasattr(cls, 'REMAT'):  # lsp
      stacked_p.remat = cls.REMAT
#================================================================================================

    # XD
    for name in ['num_groups', 'project_logits', 'project_probs', 
                'logits_residual', 'probs_residual', 'logits_absorb_residual', 'probs_absorb_residual',
                'logits_squeeze_ratio', 'logits_squeeze_activation_cls', 'logits_output_activation_cls',
                'probs_squeeze_ratio', 'probs_squeeze_activation_cls', 'probs_output_activation_cls', 'left_mul',
                'dim_per_head_v', 'value_gate_activation_cls', 'o_gate_activation_cls',
                'float32_logits', 'float32_probs', 'float32_value', 'qk_norm', 'transpose_logits',
                'shared_qk_dim', 'shared_ov_dim', 'dim_per_shared_head', 'scale_shared_key', 'scale_init', 'scale_bias', 'rotate_shared_qk',
                ]:
      NAME = name.upper()
      if prefix == 'early_' and hasattr(cls, NAME + '_EARLY'):
        NAME = NAME + '_EARLY'
      if hasattr(cls, NAME):
        setattr(transformer_layer_p.tr_atten_tpl, name, getattr(cls, NAME))

    dynamic_w_attrs = ['dynamic_w_init', 'dynamic_d_init', 
        'dw_activation_cls', 'dw_activation_weights', 'dynamic_squeeze_ratio',
        'dw_cap', 'learned_dw_cap', 'use_dw_cap_bias', 'decompose_dynamic_w',
        'dynamic_w_hidden_dim', 'dynamic_d_hidden_dim', 'dw_hidden_activation_cls',
        'use_dw_hidden_bias', 'merge_dynamic_w_hidden', 'dw_hidden_gate_act_cls',
        'dw1_norm_cls', 'dw1_norm_dbias_init', 'dw1_norm_bias_init', 'dw1_norm_bias_const', 'square_dw1_norm_bias',
        'dw_gate_activation_cls', 'dw_gate_weights', 'dd_gate_activation_cls', 'dd_activation_cls', 'summary_verbosity',
    ]
    for name in ['input_activation_cls', 'use_input_bias', 'merge_dw_op',
        'use_squeeze_bias', 'transpose', 'learnable_diag', 'relative_scale', 'skip_ffn_weight_decay',
        'dynamic_squeeze_gate_act_cls', 'gate_relative_scale', 'addictive_gate', 'use_static_w',
        'src_dependent', 'tgt_dependent', 'skip_bias', 'summary_verbosity', 'loop_over_dynamic_hd', # 'squeeze_gate_activation_cls', 
      ] + dynamic_w_attrs:
      NAME = name.upper()
      if prefix == 'early_' and any(hasattr(cls, s + NAME + '_EARLY') for s in ['', 'LOGITS_', 'PROBS_']):
        NAME = NAME + '_EARLY'
      if hasattr(cls, NAME) and not hasattr(cls, 'LOGITS_' + NAME) and not hasattr(cls, 'PROBS_' + NAME):
        setattr(transformer_layer_p.tr_atten_tpl.cross_head_pre_proj_tpl, name, getattr(cls, NAME))
        setattr(transformer_layer_p.tr_atten_tpl.cross_head_post_proj_tpl, name, getattr(cls, NAME))
      if hasattr(cls, 'LOGITS_' + NAME):
        setattr(transformer_layer_p.tr_atten_tpl.cross_head_pre_proj_tpl, name, getattr(cls, 'LOGITS_' + NAME))
      if hasattr(cls, 'PROBS_' + NAME):
        setattr(transformer_layer_p.tr_atten_tpl.cross_head_post_proj_tpl, name, getattr(cls, 'PROBS_' + NAME))
    if getattr(cls, 'QUERY_CHUNK_SIZE', None) is not None:
      transformer_layer_p.tr_atten_tpl.query_chunk_size = cls.QUERY_CHUNK_SIZE
      if getattr(cls, 'WINDOW_SIZE', None) is not None:
        transformer_layer_p.tr_atten_tpl.window_size = cls.WINDOW_SIZE
      for name in dynamic_w_attrs:
        NAME = name.upper()
        if prefix == 'early_' and any(hasattr(cls, s + NAME + '_EARLY') for s in ['', 'LOGITS_', 'PROBS_']):
          NAME = NAME + '_EARLY'
        if hasattr(cls, NAME) and not hasattr(cls, 'LOGITS_' + NAME) and not hasattr(cls, 'PROBS_' + NAME):
          setattr(transformer_layer_p.tr_atten_tpl.dynamic_w_pre_proj_tpl, name, getattr(cls, NAME))
          setattr(transformer_layer_p.tr_atten_tpl.dynamic_w_post_proj_tpl, name, getattr(cls, NAME))
        if hasattr(cls, 'LOGITS_' + NAME):
          setattr(transformer_layer_p.tr_atten_tpl.dynamic_w_pre_proj_tpl, name, getattr(cls, 'LOGITS_' + NAME))
        if hasattr(cls, 'PROBS_' + NAME):
          setattr(transformer_layer_p.tr_atten_tpl.dynamic_w_post_proj_tpl, name, getattr(cls, 'PROBS_' + NAME))

    transformer_layer_p.tr_fflayer_tpl.has_bias = not cls.USE_GATED_ACTIVATION or cls.USE_BIAS  # XD add
    if cls.ACTIVATION_CLS == layers.GELU: transformer_layer_p.tr_fflayer_tpl.activation_tpl.approximate = True  # XD: add if
    if not cls.USE_GATED_ACTIVATION: transformer_layer_p.hidden_dims = cls.MODEL_DIMS * 4 # XD add

    for atten_p in (
        transformer_layer_p.tr_atten_tpl,
        transformer_layer_p.cross_atten_tpl,
    ):
      if atten_p is None:
        continue
      atten_wp = atten_p.weight_split_dims_mapping
      atten_wp.proj = ['data', 'mdl', None]

  if task_p.early_stopping_fn is None:
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
  TARGET_LOG_PPLX = 2.69 - 2.69  # XD

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
  MAX_SEQ_LEN = 2048 // 2  # XD
  NUM_LAYERS = 12
  MODEL_DIMS = 768
  ACTIVATION_CLS = layers.SiLU  # layers.SiLU/GELU  # XD
  USE_GATED_ACTIVATION = True  # XD
  HIDDEN_DIMS = MODEL_DIMS * 4 # 2048  # XD: MODEL_DIMS * 4
  NUM_HEADS = 12
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  COMBINE_QKV = False
  USE_BIAS = False # XD add
  NORMALIZATION_CLS = normalizations.RmsNorm  # XD add RmsNorm
  SKIP_RMSNORM_WD = True
  LEARNING_RATE = 2e-4  # XD
  PERCORE_BATCH_SIZE = 4
  FPROP_DTYPE = jnp.bfloat16

  # ICI_MESH_SHAPE = [1, 8, 4]
  ICI_MESH_SHAPE = [1, 4, 2]

  SEPARATE_EMBEDDING = True  # XD
  USE_ROTARY_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 2048  # XD add

  SUMMARY_INTERVAL_STEPS = 10  # XD
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 1

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    if self.USE_ROTARY_POSITION_EMB: task_p.model.lm_tpl.position_emb_tpl = None  # XD: add if
    return task_p

@experiment_registry.register
class C4SpmdGpt37BRoPE(C4SpmdGpt3SmallRoPE):  # XD
  MAX_SEQ_LEN = 2048 #// 2  # XD
  VOCAB_SIZE = 32000
  NUM_LAYERS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 32
  NUM_GROUPS = -1  # XD
  # DIMS_PER_HEAD = 128
  COMBINE_QKV = True
  
  PERCORE_BATCH_SIZE = 8 * 2
  #ICI_MESH_SHAPE = [1, 8 // NUM_GROUPS, NUM_GROUPS]
  # ICI_MESH_SHAPE = [4, 1, 8]  # bs=2, 0.146, combine_qkv 0.1514  by xd
  # ICI_MESH_SHAPE = [1, 8, 4]  # bs=8, 0.044, combine_qkv 0.045  by xd
  #ICI_MESH_SHAPE = [1, 32, 1]  # bs=4, combine_qkv 0.0935  by xd
  ICI_MESH_SHAPE = [1, 32, 1]  # v5-64 bs=8 0.131 by xd
  CHECKPOINT_EVERY_N_STEPS=50

@experiment_registry.register
class C4SpmdGpt37BRoPEv4(C4SpmdGpt37BRoPE):
  PERCORE_BATCH_SIZE = 8
  # ICI_MESH_SHAPE = [1, 32, 1]  # bs=16*2*16*1, v4-64: 0.175 by lsp seqlen 1024
  #ICI_MESH_SHAPE = [1, 16, 1]  # bs=4*1*16*1, v4-16: seqlen 1024 0.388  seqlen 2048 0.23
  # ICI_MESH_SHAPE = [1, 8, 2]  # v4-32 bs=8 0.119  by xd
  ICI_MESH_SHAPE = [1, 16, 1]  # v4-32 bs=8 0.138 by lsp
  # ICI_MESH_SHAPE = [1, 16, 2]  # v4-64 bs=8 0.121 by xd

@experiment_registry.register
class C4SpmdGpt313BRoPE(C4SpmdGpt3SmallRoPE):  # XD
  VOCAB_SIZE = 32000  # XD
  NUM_LAYERS = 40
  MODEL_DIMS = 5120
  HIDDEN_DIMS = 13824  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 40
  # DIMS_PER_HEAD = 128
  COMBINE_QKV = False
  
  PERCORE_BATCH_SIZE = 6
  ICI_MESH_SHAPE = [1, 16, 4]  # bs=6*8*8, 0.032

@experiment_registry.register
class C4SpmdLlamaSmall(C4SpmdGpt3SmallRoPE):
  NUM_LAYERS = 12
  MODEL_DIMS = 768
  HIDDEN_DIMS = 2048
  NUM_HEADS = 12
  DIMS_PER_HEAD = 64

@experiment_registry.register
class C4SpmdLlamaMedium(C4SpmdGpt3SmallRoPE):
  NUM_LAYERS = 24
  MODEL_DIMS = 1024
  HIDDEN_DIMS = 2816  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 16
  DIMS_PER_HEAD = 64
  # NUM_GROUPS = 1  # XD
  COMBINE_QKV = False
  
  # LEARNING_RATE = 6e-5
  LR_COS_WARMUP = 256   # XD
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 50000

  PERCORE_BATCH_SIZE = 16
  ICI_MESH_SHAPE = [1, 32, 1]  # 0.549， combine_qkv 0.493???!!!, v5 0.436???
  # ICI_MESH_SHAPE = [32, 1, 1]

@experiment_registry.register
class C4SpmdGPT:
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  USE_ROTARY_POSITION_EMB = False

@experiment_registry.register
class C4SpmdGPTSepEmb:
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  USE_ROTARY_POSITION_EMB = False

@experiment_registry.register
class C4SpmdGPTMedium(C4SpmdGPT, C4SpmdLlamaMedium):
  HIDDEN_DIMS = 4096  # v3 bug (USE_ROTARY_POSITION_EMB==True) 0.585??? vs fix 0.571

@experiment_registry.register
class C4SpmdGPTMediumSepEmb(C4SpmdGPTSepEmb, C4SpmdLlamaMedium):
  HIDDEN_DIMS = 4096  # v3

@experiment_registry.register
class C4SpmdLlamaMediumNoSwiGLU(C4SpmdLlamaMedium):
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False  # v3 0.55
  HIDDEN_DIMS = 4096

@experiment_registry.register
class C4SpmdLlamaMediumNoRoPE(C4SpmdLlamaMedium):
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True  # v3 0.561
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  USE_ROTARY_POSITION_EMB = False

@experiment_registry.register
class C4SpmdLlamaMediumNoRoPESepEmb(C4SpmdLlamaMedium):
  TRAINABLE_POSITION_EMB = True  # v3
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  USE_ROTARY_POSITION_EMB = False

@experiment_registry.register
class C4SpmdLlamaMediumNoRoPESepScaleEmb(C4SpmdLlamaMedium):
  EXPLICIT_INIT = False
  SCALE_EMBEDDING = True
  TRAINABLE_POSITION_EMB = False
  USE_ROTARY_POSITION_EMB = False
  
@experiment_registry.register
class C4SpmdLlamaMediumSeqLen512(C4SpmdLlamaMedium):
  MAX_SEQ_LEN = 512  # 0.648
  PERCORE_BATCH_SIZE = 32

@experiment_registry.register
class C4SpmdLlamaMediumSeqLen2K(C4SpmdLlamaMedium):
  MAX_SEQ_LEN = 2048  # 0.394
  PERCORE_BATCH_SIZE = 8

@experiment_registry.register
class C4SpmdLlamaMediumSkipRmsNormWD(C4SpmdLlamaMedium):
  SKIP_RMSNORM_WD = True

@experiment_registry.register
class C4SpmdLlamaMediumWD01(C4SpmdLlamaMedium):
  WEIGHT_DECAY = 0.01

@experiment_registry.register
class C4SpmdLlamaMediumNoWD(C4SpmdLlamaMedium):
  WEIGHT_DECAY = None

@experiment_registry.register
class C4SpmdLlamaLarge(C4SpmdGpt3SmallRoPE):
  NUM_LAYERS = 24
  MODEL_DIMS = 1536
  HIDDEN_DIMS = 4096  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 12 * 2
  DIMS_PER_HEAD = 128 // 2
  COMBINE_QKV = False

  PERCORE_BATCH_SIZE = 16  # 0.299
  ICI_MESH_SHAPE = [1, 32, 1]

@experiment_registry.register
class C4SpmdLlamaXL(C4SpmdGpt3SmallRoPE):
  NUM_LAYERS = 28
  MODEL_DIMS = 2048
  HIDDEN_DIMS = 5504  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 32
  DIMS_PER_HEAD = 64
  COMBINE_QKV = False
  
  # LEARNING_RATE = 6e-5
  LR_COS_WARMUP = 256   # XD
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 65536

  PERCORE_BATCH_SIZE = 16  # 0.168, v4 0.189!?
  ICI_MESH_SHAPE = [1, 64, 1]

@experiment_registry.register
class C4SpmdLlamaXXL(C4SpmdLlamaXL):
  NUM_LAYERS = 26  # v4 0.139
  MODEL_DIMS = 3072
  HIDDEN_DIMS = 8192  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 24
  DIMS_PER_HEAD = 128

@experiment_registry.register
class C4SpmdLlama7B(C4SpmdLlamaXXL):
  MAX_SEQ_LEN = 2048
  NUM_LAYERS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 32
  DIMS_PER_HEAD = 128
  
  LR_COS_WARMUP = 256
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 65536

  # ICI_MESH_SHAPE = [1, 128, 1]  # v3-128 2048*8*64*2 0.048, 2048*4*128*1 OOM!
  QUERY_CHUNK_SIZE = 128
  SUMMARY_INTERVAL_STEPS = 10

class _Llama(C4SpmdGpt3AdamOrgHP):
  ACTIVATION_CLS = layers.SiLU  # layers.SiLU/GELU  # XD
  USE_GATED_ACTIVATION = True  # XD
  COMBINE_QKV = False
  USE_BIAS = False # XD add
  NORMALIZATION_CLS = normalizations.RmsNorm  # XD add RmsNorm
  SKIP_RMSNORM_WD = True
  FPROP_DTYPE = jnp.bfloat16

  SEPARATE_EMBEDDING = True  # XD
  USE_ROTARY_POSITION_EMB = True

  CHECKPOINT_MAX_TO_KEEP = 2

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    task_p = super().task()
    if self.USE_ROTARY_POSITION_EMB: task_p.model.lm_tpl.position_emb_tpl = None  # XD: add if
    return task_p

class _Llama3B(_Llama):
  NUM_LAYERS = 32
  MODEL_DIMS = 2560
  HIDDEN_DIMS = 6912
  NUM_HEADS = 32
  DIMS_PER_HEAD = 80

class _Llama7B(_Llama):
  NUM_LAYERS = 32
  MODEL_DIMS = 4096
  HIDDEN_DIMS = 11008
  NUM_HEADS = 32
  DIMS_PER_HEAD = 128

class _Llama13B(_Llama):
  NUM_LAYERS = 40
  MODEL_DIMS = 5120
  HIDDEN_DIMS = 13824
  NUM_HEADS = 40
  DIMS_PER_HEAD = 128

class _Llama33B(_Llama):
  NUM_LAYERS = 60
  MODEL_DIMS = 6656
  HIDDEN_DIMS = 17920
  NUM_HEADS = 52
  DIMS_PER_HEAD = 128

class _SlimLlama7B(_Llama7B):
  NUM_LAYERS = 48
  HIDDEN_DIMS = 5504  # XD: MODEL_DIMS * 4 // 3

@experiment_registry.register
class Pythia7B(C4SpmdLlama7B):
  USE_BIAS = True
  NORMALIZATION_CLS = normalizations.LayerNorm
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  HIDDEN_DIMS = 16384
  GPT_J_RESIDUAL = True

  QUERY_CHUNK_SIZE = 128
  EMBEDDING_LOOKUP_STYLE = 'index'
  LM_HEAD_CHUNK_SIZE = 512

  LEARNING_RATE = 1.2e-4
  WEIGHT_DECAY = 0.01
  LR_COS_WARMUP = 1430
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 143000

@experiment_registry.register
class Pythia33B(Pythia7B):
  # NUM_LAYERS = 60   # llama
  # MODEL_DIMS = 6656
  # NUM_HEADS = 52
  # HIDDEN_DIMS = 26624
  NUM_LAYERS = 48  # llama-2
  MODEL_DIMS = 8192
  NUM_HEADS = 64
  HIDDEN_DIMS = 32768
  DIMS_PER_HEAD = 128

  LEARNING_RATE = 1e-4

@experiment_registry.register
class Pythia3B(Pythia7B):
  NUM_LAYERS = 32
  MODEL_DIMS = 2560
  NUM_HEADS = 32
  DIMS_PER_HEAD = 80
  HIDDEN_DIMS = 10240

  LEARNING_RATE = 1.6e-4

@experiment_registry.register
class PythiaXL(Pythia7B):
  NUM_LAYERS = 24
  MODEL_DIMS = 2048
  NUM_HEADS = 16
  DIMS_PER_HEAD = 128
  HIDDEN_DIMS = 8192

  LEARNING_RATE = 2e-4

@experiment_registry.register
# class Pythia410M(C4SpmdLlama7B):  # TODO: C4SpmdLlama7B -> Pythia7B
class Pythia410M(Pythia7B):  # TODO: C4SpmdLlama7B -> Pythia7B
  NUM_LAYERS = 24
  MODEL_DIMS = 1024
  NUM_HEADS = 16
  DIMS_PER_HEAD = 64
  HIDDEN_DIMS = 4096

  LEARNING_RATE = 3e-4

@experiment_registry.register
class Pythia160M(Pythia7B):
  NUM_LAYERS = 12
  MODEL_DIMS = 768
  NUM_HEADS = 12
  DIMS_PER_HEAD = 64
  HIDDEN_DIMS = 3072

  LEARNING_RATE = 6e-4

@experiment_registry.register
class Pythia70M(Pythia7B):
  NUM_LAYERS = 6
  MODEL_DIMS = 512
  NUM_HEADS = 8
  DIMS_PER_HEAD = 64
  HIDDEN_DIMS = 2048

  LEARNING_RATE = 10e-4

@experiment_registry.register
class C4SpmdLlama7BFFN16512:
  NUM_LAYERS = 24
  HIDDEN_DIMS = 16512

@experiment_registry.register
class C4SpmdLlama7B128x1(C4SpmdLlama7B):
  PERCORE_BATCH_SIZE = 2 * 2 * 2
  ICI_MESH_SHAPE = [1, 128, 1]  # v3 pbs8 0.0519, v4 pbs8 0.0900
  # ICI_MESH_SHAPE = [1, 64, 2]  # v3 pbs8 win256 0.0487, v4 error
  # ICI_MESH_SHAPE = [1, 32, 4]  # v3 pbs8 win256 0.0448, v4 0.0612
  # NUM_LAYERS_PER_BLOCK = 2
  # WINDOW_SIZE = [256, None]  # v3 pbs4/8 0.0997/0.0528, v4 pbs8 0.0959!
  EMBEDDING_LOOKUP_STYLE = 'index'
  LM_HEAD_CHUNK_SIZE = 512

@experiment_registry.register
class C4SpmdLlama7B256x1(C4SpmdLlama7B):
  PERCORE_BATCH_SIZE = 2 * 2 #* 2
  ICI_MESH_SHAPE = [1, 256, 1] # v4-256 2048*2*256*1 0.295, 1024*4*256*1 0.32
  QUERY_CHUNK_SIZE = 128   # v4 pbs4 NoWin chunk 64/256  0.155/0.168

  # NUM_LAYERS_PER_BLOCK = 2
  # WINDOW_SIZE = [None, 256]  # v4 pbs2 0.319; pbs4 lmchunk512 win128/256/384 0.1731/0.1750/0.1732 win256 is fastest!?; pbs8 win256 0.095 little faster than NoWin (0.094)

  EMBEDDING_LOOKUP_STYLE = 'index'
  LM_HEAD_CHUNK_SIZE = 512  # v4 NoWin chunk 512 PBS4/8 0.166/0.09; Win128: PBS 4 chunk128/512/1024 0.171/0.174/0.175, PBS 8 chunk512 0.094, PBS 12 chunk512 0.06
  # FFN_CHUNK_SIZE = 2752  # 0.087

class PythiaInit:
  INIT_METHOD = 'small_init'
  OUTPUT_LAYER_INIT_METHOD = 'wang_init'

class PythiaInit7B:
  INIT_METHOD = 0.02
  OUTPUT_LAYER_INIT_METHOD = 0.0025

@experiment_registry.register
class Pythia410M64x1(PythiaInit, Pythia410M):
  PERCORE_BATCH_SIZE = 2 * 2 * 2 * 2
  ICI_MESH_SHAPE = [1, 64, 1]

@experiment_registry.register
class Pythia410M128x1(PythiaInit, Pythia410M):
  PERCORE_BATCH_SIZE = 2 * 2 * 2
  ICI_MESH_SHAPE = [1, 128, 1]

@experiment_registry.register
class Pythia70M64x1(PythiaInit, Pythia70M):
  PERCORE_BATCH_SIZE = 16
  ICI_MESH_SHAPE = [1, 64, 1]

@experiment_registry.register
class PythiaXL128x1(PythiaInit, PythiaXL):
  PERCORE_BATCH_SIZE = 2 * 2 * 2
  ICI_MESH_SHAPE = [1, 128, 1]

@experiment_registry.register
class Pythia7B128x1(PythiaInit, Pythia7B):
  PERCORE_BATCH_SIZE = 2 * 2 * 2
  ICI_MESH_SHAPE = [1, 128, 1]  # v3 win256 0.0525, v4 win256 0.0972
  # NUM_LAYERS_PER_BLOCK = 2
  # WINDOW_SIZE = [256, None]

@experiment_registry.register
class Pythia7B256x1(Pythia7B128x1):
  PERCORE_BATCH_SIZE = 2 * 2
  ICI_MESH_SHAPE = [1, 256, 1]

# @experiment_registry.register
# class Pythia7B256x1(C4SpmdLlama7B256x1):
#   GPT_J_RESIDUAL = True  # v4 NoWin 0.1771, win256 0.1777

# @experiment_registry.register
# class Pythia7B128x1PythiaInit(PythiaInit, Pythia7B128x1): pass  # v4 0.0972

@experiment_registry.register
class C4SpmdLlama7B64x4(C4SpmdLlama7B):
  PERCORE_BATCH_SIZE = 8
  ICI_MESH_SHAPE = [1, 64, 4] # v4-256 2048*8*64*4 0.06, v4-256 1024*16*64*4 0.065

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  DIM_PER_SHARED_HEAD = 16
  FLOAT32_LOGITS = True

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x64(C4SpmdLlamaMediumShareHeads):
  SHARED_QK_DIM = 1024  # 0.233, float32_logits 0.218, float32_logits v4 0.287 / 0.309 (fix probs fp32->fp16)
  SHARED_OV_DIM = 1024
  NUM_SHARED_HEADS = 16
  DIM_PER_SHARED_HEAD = 64

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x16(C4SpmdLlamaMediumShareHeads):
  DIMS_PER_HEAD = 48
  SHARED_QK_DIM = 256  # v4 0.514
  SHARED_OV_DIM = 256
  NUM_SHARED_HEADS = 16

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x16NoRot(C4SpmdLlamaMediumShareHeads16x16):
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads128x2(C4SpmdLlamaMediumShareHeads):
  DIMS_PER_HEAD = 48
  SHARED_QK_DIM = 256  # 0.321
  SHARED_OV_DIM = 256
  NUM_SHARED_HEADS = 128
  DIM_PER_SHARED_HEAD = 2

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads128x2NoRot(C4SpmdLlamaMediumShareHeads128x2):
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x64FP32value(C4SpmdLlamaMediumShareHeads16x64):
  FLOAT32_VALUE = True  # v4 0.282

@experiment_registry.register
class C4SpmdLlamaMediumShareQK16x64(C4SpmdLlamaMediumShareHeads16x64):
  SHARED_OV_DIM = 0  # 0.298

@experiment_registry.register
class C4SpmdLlamaMediumShareOV16x64(C4SpmdLlamaMediumShareHeads16x64):
  SHARED_QK_DIM = 0  # 0.296

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads128(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_QK_DIM = 128  # 0.464
  SHARED_OV_DIM = 128
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x16ScaleInit05_15(C4SpmdLlamaMedium):
  DIMS_PER_HEAD = 48
  NUM_GROUPS = 1
  SHARED_QK_DIM = 256  # 0.394
  SHARED_OV_DIM = 256
  DIM_PER_SHARED_HEAD = 16
  SCALE_INIT = WeightInit.Uniform(0.05)
  SCALE_BIAS = 0.1
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x13ScaleInit05_15(C4SpmdLlamaMedium):
  DIMS_PER_HEAD = 48
  NUM_GROUPS = 1
  SHARED_QK_DIM = 208  # 0.424
  SHARED_OV_DIM = 208
  DIM_PER_SHARED_HEAD = 16
  SCALE_INIT = WeightInit.Uniform(0.05)
  SCALE_BIAS = 0.1
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x8x2ScaleInit05_15(C4SpmdLlamaMedium):
  DIMS_PER_HEAD = 48
  NUM_GROUPS = 2
  SHARED_QK_DIM = 256  # 0.462
  SHARED_OV_DIM = 256
  DIM_PER_SHARED_HEAD = 16
  SCALE_INIT = WeightInit.Uniform(0.05)
  SCALE_BIAS = 0.1
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x8ScaleInit05_15(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_QK_DIM = 128  # 0.461
  SHARED_OV_DIM = 128
  DIM_PER_SHARED_HEAD = 16
  SCALE_INIT = WeightInit.Uniform(0.05)
  SCALE_BIAS = 0.1
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareHeadsRot16x8ScaleInit05_15(C4SpmdLlamaMediumShareHeads16x8ScaleInit05_15):
  ROTATE_SHARED_QK = True  # 0.45

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads16x8ScaleInit00_10(C4SpmdLlamaMediumShareHeads16x8ScaleInit05_15):
  SCALE_BIAS = 0.05

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads128x1ScaleInit05_15(C4SpmdLlamaMediumShareHeads16x8ScaleInit05_15):
  DIM_PER_SHARED_HEAD = 1  # 0.464

@experiment_registry.register
class C4SpmdLlamaMediumShareHeads128x1ScaleInit25_75(C4SpmdLlamaMediumShareHeads16x8ScaleInit05_15):
  SCALE_INIT = WeightInit.Uniform(0.25)
  SCALE_BIAS = 0.5

@experiment_registry.register
class C4SpmdLlamaMediumShareHeadsRot(C4SpmdLlamaMediumShareHeads):
  ROTATE_SHARED_QK = True  # 0.470

@experiment_registry.register
class C4SpmdLlamaMediumShareHeadsRot128(C4SpmdLlamaMediumShareHeads128):
  ROTATE_SHARED_QK = True  # 0.455

@experiment_registry.register
class C4SpmdLlamaMediumShareHeadsScaleKRot128(C4SpmdLlamaMediumShareHeadsRot128):
  SCALE_SHARED_KEY = True  # 0.436

@experiment_registry.register
class C4SpmdLlamaMediumDimPerHead128(C4SpmdLlamaMedium):
  DIMS_PER_HEAD = 128   # 192 0.4, 128 0.47
  # DIMS_PER_HEAD = 160  # 0.419

@experiment_registry.register
class C4SpmdLlamaMediumHead16x128(C4SpmdLlamaMedium):
  DIMS_PER_HEAD = 128   # v4 0.515

@experiment_registry.register
class C4SpmdLlamaMediumHead8x128(C4SpmdLlamaMedium):
  NUM_HEADS = 8  # 0.666
  DIMS_PER_HEAD = 128

@experiment_registry.register
class C4SpmdLlamaMediumShareQK(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_QK_DIM = 96  # 0.207
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareOV(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_OV_DIM = 96  # 0.204

@experiment_registry.register
class C4SpmdLlamaMediumShareQK128(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_QK_DIM = 128  #
  ROTATE_SHARED_QK = False

@experiment_registry.register
class C4SpmdLlamaMediumShareOV128(C4SpmdLlamaMedium):
  NUM_GROUPS = 1
  SHARED_OV_DIM = 128  # 0.489

@experiment_registry.register
class C4SpmdLlamaMediumv4(C4SpmdLlamaMedium):
  PERCORE_BATCH_SIZE = 16 * 2
  ICI_MESH_SHAPE = [1, 32 // 2, 1]  # v4 0.619

@experiment_registry.register
class C4SpmdLlamaMediumGA(C4SpmdLlamaMedium):
  DIM_PER_HEAD_V = 128  # 0.6
  VALUE_GATE_ACTIVATION_CLS = layers.SiLU
  HIDDEN_DIMS = 1408

@experiment_registry.register
class C4SpmdLlamaMediumGAv4(C4SpmdLlamaMediumGA):
  PERCORE_BATCH_SIZE = 16 * 2
  ICI_MESH_SHAPE = [1, 32 // 2, 1]  # v4 0.618

@experiment_registry.register
class C4SpmdLlamaMediumGA256x8(C4SpmdLlamaMediumGA):
  NUM_HEADS = 8
  DIM_PER_HEAD_V = 256  # 0.642, v5 0.544???

@experiment_registry.register
class C4SpmdLlamaMediumGA256x8FP32logits(C4SpmdLlamaMediumGA256x8):
  NUM_HEADS = 8
  DIM_PER_HEAD_V = 256
  FLOAT32_LOGITS = True

@experiment_registry.register
class C4SpmdLlamaMediumGA256x8QKNorm(C4SpmdLlamaMediumGA256x8):
  NUM_HEADS = 8
  DIM_PER_HEAD_V = 256
  QK_NORM = True

@experiment_registry.register
class C4SpmdLlamaMediumGA256x8v4(C4SpmdLlamaMediumGA256x8):
  PERCORE_BATCH_SIZE = 16 * 2
  ICI_MESH_SHAPE = [1, 32 // 2, 1]  # v4 0.733

@experiment_registry.register
class C4SpmdLlamaMediumResTH(C4SpmdLlamaMedium):
  NUM_GROUPS = 1  # 0.37, res 0.208/0.211, v4 0.336, all no_absorbres
  PROJECT_LOGITS = True
  PROJECT_PROBS = True
  LOGITS_ABSORB_RESIDUAL = True
  PROBS_ABSORB_RESIDUAL = True

@experiment_registry.register
class C4SpmdLlamaXLHead16x128(C4SpmdLlamaXL):
  NUM_HEADS = 16  # 0.20
  DIMS_PER_HEAD = 128

@experiment_registry.register
class _LlamaXLHead24x80:
  NUM_HEADS = 24
  DIMS_PER_HEAD = 80
  HIDDEN_DIMS = 5632  # (2048 * 8 + (2048 + 80 * 24) * 4) / 3

@experiment_registry.register
class _LlamaXLHead24x96:
  NUM_HEADS = 24
  DIMS_PER_HEAD = 96
  HIDDEN_DIMS = 5120  # (2048 * 8 + (96 * 24 - 2048) * 4) / 3

@experiment_registry.register
class C4SpmdGPTXLSepEmb(C4SpmdGPTSepEmb, C4SpmdLlamaXLHead16x128):
  HIDDEN_DIMS = 8192  # v3 0.204

@experiment_registry.register
class C4SpmdLlamaXLNoSwiGLU:  # (C4SpmdLlamaXLHead16x128):
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False  # v3 0.208 vs v4 merely 0.21 !?
  HIDDEN_DIMS = 8192

@experiment_registry.register
class C4SpmdLlamaXLMQA(C4SpmdLlamaXLHead16x128):
  NUM_KV_HEADS = 1  # v3 0.202
  HIDDEN_DIMS = 6784

@experiment_registry.register
class C4SpmdLlamaXLWin128(C4SpmdLlamaXLHead16x128):
  EMBEDDING_LOOKUP_STYLE = 'index'
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [None, 128]  # v3 0.211

@experiment_registry.register
class C4SpmdLlamaXL16x64(C4SpmdLlamaXL):
  NUM_HEADS = 16  # v3 0.201
  DIMS_PER_HEAD = 64
  HIDDEN_DIMS = 6912  # XD: MODEL_DIMS * 4 * 2 // 3

@experiment_registry.register
class C4SpmdLlamaXLFP32logits(C4SpmdLlamaXL):
  FLOAT32_LOGITS = True  # 0.155

@experiment_registry.register
class C4SpmdLlamaXLResTH(C4SpmdLlamaXL):
  NUM_GROUPS = 1  #  v4 absorbres 0.150  v4 no_absorbres 0.117
  PROJECT_LOGITS = True
  PROJECT_PROBS = True
  LOGITS_ABSORB_RESIDUAL = True
  PROBS_ABSORB_RESIDUAL = True

@experiment_registry.register
class C4SpmdLlamaXLResTHGaussian04(C4SpmdLlamaXLResTH):
  SCALE_INIT = WeightInit.Gaussian(0.04)

@experiment_registry.register
class C4SpmdLlamaXLHead16x128ResTH(C4SpmdLlamaXLResTH):
  NUM_HEADS = 16  # v4 0.186
  DIMS_PER_HEAD = 128

@experiment_registry.register
class C4SpmdLlamaXLHeadResTHFFN16(C4SpmdLlamaXLHead16x128ResTH):
  LOGITS_SQUEEZE_RATIO = 16  # v4 0.187
  PROBS_SQUEEZE_RATIO = 16
  USE_SQUEEZE_BIAS = False
  LOGITS_ABSORB_RESIDUAL = False
  PROBS_ABSORB_RESIDUAL = False
  SKIP_FFN_WEIGHT_DECAY = True

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16(C4SpmdLlamaXLHeadResTHFFN16):
  USE_BIAS = False  # fix init_fn bug. fix class name bug by the way

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN4(C4SpmdLlamaXLResTHFFN16):
  LOGITS_SQUEEZE_RATIO = 4  # v4 0.156
  PROBS_SQUEEZE_RATIO = 4

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN4LogitsGELU(C4SpmdLlamaXLResTHFFN4):
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU  # v4 0.155
  USE_BIAS = True

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN4DynW0003LearnDiagDWTanh(C4SpmdLlamaXLResTHFFN4LogitsGELU):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0003)  # 0.117
  TRANSPOSE = True
  LEARNABLE_DIAG = True
  DW_ACTIVATION_CLS = layers.Tanh
  PROBS_OUTPUT_ACTIVATION_CLS = layers.ReLU

@experiment_registry.register
class C4SpmdLlamaXLHeadResTHFFN16DynW0003(C4SpmdLlamaXLHeadResTHFFN16):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0003)
  TRANSPOSE = True

@experiment_registry.register
class C4SpmdLlamaXLHeadResTHFFN16DynW0003DWTanh(C4SpmdLlamaXLHeadResTHFFN16DynW0003):
  DW_ACTIVATION_CLS = layers.Tanh  # v4 0.136??? vs C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWTanh
  PROBS_OUTPUT_ACTIVATION_CLS = layers.ReLU

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16DynW0003DWTanh(C4SpmdLlamaXLHeadResTHFFN16DynW0003DWTanh):
  DW_ACTIVATION_CLS = layers.Tanh # fix init_fn bug. fix class name bug by the way

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWTanh_test(C4SpmdLlamaXLHeadResTHFFN16DynW0003DWTanh):
  LEARNABLE_DIAG = True  # v4 0.142 to compare with C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWTanh

@experiment_registry.register
class C4SpmdLlamaXLHeadResTHFFN16DynW0003LearnDiag(C4SpmdLlamaXLHeadResTHFFN16DynW0003):
  LEARNABLE_DIAG = True # v4 0.151

@experiment_registry.register
class C4SpmdLlamaXLHeadResTHFFN16DynW0003LearnDiagSave(C4SpmdLlamaXLHeadResTHFFN16DynW0003LearnDiag):
  SAVE_ON_STEPS = [10460, 10490, 10530, 10540, 10570, 11120, 11140, 11150, 13570, 13610, 13640] + list(range(7000, 25000, 1000))

@experiment_registry.register
class C4SpmdLlamaXLHeadResTHFFN16DynW0003LearnDiagSaveAlignRelScale(C4SpmdLlamaXLHeadResTHFFN16DynW0003LearnDiagSave):
  RELATIVE_SCALE = 1.  # replicate C4SpmdLlamaXLHeadResTHFFN16DynW0003LearnDiag

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWTanh(C4SpmdLlamaXLHeadResTHFFN16DynW0003LearnDiag):
  DW_ACTIVATION_CLS = layers.Tanh  # v4 0.142
  PROBS_OUTPUT_ACTIVATION_CLS = layers.ReLU

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWTanhSave(C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWTanh):
  SAVE_ON_STEPS = [10460, 10490, 10530, 10540, 10570, 11120, 11140, 11150, 13570, 13610, 13640] + list(range(7000, 25000, 1000))

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWTanhSaveAlignRelScale(C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWTanhSave):
  RELATIVE_SCALE = 1.  # replicate C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWTanh

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWClip(C4SpmdLlamaXLHeadResTHFFN16DynW0003LearnDiag):
  DW_ACTIVATION_CLS = activations.Clip  # v4 0.142
  PROBS_OUTPUT_ACTIVATION_CLS = layers.ReLU

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16DynW0003OnlyLearnDiagDWTanh(C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWTanh):
  TGT_DEPENDENT = False  # v4 0.158
  SRC_DEPENDENT = False

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16OnlyDynW0003LearnDiagDWTanh(C4SpmdLlamaXLResTHFFN16DynW0003LearnDiagDWTanh):
  USE_STATIC_W = False  # v4 0.14

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16OnlyDynWHD32LearnDiagDWTanh(C4SpmdLlamaXLResTHFFN16OnlyDynW0003LearnDiagDWTanh):
  DYNAMIC_W_HIDDEN_DIM = 32  # v4 wrong code 0.043 
  DW_HIDDEN_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16OnlyDynWHD32LearnDiagDWBiasTanh(C4SpmdLlamaXLResTHFFN16OnlyDynWHD32LearnDiagDWTanh):
  use_dw_bias = True  # v4 0.095

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16OnlyDynWHD16LearnDiagDWBiasTanh(C4SpmdLlamaXLResTHFFN16OnlyDynWHD32LearnDiagDWTanh):
  DYNAMIC_W_HIDDEN_DIM = 16  # v4 wrong code w/o bias 0.063, v4 0.112
  use_dw_bias = True

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16OnlyDynW001HD16LearnDiagDWBiasTanh(C4SpmdLlamaXLResTHFFN16OnlyDynWHD16LearnDiagDWBiasTanh):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.001)  # v3 0.083

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16OnlyDynW003HD16LearnDiagDWBiasTanh(C4SpmdLlamaXLResTHFFN16OnlyDynWHD16LearnDiagDWBiasTanh):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.003)

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN16OnlyDynWHD16LearnDiagDWBias(C4SpmdLlamaXLResTHFFN16OnlyDynWHD16LearnDiagDWBiasTanh):
  DW_ACTIVATION_CLS = None  # v3 0.084

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN4OnlyDynW0003LearnDiagDWTanh(C4SpmdLlamaXLResTHFFN16OnlyDynW0003LearnDiagDWTanh):
  LOGITS_SQUEEZE_RATIO = 4  # v4 0.140
  PROBS_SQUEEZE_RATIO = 4

@experiment_registry.register
class C4SpmdLlamaXLResTHFFN4LogitsGELUOnlyDynW0003LearnDiagDWTanh(C4SpmdLlamaXLResTHFFN4OnlyDynW0003LearnDiagDWTanh):
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU  # v4 0.140
  # USE_BIAS = True
  USE_SQUEEZE_BIAS = True

@experiment_registry.register
class C4SpmdLlamaXLHead32x64ResTHFFN32DynW0003LearnDiag(C4SpmdLlamaXLHeadResTHFFN16DynW0003LearnDiag):
  NUM_HEADS = 32  # v4 0.093
  DIMS_PER_HEAD = 64
  LOGITS_SQUEEZE_RATIO = 32
  PROBS_SQUEEZE_RATIO = 32

@experiment_registry.register
class C4SpmdLlamaMediumResTHLearnDiag(C4SpmdLlamaMediumResTH):
  LEARNABLE_DIAG = True  # v4 0.503! vs C4SpmdLlamaMediumResTHAbsorbRes
  SCALE_INIT = WeightInit.Gaussian(0.04)

@experiment_registry.register
class C4SpmdLlamaMediumResTHAbsorbRes(C4SpmdLlamaMediumResTH):
  ABSORB_RESIDUAL = True  # 0.355 v4 0.449, v5 0.298~0.376?! very unstable

@experiment_registry.register
class C4SpmdLlamaMediumResTHFP32logitsprobs(C4SpmdLlamaMediumResTH):
  FLOAT32_LOGITS = True
  FLOAT32_PROBS = True  # 0.152

@experiment_registry.register
class C4SpmdLlamaMediumResTHAbsorbResFP32logitsprobs(C4SpmdLlamaMediumResTHAbsorbRes):
  FLOAT32_LOGITS = True
  FLOAT32_PROBS = True  # 0.343

@experiment_registry.register
class C4SpmdLlamaMediumResTHLeftMul(C4SpmdLlamaMediumResTH):
  LEFT_MUL = True  # v4 0.336

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN4(C4SpmdLlamaMediumResTH):
  LOGITS_SQUEEZE_RATIO = 4  # v4 0.32
  PROBS_SQUEEZE_RATIO = 4
  USE_BIAS = False
  LOGITS_ABSORB_RESIDUAL = False
  PROBS_ABSORB_RESIDUAL = False

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN8SkipWD(C4SpmdLlamaMediumResTHFFN4):
  LOGITS_SQUEEZE_RATIO = 8  # v4 0.33
  PROBS_SQUEEZE_RATIO = 8
  SKIP_FFN_WEIGHT_DECAY = True

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16SkipWD(C4SpmdLlamaMediumResTHFFN4):
  LOGITS_SQUEEZE_RATIO = 16  # v4 0.48
  PROBS_SQUEEZE_RATIO = 16
  SKIP_FFN_WEIGHT_DECAY = True

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16LogitsGELU(C4SpmdLlamaMediumResTHFFN16SkipWD):
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU
  USE_BIAS = True

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16GELU(C4SpmdLlamaMediumResTHFFN16LogitsGELU):
  PROBS_SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN4LearnDiag(C4SpmdLlamaMediumResTHFFN4):
  LEARNABLE_DIAG = True
  TRANSPOSE = True  # 0.307

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN4DynGate(C4SpmdLlamaMediumResTHFFN4):
  DYNAMIC_SQUEEZE_GATE_ACT_CLS = layers.SiLU  # v4 0.317 transpose==no_transpose
  GATE_RELATIVE_SCALE = 0.01
  TRANSPOSE = True
  SKIP_FFN_WEIGHT_DECAY = True
  
@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN4DynGateSigmoid(C4SpmdLlamaMediumResTHFFN4DynGate):
  DYNAMIC_SQUEEZE_GATE_ACT_CLS = layers.Sigmoid  # v4 0.317
  GATE_RELATIVE_SCALE = 0.05

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN4DynAddGate(C4SpmdLlamaMediumResTHFFN4DynGate):
  ADDICTIVE_GATE = True  # v4 0.316
  GATE_RELATIVE_SCALE = 0.0002

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN4DynW006(C4SpmdLlamaMediumResTHFFN4):  # to delete
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.006)  # v4 0.194
  TRANSPOSE = True
  SKIP_FFN_WEIGHT_DECAY = True

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW006(C4SpmdLlamaMediumResTHFFN16SkipWD):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.006)  # v4 0.314
  TRANSPOSE = True
  SKIP_FFN_WEIGHT_DECAY = True

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW0006(C4SpmdLlamaMediumResTHFFN16DynW006):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0006)

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW0003(C4SpmdLlamaMediumResTHFFN16DynW006):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0003)

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW0001(C4SpmdLlamaMediumResTHFFN16DynW006):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0001)

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW0003SrcInd(C4SpmdLlamaMediumResTHFFN16DynW0003):
  SRC_DEPENDENT = False # v4 0.394

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiag(C4SpmdLlamaMediumResTHFFN16DynW0003):
  # RELATIVE_SCALE = 1.  # actual static w2.init.relative_scale=1 due to init_fn bug
  LEARNABLE_DIAG = True # v4 0.308

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiagDWTanh(C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiag):
  DW_ACTIVATION_CLS = layers.Tanh  # v3 0.163 much slower than v4 with C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiag
  PROBS_OUTPUT_ACTIVATION_CLS = layers.ReLU

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW0003LearnOnlyDiagDWTanh(C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiag):
  RELATIVE_SCALE = 1.
  DW_ACTIVATION_CLS = layers.Tanh
  SAVE_ON_STEPS = [200, 1000, 5000] + list(range(10000, 60000, 10000))

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiagRelScale10th(C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiag):
  PROBS_OUTPUT_ACTIVATION_CLS = layers.ReLU
  SAVE_ON_STEPS = [200, 1000, 5000] + list(range(10000, 60000, 10000))

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiagOnlyRelScale10th(C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiag):
  PROBS_OUTPUT_ACTIVATION_CLS = None  # in fact unnecessary
  SAVE_ON_STEPS = [200, 1000, 5000] + list(range(10000, 60000, 10000))

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiagSeqLen512(C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiag):
  MAX_SEQ_LEN = 512  # v4 0.46
  PERCORE_BATCH_SIZE = 32

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiagSeqLen2K(C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiag):
  MAX_SEQ_LEN = 2048  # v4 0.184
  PERCORE_BATCH_SIZE = 8

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16OnlyDynW0003OnlyLearnDiagDWTanh(C4SpmdLlamaMediumResTHFFN16DynW0003):
  DW_ACTIVATION_CLS = layers.Tanh
  PROBS_OUTPUT_ACTIVATION_CLS = layers.ReLU
  USE_STATIC_W = False
  LEARNABLE_DIAG = True  # v3 0.364
  TGT_DEPENDENT = False
  SRC_DEPENDENT = False

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16OnlyDynW0003OnlyProbsLearnDiagDWTanh(C4SpmdLlamaMediumResTHFFN16OnlyDynW0003OnlyLearnDiagDWTanh):
  PROJECT_LOGITS = False
  PROBS_LEARNABLE_DIAG = True # v3 0.373

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16OnlyDynW0003(C4SpmdLlamaMediumResTHFFN16DynW0003):
  USE_STATIC_W = False  # 0.401!? much faster than C4SpmdLlamaMediumResTHFFN16OnlyDynW0003LearnDiag
  
@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16OnlyDynW0003LearnDiag(C4SpmdLlamaMediumResTHFFN16DynW0003):
  USE_STATIC_W = False
  LEARNABLE_DIAG = True  # v4 0.320

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN16OnlyDynW0003LearnDiagDWTanh(C4SpmdLlamaMediumResTHFFN16OnlyDynW0003LearnDiag):
  DW_ACTIVATION_CLS = layers.Tanh  # v4 0.267??? vs C4SpmdLlamaMediumResTHFFN16OnlyDynW0003LearnDiag
  PROBS_OUTPUT_ACTIVATION_CLS = layers.ReLU

@experiment_registry.register
class C4SpmdLlamaMediumHead32x32ResTHFFN32(C4SpmdLlamaMediumResTH):
  NUM_HEADS = 32
  DIMS_PER_HEAD = 32
  LOGITS_SQUEEZE_RATIO = 32  #
  PROBS_SQUEEZE_RATIO = 32
  USE_BIAS = False
  LOGITS_ABSORB_RESIDUAL = False
  PROBS_ABSORB_RESIDUAL = False
  SKIP_FFN_WEIGHT_DECAY = True

@experiment_registry.register
class C4SpmdLlamaMediumHead8x128ResTHFFN8(C4SpmdLlamaMediumHead32x32ResTHFFN32):
  NUM_HEADS = 8
  DIMS_PER_HEAD = 128
  LOGITS_SQUEEZE_RATIO = 8  #
  PROBS_SQUEEZE_RATIO = 8

@experiment_registry.register
class C4SpmdLlamaMediumHead32x32ResTHFFN32DynW0003(C4SpmdLlamaMediumHead32x32ResTHFFN32):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0003)  # v4 0.19
  TRANSPOSE = True

@experiment_registry.register
class C4SpmdLlamaMediumHead8x128ResTHFFN8DynW0006(C4SpmdLlamaMediumHead8x128ResTHFFN8):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0006)  # v4 0.465
  TRANSPOSE = True

@experiment_registry.register
class C4SpmdLlamaMediumResTHFFN8DynW0003(C4SpmdLlamaMediumResTHFFN8SkipWD):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0003)  # v4 0.207
  TRANSPOSE = True

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELUProbs(C4SpmdLlamaMediumResTH):
  LOGITS_SQUEEZE_RATIO = 2  
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUProbs(C4SpmdLlamaXLResTH):
  LOGITS_SQUEEZE_RATIO = 2   # v4 0.112 v4 absorbres 0.115
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU
  LOGITS_ABSORB_RESIDUAL = False

@experiment_registry.register
class C4SpmdLlamaXLHead16x128ResTHLogitsFFN2GELUProbs(C4SpmdLlamaXLResTHLogitsFFN2GELUProbs):
  NUM_HEADS = 16  # v4 0.159
  DIMS_PER_HEAD = 128
  # RELATIVE_SCALE = 1.0  # init_fn has bug. Intended scale is 0.1, but the actual scale is 1.0

@experiment_registry.register
class C4SpmdLlamaXLHead16x128ResTHLogitsFFN2GELUProbsRelScale025(C4SpmdLlamaXLHead16x128ResTHLogitsFFN2GELUProbs):
  RELATIVE_SCALE = 0.025

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUProbsDynW0003LearnDiagDWTanh(C4SpmdLlamaXLHead16x128ResTHLogitsFFN2GELUProbs):
  TRANSPOSE = True
  PROBS_DYNAMIC_W_INIT = WeightInit.Gaussian(0.0003)  # v3 0.096
  PROBS_LEARNABLE_DIAG = True
  PROBS_DW_ACTIVATION_CLS = layers.Tanh
  PROBS_OUTPUT_ACTIVATION_CLS = layers.ReLU

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagProbsDWTanh(C4SpmdLlamaXLResTHLogitsFFN2GELUProbsDynW0003LearnDiagDWTanh):
  # loss worse than C4SpmdLlamaXL baseline and goes to nan at ~1600 step if ONLY LogitsDynW is used in place of ProbsDynW (due to bug)
  LOGITS_DYNAMIC_W_INIT = WeightInit.Gaussian(0.0003)  # v4 0.122
  LOGITS_LEARNABLE_DIAG = True

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagProbsDWTanh):
  LOGITS_DW_ACTIVATION_CLS = layers.Tanh  # v4 0.144

@experiment_registry.register
class MQAXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh):
  NUM_KV_HEADS = 1  # v4 0.109
  HIDDEN_DIMS = 6784
  USE_SQUEEZE_BIAS = False
  SUMMARY_VERBOSITY = 3
  PROBS_ABSORB_RESIDUAL = False
  
@experiment_registry.register
class MQAXLResTHLogitsFFN2GELUDynWHD32LearnDiagDWTanh(MQAXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh):
  DYNAMIC_W_HIDDEN_DIM = 32  # v4 0.109
  DW_HIDDEN_ACTIVATION_CLS = layers.GELU
  USE_DW_HIDDEN_BIAS = False

@experiment_registry.register
class LlamaXL16x64ResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh):
  NUM_HEADS = 16
  DIMS_PER_HEAD = 64  # v4 0.109
  HIDDEN_DIMS = 6912
  USE_SQUEEZE_BIAS = False
  SUMMARY_VERBOSITY = 3
  PROBS_ABSORB_RESIDUAL = False

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GateSiLUDynW0003LearnDiagDWTanh(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh):
  SQUEEZE_ACTIVATION_CLS = layers.Identity
  SQUEEZE_GATE_ACTIVATION_CLS = layers.SiLU  # v4 0.104
  SUMMARY_VERBOSITY = 3
  PROBS_ABSORB_RESIDUAL = False

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWGateSiLU(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh):
  DW_ACTIVATION_CLS = None
  DW_GATE_ACTIVATION_CLS = layers.SiLU  # v4 0.111 absorres 0.107!?
  PROBS_OUTPUT_ACTIVATION_CLS = None

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW001LearnDiagDW1GateSiLU(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWGateSiLU):
  DW_GATE_WEIGHTS = ['qw1', 'kw1']  # v4 0.111
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.001)

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW001LearnDiagDW1GateSigmoid(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW001LearnDiagDW1GateSiLU):
  DW_GATE_ACTIVATION_CLS = layers.Sigmoid  # v4 0.111

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWDDGateSiLU(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWGateSiLU):
  DD_GATE_ACTIVATION_CLS = layers.SiLU  # v4 0.110

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW004LearnDiagDWDDGateSiLU(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWDDGateSiLU):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.004)

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNorm(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh):
  DW_ACTIVATION_CLS = None
  DW1_NORM_CLS = normalizations.RmsNormNoScale  # v4 0.128
  PROBS_OUTPUT_ACTIVATION_CLS = None
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00003)
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00012) # DYNAMIC_W_INIT.scale * sqrt(n)
  SAVE_ON_STEPS = [0, 200, 1000, 5000] + list(range(10000, 60000, 10000))
  SKIP_BIAS = True  # squeeze_bias is in fact not used due to bug

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUBiasDynW00003LearnDiagDW1RmsNorm(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNorm):
  SKIP_BIAS = False  # v4 0.112
  SUMMARY_VERBOSITY = 3
  PROBS_ABSORB_RESIDUAL = False
  SAVE_ON_STEPS = None

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormNoAbsorbRes(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNorm):
  DW1_NORM_BIAS_INIT = 1e-6
  SKIP_BIAS = True
  # SUMMARY_VERBOSITY = 3
  # PROBS_ABSORB_RESIDUAL = False
  PROBS_ABSORB_RESIDUAL = True
  SAVE_ON_STEPS = None

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBias1e_6(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNorm):
  # curiouly, not exactly reproducable, maybe due to subtle difference in add_summaries (more likely) or absorb_res (less likely)
  DW1_NORM_BIAS_INIT = 1e-6  # v4 0.112  w/o summary 0.134
  SUMMARY_VERBOSITY = 3  # 3 -> 9 at step 38700, back to 3 at step 53000 after loss spike
  PROBS_ABSORB_RESIDUAL = False  # TODO: disable absorb_res may decrease loss a little bit?
  SAVE_ON_STEPS = None
  SKIP_BIAS = True  # added after fix the bias not used bug to *preserve* the bug when continuing training at step 29500

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBias1e_5(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNorm):
  DW1_NORM_BIAS_INIT = 1e-5  # v4 0.112
  SUMMARY_VERBOSITY = 3
  PROBS_ABSORB_RESIDUAL = False
  SAVE_ON_STEPS = None

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBias1e_6Const(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNorm):
  DW1_NORM_BIAS_CONST = 1e-6
  PROBS_ABSORB_RESIDUAL = False # v4 0.134 why faster than C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNorm?
  PROBS_ABSORB_RESIDUAL = True # v4 0.128 because absorbres is slower!!??
  SAVE_ON_STEPS = None

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormFixBias(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBias1e_6):
  # loss nan at very beginning after fix dw1_norm_bias
  USE_SQUEEZE_BIAS = False

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormSquareBias(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBias1e_6):
  USE_SQUEEZE_BIAS = False  # only fix dw1_norm_bias but not squeeze_bias
  SQUARE_DW1_NORM_BIAS = True  # v4 0.112
  DW1_NORM_BIAS_INIT = 1e-3

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD32LearnDiagDW1RmsNormBias1e_6(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBias1e_6):
  DYNAMIC_W_HIDDEN_DIM = 32  # v4 0.112
  DW_HIDDEN_ACTIVATION_CLS = layers.GELU
  SKIP_BIAS = True

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD32NoBiasLearnDiagDW1RmsNorm(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD32LearnDiagDW1RmsNormBias1e_6):
  # First 3220 steps has dynamic_d_hidden_dim params dd1, qd, kw as well as dd, but changed little, probably only due to weight decay
  # Some params's and activations std's curves turn @9020 when C4SpmdLlamaXLResTHLogitsFFN2GELUDynWDHD32NoBiasLearnDiagDW1RmsNorm stops training
  # Another turn @~22480 due to adding dw_cap @22400, which is too late, apparently hurting performance
  USE_DW_HIDDEN_BIAS = False  # v4 0.112
  SUMMARY_VERBOSITY = 3
  # dd caps duplicate with DW_ACTIVATION_CLS = layers.Tanh
  # DW_CAP = {'qw2': 2., 'kw2': 2., 'dd': 1.}  # add @1600 in _fix run, @22400 in original run
  # PROBS_DW_CAP = {'qw2': 2., 'kw2': 2., 'qdd': 1., 'kdd': 1.}  # switch @13800 in _fix run, @32200 in original run, accidentally removed for original run @44300
  SAVE_ON_STEPS = list(range(10000, 70000, 10000))  # add @27400
  # LOGITS/PROBS_DW_ACTIVATION_CLS = layers.Tanh  # wrongly inherited from C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh, only effective on dd

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWDHD32NoBiasLearnDiagDW1RmsNorm(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD32NoBiasLearnDiagDW1RmsNorm):
  # has dd params, but changed little, probably these params are not used and changed due to weight decay
  DYNAMIC_D_HIDDEN_DIM = 16  # v4 0.112
  SUMMARY_VERBOSITY = 3

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN1GELUDynWHD32DW1RmsNorm(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD32NoBiasLearnDiagDW1RmsNorm):
  # unnecesarily add '_fix' suffix
  LOGITS_SQUEEZE_RATIO = 1  # v4 0.110
  DW_CAP = {'qw2': 2., 'kw2': 2., 'dd': 1.}  # add @700

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNorm(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD32NoBiasLearnDiagDW1RmsNorm):
  DYNAMIC_SQUEEZE_RATIO = 8  # v4 0.093
  DYNAMIC_W_HIDDEN_DIM = 64
  # dd caps duplicate with DW_ACTIVATION_CLS = layers.Tanh
  # DW_CAP = {'qw2': 2., 'kw2': 2., 'dd': 1.}  # add @600
  # PROBS_DW_CAP = {'qw2': 2., 'kw2': 2., 'qdd': 1., 'kdd': 1.}  # switch @10300
  SAVE_ON_STEPS = list(range(10000, 70000, 10000))
  # CHECKPOINT_MAX_TO_KEEP = 20  # for last steps of _nocap run

  # LOGITS/PROBS_DW_ACTIVATION_CLS = layers.Tanh  # wrongly inherited from C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh, only effective on dd
  # for _nocap run, use PROBS_DW_CAP before step 6200 (though commented out in this class, somehow wrongly inherited from parent),
  # and remove PROBS_DW_CAP afterwards on a reboot (dd cap remains due to DW_ACTIVATION_CLS = layers.Tanh bug), causing a slight turn of params and activations @6200

@experiment_registry.register  # praxis 29dcf7b, paxml bcccfea 
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormNoDDTanh(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNorm):
  LOGITS_DW_ACTIVATION_CLS = None
  PROBS_DW_ACTIVATION_CLS = None

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormNoDDTanhChunk128(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormNoDDTanh):
  # QUERY_CHUNK_SIZE = 64  # v4 0.131
  # QUERY_CHUNK_SIZE = 256  # v4 0.135
  QUERY_CHUNK_SIZE = 128  # v4 +summary 0.142, 0.152, -transpose 0.153, -transpose+tri-einsum 0.153, -transpose+loop 0.162!
  TRANSPOSE = False
  LOOP_OVER_DYNAMIC_HD = True
  SUMMARY_VERBOSITY = 9
  # _loophd run praxis@a929fe4

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormNoDDTanhQKNorm(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormNoDDTanhChunk128):
  QK_NORM = True  # v4 0.159, rmsnorm 0.160

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNorm):
  QUERY_CHUNK_SIZE = 128  # v4 0.16, v3 0.132
  TRANSPOSE = False
  LOOP_OVER_DYNAMIC_HD = True
  QK_NORM = True
  SUMMARY_VERBOSITY = 9

@experiment_registry.register
class C4SpmdLlamaXLNoSwiGLUResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole(C4SpmdLlamaXLNoSwiGLU, C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  QK_NORM = False  # v3 0.138

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64Win128(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [None, 128]  # v3 0.148

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64Win128LHChunk128(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64Win128):
  EMBEDDING_LOOKUP_STYLE = 'index' # v3 0.152
  LM_HEAD_CHUNK_SIZE = 128  # v3 0.146

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64Win256(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  EMBEDDING_LOOKUP_STYLE = 'index'  # @600 v3 0.150
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [None, 256]  # v3 0.143

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64Win256Rev(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  EMBEDDING_LOOKUP_STYLE = 'index'  # v3 MI/IM 0.1496/0.1496, merge_dw_proj MI/IM 0.1506/0.1506
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64Win384(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  EMBEDDING_LOOKUP_STYLE = 'index'  # v3 0.146
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [None, 384]

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD641to4Win128(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  EMBEDDING_LOOKUP_STYLE = 'index'  # @200 v3 0.166
  NUM_EARLY_LAYERS = 4
  NUM_LAYERS_PER_BLOCK_EARLY = 4
  WINDOW_SIZE = [None, 128, None, 128]
  LOGITS_USE_STATIC_W_EARLY = True
  PROBS_USE_STATIC_W_EARLY = True
  LOGITS_DYNAMIC_W_INIT_EARLY = WeightInit.Gaussian(0.00003)
  PROBS_DYNAMIC_W_INIT_EARLY = WeightInit.Gaussian(0.00003)
  LOGITS_DYNAMIC_D_INIT_EARLY = WeightInit.Gaussian(0.00012)
  PROBS_DYNAMIC_D_INIT_EARLY = WeightInit.Gaussian(0.00012)

  NUM_LAYERS_PER_BLOCK = 4
  WINDOW_SIZE = [None, 128, None, 128]  # v3 0.159
  LOGITS_USE_STATIC_W = [False, False, False, True]
  PROBS_USE_STATIC_W = [True, True, True, True]
  LOGITS_DYNAMIC_W_INIT = [None, None, None, WeightInit.Gaussian(0.00003)]
  PROBS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003)]
  LOGITS_DYNAMIC_D_INIT = [None, None, None, WeightInit.Gaussian(0.00012)]
  PROBS_DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012)]

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD641to4Win128NoEarly(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  EMBEDDING_LOOKUP_STYLE = 'index'
  NUM_LAYERS_PER_BLOCK = 4
  WINDOW_SIZE = [None, 128, None, 128]  #
  LOGITS_USE_STATIC_W = [False, False, False, True]
  PROBS_USE_STATIC_W = [True, True, True, True]
  LOGITS_DYNAMIC_W_INIT = [None, None, None, WeightInit.Gaussian(0.00003)]
  PROBS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003)]
  LOGITS_DYNAMIC_D_INIT = [None, None, None, WeightInit.Gaussian(0.00012)]
  PROBS_DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012)]

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD641to4Win256(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  EMBEDDING_LOOKUP_STYLE = 'index'
  NUM_EARLY_LAYERS = 4  # v3 0.156
  NUM_LAYERS_PER_BLOCK_EARLY = 4
  WINDOW_SIZE_EARLY = [256, None, 256, None]
  LOGITS_USE_STATIC_W_EARLY = True
  PROBS_USE_STATIC_W_EARLY = True
  LOGITS_DYNAMIC_W_INIT_EARLY = WeightInit.Gaussian(0.00003)
  PROBS_DYNAMIC_W_INIT_EARLY = WeightInit.Gaussian(0.00003)
  LOGITS_DYNAMIC_D_INIT_EARLY = WeightInit.Gaussian(0.00012)
  PROBS_DYNAMIC_D_INIT_EARLY = WeightInit.Gaussian(0.00012)

  NUM_LAYERS_PER_BLOCK = 4
  WINDOW_SIZE = [256, None, 256, None]
  LOGITS_USE_STATIC_W = [False, False, False, True]
  PROBS_USE_STATIC_W = [True, True, True, True]
  LOGITS_DYNAMIC_W_INIT = [None, None, None, WeightInit.Gaussian(0.00003)]
  PROBS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003)]
  LOGITS_DYNAMIC_D_INIT = [None, None, None, WeightInit.Gaussian(0.00012)]
  PROBS_DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012)]

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD641to4Win256NoEarly(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  EMBEDDING_LOOKUP_STYLE = 'index'
  NUM_LAYERS_PER_BLOCK = 4  # v3 0.166
  WINDOW_SIZE = [256, None, 256, None]
  LOGITS_USE_STATIC_W = [False, True, False, False]
  PROBS_USE_STATIC_W = [True, True, True, True]
  LOGITS_DYNAMIC_W_INIT = [None, WeightInit.Gaussian(0.00003), None, None]
  PROBS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003)]
  LOGITS_DYNAMIC_D_INIT = [None, WeightInit.Gaussian(0.00012), None, None]
  PROBS_DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012)]

@experiment_registry.register
class C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  LOGITS_SQUEEZE_RATIO = None  # v3 0.134

@experiment_registry.register
class C4SpmdLlamaXLResTHNG2DynWFFN8HD16(C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  ICI_MESH_SHAPE = [1, 32, 2]
  NUM_GROUPS = 2  # v3 0.134
  DYNAMIC_W_HIDDEN_DIM = 16

@experiment_registry.register
class C4SpmdLlamaXLResTHNG2DynWFFN4HD32(C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  ICI_MESH_SHAPE = [1, 32, 2]
  NUM_GROUPS = 2  # v3 0.123!?
  DYNAMIC_SQUEEZE_RATIO = 4
  DYNAMIC_W_HIDDEN_DIM = 32

@experiment_registry.register
class C4SpmdLlamaXLResTHDynW0003FFN8HD64DW1RmsNormWhole(C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  EMBEDDING_LOOKUP_STYLE = 'index' # @100 v3 0.140
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0003)  # v3 0.135
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.0003)

@experiment_registry.register
class C4SpmdLlamaXLResTHDynWNoFFNHD64DW1RmsNormWhole(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  DECOMPOSE_DYNAMIC_W = False  # v4 0.145 vs C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole v4 0.16
  DYNAMIC_D_INIT = None

@experiment_registry.register
class C4SpmdLlamaXLResTHDynWNoFFNHD128DW1RmsNormWhole(C4SpmdLlamaXLResTHDynWNoFFNHD64DW1RmsNormWhole):
  DYNAMIC_W_HIDDEN_DIM = 128  # v4 0.145

@experiment_registry.register
class C4SpmdLlamaXLResTHDynWNoFFNHD128DW1RmsNormWholeDD(C4SpmdLlamaXLResTHDynWNoFFNHD128DW1RmsNormWhole):
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00012)  # v4 0.145

@experiment_registry.register
class C4SpmdLlamaXLResTHDynWNoFFNHD256DW1RmsNormWhole(C4SpmdLlamaXLResTHDynWNoFFNHD64DW1RmsNormWhole):
  DYNAMIC_W_HIDDEN_DIM = 256  # v4 0.144

@experiment_registry.register
class C4SpmdLlamaXLResTHDynWNoFFNDW1RmsNormWhole(C4SpmdLlamaXLResTHDynWNoFFNHD64DW1RmsNormWhole):
  DYNAMIC_W_HIDDEN_DIM = None  # v4 0.145

@experiment_registry.register
class C4SpmdLlamaXLFFN8192(C4SpmdLlamaXLHead16x128):
  NUM_LAYERS = 21
  HIDDEN_DIMS = 8192

@experiment_registry.register
class C4SpmdLlamaXLFFN10880(C4SpmdLlamaXLHead16x128):
  NUM_LAYERS = 17
  HIDDEN_DIMS = 10880

@experiment_registry.register
class C4SpmdLlamaXLFFN8192ResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole(C4SpmdLlamaXLFFN8192, C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  pass  # v3 0.152 vs C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole v3 0.132

@experiment_registry.register
class C4SpmdLlamaXLFFN10880ResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole(C4SpmdLlamaXLFFN10880, C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  pass  # v3 0.164

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNorm11to4(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  NUM_EARLY_LAYERS = 4
  NUM_LAYERS_PER_BLOCK = 4

  LOGITS_USE_STATIC_W_EARLY = True
  PROBS_USE_STATIC_W_EARLY = True
  LOGITS_DYNAMIC_W_INIT_EARLY = WeightInit.Gaussian(0.00003)
  PROBS_DYNAMIC_W_INIT_EARLY = WeightInit.Gaussian(0.00003)
  LOGITS_DYNAMIC_D_INIT_EARLY = WeightInit.Gaussian(0.00012)
  PROBS_DYNAMIC_D_INIT_EARLY = WeightInit.Gaussian(0.00012)

  LOGITS_USE_STATIC_W = [True, False, False, False]  # v3 0.148
  PROBS_USE_STATIC_W = [True, True, True, True]
  LOGITS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), None, None, None]
  PROBS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003)]
  # LOGITS_DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), None, None, None] # should be trained with this
  LOGITS_DYNAMIC_D_INIT_EARLY = [WeightInit.Gaussian(0.00012), None, None, None]  # actually be trained with this, so LOGITS_DYNAMIC_D is not used
  PROBS_DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012)]

@experiment_registry.register
class C4SpmdLlamaXLFFN8192ResTHDynWFFN8HD64DW1RmsNorm11to4(C4SpmdLlamaXLFFN8192, C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  NUM_EARLY_LAYERS = 1
  LOGITS_SQUEEZE_RATIO = None  # v3

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalf(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole):
  NUM_EARLY_LAYERS = 2
  NUM_LAYERS_PER_BLOCK = 2

  PROJECT_LOGITS_EARLY = True
  LOGITS_USE_STATIC_W_EARLY = True
  PROBS_USE_STATIC_W_EARLY = True
  LOGITS_DYNAMIC_W_INIT_EARLY = WeightInit.Gaussian(0.00003)
  PROBS_DYNAMIC_W_INIT_EARLY = WeightInit.Gaussian(0.00003)
  DYNAMIC_D_INIT_EARLY = WeightInit.Gaussian(0.00012)
  DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012)]

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalfOnlyProbs(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalf):
  # NUM_EARLY_LAYERS = 0
  NUM_LAYERS_PER_BLOCK = 1
  
  LOGITS_USE_STATIC_W = False
  PROBS_USE_STATIC_W = True  # v4 w early_layers 0.171?, w/o early_layers 0.173?
  LOGITS_DYNAMIC_W_INIT = None
  PROBS_DYNAMIC_W_INIT = WeightInit.Gaussian(0.00003)
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00012)
  # TRAINING_NUM_BATCHES_TO_SKIP = 0

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalfOnlyProbsNoDD(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalfOnlyProbs):
  # NUM_EARLY_LAYERS = 0
  PROJECT_LOGITS = False  # v4 w/o early_layers 0.184, w early_layers 0.182
  PROBS_DYNAMIC_D_INIT = WeightInit.Gaussian(0.00012)

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalfOnlyProbsNoDDNoEarly(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalfOnlyProbsNoDD):
  NUM_EARLY_LAYERS = 0  # v3 w/o early_layers 0.160 vs OnlyProbsNoDD v4 0.184

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalfOnlyDD(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalfOnlyProbs):
  NUM_EARLY_LAYERS = 0
  LOGITS_USE_STATIC_W = False
  PROBS_USE_STATIC_W = False
  LOGITS_DYNAMIC_W_INIT = None
  PROBS_DYNAMIC_W_INIT = None
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00012)  # 0.215!

@experiment_registry.register
class C4SpmdLlamaXLBaseline(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalfOnlyProbs):
  NUM_EARLY_LAYERS = 0
  PROJECT_LOGITS = False
  PROJECT_PROBS = False  # 0.215

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalfAligned(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalf):
  LOGITS_USE_STATIC_W = [True, False]  # v4 all dd 0.184!
  PROBS_USE_STATIC_W = [True, False]
  LOGITS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), None]
  PROBS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), None]
  # DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), None]  # v4 half dd 0.184
  # DYNAMIC_D_INIT = [None, None]
  # TRAINING_NUM_BATCHES_TO_SKIP = 0

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalfAlignedNoDD(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalfAligned):
  DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), None]

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormInterleaveda(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalf):
  LOGITS_USE_STATIC_W = [True, False]  # v4 all dd 0.178
  PROBS_USE_STATIC_W = [False, True]
  LOGITS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), None]
  PROBS_DYNAMIC_W_INIT = [None, WeightInit.Gaussian(0.00003)]

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormQuarter(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormHalf):
  PROJECT_LOGITS = False
  LOGITS_USE_STATIC_W = [False, False]  # v3 0.172 vs HalfOnlyProbsNoDD v4 0.182
  PROBS_USE_STATIC_W = [True, False]
  LOGITS_DYNAMIC_W_INIT = [None, None]
  PROBS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), None]
  LOGITS_DYNAMIC_D_INIT = [None, None]
  PROBS_DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), None]

@experiment_registry.register
class XXLResTHLogitsFFN2GELUDynWFFN8HD64Chunk128(C4SpmdLlamaXXL, C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormNoDDTanh):
  NUM_LAYERS = 13
  PERCORE_BATCH_SIZE = 16
  ICI_MESH_SHAPE = [1, 32, 1]
  SUMMARY_INTERVAL_STEPS = 5

  QUERY_CHUNK_SIZE = 128  # v4 0.228
  TRANSPOSE = False
  SUMMARY_VERBOSITY = 9

@experiment_registry.register
class XXLResTHLogitsFFN2GELUDynWFFN12HD96Chunk128(XXLResTHLogitsFFN2GELUDynWFFN8HD64Chunk128):
  DYNAMIC_SQUEEZE_RATIO = 12
  DYNAMIC_W_HIDDEN_DIM = 96  # v4 0.23, SW 0.292
  QUERY_CHUNK_SIZE = 128  # None v4 0.17, SW 0.228
  PROBS_ABSORB_RESIDUAL = False  # True v4 0.222, SW 0.299, SW NoChunk 0.234
  # NoMLP  # v4 SW 0.471

@experiment_registry.register
class XXLBaseline(C4SpmdLlamaXXL):
  NUM_LAYERS = 13  # v4 0.325, Chunk128 0.331
  PERCORE_BATCH_SIZE = 16
  ICI_MESH_SHAPE = [1, 32, 1]
  SUMMARY_INTERVAL_STEPS = 5

  SUMMARY_VERBOSITY = 9
  # QUERY_CHUNK_SIZE = 128
  # NoMLP  # v4 0.561

@experiment_registry.register
class Llama7B128x1DynWFFN16HD128Whole(C4SpmdLlama7B128x1, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  ICI_MESH_SHAPE = [1, 128, 1]
  NUM_GROUPS = 1
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00013)  # sqrt(1 / HD) * 2 / (M + I) * 0.025, total_scale = 0.02
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.0002) # sqrt(2 / (D + M * 2)) * 0.01, total_scale = 0.02
  DYNAMIC_SQUEEZE_RATIO = 16 #// 2
  DYNAMIC_W_HIDDEN_DIM = 128 #* 2
  QUERY_CHUNK_SIZE = 128
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]  # SEP_DW_PROJ v3 error!?, v4 0.0648
  MERGE_DW_PROJ = True # v3 0.0338, v4 0.0650
  SUMMARY_INTERVAL_STEPS = 5

@experiment_registry.register
class Llama7BNG4DynWFFN8HD64Whole(C4SpmdLlama7B128x1, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  ICI_MESH_SHAPE = [1, 128, 1]  # v3 oom shard_dw_proj oom; v4 0.0628 shard_dw_proj 0.0647!? rank2 shard_dw_proj 0.0554
  # ICI_MESH_SHAPE = [1, 32, 4]  # v3 0.0346 shard_dw_proj 0.0346, v4 oom!? shard_dw_proj oom
  NUM_GROUPS = 4
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00013)  # sqrt(1 / HD) * 2 / (M + I) * 0.025, total_scale = 0.02
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.0002) # sqrt(2 / (D + M * 2)) * 0.01, total_scale = 0.02
  DYNAMIC_SQUEEZE_RATIO = 8 #// 2
  DYNAMIC_W_HIDDEN_DIM = 64 #* 2
  QUERY_CHUNK_SIZE = 128
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  MERGE_DW_PROJ = True
  SUMMARY_INTERVAL_STEPS = 5

@experiment_registry.register
class Pythia7BDynWFFN16HD128Win256(Pythia7B, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0001)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.0001) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01
  DYNAMIC_SQUEEZE_RATIO = 16
  DYNAMIC_W_HIDDEN_DIM = 128  # v4 0.0693
  QUERY_CHUNK_SIZE = 128
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  MERGE_DW_PROJ = True 

# @experiment_registry.register
# class Pythia7B128x1DynWFFN16HD128Win256(PythiaInit, Pythia7BDynWFFN16HD128Win256):
#   PERCORE_BATCH_SIZE = 2 * 2 * 2
#   ICI_MESH_SHAPE = [1, 128, 1] 

@experiment_registry.register
class Pythia7B128x1DynWFFN16HD128Win256(Pythia7B128x1, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0001)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.0001) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01
  DYNAMIC_SQUEEZE_RATIO = 16
  DYNAMIC_W_HIDDEN_DIM = 128  # v4 0.0693
  QUERY_CHUNK_SIZE = 128
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  MERGE_DW_PROJ = True
  # SUMMARY_INTERVAL_STEPS = 5

@experiment_registry.register
class Pythia7B128x1NG2DynWFFN8HD128Win256(Pythia7B128x1DynWFFN16HD128Win256):
  ICI_MESH_SHAPE = [1, 128, 1]  # v3 0.0356
  # ICI_MESH_SHAPE = [1, 64, 2]  # v3 0.0356
  # ICI_MESH_SHAPE = [1, 32, 4]  # v3 oom???
  NUM_GROUPS = 2
  DYNAMIC_SQUEEZE_RATIO = 8
  DYNAMIC_W_HIDDEN_DIM = 128
  # SUMMARY_INTERVAL_STEPS = 5

@experiment_registry.register
class Pythia7B128x1NG4DynWFFN8HD64Win256(Pythia7B128x1DynWFFN16HD128Win256):
  NUM_GROUPS = 4
  DYNAMIC_SQUEEZE_RATIO = 8
  DYNAMIC_W_HIDDEN_DIM = 64  # v4 0.0697  # TODO: should be 8*1*2=16?

@experiment_registry.register
class Pythia7B128x1NG4DynWFFN4HD128Win256(Pythia7B128x1DynWFFN16HD128Win256):
  NUM_GROUPS = 4
  DYNAMIC_SQUEEZE_RATIO = 4
  DYNAMIC_W_HIDDEN_DIM = 128

@experiment_registry.register
class Pythia7B128x1DynWFFN8HD256Win256(Pythia7B128x1DynWFFN16HD128Win256):
  DYNAMIC_SQUEEZE_RATIO = 8
  DYNAMIC_W_HIDDEN_DIM = 256  # v3/v4 oom

@experiment_registry.register
class Pythia7B256x1DynWFFN16HD128Win256(PythiaInit, Pythia7BDynWFFN16HD128Win256):
  PERCORE_BATCH_SIZE = 2 * 2
  ICI_MESH_SHAPE = [1, 256, 1] 

@experiment_registry.register
class C4Pythia7B256x1DynWFFN16HD128Win256(DataParams, Pythia7B256x1DynWFFN16HD128Win256):
  EVAL_LOOP_NUM_BATCHES = 50  # v4 0.130

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256(PileDataParams, Pythia7B256x1DynWFFN16HD128Win256):
  SAVE_ON_STEPS = [1000,] + list(range(3000, 143000, 10000))  # restart @900 to add 1000 to save_on_steps
  # _fix run: restart @2090x @5240 @13760
  # _fix_dw1norm run: v4 0.130 restart @2830 v3 0.0683

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256Aligned(PythiaInit7B, PilePythia7B256x1DynWFFN16HD128Win256):
  PYTHIA_ROTARY = True  # restart @10500/19990/29820/40210/50590/53190/63510/73910/82299/82400/92730/99970/106940/117320/127870/1375xx
                        # worker1 died due to kfang @~138300, then somehow recovered automatically (maybe after preemption several hours later)
  CHECKPOINT_MAX_TO_KEEP = 2

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256TransLogits(PilePythia7B256x1DynWFFN16HD128Win256):
  TRANSPOSE_LOGITS = True  # v4 0.123 v4 0.646, BTNS v3 0.06459
  SUMMARY_INTERVAL_STEPS = 5

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256RellocA4M4L48(PilePythia7B256x1DynWFFN16HD128Win256Aligned):
  HIDDEN_DIMS = 16384 // 2  # v4 0.105
  NUM_LAYERS = 48

@experiment_registry.register
class PilePythia410M64x1(PileDataParams, Pythia410M64x1):
  pass

@experiment_registry.register
class PilePythia410M128x1(PileDataParams, Pythia410M128x1):
  SAVE_ON_STEPS = [1000,] + list(range(3000, 143000, 10000)) # v4 0.601!
  # CHECKPOINT_EVERY_N_STEPS = 50
  # EVAL_SKIP_TRAIN = True
  # ctrl+c @200/600, delete pod @1200

@experiment_registry.register
class PilePythia70M64x1(PileDataParams, Pythia70M64x1):
  CHECKPOINT_MAX_TO_KEEP = 15  # v4 llama arch 0.103
  SAVE_ON_STEPS = [1000,] + list(range(3000, 143000, 10000))
  QUERY_CHUNK_SIZE = 128  # 128/None 1.49/1.15
  # EVAL_SKIP_TRAIN = True
  EVAL_INTERVAL_STEPS = 100
  # ICI_MESH_SHAPE = [1, 8, 1]
  # PERCORE_BATCH_SIZE = 1
  # ONLY_EVAL = True
  # RESET_FOR_EVAL = True # True: test while test dataset


@experiment_registry.register
class SeqioPilePythia410M128x1(SeqioPileDataParams, Pythia410M128x1):
  SAVE_ON_STEPS = [1000,] + list(range(3000, 143000, 10000))

@experiment_registry.register
class PilePythia410M64x1FixRot(PileDataParams, Pythia410M64x1):
  PYTHIA_ROTARY = True  # v4 0.222, v3 0.2644!?
  SAVE_ON_STEPS = [1000, 6000, 9000] + list(range(3000, 143000, 10000))
  
@experiment_registry.register
class PilePythia410M128x1FixRot(PileDataParams, Pythia410M128x1):
  PYTHIA_ROTARY = True
  SAVE_ON_STEPS = list(range(3000, 143000, 10000))

@experiment_registry.register
class PilePythiaXL128x1FixRot(PileDataParams, PythiaXL128x1):
  PYTHIA_ROTARY = True  # v4 0.382 significantly faster than PileSpmdLlamaXLHead16x128LR001 v4 0.331!? v3 0.224
  SAVE_ON_STEPS = [1000, 6000, 9000] + list(range(3000, 143000, 10000))  # restart @6120
  
@experiment_registry.register
class PilePythia3B128x1FixRot(PileDataParams, PythiaInit, Pythia3B):
  PYTHIA_ROTARY = True
  PERCORE_BATCH_SIZE = 2 * 2 * 2
  ICI_MESH_SHAPE = [1, 128, 1]

@experiment_registry.register
class PilePythia33B512x1FixRot(PileDataParams, PythiaInit, Pythia33B):
  PERCORE_BATCH_SIZE = 2
  ICI_MESH_SHAPE = [1, 512, 1]  # v3 0.035
  PYTHIA_ROTARY = True
  SUMMARY_INTERVAL_STEPS = 5

@experiment_registry.register
class PilePythia33B128x4FixRot(PilePythia33B512x1FixRot):
  ICI_MESH_SHAPE = [1, 128, 4]  # v3 0.03345

@experiment_registry.register
class PilePythia33B64x8FixRot(PilePythia33B512x1FixRot):
  ICI_MESH_SHAPE = [1, 64, 8]  # v3 0.3345

@experiment_registry.register
class PilePythia410M64x1DynWFFN8HD64Win256Aligned(PilePythia410M64x1FixRot, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00013)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00022) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01
  DYNAMIC_SQUEEZE_RATIO = 8  # v4 0.137, v3 0.115
  DYNAMIC_W_HIDDEN_DIM = 64
  QUERY_CHUNK_SIZE = 128
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  MERGE_DW_PROJ = True
  SAVE_ON_STEPS = list(range(3000, 143000, 10000))
  CHECKPOINT_MAX_TO_KEEP = 2

@experiment_registry.register
class PilePythia410M128x1DynWFFN8HD64Win256Aligned(PilePythia410M128x1FixRot, PilePythia410M64x1DynWFFN8HD64Win256Aligned):
  pass  # v3 0.232! vs 64x1 v3 0.115  interrupted by kf @51000

@experiment_registry.register
class PilePythiaXLDynWFFN8HD64Win256Aligned(PilePythiaXL128x1FixRot, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  # should be PilePythiaXL128x1DynWFFN8HD64Win256Aligned
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00013)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00015) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01
  DYNAMIC_SQUEEZE_RATIO = 8  # v4 0.2356,  v3 0.142
  DYNAMIC_W_HIDDEN_DIM = 64
  QUERY_CHUNK_SIZE = 128
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  MERGE_DW_PROJ = True
  SAVE_ON_STEPS = list(range(3000, 143000, 10000))  # interrupted by kf @59900
  CHECKPOINT_MAX_TO_KEEP = 2

@experiment_registry.register
class PilePythiaXLDynWFFN8HD64Win256Head32x64OrigInit(PilePythiaXLDynWFFN8HD64Win256Aligned): # wrong
  NUM_HEADS = 32
  DIMS_PER_HEAD = 64  # v4 0.136 << PilePythiaXLDynWFFN8HD64Win256Aligned v4 0.236
  INIT_METHOD = None
  OUTPUT_LAYER_INIT_METHOD = None

@experiment_registry.register
class PilePythiaXLDynWFFN16HD128Win256Head32x64(PilePythiaXLDynWFFN8HD64Win256Aligned):
  NUM_HEADS = 32
  DIMS_PER_HEAD = 64  # v4 0.160
  DYNAMIC_SQUEEZE_RATIO = 16
  DYNAMIC_W_HIDDEN_DIM = 128

@experiment_registry.register
class PilePythiaXLDynWFFN16HD128Win256Head32x64OrigInit(PilePythiaXLDynWFFN8HD64Win256Aligned):
  NUM_HEADS = 32
  DIMS_PER_HEAD = 64  # v4 0.160 << PilePythiaXLDynWFFN8HD64Win256Aligned v4 0.236
  DYNAMIC_SQUEEZE_RATIO = 16
  DYNAMIC_W_HIDDEN_DIM = 128

  INIT_METHOD = None
  OUTPUT_LAYER_INIT_METHOD = None

@experiment_registry.register
class PilePythiaXLDynWFFN8HD64Win256OrigInit(PilePythiaXLDynWFFN8HD64Win256Aligned):
  INIT_METHOD = None  # v3 0.142
  OUTPUT_LAYER_INIT_METHOD = None

@experiment_registry.register
class PilePythia3BDynWFFN16HD128Win256Aligned(PilePythia3B128x1FixRot, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0001)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.0001) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01
  DYNAMIC_SQUEEZE_RATIO = 16
  DYNAMIC_W_HIDDEN_DIM = 128  # v4 0.106 v3 0.0619
  QUERY_CHUNK_SIZE = 128
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  MERGE_DW_PROJ = True
  SAVE_ON_STEPS = list(range(3000, 143000, 10000))
  CHECKPOINT_MAX_TO_KEEP = 2

  PERCORE_BATCH_SIZE = 2 * 2
  ICI_MESH_SHAPE = [1, 256, 1]  # v4 0.206  acutally trained with this config in order to be faster
  # ZERO_LOSS = False

@experiment_registry.register
class PilePythia3BDynWFFN16HD128Win256AlignedSW(PilePythia3BDynWFFN16HD128Win256Aligned):
  LOGITS_DYNAMIC_W_INIT = None
  PROBS_DYNAMIC_W_INIT = None
  DYNAMIC_W_INIT = None  # v4 0.28
  DYNAMIC_D_INIT = None

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256AlignedSW(PilePythia7B256x1DynWFFN16HD128Win256Aligned):
  LOGITS_DYNAMIC_W_INIT = None
  PROBS_DYNAMIC_W_INIT = None
  DYNAMIC_W_INIT = None  # v3 0.078
  DYNAMIC_D_INIT = None

@experiment_registry.register
class PilePythia33B512x1DynWFFN32HD256Win256Aligned(PilePythia33B512x1FixRot, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00013)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00015) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01
  DYNAMIC_SQUEEZE_RATIO = 32  # v3 0.02857
  DYNAMIC_W_HIDDEN_DIM = 256
  QUERY_CHUNK_SIZE = 128
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  MERGE_DW_PROJ = True

@experiment_registry.register
class PilePythia33B128x4NG4DynWFFN32HD256Win256Aligned(PilePythia33B128x4FixRot, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00013)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00015) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01
  NUM_GROUPS = 4
  DYNAMIC_SQUEEZE_RATIO = 16  # v3 NUM_GROUPS mdl 0.0278@20
  DYNAMIC_W_HIDDEN_DIM = 32
  QUERY_CHUNK_SIZE = 128
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  MERGE_DW_PROJ = True

@experiment_registry.register
class PilePythia33B64x8NG8DynWFFN32HD256Win256Aligned(PilePythia33B64x8FixRot, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00013)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00015) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01
  NUM_GROUPS = 8
  DYNAMIC_SQUEEZE_RATIO = 8  # v3 NUM_GROUPS mdl/None 0.02746/0.0271 @20 
  DYNAMIC_W_HIDDEN_DIM = 16
  QUERY_CHUNK_SIZE = 128
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  MERGE_DW_PROJ = True

@experiment_registry.register
class PileSpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWin256(PileDataParams, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  NUM_LAYERS = 24  # v4 0.229
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  MERGE_DW_PROJ = True

  QUERY_CHUNK_SIZE = 128
  EMBEDDING_LOOKUP_STYLE = 'index'
  LM_HEAD_CHUNK_SIZE = 512

  LEARNING_RATE = 2e-4
  LR_COS_WARMUP = 1430
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 143000

  PERCORE_BATCH_SIZE = 2 * 2 * 2
  ICI_MESH_SHAPE = [1, 128, 1]
  SAVE_ON_STEPS = list(range(3000, 143000, 10000))
  CHECKPOINT_MAX_TO_KEEP = 2

@experiment_registry.register
class PileSpmdLlamaXLHead24x80ResTHDynWFFN8HD64DW1RmsNormWin256(_LlamaXLHead24x80, PileSpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWin256):
  pass  # v4 0.173!

@experiment_registry.register
class PileSpmdLlamaXLHead24x96ResTHDynWFFN8HD64DW1RmsNormWin256(_LlamaXLHead24x96, PileSpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWin256):
  pass  # v3 0.0841

@experiment_registry.register
class PileSpmdLlamaXLHead16x128(PileDataParams, C4SpmdLlamaXLHead16x128):
  NUM_LAYERS = 24  # v3 0.215

  QUERY_CHUNK_SIZE = 128
  EMBEDDING_LOOKUP_STYLE = 'index'
  LM_HEAD_CHUNK_SIZE = 512

  LEARNING_RATE = 2e-4
  LR_COS_WARMUP = 1430
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 143000
  
  PERCORE_BATCH_SIZE = 2 * 2 * 2
  ICI_MESH_SHAPE = [1, 128, 1]
  SAVE_ON_STEPS = list(range(3000, 143000, 10000))
  CHECKPOINT_MAX_TO_KEEP = 2

@experiment_registry.register
class PileSpmdLlamaXLHead24x80(_LlamaXLHead24x80, PileSpmdLlamaXLHead16x128):
  pass  # v3 0.202

@experiment_registry.register
class PileSpmdLlamaXLHead24x96(_LlamaXLHead24x96, PileSpmdLlamaXLHead16x128):
  pass  # v3 0.1996

@experiment_registry.register
class PileSpmdLlamaXLHead16x128LR001(PileSpmdLlamaXLHead16x128):
  LEARNING_RATE = 1e-3  # v4 0.331
  LR_COS_MIN_RATIO = 0.01
  QK_NORM = True
  
@experiment_registry.register
class PileSpmdLlamaXLHead16x128LR0006(PileSpmdLlamaXLHead16x128):
  LEARNING_RATE = 6e-4
  LR_COS_MIN_RATIO = 1 / 60
  QK_NORM = True

# C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagProbsDWTanh  LOGITS/PROBS_DYNAMIC_W_INIT
class _DC:
  NUM_GROUPS = 1
  QK_NORM = True
  PROJECT_LOGITS = True
  PROJECT_PROBS = True
  # LOGITS_SQUEEZE_RATIO = None
  SKIP_BIAS = True  # useless
  TRANSPOSE = False
  LEARNABLE_DIAG = True
  ABSORB_RESIDUAL = False
  PROBS_OUTPUT_ACTIVATION_CLS = None

  USE_DW_HIDDEN_BIAS = False  # useless for merge_dw_proj
  DW_HIDDEN_ACTIVATION_CLS = layers.GELU
  DW1_NORM_CLS = normalizations.RmsNormNoScale
  DW1_NORM_BIAS_INIT = 1e-6
  DW_ACTIVATION_CLS = layers.Tanh  # only effective for dd
  LOOP_OVER_DYNAMIC_HD = True
  MERGE_DW_PROJ = True
  
  # for DCLlama7B
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00005)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.0001)  # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01

  QUERY_CHUNK_SIZE = 128
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  SUMMARY_VERBOSITY = 9

class DCLlama3B(_DC, _Llama3B):  # TODO: add init
  DYNAMIC_SQUEEZE_RATIO = 16
  DYNAMIC_W_HIDDEN_DIM = 128

class DCLlama7B(_DC, _Llama7B):
  DYNAMIC_SQUEEZE_RATIO = 16
  DYNAMIC_W_HIDDEN_DIM = 128
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00005)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.0001)  # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01

class DCSlimLlama7B(_DC, _SlimLlama7B):
  DYNAMIC_SQUEEZE_RATIO = 16
  DYNAMIC_W_HIDDEN_DIM = 128
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00005)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.0001)  # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01

class DCLlama13B(_DC, _Llama13B):  # TODO: add init
  DYNAMIC_SQUEEZE_RATIO = 20
  DYNAMIC_W_HIDDEN_DIM = 160

class DCLlama33B(_DC, _Llama33B):  # TODO: add init
  DYNAMIC_SQUEEZE_RATIO = 52
  DYNAMIC_W_HIDDEN_DIM = 208

class DCSlimLlama7BNG4(DCSlimLlama7B):
  NUM_GROUPS = 4
  DYNAMIC_SQUEEZE_RATIO = 8
  DYNAMIC_W_HIDDEN_DIM = 16
  # TODO: DYNAMIC_W_INIT and DYNAMIC_D_INIT should also be changed

class DCSlimLlama7BNG2(DCSlimLlama7B):
  NUM_GROUPS = 2
  DYNAMIC_SQUEEZE_RATIO = 8
  DYNAMIC_W_HIDDEN_DIM = 16 * (16 // 8) * 2
  # TODO: DYNAMIC_W_INIT and DYNAMIC_D_INIT should also be changed

@experiment_registry.register
class PileDCSlimLlama7B2Kx4x512x1(DataParams, PythiaInit, DCSlimLlama7B):
  MAX_SEQ_LEN = 2048
  LEARNING_RATE = 3e-4
  LR_COS_WARMUP = 2000
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 200000  # 800B tokens
  LR_COS_MIN_RATIO = 0.1

  PERCORE_BATCH_SIZE = 8
  ICI_MESH_SHAPE = [1, 32, 1]
  EMBEDDING_LOOKUP_STYLE = 'index'
  SAVE_ON_STEPS = list(range(0, 300000, 10000))

@experiment_registry.register
class PileDCSlimLlama7B8Kx1x512x1Win256_4K(PileDCSlimLlama7B2Kx4x512x1):
  MAX_SEQ_LEN = 8192 // 2
  WINDOW_SIZE = [256, 4096]
  PERCORE_BATCH_SIZE = 1
  QUERY_CHUNK_SIZE = 256
  LM_HEAD_CHUNK_SIZE = 512
  ICI_MESH_SHAPE = [1, 4, 1]
  DATA_FULL_SHARD = False
  # FFN_CHUKN_SIZE = 5504 // 2

@experiment_registry.register
class PileDCSlimLlama7B32Kx1x512x1Win256_4K(PileDCSlimLlama7B2Kx4x512x1):
  #MAX_SEQ_LEN = 8192 * 4 // 2
  NUM_LAYERS=48
  MAX_SEQ_LEN = 8192 * 4
  WINDOW_SIZE = [256, 4096]
  # WINDOW_SIZE = None
  PERCORE_BATCH_SIZE = 0.25 #/ 4
  QUERY_CHUNK_SIZE = 512
  LM_HEAD_CHUNK_SIZE = 512
  ICI_MESH_SHAPE = [1, 64, 4]
  DATA_FULL_SHARD = False
  FFN_CHUKN_SIZE = 5504 // 8
  PRE_COMPUTE_ATTEN_MASK = False
  EVAL_INTERVAL_STEPS = 100
  EVAL_LOOP_NUM_BATCHES = 20


@experiment_registry.register
class _TrainConfig2Kx2x512x1:
  LEARNING_RATE = 3e-4  # v3 0.0331
  LR_COS_WARMUP = 2000
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 200000  # 800B tokens
  LR_COS_MIN_RATIO = 0.1

  PERCORE_BATCH_SIZE = 2
  ICI_MESH_SHAPE = [1, 512, 1]
  EMBEDDING_LOOKUP_STYLE = 'index'
  SUMMARY_INTERVAL_STEPS = 5

@experiment_registry.register
class _TrainConfig2Kx4x256x1(_TrainConfig2Kx2x512x1):
  PERCORE_BATCH_SIZE = 4
  ICI_MESH_SHAPE = [1, 256, 1]

@experiment_registry.register
class PileDCLlama3B2Kx4x256x1(_TrainConfig2Kx4x256x1, PileDataParams, PythiaInit, DCLlama3B):
  LEARNING_RATE = 5e-4
  LR_COS_WARMUP = 1430
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 143000
  LR_COS_MIN_RATIO = 0.01
  SAVE_ON_STEPS = list(range(3000, 143000, 10000))

@experiment_registry.register
class PileDCLlama3B2Kx4x256x1DWDD(PileDCLlama3B2Kx4x256x1):
  USE_STATIC_W = False  # v4 0.221

@experiment_registry.register
class PileDCLlama3B2Kx4x256x1DWDDLR00032(PileDCLlama3B2Kx4x256x1DWDD):
  LEARNING_RATE = 3.2e-4


@experiment_registry.register
class RepeatLayerTest(PileDCSlimLlama7B2Kx4x512x1):
  #MAX_SEQ_LEN = 8192 * 4 // 2
  NUM_LAYERS=48
  NUM_LAYERS_PER_BLOCK = 2
#  NUM_HEADS = 32
  MAX_SEQ_LEN = 1 * 8192 // 4
  # WINDOW_SIZE = [256, 4096] * (NUM_LAYERS_PER_BLOCK // 2)
  # WINDOW_SIZE = [256, 4096] * (NUM_LAYERS // NUM_LAYERS_PER_BLOCK)
  WINDOW_SIZE = [256, 4096]
  HIDDEN_DIMS = 11008 // 2
  PERCORE_BATCH_SIZE = 8
  QUERY_CHUNK_SIZE = 512
  LM_HEAD_CHUNK_SIZE = 512
  ICI_MESH_SHAPE = [1, 4, 1]
  DATA_FULL_SHARD = True
  # FFN_CHUKN_SIZE = 5504 // 8
  DATA_PATH = {
                'train': 'gs://common_datasets_us-east5/',
                'test':  'gs://common_datasets_us-east5/',
                }
  VOCAB_FILE = 'gs://common_datasets_us-east5/vocab/c4_en_301_5Mexp_spm.model'
  VOCABULARY = t5.data.SentencePieceVocabulary(VOCAB_FILE)
  USE_REPEATED_LAYER = True
  USE_STATIC_W = False
  REMAT = True
  # CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_EVERYTHING


@experiment_registry.register
class MoeTest(PileDCSlimLlama7B2Kx4x512x1):
  #MAX_SEQ_LEN = 8192 * 4 // 2
  NUM_LAYERS=30
  MAX_SEQ_LEN = 8192 // 4
  WINDOW_SIZE = [256, 4096]
  HIDDEN_DIMS = 5504 // 4
  # WINDOW_SIZE = None
  PERCORE_BATCH_SIZE = 1
  QUERY_CHUNK_SIZE = 512
  LM_HEAD_CHUNK_SIZE = 512
  ICI_MESH_SHAPE = [1, 8, 1]
  DATA_FULL_SHARD = False
  # FFN_CHUKN_SIZE = 5504 // 8
  NUM_EXPERTS = 4
  MOE_GATED_ACTIVATION=True
  CAPACITY_FACTOR = 1.25
  MOE_LAYERS = list(range(NUM_LAYERS))
  PRE_COMPUTE_ATTEN_MASK = False
  EVAL_INTERVAL_STEPS = 100
  EVAL_LOOP_NUM_BATCHES = 20

@experiment_registry.register
class MgateTest(PileDCSlimLlama7B2Kx4x512x1):
  #MAX_SEQ_LEN = 8192 * 4 // 2
  NUM_LAYERS=10
  MAX_SEQ_LEN = 2048
  WINDOW_SIZE = [256, 4096]
  # WINDOW_SIZE = None
  PERCORE_BATCH_SIZE = 4 #/ 4
  QUERY_CHUNK_SIZE = 512
  LM_HEAD_CHUNK_SIZE = 512
  ICI_MESH_SHAPE = [1, 8, 1]
  DATA_FULL_SHARD = True
  FFN_CHUKN_SIZE = 5504 // 8
  PRE_COMPUTE_ATTEN_MASK = False
  EVAL_INTERVAL_STEPS = 100
  EVAL_LOOP_NUM_BATCHES = 20
  MGATE = True

@experiment_registry.register
class DSMgateTest(PileDCSlimLlama7B2Kx4x512x1):
  #MAX_SEQ_LEN = 8192 * 4 // 2
  MAX_SEQ_LEN = 2048
  WINDOW_SIZE = [256, 4096]
  # WINDOW_SIZE = None
  PERCORE_BATCH_SIZE = 1 #/ 4
  QUERY_CHUNK_SIZE = 256
  LM_HEAD_CHUNK_SIZE = 512
  ICI_MESH_SHAPE = [1, 4, 1]
  DATA_FULL_SHARD = False
  FFN_CHUKN_SIZE = 5504 // 8
  PRE_COMPUTE_ATTEN_MASK = False
  EVAL_INTERVAL_STEPS = 100
  EVAL_LOOP_NUM_BATCHES = 20
  MGATE = True
  DSM = True
  DATA_PATH = {
                'train': 'gs://common_datasets_us-east5/', 
                'test':  'gs://common_datasets_us-east5/', 
                }
  VOCAB_FILE = 'gs://common_datasets_us-east5/vocab/c4_en_301_5Mexp_spm.model'
  VOCABULARY = t5.data.SentencePieceVocabulary(VOCAB_FILE)


@experiment_registry.register
class MultiSliceTest(PileDCSlimLlama7B2Kx4x512x1):
  NUM_LAYERS=20
  MAX_SEQ_LEN = 8192 // 4
  # WINDOW_SIZE = [256, 4096]
  WINDOW_SIZE = None
  PERCORE_BATCH_SIZE = 1 #/ 4
  QUERY_CHUNK_SIZE = 512
  LM_HEAD_CHUNK_SIZE = None
  ICI_MESH_SHAPE = [1, 4, 1]
  DATA_FULL_SHARD = True
  FFN_CHUKN_SIZE = None
  PRE_COMPUTE_ATTEN_MASK = True
  EVAL_INTERVAL_STEPS = 100
  EVAL_LOOP_NUM_BATCHES = 20
  DCN_MESH_SHAPE = [2, 1, 1] # 2表示slice count

@experiment_registry.register
class PileDCLlama33B2Kx2x512x1(_TrainConfig2Kx2x512x1, PileDataParams, PythiaInit, DCLlama33B):
  pass  # v3 0.0331

@experiment_registry.register
class PileDCLlama33B2Kx2x512x1DWDD(PileDCLlama33B2Kx2x512x1):
  USE_STATIC_W = False

@experiment_registry.register
class PileLlama33B2Kx2x512x1(_TrainConfig2Kx2x512x1, PileDataParams, PythiaInit, _Llama33B):
  QUERY_CHUNK_SIZE = 512  # v3 0.04104
  # pass  # v3 0.0399

@experiment_registry.register
class PileDCLlama13B2Kx2x512x1(_TrainConfig2Kx2x512x1, PileDataParams, PythiaInit, DCLlama13B):
  pass  # v3 0.07599

@experiment_registry.register
class PileDCLlama13B2Kx2x512x1DWDD(PileDCLlama13B2Kx2x512x1):
  USE_STATIC_W = False

@experiment_registry.register
class PileLlama13B2Kx2x512x1(_TrainConfig2Kx2x512x1, PileDataParams, PythiaInit, _Llama13B):
  QUERY_CHUNK_SIZE = 512  # v3 0.099819
  # pass  # v3 0.096595

@experiment_registry.register
class PileDCLlama13B2Kx4x256x1(_TrainConfig2Kx4x256x1, PileDataParams, PythiaInit, DCLlama13B):
  pass  # v3

@experiment_registry.register
class PileLlama13B2Kx4x256x1(_TrainConfig2Kx4x256x1, PileDataParams, PythiaInit, _Llama13B):
  QUERY_CHUNK_SIZE = 512  # v3 0.099819
  # pass  # v3 0.096595

@experiment_registry.register
class PileDCLlama7B2Kx4x256x1(_TrainConfig2Kx4x256x1, PileDataParams, PythiaInit, DCLlama7B):
  pass  # v3

@experiment_registry.register
class PileDCLlama7B2Kx4x256x1DWDD(PileDCLlama7B2Kx4x256x1):
  USE_STATIC_W = False

@experiment_registry.register
class PileLlama7B2Kx4x256x1(_TrainConfig2Kx4x256x1, PileDataParams, PythiaInit, _Llama7B):
  QUERY_CHUNK_SIZE = 512  # v3
  # pass  # v3

@experiment_registry.register
class _TrainConfig2Kx8x128x1(_TrainConfig2Kx2x512x1):
  PERCORE_BATCH_SIZE = 8
  ICI_MESH_SHAPE = [1, 128, 1]

@experiment_registry.register
class PileDCLlama3B2Kx8x128x1(_TrainConfig2Kx8x128x1, PileDataParams, PythiaInit, DCLlama3B):
  pass  # v3

# @experiment_registry.register
# class PileDCLlama3B2Kx4x256x1(_TrainConfig2Kx4x256x1, PileDataParams, PythiaInit, DCLlama3B):
#   pass  # v3

@experiment_registry.register
class PileLlama3B2Kx8x128x1(_TrainConfig2Kx8x128x1, PileDataParams, PythiaInit, _Llama3B):
  QUERY_CHUNK_SIZE = 512  # v3 0.099819
  # pass  # v3 0.096595

class _SWConfig:
  LOGITS_DYNAMIC_W_INIT = None
  PROBS_DYNAMIC_W_INIT = None
  DYNAMIC_W_INIT = None
  DYNAMIC_D_INIT = None

class _CommonConfig:
  QUERY_CHUNK_SIZE = 128
  EMBEDDING_LOOKUP_STYLE = 'index'
  LM_HEAD_CHUNK_SIZE = 512
  CHECKPOINT_MAX_TO_KEEP = 2

class _DCConfig:
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]
  MERGE_DW_PROJ = True

class _XLConfig(_CommonConfig):
  LEARNING_RATE = 2e-4
  LR_COS_WARMUP = 500
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 50000
  LR_COS_MIN_RATIO = 0.1
  
  PERCORE_BATCH_SIZE = 2 * 2
  ICI_MESH_SHAPE = [1, 64, 1]
  SAVE_ON_STEPS = list(range(10000, 60000, 10000))

@experiment_registry.register
class PileLlamaXL(PileDataParams, _XLConfig, C4SpmdLlamaXL):
  NUM_LAYERS = 24 # v3 0.345, v4 0.396

@experiment_registry.register
class PileLlamaXLHead16x128(PileLlamaXL):
  NUM_HEADS = 16
  DIMS_PER_HEAD = 128  # v3 0.401

@experiment_registry.register
class PileGPTXL(C4SpmdGPTSepEmb, PileLlamaXL):
  HIDDEN_DIMS = 8192  # v3 0.353

@experiment_registry.register
class PileDCLlamaXL(PileDataParams, _DCConfig, _XLConfig, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  NUM_LAYERS = 24  # v4 0.238
  NUM_HEADS = 32
  DIMS_PER_HEAD = 64

@experiment_registry.register
class PileDCLlamaXLFixDWInit(PileDCLlamaXL):
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00013)  # should be 5e-5 for NUM_HEAD 32. sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00015) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01

@experiment_registry.register
class PileDCLlamaXLFixDWShape(PileDCLlamaXLFixDWInit):
  DYNAMIC_SQUEEZE_RATIO = 16  # 8 -> 16
  DYNAMIC_W_HIDDEN_DIM = 128  # 64 -> 128  v4 0.267

@experiment_registry.register
class PileDCGPTXLDWDDNoQKNorm(C4SpmdGPTSepEmb, PileDCLlamaXLFixDWShape):
  HIDDEN_DIMS = 8192
  USE_STATIC_W = False
  QK_NORM = False

@experiment_registry.register
class PileDCLlamaXLHead16x128(PileDCLlamaXLFixDWInit):
  NUM_HEADS = 16
  DIMS_PER_HEAD = 128  # v4 0.364! >> PileDCLlamaXL v4 0.238
  # also >> C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormWhole v4 0.16 * 2
  # may be even faster if pbs 4 -> 8

@experiment_registry.register
class PileDCLlamaXLRellocA6M6(PileDCLlamaXLFixDWInit):
  HIDDEN_DIMS = 4096  # v4 0.235
  NUM_HEADS = 32
  DIMS_PER_HEAD = 96

@experiment_registry.register
class PileDCLlamaXLRellocA6M6Interleave(PileDCLlamaXLFixDWInit):
  HIDDEN_DIMS = [5504, 4096]  # v4 0.257
  NUM_HEADS = [16, 32]
  DIMS_PER_HEAD = [128, 96]

@experiment_registry.register
class PileDCLlamaXLRellocA4M4L36(PileDCLlamaXLFixDWInit):
  HIDDEN_DIMS = 2752  # 5504 // 2
  NUM_LAYERS = 36  # v4 0.180

@experiment_registry.register
class PileLlamaXLRellocA4M4L36(PileLlamaXL):
  HIDDEN_DIMS = 2752  # 5504 // 2
  NUM_LAYERS = 36  # v4 0.331

@experiment_registry.register
class PileDCLlamaXLHead16x128GQA4(PileDCLlamaXLHead16x128):
  NUM_KV_HEADS = 4  # v3 0.265
  HIDDEN_DIMS = 6528

@experiment_registry.register
class PileDCLlamaXLGQA8(PileDCLlamaXLFixDWShape):
  NUM_KV_HEADS = 8  # v3 0.191
  HIDDEN_DIMS = 6528

@experiment_registry.register
class PileDCLlamaXLGQA8NG4DiffKV(PileDCLlamaXLGQA8):
  NUM_GROUPS = 4  # v3 0.193
  DYNAMIC_SQUEEZE_RATIO = 8
  DYNAMIC_W_HIDDEN_DIM = 16
  
@experiment_registry.register
class PileDCLlamaXLGQA4NG4SameKV(PileDCLlamaXLGQA8NG4DiffKV):
  NUM_KV_HEADS = 4  # v3 0.194
  INTERLEAVE_KV_HEADS = False
  HIDDEN_DIMS = 6656

@experiment_registry.register
class PileDCLlamaXLGQA4NG8DiffKV(PileDCLlamaXLGQA4NG4SameKV):
  NUM_GROUPS = 8  # v3 0.158 << others
  INTERLEAVE_KV_HEADS = True
  DYNAMIC_SQUEEZE_RATIO = 4
  DYNAMIC_W_HIDDEN_DIM = 8

@experiment_registry.register
class PileDCLlamaXLGQA8NG4MixKV(PileDCLlamaXLGQA8NG4DiffKV):
  INTERLEAVE_KV_HEADS = None  # v3 0.193

@experiment_registry.register
class PileDCLlamaXLGQA8NG4MixKVRank2(PileDCLlamaXLGQA8NG4MixKV):
  DYNAMIC_SQUEEZE_RATIO = 4  # v3 0.167
  DYNAMIC_W_HIDDEN_DIM = 32

@experiment_registry.register
class PileLlamaXLQKNorm(PileLlamaXL):
  QK_NORM = True

@experiment_registry.register
class PileDCLlamaXLSW(_SWConfig, PileDCLlamaXLFixDWShape): pass

@experiment_registry.register
class PileDCLlamaXLSWNoWinNoQKNorm(PileDCLlamaXLSW):
  NUM_LAYERS_PER_BLOCK = 1
  WINDOW_SIZE = None
  QK_NORM = False

@experiment_registry.register
class PileDCLlamaXLDWDD(PileDCLlamaXLFixDWShape):
  USE_STATIC_W = False  # v4 0.282

@experiment_registry.register
class PileDCLlamaXLNoQKNorm(PileDCLlamaXLFixDWShape):
  QK_NORM = False  # v4 0.262

@experiment_registry.register
class PileDCLlamaXLDWDDNoQKNorm(PileDCLlamaXLDWDD):
  QK_NORM = False  # v4 0.284

@experiment_registry.register
class PileDCLlamaXLHead16x128SW(_SWConfig, PileDCLlamaXLHead16x128): pass

@experiment_registry.register
class PileDCLlamaXLHead16x128SWNoWinNoQKNorm(PileDCLlamaXLHead16x128SW):
  NUM_LAYERS_PER_BLOCK = 1
  WINDOW_SIZE = None
  QK_NORM = False

@experiment_registry.register
class PileDCLlamaXLNoQKNormTHADyn(PileDCLlamaXLNoQKNorm):
  MERGE_DW_PROJ = False
  DYNAMIC_W_HIDDEN_DIM = 32*32
  DECOMPOSE_DYNAMIC_W = False  # v4
  DYNAMIC_D_INIT = None
  DW1_NORM_CLS = None

@experiment_registry.register
class PileDCLlamaXLNoQKNormTHADynOrig(PileDCLlamaXLNoQKNormTHADyn):
  DYNAMIC_W_HIDDEN_DIM = None  # v4 0.2097
  LOGITS_DW_ACTIVATION_CLS = None
  PROBS_DW_ACTIVATION_CLS = None
  DW_ACTIVATION_CLS = None
  
class _LargeConfig(_CommonConfig):
  LEARNING_RATE = 2.5e-4
  LR_COS_WARMUP = 290
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 29000
  LR_COS_MIN_RATIO = 0.1
  
  PERCORE_BATCH_SIZE = 8
  ICI_MESH_SHAPE = [1, 32, 1]
  SAVE_ON_STEPS = list(range(5000, 30000, 5000)) + [29000]

@experiment_registry.register
class PileLlamaLarge(PileDataParams, _LargeConfig, C4SpmdLlamaLarge):
  pass  # v3  # v3 0.293

@experiment_registry.register
class PileGPTLarge(C4SpmdGPTSepEmb, PileLlamaLarge):
  HIDDEN_DIMS = 6144  # # v3 0.301

@experiment_registry.register
class PileDCLlamaLarge(PileDataParams, _DCConfig, _LargeConfig, C4SpmdLlamaLarge, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  DYNAMIC_SQUEEZE_RATIO = 12  # v4 0.206
  DYNAMIC_W_HIDDEN_DIM = 96
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00008)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00018) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01

@experiment_registry.register
class PileDCLlamaLargeNoQKNorm(PileDCLlamaLarge):
  QK_NORM = False

@experiment_registry.register
class PileDCGPTLargeDWDDNoQKNorm(C4SpmdGPTSepEmb, PileDCLlamaLarge):
  HIDDEN_DIMS = 6144
  USE_STATIC_W = False
  QK_NORM = False

@experiment_registry.register
class PileDCLlamaLargeDWDDNoQKNorm(PileDCLlamaLarge):
  USE_STATIC_W = False
  QK_NORM = False

class _MediumConfig(_CommonConfig):
  LR_COS_WARMUP = 135 #* 4
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_MIN_RATIO = 0.1
  
  ICI_MESH_SHAPE = [1, 32, 1]
  # v1
  PERCORE_BATCH_SIZE = 8
  LEARNING_RATE = 3e-4
  LR_COS_DECAY_END = 13500 #* 4
  SAVE_ON_STEPS = [13500] # list(range(3000, 15000, 3000))
  # SAVE_ON_STEPS = list(range(3000 * 4, 15000 * 4, 3000 * 4)) + [13500 * 4]
  # stepsx2
  # PERCORE_BATCH_SIZE = 4
  # LEARNING_RATE = 2.5e-4
  # LR_COS_DECAY_END = 27000
  # SAVE_ON_STEPS = list(range(3000, 30000, 3000))

@experiment_registry.register
class PileLlamaMedium(PileDataParams, _MediumConfig, C4SpmdLlamaMedium):
  ZERO_LOSS = False
  # pass  # v3 0.520
  # TODO: _stepsx4 run should restart @42000

@experiment_registry.register
class PileGPTMedium(C4SpmdGPTSepEmb, PileLlamaMedium):
  HIDDEN_DIMS = 4096  # v3 0.530
  # TODO: _stepsx4 run should restart @42000

@experiment_registry.register
class PileDCLlamaMedium(PileDataParams, _DCConfig, _MediumConfig, C4SpmdLlamaMedium, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  DYNAMIC_SQUEEZE_RATIO = 8  # v3 0.232 v4 0.320; rerun 2024.1.19 v4 0.312??
  DYNAMIC_W_HIDDEN_DIM = 64
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00014)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00022) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01
  # TODO: _stepsx4 run should restart @18000

@experiment_registry.register
class PileDCGPTMediumDWDDNoQKNorm(C4SpmdGPTSepEmb, PileDCLlamaMedium):
  HIDDEN_DIMS = 4096
  USE_STATIC_W = False
  QK_NORM = False

@experiment_registry.register
class PileDCLlamaMedium2(PileDataParams, _DCConfig, _MediumConfig, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  NUM_LAYERS = 24
  MODEL_DIMS = 1024
  HIDDEN_DIMS = 2816  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 16
  DIMS_PER_HEAD = 64
  
  DYNAMIC_SQUEEZE_RATIO = 8
  DYNAMIC_W_HIDDEN_DIM = 64
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00014)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00022) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01

@experiment_registry.register
class PileDCLlamaMediumRellocA4M4L36(PileDCLlamaMedium):
  HIDDEN_DIMS = 1408  # 2816 / 2. v4
  NUM_LAYERS = 36

@experiment_registry.register
class PileDCLlamaMediumRellocA4M4D1280(PileDCLlamaMedium):
  MODEL_DIMS = 1280
  NUM_HEADS = 20  # 16 * 1.25
  HIDDEN_DIMS = 1600  # 1280 * 3.68 / 3

@experiment_registry.register
class PileDCLlamaMediumGO(PileDCLlamaMedium):
  O_GATE_ACTIVATION_CLS = layers.SiLU  # v3 0.230
  HIDDEN_DIMS = 2390  # MODEL_DIMS * 7 / 3

@experiment_registry.register
class PileDCLlamaMediumGVO(PileDCLlamaMedium):
  VALUE_GATE_ACTIVATION_CLS = layers.SiLU
  O_GATE_ACTIVATION_CLS = layers.SiLU  # v3 0.229
  HIDDEN_DIMS = 2048  # MODEL_DIMS * 6 / 3

@experiment_registry.register
class PileDCLlamaMediumDimVO128(PileDCLlamaMedium):
  DIM_PER_HEAD_V = 128  # v3 0.231
  HIDDEN_DIMS = 2048  # MODEL_DIMS * 6 / 3

@experiment_registry.register
class PileDCLlamaMediumSeg64(PileDCLlamaMedium):
  SEGMENT_SIZE = 64  # v3 0.2266
  HIDDEN_DIMS = 2752
  # RESIDUAL_CROSS_ACT_PROJ = False
  
@experiment_registry.register
class PileDCLlamaMediumResSeg64(PileDCLlamaMedium):
  SEGMENT_SIZE = 64  # v3 0.2266
  HIDDEN_DIMS = 2752
  RESIDUAL_CROSS_ACT_PROJ = True

@experiment_registry.register
class PileDCLlamaMediumProbsInpGELU(PileDCLlamaMedium):
  PROBS_INPUT_ACTIVATION_CLS = layers.GELU  # v3 0.229

@experiment_registry.register
class PileDCLlamaMediumInpGELU(PileDCLlamaMedium):
  LOGITS_INPUT_ACTIVATION_CLS = layers.GELU  # v3 0.221
  PROBS_INPUT_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class PileDCLlamaMediumDWDD(PileDCLlamaMedium):
  USE_STATIC_W = False  # v4 0.366

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNorm(PileDCLlamaMediumDWDD):
  QK_NORM = False # v4 0.370

# gs://llm_projects_us-central2/log/PileDCLlamaMediumDWDDNoQKNormWindowLGLLv4/checkpoints/checkpoint_00013500
@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormWindowLGLL(PileDCLlamaMediumDWDDNoQKNorm): # mqy
  WINDOW_SIZE = [256,None,256,256] #LGLL local:global 3:1 
  NUM_LAYERS_PER_BLOCK = 4

# gs://llm_projects/log/PileDCLlamaMediumDWDDNoQKNormWindowLGLLQW/checkpoints/checkpoint_00013500
@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormWindowLGLLQW(PileDCLlamaMediumDWDDNoQKNormWindowLGLL): # mqy
  SRC_DEPENDENT = False 

# gs://llm_projects/log/PileDCLlamaMediumDWDDNoQKNormWindowLGL6/checkpoints/checkpoint_00013500
@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormWindowLGL6(PileDCLlamaMediumDWDDNoQKNorm): # mqy
  WINDOW_SIZE = [256,None,256,256,256,256,256,256] #LGL6 local:global 7:1 
  NUM_LAYERS_PER_BLOCK = 8

@experiment_registry.register
class PileDCLlamaMediumDWNoQKNorm(PileDCLlamaMediumDWDDNoQKNorm):
  DYNAMIC_D_INIT = None  # v4 0.375

@experiment_registry.register
class PileDCLlamaMediumDDNoQKNorm(PileDCLlamaMediumDWDDNoQKNorm):
  LOGITS_DYNAMIC_W_INIT = None
  PROBS_DYNAMIC_W_INIT = None
  DYNAMIC_W_INIT = None  # v4 0.645

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormSrc(PileDCLlamaMediumDWDDNoQKNorm):
  TGT_DEPENDENT = False  # v4 0.454

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormTgt(PileDCLlamaMediumDWDDNoQKNorm):
  SRC_DEPENDENT = False  # v4 0.457

@experiment_registry.register
class PileDCLlamaMediumNoQKNormSrc(PileDCLlamaMediumDWDDNoQKNormSrc):
  USE_STATIC_W = True

@experiment_registry.register
class PileDCLlamaMediumSW(PileDCLlamaMedium):  # wrong warmup = 135 * 4
  LOGITS_DYNAMIC_W_INIT = None
  PROBS_DYNAMIC_W_INIT = None
  DYNAMIC_W_INIT = None  # v4 0.49
  DYNAMIC_D_INIT = None

@experiment_registry.register
class PileDCLlamaMediumSWNoWinNoQKNorm(PileDCLlamaMediumSW):
  NUM_LAYERS_PER_BLOCK = 1
  WINDOW_SIZE = None
  QK_NORM = False

@experiment_registry.register
class PileDCLlamaMediumNoWinNoQKNorm(PileDCLlamaMedium):
  NUM_LAYERS_PER_BLOCK = 1
  WINDOW_SIZE = None
  QK_NORM = False

@experiment_registry.register
class PileDCLlamaMediumSWNoOKNorm(PileDCLlamaMediumSW):  # should be NoQKNorm
  QK_NORM = False  # loss bump @710 after a preemption

@experiment_registry.register
class PileDCLlamaMediumSWNoQKNorm(PileDCLlamaMediumSWNoOKNorm):  # fix classname and rerun
  pass

@experiment_registry.register
class PileDCLlamaMediumSWNoWin(PileDCLlamaMediumSWNoWinNoQKNorm):
  NUM_LAYERS_PER_BLOCK = 1
  WINDOW_SIZE = None

@experiment_registry.register
class PileDCLlamaMediumNoOKNorm(PileDCLlamaMedium):  # should be NoQKNorm
  QK_NORM = False  # v4 0.315
  # QUERY_CHUNK_SIZE = 2048  # v4 0.125

@experiment_registry.register
class PileDCLlamaMediumNoQKNorm(PileDCLlamaMediumNoOKNorm): pass  # just fix name
  
@experiment_registry.register
class PileDCLlamaMediumNoQKNormProjLogits(PileDCLlamaMediumNoOKNorm):
  PROJECT_PROBS = False  # v4 0.516 / fixed 0.267
  # MERGE_DW_PROJ = False  # v4 0.186

@experiment_registry.register
class PileDCLlamaMediumNoQKNormProjProbs(PileDCLlamaMediumNoOKNorm):
  PROJECT_LOGITS = False  # v4 OOvmem
  QUERY_CHUNK_SIZE = 2048  # v4 0.061
  # MERGE_DW_PROJ = False  # v4 0.061
  # keep SW v4 0.376
  LOGITS_USE_STATIC_W = False
  # QUERY_CHUNK_SIZE = None  # v4 0.237

@experiment_registry.register
class PileDCLlamaMediumNoQKNormProjLogitsWorkaround(PileDCLlamaMediumNoOKNorm): # pre_proj is created but do nothing
  PROJECT_PROBS = False  # v4 0.392
  PROBS_USE_STATIC_W = False

@experiment_registry.register
class PileDCLlamaMediumNoQKNormProjProbsWorkaround(PileDCLlamaMediumNoOKNorm): # post_proj is created but do nothing
  PROJECT_LOGITS = False  # v4 0.355
  LOGITS_USE_STATIC_W = False

@experiment_registry.register
class PileDCLlamaMediumNoQKNormNG2R1(PileDCLlamaMediumNoQKNorm):
  NUM_GROUPS = 2
  DYNAMIC_SQUEEZE_RATIO = 8
  DYNAMIC_W_HIDDEN_DIM = 16

@experiment_registry.register
class PileDCLlamaMediumNoQKNormNG4R1(PileDCLlamaMediumNoQKNorm):
  NUM_GROUPS = 4
  DYNAMIC_SQUEEZE_RATIO = 4
  DYNAMIC_W_HIDDEN_DIM = 8

@experiment_registry.register
class PileDCLlamaMediumNoQKNormR1(PileDCLlamaMediumNoQKNorm):
  DYNAMIC_SQUEEZE_RATIO = 16
  DYNAMIC_W_HIDDEN_DIM = 32

@experiment_registry.register
class PileDCLlamaMediumNoQKNormR4(PileDCLlamaMediumNoQKNorm):
  DYNAMIC_SQUEEZE_RATIO = 4
  DYNAMIC_W_HIDDEN_DIM = 128

@experiment_registry.register
class PileDCLlamaMediumNoQKNormTHADyn(PileDCLlamaMediumNoQKNorm):
  MERGE_DW_PROJ = False
  DYNAMIC_W_HIDDEN_DIM = 16*16
  DECOMPOSE_DYNAMIC_W = False  # v4 0.2657
  DYNAMIC_D_INIT = None

@experiment_registry.register
class PileDCLlamaMediumNoQKNormTHADynNoDW1Norm(PileDCLlamaMediumNoQKNormTHADyn):
  DW1_NORM_CLS = None  # v3 0.1589
  # loss is quite low, but suddenly turns to nan @7740 without preemption

@experiment_registry.register
class PileDCLlamaMediumNoQKNormTHADynOrig(PileDCLlamaMediumNoQKNormTHADyn):
  DYNAMIC_W_HIDDEN_DIM = None  # v4 0.2688
  LOGITS_DW_ACTIVATION_CLS = None
  PROBS_DW_ACTIVATION_CLS = None
  DW_ACTIVATION_CLS = None

@experiment_registry.register
class PileDCLlamaMediumNoQKNormTHADynOrigNoDW1Norm(PileDCLlamaMediumNoQKNormTHADynOrig):
  DW1_NORM_CLS = None  # v3 0.1579

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormMergeDWMulMul(PileDCLlamaMediumDWDDNoQKNorm):
  MERGE_DW_OP = [operator.mul, operator.mul]

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormMergeDWMulAdd(PileDCLlamaMediumDWDDNoQKNorm):
  MERGE_DW_OP = [operator.mul, operator.add]

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormMergeDWAddMul(PileDCLlamaMediumDWDDNoQKNorm):
  MERGE_DW_OP = [operator.add, operator.mul]  # slightly better than PileDCLlamaMediumDWDDNoQKNorm

@experiment_registry.register
class C4DCLlamaMedium1Kx16(DataParams, _DCConfig, _MediumConfig, C4SpmdLlamaMedium, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  LR_COS_WARMUP = 256
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 50000
  LEARNING_RATE = 2e-4

  PERCORE_BATCH_SIZE = 16
  MAX_SEQ_LEN = 1024

  DYNAMIC_SQUEEZE_RATIO = 8
  DYNAMIC_W_HIDDEN_DIM = 64
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00014)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00022) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01

@experiment_registry.register
class C4DCLlamaMedium1Kx16SW(C4DCLlamaMedium1Kx16):
  LOGITS_DYNAMIC_W_INIT = None
  PROBS_DYNAMIC_W_INIT = None
  DYNAMIC_W_INIT = None
  DYNAMIC_D_INIT = None

@experiment_registry.register
class C4DCLlamaMedium1Kx16NoWinNoQKNorm(C4DCLlamaMedium1Kx16):
  NUM_LAYERS_PER_BLOCK = 1
  WINDOW_SIZE = None
  QK_NORM = False

@experiment_registry.register
class C4DCLlamaMedium1Kx16SWNoWinNoQKNorm(C4DCLlamaMedium1Kx16SW):
  NUM_LAYERS_PER_BLOCK = 1
  WINDOW_SIZE = None
  QK_NORM = False

@experiment_registry.register
class C4DCLlamaMedium1Kx16NoQKNorm(C4DCLlamaMedium1Kx16):
  QK_NORM = False

@experiment_registry.register
class C4DCLlamaMedium1Kx16SWNoQKNorm(C4DCLlamaMedium1Kx16SW):
  QK_NORM = False

@experiment_registry.register
class C4DCLlamaXL1Kx16(DataParams, _DCConfig, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  MAX_SEQ_LEN = 1024

  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00013)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00015) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01

@experiment_registry.register
class C4DCLlamaXL1Kx16SW(_SWConfig, C4DCLlamaXL1Kx16): pass

@experiment_registry.register
class C4DCLlamaXL1Kx16SWNoWinNoQKNorm(C4DCLlamaXL1Kx16SW):
  NUM_LAYERS_PER_BLOCK = 1
  WINDOW_SIZE = None
  QK_NORM = False

@experiment_registry.register
class C4LlamaXL1Kx16(DataParams, C4SpmdLlamaXLHead16x128):
  MAX_SEQ_LEN = 1024

@experiment_registry.register
class C4LlamaXL1Kx16QKNorm(C4LlamaXL1Kx16):
  QK_NORM = True
  
@experiment_registry.register
class PileLlamaMediumWin256(PileLlamaMedium):
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]

@experiment_registry.register
class PileLlamaMediumWin256QKNorm(PileLlamaMediumWin256):
  QK_NORM = True

@experiment_registry.register
class PileLlamaMediumQKNorm(PileLlamaMedium):
  QK_NORM = True

@experiment_registry.register
class PileGPTMediumQKNorm(PileGPTMedium):
  QK_NORM = True

class _SmallConfig(_CommonConfig):
  LR_COS_WARMUP = 48
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_MIN_RATIO = 0.1
  
  ICI_MESH_SHAPE = [1, 32, 1]
  # v1
  PERCORE_BATCH_SIZE = 8
  LEARNING_RATE = 6e-4
  LR_COS_DECAY_END = 4800
  SAVE_ON_STEPS = list(range(1000, 5000, 1000)) + [4800]
  # stepsx4
  # PERCORE_BATCH_SIZE = 2
  # LEARNING_RATE = 3e-4
  # LR_COS_DECAY_END = 19200
  # SAVE_ON_STEPS = list(range(5000, 20000, 5000)) + [19200]

@experiment_registry.register
class PileLlamaSmall(PileDataParams, _SmallConfig, C4SpmdLlamaSmall):
  pass  # v3 1.43; stepsx4 v3 4.75

@experiment_registry.register
class PileLlamaSmallLR0006(PileDataParams, _SmallConfig, C4SpmdLlamaSmall):
  LEARNING_RATE = 6e-4  # stepsx4 3e-r -> 6e-4

@experiment_registry.register
class PileGPTSmall(C4SpmdGPTSepEmb, PileLlamaSmall):
  HIDDEN_DIMS = 3072  # v3 1.46; stepsx4 v3 4.84

@experiment_registry.register
class PileDCLlamaSmall(PileDataParams, _DCConfig, _SmallConfig, C4SpmdLlamaSmall, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  DYNAMIC_SQUEEZE_RATIO = 6  # v3 0.612
  DYNAMIC_W_HIDDEN_DIM = 48
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.0002)  # sqrt(1 / HD) * 2 / (M + I) * 0.01, total_scale <= 0.01
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00025) # sqrt(2 / (D + M)) * 0.005, total_scale <= 0.01

@experiment_registry.register
class PileDCLlamaSmallRellocA4M4L18(PileDCLlamaSmall):
  HIDDEN_DIMS = 1024  # v3 0.4357
  NUM_LAYERS = 18

@experiment_registry.register
class PileDCLlamaSmallRellocA4M4D960(PileDCLlamaSmall):
  MODEL_DIMS = 960  # v3 dhead128/64 0.191/222
  NUM_HEADS = 15  # 12 * 1.25
  HIDDEN_DIMS = 1184  # 960 * 3.68 / 3

@experiment_registry.register
class PileLlamaSmallQKNorm(PileLlamaSmall):
  QK_NORM = True

@experiment_registry.register
class PileDCLlamaSmallNoQKNorm(PileDCLlamaSmall):
  QK_NORM = False

@experiment_registry.register
class PilePythia7B128x1(PileDataParams, Pythia7B128x1):
  SAVE_ON_STEPS = [1000, 6000, 9000] + list(range(3000, 143000, 10000))  # v3 0.521
  # _fix run: restart @8070

@experiment_registry.register
class PilePythia7B128x1FixRot(PilePythia7B128x1):
  PYTHIA_ROTARY = True  # v3 0.519  restart @600
  SAVE_ON_STEPS = [0, 1, 4, 8, 128, 256, 512] + [1000, 6000, 9000] + list(range(3000, 143000, 10000))

@experiment_registry.register
class PilePythia7B128x1FixRotAlignInit(PythiaInit7B, PilePythia7B128x1FixRot):
  pass

@experiment_registry.register
class PilePythia7B256x1FixRotAlignInit(PileDataParams, PythiaInit7B, Pythia7B256x1):
  PYTHIA_ROTARY = True  # v3 0.0892 vs PilePythia7B256x1Win256 v4 0.182 much slower
  SAVE_ON_STEPS = [0, 1, 4, 8, 128, 256, 512] + [1000, 6000, 9000] + list(range(3000, 143000, 10000))
  # _earlyckpt run: restart @7470

@experiment_registry.register
class PilePythia7B256x1Win256(PileDataParams, Pythia7B256x1):
  NUM_LAYERS_PER_BLOCK = 2
  WINDOW_SIZE = [256, None]  # v3 0.09; v4 0.182 vs PilePythia7B256x1DynWFFN16HD128Win256 v4 0.130
  PYTHIA_ROTARY = True  # v3 0.08974 vs PilePythia7B256x1FixRotAlignInit v3 0.0892 very close
  QK_NORM = True  # v3 0.08874
  SUMMARY_INTERVAL_STEPS = 5

@experiment_registry.register
class Llama7B256x1DynWFFN16HD128Whole(C4SpmdLlama7B256x1, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  NUM_GROUPS = 1  # v4 pbs2 chunk128/256 0.202 (vs baseline 0.295, +46%) chunk512 0.184, rank4 0.186; pbs4 0.108 (vs baseline 0.166, +53.7%!); pbs8 0.0554 (vs baseline 0.09, +62.5%)
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00013)  # sqrt(1 / HD) * 2 / (M + I) * 0.025, total_scale = 0.02
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.0002) # sqrt(2 / (D + M * 2)) * 0.01, total_scale = 0.02
  DYNAMIC_SQUEEZE_RATIO = 16 #// 2
  DYNAMIC_W_HIDDEN_DIM = 128 #* 2
  QUERY_CHUNK_SIZE = 128   # pbs4 NoWin chunk64/256 0.103/0.1038
  NUM_LAYERS_PER_BLOCK = 2
  # WINDOW_SIZE = [None, 128]  # pbs2 chunk128 0.231, win128 0.279
  WINDOW_SIZE = [None, 256]  # pbs2 win 256 ~0.237 unstable (vs baseline 0.319 +34.6); pbs4 win128/256/384 0.1276/0.1267/0.123 (vs baseline win256 0.1750, +38%); pbs8 win256 0.0647 (vs baseline 0.095, +46.8%)
  SUMMARY_INTERVAL_STEPS = 5

@experiment_registry.register
class Llama7B64x4DynWFFN8HD16Whole(C4SpmdLlama7B64x4, C4SpmdLlamaXLResTHDynWFFN8HD64DW1RmsNormWhole):
  # NUM_GROUPS = 2  # v3 chunk128/256 0.03/0.029
  NUM_GROUPS = 4  # v4 0.0418?
  DYNAMIC_W_INIT = WeightInit.Gaussian(0.00013)  # sqrt(1 / HD) * 2 / (M + I) * 0.025, total_scale = 0.02
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.0002) # sqrt(2 / (D + M * 2)) * 0.01, total_scale = 0.02
  DYNAMIC_SQUEEZE_RATIO = 8
  DYNAMIC_W_HIDDEN_DIM = 16
  QUERY_CHUNK_SIZE = 128
  SUMMARY_INTERVAL_STEPS = 5

@experiment_registry.register
class Pythia7B256x1DynWFFN16HD128Whole(Llama7B256x1DynWFFN16HD128Whole):
  # pbs4
  GPT_J_RESIDUAL = True  # v4 NoWin 0.109

@experiment_registry.register
class Llama7B256x1DynWFFN16HD1281to4(Llama7B256x1DynWFFN16HD128Whole):
  # NUM_EARLY_LAYERS = 4
  NUM_LAYERS_PER_BLOCK = 4  # v4 pbs2 0.206 improvement so small vs Llama7B256x1DynWFFN16HD128Whole v4 0.202
  # NoEarly v4 pbs4 NoWin 0.1072 vs Whole 0.108 slower!?, win256 0.1328 vs Whole 0.1267 little faster

  LOGITS_USE_STATIC_W_EARLY = True
  PROBS_USE_STATIC_W_EARLY = True
  LOGITS_DYNAMIC_W_INIT_EARLY = WeightInit.Gaussian(0.00003)
  PROBS_DYNAMIC_W_INIT_EARLY = WeightInit.Gaussian(0.00003)
  LOGITS_DYNAMIC_D_INIT_EARLY = WeightInit.Gaussian(0.00012)
  PROBS_DYNAMIC_D_INIT_EARLY = WeightInit.Gaussian(0.00012)

  WINDOW_SIZE = [None, 256, None, 256]
  # WINDOW_SIZE = [None, None, None, None]
  LOGITS_USE_STATIC_W = [True, False, False, False]
  PROBS_USE_STATIC_W = [True, True, True, True]
  LOGITS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), None, None, None]
  PROBS_DYNAMIC_W_INIT = [WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003), WeightInit.Gaussian(0.00003)]
  LOGITS_DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), None, None, None]
  PROBS_DYNAMIC_D_INIT = [WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012), WeightInit.Gaussian(0.00012)]

@experiment_registry.register
class Llama7BFat256x1DynWFFN16HD128Whole(C4SpmdLlama7BFFN16512, Llama7B256x1DynWFFN16HD128Whole):
  pass  # v4 0.226

@experiment_registry.register
class Llama7BFat256x1DynWFFN16HD1281to4(C4SpmdLlama7BFFN16512, Llama7B256x1DynWFFN16HD1281to4):
  pass  # v4 0.229

@experiment_registry.register
class Llama7BResTHLogitsFFN2GELUDynWChunk(C4SpmdLlama7B, C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormNoDDTanh):
  MAX_SEQ_LEN = 1024 #* 2  # v4 rank2 chunk128 0.120, rank2 chunk256 0.118
  NUM_LAYERS = 13
  PERCORE_BATCH_SIZE = 16 #// 2
  ICI_MESH_SHAPE = [1, 32, 1]
  SUMMARY_INTERVAL_STEPS = 5

  PROJECT_LOGITS = True  # rank2looponly 0.198 compilation and first few steps very slow?
  PROJECT_PROBS = True
  DYNAMIC_SQUEEZE_RATIO = 8 * 2
  DYNAMIC_W_HIDDEN_DIM = 256 // 2
  QUERY_CHUNK_SIZE = 128  # v4 rank4 0.156, 
  # rank1only-dd 0.199, ran4looponly-dd 0.167!?, rank4only-dd 0.169
  # rank2 0.159, rank2only 0.170, rank2only-dd 172, SW 0.198, SW+dd 0.193, ddonly 0.217/0.22?!, 
  # rank2loop 0.163, rank2looponly 0.168?, rank2looponly-dd 0.181
  # rank2 NoChunk 0.119 rank16 0.255/2=0.128 rank2probs 0.183
  # btns 0.154

  # baseline 1/0.22 = 4.545
  # ran1only-dd 1/0.199 = +0.48 5.025
  # rand2looponly-dd 1/0.181 = +0.98 5.525 ~ ran1only-dd*2
  # SW+dd 1/0.193 = +0.6363 5.1813
  # rand2loop 1/0.163 = +1.59 6.135 ~ rand2looponly-dd + SW+dd
  
  # SW 1/0.198 = + 0.5055 5.0505
  # rand2looponly 1/0.168 = +1.4074 5.9524
  # SW + rand2looponly > rand2loop

  # rank2only 1/0.170 = +1.337 5.882
  # rank2 1/0.159 = +1.7443 6.289 ~ rank2only + SW
  TRANSPOSE = False
  PROBS_ABSORB_RESIDUAL = False
  LOOP_OVER_DYNAMIC_HD = True
  SUMMARY_VERBOSITY = 9
  # ParaMLP # v4 rank2 0.158
  # remove dim G  # rank2 0.136?!
  
@experiment_registry.register
class Llama7BBaseline(C4SpmdLlama7B):
  MAX_SEQ_LEN = 1024 #* 2  # v4 0.174
  NUM_LAYERS = 13  # v4 0.212, Chunk128 0.22
  PERCORE_BATCH_SIZE = 16 #// 2
  ICI_MESH_SHAPE = [1, 32, 1]
  SUMMARY_INTERVAL_STEPS = 5

  SUMMARY_VERBOSITY = 9
  # QUERY_CHUNK_SIZE = 128
  # NoMLP  # v4 0.561
  # ParaMLP # v4 0.219

@experiment_registry.register  # praxis 29dcf7b, paxml bcccfea
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormLearnedDDCap(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormNoDDTanh):
  LEARNED_DW_CAP = {'qdd': 1., 'kdd': 1.}

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormProbsDWCap(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNorm):  
  # debug, exactly align with C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNorm_cap
  PROBS_DW_CAP = {'qw2': 2., 'kw2': 2., 'qdd': 1., 'kdd': 1.}

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNormQKBias(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNorm):
  USE_QK_BIAS = True
  
@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD32DW1RmsNormLearnedCap(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD32NoBiasLearnDiagDW1RmsNorm):
  PROBS_DW_CAP = None
  LEARNED_DW_CAP = {'qw2': 1., 'kw2': 1., 'qdd': 1., 'kdd': 1.}
  SAVE_ON_STEPS = list(range(10000, 70000, 10000))

@experiment_registry.register
class MediumResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNorm(C4SpmdLlamaMedium, C4SpmdLlamaXLResTHLogitsFFN2GELUDynWFFN8HD64DW1RmsNorm):
  SUMMARY_VERBOSITY = 9 # v3 w/ summary 0.081, w/o summary 0.102 << C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiagDWTanh v3 0.163 << C4SpmdLlamaMediumResTHFFN16DynW0003LearnDiag v4 0.308
  SAVE_ON_STEPS = None

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD32NoBiasLearnDiagDWTanh(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh):
  USE_SQUEEZE_BIAS = False
  DYNAMIC_W_HIDDEN_DIM = 32  # v4
  DW_HIDDEN_ACTIVATION_CLS = layers.GELU
  USE_DW_HIDDEN_BIAS = False
  PROBS_ABSORB_RESIDUAL = False
  SUMMARY_VERBOSITY = 3

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD32NoBiasLearnDiag(C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD32NoBiasLearnDiagDWTanh):
  LOGITS_DW_ACTIVATION_CLS = None
  PROBS_DW_ACTIVATION_CLS = None
  DW_CAP = {'qw1': 1., 'kw1': 1., 'qw2': 2., 'kw2': 2., 'dd': 1.}  # add @

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynWHD24GateSiLULearnDiagDW1RmsNormBias1e_6(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBias1e_6):
  DYNAMIC_W_HIDDEN_DIM = 24  # v4 0.111
  DW_HIDDEN_GATE_ACT_CLS = layers.SiLU
  SKIP_BIAS = True
  SUMMARY_VERBOSITY = 3

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormDBias(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBias1e_6):
  DW1_NORM_DBIAS_INIT = WeightInit.Gaussian(0.00003)  # v4 0.111
  DW1_NORM_BIAS_CONST = -1e-6  # cancel out RmsNormNoScale.epsilon

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBiasDDSiLU(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBias1e_6):
  DW_ACTIVATION_WEIGHTS = ['dd']  # v4 0.093

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormOnlyDiagHD16(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBias1e_6): # diverge
  DW_ACTIVATION_WEIGHTS = ['dd']  # v4 0.096
  DD_ACTIVATION_CLS = layers.GELU
  SUMMARY_VERBOSITY = 3

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELUDynWHD32LearnDiagDW1RmsNormBiasAllBias(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNorm):
  # class name should be C4SpmdLlamaMediumResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBiasAllBias
  NUM_LAYERS = 24
  MODEL_DIMS = 1024
  HIDDEN_DIMS = 2816  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 16
  DIMS_PER_HEAD = 64
  
  LR_COS_DECAY_END = 50000
  ICI_MESH_SHAPE = [1, 32, 1] 

  DW1_NORM_BIAS_INIT = 1e-6  # v3 0.152
  USE_BIAS = True
  SAVE_ON_STEPS = [0, 10, 20, 40, 100, 200, 400, 800]

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELUDynWHD32LearnDiagDW1RmsNormBiasNoBias(C4SpmdLlamaMediumResTHLogitsFFN2GELUDynWHD32LearnDiagDW1RmsNormBiasAllBias):
  # class name should be C4SpmdLlamaMediumResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormBiasNoBias
  # loss nan at very beginning after fix dw1_norm_bias
  USE_BIAS = False  # v3 0.152

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELUDynW00003LearnDiagDW1RmsNormSquareBiasFP32logitsLRx10(C4SpmdLlamaMediumResTHLogitsFFN2GELUDynWHD32LearnDiagDW1RmsNormBiasAllBias):
  USE_BIAS = False
  SQUARE_DW1_NORM_BIAS = True
  DW1_NORM_BIAS_INIT = 1e-3
  FLOAT32_LOGITS = True
  LEARNING_RATE = 2e-3

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW00003LearnDiagDW1Tanh(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh):
  # W00003 is wrongly copied and misleading. Actually W0003
  DW_ACTIVATION_WEIGHTS = ['qw1', 'kw1']  # v4 0.128 why slower than C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh
  PROBS_OUTPUT_ACTIVATION_CLS = None
  DYNAMIC_D_INIT = WeightInit.Gaussian(0.00012)
  SAVE_ON_STEPS = [0, 200, 1000, 5000] + list(range(10000, 60000, 10000))

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagProbsDW1Tanh(C4SpmdLlamaXLResTHLogitsFFN2GELUDynW0003LearnDiagDWTanh):
  PROBS_DW_ACTIVATION_WEIGHTS = ['qw1', 'kw1']
  PROBS_OUTPUT_ACTIVATION_CLS = None
  SAVE_ON_STEPS = [0, 200, 1000, 5000] + list(range(10000, 60000, 10000))

@experiment_registry.register
class C4SpmdLlamaXLResTHLogitsFFN2GELUProbsDynWHD16LearnDiagDWTanh(C4SpmdLlamaXLResTHLogitsFFN2GELUProbsDynW0003LearnDiagDWTanh):
  PROBS_DYNAMIC_W_HIDDEN_DIM = 16  # v3 0.078
  PROBS_DW_HIDDEN_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaXLTHLogitsFFN2GELUResProbs(C4SpmdLlamaXLResTHLogitsFFN2GELUProbs):
  LOGITS_RESIDUAL = False   # v4 0.154

@experiment_registry.register
class C4SpmdLlamaXLHead16x128THLogitsFFN2GELUResProbs(C4SpmdLlamaXLTHLogitsFFN2GELUResProbs):
  NUM_HEADS = 16  # 0.189
  DIMS_PER_HEAD = 128

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELUProbsBS1(C4SpmdLlamaMediumResTHLogitsFFN2GELUProbs):
  PERCORE_BATCH_SIZE = 1

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELUProbsReLU(C4SpmdLlamaMediumResTHLogitsFFN2GELUProbs):
  PROBS_OUTPUT_ACTIVATION_CLS = layers.ReLU  # 0.311

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELUProbsv4(C4SpmdLlamaMediumResTHLogitsFFN2GELUProbs):
  PERCORE_BATCH_SIZE = 32
  ICI_MESH_SHAPE = [1, 16, 1]  # 0.318, transpose 0.323, global transpose 0.322

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogits(C4SpmdLlamaMedium):
  NUM_GROUPS = 1  # 0.307
  PROJECT_LOGITS = True 

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsv4(C4SpmdLlamaMediumResTHLogits):
  PERCORE_BATCH_SIZE = 32
  ICI_MESH_SHAPE = [1, 16, 1]  # 0.409

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsGaussian05(C4SpmdLlamaMediumResTHLogits):
  SCALE_INIT = WeightInit.Gaussian(0.05)

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2ProbsFFN2(C4SpmdLlamaMediumResTH):
  LOGITS_SQUEEZE_RATIO = 2  
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU
  PROBS_SQUEEZE_RATIO = 2    # v4
  PROBS_SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumTHLogits(C4SpmdLlamaMediumResTHLogits):
  LOGITS_RESIDUAL = False  # 0.38

@experiment_registry.register
class C4SpmdLlamaMediumTH(C4SpmdLlamaMediumResTH):
  LOGITS_RESIDUAL = False
  PROBS_RESIDUAL = False

@experiment_registry.register
class C4SpmdLlamaMediumTHGaussian25(C4SpmdLlamaMediumTH):
  SCALE_INIT = WeightInit.Gaussian(0.25) # 0.383

@experiment_registry.register
class C4SpmdLlamaMediumTHLogitsFFNProbs(C4SpmdLlamaMediumTH):
  LOGITS_SQUEEZE_RATIO = 2   # v4 0.47
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumTHLogitsFFNProbsFFN(C4SpmdLlamaMediumTH):
  LOGITS_SQUEEZE_RATIO = 2  
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU
  PROBS_SQUEEZE_RATIO = 2    # v4 0.437
  PROBS_SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumTHHEAD64x16Gaussian125(C4SpmdLlamaMediumTH):
  NUM_HEADS = 64
  DIMS_PER_HEAD = 16
  SCALE_INIT = WeightInit.Gaussian(0.125)  # 0.21

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN1GELU(C4SpmdLlamaMediumResTHLogits):
  SQUEEZE_RATIO = 1  # 0.279, bias 0.271
  SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2GELU(C4SpmdLlamaMediumResTHLogits):
  LOGITS_SQUEEZE_RATIO = 2  # 0.293, bias 0.290
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.GELU

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2ReLU(C4SpmdLlamaMediumResTHLogits):
  LOGITS_SQUEEZE_RATIO = 2  # 
  LOGITS_SQUEEZE_ACTIVATION_CLS = layers.ReLU

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2ReLUv4(C4SpmdLlamaMediumResTHLogitsFFN2ReLU):
  PERCORE_BATCH_SIZE = 32
  ICI_MESH_SHAPE = [1, 16, 1]  # 0.386

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN2(C4SpmdLlamaMediumResTHLogits):
  SQUEEZE_RATIO = 2  #

@experiment_registry.register
class C4SpmdLlamaMediumTHLogitsFFN2GELU(C4SpmdLlamaMediumResTHLogitsFFN2GELU):
  LOGITS_RESIDUAL = False

@experiment_registry.register
class C4SpmdLlamaMediumResTHLogitsFFN4GELU(C4SpmdLlamaMediumResTHLogitsFFN2GELU):
  SQUEEZE_RATIO = 4  # 

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbs(C4SpmdLlamaMedium):
  NUM_GROUPS = 1  # 0.304
  PROJECT_PROBS = True

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbsFFN2(C4SpmdLlamaMediumResTHProbs):
  PROBS_SQUEEZE_RATIO = 2  #

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbsFFN2GELU(C4SpmdLlamaMediumResTHProbsFFN2):
  SQUEEZE_ACTIVATION_CLS = layers.GELU  #

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbsFFN2v4(C4SpmdLlamaMediumResTHProbs):
  PERCORE_BATCH_SIZE = 32
  ICI_MESH_SHAPE = [1, 16, 1]  # 0.395
  SQUEEZE_RATIO = 2  #

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbsFFN2GELUv4(C4SpmdLlamaMediumResTHProbsFFN2v4):
  SQUEEZE_ACTIVATION_CLS = layers.GELU  # 0.385

@experiment_registry.register
class C4SpmdLlamaMediumResTHProbsFFN2ReLUv4(C4SpmdLlamaMediumResTHProbsFFN2v4):
  SQUEEZE_ACTIVATION_CLS = layers.ReLU

@experiment_registry.register
class C4SpmdLlamaMediumResTHFP32logits(C4SpmdLlamaMediumResTH):
  FLOAT32_LOGITS = True  # 0.149

@experiment_registry.register
class C4SpmdLlamaMediumResTHGaussian05(C4SpmdLlamaMediumResTH):
  SCALE_INIT = WeightInit.Gaussian(0.05)

@experiment_registry.register
class C4SpmdLlamaMediumResTHx2(C4SpmdLlamaMediumResTH):
  NUM_HEADS = 16 * 2  # res 0.105, v5 0.195

@experiment_registry.register
class C4SpmdLlamaMediumResTHx4(C4SpmdLlamaMediumResTH):
  NUM_HEADS = 16 * 4  # res 0.059

@experiment_registry.register
class C4SpmdGpt3XLRoPE(C4SpmdGpt3SmallRoPE):
  NUM_LAYERS = 24
  MODEL_DIMS = 2048
  HIDDEN_DIMS = 5504  # XD: MODEL_DIMS * 4 * 2 // 3
  NUM_HEADS = 16
  # DIMS_PER_HEAD = 128
  
  PERCORE_BATCH_SIZE = 4 * 4
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

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGLL(PilePythia7B256x1DynWFFN16HD128Win256Aligned): #mqy
  WINDOW_SIZE = [256,None,256,256] #LGLL local:global 3:1 # v4:0.1403
  NUM_LAYERS_PER_BLOCK = 4

@experiment_registry.register
class Pythia7BLSP(DataParams, C4SpmdGpt37BRoPE):
    NUM_LAYERS = 32
    NUM_HEADS = 32
    DIMS_PER_HEAD = 128
    COMBINE_QKV = False
    MODEL_DIMS = 4096
    HIDDEN_DIMS = 16384
    VOCAB_SIZE = 50432

    PERCORE_BATCH_SIZE = 2
    ICI_MESH_SHAPE = [1, 8, 1]

    CHECKPOINT_EVERY_N_STEPS = 500
    EVAL_LOOP_NUM_BATCHES = 50
    EVAL_INTERVAL_STEPS = 250
    CHECKPOINT_MAX_TO_KEEP = 2
   
    LAYERNORM_EPSILON = 1e-05
    # Learning rate schedule
    LEARNING_RATE = 1e-5
    LR_SCHEDULE = "linear_rampup_cosine_decay"
    # 最大学习率 * LR_LRED_MIN_RATIO： 最后保持稳定的学习率,即step > LR_COS_DECAY_END时的学习率
    LR_COS_MIN_RATIO = 0.1
    LR_COS_MAX = 1.0  # 这是cos曲线的最大值，和pytorch的cos曲线的学习率不是一个值，这个值 * LEARNING_RATE就是pytorch设定的值
    # warmup step: 学习率从 0 -> LR_COS_MAX的步数, easyl: ratio, 0.02 * LR_COS_DECAY_END = 1170
    LR_COS_WARMUP = 200
    LR_COS_DECAY_START = LR_COS_WARMUP + 1  # decay start step: 学习率开始衰减的步数
    LR_COS_DECAY_END = 10000  # decay end step # 学习率最后保持恒定的步数
    WEIGHT_DECAY = 0.0
    ADAM_BETA2 = 0.95
    ADAM_BETA1 = 0.9
    ADAM_EPSILON = 1e-8
    CLIP_GRADIENT_NORM_TO_VALUE = 1.0

    # TASK_NAME = "Pythia7B"
    # WANDB_PROJECT = "pythia_7b_test"

    TRAINING_SEED = 1234

    QUERY_CHUNK_SIZE = 512
    Z_LOSS_WEIGHT = 0.0
    LM_HEAD_NORM = False
    TRAINABLE_POSITION_EMB = False
    USE_ROTARY_POSITION_EMB = True
    USE_ALIBI_POSITION_EMB = False
    NORMALIZATION_CLS = normalizations.LayerNorm
    USE_BIAS = True
    USE_GATED_ACTIVATION = False # no ff1_layer_gate
    ACTIVATION_CLS = layers.GELU
    ROTARY_TYPE = 'pythia'

    MAX_SEQ_LEN = 2049  # ps：pythia读取的数据长度为2049

    LOAD_SEQIO_TEXT = False
    SHUFFLE = {'train': False, 'test': False}
    VOCABULARY = t5.data.PassThroughVocabulary(size=VOCAB_SIZE)
    KEY_MAP = {"targets": "input_ids", "masks": "input_ids"}
    DATA_PATH = {
                'train': 'gs://common_datasets/pythia_pile_idxmaps_tfrecored', 
                'test':  'gs://common_datasets/pythia_pile_idxmaps_tfrecored', 
                }
    DATA_FUNC = extract_pythia_datapath
    # LOAD_SEQIO_ID = False

    LOAD_SEQIO_ID = True
    TASK_NAME = 'pythia'

@experiment_registry.register
class BaseEval():
    TRAINING_NUM_BATCHES_TO_SKIP = None
    ICI_MESH_SHAPE = [1, 32, 1]
    PERCORE_BATCH_SIZE = 32
    LM_HEAD_CHUNK_SIZE = None
    FPROP_DTYPE = jnp.bfloat16
    SHUFFLE = {"train": False, "test": False}
    ONLY_EVAL = True
    TEST_RATIO = 1
    RESET_FOR_EVAL = False
    EVAL_LOOP_NUM_BATCHES = 1
    DATA_FUNC = extract_pythia_datapath

@experiment_registry.register
class PileEval(BaseEval):
    ZERO_LOSS = True
    LOSS_BATCH_MEAN = False
    ACC_BATCH_MEAN = False

    DATA_PATH = {
                'train': 'gs://common_datasets_us-east5/pythia_model_test/pile_test/', 
                'test':  'gs://common_datasets_us-east5/pythia_model_test/pile_test/', 
                }
    KEY_MAP = {"targets": "input_ids", "masks": "input_ids"}
    TASK_NAME = 'Pile'
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = True
    
@experiment_registry.register
class FlanMiniEval(BaseEval):
    ZERO_LOSS = False
    LOSS_BATCH_MEAN = True
    ACC_BATCH_MEAN = True

    DATA_PATH = {
                'train': 'gs://common_datasets/pythia_model_test/flan_test3/', 
                'test':  'gs://common_datasets/pythia_model_test/flan_test3/', 
                }
    KEY_MAP = {"targets": "input_ids", "labels": "labels"}
    TASK_NAME = 'FlanMini'
    EVAL_LOOP_NUM_BATCHES = 320

@experiment_registry.register
class PilePythiaXLDynWFFN8HD64Win256AlignedPileEval(PileEval, PilePythiaXLDynWFFN8HD64Win256Aligned):
  TASK_NAME = 'PilePythiaXLDynWFFN8HD64Win256AlignedPileEval'

@experiment_registry.register
class PilePythia410M128x1DynWFFN8HD64Win256AlignedPileEval(PileEval, PilePythia410M128x1DynWFFN8HD64Win256Aligned):
  TASK_NAME = 'SelfModelPile'

@experiment_registry.register
class PilePythiaXLDynWFFN8HD64Win256AlignedFlanMiniEval(FlanMiniEval, PilePythiaXLDynWFFN8HD64Win256Aligned):
  TASK_NAME = 'PilePythiaXLDynWFFN8HD64Win256AlignedFlanMini'

@experiment_registry.register
class PilePythia410M128x1DynWFFN8HD64Win256AlignedFlanMiniEval(FlanMiniEval, PilePythia410M128x1DynWFFN8HD64Win256Aligned):
  TASK_NAME = 'SelfModelPile'

@experiment_registry.register
class PilePythiaXL128x1FixRotPileEval(PileEval, PilePythiaXL128x1FixRot):
  TASK_NAME = 'SelfModelPile'

@experiment_registry.register
class PilePythiaXL128x1FixRotFlanMiniEval(FlanMiniEval, PilePythiaXL128x1FixRot):
  TASK_NAME = 'SelfModelFlanMini'

@experiment_registry.register
class PilePythia410M64x1FixRotPileEval(PileEval, PilePythia410M64x1FixRot):
  TASK_NAME = 'SelfModelPile'

@experiment_registry.register
class PilePythia410M64x1FixRotFlanMiniEval(FlanMiniEval, PilePythia410M64x1FixRot):
  TASK_NAME = 'SelfModelFlanMini'

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256AlignedPileEval(PileEval, PilePythia7B256x1DynWFFN16HD128Win256Aligned):
  TASK_NAME = 'PilePythia7B256x1DynWFFN16HD128Win256AlignedPileEval'

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256AlignedFlanMiniEval(FlanMiniEval, PilePythia7B256x1DynWFFN16HD128Win256Aligned):
  TASK_NAME = 'SelfModelFlanMiniSmallPile'

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256PileEval(PileEval, PilePythia7B256x1DynWFFN16HD128Win256):
  TASK_NAME = 'PilePythia7B256x1DynWFFN16HD128Win256PileEval'

@experiment_registry.register
class PilePythia7B128x1PileEval(PileEval, PilePythia7B128x1):
  TASK_NAME = 'PilePythia7B128x1PileEval'

@experiment_registry.register
class PileDCLlamaXLFixDWInitPileEval(PileEval, PileDCLlamaXLFixDWInit):
  TASK_NAME = 'PileDCLlamaXLFixDWInitPileEvalWhole'

@experiment_registry.register
class PileDCLlamaXLFixDWShapePileEval(PileEval, PileDCLlamaXLFixDWShape):
  TASK_NAME = 'PileDCLlamaXLFixDWShapePileEvalWhole'

@experiment_registry.register
class PileDCLlamaLargePileEval(PileEval, PileDCLlamaLarge):
  TASK_NAME = 'PileDCLlamaLargePileEvalWhole'


@experiment_registry.register
class PileDCLlamaMediumPileEval(PileEval, PileDCLlamaMedium):
  TASK_NAME = 'PileDCLlamaMediumPileEvalWhole'

@experiment_registry.register
class PileGPTSmallPileEval(PileEval, PileGPTSmall):
  TASK_NAME = 'PileGPTSmallPileEvalWhole'

@experiment_registry.register
class PileLlamaSmallPileEval(PileEval, PileLlamaSmall):
  TASK_NAME = 'PileLlamaSmallPileEvalWhole'

@experiment_registry.register
class PileDCLlamaSmallPileEval(PileEval, PileDCLlamaSmall):
  TASK_NAME = 'PileDCLlamaSmallPileEvalWhole'

@experiment_registry.register
class PilePythia3BDynWFFN16HD128Win256AlignedPileEval(PileEval, PilePythia3BDynWFFN16HD128Win256Aligned):
  TASK_NAME = 'PilePythia3BDynWFFN16HD128Win256AlignedPileEvalWhole'

@experiment_registry.register
class PilePythia3BDynWFFN16HD128Win256AlignedFlanMiniEval(FlanMiniEval, PilePythia3BDynWFFN16HD128Win256Aligned):
  TASK_NAME = 'PilePythia3BDynWFFN16HD128Win256AlignedFlanMiniEvalWhole'

# ============= scaling law ============================
@experiment_registry.register
class PileGPTMediumPileEval(PileEval, PileGPTMedium):
  ZERO_LOSS = False
  TASK_NAME = 'PileGPTMediumPileEvalWholeZeroFalse'

@experiment_registry.register
class PileLlamaMediumPileEval(PileEval, PileLlamaMedium):
  ZERO_LOSS = False
  # PileDCLlamaMediumSWNoQKNorm
  TASK_NAME = 'PileLlamaMediumPileEvalWholeZeroFalse'

@experiment_registry.register
class PileGPTLargePileEval(PileEval, PileGPTLarge):
  ZERO_LOSS = False
  TASK_NAME = 'PileGPTLargePileEvalWholeZeroFalse'

@experiment_registry.register
class PileLlamaLargePileEval(PileEval, PileLlamaLarge):
  ZERO_LOSS = False
  TASK_NAME = 'PileLlamaLargePileEvalWholeZeroFalse'

@experiment_registry.register
class PileGPTXLPileEval(PileEval, PileGPTXL):
  # PileGPTXL
  ZERO_LOSS = False
  TASK_NAME = 'PileGPTXLPileEvalWholeZeroFalse'

@experiment_registry.register
class PileLlamaXLPileEval(PileEval, PileLlamaXL):
  # PileLlamaXL
  ZERO_LOSS = False
  TASK_NAME = 'PileLlamaXLPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCGPTMediumDWDDNoQKNormPileEval(PileEval, PileDCGPTMediumDWDDNoQKNorm):
  # PileDCGPTMediumDWDDNoQKNormv4
  ZERO_LOSS = False
  TASK_NAME = 'PileDCGPTMediumDWDDNoQKNormPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCGPTLargeDWDDNoQKNormPileEval(PileEval, PileDCGPTLargeDWDDNoQKNorm):
  # PileDCGPTLargeDWDDNoQKNormv4
  ZERO_LOSS = False
  TASK_NAME = 'PileDCGPTLargeDWDDNoQKNormPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCGPTXLDWDDNoQKNormPileEval(PileEval, PileDCGPTXLDWDDNoQKNorm):
  # PileDCGPTXLDWDDNoQKNormv4
  ZERO_LOSS = False
  TASK_NAME = 'PileDCGPTXLDWDDNoQKNormPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormPileEval(PileEval, PileDCLlamaMediumDWDDNoQKNorm):
  # PileDCLlamaMediumDWDDNoQKNormv4
  ZERO_LOSS = False
  TASK_NAME = 'PileDCLlamaMediumDWDDNoQKNormPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaLargeDWDDNoQKNormPileEval(PileEval, PileDCLlamaLargeDWDDNoQKNorm):
  # PileDCLlamaLargeDWDDNoQKNormv4
  ZERO_LOSS = False
  TASK_NAME = 'PileDCLlamaLargeDWDDNoQKNormPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaXLDWDDNoQKNormPileEval(PileEval, PileDCLlamaXLDWDDNoQKNorm):
  # PileDCLlamaXLDWDDNoQKNormv4
  ZERO_LOSS = False
  TASK_NAME = 'PileDCLlamaXLDWDDNoQKNormPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaMediumNoOKNormPileEval(PileEval, PileDCLlamaMediumNoOKNorm):
    # PileDCLlamaMediumNoOKNormv4
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumNoOKNormPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaMediumSWNoQKNormPileEval(PileEval, PileDCLlamaMediumSWNoQKNorm):
    # PileDCLlamaMediumSWNoQKNormv4_fix
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumSWNoQKNormPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaMediumDWNoQKNormPileEval(PileEval, PileDCLlamaMediumDWNoQKNorm):
    # PileDCLlamaMediumDWNoQKNormv4
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumDWNoQKNormv4PileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaMediumDDNoQKNormPileEval(PileEval, PileDCLlamaMediumDDNoQKNorm):
    # PileDCLlamaMediumDDNoQKNormv4
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumDDNoQKNormPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormTgtPileEval(PileEval, PileDCLlamaMediumDWDDNoQKNormTgt):
    # PileDCLlamaMediumDWDDNoQKNormTgtv4
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumDWDDNoQKNormTgtPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormSrcPileEval(PileEval, PileDCLlamaMediumDWDDNoQKNormSrc):
    # PileDCLlamaMediumDWDDNoQKNormSrcv4
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumDWDDNoQKNormSrcPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaMediumNoQKNormProjLogitsWorkaroundPileEval(PileEval, PileDCLlamaMediumNoQKNormProjLogitsWorkaround):
    # PileDCLlamaMediumNoQKNormProjLogitsWorkaroundv4
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumNoQKNormProjLogitsWorkaroundPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaMediumNoQKNormProjProbsWorkaroundPileEval(PileEval, PileDCLlamaMediumNoQKNormProjProbsWorkaround):
    # PileDCLlamaMediumNoQKNormProjProbsWorkaroundv4
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumNoQKNormProjProbsWorkaroundPileEvalWholeZeroFalse'


@experiment_registry.register
class PileDCLlamaMediumNoQKNormR1PileEval(PileEval, PileDCLlamaMediumNoQKNormR1):
    # PileDCLlamaMediumNoQKNormR1
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumNoQKNormR1PileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaMediumNoOKNormR1PileEval(PileEval, PileDCLlamaMediumNoOKNorm):
    # PileDCLlamaMediumNoOKNorm
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumNoOKNormPileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlamaMediumNoQKNormR4PileEval(PileEval, PileDCLlamaMediumNoQKNormR4):
    # PileDCLlamaMediumNoQKNormR4v4
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumNoQKNormR4PileEvalWholeZeroFalse'

@experiment_registry.register
class PileDCLlama3B2Kx4x256x1DWDDLR00032PileEval(PileEval, PileDCLlama3B2Kx4x256x1DWDDLR00032):
    # PileDCLlama3B2Kx4x256x1DWDDLR00032v4
    ZERO_LOSS = True
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlama3B2Kx4x256x1DWDDLR00032PileEvalWholeZeroTrue'

@experiment_registry.register
class PileDCLlama3B2Kx4x256x1DWDDLR00032FlanMiniEval(FlanMiniEval, PileDCLlama3B2Kx4x256x1DWDDLR00032):
    # PileDCLlama3B2Kx4x256x1DWDDLR00032v4
    ZERO_LOSS = False
    EVAL_LOOP_NUM_BATCHES = 320
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlama3B2Kx4x256x1DWDDLR00032FlanMiniEvalWholeZeroTrue'

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGLLPileEval(PileEval, PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGLL):
    # PileDCLlamaMediumNoQKNormR4v4
    ZERO_LOSS = True
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGLLPileEval'
    ICI_MESH_SHAPE = [1, 32, 1]
    PERCORE_BATCH_SIZE = 32

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormWindowLGLLPileEval(PileEval, PileDCLlamaMediumDWDDNoQKNormWindowLGLL):
  # PileDCLlamaMediumNoQKNormR4v4  13500
# gs://llm_projects_us-central2/log/PileDCLlamaMediumDWDDNoQKNormWindowLGLLv4/checkpoints/checkpoint_00013500
    ZERO_LOSS = True
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumDWDDNoQKNormWindowLGLLPileEval'
    ICI_MESH_SHAPE = [1, 4, 1]
    PERCORE_BATCH_SIZE = 256

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormWindowLGLLQWPileEval(PileEval, PileDCLlamaMediumDWDDNoQKNormWindowLGLLQW):
  # PileDCLlamaMediumNoQKNormR4v4
# gs://llm_projects/log/PileDCLlamaMediumDWDDNoQKNormWindowLGLLQW/checkpoints/checkpoint_00013500
    ZERO_LOSS = True
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumDWDDNoQKNormWindowLGLLQWPileEval'
    ICI_MESH_SHAPE = [1, 4, 1]
    PERCORE_BATCH_SIZE = 256

@experiment_registry.register
class PileDCLlamaMediumDWDDNoQKNormWindowLGL6PileEval(PileEval, PileDCLlamaMediumDWDDNoQKNormWindowLGL6):
  # PileDCLlamaMediumNoQKNormR4v4
# gs://llm_projects/log/PileDCLlamaMediumDWDDNoQKNormWindowLGL6/checkpoints/checkpoint_00013500
    ZERO_LOSS = True
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    TASK_NAME = 'PileDCLlamaMediumDWDDNoQKNormWindowLGL6PileEval'
    ICI_MESH_SHAPE = [1, 4, 1]
    PERCORE_BATCH_SIZE = 256

# gs://llm_projects_us-central2/log/PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGL6v4/checkpoints/checkpoint_00013000
@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGL6(PilePythia7B256x1DynWFFN16HD128Win256Aligned): #mqy
  WINDOW_SIZE = [256,None,256,256,256,256,256,256] #LGL6 local:global 7:1 # v4: 0.1335
  NUM_LAYERS_PER_BLOCK = 8 

# gs://llm_projects_us-central2/log/PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGQWv4/checkpoints/checkpoint_00013000
@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGQW(PilePythia7B256x1DynWFFN16HD128Win256Aligned): #mqy
  SRC_DEPENDENT = False # LG query wise # v4

# gs://llm_projects_us-central2/log/PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGLLQWv4/checkpoints/checkpoint_00013000
@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGLLQW(PilePythia7B256x1DynWFFN16HD128Win256Aligned): #mqy
  SRC_DEPENDENT = False # query wise # v4:
  WINDOW_SIZE = [256,None,256,256] #LGLL local:global 3:1 # v4:0.1403
  NUM_LAYERS_PER_BLOCK = 4


@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGL6PileEval(PileEval, PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGL6):
  # PileDCLlamaMediumNoQKNormR4v4
# gs://llm_projects/log/PileDCLlamaMediumDWDDNoQKNormWindowLGL6/checkpoints/checkpoint_00013500
    ZERO_LOSS = True
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    CLASS_NAME = 'PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGL6'
    TASK_NAME = CLASS_NAME + 'PileEval'
    ICI_MESH_SHAPE = [1, 16, 1]
    PERCORE_BATCH_SIZE = 64

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGQWPileEval(PileEval, PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGQW):
  # PileDCLlamaMediumNoQKNormR4v4
# gs://llm_projects/log/PileDCLlamaMediumDWDDNoQKNormWindowLGL6/checkpoints/checkpoint_00013500
    ZERO_LOSS = True
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    CLASS_NAME = 'PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGQW'
    TASK_NAME = CLASS_NAME + 'PileEval'
    ICI_MESH_SHAPE = [1, 16, 1]
    PERCORE_BATCH_SIZE = 64

@experiment_registry.register
class PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGLLQWPileEval(PileEval, PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGLLQW):
  # PileDCLlamaMediumNoQKNormR4v4
# gs://llm_projects/log/PileDCLlamaMediumDWDDNoQKNormWindowLGL6/checkpoints/checkpoint_00013500
    ZERO_LOSS = True
    EVAL_LOOP_NUM_BATCHES = 162
    RESET_FOR_EVAL = False
    CLASS_NAME = 'PilePythia7B256x1DynWFFN16HD128Win256AlignedWindowLGLLQW'
    TASK_NAME = CLASS_NAME + 'PileEval'
    ICI_MESH_SHAPE = [1, 16, 1]
    PERCORE_BATCH_SIZE = 64


class MyDatasets(base_input.BaseInput):
    path: Optional[str] = None
    num_infeed_hosts: int = 0
    reset_for_eval: bool = False
    is_training: bool = True
    batch_size: int = 8
    seq_len: int = 2048
    repeat: int = 1
    train_seed: int = 9876
    task_features: Optional[dict] = None
    shuffle_buffer_size: Optional[int] = None
    pad_id: int = 0
    drop_remainder: bool = True
    iter_file_nums: int = 2 # 100  500 steps/file
    meta_dict: Optional[dict] = None
    num_batches_to_skip: Optional[int] = None
    only_eval: bool = False
    zero_loss: bool = True

    def __post_init__(self):
        if self.num_infeed_hosts == 0:
            self.num_infeed_hosts = jax.process_count()

        if not self.meta_dict or self.only_eval:
            self.meta_dict = {}
            self.init_meta()
        else:
            if self.meta_dict["file_in_data"] != 0:
                assert self.meta_dict["iter_file_nums"] == self.iter_file_nums, print(
                    f'iter_file_nums in meta_dict is not equal to cur args. => {self.meta_dict["iter_file_nums"]}≠'
                    f" {self.iter_file_nums}"
                )
            self.step_in_file = self.meta_dict.get('step_in_file')  # XD fix
        logging.info(f'meta_dict: {self.meta_dict}')
        self.train_seed = self.meta_dict['seed']
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)
        self._peek = None
        self._state_before_peek = None

    def init_meta(self):
        self.meta_dict = {
                "seed": self.train_seed,
                "cur_files": self.meta_dict.get('cur_files', []),
                "file_in_data": 0,
                "step_in_file": 0,
                "iter_file_nums": self.iter_file_nums,
                "checkpoint_step": self.meta_dict.get('checkpoint_step', None),
            }
        self.step_in_file = 0

 #   def peek_padded(self):
  #      return self.get_next_padded()

    def reset(self):
        self.init_meta()
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)

    def get_next_padded(self):
        if self._peek is not None:
          output = self._peek
          self._peek = None
          self._state_before_peek = None
          return output
        unpadded = next(self.dataset)
        pad_size = int(self.batch_padding_size)
        if pad_size == 0:
            return unpadded
        return jax.tree_util.tree_map(
            lambda x: np.pad(x, [[0, pad_size]] + [[0, 0]] * (x.ndim - 1)),
            unpadded,
        )

    def get_global_batch_size(self, train_input):
        logging.info(f"train_input: {train_input} type: {type(train_input)}")
        return self.batch_size * self.num_infeed_hosts

    def _parse_function(self, example_proto):
        feature_desc = {key: tf.io.VarLenFeature(tf.int64) for key in self.task_features}
        example = tf.io.parse_single_example(example_proto, feature_desc)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = tf.sparse.to_dense(t, default_value=0)[ :self.seq_len]
        return example

    def convert(self, data):
        seq_len = self.seq_len
        model_needed_inputs = NestedMap()
        model_needed_inputs.ids = data["input_ids"][:, : seq_len - 1]
        model_needed_inputs.labels = data["input_ids"][:, 1:seq_len]
        key = 'labels' if "labels" in data else 'input_ids'
        weights = data[key] >= 0 if self.zero_loss else data[key] > 0
        model_needed_inputs.weights = weights[:, 1:seq_len]
        model_needed_inputs.paddings = tf.zeros_like(model_needed_inputs.ids)
        model_needed_inputs.segment_ids = tf.ones_like(model_needed_inputs.ids)
        pos = tf.range(seq_len - 1)
        model_needed_inputs.segment_pos = model_needed_inputs.segment_ids * pos
        return model_needed_inputs

    def _load_file_dataset(self, fname):
        tf.random.set_seed(self.train_seed)
        ds = tf.data.Dataset.from_tensor_slices(fname)
        ds = ds.apply(tf.data.TFRecordDataset)
        # shard host data
        process_index = jax.process_index()
        # logging.info(f"num_infeed_hosts: {self.num_infeed_hosts} || process_index: {process_index}")  # XD fix
        ds = ds.shard(self.num_infeed_hosts, process_index)
        ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle_buffer_size is not None:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)
        padded_shapes = {key: self.seq_len for key in self.task_features}
        padding_values = {key: self.pad_id for key in self.task_features}
        ds = ds.padded_batch(
            batch_size=np.prod(self.batch_size),
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=True,
        )
        ds = ds.map(self.convert)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        if self.step_in_file: ds = ds.skip(self.step_in_file)  # XD fix
        return ds

    def load_tfrecord_dataset(self, fnames):
        tf.random.set_seed(self.train_seed)
        assert isinstance(fnames, list)
        repeat_fnames = fnames * self.repeat
        N = math.ceil(len(repeat_fnames) / self.iter_file_nums)
        file_in_data = self.meta_dict["file_in_data"]
        logging.info(f'file_in_data: {file_in_data} N: {N}')
        for n in range(file_in_data, N, 1):
            fname = repeat_fnames[n * self.iter_file_nums : (n + 1) * self.iter_file_nums]
            self.meta_dict["cur_files"] = fname
            ds = self._load_file_dataset(fname)
            ds = ds.as_numpy_iterator()
            for batch in ds:
                # self.meta_dict["step_in_file"] += 1  # XD fix
                self.step_in_file += 1
                yield batch
            self.meta_dict["file_in_data"] += 1
            # self.meta_dict["step_in_file"] = 0  # XD fix
            self.step_in_file = 0
