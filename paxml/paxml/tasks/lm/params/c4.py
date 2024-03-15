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
from collections import defaultdict
import os
import random
import json
from functools import partial

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
from praxis import py_utils

from paxml import checkpoint_paths

from paxml.utils import tfids_registry, c4_registry, extract_pythia_datapath, extract_qwen_datapath, extract_bc2_datapath1213_shuffled, extract_qwen_datapath1208, extract_train_skip_step, extract_sft_datapath,extract_sft_datapath2
from praxis import aqt_utils


from paxml.tasks.lm.params import global_cfg

WeightInit = base_layer.WeightInit
NestedMap = py_utils.NestedMap

GPT_SPM_PATH = global_cfg.GPT_SPM_PATH
GPT_EOS_ID = 1
GPT_VOCABULARY = t5.data.SentencePieceVocabulary(GPT_SPM_PATH)
PASS_THROUGH_VOCABULARY = t5.data.PassThroughVocabulary(size=50257)


NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit
GPT_EOS_ID = 1


@experiment_registry.register
class DataParams():
    LOAD_SEQIO_ID = False
    LOAD_SEQIO_TEXT = True
    TRAINING_NUM_BATCHES_TO_SKIP = None
    TEST_RATIO = 0.02
    SHUFFLE = {"train": True, "test": False}
    SHUFFLE_SIZE = {"train": 100000, "test": 10000}
    MAX_SEQ_LEN = 2048
    # default
    KEY_MAP = {"inputs": None, "targets": "text"}
    VOCAB_FILE = 'gs://common_datasets/vocab/c4_en_301_5Mexp_spm.model'
    VOCABULARY = t5.data.SentencePieceVocabulary(VOCAB_FILE)
    DATA_PATH = {
                'train': 'gs://common_datasets/', 
                'test':  'gs://common_datasets/', 
                }
    DATA_FUNC = c4_registry
    TASK_NAME = 'DataParams'


class C4UnsupervisedDataset(base_experiment.BaseExperiment):
    """Used for training Baseline ULM."""

    PERCORE_BATCH_SIZE = 1
    PERCORE_EVAL_BATCH_SIZE = None
    MAX_SEQ_LEN = 1024
    TRAINING_SEED = 9876
    TRAINING_NUM_BATCHES_TO_SKIP = None

    def _dataset_common(
        self, is_training: bool, job_log_dir=None
    ) -> pax_fiddle.Config[base_input.BaseInput]:
        LOAD_SEQIO_ID = getattr(self, 'LOAD_SEQIO_ID', False)
        LOAD_SEQIO_TEXT = getattr(self, 'LOAD_SEQIO_TEXT', True)
        # if not LOAD_SEQIO_ID and not LOAD_SEQIO_TEXT: 
        meta_dict = extract_train_skip_step(job_log_dir=job_log_dir, step=self.TRAINING_NUM_BATCHES_TO_SKIP, only_eval=getattr(self, 'ONLY_EVAL', False),)
        num_batches_to_skip = meta_dict.get('checkpoint_step', self.TRAINING_NUM_BATCHES_TO_SKIP)

        if is_training:
            percore_batch_size = self.PERCORE_BATCH_SIZE
        else:
            if self.PERCORE_EVAL_BATCH_SIZE is not None:
                percore_batch_size = self.PERCORE_EVAL_BATCH_SIZE
            else:
                percore_batch_size = self.PERCORE_BATCH_SIZE

        num_local_devices = jax.local_device_count()  # 8
        # lsp: global_batch_size: percore_batch_size * 8 * N
        global_batch_size = int(percore_batch_size * num_local_devices * jax.process_count() + 1e-6)
        if percore_batch_size >= 1:
            assert global_batch_size % num_local_devices == 0
            batch_size_per_process = int(math.ceil(percore_batch_size) * num_local_devices + 1e-6)
            num_infeed_hosts = global_batch_size // batch_size_per_process
        else:
            if jax.process_count() > 1:
                # assert global_batch_size % num_local_devices == 0  # XD: bug?
                # batch_size_per_process = num_local_devices  # XD: bug?
                batch_size_per_process = int(percore_batch_size * num_local_devices + 1e-6)
                # N hosts
                num_infeed_hosts = global_batch_size // batch_size_per_process
            else:
                batch_size_per_process = int(percore_batch_size * num_local_devices + 1e-6)
                num_infeed_hosts = 1
        # batch_size_per_process, num_infeed_hosts = 4, 2  # XD
        seed = None
        if is_training:
            seed = self.TRAINING_SEED
            # TODO(sgpyc): enable sync of seeds across hosts, currently the
            # following failed because of "sync_global_devices name mismatch"
            # seed = jnp.int32(multihost_utils.broadcast_one_to_all(seed))
            logging.info("Train input seed: %s", "None" if seed is None else seed)

        if self.LOAD_SEQIO_ID:
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

        elif self.LOAD_SEQIO_TEXT:
            logging.info(f"Load seqio text data......")
            DataFeature = seqio_input.LanguageModelFeatures
            mixture_name = "c4.train" if is_training else "c4.test"
            name = "C4Train" if is_training else "C4Validation"
            split_name = "train" if is_training else "validation"
            task_feature_lengths = {"targets": self.MAX_SEQ_LEN}
            shuffle = None

        else:
            logging.info(f"Load mesh id data......")
            shuffle_key = "train" if is_training else "test"
            shuffle_buffer_size = self.SHUFFLE_SIZE[shuffle_key] if self.SHUFFLE[shuffle_key] else None

        print(f'is_training: {is_training}')
        DATA_PATH = self.DATA_FUNC(mode='train' if is_training else 'test')
        if self.LOAD_SEQIO_ID or self.LOAD_SEQIO_TEXT:
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
                batch_size=int(batch_size_per_process),  # lsp
                drop_remainder=True if is_training else False,
                num_batches_to_skip=num_batches_to_skip,  # lsp: add skip batch step
                # num_infeed_hosts=num_infeed_hosts,
                num_infeed_hosts=num_infeed_hosts,

                # reset_for_eval=False if is_training else True, # eval的时候为True
                reset_for_eval=getattr(self, 'RESET_FOR_EVAL', False),  # eval的时候为True -> False
                annotate_padding_fields=True,
                eval_loop_num_batches=self.EVAL_LOOP_NUM_BATCHES,
            )
            return p
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
                repeat=getattr(self, 'DATA_REPEAT', {'train': 1})['train'] if is_training else getattr(self, 'DATA_REPEAT', {'test': 1})['test'],
                eval_loop_num_batches=self.EVAL_LOOP_NUM_BATCHES,
                train_seed=self.TRAINING_SEED,
                task_features=list(self.KEY_MAP.values()),
                shuffle_buffer_size=shuffle_buffer_size,
                num_batches_to_skip=num_batches_to_skip,
                only_eval=getattr(self, 'ONLY_EVAL', False),
            )
            return p  

    # lsp: 数据
    def datasets(self, job_log_dir=None) -> List[pax_fiddle.Config[base_input.BaseInput]]:
        """Returns a list of dataset parameters."""
        # if not (hasattr(self, 'train_datasets') and hasattr(self, 'eval_datasets')):
        self.train_datasets = self._dataset_common(is_training=True, job_log_dir=job_log_dir)
        self.eval_datasets = self._dataset_common(is_training=False)
        return [self.train_datasets, self.eval_datasets]
        

def set_adam_and_learning_rate_schedule(
    cls,
    task_p: pax_fiddle.Config[tasks_lib.SingleTask],
) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Sets the Adam optimizer and the learning rate schedule."""
    lp = task_p.train.learner
    lp.loss_name = "total_loss"

    lp.optimizer = pax_fiddle.Config(
        optimizers.Adam,
        beta1=cls.ADAM_BETA1 if cls.ADAM_BETA1 else 0.9,
        beta2=cls.ADAM_BETA2 if cls.ADAM_BETA2 else 0.999,
        # weight_decay=cls.WEIGHT_DECAY if cls.WEIGHT_DECAY else 0.0, # lsp: DEPRECATION
        # l2_regularizer_weight=cls.WEIGHT_DECAY if cls.WEIGHT_DECAY else 0.0,
        decoupled_weight_decay=cls.WEIGHT_DECAY if cls.WEIGHT_DECAY else 0.0,
        epsilon=cls.ADAM_EPSILON if cls.ADAM_EPSILON else 1e-6,
        epsilon_root=cls.ADAM_EPSILON_ROOT if cls.ADAM_EPSILON_ROOT else 0.0,
        clip_gradient_norm_to_value=cls.CLIP_GRADIENT_NORM_TO_VALUE  # lsp: 1.0
        if cls.CLIP_GRADIENT_NORM_TO_VALUE
        else 5.0,
        clip_threshold=cls.ADAM_CLIP_THRESHOLD if cls.ADAM_CLIP_THRESHOLD else 1.0,
        sharded_adam=True,
    )
    # # lsp
    # num_sub_batches = 3
    # lp.optimizer = pax_fiddle.Config(
    #     optimizers.ShardedStaticAccumulator,
    #     optimizer_tpl=lp.optimizer,
    #     num_sub_batches=num_sub_batches,
    # )

    if hasattr(cls, "PERCORE_BATCH_SIZE"):
        global_batch_size = int(cls.PERCORE_BATCH_SIZE * jax.device_count() + 1e-6)
        if global_batch_size == 0:
            logging.warning(
                "Found global_batch_size = 0: cls.PERCORE_BATCH_SIZE=%s, jax.device_count()=%s",
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

    if cls.LR_SCHEDULE == "linear_rampup_exponential_decay":
        lp.optimizer.lr_schedule = pax_fiddle.Config(
            schedules.LinearRampupExponentialDecay,
            warmup_steps=cls.LR_LRED_WARMUP,
            decay_start=cls.LR_LRED_DECAY_START,
            decay_end=cls.LR_LRED_DECAY_END,
            min_ratio=cls.LR_LRED_MIN_RATIO,
            max=cls.LR_LRED_MAX,
        )
    elif cls.LR_SCHEDULE == "linear_rampup_cosine_decay":
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
        raise NotImplementedError(f"Learning rate schedule {cls.LR_SCHEDULE} is not supported.")

    return task_p


# 这个类继承
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
    EMBEDDING_LOOKUP_STYLE = "matmul"

    # optimizer related
    LEARNING_RATE = 2e-4  # XD: 1e-3
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.95  # XD: 0.99
    ADAM_CLIP_THRESHOLD = 1.0
    ADAM_EPSILON = 1e-8  # XD: -6
    ADAM_EPSILON_ROOT = 0.0

    # Learning rate schedule
    LR_SCHEDULE = "linear_rampup_cosine_decay"  # XD: exponential
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
        # pytype: disable=attribute-error  # enable-nested-classes
        model_p.lm_tpl.packed_input = self.PACKED_INPUT
        # stacked_p: StackedTransformerRepeated
        # pytype: disable=attribute-error  # enable-nested-classes
        stacked_p = model_p.lm_tpl.stacked_transformer_tpl
        if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
            stacked_p = stacked_p.pipeline_stage
        if self.USE_REPEATED_LAYER:
            # stacked_p.block: StackedTransformer
            stacked_p = stacked_p.block
        transformer_layer_p = stacked_p.transformer_layer_params_tpl
        # lsp： 在这里设置的attn bias
        # qkv bias
        transformer_layer_p.tr_atten_tpl.use_bias = getattr(self, 'QKV_BIAS', False)
        # post bias | mlp bias
        transformer_layer_p.tr_atten_tpl.o_bias = getattr(self, 'O_BIAS', False)
        # lsp: 设置优化器
        task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)
        return task_p


class TransformerLmSpmdPipelineAdam(model_params.TransformerLmSpmdPipelineAdafactor):
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
    EMBEDDING_LOOKUP_STYLE = "matmul"

    # optimizer related
    LEARNING_RATE = 1e-3
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = 0.99
    ADAM_CLIP_THRESHOLD = 1.0
    ADAM_EPSILON = 1e-6
    ADAM_EPSILON_ROOT = 0.0

    # Learning rate schedule
    LR_SCHEDULE = "linear_rampup_exponential_decay"
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
        # pytype: disable=attribute-error  # enable-nested-classes
        model_p.lm_tpl.packed_input = self.PACKED_INPUT

        # pytype: disable=attribute-error  # enable-nested-classes
        stacked_p = model_p.lm_tpl.stacked_transformer_tpl
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
        if "eval_test_C4Validation/metrics/log_pplx" not in metrics.keys():
            return False
        log_pplx = metrics["eval_test_C4Validation/metrics/log_pplx"]

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

    # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.eos_id = GPT_EOS_ID
    # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.seqlen = cls.MAX_SEQ_LEN
    # lsp: 每个参数都是WeightHParams类，该类有个init函数，调用params_init。没有指定的默认WeightInit.Xavier(_DEFAULT_XAVIER_INIT)， _DEFAULT_XAVIER_INIT： 1.000001
    model_p.params_init = WeightInit.Gaussian(0.006)

    softmax_init = WeightInit.Gaussian(0.006)
    # lsp: lm_tpl: TransformerLM
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init
    model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False
    model_p.lm_tpl.softmax_tpl.soft_cap_logits = None
    # lsp: True
    if cls.SEPARATE_EMBEDDING:
        # scale_sqrt_depth is true时，会对embedding进行scale
        model_p.lm_tpl.separate_embedding_tpl.scale_sqrt_depth = False
        # matmul
        model_p.lm_tpl.separate_embedding_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE
        
    else:
        model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = False
        model_p.lm_tpl.softmax_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE  # matmul
    if cls.TRAINABLE_POSITION_EMB:
        model_p.lm_tpl.position_emb_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE
    # lsp: 增加embed drop ratio
    model_p.lm_tpl.embed_dropout_prob = cls.EMBED_DROPOUT_PROB

    # lsp: 设置transformer的属性
    stacked_p = model_p.lm_tpl.stacked_transformer_tpl
    if fdl.get_callable(stacked_p) == transformers.PipelinedTransformer:
        stacked_p = stacked_p.pipeline_stage
    # issubclass： 判断stacked_p是否继承了
    # fdl.get_callable(stacked_p)其实就是stacked_p未被pax_fiddle.Config(stacked_p)之前的类
    # lsp: stacked_p 继承 StackedTransformerRepeated为True
    if issubclass(fdl.get_callable(stacked_p), transformers.StackedTransformerRepeated):
        # 如果stacked_p继承了StackedTransformerRepeated，那么stacked_p.block应该是StackedTransformerRepeated
        # 然后去设置block的属性
        stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    # lsp: layer_norm
    transformer_layer_p.ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
    transformer_layer_p.tr_fflayer_tpl.ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
    model_p.lm_tpl.final_ln_tpl = pax_fiddle.Config(cls.NORMALIZATION_CLS)  # XD add
    if cls.NORMALIZATION_CLS == normalizations.RmsNorm:  # XD
        # transformer_layer_p.ln_tpl.intermediate_dtype = jnp.float32 # lsp：inputs采用float32
        # transformer_layer_p.tr_fflayer_tpl.ln_tpl.intermediate_dtype = jnp.float32
        # model_p.lm_tpl.final_ln_tpl.intermediate_dtype = jnp.float32
        transformer_layer_p.ln_tpl.intermediate_dtype = jnp.bfloat16  # lsp：inputs采用float32
        transformer_layer_p.tr_fflayer_tpl.ln_tpl.intermediate_dtype = jnp.bfloat16
        model_p.lm_tpl.final_ln_tpl.intermediate_dtype = jnp.bfloat16

    if True or cls.NORMALIZATION_CLS == normalizations.LayerNorm:  # XD
        transformer_layer_p.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
        transformer_layer_p.tr_fflayer_tpl.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
        model_p.lm_tpl.final_ln_tpl.epsilon = cls.LAYERNORM_EPSILON

    transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
    # lsp 优先级最高：atten qkv bias
    transformer_layer_p.tr_atten_tpl.use_bias = getattr(cls, 'QKV_BIAS', False)
    transformer_layer_p.tr_atten_tpl.o_bias = getattr(cls, 'O_BIAS', False)

    # lsp:
    # transformer_layer_p.tr_atten_tpl.atten_dropout_prob = cls.ATTEN_DROPOUT_PROB # 会被transformer_layer_p的atten_dropout_prob覆盖
    # transformer_layer_p.atten_dropout_prob = cls.ATTEN_DROPOUT_PROB # 会被stacked_p的atten_dropout_prob覆盖
    stacked_p.atten_dropout_prob = cls.ATTEN_DROPOUT_PROB

    transformer_layer_p.tr_fflayer_tpl.has_bias = not cls.USE_GATED_ACTIVATION or cls.USE_BIAS  # XD add
    if cls.ACTIVATION_CLS == layers.GELU:
        transformer_layer_p.tr_fflayer_tpl.activation_tpl.approximate = True  # XD: add if
    
    # lsp
    transformer_layer_p.tr_fflayer_tpl.chunk_size = getattr(cls, 'FFN_CHUNK_SIZE', None)  # XD: add if
    # lsp: default True
    model_p.data_full_shard = getattr(cls, 'DATA_FULL_SHARD', True)

    for atten_p in (
        transformer_layer_p.tr_atten_tpl,
        transformer_layer_p.cross_atten_tpl,
    ):
        if atten_p is None:
            continue
        atten_wp = atten_p.weight_split_dims_mapping
        atten_wp.proj = ["data", "mdl", None]

    if task_p.early_stopping_fn is None:
        # lsp: EarlyStoppingFn: 当ppl变大的时候则停止
        task_p.early_stopping_fn = pax_fiddle.Config(EarlyStoppingFn)
        task_p.early_stopping_fn.target_log_pplx = cls.TARGET_LOG_PPLX

    if hasattr(cls, 'MGATE'):  # lsp
      transformer_layer_p.tr_fflayer_tpl.mgate = cls.MGATE
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
    if hasattr(cls, 'MOE_NUM_GROUPS'):  # lsp
      stacked_p.num_groups = cls.MOE_NUM_GROUPS
    if hasattr(cls, 'GATING_FUNC'):  # lsp
      stacked_p.gating_func = cls.GATING_FUNC
    if hasattr(cls, 'MIN_GROUP_SIZE'):  # lsp
      stacked_p.min_group_size = cls.MIN_GROUP_SIZE
    if hasattr(cls, 'EXPERT_CHUNK_SIZE'):  # lsp
      stacked_p.expert_chunk_size = cls.EXPERT_CHUNK_SIZE
    if hasattr(cls, 'ROUTER_Z_LOSS'):  # lsp
      stacked_p.router_z_loss = cls.ROUTER_Z_LOSS

    if hasattr(cls, 'QUANT'):  # lsp
      logging.info(f'quant is True')
      aqt_config = aqt_utils.AqtCfg(quantization=cls.QUANT)
      logging.info(f'aqt_config: {aqt_config}')
      quant_config = aqt_utils.configure_quantization(aqt_config)
      logging.info(f'quant_config: {quant_config}')
      transformer_layer_p.quant = quant_config # 可有可无
      # ffn aqt
      transformer_layer_p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.quant = quant_config 
      # atten aqt，设置后变慢了一点 qkv weight 
      transformer_layer_p.tr_atten_tpl.proj_tpl.quant = quant_config
      # qkv activation 快一点点
    #   transformer_layer_p.tr_atten_tpl.quant = quant_config
      # embedding 快一点点
      model_p.lm_tpl.separate_embedding_tpl.quant = quant_config
      # lm head 变慢了一点点
    #   model_p.lm_tpl.softmax_tpl.feed_forward_tpl.linear_tpl.quant = quant_config

    return task_p


@experiment_registry.register
class C4SpmdAdam(TransformerLmSpmdAdam, C4UnsupervisedDataset):
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
        # pytype: disable=attribute-error  # enable-nested-classes
        model_p.decoder_tpl.eos_id = GPT_EOS_ID
        # pytype: disable=attribute-error  # enable-nested-classes
        model_p.decoder_tpl.seqlen = self.MAX_SEQ_LEN

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
    LAYERNORM_EPSILON = 1e-6

    # In units of steps for BS1.5k
    LR_SCHEDULE = "linear_rampup_cosine_decay"
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
    r"""small GPT-3 config with RoPE."""
    VOCAB_SIZE = 32000  # XD
    NUM_LAYERS = 12
    MODEL_DIMS = 768
    ACTIVATION_CLS = layers.SiLU  # layers.SiLU/GELU  # XD
    USE_GATED_ACTIVATION = True  # XD
    HIDDEN_DIMS = MODEL_DIMS * 4  # 2048  # XD: MODEL_DIMS * 4
    NUM_HEADS = 12
    # Defaults to MODEL_DIMS // NUM_HEADS.
    DIMS_PER_HEAD = None
    USE_BIAS = False  # XD add
    # NORMALIZATION_CLS = normalizations.LayerNorm  # XD add RmsNorm
    NORMALIZATION_CLS = normalizations.RmsNorm  # XD add RmsNorm

    LEARNING_RATE = 2e-4  # XD
    PERCORE_BATCH_SIZE = 4
    # FPROP_DTYPE = jnp.bfloat16
    FPROP_DTYPE = jnp.bfloat16

    ICI_MESH_SHAPE = [1, 8, 4]

    SEPARATE_EMBEDDING = True  # XD
    USE_ROTARY_POSITION_EMB = True
    TRAINABLE_PE_MAX_SEQ_LEN = 2048  # XD add
    # lsp: 调用这个task

    def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
        task_p = super().task()
        # lsp: 位置向量在这里设置为None了
        if self.USE_ALIBI_POSITION_EMB or self.USE_ROTARY_POSITION_EMB:
            task_p.model.lm_tpl.position_emb_tpl = None  # XD: add if
        return task_p


@experiment_registry.register
class C4SpmdGpt37BRoPE(C4SpmdGpt3SmallRoPE):  # XD
    NUM_LAYERS = 32
    MODEL_DIMS = 4096
    HIDDEN_DIMS = 11008  # XD: MODEL_DIMS * 4 * 2 // 3
    # HIDDEN_DIMS = 13696  # XD: MODEL_DIMS * 4 * 2 // 3
    NUM_HEADS = 32
    # DIMS_PER_HEAD = 128
    COMBINE_QKV = False  # False 占用显存小于 True 1G+
    NUM_GROUPS = -1
    NUM_TRAIN_STEPS = 1e7
    # ICI_MESH_SHAPE = [1, 4, 2]  # bs=1*8, 0.315 paxml 0.273 mesh
    # ICI_MESH_SHAPE = [1, 8, 1]  # bs=1*8, 0.311 paxml 0.272 mesh
    # ICI_MESH_SHAPE = [4, 1, 8]  # bs=2*8, 0.146, combine_qkv 0.1514
    # ICI_MESH_SHAPE = [1, 8, 4]  # bs=8*8, 0.176, combine_qkv 0.180
    # ICI_MESH_SHAPE = [1, 16, 1] # 16 * 1 * 16 * 1 oom: 30M, combine_qkv: False
    # ICI_MESH_SHAPE = [1, 16, 1] # 8 * 1 * 16 * 1 combine_qkv: True, 0.138 * 2
    # ICI_MESH_SHAPE = [1, 16, 1] # 16 * 1 * 16 * 1 combine_qkv: True,
    # ICI_MESH_SHAPE = [4, 1, 8]  # bs=2*8*4, 0.146, combine_qkv 0.1514
    # ICI_MESH_SHAPE = [1, 8, 4]  # seq: 4096 bs=1*8*4, 0.0692 paxml
    # ICI_MESH_SHAPE = [1, 32, 4]  # seq: 4096 bs=2*32*4, 0.0349 paxml

    PERCORE_BATCH_SIZE = 1
    ICI_MESH_SHAPE = [1, 8, 4]
    DCN_MESH_SHAPE = [1, 1, 1]  # lsp： [2, 1, 1] 表示2个node，但是会报错，不知道啥情况

    MAX_SEQ_LEN = 4096 * 2
    VOCAB_SIZE = 64000
    # VOCAB_SIZE = 125696

    LAYERNORM_EPSILON = 1e-06
    # Learning rate schedule
    LEARNING_RATE = 8e-6
    LR_SCHEDULE = "linear_rampup_cosine_decay"
    # 最大学习率 * LR_LRED_MIN_RATIO： 最后保持稳定的学习率,即step > LR_COS_DECAY_END时的学习率
    LR_COS_MIN_RATIO = 0.1
    LR_COS_MAX = 1.0  # 这是cos曲线的最大值，和pytorch的cos曲线的学习率不是一个值，这个值 * LEARNING_RATE就是pytorch设定的值
    # warmup step: 学习率从 0 -> LR_COS_MAX的步数, easyl: ratio, 0.02 * LR_COS_DECAY_END = 1170
    LR_COS_WARMUP = int(58497 * 0.02 * 1)
    LR_COS_DECAY_START = LR_COS_WARMUP + 1  # decay start step: 学习率开始衰减的步数
    LR_COS_DECAY_END = int(19499 * 1)  # decay end step # 学习率最后保持恒定的步数
    WEIGHT_DECAY = 0.001
    ADAM_BETA2 = 0.999
    ADAM_BETA1 = 0.9
    ADAM_EPSILON = 1e-8
    CLIP_GRADIENT_NORM_TO_VALUE = 1.0

    TRAINING_NUM_BATCHES_TO_SKIP = None

    EMBED_DROPOUT_PROB = 0.0
    ATTEN_DROPOUT_PROB = 0.0
    TRAINABLE_POSITION_EMB = False

    CHECKPOINT_EVERY_N_STEPS = 100
    EVAL_LOOP_NUM_BATCHES = 25  # 每次评测多少batch
    EVAL_INTERVAL_STEPS = 100  # 每隔多少step评测一次
    CHECKPOINT_MAX_TO_KEEP = 2  # 保留n个checkpoint

    WANDB_PROJECT = "debug"

    TRAINING_SEED = 1234
    USE_ROTARY_POSITION_EMB = True
    USE_ALIBI_POSITION_EMB = False

    LOAD_SEQIO_ID = False
    LOAD_SEQIO_TEXT = True
    # eval loss小于等于这个值会自动停止，paxml默认2.69，设置-1让它一直训练
    TARGET_LOG_PPLX = -1
    SAVE_ON_STEPS = list(range(2000, 1000000, 2000))


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
    EMB_W_DATA_DIMS = ("replica", "data")

    def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
        """Returns the task parameters."""
        task_p = super().task()
        model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
        # pytype: disable=attribute-error  # enable-nested-classes
        model_p.decoder_tpl.eos_id = GPT_EOS_ID
        # pytype: disable=attribute-error  # enable-nested-classes
        model_p.decoder_tpl.seqlen = self.MAX_SEQ_LEN

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
    LAYERNORM_EPSILON = 1e-6

    # In units of steps for BS1.5k
    LR_SCHEDULE = "linear_rampup_cosine_decay"
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
class C4SpmdPipelineGpt3AdamMLPerfHPBS1p5k768Replicas(C4SpmdPipelineGpt3AdamMLPerfHP):
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
class C4SpmdPipelineGpt3AdamMLPerfHPBS2k512Replicas(C4SpmdPipelineGpt3AdamMLPerfHP):
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
class C4SpmdPipelineGpt3AdamMLPerfHPBS3k768Replicas(C4SpmdPipelineGpt3AdamMLPerfHP):
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
class C4SpmdPipelineGpt3AdamMLPerfHPBS4k1024Replicas(C4SpmdPipelineGpt3AdamMLPerfHP):
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
class C4SpmdPipelineGpt3AdamMLPerfHPBS8k1024Replicas(C4SpmdPipelineGpt3AdamMLPerfHP):
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
    PERCORE_BATCH_SIZE = 16  # // 8 # XD: v4->v3
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
class Llama7B(C4SpmdGpt37BRoPE):
    NUM_LAYERS = 32
    MODEL_DIMS = 4096
    HIDDEN_DIMS = 11008 // 2
    NUM_HEADS = 32
    # DIMS_PER_HEAD = 256
    PERCORE_BATCH_SIZE = 1
    ICI_MESH_SHAPE = [1, 8, 1]  # [1, 8, 4], bsz = 1 * 1 * 8 * 4=32， mesh_tf: 0.0686step/s
    MAX_SEQ_LEN = 8192 // 4
    VOCAB_SIZE = 50257
    DATA_FULL_SHARD = True

    LAYERNORM_EPSILON = 1e-06
    LEARNING_RATE = 1e-5
    LR_SCHEDULE = "linear_rampup_exponential_decay"  # constant_with_warmup
    LR_LRED_WARMUP = 2000
    LR_LRED_DECAY_START = 2001
    LR_LRED_DECAY_END = 200000
    LR_LRED_MIN_RATIO = 1.0
    LR_LRED_MAX = 1.0
    Z_LOSS_WEIGHT = 0.0

    ADAM_BETA2 = 0.95
    ADAM_BETA1 = 0.9
    ADAM_EPSILON = 1e-8  # baichuan2 use default 1e-8
    CLIP_GRADIENT_NORM_TO_VALUE = 1.0
    WEIGHT_DECAY = 0.005  # baichuan2 finetune: 0.005  pretrain: 0.1

    TRAINING_NUM_BATCHES_TO_SKIP = None
    TRAINABLE_POSITION_EMB = False
    USE_ROTARY_POSITION_EMB = True
    USE_ALIBI_POSITION_EMB = False
    ROTARY_TYPE = 'paxml'
    NORMALIZATION_CLS = normalizations.RmsNorm
    QKV_BIAS = True
    O_BIAS = False
    USE_BIAS = False
    FPROP_DTYPE = jnp.bfloat16

    CHECKPOINT_EVERY_N_STEPS = 200000
    EVAL_LOOP_NUM_BATCHES = 102
    EVAL_INTERVAL_STEPS = 10000
    CHECKPOINT_MAX_TO_KEEP = 2

    WANDB_PROJECT = "debug"
    LM_HEAD_NORM = False

    QUERY_CHUNK_SIZE = None
    LM_HEAD_CHUNK_SIZE = None
    FFN_CHUNK_SIZE = HIDDEN_DIMS // 1
    RESET_FOR_EVAL = False
    TASK_NAME = "Llama7B"
    TARGET_LOG_PPLX = -1
    SHUFFLE = {"train": True, "test": True}
    SHUFFLE_SIZE = {"train": 100000, "test": 10000}
    TRAINING_SEED = 1234
    TEST_RATIO = 0.02
    RESET_FOR_EVAL = False # when True, eval whole eval dataset

    LOAD_SEQIO_TEXT = True
    KEY_MAP = {"inputs": None, "targets": "text"}
    VOCABULARY = t5.data.SentencePieceVocabulary(GPT_SPM_PATH)
    DATA_PATH = {
                'train': 'gs://common_datasets/', 
                'test':  'gs://common_datasets/', 
                }
    DATA_FUNC = c4_registry

@experiment_registry.register
class Llama7BMoe(Llama7B):
  MOE_GATED_ACTIVATION = True
  NUM_EXPERTS = 8
  GATING_FUNC = 'openmoe_top2'
  NUM_LAYERS = 12
  NUM_HEADS = 32
  MOE_LAYERS = list(range(NUM_LAYERS))
  CAPACITY_FACTOR = 1.25
  HIDDEN_DIMS = 5504
  MODEL_DIMS = 1024 * 4
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 2, 8]
  MOE_NUM_GROUPS = PERCORE_BATCH_SIZE * ICI_MESH_SHAPE[1] * ICI_MESH_SHAPE[2]
  MIN_GROUP_SIZE = 10
  FFN_CHUNK_SIZE = None
  QUERY_CHUNK_SIZE = None
  EXPERT_CHUNK_SIZE = None
  ROUTER_Z_LOSS = False

@experiment_registry.register
class Llama7BDense(Llama7B):
  QUERY_CHUNK_SIZE = None
  MOE_GATED_ACTIVATION = True
  NUM_EXPERTS = 0
  NUM_LAYERS = 12
  NUM_HEADS = 32
#   MOE_LAYERS = list(range(NUM_LAYERS))
  MOE_LAYERS = []
  HIDDEN_DIMS = 5504 * 1
  MODEL_DIMS = 1024 * 4
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 2, 8]
  FFN_CHUNK_SIZE = None

@experiment_registry.register
class Llama7BMoeV4x256(Llama7B):
  MOE_GATED_ACTIVATION = True
  NUM_EXPERTS = 8
  GATING_FUNC = 'openmoe_top2'
  NUM_LAYERS = 48
  NUM_HEADS = 32
  MOE_LAYERS = list(range(NUM_LAYERS))
  CAPACITY_FACTOR = 1.25
  HIDDEN_DIMS = 5504
  MODEL_DIMS = 1024 * 4
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 128, 1]
  MOE_NUM_GROUPS = PERCORE_BATCH_SIZE * ICI_MESH_SHAPE[1]
  MIN_GROUP_SIZE = 10
  FFN_CHUNK_SIZE = None
  QUERY_CHUNK_SIZE = None
  EXPERT_CHUNK_SIZE = None

@experiment_registry.register
class Llama7BDenseV4x256(Llama7B):
  QUERY_CHUNK_SIZE = None
  MOE_GATED_ACTIVATION = True
  NUM_EXPERTS = 0
  NUM_LAYERS = 48
  NUM_HEADS = 32
#   MOE_LAYERS = list(range(NUM_LAYERS))
  MOE_LAYERS = []
  HIDDEN_DIMS = 5504 * 8
  MODEL_DIMS = 1024 * 4
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 128, 1]
  FFN_CHUNK_SIZE = None


@experiment_registry.register
class Llama7BMultiSlice(Llama7B):
    NUM_LAYERS = 48
    ICI_MESH_SHAPE = [1, 4, 2]  # [1, 8, 4], bsz = 1 * 1 * 8 * 4=32， mesh_tf: 0.0686step/s
    MAX_SEQ_LEN = 8192 // 4
    QUERY_CHUNK_SIZE = 512
    LM_HEAD_CHUNK_SIZE = 512
    DATA_FULL_SHARD = True
    PERCORE_BATCH_SIZE = 2
    FFN_CHUNK_SIZE = 5504 // 8
    DCN_MESH_SHAPE = [2, 1, 1]


@experiment_registry.register
class Llama7Bv4x16(Llama7B):
    NUM_LAYERS = 48
    ICI_MESH_SHAPE = [1, 8, 1]  # [1, 8, 4], bsz = 1 * 1 * 8 * 4=32， mesh_tf: 0.0686step/s
    MAX_SEQ_LEN = 2048
    QUERY_CHUNK_SIZE = None
    LM_HEAD_CHUNK_SIZE = None
    DATA_FULL_SHARD = False
    PERCORE_BATCH_SIZE = 1
    FFN_CHUNK_SIZE = 5504

@experiment_registry.register
class Llama7Bv5px8(Llama7B):
    NUM_LAYERS = 32
    ICI_MESH_SHAPE = [1, 4, 1]  # [1, 8, 4], bsz = 1 * 1 * 8 * 4=32， mesh_tf: 0.0686step/s
    MAX_SEQ_LEN = 2048
    QUERY_CHUNK_SIZE = None
    LM_HEAD_CHUNK_SIZE = None
    DATA_FULL_SHARD = False
    PERCORE_BATCH_SIZE = 8
    # FFN_CHUNK_SIZE = 5504 * 2
    LEARNING_RATE = 2e-4  # XD: 1e-3
    LR_LRED_WARMUP = 500
    LR_LRED_DECAY_START = 501
    LR_LRED_DECAY_END = 200000
    HIDDEN_DIMS = 11008
    QUANT = 'int8'
    DATA_PATH = {
                'train': 'gs://common_datasets_us-east5/', 
                'test':  'gs://common_datasets_us-east5/', 
                }

@experiment_registry.register
class Llama7Bv5px32(Llama7Bv5px8):
    ICI_MESH_SHAPE = [1, 16, 1]  # [1, 8, 4], bsz = 1 * 1 * 8 * 4=32， mesh_tf: 0.0686step/s


@experiment_registry.register
class Llama7Bv32(Llama7B):
    NUM_LAYERS = 48
    ICI_MESH_SHAPE = [1, 16, 2]  # [1, 8, 4], bsz = 1 * 1 * 8 * 4=32， mesh_tf: 0.0686step/s
    MAX_SEQ_LEN = 8192 * 4
    QUERY_CHUNK_SIZE = 512
    LM_HEAD_CHUNK_SIZE = 512
    DATA_FULL_SHARD = False
    PERCORE_BATCH_SIZE = 0.5
    FFN_CHUNK_SIZE = 5504 // 8


@experiment_registry.register
class Llama7Bv64(Llama7B):
    NUM_LAYERS = 48
    ICI_MESH_SHAPE = [1, 64, 1]  # [1, 8, 4], bsz = 1 * 1 * 8 * 4=32， mesh_tf: 0.0686step/s
    MAX_SEQ_LEN = 8192 * 2
    QUERY_CHUNK_SIZE = 1024
    LM_HEAD_CHUNK_SIZE = 1024

@experiment_registry.register
class BC2Gpt13B(C4SpmdGpt37BRoPE):
    NUM_LAYERS = 40
    MODEL_DIMS = 5120
    HIDDEN_DIMS = 13696
    NUM_HEADS = 40
    PERCORE_BATCH_SIZE = 2
    ICI_MESH_SHAPE = [1, 32, 4]  # [1, 8, 4], bsz = 1 * 1 * 8 * 4=32， mesh_tf: 0.0686step/s
    MAX_SEQ_LEN = 4097
    VOCAB_SIZE = 125696

    LAYERNORM_EPSILON = 1e-06
    LEARNING_RATE = 1e-5
    LR_SCHEDULE = "linear_rampup_exponential_decay"  # constant_with_warmup
    LR_LRED_WARMUP = 2000
    LR_LRED_DECAY_START = 2001
    LR_LRED_DECAY_END = 200000
    LR_LRED_MIN_RATIO = 1.0
    LR_LRED_MAX = 1.0
    Z_LOSS_WEIGHT = 0.0

    # LR_SCHEDULE = "linear_rampup_cosine_decay"
    # # 最大学习率 * LR_LRED_MIN_RATIO： 最后保持稳定的学习率,即step > LR_COS_DECAY_END时的学习率
    # LR_COS_MIN_RATIO = 0.1
    # LR_COS_MAX = 1.0  # 这是cos曲线的最大值，和pytorch的cos曲线的学习率不是一个值，这个值 * LEARNING_RATE就是pytorch设定的值
    # # warmup step: 学习率从 0 -> LR_COS_MAX的步数, easyl: ratio, 0.02 * LR_COS_DECAY_END = 1170
    # LR_COS_WARMUP = 200
    # LR_COS_DECAY_START = LR_COS_WARMUP + 1  # decay start step: 学习率开始衰减的步数
    # LR_COS_DECAY_END = 10000  # decay end step # 学习率最后保持恒定的步数

    ADAM_BETA2 = 0.95
    ADAM_BETA1 = 0.9
    ADAM_EPSILON = 1e-8  # baichuan2 use default 1e-8
    CLIP_GRADIENT_NORM_TO_VALUE = 1.0
    WEIGHT_DECAY = 0.1  # baichuan2 finetune: 0.005  pretrain: 0.1

    TRAINING_NUM_BATCHES_TO_SKIP = None
    TRAINABLE_POSITION_EMB = False
    USE_ROTARY_POSITION_EMB = False
    USE_ALIBI_POSITION_EMB = True
    ROTARY_TYPE = 'paxml'

    CHECKPOINT_EVERY_N_STEPS = 200
    EVAL_LOOP_NUM_BATCHES = 20
    EVAL_INTERVAL_STEPS = 100
    CHECKPOINT_MAX_TO_KEEP = 3

    WANDB_PROJECT = "BC213B"
    LM_HEAD_NORM = True
    LOAD_SEQIO_ID = False
    LOAD_SEQIO_TEXT = False

    QUERY_CHUNK_SIZE = 512
    LM_HEAD_CHUNK_SIZE = 512
    RESET_FOR_EVAL = False
    TASK_NAME = "BC2Gpt13B"
    TARGET_LOG_PPLX = -1
    SHUFFLE = {"train": True, "test": True}
    SHUFFLE_SIZE = {"train": 100000, "test": 10000}

    TRAINING_SEED = 1234
    TEST_RATIO = 0.02
    RESET_FOR_EVAL = False # when True, eval whole eval dataset
    # # novel xiaomeng zh en
    # LOAD_SEQIO_TEXT = False
    # VOCABULARY = t5.data.PassThroughVocabulary(size=VOCAB_SIZE)
    # KEY_MAP = {"targets": "input_ids", "masks": "input_ids"}
    # SPLIT_BSZ = {"zh": 7, "en": 20}  # 7表示这本书取了前7次
    # DATA_FUNC = extract_zh_en_novel_datapath

    # # c4 text datasets. when LOAD_SEQIO_TEXT is True ，recovery code
    # LOAD_SEQIO_TEXT = True
    # KEY_MAP = {"inputs": None, "targets": "text"}
    # VOCAB_FILE = "gs://llm_base_models/baichuan2-13b-hf/tokenizer.model"
    # VOCABULARY = t5.data.SentencePieceVocabulary(VOCAB_FILE)
    # DATA_PATH = {
    #             'train': 'gs://common_datasets/', 
    #             'test':  'gs://common_datasets/', 
    #             }
    # DATA_FUNC = c4_registry

    # baichuan1指令数据集
    # LOAD_SEQIO_ID = True
    # LOAD_SEQIO_TEXT = False
    # KEY_MAP = {"targets": "input_ids", "masks": "input_ids"}
    # VOCABULARY = t5.data.PassThroughVocabulary(size=VOCAB_SIZE)
    # DATA_PATH = {
    #     "train": ["gs://jax_llm_data/data-baichuan/dreamily_translation_general.train.tfrecords"],
    #     "test": ["gs://jax_llm_data/data-baichuan/dreamily_translation_general.test.tfrecords"],
    # }
    # DATA_FUNC = tfids_registry
    DATA_REPEAT = {'train': 1, 'test': 10}
    LOAD_SEQIO_TEXT = False
    LOAD_SEQIO_ID = False
    VOCABULARY = t5.data.PassThroughVocabulary(size=VOCAB_SIZE)
    KEY_MAP = {"targets": "input_ids", "masks": "labels"}
    DATA_PATH = {
                'train': ['gs://jax_llm_data/xiaomeng/zh_data_Baichuan2-13B-Base_1213',
                          'gs://jax_llm_data/xiaomeng/en_data_Baichuan2-13B-Base_1213'], 
                 'test': ['gs://jax_llm_data/xiaomeng/zh_data_Baichuan2-13B-Base_1213',
                          'gs://jax_llm_data/xiaomeng/en_data_Baichuan2-13B-Base_1213'], 
                }
    # DATA_FUNC = extract_bc2_datapath1213
    DATA_FUNC = extract_bc2_datapath1213_shuffled


@experiment_registry.register
class Qwen7B(C4SpmdGpt37BRoPE):
    NUM_LAYERS = 32
    MODEL_DIMS = 4096
    HIDDEN_DIMS = 11008
    NUM_HEADS = 32
    PERCORE_BATCH_SIZE = 1
    ICI_MESH_SHAPE = [1, 64, 1]  # [1, 8, 4], bsz = 1 * 1 * 8 * 4=32， mesh_tf: 0.0686step/s
    MAX_SEQ_LEN = 4097
    VOCAB_SIZE = 151936

    LAYERNORM_EPSILON = 1e-06
    LEARNING_RATE = 1e-5
    LR_SCHEDULE = "linear_rampup_exponential_decay"  # constant_with_warmup
    LR_LRED_WARMUP = 2000
    LR_LRED_DECAY_START = 2001
    LR_LRED_DECAY_END = 200000
    LR_LRED_MIN_RATIO = 1.0
    LR_LRED_MAX = 1.0
    Z_LOSS_WEIGHT = 0.0

    ADAM_BETA2 = 0.95
    ADAM_BETA1 = 0.9
    ADAM_EPSILON = 1e-8  # baichuan2 use default 1e-8
    CLIP_GRADIENT_NORM_TO_VALUE = 1.0
    WEIGHT_DECAY = 0.005  # baichuan2 finetune: 0.005  pretrain: 0.1

    TRAINING_NUM_BATCHES_TO_SKIP = None
    TRAINABLE_POSITION_EMB = False
    USE_ROTARY_POSITION_EMB = True
    USE_ALIBI_POSITION_EMB = False
    ROTARY_TYPE = 'paxml'
    NORMALIZATION_CLS = normalizations.RmsNorm
    QKV_BIAS = True
    O_BIAS = False
    USE_BIAS = False
    FPROP_DTYPE = jnp.bfloat16

    CHECKPOINT_EVERY_N_STEPS = 20
    EVAL_LOOP_NUM_BATCHES = 102
    EVAL_INTERVAL_STEPS = 100
    CHECKPOINT_MAX_TO_KEEP = 2

    WANDB_PROJECT = "debug"
    LM_HEAD_NORM = False

    QUERY_CHUNK_SIZE = 128
    LM_HEAD_CHUNK_SIZE = 512
    RESET_FOR_EVAL = False
    TASK_NAME = "Qwen7B"
    TARGET_LOG_PPLX = -1
    SHUFFLE = {"train": True, "test": True}
    SHUFFLE_SIZE = {"train": 100000, "test": 10000}

    TRAINING_SEED = 1234
    TEST_RATIO = 0.02
    RESET_FOR_EVAL = False # when True, eval whole eval dataset

    # c4 text datasets. when LOAD_SEQIO_TEXT is True ，recovery code
    LOAD_SEQIO_TEXT = False
    LOAD_SEQIO_ID = False
    KEY_MAP = {"targets": "input_ids", "masks": "labels"}
    DATA_PATH = {
                'train': ['gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/', 
                          'gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117'], 
                'test':  ['gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/', 
                          'gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117']
                }
    DATA_FUNC = extract_qwen_datapath


@experiment_registry.register
class Qwen14B(C4SpmdGpt37BRoPE):
    TASK_NAME = "Qwen14B"
    NUM_LAYERS = 40
    MODEL_DIMS = 5120
    HIDDEN_DIMS = 13696
    NUM_HEADS = 40
    PERCORE_BATCH_SIZE = 16
    ICI_MESH_SHAPE = [1, 16, 1]  # [1, 8, 4], bsz = 1 * 1 * 8 * 4=32， mesh_tf: 0.0686step/s
    # MAX_SEQ_LEN = 4097
    MAX_SEQ_LEN = 2049
    VOCAB_SIZE = 152064

    LAYERNORM_EPSILON = 1e-06
    LEARNING_RATE = 1e-5
    LR_SCHEDULE = "linear_rampup_exponential_decay"  # constant_with_warmup
    LR_LRED_WARMUP = 200
    LR_LRED_DECAY_START = 201
    LR_LRED_DECAY_END = 2000
    LR_LRED_MIN_RATIO = 1.0
    LR_LRED_MAX = 1.0
    Z_LOSS_WEIGHT = 0.0

    ADAM_BETA2 = 0.95
    ADAM_BETA1 = 0.9
    ADAM_EPSILON = 1e-8  # baichuan2 use default 1e-8
    CLIP_GRADIENT_NORM_TO_VALUE = 1.0
    WEIGHT_DECAY = 0.01  # baichuan2 finetune: 0.005  pretrain: 0.1

    TRAINING_NUM_BATCHES_TO_SKIP = None
    TRAINABLE_POSITION_EMB = False
    USE_ROTARY_POSITION_EMB = True
    USE_ALIBI_POSITION_EMB = False
    ROTARY_TYPE = 'qwen'
    NORMALIZATION_CLS = normalizations.RmsNorm
    QKV_BIAS = True
    O_BIAS = False
    USE_BIAS = False
    FPROP_DTYPE = jnp.bfloat16

    CHECKPOINT_EVERY_N_STEPS = 200
    EVAL_LOOP_NUM_BATCHES = 25
    EVAL_INTERVAL_STEPS = 100
    CHECKPOINT_MAX_TO_KEEP = 2

    WANDB_PROJECT = "Qwen14B"
    LM_HEAD_NORM = False

    QUERY_CHUNK_SIZE = 256
    LM_HEAD_CHUNK_SIZE = 512
    RESET_FOR_EVAL = False
    TARGET_LOG_PPLX = -1
    SHUFFLE = {"train": True, "test": True}
    DATA_REPEAT = {'train': 5, 'test': 40}
    SHUFFLE_SIZE = {"train": 1300000, "test": 40000}
    TRAINING_SEED = 1234
    TEST_RATIO = 0.02

    # c4 text datasets. when LOAD_SEQIO_TEXT is True ，recovery code
    LOAD_SEQIO_TEXT = False
    LOAD_SEQIO_ID = False
    KEY_MAP = {"targets": "input_ids", "masks": "labels"}
    # DATA_PATH = {
    #             'train': ['gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208', 
    #                       'gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208'], 
    #             'test':  ['gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208', 
    #                       'gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208']
    #             }
    DATA_PATH = {
                'train': ['gs://jax_llm_data_us-east5/xiaomeng/sft_target/tfrecord_len2k/'],
                'test':  ['gs://jax_llm_data_us-east5/xiaomeng/sft_target/tfrecord_len2k/'], 
                }
    DATA_FUNC = extract_sft_datapath
    # DATA_FUNC = extract_qwen_datapath_shuffled
    # DATA_FUNC = extract_qwen_datapath2
    # DATA_FUNC = extract_qwen_datapath1208
    # DATA_FUNC = extract_qwen_datapath1208_shuffled
    SAVE_ON_STEPS = list(range(1000, 100000, 1000)) + [11800]
    ONLY_EVAL = False

   
@experiment_registry.register
class BaseEval():
    ONLY_EVAL = True
    TEST_RATIO = 1
    RESET_FOR_EVAL = True # True: test while test dataset
    DATA_PATH = {
                'train': 'gs://common_datasets/pythia_model_test/pile_test',
                'test':  'gs://common_datasets/pythia_model_test/pile_test',
                }
    DATA_FUNC = extract_pythia_datapath
    ICI_MESH_SHAPE = [1, 32, 1]
    PERCORE_BATCH_SIZE = 32


@experiment_registry.register
class Qwen7BEval(BaseEval, Qwen7B):
    TEST_RATIO = 1
    RESET_FOR_EVAL = False # True: test while test dataset
    KEY_MAP = {"targets": "input_ids", "masks": "input_ids"}
    # DATA_PATH = {
    #             'train': 'gs://jax_llm_data/xiaomeng/processed_zh_data_qwen7B_test1024/', 
    #             'test':  'gs://jax_llm_data/xiaomeng/processed_zh_data_qwen7B_test1024/', 
    #             }
    # DATA_FUNC = extract_pythia_datapath
    ICI_MESH_SHAPE = [1, 8, 4]
    PERCORE_BATCH_SIZE = 1
    EVAL_LOOP_NUM_BATCHES = 10
    MAX_SEQ_LEN = 4097
    SHUFFLE = {"train": False, "test": False}
    KEY_MAP = {"targets": "input_ids", "masks": "input_ids"}
    DATA_PATH = {
            'train': ['gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/', 
                        'gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117'], 
            'test':  ['gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/', 
                        'gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117']
            }
    DATA_FUNC = extract_qwen_datapath


@experiment_registry.register
class Qwen14BEval(BaseEval, Qwen14B):
    TEST_RATIO = 0.02
    RESET_FOR_EVAL = False # True: test while test dataset
    ICI_MESH_SHAPE = [1, 16, 2]
    PERCORE_BATCH_SIZE = 4
    EVAL_LOOP_NUM_BATCHES = 20
    MAX_SEQ_LEN = 4097
    SHUFFLE = {"train": True, "test": True}
    KEY_MAP = {"targets": "input_ids", "masks": "labels"}
    # KEY_MAP = {"targets": "input_ids", "masks": "input_ids"}
    FPROP_DTYPE = jnp.bfloat16
    DATA_PATH = {
            'train': ['gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/', 
                        'gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117'], 
            'test':  ['gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/', 
                        'gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117']
            }
    QUERY_CHUNK_SIZE = 128
    LM_HEAD_CHUNK_SIZE = None
    ROTARY_TYPE = 'qwen'
    # DATA_FUNC = extract_qwen_datapath
    DATA_FUNC = extract_qwen_datapath1208
    # DATA_FUNC = extract_qwen_datapath_shuffled
    TRAINING_NUM_BATCHES_TO_SKIP = 9000

@experiment_registry.register
class BC2Gpt13BEval(BaseEval, BC2Gpt13B):
    TRAINING_NUM_BATCHES_TO_SKIP = None
    ICI_MESH_SHAPE = [1, 8, 1]
    PERCORE_BATCH_SIZE = 4
    # DATA_PATH = {
    #             'train': 'gs://common_datasets/pythia_model_test/pile_test', 
    #             'test':  'gs://common_datasets/pythia_model_test/pile_test', 
    #             }
    # DATA_FUNC = extract_pythia_datapath

    RESET_FOR_EVAL = False
    EVAL_LOOP_NUM_BATCHES = 20

     # c4 text datasets. when LOAD_SEQIO_TEXT is True ，recovery code
    LOAD_SEQIO_TEXT = True
    KEY_MAP = {"inputs": None, "targets": "text"}
    VOCAB_FILE = "gs://llm_base_models/baichuan2-13b-hf/tokenizer.model"
    VOCABULARY = t5.data.SentencePieceVocabulary(VOCAB_FILE)
    DATA_PATH = {
                'train': 'gs://common_datasets/', 
                'test':  'gs://common_datasets/', 
                }
    DATA_FUNC = c4_registry

 
class MyDatasets(base_input.BaseInput):
    path: Optional[str] = None
    num_infeed_hosts: int = 0
    reset_for_eval: bool = False
    is_training: bool = True
    batch_size: int = 8
    seq_len: int = 2048
    repeat: int = 1
    train_seed: int = 1234
    task_features: Optional[dict] = None
    shuffle_buffer_size: Optional[int] = None
    pad_id: int = 0
    drop_remainder: bool = True
    iter_file_nums: int = 100 # 100  500 steps/file
    meta_dict: Optional[dict] = None
    num_batches_to_skip: Optional[int] = None
    only_eval: bool = False

    def __post_init__(self):
        if self.num_infeed_hosts == 0:
            self.num_infeed_hosts = jax.process_count()

        if not self.meta_dict or self.only_eval:
            self.meta_dict = {
                "seed": self.train_seed,
                "cur_files": [],
                "file_in_data": 0,
                "step_in_file": 0,
                "iter_file_nums": self.iter_file_nums,
                "checkpoint_step": None,
            }
            self.step_in_file = 0  # XD fix
        else:
            if self.meta_dict["file_in_data"] != 0:
                assert self.meta_dict["iter_file_nums"] == self.iter_file_nums, print(
                    f'iter_file_nums in meta_dict is not equal to cur args. => {self.meta_dict["iter_file_nums"]}≠'
                    f" {self.iter_file_nums}"
                )
            self.step_in_file = self.meta_dict['step_in_file']  # XD fix
        logging.info(f'meta_dict: {self.meta_dict}')
        self.train_seed = self.meta_dict['seed']
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)
        self._peek = None
        self._state_before_peek = None
        self.label_flag = 0

 #   def peek_padded(self):
  #      return self.get_next_padded()

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
            example[name] = tf.sparse.to_dense(t, default_value=0)
        return example

    def reset(self) -> None:
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)

    def convert(self, data):
        seq_len = self.seq_len
        model_needed_inputs = NestedMap()
        model_needed_inputs.ids = data["input_ids"][:, : seq_len - 1]
        logging.info(f'process index {jax.process_index()} load input_ids: {model_needed_inputs.ids}')
        model_needed_inputs.labels = data["input_ids"][:, 1:seq_len]
        if "labels" in data:
            # lsp: 第一次打印数据
            if self.label_flag == 0:
                logging.info(f'=================data:\n{data}')
            self.label_flag = 1
            weights = data["labels"] > 0
        else:
            weights = data["input_ids"] >= 0
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
        # ds = ds.shard(self.num_infeed_hosts, process_index)
        ds = ds.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        if self.shuffle_buffer_size is not None:
            logging.info(f'[lsp]shuffle_buffer_size: {self.shuffle_buffer_size}')
            ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)
        padded_shapes = {key: self.seq_len for key in self.task_features}
        # padded_shapes = {key: 4097 for key in self.task_features}
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