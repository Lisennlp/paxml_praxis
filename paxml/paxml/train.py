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

"""Training loop for Pax model."""

import contextlib
import typing
from typing import Type
import os
import re
import subprocess
import json

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
from paxml import base_experiment
from paxml import checkpoint_creators
from paxml import checkpoint_types
from paxml import decode_programs as decode_programs_lib
from paxml import executors
from paxml import experiment_utils
from paxml import partitioning
from paxml import tasks_lib
from paxml import trainer_lib
from praxis import base_hyperparams
from praxis import base_input
from praxis import pax_fiddle
from praxis import py_utils
import tensorflow.compat.v2 as tf

import smart_open
from paxml import checkpoints  # mapped to internal
from paxml import checkpoint_paths

try:
    import wandb
except:
    command = "pip install wandb"
    subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    import wandb

if jax.process_index() == 0:
    wandb.login(key="7988c805dfe3fed4d6e4017f616555a5160fd2c2")

Checkpointer = checkpoints.Checkpointer
CheckpointType = checkpoints.CheckpointType
instantiate = base_hyperparams.instantiate
PaxCheckpointHandler = checkpoints.PaxCheckpointHandler

RunningMode = trainer_lib.RunningMode
SummaryWriter = tf.summary.SummaryWriter
# pylint: disable=protected-access
_checkpoint_dir = checkpoint_creators._checkpoint_dir
_create_checkpointer = checkpoint_creators._create_checkpointer
# pylint: enable=protected-access


@py_utils.benchmark("[PAX STATUS]: ")
def write_hparams_file(
    model_config: base_experiment.BaseExperiment,
    job_log_dir: epath.Path,
    filename_prefix: str = "",
) -> None:
    """Writes a params file into the root `job_log_dir`."""
    if jax.process_index() == 0:
        job_log_dir.mkdir(parents=True, exist_ok=True)
        params_fpath = job_log_dir / f"{filename_prefix}model_params.txt"
        with params_fpath.open("w") as hparams_file:
            for dataset in model_config.datasets():
                hparams_file.write(base_hyperparams.nested_struct_to_text(dataset))
                hparams_file.write("\n\n")
            for decoder_dataset in model_config.decoder_datasets():
                hparams_file.write("decoder dataset hparams\n")
                hparams_file.write(base_hyperparams.nested_struct_to_text(decoder_dataset))
                hparams_file.write("\n\n")
            hparams_file.write(base_hyperparams.nested_struct_to_text(model_config.task()))


def write_experiment_class_vars_file(
    exp_cls: Type[base_experiment.BaseExperiment],
    job_log_dir: epath.Path,
    filename_prefix: str = "",
) -> None:
    """Writes a params file into the root `job_log_dir`."""
    if jax.process_index() == 0:
        exp_summary_fpath = job_log_dir / f"{filename_prefix}experiment_cls_vars.txt"
        job_log_dir.mkdir(parents=True, exist_ok=True)

        cls_vars_summary = experiment_utils.get_cls_vars_summary(exp_cls)
        # epath对象
        exp_summary_fpath.write_text(cls_vars_summary)
        

@py_utils.benchmark("[PAX STATUS]: ")
def train_and_evaluate(
    experiment_config: base_experiment.BaseExperiment,
    job_log_dir: epath.PathLike,
    maybe_use_persistence_checkpointing: bool,
    eval_on_test: bool | None,
    checkpoint_todelete_subdir: str | None = None,
    early_stopping_fn: trainer_lib.EarlyStoppingFn | None = None,
    run_decode: bool = False,
    enable_auto_sharding: bool = False,
    enable_async_checkpointing: bool = False,
    enable_checkpoint_saving: bool = True,
    enforce_restore_shape_check: bool = False,
    tensorstore_use_ocdbt: bool = False,
    exit_after_ondemand_checkpoint: bool = False,
) -> None:
    """The shared path to run the training and evaluation loop.

    Args:
      experiment_config: an instance of BaseExperiment for the experiment to train
        and evaluate.
      job_log_dir: The directory for the job logs.
      maybe_use_persistence_checkpointing: If set, it will try to use
        persistence-based checkpointing if suitable.
      eval_on_test: Whether to eval on test as a part of the training loop.
      checkpoint_todelete_subdir: If set, checkpoints to be deleted will be only
        renamed into the provided subdirectory. Otherwise, they will be directly
        deleted from the file system. This is useful, when checkpoint deletion is
        time consuming.
      early_stopping_fn: An optional callable object for reporting eval metrics
        and determining whether to early stop current training. The callable
        object has signature: (metrics_by_dataset, ckpt_step, is_final_ckpt) ->
        should_stop_early.
      run_decode: whether to periodically run decode as part of the training loop.
        If and only if this is True, every `task_p.train.decode_interval_steps` of
        training, model runs decode.
      enable_auto_sharding: Enables the XLA Auto SPMD partitioner.
      enable_async_checkpointing: Allows training to continue when checkpointing
        is going on as checkpointing happens in a different thread.
      enable_checkpoint_saving: Whether to perform checkpoint saving or not.
      enforce_restore_shape_check: Raises an error if restore shapes do not match
        checkpoint shapes.
      tensorstore_use_ocdbt: Uses OCDBT format for saving new checkpoints.
      exit_after_ondemand_checkpoint: If True, exists immediately after saving an
        on-demand checkpoint due to preemption.
    """
    jax.monitoring.record_event("/jax/pax/train_and_evaluate/beacon")

    task_p = experiment_config.task()  # 怎么又设置了一遍参数？
    task_p = typing.cast(pax_fiddle.Config[tasks_lib.SingleTask], task_p)

    if jax.process_index() == 0:
        wandb_name = task_p.name
        wandb.init(
            project=experiment_config.WANDB_PROJECT,
            name=wandb_name,
            config=experiment_config,
            resume=True,
        )

    # in case the user passed in a string dtype, convert it to an actual dtype
    # jnp.bfloat16
    task_p.model.fprop_dtype = jnp.dtype(task_p.model.fprop_dtype)

    logging.info("[PAX STATUS]: Getting dataset configurations.")

    # [train_p datasets, notrain_p datasets]的pax_fiddle.Config对象
    # num_batches_to_skip = extract_train_skip_step(job_log_dir=job_log_dir)
    input_p = experiment_config.datasets(job_log_dir=job_log_dir)
    for inp in input_p:
        if not isinstance(
            inp,
            (
                pax_fiddle.Config,
                base_input.DistributedInputHParams,
            ),
        ):
            raise ValueError(
                f"Expecting pax_fiddle.Config[BaseInput] from datasets(), got: {inp.ToText()}"
            )
    train_input_p = [v for v in input_p if v.is_training]
    if len(train_input_p) != 1:
        raise ValueError(f"Expecting exactly one training split. Got `{len(train_input_p)}`.")
    train_input_p = train_input_p[0]
    logging.info("[PAX STATUS]: Done getting dataset configurations.")
    # 打印训练数据集参数
    logging.info("train_input_p:")
    for line in base_hyperparams.nested_struct_to_text(
        train_input_p
    ).splitlines():  # pytype: disable=attribute-error
        logging.info("  %s", line)
    # 打印训练task相关参数
    logging.info("task_p:")
    for line in base_hyperparams.nested_struct_to_text(
        task_p
    ).splitlines():  # pytype: disable=attribute-error
        logging.info("  %s", line)
    # Creates the task.
    logging.info("[PAX STATUS]: Creating task")
    # <PaxConfig[SingleTask( -> SingleTask(model=LanguageModel(
    # 实例化task_p，这样的话，jax_task就是一个SingleTask对象，之前task_p是一个pax_config对象
    jax_task = instantiate(task_p)
    # <CheckpointType.GDA: 'gda'>
    checkpoint_type = checkpoint_types.retrieve_checkpoint_type(
        maybe_use_persistence_checkpointing, jax_task
    )
    # PosixGPath('gs://llm_projects/log/C4SpmdGpt37BRoPE')
    job_log_dir = epath.Path(job_log_dir)
    # checkpoint dir: job_log_dir / 'checkpoints'
    # 初始化模型参数加载与保存的对象
    checkpointer = _create_checkpointer(
        task_p,
        job_log_dir,
        checkpoint_type,
        checkpoint_todelete_subdir,
        train_input_p=train_input_p,
        enable_async_checkpointing=enable_async_checkpointing,  # false
        enable_checkpoint_saving=enable_checkpoint_saving,  # false
        enforce_restore_shape_check=enforce_restore_shape_check,
        maybe_use_persistence_checkpointing=maybe_use_persistence_checkpointing,
        tensorstore_use_ocdbt=tensorstore_use_ocdbt,
    )
    # enable_checkpoint_saving: false
    if not enable_checkpoint_saving:
        logging.info("Checkpointing is disabled and no checkpoint will be saved to disk.")
    # EarlyStoppingFn(target_log_pplx=2.69, name='')
    if jax_task.early_stopping_fn is not None:
        if early_stopping_fn is None:
            early_stopping_fn = jax_task.early_stopping_fn
        else:
            raise ValueError(
                "early_stopping_fn is set in both task and train_and_evel function parameter."
            )

    logging.info("[PAX STATUS]: Initializing partitioner")
    # Creates the partitioner, which will be set up later.
    # none
    partitioner = experiment_config.partitioner()
    if not partitioner:
        # For the input pipeline on the Pathways client, the inputs are numpy
        # arrays. We rely on the Pathways to transfer the inputs, since
        # jax.device_put() has a larger performance overhead.
        # true
        reshard_inputs = (
            checkpointer.checkpoint_type != CheckpointType.PERSISTENCE
            or train_input_p.experimental_remote_input
        )
        # lsp: partitioner： PjitPartitioner || enable_auto_sharding: false
        partitioner = partitioning.create_partitioner(
            jax_task,
            reshard_inputs=reshard_inputs,
            auto_sharding_mode=RunningMode.TRAIN if enable_auto_sharding else None,
        )

    # Creates the train/eval/decode programs.
    logging.info("[PAX STATUS]: Initializing train program.")
    # train_program: Class SingleTaskTrainProgram
    train_program = experiment_config.train_program()

    logging.info("[PAX STATUS]: Initializing eval programs.")
    eval_programs = []
    if (
        eval_on_test
        and task_p.train.eval_interval_steps is not None
        and task_p.train.eval_interval_steps > 0
    ):
        eval_programs = experiment_config.eval_programs()

    logging.info("[PAX STATUS]: Initializing decode programs.")
    if (
        run_decode
        and task_p.train.decode_interval_steps is not None
        and task_p.train.decode_interval_steps > 0
    ):
        decode_input_p = experiment_config.decoder_datasets()
    else:
        # here
        decode_input_p = []
    # TODO(wangpeng): Make decode programs configurable.
    # []
    decode_programs = [
        decode_programs_lib.SingleTaskDecodeProgram(input_p) for input_p in decode_input_p
    ]

    # Creates the executor and run the training pipeline.
    logging.info("[PAX STATUS]: Creating executor.")
    executor = experiment_config.executor()
    # lsp: None
    if not executor:
        executor = executors.DefaultExecutor()
    logging.info("[PAX STATUS]: Setting up executor.")
    with partitioner.global_mesh or contextlib.nullcontext():
        executor.setup(
            jax_task,
            job_log_dir,
            checkpointer,
            partitioner,
            instantiate(experiment_config.get_input_specs_provider_params()),
            train_input_p,
            train_program,
            eval_programs,
            decode_programs,
            early_stopping_fn,
            exit_after_ondemand_checkpoint=exit_after_ondemand_checkpoint,
        )
        executor.start()
