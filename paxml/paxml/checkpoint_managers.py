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

"""Module to manage checkpoint metadata and automatic checkpoint deletion."""

import dataclasses
import os
import typing
from typing import Any, Mapping, Optional, Sequence, Union

from absl import logging
from etils import epath
import jax
import jax.numpy as jnp
import orbax.checkpoint
from orbax.checkpoint import utils
from paxml import checkpoint_metadata
from paxml import checkpoint_paths
from paxml import checkpoint_types
from paxml import checkpoint_version
from praxis import base_input
from praxis import pytypes
import tensorflow.compat.v2 as tf

from paxml import preemption  # mapped to internal
from praxis import py_utils
from praxis import base_layer


Nested = pytypes.Nested
# TODO(pax-dev): pytyping doesn't like either
# Optional[pytypes.NestedShapeDtypeStruct]
# or pytypes.NestedShapeDtypeStruct | None,
# Switch to the right type hint once pytyping versions are in sync.
OptionalNestedShapeDtypeStruct = Any

STATE_ITEM_NAME = checkpoint_paths.STATE_ITEM_NAME
METADATA_ITEM_NAME = checkpoint_metadata.METADATA_ITEM_NAME
INPUT_ITEM_NAME = checkpoint_paths.INPUT_ITEM_NAME
_SUPPORTED_ITEMS = frozenset({STATE_ITEM_NAME, METADATA_ITEM_NAME})
CheckpointType = checkpoint_types.CheckpointType


def _get_checkpoint_version(
    checkpoint_type: CheckpointType,
    directory: epath.Path,
    step: int,
    use_digit_step_subdirectory: bool = False,
) -> float:
    """Gets checkpoint version from saved metadata."""
    checkpoint_step_dir = checkpoint_paths.make_checkpoint_step_dir(
        directory,
        step,
        checkpoint_type=checkpoint_type,
        use_digit_step_subdirectory=use_digit_step_subdirectory,
    )
    version = 0.0
    # Necessary because some checkpoints do not conform to Orbax directory
    # structure. Could rely exclusively on actual version if all checkpoints
    # conformed.
    if checkpoint_metadata.metadata_exists(checkpoint_step_dir):
        version = checkpoint_metadata.restore_metadata(checkpoint_step_dir)[
            checkpoint_version.get_version_key()
        ]
    return version


def _update_args_with_version(item_kwargs, version):
    kwargs = {STATE_ITEM_NAME: {checkpoint_version.get_version_key(): version}}
    if item_kwargs is not None:
        kwargs[STATE_ITEM_NAME].update(item_kwargs)
    return kwargs


def _create_items_dict_with_metadata(
    train_state,
    train_state_unpadded_shape_dtype_struct,
    version,
    tensorstore_use_ocdbt: Optional[bool] = None,
):
    """Returns items dict with metadata."""
    # (padded) train_state
    items = {STATE_ITEM_NAME: train_state}

    if version > 0:
        metadata = checkpoint_metadata.make_metadata(
            version,
            train_state,
            train_state_unpadded_shape_dtype_struct,
            tensorstore_use_ocdbt=tensorstore_use_ocdbt,
        )
        items.update({METADATA_ITEM_NAME: metadata})

    return items


def _is_legacy_flax_checkpoint(path: epath.Path) -> bool:
    """Returns whether the checkpoint is a legacy Flax checkpoint format.

    Old-format Flax checkpoint conforming to
    'path/to/dir/checkpoints/checkpoint_100'.
    Contrast with 'standard' old-format Flax checkpoint conforming to
    'path/to/dir/checkpoints/checkpoint_100/checkpoint'.
    The former is not considered a valid checkpoint by Orbax because it is not a
    directory. It thus requires special handling.

    Args:
      path: the checkpoint path.

    Returns:
      Boolean indicating whether the path is legacy Flax checkpoint or not.
    """
    return checkpoint_paths.is_checkpoint_asset(path) and (
        not checkpoint_paths.is_tmp_checkpoint_asset(path) and path.is_file()
    )


def _has_digit_step_subdirectory(directory) -> bool:
    """Indicates whether the checkpoints have digit-like step subdirectories."""
    return False  # mapped to internal digit step impl.


@dataclasses.dataclass
class CheckpointManagerOptions(orbax.checkpoint.CheckpointManagerOptions):
    """Options for constructing OrbaxCheckpointManager.

    See superclass.

    Attributes:
      todelete_subdir: If set, checkpoints to be deleted will be only renamed into
        a subdirectory with the provided string. Otherwise, they will be directly
        deleted from the file system. Useful if checkpoint deletion is time
        consuming. By default, delete the checkpoint assets. TODO(b/278901950):
        Remove this option when it is available in Orbax OSS.
      cleanup_tmp_directories: if True, cleans up any existing temporary
        directories on CheckpointManager creation. Set to False by default and
        overridden in select places (like train.py). Otherwise, the
        CheckpointManager is typically a one-off and does not need to concern
        itself with much directory management.
    """

    todelete_subdir: Optional[str] = None
    cleanup_tmp_directories: bool = False


class _CheckpointManagerImpl(orbax.checkpoint.CheckpointManager):
    """Provides Pax-specific logic for orbax.checkpoint.CheckpointManager.

    Pax only supports a single checkpointable item (TrainState) and checkpoints
    are saved under a different folder name in a flat manner (no per-item
    sub-directories).

    Additionally, Pax supports extra options provided via CheckpointManagerOptions
    (see above).

    An instance of this class can be created on several JAX processes.
    All public APIs may be called by all processes.
    """

    def __init__(
        self,
        directory: epath.PathLike,
        *args,
        checkpoint_type: CheckpointType = CheckpointType.UNSPECIFIED,
        tensorstore_use_ocdbt: Optional[bool] = None,
        **kwargs,
    ):
        if checkpoint_type == CheckpointType.UNSPECIFIED:
            raise ValueError("Must specify checkpoint type.")
        self._checkpoint_type = checkpoint_type

        self._version = checkpoint_version.get_version(tensorstore_use_ocdbt)
        # Check for existing checkpoints and retrieve version information. The
        # specific version may impact the checkpoint format, so it must be known in
        # advance of any operations.
        self._directory = epath.Path(directory)
        self._use_digit_step_subdirectory = _has_digit_step_subdirectory(self._directory)
        if self._directory.exists():
            step = self.any_step()
            if step is not None:
                version = _get_checkpoint_version(
                    self._checkpoint_type,
                    self._directory,
                    step,
                    use_digit_step_subdirectory=self._use_digit_step_subdirectory,
                )
                logging.info(
                    "Found existing checkpoint with version: %s, step: %s",
                    version,
                    step,
                )
                if version != self._version:
                    logging.warning(
                        (
                            "Found existing checkpoints with old version %s, compared to "
                            "latest version %s. Use version of existing checkpoints for "
                            "restoring and saving future checkpoints."
                        ),
                        version,
                        self._version,
                    )
                    self._version = version

        super().__init__(directory, *args, **kwargs)
        # Set to 1 if not provided or set to 0.
        self._options.save_interval_steps = self._options.save_interval_steps or 1
        self._options.step_prefix = checkpoint_paths.checkpoint_prefix(self._checkpoint_type)
        self._options.step_format_fixed_length = checkpoint_paths.checkpoint_name_fixed_length(
            self._checkpoint_type
        )

    @property
    def version(self) -> float:
        return self._version

    def all_steps(self, read: bool = False) -> Sequence[int]:
        steps = list(super().all_steps(read=read))
        if read:
            for path in self.directory.iterdir():
                if _is_legacy_flax_checkpoint(path):
                    steps.append(checkpoint_paths.get_step_from_checkpoint_asset(path))
        return steps

    def any_step(self) -> Optional[int]:
        """Returns any step tracked by the checkpoint manager.

        Returns:
          A step (integer) or None.
        """
        any_step = utils.any_checkpoint_step(self.directory)
        if any_step is not None:
            return any_step

        for path in self.directory.iterdir():
            if _is_legacy_flax_checkpoint(path):
                return checkpoint_paths.get_step_from_checkpoint_asset(path)
        return None

    def _checkpoint_name(self, step: int) -> str:
        return checkpoint_paths.checkpoint_name(
            step,
            checkpoint_type=self._checkpoint_type,
            use_digit_step_subdirectory=self._use_digit_step_subdirectory,
        )

    def reached_preemption(self, step: int) -> bool:
        """Returns True if a preemption sync point has been reached."""
        return preemption.reached_preemption_sync_point(step)

    def should_save(self, step: int) -> bool:
        """Indicates whether there is a need to save a checkpoint."""
        if self._use_digit_step_subdirectory:
            raise NotImplementedError(
                "Checkpoints with digit step subdirectories do not support the saving mode."
            )

        # Whether to save an on-demand checkpoint due to preemption
        if self.reached_preemption(step):
            return True
        last_checkpoint_step = self._last_checkpoint.step if self._last_checkpoint else None
        # Ensure current step is between the last step and next step (accounting for
        # save interval). The `last_checkpoint_step` may not be initialized, in
        # which case we should save. Otherwise, step must fall on the specified
        # save interval. This condition accounts for the possibility of saving
        # on preemption, in which case we want to maintain the same save period as
        # if preemption had not happened.
        return last_checkpoint_step is None or (
            last_checkpoint_step < step and step % self._options.save_interval_steps == 0
        )

    def delete(self, step: int):
        """Deletes a step checkpoint."""
        if self._use_digit_step_subdirectory:
            raise NotImplementedError(
                "Checkpoints with digit step subdirectories do not support deletions."
            )
        super().delete(step)

    def save(self, *args, **kwargs) -> bool:
        """Saves the provided items."""
        if self._use_digit_step_subdirectory:
            raise NotImplementedError(
                "Checkpoints with digit step subdirectories do not support the saving mode."
            )
        return super().save(*args, **kwargs)

    def _get_save_directory(
        self,
        step: int,
        directory: epath.Path,
        key_name: Optional[str] = None,
        tmp_directory: Optional[epath.Path] = None,
    ) -> epath.Path:
        """Returns the standardized path to a save directory for a single item."""
        if tmp_directory is None:
            step_dir = checkpoint_paths.make_checkpoint_step_dir(
                directory,
                step,
                checkpoint_type=self._checkpoint_type,
                use_digit_step_subdirectory=self._use_digit_step_subdirectory,
            )
        else:
            step_dir = tmp_directory
        if self._version < 1 or key_name is None:
            return step_dir
        return step_dir / key_name

    def _create_tmp_directory(self, directory: epath.Path) -> epath.Path:
        if self._version < 1:
            # Construct the path without returning. This is because Checkpointer must
            # be allowed to create the path. Only needed for legacy compatibility.
            return orbax.checkpoint.utils.get_tmp_directory(directory)
        return super()._create_tmp_directory(directory)

    def _delete_directory(self, step: int):
        if jax.process_index() != 0:
            return
        options = typing.cast(CheckpointManagerOptions, self._options)
        todelete_subdir = options.todelete_subdir
        checkpoint_name = self._checkpoint_name(step)

        if todelete_subdir:
            rename_dir = self.directory / todelete_subdir
            if not rename_dir.exists():
                rename_dir.mkdir(parents=True)
            src = self.directory / checkpoint_name
            dst = rename_dir / checkpoint_name
            # TODO(pax-team): Check if dst already exists?
            tf.io.gfile.rename(src, dst)
        else:
            super()._delete_directory(step)

    def structure(self) -> Union[Any, Mapping[str, Any]]:
        if self._checkpoint_type == CheckpointType.FLAX:
            raise ValueError("`structure` not supported for Flax format checkpoints.")
        return super().structure()


class OrbaxCheckpointManager:
    """Wrapper class for overridden _CheckpointManagerImpl."""

    def __init__(
        self,
        directory: epath.Path,
        checkpointer: orbax.checkpoint.AbstractCheckpointer,
        train_input_checkpointer: Optional[orbax.checkpoint.Checkpointer] = None,
        options: Optional[CheckpointManagerOptions] = None,
        checkpoint_type: CheckpointType = CheckpointType.UNSPECIFIED,
        tensorstore_use_ocdbt: Optional[bool] = None,
    ):
        self._tensorstore_use_ocdbt = tensorstore_use_ocdbt
        checkpointers = {
            STATE_ITEM_NAME: checkpointer,
            METADATA_ITEM_NAME: orbax.checkpoint.Checkpointer(
                orbax.checkpoint.JsonCheckpointHandler()
            ),
        }

        if train_input_checkpointer:
            checkpointers[INPUT_ITEM_NAME] = train_input_checkpointer

        # Internal Orbax infra configuration

        self._manager = _CheckpointManagerImpl(
            directory,
            checkpointers,
            options=options,
            checkpoint_type=checkpoint_type,
            tensorstore_use_ocdbt=tensorstore_use_ocdbt,
        )

    @property
    def version(self) -> float:
        return self._manager.version

    @property
    def directory(self) -> epath.Path:
        return self._manager.directory

    def all_steps(self) -> Sequence[int]:
        return self._manager.all_steps()

    def latest_step(self) -> Optional[int]:
        return self._manager.latest_step()

    def check_for_errors(self):
        self._manager.check_for_errors()

    def wait_until_finished(self):
        self._manager.wait_until_finished()

    def reached_preemption(self, step: int) -> bool:
        return self._manager.reached_preemption(step)

    def should_save(self, step: int) -> bool:
        return self._manager.should_save(step)

    def _train_checkpoint_exists(self, step: int) -> bool:
        path = self._manager._get_save_directory(  # pylint: disable=protected-access
            step, self.directory, INPUT_ITEM_NAME
        )
        return path.exists()

    def save(
        self,
        step: int,
        train_state: Any,
        train_state_unpadded_shape_dtype_struct: OptionalNestedShapeDtypeStruct = None,
        train_input_pipeline: Optional[base_input.BaseInput] = None,
        force: Optional[bool] = False,
    ) -> bool:
        """See superclass documentation."""
        if self.version > 1.0 and train_state_unpadded_shape_dtype_struct is None:
            raise ValueError(
                """For checkpoint version > 1.0, we require users to provide
          `train_state_unpadded_shape_dtype_struct` during checkpoint
          saving/restoring, to avoid potential silent bugs when loading
          checkpoints to incompatible unpadded shapes of TrainState."""
            )

        # save_kwargs
        # lsp：版本信息{'state': {'version': ...}}
        save_kwargs = _update_args_with_version(None, self.version)

        # items，如果self.version > 0则加入metadata数据进行保存, 没有的话返回{'state': train_state}
        items = _create_items_dict_with_metadata(
            train_state,
            train_state_unpadded_shape_dtype_struct,
            self.version,
            tensorstore_use_ocdbt=self._tensorstore_use_ocdbt,
        )
        if train_input_pipeline:
            items[INPUT_ITEM_NAME] = train_input_pipeline
        # lsp
        # py_utils.sync_global_devices('Model start finish.......')
        # with base_layer.JaxContext.new_context():
        # items = jax.tree_map(lambda x: self.convert_to_float16(x), items)
        return self._manager.save(step, items, save_kwargs=save_kwargs, force=force)
        # return self._manager.save(step, items, force=force)

    def convert_to_float16(self, x):
        if hasattr(x, "dtype"):
            x = x.astype(jnp.float16)
        return x

    # lsp
    def restore(
        self,
        step: int,
        train_state: Any,  # 参数shape和dtype
        train_state_unpadded_shape_dtype_struct: OptionalNestedShapeDtypeStruct = None,
        train_input_pipeline: Optional[base_input.BaseInput] = None,
        restore_kwargs: Optional[Any] = None,  # shard
    ) -> Any:
        """See superclass documentation."""
        uses_transformations = (
            restore_kwargs
            and "transforms" in restore_kwargs
            and restore_kwargs["transforms"] is not None
        )
        # Propagate version to CheckpointHandler. lsp: 参数 shard specs 更新版本信息
        restore_kwargs = _update_args_with_version(restore_kwargs, self.version)
        # lsp: train_state: padded_shape_dtype, 和train_state_unpadded_shape_dtype_struct一样？
        # 根据版本信息，决定是否需要加载metadata数据信息，没有的话就返回{'state': train_state}
        items = _create_items_dict_with_metadata(
            train_state,
            train_state_unpadded_shape_dtype_struct,
            self.version,
            tensorstore_use_ocdbt=self._tensorstore_use_ocdbt,
        )

        # Train input checkpoint may not exist if input checkpointing wasn't
        # previously enabled
        # train_input_pipeline： 输入数据的的shape和dtype
        if train_input_pipeline and self._train_checkpoint_exists(step):
            items[INPUT_ITEM_NAME] = train_input_pipeline

        logging.info(f"items: {items}")
        logging.info(f"restore_kwargs: {restore_kwargs}")
        logging.info(f"step: {step}")
        restored = self._manager.restore(step, items=items, restore_kwargs=restore_kwargs)

        # Skip metadata checks if using transformations, since the TrainState may be
        # completely altered.
        # 检查metadata信息是否正确
        if self.version > 1.0 and not uses_transformations and 0:
            # If unpadded shapes were not provided, skip the shape check for now, as
            # there are many callers that need to be changed.
            if train_state_unpadded_shape_dtype_struct is None:
                logging.error(
                    """For checkpoint version > 1.0, we require users to provide
          `train_state_unpadded_shape_dtype_struct` during checkpoint
          saving/restoring, to avoid potential silent bugs when loading
          checkpoints to incompatible unpadded shapes of TrainState."""
                )
            else:
                restored_metadata = checkpoint_metadata.PaxMetadata.from_dict(
                    restored[METADATA_ITEM_NAME]
                )
                metadata = checkpoint_metadata.PaxMetadata.from_dict(items[METADATA_ITEM_NAME])
                if not metadata.is_compatible(restored_metadata):
                    raise ValueError(
                        "PaxMetadata is not compatible with the restored PaxMetadata. "
                        f"expected PaxMetadata = {restored_metadata}. "
                        f"actual PaxMetadata = {metadata}."
                    )
        # lsp: 返回的已经是在tpu上的参数了
        return restored[STATE_ITEM_NAME]
