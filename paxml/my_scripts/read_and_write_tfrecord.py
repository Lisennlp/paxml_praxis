import sys

sys.path.append('/nas/xd/projects/paxml/paxml/')
sys.path.append('/nas/xd/projects/praxis/praxis/')

import tensorflow as tf


class MyDatasets(base_input.BaseInput):
    # Required params. lsp - note: 参数一定要注明类型，不然在初始化的时候就不能传入，会报错没有这个参数
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
    iter_file_nums: int = 100
    meta_dict: Optional[dict] = None
    num_batches_to_skip: Optional[int] = None
    only_eval: bool = False

    def __post_init__(self):
        if self.num_infeed_hosts == 0:
            self.num_infeed_hosts = jax.process_count()

        if not self.meta_dict or self.only_eval: # lsp
            self.meta_dict = {
                "seed": self.train_seed,
                "cur_files": [],
                "file_in_data": 0,
                "step_in_file": 0,
                "iter_file_nums": self.iter_file_nums,
                "checkpoint_step": None,
            }
        else:
            if self.meta_dict["file_in_data"] != 0:
                assert self.meta_dict["iter_file_nums"] == self.iter_file_nums, print(
                    f'iter_file_nums in meta_dict is not equal to cur args. => {self.meta_dict["iter_file_nums"]}≠'
                    f" {self.iter_file_nums}"
                )
        logging.info(f'meta_dict: {self.meta_dict}')
        self.train_seed = self.meta_dict['seed']
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)

    def reset(self) -> None:
        self.dataset = self.load_tfrecord_dataset(fnames=self.path)

    def peek_padded(self):
        return self.get_next_padded()

    def get_next_padded(self):
        unpadded = next(self.dataset)
        pad_size = int(self.batch_padding_size)
        if pad_size == 0:
            return unpadded
        return jax.tree_util.tree_map(
            lambda x: np.pad(x, [[0, pad_size]] + [[0, 0]] * (x.ndim - 1)),
            unpadded,
        )
        # if self.num_infeed_hosts > 1:
        #   x = host_local_array_to_global_array(x, self.mesh, P(('replica', 'data', 'mdl'), None))
        # return x

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

    def convert(self, data):
        seq_len = self.seq_len
        model_needed_inputs = NestedMap()
        model_needed_inputs.ids = data["input_ids"][:, : seq_len - 1]
        model_needed_inputs.labels = data["input_ids"][:, 1:seq_len]
        if "labels" in data:
            weights = data["labels"] >= 0
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
        logging.info(f"num_infeed_hosts: {self.num_infeed_hosts} || process_index: {process_index}")
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
        if self.meta_dict["step_in_file"]:
            ds = ds.skip(self.meta_dict["step_in_file"])
        return ds

    def load_tfrecord_dataset(self, fnames):
        tf.random.set_seed(self.train_seed)
        assert isinstance(fnames, list)
        repeat_fnames = fnames * self.repeat
        N = math.ceil(len(repeat_fnames) / self.iter_file_nums)
        file_in_data = self.meta_dict["file_in_data"]
        flag = 0
        for n in range(file_in_data, N, 1):
            fname = repeat_fnames[n * self.iter_file_nums : (n + 1) * self.iter_file_nums]
            self.meta_dict["cur_files"] = fname
            ds = self._load_file_dataset(fname)
            ds = ds.as_numpy_iterator()
            for batch in ds:
                self.meta_dict["step_in_file"] += 1
                flag = 1
                yield batch
            if flag:
                self.meta_dict["file_in_data"] += 1
                self.meta_dict["step_in_file"] = 0



fname = ['gs://common_datasets/pythia_pile_idxmaps_tfrecord/pile.tfrecord.b19500']

skip_file0 = {"seed": 1234, "cur_files": ["gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/en_R218_F1_1", "gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R124_F2_286", "gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/en_R1_F1_1", "gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R75_F1_1", "gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R236_F1_1", "gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R100_F2_319", "gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/en_R38_F1_1", "gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R65_F1_1", "gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R103_F2_682", "gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R101_F1_1"], "file_in_data": 0, "step_in_file": 200, "iter_file_nums": 10, "checkpoint_step": 200}
skip_file1 = {"seed": 1234, "cur_files": ["gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R215_F1_1", "gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R145_F1_1", "gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R35_F1_1", "gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/en_R196_F1_1", "gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/en_R94_F1_1", "gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R32_F1_1", "gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/en_R24_F1_1", "gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/en_R71_F1_1", "gs://jax_llm_data/xiaomeng/processed_zh_data_qwen14B_KeepChapter1117/zh_R117_F1_1", "gs://jax_llm_data/xiaomeng/processed_en_data_qwen14B_KeepChapter1117/en_R67_F1_1"], "file_in_data": 1, "step_in_file": 349, "iter_file_nums": 10, "checkpoint_step": 1200}
# fname = ['gs://common_datasets/pile.tfrecord.b19500']
fname = skip_file1['cur_files']
def _parse_function(example_proto):
    feature_desc = {'input_ids': tf.io.VarLenFeature(tf.int64)}
    example = tf.io.parse_single_example(example_proto, feature_desc)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = tf.sparse.to_dense(t, default_value=0)
    return example


tf.random.set_seed(1234)
ds = tf.data.Dataset.from_tensor_slices(fname)
ds = ds.apply(tf.data.TFRecordDataset)
ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
padded_shapes = {'input_ids': 4097}
padding_values = {'input_ids': 0}
ds = ds.padded_batch(
    batch_size=256,
    padded_shapes=padded_shapes,
    padding_values=padding_values,
    drop_remainder=True,
)

iter_ds2 = ds.as_numpy_iterator()



import time
start = time.time()
count = 0
for d in iter_ds2:
    print(f'count: {count} shape: {d["input_ids"].shape} take: {time.time() -start}s')
    count += 1



import math
import os

import json
import time
import numpy as np
import tensorflow as tf


def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



data = np.load(data_path)

seq_length = 2049
num_batches = math.ceil(data.shape[0] / seq_length)


start = time.time()

wp = 'gs://llm_projects/pile/val_with_eos.tfrecord'

with tf.io.TFRecordWriter(wp) as writer:
     for index in range(500):
        example = next(iter_ds)
        example = line
        if index % 100 == 0:
            print(f'processed: {index}/{num_batches} take: {time.time() - start}s')
        feature = {
            "input_ids": _int64_feature(example),
                  }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

         