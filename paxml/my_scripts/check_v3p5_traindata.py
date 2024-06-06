# pip install tiktoken


from transformers import AutoTokenizer


TOKENIZER_PATH = "/home/lishengping/tokenizer"
MAX_LEN = 4097
EOS_ID = [151643] # <|endoftext|>
BOS_ID = [151646] #  <|extra_0|>

tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH, use_fast=True, trust_remote_code=True
        )


import os
import time
import argparse
import socket
import random
from collections import defaultdict
os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
import jax
import numpy as np


seq_len = 2048

def _parse_function(example_proto):
    # seq_len = 1024
    feature_desc = {key: tf.io.VarLenFeature(tf.int64) for key in task_features}
    example = tf.io.parse_single_example(example_proto, feature_desc)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        # example[name] = tf.sparse.to_dense(t, default_value=0)[: 2 * seq_len - 2]
        # example[name] = tf.reshape(example[name], [2, seq_len - 1])
        # t = tf.constant([[10], [10]], dtype=tf.int32)
        # example[name] = tf.concat([t, example[name]], 1)
        example[name] = tf.sparse.to_dense(t, default_value=0)[: seq_len]
        print(f'example[name]: {example[name]}')
    return example

task_features = {'input_ids': None}
train_seed = 1234
num_infeed_hosts = 1
shuffle_buffer_size = None
pad_id = 0
batch_size = 32

fname = ['gs://jax_llm_data/xiaomeng/sft_target/tfrecord_len2k/en.test.continue_write.tfrecord']
fname = ['gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfrecord/valid_concat.tfrecord']
fname = ['gs://jax_llm_data_us-central2/xiaomeng/v3.5/val_from_train/B039.F009.val.tfrecord']
fname = ['gs://jax_llm_data_us-east5/xiaomeng/v3.5/novel_from_train/B039.F009.val']
fname = ['gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfids0527/B020/F009/009']
fname = ['gs://jax_llm_data_us-central2/xiaomeng/v3.5/tfids0527/B023/F009/009']

# fname = ['gs://jax_llm_data/xiaomeng/sft_target/tfrecord_len2k/en.test.continue_write.tfrecord']
tf.random.set_seed(train_seed)
ds = tf.data.Dataset.from_tensor_slices(fname)
ds = ds.apply(tf.data.TFRecordDataset)
# shard host data
ds = ds.shard(num_infeed_hosts, 0)
ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
if shuffle_buffer_size is not None:
    ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)
padded_shapes = {key: seq_len for key in task_features}
padding_values = {key: pad_id for key in task_features}
ds = ds.padded_batch(
    batch_size=np.prod(batch_size),
    padded_shapes=padded_shapes,
    padding_values=padding_values,
    drop_remainder=True,
)
# ds = ds.map(self.convert)
# ds = ds.prefetch(tf.data.AUTOTUNE)

iter_ds = ds.as_numpy_iterator()
while 1:
    a = next(iter_ds)
    input_ids = a['input_ids']
    fir, sec = np.where(input_ids == 151643)
    for i in range(len(fir)):
        x = fir[i]
        y = sec[i]
        selected_ids =  input_ids[x, y-10: y+1000]
        tokens = tokenizer.decode(selected_ids)
        print(f'i: {i} tokens: {tokens}\n\n==========================\n\n\n')
        # break
    break
    