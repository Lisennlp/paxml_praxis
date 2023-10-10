import os

os.environ["JAX_PLATFORMS"] = "cpu"

import json
import time

import mlxu
import tensorflow as tf


def shard(data, batch_size=None):  # XD
    return jax.tree_map(lambda x: x.numpy().reshape(batch_size + x.shape[1:]), data)  # mtj


def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


start = time.time()

rp = 'gs://jax_llm_data/xiaomeng/compare_torch_data/sample_50k_data_test.jsonl'
rp = 'gs://jax_llm_data/xiaomeng/compare_torch_data/sample_0.9M_data_train.jsonl'

wp = 'gs://jax_llm_data/xiaomeng/compare_torch_data/tfrecord/sample_50k_data_test.tfrecord'
wp = 'gs://jax_llm_data/xiaomeng/compare_torch_data/tfrecord/sample_0.9M_data_train.tfrecord'

N = 50000
N = 900000
with tf.io.TFRecordWriter(wp) as writer, mlxu.open_file(rp, 'r') as fin:
     for index, line in enumerate(fin):
        example = json.loads(line)
        if index % 100 == 0:
            print(f'processed: {index}/{N} take: {time.time() - start}s')
        feature = {
            "input_ids": _int64_feature(example['input_ids']),
                  }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())