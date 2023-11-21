import os

os.environ["JAX_PLATFORMS"] = "cpu"

import json
import time

from etils import epath

import tensorflow as tf


def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


start = time.time()

rp = epath.Path('gs://jax_llm_data/xiaomeng/compare_torch_data/sample_0.9M_data_train.jsonl')

wp = 'gs://jax_llm_data/xiaomeng/compare_torch_data/tfrecord/sample_0.9M_data_train.tfrecord'

N = 50000
N = 900000
with tf.io.TFRecordWriter(wp) as writer, rp.open('r') as fin:
     for index, line in enumerate(fin):
        example = json.loads(line)
        if index % 100 == 0:
            print(f'processed: {index}/{N} take: {time.time() - start}s')
        feature = {
            "input_ids": _int64_feature(example['input_ids']),
                  }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())