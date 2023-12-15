import json
import functools
from collections import defaultdict
import random
import os
import math
import time

import tensorflow as tf
import numpy as np
from google.cloud import storage # google-cloud-storage


def read_bucket(path, substrings=None, split='_'):
    path = path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path  + '/'
    print(f"bucket_name: {bucket_name} directory_path: {directory_path}")
    step_map_path = defaultdict(list)
    rerank = 0
    files = defaultdict(list)
    client = storage.Client()
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        filename = blob.name
        if substrings is None:
            substrings = []
        elif isinstance(substrings, str):
            substrings = [substrings]
        if any(substring in filename for substring in substrings):
            # print(f"Successful filename: {filename}=====")
            try:
                step = int(filename.rsplit(split, maxsplit=1)[-1])
            except:
                step = rerank
                rerank += 1
            path = f'gs://{os.path.join(bucket_name, filename)}'
            files[step].append(path)
        else:
            print(f"Failed filename: {filename}=====")
    return files
            

path = 'gs://jax_llm_data/xiaomeng/zh_data_Baichuan2-13B-Base_1213'
zh_files = read_bucket(path, substrings=['_R', '_F'], split='_b')
path = 'gs://jax_llm_data/xiaomeng/en_data_Baichuan2-13B-Base_1213'
en_files = read_bucket(path, substrings=['_R', '_F'], split='_b')
total_files = []
for key in range(0, 10001, 10000):
    zh_file = zh_files.get(key, None)
    en_file = en_files.get(key, None)
    
    if zh_file is not None:
        total_files.extend(zh_file)
    if en_file is not None:
        if key == 10000:
            en_file = random.sample(en_file, k=len(en_file) // 2)
        total_files.extend(en_file)
        
random.seed(1234)
random.shuffle(total_files)


task_features = {'input_ids': 4097, 'labels': 4097}


def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _parse_function(example_proto):
    feature_desc = {key: tf.io.VarLenFeature(tf.int64) for key in task_features}
    example = tf.io.parse_single_example(example_proto, feature_desc)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = tf.sparse.to_dense(t, default_value=0)
    return example
    
 
data_type = 'zh_en'
model_name = 'bc2_13b'
shuffle_buffer_size = 2000000
tf.random.set_seed(1234)

ds = tf.data.Dataset.from_tensor_slices(total_files)
ds = ds.apply(tf.data.TFRecordDataset)
ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
padded_shapes = {'input_ids': 4097, 'labels': 4097}
padding_values = {'input_ids': 0, 'labels': 0}
ds = ds.shuffle(buffer_size=shuffle_buffer_size)

ds = ds.padded_batch(
    batch_size=1,
    padded_shapes=padded_shapes,
    padding_values=padding_values,
    drop_remainder=True,
)

iter_ds2 = ds.as_numpy_iterator()

start = time.time()
wp = f'gs://jax_llm_data/xiaomeng/{data_type}_data_{model_name}_1214_shuffled/{data_type}_b0'
writer = tf.io.TFRecordWriter(wp)
for step, d in enumerate(iter_ds2):
    if (step + 1) % 10000 == 0:
        writer.close()
        wp = f'gs://jax_llm_data/xiaomeng/{data_type}_data_{model_name}_1214_shuffled/{data_type}_b{step + 1}'
        writer = tf.io.TFRecordWriter(wp)
    input_ids = d['input_ids'][0]
    labels = d['labels'][0]
    if step % 100 == 0:
        print(f'processed: {step} take: {time.time() - start}s')
    feature = {
        "input_ids": _int64_feature(input_ids),
        "labels": _int64_feature(labels),
              }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()
