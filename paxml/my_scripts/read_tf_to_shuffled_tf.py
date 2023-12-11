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
            

path = 'gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208'
zh_files = read_bucket(path, substrings=['_R', '_F'], split='_b')
path = 'gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208'
en_files = read_bucket(path, substrings=['_R', '_F'], split='_b')
total_files = []
for key in range(10000, 20001, 10000):
    zh_file = zh_files.get(key, None)
    en_file = en_files.get(key, None)
    if zh_file is None or en_file is None:
        break
    total_files.extend(zh_file)
    total_files.extend(en_file)

total_files2 = []
for key in range(30000, 70001, 10000):
    zh_file = zh_files.get(key, None)
    en_file = en_files.get(key, None)
    if zh_file is not None:
        total_files2.extend(zh_file)
    if en_file is not None:
        total_files2.extend(en_file)
    
random.seed(1234)
random.shuffle(total_files)
random.shuffle(total_files2)

test_nums = int(len(total_files) * 0.02)
test = total_files[: test_nums]
train = total_files[test_nums:]

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
    
meta_dict = {"seed": 1234, "cur_files": ["gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R9_E31_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R118_E9_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R171_E34_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R35_E29_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R168_E36_b20000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R15_E32_b20000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R88_E32_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R71_E30_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R162_E9_b10000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R154_E9_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R149_E14_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R159_E15_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R66_E33_b20000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R147_E14_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R98_E32_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R27_E9_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R47_E32_b20000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R68_E34_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R15_E9_b10000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R7_E9_b10000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R1_E9_b10000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R2_E9_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R148_E33_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R72_E9_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R130_E30_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R142_E9_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R26_E13_b10000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R173_E29_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R9_E9_b10000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R19_E36_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R108_E33_b20000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R56_E34_b20000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R148_E10_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R19_E13_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R63_E14_b10000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R162_E34_b20000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R72_E14_b10000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R170_E31_b20000", "gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208/zh_R22_E14_b10000", "gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208/en_R153_E33_b20000"], "file_in_data": 6, "step_in_file": 82, "iter_file_nums": 40, "checkpoint_step": 8200}


def load_tfrecord_dataset(fnames):
    tf.random.set_seed(1234)
    assert isinstance(fnames, list)
    repeat_fnames = fnames * 1
    N = math.ceil(len(repeat_fnames) / 40)
    file_in_data = meta_dict["file_in_data"]
    print(f'file_in_data: {file_in_data} N: {N}')
    noruned_files = []
    for n in range(file_in_data, N, 1):
        fname = repeat_fnames[n * 40 : (n + 1) * 40]
        meta_dict["cur_files"] = fname
        noruned_files.append(fname)
    return noruned_files

noruned_files_20k = load_tfrecord_dataset(train)
noruned_files_20k_zh = [f for fs in noruned_files_20k for f in fs if 'zh_' in f]
noruned_files_20k_en = [f for fs in noruned_files_20k for f in fs if 'en_' in f]
noruned_files_gt20k_zh = [f for f in total_files2 if 'zh_' in f]
noruned_files_zh = noruned_files_20k_zh + noruned_files_gt20k_zh
random.shuffle(noruned_files_zh)
# noruned_files_zh = random.sample(noruned_files_zh, k=300)
print(f'noruned_files_zh: {len(noruned_files_zh)} noruned_files_20k_en: {len(noruned_files_20k_en)}')


data_type = 'en'
shuffle_buffer_size = 2000000
tf.random.set_seed(1234)
if data_type == 'en':
    ds = tf.data.Dataset.from_tensor_slices(noruned_files_20k_en)
else:
    ds = tf.data.Dataset.from_tensor_slices(noruned_files_zh)
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
wp = f'gs://jax_llm_data/xiaomeng/{data_type}_data_Qwen-14B_1208_shuffled/{data_type}_b0'
writer = tf.io.TFRecordWriter(wp)
for step, d in enumerate(iter_ds2):
    if (step + 1) % 10000 == 0:
        writer.close()
        wp = f'gs://jax_llm_data/xiaomeng/{data_type}_data_Qwen-14B_1208_shuffled/{data_type}_b{step + 1}'
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
