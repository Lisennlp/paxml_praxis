from transformers import AutoTokenizer

import os
import json
import time
import tensorflow as tf
import random
import time
import os
import numpy as np

from etils import epath
from google.cloud import storage # google-cloud-storage
from collections import defaultdict
import tensorflow as tf
def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


tokenizer_path = "Qwen/Qwen-14B"

tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=False, trust_remote_code=True
        )


random.seed(42)




def get_inputs(a, data_dtype):
    max_length = 2049

    # random.seed(42)
    if data_dtype == 'zh':
        prompt_target_ids = [271, 99448, 61443, 28311]
        prompt_input_ids = [110644, 28311]
        split_ids = [1773, 8997]
    else:
        prompt_target_ids = [271, 23526, 510]
        prompt_input_ids = [1178, 510]
        split_ids = [13, 624]
    # 151644 151645均不加
    target_start_id = []
    # prompt_target_ids += target_start_id
    # 续写不要结束id
    target_end_id = []
    
    prompt_ids_len = len(prompt_input_ids) + len(prompt_target_ids)
    input_ids = a['input_ids'][0, :max_length]
    labels = a['labels'][0, :max_length]
    
    indices = np.where((input_ids == split_ids[0]) | (input_ids == split_ids[1]))
    # print(f'indices: {indices}')
    # 1/3概率选择完整的
    if len(indices[0]) > 1 and random.randint(1, 3) == 1:
        split_loc = random.choice(indices[0][:-1])
    else: 
        split_loc =  random.randint(1, len(input_ids) - 100)
    # print(f'split_loc: {split_loc}')
    input_ids = input_ids.tolist()
    labels = labels.tolist()
    
    dealed_input_ids = prompt_input_ids + input_ids[:split_loc + 1] + prompt_target_ids +  \
                       input_ids[split_loc + 1: ] + target_end_id
    dealed_labels = [0] * (len(prompt_input_ids) + len(labels[:split_loc + 1]) + len(prompt_target_ids)) +  \
                       labels[split_loc + 1: ] + len(target_end_id) * [1]
    
    dealed_input_ids = dealed_input_ids[:max_length]
    dealed_labels = dealed_labels[:max_length]
    
    return dealed_input_ids, dealed_labels



data_dtype = 'en'
p0 = 'gs://jax_llm_data_us-east5/xiaomeng/sft_target/tfrecord_len4k/test.tfrecord'

zh_files = []
for b in range(2000000, 2500000, 10000):
    p = f'gs://jax_llm_data_us-east5/xiaomeng/zh_data_Qwen-14B_1208_shuffled/zh_b{b}'
    zh_files.append(p)

en_files = []
for b in range(0, 400000, 10000):
    p = f'gs://jax_llm_data_us-east5/xiaomeng/en_data_Qwen-14B_1208_shuffled/en_b{b}'
    en_files.append(p)


def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _parse_function(example_proto):
    padding_values = {'input_ids': 0, 'labels': 0}
    feature_desc = {key: tf.io.VarLenFeature(tf.int64) for key in padding_values}
    example = tf.io.parse_single_example(example_proto, feature_desc)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = tf.sparse.to_dense(t, default_value=0)
    return example

if data_dtype == 'zh':
    frames = zh_files
else:
    frames = en_files
print(f'frames: {frames}')

# frames = frames[:1]
shuffle_buffer_size = 100000
tf.random.set_seed(1234)
ds = tf.data.Dataset.from_tensor_slices(frames)
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

iter_ds = ds.as_numpy_iterator()


start = time.time()

total_nums = 500000 if data_dtype == 'zh' else 400000
wp_train = f'gs://jax_llm_data/xiaomeng/sft_target/tfrecord_len2k/{data_dtype}.train.continue_write.tfrecord'
wp_test = f'gs://jax_llm_data/xiaomeng/sft_target/tfrecord_len2k/{data_dtype}.test.continue_write.tfrecord'

train_writer = tf.io.TFRecordWriter(wp_train)
test_writer = tf.io.TFRecordWriter(wp_test)

for step, inp in enumerate(iter_ds):
    try:
        input_ids, labels = get_inputs(inp, data_dtype)
    except:
        print(f'step: {step} inp: {inp["input_ids"].tolist()}')
        continue
    if step % 100 == 0:
        print(f'processed: {step}/{total_nums} take: {time.time() - start}s')
    feature = {
        "input_ids": _int64_feature(input_ids),
        "labels": _int64_feature(labels),
              }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    if step < 10000:
        test_writer.write(example.SerializeToString())
    else:
        train_writer.write(example.SerializeToString())

train_writer.close()
test_writer.close()
