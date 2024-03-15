from transformers import AutoTokenizer

import os
import json
import time
import tensorflow as tf
import random
import time
import os


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

# local
# read_dir = '/nas2/yuhe/chatllm/sft_target/zh'
# files = os.listdir(read_dir)
# start = time.time()
# max_length = 4096
# total_lines = []
# for f in files[:1]:
#     abs_path = os.path.join(read_dir, f)
#     with open(abs_path, 'r') as f:
#         lines = f.readlines()
#         total_lines.extend(lines)

# bucket
# zh_read_dir = 'gs://jax_llm_data/xiaomeng/sft_target/zh/'
# en_read_dir = 'gs://jax_llm_data/xiaomeng/sft_target/en/'

client = storage.Client()
bucket_name = 'jax_llm_data_us-east5'
pathes = []
for lang in ['zh', 'en']:
    directory_path = f'xiaomeng/sft_target/{lang}/'
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        filename = blob.name
        path = os.path.join('gs://', bucket_name, filename)
        # path = f'gs://{bucket_name}/{directory_path}'
        pathes.append(path)
print(pathes)


start = time.time()
max_length = 2049
total_lines = []
for path in pathes:
    path = epath.Path(path)
    with path.open('r') as f:
        lines = f.readlines()
        total_lines.extend(lines)

random.shuffle(total_lines)

print(f'Read data and shuffle take: {time.time() - start}s data length: {len(total_lines)}')
train_path = 'gs://jax_llm_data_us-east5/xiaomeng/sft_target/tfrecord_len2k/train.summary.etc.tfrecord'
test_path = 'gs://jax_llm_data_us-east5/xiaomeng/sft_target/tfrecord_len2k/test.summary.etc.tfrecord'

train_writer = tf.io.TFRecordWriter(train_path)
test_writer = tf.io.TFRecordWriter(test_path)


test_nums = 10000

start = time.time()
for i, line in enumerate(total_lines):
    line = json.loads(line)
    text = line['text']
    target = line['target']
    text_ids = tokenizer.encode(text) if text else []
    # text_ids = text_ids[: max_length - 1]
    labels = [0] * len(text_ids) 
    target_ids = tokenizer.encode(target)
    # <|im_start| <|im_end|>
    if text_ids:
        start_id = [151644]
        end_id = [151645]
    else:
        # 去掉续写数据
        continue
        start_id = []
        end_id = []
    input_ids =  text_ids + start_id + target_ids + end_id
    assert sum(labels) == 0
    # labels = labels + start_id + target_ids + end_id
    # start_id不计算loss
    labels = labels + [0] + target_ids + end_id

    assert len(input_ids) == len(labels)
    if len(input_ids) > max_length: 
        print(f'i: {i} len: {len(input_ids)}')
        continue
    feature = {
        "input_ids": _int64_feature(input_ids),
        "labels": _int64_feature(labels),
                }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    if i < test_nums:
        test_writer.write(example.SerializeToString())
    else:
        train_writer.write(example.SerializeToString())
    if i % 100 == 0:
        print(f'Processing: {i}...... take: {time.time() - start}s')


test_writer.close()
train_writer.close()




# =====================================================================

p0 = 'gs://jax_llm_data/xiaomeng/sft_target/en_tfrecord'
p1 = 'gs://jax_llm_data/xiaomeng/sft_target/en_tfrecord'

files = defaultdict(list)
rerank = 0
for lang in ['zh', 'en']:
    client = storage.Client()
    bucket_name = 'jax_llm_data'
    directory_path = f'xiaomeng/sft_target/{lang}_tfrecord'
    substrings = 'tfrecord'
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        filename = blob.name
        if substrings is None:
            substrings = []
        elif isinstance(substrings, str):
            substrings = [substrings]
        if any(substring in filename for substring in substrings):
            try:
                step = int(filename.rsplit(split, maxsplit=1)[-1])
            except:
                step = rerank
                rerank += 1
            path = f'gs://{os.path.join(bucket_name, filename)}'
            files[step].append(path)

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
    
shuffle_buffer_size = 500000
tf.random.set_seed(1234)
frames = [v[0] for k, v in files.items()]
ds = tf.data.Dataset.from_tensor_slices(frames)
ds = ds.apply(tf.data.TFRecordDataset)
ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
padded_shapes = {'input_ids': 2049, 'labels': 2049}
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

wp_train = 'gs://jax_llm_data/xiaomeng/sft_target/tfrecord/train.tfrecord'
wp_test = 'gs://jax_llm_data/xiaomeng/sft_target/tfrecord/test.tfrecord'

train_writer = tf.io.TFRecordWriter(wp_train)
test_writer = tf.io.TFRecordWriter(wp_test)

for step, d in enumerate(iter_ds):
    
    input_ids = d['input_ids'][0]
    labels = d['labels'][0]
    if step % 100 == 0:
        print(f'processed: {step} take: {time.time() - start}s')
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
