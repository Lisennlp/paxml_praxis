import os
import time
import argparse
import socket
import random
from collections import defaultdict
os.environ["JAX_PLATFORMS"] = "cpu"
import pickle

import tensorflow as tf
from google.cloud import storage
import seqio
import functools
from t5.data import preprocessors as t5_preprocessors
import jax
import numpy as np


TEST_RATIO = 0.02
TRAINING_SEED = 1234
SPLIT_BSZ = {"zh": 7, "en": 20}  # 7表示这本书取了前7次

def extract_datapath(test_ratio, seed, split_batch):
    random.seed(seed)
    dataset = defaultdict(list)
    client = storage.Client()
    bucket_name = "jax_llm_data"
    for lang in ["zh", "en"]:
        # directory_path = f'xiaomeng/processed_{lang}_data_split'
        directory_path = f"xiaomeng/processed_{lang}_data_1001"
        for blob in client.list_blobs(bucket_name, prefix=directory_path):
            if not blob.name or "_R" not in blob.name:
                continue
            if len(dataset[lang]) > 5:
                break
            index = int(blob.name.rsplit("_", maxsplit=1)[-1])
            # 每本书的前多少个4096
            if index < split_batch[lang]:
                path = os.path.join(f"gs://{bucket_name}", blob.name)
                dataset[lang].append(path)
    total = dataset["zh"] + dataset["en"]
    random.shuffle(total)
    test_num = int(len(total) * test_ratio)
    test_num = max(test_num, 1)

    train_test_dataset = {"test": total[:test_num], "train": total[test_num:]}
    print(f'Train file: {len(train_test_dataset["train"])},  test file: {len(train_test_dataset["test"])}')
    return train_test_dataset

DATA_PATH = extract_datapath(TEST_RATIO, TRAINING_SEED, SPLIT_BSZ)
# ds = ds.as_numpy_iterator()

def _parse_function(example_proto):  # https://zhuanlan.zhihu.com/p/552951305  # XD
    feature_desc = {"input_ids": tf.io.VarLenFeature(tf.int64), "labels": tf.io.VarLenFeature(tf.int64)}
    example = tf.io.parse_single_example(example_proto, feature_desc)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = tf.sparse.to_dense(t, default_value=0)
    return example


def shard(data, batch_size=None):  # XD
    return jax.tree_map(lambda x: x.numpy().reshape(batch_size + x.shape[1:]), data)  # mtj


def load_tfrecord_dataset(index_fname, batch_size, seq_len, restore_state=None, repeat=3, skip_step=0):  # XD
    #     tf.random.set_seed(42)
    tf.random.set_seed(1234)
    fnames = index_fname
    ds = tf.data.Dataset.from_tensor_slices(fnames)  # .repeat()
    ds = ds.apply(tf.data.TFRecordDataset)
    # shard host data
    ds = ds.shard(jax.process_count(), jax.process_index())
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=10000)  # 从文件中取buffer_size数据，然后打乱
    ds = ds.padded_batch(
        batch_size=np.prod(batch_size),
        padded_shapes={'input_ids': [seq_len], 'labels': [seq_len]},
        padding_values={'input_ids': 0, 'labels': 0},
        drop_remainder=True,
    )
    ds = ds.prefetch(10)
    ds = ds.repeat(repeat)
    return map(lambda x: shard(x, batch_size=batch_size), iter(ds))

index_fname = DATA_PATH['train']
batch_size = 1
seq_len = 4096
results = load_tfrecord_dataset(index_fname, batch_size, seq_len, restore_state=None, repeat=3)


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

os.environ["JAX_PLATFORMS"] = "cpu"
model_dir = '/home/lishengping/baichuan2-13b-hf/'
sys.path.append(model_dir)

import modeling_baichuan


tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
model = modeling_baichuan.BaichuanForCausalLM(config)

model.eval()

ckpt_pathes = [f for f in os.listdir(model_dir) if f.endswith('.bin')]
weights = {}
for p in ckpt_pathes:
    abs_path = os.path.join(model_dir, p)
    w = torch.load(abs_path, map_location='cpu')
    weights.update(w)
model.load_state_dict(weights)

p = '/home/lishengping/projects/paxml/input_and_loss_1.pkl'
paxml_input_loss = pickle.load(open(p, 'rb'))

# split = 200
# inputs = next(results)
# input_ids = torch.from_numpy(inputs['input_ids'])[:, :split].long()
import time

start_time = time.time()
batch_index = 3
seq_len = 1000

input_ids = paxml_input_loss['input_ids'][batch_index, :seq_len]
paxml_loss = paxml_input_loss['loss'][batch_index, :seq_len-1]

input_ids = torch.tensor([input_ids]).long()
targets = input_ids
output = model.forward(input_ids=input_ids, labels=targets)
end_time = time.time()
print(f'take: {end_time - start_time}s')

torch_loss = output.loss
torch_mean_loss = output.loss.mean()
print(f'torch_mean_loss: {torch_mean_loss} shape: {torch_loss.shape}')

paxml_mean_loss = paxml_loss.mean()
print(f'paxml_mean_loss: {paxml_mean_loss} shape: {paxml_loss.shape}')

id_list = input_ids.reshape(-1).tolist()[1:]
input_tokens = tokenizer.convert_ids_to_tokens(id_list)

for id_, token, tloss, ploss in zip(id_list, input_tokens, torch_loss, paxml_loss):
    if tloss.item() > 10:
        print(id_, token, round(tloss.item(), 4), round(ploss, 4), '===============', sep='    ')
    else:
        print(id_, token, round(tloss.item(), 4), round(ploss, 4), sep='    ')        