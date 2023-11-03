# 可能遇到的问题
# wandb > ..
# collections.abc


# torch=1.12
# deepspeed=0.6
# shortuuid

# np.float -> np.float64


# sudo apt install make
# pip install pybind11
# sudo apt install build-essential
# make

# data/indexed_dataset.py 的path exist问题

import sys
import torch
import socket
import random

sys.path.append('/home/lishengping/pythia/utils/gpt-neox')

from megatron.data.data_utils import (build_train_valid_test_datasets, 
                                    get_normalized_weights_and_num_samples, 
                                    build_weighted_datasets,
                                    make_data_loader
                                     )
from megatron.data.blendable_dataset import BlendableDataset
from megatron import mpu, print_rank_0

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import json
import time

import mlxu
import tensorflow as tf
import multiprocessing
# from multiprocessing import set_start_method


def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

class neox_args():
    
    train_iters = 14300
    eval_interval = 1000
    eval_iters = 10
    train_batch_size = 1024
    batch_size = 1024
    train_data_paths = ['/home/lishengping/common_datasets_us-central2/pythia_pile_idxmaps3/pile_20B_tokenizer_text_document']
    valid_data_paths = ['/home/lishengping/common_datasets_us-central2/pythia_pile_idxmaps3/pile_20B_tokenizer_text_document']
    test_data_paths = ['/home/lishengping/common_datasets_us-central2/pythia_pile_idxmaps3/pile_20B_tokenizer_text_document']
#    
#     valid_data_paths = None
#     test_data_paths = None
    train_data_weights = [1.0]
    valid_data_weights = [1.0]
    test_data_weights = [1.0]
    
    weight_by_num_documents = None
    data_impl = 'mmap'
    seq_length = 2048
    seed = 1234
    mmap_warmup = False
    num_workers = 1
    iteration = 0
    gradient_accumulation_steps = 2
    

train_iters = neox_args.train_iters
eval_iters = (train_iters // neox_args.eval_interval + 1) * neox_args.eval_iters
test_iters = neox_args.eval_iters
train_val_test_num_samples = [
    train_iters * neox_args.train_batch_size,
    eval_iters * neox_args.train_batch_size,
    test_iters * neox_args.train_batch_size,
]
if neox_args.train_data_paths:
    train_weights, train_num_samples = get_normalized_weights_and_num_samples(
        neox_args.train_data_weights, train_val_test_num_samples[0]
    )
    valid_weights, valid_num_samples = get_normalized_weights_and_num_samples(
        neox_args.valid_data_weights, train_val_test_num_samples[1]
    )
    test_weights, test_num_samples = get_normalized_weights_and_num_samples(
        neox_args.test_data_weights, train_val_test_num_samples[2]
    )

    # build individual datasets
    train_datasets, valid_datasets, test_datasets = build_weighted_datasets(
        neox_args,
        train_num_samples,
        valid_num_samples,
        test_num_samples,
        train_weights,
        valid_weights,
        test_weights,
        build_index_mappings=not neox_args.weight_by_num_documents,
    )

    if train_datasets:
        train_ds = BlendableDataset(train_datasets, train_weights)
    if valid_datasets:
        valid_ds = BlendableDataset(valid_datasets, valid_weights)
    if test_datasets:
        test_ds = BlendableDataset(test_datasets, test_weights)
else:
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=neox_args.data_path,
        data_impl=neox_args.data_impl,
        splits_string=neox_args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=neox_args.seq_length,
        seed=neox_args.seed,
        skip_warmup=(not neox_args.mmap_warmup),
    )

def process(args):
    workers, host_id, rank_id = args
    data_nums_perworker = 512000
    start_time = time.time()
    wp = f'/home/lishengping/common_datasets_us-central2/pythia_pile_idxmaps_tfrecord/pile_20B_tokenizer_text_document.H{host_id}_R{rank_id}'
    start = (host_id * workers + rank_id) * data_nums_perworker
    end = start + data_nums_perworker
    total_nums = len(train_ds)
    if start > total_nums - 1:
        return
    print(f'workers: {workers} host_id: {host_id} rank_id: {rank_id}')
    print(f'start: {start} end: {end}')
    with tf.io.TFRecordWriter(wp) as writer:
        for index in range(start, end, 1):
            example = train_ds[index]
            if index % 100 == 0:
                print(f'rank: {rank_id} processed: {index}/{data_nums_perworker} take: {time.time() - start_time}s')
            feature = {
                "input_ids": _int64_feature(example['text']),
                      }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            
workers = int(sys.argv[1])

hostname = socket.gethostname()
host_id = hostname
if isinstance(host_id, str) and len(host_id) > 5:
    host_id = host_id.rsplit("-", maxsplit=1)[-1]
host_id = int(host_id)
# set_start_method("spawn")  # tpu-vm
random.seed(42)

args = [workers, host_id, rank]
process(args)


