import os
import time
from functools import partial
import multiprocessing
from multiprocessing import set_start_method
import random
import sys
from collections import defaultdict
import socket

import orjson
from tqdm import tqdm
from xopen import xopen
from transformers import AutoTokenizer
import tensorflow as tf
import smart_open
from etils import epath
import json

class QwenTokenizer():
    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=False, trust_remote_code=True
        )
        self.next_ids = []
        self.count = 0

    @classmethod
    def _int64_feature(cls, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def write_to_tfrecord(self, input_ids):
        feature = {
            "input_ids": self._int64_feature(input_ids),
#             "labels": self._int64_feature(masks),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())
        self.count += 1

    def tokenize(self, text, max_len=2048, bos_id:list=[], eos_id:list=[]):
        input_ids = self.tokenizer.encode(text) + eos_id
        if bos_id:
            max_len -= 1
        self.next_ids += input_ids #  加上上个step保留的id
        while len(self.next_ids) >= max_len:
            save_ids = self.next_ids[: max_len]
            if len(save_ids) == max_len:
                save_ids = bos_id + save_ids
                self.write_to_tfrecord(save_ids)
                self.next_ids = self.next_ids[max_len: ]
            else:
                self.next_ids = save_ids
                save_ids = []


def process_wrapper(args):
    rank, read_dir, save_dir, filenames = args
    start = time.time()
    # local host
#     tokenizer_path = "/home/lishengping/qwen14b"
    # bucket remote
    if 'gs:' in read_dir:
        tokenizer_path = "Qwen/Qwen-14B"
    else:
        tokenizer_path = "/home/lishengping/qwen14b"

    # save_path = 'gs://jax_llm_data_us-east5/xiaomeng/v3.5/valid.tfrecord'
    qwen_tokenizer = QwenTokenizer(tokenizer_path)
    max_len = 4097
    eos_id = 151643 # <|endoftext|>
    bos_id = 151646 #  <|extra_0|>
    tokenize_func = partial(qwen_tokenizer.tokenize, max_len=max_len, bos_id=[bos_id], eos_id=[eos_id])
    if isinstance(filenames, str):
        filenames = [filenames]
    for filename in filenames:
        save_filename = filename.replace('.jsonl.zst', '.tfrecord')
        read_path = os.path.join(read_dir, filename)
        save_path = os.path.join(save_dir, save_filename)
        print(f'save_path: {save_path}')
        qwen_tokenizer.writer = tf.io.TFRecordWriter(save_path)
        with smart_open.open(read_path, 'rb') as combined_file:
            # for line in tqdm(combined_file, desc=f'Rank-{rank}'):
            for index, line in enumerate(combined_file):
                line = orjson.loads(line)
                text = line['text']
                text_split = text.split('\n')
                per = 1000
                if len(text_split) > 50000:
                    for lnx in tqdm(range(0, len(text_split), per), desc=f'Rank-{rank}-sub-{i}'):
                        inp = text_split[lnx * per: (lnx + 1) * per]
                        inp = '\n'.join(inp)
                        ids = tokenize_func(input)
                else:
                    ids = tokenize_func(text)

                if index % 10000 == 0:
                    print(f'Rank: {rank}, index: {index} count: {qwen_tokenizer.count} ratio: {index / 3000000:3f} take: {time.time() - start:.3f}s')
        qwen_tokenizer.writer.close()
    return qwen_tokenizer.count


if __name__ == "__main__":
    random.seed(42)
    index = sys.argv[1]
    bucket, start, end = [int(i) for i in index.split(',')]
    buckets = [bucket]

    read_dir = '/mnt/nvme2/kf/temp_data/combined_data_validexcluded' # elderberry
    read_dir = 'gs://jax_llm_data_us-east5/xiaomeng/v3.5/jsonl/' # bucket

    if 'gs:' in read_dir:
        save_dir = 'gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfid3/' # bucket
        perbucket_file_num = 10
        # 2nd-shuffled-data_bucket-25-008-of-010.jsonl.zst
        train_files = []
        for bucket in buckets:
            for fdx in range(perbucket_file_num):
                filename = f'2nd-shuffled-data_bucket-{bucket}-{fdx:03}-of-010.jsonl.zst'
                # path = os.path.join(read_dir, filename)
                train_files.append(filename)
        train_files = train_files[start: end]
    else:
        save_dir = '/mnt/nvme2/kf/temp_data/combined_data_validexcluded_tfrecord' # elderberry
        train_files = [f for f in os.listdir(read_dir)  for bucket in buckets if f.endswith('.zst') and f'bucket-{bucket}-' in f]
        train_files = sorted(train_files, key=lambda x: [int(x.split('-')[3]), x.split('-')[4]])
        # valid_file = os.path.join(read_dir, 'valid_concat.jsonl')

    data_type = 'valid1'
    if data_type == 'valid':
        train_files = ['valid_concat.jsonl']

    print(f'train_files: {train_files}')
    # buckets = list(range(40))
    WORKERS = len(train_files)

    # set_start_method("spawn")  # tpu-vm
    num_processes = multiprocessing.cpu_count()
    print(f"num_processes: {num_processes}")
    pool = multiprocessing.Pool(processes=WORKERS)

    args = ([rank, read_dir, save_dir, train_files[rank]] for rank in range(WORKERS)
        )
    results = pool.map(process_wrapper, args)  # 包含每个进程的返回值
    pool.close()
    pool.join()
    # total = sum(results)
    print(f'results: {results}')
    meta_path = f'gs://jax_llm_data_us-east5/xiaomeng/v3.5/meta7.json'
    meta_path = epath.Path(meta_path)
    meta_dict = dict(zip(train_files, results))
    with meta_path.open('a') as f:
        meta_dict = json.dumps(meta_dict, ensure_ascii=False)
        f.write(f'{meta_dict}\n')


# Usage:
# pip install tiktoken orjson xopen smart_open[gcs]
# TPU_NAME=llm-jax-mqy-v4-32-11; ZONE=us-central2-b
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install tiktoken orjson xopen smart_open[gcs]" --project=ntpu-413714
# SCRIPT=/Users/lishengping/codes/jax_projects/paxml_praxis/paxml/my_scripts/processed_single_file.py
# gcloud compute tpus tpu-vm scp $SCRIPT $TPU_NAME:/home/lishengping/processed.py  --zone=$ZONE  --worker=all  --project=ntpu-413714
# TPU_NAME=llm-jax-mqy-v4-32-11; ZONE=us-central2-b
# B=1,5,10
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=3 --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py $B"