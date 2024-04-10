import os
import random
import sys
os.environ["JAX_PLATFORMS"] = "cpu"

from google.cloud import storage
import io
import multiprocessing
from multiprocessing import Pool
from functools import partial
import tensorflow as tf
from transformers import AutoTokenizer
import os
import gcsfs
from tqdm import tqdm
import json
from multiprocessing import set_start_method
import math
import time
from etils import epath
from collections import defaultdict
import smart_open

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_tfrecord(writer, input_ids):
    feature = {
        "input_ids": _int64_feature(input_ids),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


def async_tokenizer(args):
    self, cur_rank_lines, rank, workers = args
    save_path = self.save_path + f'.rank.{rank}'
    work_input_ids = []
    for i in tqdm(range(len(cur_rank_lines)), desc=f'Rank-{rank}'):
        line = cur_rank_lines[i]
        line = json.loads(line)
        input_ids = self.partial_tokenize(line['text'], writer)
        work_input_ids.extend(input_ids)
    return work_input_ids


class QwenTokenizer():
    def __init__(self, tokenizer_path, save_path, max_len, bos_id:list=[], eos_id: list=[]):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=False, trust_remote_code=True
        )
        self.next_ids = []
        self.partial_tokenize = partial(self.tokenize, max_len=max_len, bos_id=bos_id, eos_id=eos_id)
        self.save_path = save_path
        self.count = 0

    def tokenize(self, text, writer,max_len=2048, bos_id:list=[], eos_id:list=[]):
        input_ids = self.tokenizer.encode(text) + eos_id
        if bos_id:
            max_len -= 1
        self.next_ids += input_ids #  加上上个step保留的id
        total_ids = []
        while len(self.next_ids) >= max_len:
            save_ids = self.next_ids[: max_len]
            if len(save_ids) == max_len:
                save_ids = bos_id + save_ids
                write_to_tfrecord(writer, save_ids)
                self.count += 1
                total_ids.append(save_ids)
                self.next_ids = self.next_ids[max_len: ]
            else:
                self.next_ids = save_ids
                save_ids = []
        return total_ids

    def encode_file(self, pathes, workers=6):
        workers_thread = []
        total_lines = []
        for idx, path in enumerate(pathes):
            with smart_open.open(path, 'rb') as f:
                lines = f.readlines()
                total_lines.extend(lines)
                print(f'path:{path}, {len(lines)}')
        print(f'total_lines: {len(total_lines)}')
        pool = Pool(processes=workers)
        perrank_line_num = math.ceil(len(total_lines) / workers)
        args = ([self, total_lines[rank * perrank_line_num: (rank + 1) * perrank_line_num], rank, workers] for rank in range(workers))
        results = pool.map(async_tokenizer, args)  # 包含每个进程的返回值
        pool.close()
        pool.join()
        return results


def extract_files(bucket_name, directory_path):
    client = storage.Client()
    pathes = defaultdict(list)
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        if 'valid' in path:
            pathes['valid'].append(path)
        else:
            pathes['train'].append(path)
    return pathes

import socket
if __name__ == "__main__":
    random.seed(42)
    # file_index = sys.argv[1]
    # file_start, file_end = [int(a) for a in file_index.split('-')]
    hostname = socket.gethostname()
    host_id = hostname
    if isinstance(host_id, str) and len(host_id) > 5:
        host_id = host_id.rsplit("-", maxsplit=1)[-1]
    host_id = int(host_id)
    # file_start = host_id * 1
    # file_end = file_start + 400
    # print(f'host_id: {host_id} file_start: {file_start} file_end: {file_end}')

    # set_start_method("spawn")  # tpu-vm
    num_processes = multiprocessing.cpu_count()
    print(f"num_processes: {num_processes}")
    meta_path = f'gs://jax_llm_data_us-east5/xiaomeng/v3.5/meta5.json'
    meta_path = epath.Path(meta_path)
    tokenizer_path = "Qwen/Qwen-14B"
    max_len = 4097
    eos_id = [151643] # <|endoftext|>
    bos_id = [151646] #  <|extra_0|>
    bucket_name = 'jax_llm_data_us-east5'  # 存储桶名称
    directory_path = 'xiaomeng/v3.5/jsonl'  # 文件在存储桶中的路径
    # pathes = extract_files(bucket_name, directory_path)

    type_ = 'train'
    if type_ == 'valid':
        pathes = ['gs://jax_llm_data_us-east5/xiaomeng/v3.5/jsonl/valid_concat.jsonl']
        save_path = f'gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfrecord/valid_concat.tfrecord'
        print(f'save_path: {save_path}')
    else:
        bucketes = [1]
        pathes = []
        for bucket in bucketes:
            for index in range(10):
                path = f'gs://jax_llm_data_us-east5/xiaomeng/v3.5/jsonl/2nd-shuffled-data_bucket-{bucket}-{index:03}-of-010.jsonl.zst'
                print(f'path: {path}')
                name = os.path.basename(path).split('.jsonl')[0]
                save_path = f'gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfids6/{bucket}-{index}.tfrecord'
                writer = tf.io.TFRecordWriter(save_path)
                print(f'save_path: {save_path}')
                workers = 50
                qwen_tokenizer = QwenTokenizer(tokenizer_path, save_path, max_len, bos_id=bos_id, eos_id=eos_id)
                encodeds = qwen_tokenizer.encode_file([path], workers=workers)
                print(f'Start write to tfrecord: {save_path}')
                w_t = time.time()
                total_nums = 0
                for rank_index, encoded in enumerate(encodeds):
                    for input_ids in encoded:
                        write_to_tfrecord(writer, input_ids)
                    print(f'Write rank-{rank_index} finished, length: {len(encoded)} take: {time.time() - w_t:.2f}s')
                    total_nums += len(encoded)
                writer.close()
                meta_dict = {save_path: total_nums}
                print(meta_dict)
                with meta_path.open('a') as f:
                    meta_dict = json.dumps(meta_dict, ensure_ascii=False)
                    f.write(f'{meta_dict}\n')

# # Usage:
# TPU_NAME=llm-jax-mqy-v4-32-14; ZONE=us-central2-b
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install tiktoken smart_open[gcs] gcsfs" --project=ntpu-413714
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install smart_open[gcs]" --project=ntpu-413714
# SCRIPT=/Users/lishengping/codes/jax_projects/paxml_praxis/paxml/my_scripts/processed_lines.py
# gcloud compute tpus tpu-vm scp $SCRIPT $TPU_NAME:/home/lishengping/processed.py  --zone=$ZONE  --worker=all  --project=ntpu-413714
# TPU_NAME=llm-jax-mqy-v4-32-14; ZONE=us-central2-b
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py" --project=ntpu-413714