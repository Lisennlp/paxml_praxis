import time
import datetime
import os
import re
import sys
import json
import random
import math
import subprocess
import logging
import argparse
from collections import defaultdict
import multiprocessing
from multiprocessing import set_start_method
import socket

os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
from transformers import AutoTokenizer

try:
    import smart_open
    from google.cloud import storage
except:
    command = 'pip install google-cloud-storage && pip install smart_open[gcs]'
    subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    import smart_open
    from google.cloud import storage

# def check_bucket_exist(filepath):
#     # filepath = 'gs://common_datasets_us-central2/pile_seq1024_tokenized/OpenSubtitles/data-00000-of-00096.test.tfrecord'
#     command = f"gsutil stat {filepath}"
#     response = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
#     if response.returncode == 0:
#         return True
#     else:
#         return False


logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Data_processed')


class DataProcessor:
    def __init__(
        self,
        read_bucket,
        read_data_dir,
        tokenizer_path,
        save_dir,
        max_seq_len=2048,
        shuffle=True,
        seed=42,
    ):
        bucket = True
        self.shuffle = shuffle
        logger.info("Init tokenizer.....")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     tokenizer_path,
        #     revision="step3000",
        #     cache_dir="./pythia-70m-deduped/step3000",
        # )
        logger.info("Init tokenizer finished.....")
        self.max_seq_len = max_seq_len
        self.save_dir = save_dir
        self.book_index = 0
        self.file_pathlist = []
        self.write_line = 0
        self.writer = None
        self.clear_threshold_length = 200
        self.book_input_ids = []
        self.read_bucket = read_bucket
        self.read_data_dir = read_data_dir
        self.bos = 1
        self.eos = 0
        self.seed = seed
        self.extract_target_filepath()
        self.extract_filepath()

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def extract_filepath(self):
        # 一定要设定seed，不然不同进程之间的文件顺序可能不一样
        random.seed(self.seed)
        dataset = defaultdict(list)
        client = storage.Client()
        logger.info(f'read_bucket: {self.read_bucket}, read_data_dir: {self.read_data_dir}')
        for blob in client.list_blobs(self.read_bucket, prefix=self.read_data_dir):
            path = os.path.join(f"gs://{self.read_bucket}", blob.name)
            if not blob.name:
                continue
            logger.info(f"Source path: {path}")
            if path in self.target_file_pathlist:
                continue
            if "test" in blob.name:
                dataset["test"].append(path)
            else:
                dataset["train"].append(path)
            logger.info(f'Filter path: {blob.name}===== total train: {len(dataset["train"])}; test: {len(dataset["test"])}')

        self.file_pathlist = dataset["test"] + dataset["train"]
        if self.shuffle:
            random.shuffle(self.file_pathlist)
        logger.info(f"Final data: {len(self.file_pathlist)}")

    def extract_target_filepath(self):
        dataset = defaultdict(list)
        client = storage.Client()
        for blob in client.list_blobs(self.read_bucket, prefix=self.save_dir):
            path = os.path.join(f"gs://{self.read_bucket}", blob.name)
            path = path.replace(self.save_dir, self.read_data_dir)
            re.subn(f"tfrecordF?(\d+)?", "jsonl", path)[0]
            if not blob.name:
                continue
            if "test" in blob.name:
                dataset["test"].append(path)
            else:
                dataset["train"].append(path)
        self.target_file_pathlist = set(dataset["test"] + dataset["train"])

    def convert_line_to_ids(self, line):
        return self.tokenizer.encode(line)

    def write_file(self, save_path, ids, target_ids):
        if self.write_line == 0:
            path = f"{save_path}{self.file_suffix}"
            self.writer = tf.io.TFRecordWriter(path)
        target_length = len(target_ids)
        target_masks = target_ids
        input_length = self.max_seq_len - target_length
        input_ids = ids[ : input_length]
        input_masks = len(input_ids) * [-100]

        input_target_ids = input_ids + target_ids
        input_target_masks = input_masks + target_masks

        assert len(input_target_ids) == len(input_target_masks) <= self.max_seq_len

        feature = {"input_ids": self._int64_feature(input_target_ids), "labels": self._int64_feature(input_target_masks)}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())
        self.write_line += 1
        if self.write_line == self.line_num_perfile and not self.save_book_unit:
            self.file_suffix += 1
            self.writer.close()
            self.write_line = 0

    def process_book(self, path):
        save_path = path.replace(self.read_data_dir, self.save_dir).replace("jsonl", "tfrecord")
        logger.info(f"Start to process file path: {save_path}")
        with smart_open.open(path) as fr:
            for i, line in enumerate(fr):
                line = json.loads(line)
                text = line["text"]
                target = line['target']
                if text.strip():
                    ids = self.convert_line_to_ids(text)
                if target.strip():
                    target_ids = self.convert_line_to_ids(target)
                if len(target_ids) > self.max_seq_len:
                    continue
                self.write_file(save_path, ids, target_ids)

        self.book_index += 1
        if self.save_book_unit:
            self.writer.close()
            self.write_line = 0
            self.book_input_ids = []

    def __len__(self):
        return self.size

    def size(self):
        return len(self.file_pathlist)

    def run(self, start, end, rank):
        N = end - start
        time_start = time.time()
        logger.info(f"Rank: {rank}|| book: {len(self.file_pathlist)} || start: {start} || end: {end}")
        for index, path in enumerate(self.file_pathlist[start:end]):
            try:
                self.process_book(path)
            except Exception as e:
                logger.info(f"Rank: {rank}, error: {e} path: {path}")
            time_end = time.time()
            logger.info(f"{rank}-processed: {index}/{N}, path: ‘{path}’ deal finished, take: {time_end - time_start}s.")
        try:
            self.writer.close()
        except Exception as e:
            logger.info(f"Final close......")


def process_book_wrapper(args):
    rank, workers, max_seq_len, read_bucket, read_data_dir, host_id, host_num, tokenizer_path, save_dir, seed, line_num_perfile, save_book_unit = args
    processor = DataProcessor(
        read_bucket, read_data_dir, tokenizer_path, save_dir, max_seq_len=max_seq_len, seed=seed
    )
    processor.rank = rank
    processor.host_id = host_id
    processor.host_num = host_num
    processor.line_num_perfile = line_num_perfile
    processor.save_book_unit = save_book_unit
    
    processor.file_suffix = '' if save_book_unit else 0

    every_host_nums = math.ceil(len(processor.file_pathlist) / host_num)
    processor.file_pathlist = processor.file_pathlist[host_id * every_host_nums : (host_id + 1) * every_host_nums]
    every_rank_nums = math.ceil(len(processor.file_pathlist) / workers)
    start = int(rank * every_rank_nums)
    end = int((rank + 1) * every_rank_nums)
    processor.run(start, end, rank)
    return rank


def split_bucket_and_dir(path):
    if "gs://" in path:
        subdir = path[5:]
    bucket, subdir = subdir.split("/", maxsplit=1)
    return bucket, subdir + "/" if not subdir.endswith('/') else subdir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data process script")
    parser.add_argument("--read_dir", type=str, help="Data read cloud bucket dir.")
    parser.add_argument("--save_dir", type=str, help="Save model weight file path, it is a dir.")
    parser.add_argument("--host_num", type=int, default=1, help="vm host numbers")
    parser.add_argument("--seed", type=int, default=42, help="Program shuffle seed")
    parser.add_argument("--tokenizer_path", type=str, help="Tokenizer path")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Tokenizer max sequence length")
    parser.add_argument("--workers", type=int, default=10, help="Processed program numbers at same time")
    parser.add_argument("--line_num_perfile", type=int, default=10000, help="Save data numbers per file when length is max_seq_len")
    parser.add_argument("--save_book_unit", action="store_true", default=False, help="whether to save single book unit")

    hostname = socket.gethostname()
    args = parser.parse_args()

    seed = args.seed
    # data_full_dir = 'gs://common_datasets_us-central2/pile/'
    data_full_dir = args.read_dir
    # save_dir = f"pile_seq2048_tokenized/"
    save_dir = args.save_dir
    # tokenizer_path = "EleutherAI/pythia-70m-deduped" || "baichuan-inc/Baichuan2-13B-Base"
    tokenizer_path = args.tokenizer_path
    host_num = args.host_num
    host_id = hostname
    # 基于tpu vm $hostname环境变量传递，但是需要进行处理，得到编号
    if isinstance(host_id, str) and len(host_id) > 5:
        host_id = host_id.rsplit("-", maxsplit=1)[-1]
    host_id = int(host_id)
    host_num = int(host_num)

    assert host_id < host_num
    read_bucket, read_data_dir = split_bucket_and_dir(data_full_dir)
    assert read_data_dir[0] != "/" and read_data_dir[-1] == "/"

    max_seq_len = args.max_seq_len
    line_num_perfile = args.line_num_perfile
    save_book_unit = args.save_book_unit
    workers = args.workers

    set_start_method("spawn")  # tpu-vm

    random.seed(seed)
    num_processes = multiprocessing.cpu_count()
    logger.info(f"num_processes: {num_processes}")
    pool = multiprocessing.Pool(processes=workers)
    args = (
        [rank, workers, max_seq_len, read_bucket, read_data_dir, host_id, host_num, tokenizer_path, save_dir, seed, line_num_perfile, save_book_unit]
        for rank in range(workers)
    )
    logger.info(f'args: {args}')

    results = pool.map(process_book_wrapper, args)  # 包含每个进程的返回值
    pool.close()
    pool.join()

# https://huggingface.co/datasets/ArmelR/the-pile-splitted
# scp
# gcloud compute tpus tpu-vm scp paxml/my_scripts/tokenizer_bucket_json.py llm-jax-v4-64-3:~/  --zone=us-central2-b  --worker=all  --project=llm-tpu
# kill pkill -f multiprocessing.spawn
# gcloud compute tpus tpu-vm ssh llm-jax-v4-64-3 --zone=us-central2-b --worker=all --command="pkill -f 'python tokenizer_bucket_json.py'"
# 用这个才能彻底kill掉
# gcloud compute tpus tpu-vm ssh llm-jax-v4-64-3 --zone=us-central2-b --worker=all --command="pkill -f 'multiprocessing.spawn'"

# run
# gcloud compute tpus tpu-vm ssh llm-jax-v4-64-3 --zone=us-central2-b --worker=all --command="/home/lishengping/miniconda3/bin/python tokenizer_bucket_json.py --read_dir gs://common_datasets_us-central2/pile/ --save_dir pile_seq4096_tokenized/ --tokenizer_path EleutherAI/pythia-70m-deduped --max_seq_len 4096 --host_num 8 --workers 4"
# vm run
# python processed.py --read_dir gs://common_datasets_us-central2/pile/  --save_dir pile_seq4096_tokenized/ --host_num 4  --tokenizer_path EleutherAI/pythia-70m-deduped --max_seq_len 4096 --workers 4