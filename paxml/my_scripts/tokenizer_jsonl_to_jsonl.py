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

from transformers import AutoTokenizer


logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Data_processed')


try:
    import mlxu
    from google.cloud import storage
except:
    command = 'pip install google-cloud-storage && pip install mlxu[gcs]'
    subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    import mlxu
    from google.cloud import storage

    

class DataProcessor:
    def __init__(
        self,
        data_full_dir,
        tokenizer_path,
        save_dir,
        max_seq_len=2048,
        shuffle=True,
        seed=42,
    ):
        bucket = True
        self.shuffle = shuffle
        print("Init tokenizer.....")
        # '/data/pretrained_model/Qwen-14B'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print("Init tokenizer finished.....")
        self.max_seq_len = max_seq_len
        self.save_dir = save_dir
        self.book_index = 0
        self.file_pathlist = []
        self.write_line = 0
        self.clear_threshold_length = 200
        self.book_input_ids = []
        self.data_full_dir = data_full_dir
        self.bos = 1
        self.eos = 0
        self.seed = seed
        self.extract_target_filepath()
        if 'gs::' in self.data_full_dir:
            self.extract_bucket_filepath()
        else:
            self.extract_local_filepath()

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def extract_local_filepath(self):
        # 一定要设定seed，不然不同进程之间的文件顺序可能不一样
        random.seed(self.seed)
        dataset = defaultdict(list)
        print(f'data_full_dir: {self.data_full_dir}')
        jsonl_files = []
        for root, dirs, files in os.walk(self.data_full_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    if 'test' in file:
                        dataset['test'].append(os.path.join(root, file))
                    else:
                        dataset['train'].append(os.path.join(root, file))
        self.file_pathlist = dataset["test"] + dataset["train"]
        if self.shuffle:
            random.shuffle(self.file_pathlist)
        print(f"Final data: {len(self.file_pathlist)}")

    def extract_bucket_filepath(self):
        # 一定要设定seed，不然不同进程之间的文件顺序可能不一样
        random.seed(self.seed)
        dataset = defaultdict(list)
        client = storage.Client()
        bucket = os.path.dirname(self.data_full_dir.rstrip('/'))
        pefix = os.path.basename(self.data_full_dir.rstrip('/')) + '/'
        print(f'bucket: {bucket}, pefix: {pefix}')
        for blob in client.list_blobs(self.bucket, prefix=pefix):
            path = os.path.join(f"gs://{bucket}", blob.name)
            if not blob.name:
                continue
            if path in self.target_file_pathlist:
                continue
            if "test" in blob.name:
                dataset["test"].append(path)
            else:
                dataset["train"].append(path)
        self.file_pathlist = dataset["test"] + dataset["train"]
        if self.shuffle:
            random.shuffle(self.file_pathlist)
        logger.info(f"Final data: {len(self.file_pathlist)}")

    def extract_target_filepath(self):
        dataset = defaultdict(list)
        files = os.listdir(self.save_dir)
        print(f'save_dir: {save_dir}')
        for file in files:
            path = os.path.join(self.save_dir, file)
            if "test" in path:
                dataset["test"].append(path)
            else:
                dataset["train"].append(path)
        self.target_file_pathlist = set(dataset["test"] + dataset["train"])

    def convert_line_to_ids(self, line):
        return self.tokenizer.encode(line)

    def write_file(self, save_path, ids, target_ids):
        if self.write_line == 0:
            path = f"{save_path}{self.file_suffix}"
            self.writer = open(path, 'w')
        target_length = len(target_ids)
        target_masks = target_ids
        input_length = self.max_seq_len - target_length
        input_ids = ids[ : input_length]
        input_masks = len(input_ids) * [-100]

        input_target_ids = input_ids + target_ids
        input_target_masks = input_masks + target_masks

        assert len(input_target_ids) == len(input_target_masks) <= self.max_seq_len

        write_data = {'input_ids': input_target_ids, 'labels': input_target_masks}
        self.writer.write(f'{write_data}\n')
        self.write_line += 1
        if self.write_line == self.line_num_perfile and not self.save_book_unit:
            self.file_suffix += 1
            self.writer.close()
            self.write_line = 0

    def process_book(self, path):
        print(f'path: {path}')
        save_path = path.replace(self.data_full_dir, self.save_dir)
        print(f"Start to process file path: {save_path}")
        with open(path) as fr:
            for i, line in enumerate(fr):
                line = json.loads(line)
                text = line["text"]
                target = line['target']
                if text and text.strip():
                    ids = self.convert_line_to_ids(text)
                else:
                    ids = []
                if target.strip():
                    target_ids = self.convert_line_to_ids(target)
                else:
                    continue
                if len(target_ids) > self.max_seq_len:
                    continue
                target_ids += [self.eos]
                self.write_file(save_path, ids, target_ids)

        self.book_index += 1
        if self.save_book_unit:
            self.writer.close()
            self.write_line = 0

    def __len__(self):
        return self.size

    def size(self):
        return len(self.file_pathlist)

    def run(self, start, end, rank):
        N = end - start
        time_start = time.time()
        print(f"Rank: {rank}|| book: {len(self.file_pathlist)} || start: {start} || end: {end}")
        for index, path in enumerate(self.file_pathlist[start:end]):
            self.process_book(path)
            time_end = time.time()
            print(f"{rank}-processed: {index}/{N}, path: ‘{path}’ deal finished, take: {time_end - time_start}s.")
        try:
            self.writer.close()
        except Exception as e:
            print(f"Final close......")


def process_book_wrapper(args):
    rank, workers, max_seq_len, data_full_dir, host_id, host_num, tokenizer_path, save_dir, seed, line_num_perfile, save_book_unit = args
    processor = DataProcessor(
        data_full_dir, tokenizer_path, save_dir, max_seq_len=max_seq_len, seed=seed
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data process script")
    parser.add_argument("--read_dir", type=str, help="Data read cloud bucket dir.")
    parser.add_argument("--save_dir", type=str, help="Save model weight file path, it is a dir.")
    parser.add_argument("--host_num", type=int, default=1, help="vm host numbers")
    parser.add_argument("--host_id", type=int, default=0, help="vm host id")
    parser.add_argument("--seed", type=int, default=42, help="Program shuffle seed")
    parser.add_argument("--tokenizer_path", type=str, help="Tokenizer path")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Tokenizer max sequence length")
    parser.add_argument("--workers", type=int, default=10, help="Processed program numbers at same time")
    parser.add_argument("--line_num_perfile", type=int, default=10000, help="Save data numbers per file when length is max_seq_len")
    parser.add_argument("--save_book_unit", action="store_true", default=False, help="whether to save single book unit")

    args = parser.parse_args()

    seed = args.seed
    # data_full_dir = 'gs://common_datasets_us-central2/pile/'
    data_full_dir = args.read_dir
    # save_dir = f"pile_seq2048_tokenized/"
    save_dir = args.save_dir
    # tokenizer_path = "EleutherAI/pythia-70m-deduped" || "baichuan-inc/Baichuan2-13B-Base"
    tokenizer_path = args.tokenizer_path
    host_num = args.host_num
    host_id = args.host_id
    if isinstance(host_id, str) and len(host_id) > 5:
        host_id = host_id.rsplit("-", maxsplit=1)[-1]
    host_id = int(host_id)
    host_num = int(host_num)
    assert host_id < host_num

    max_seq_len = args.max_seq_len
    line_num_perfile = args.line_num_perfile
    save_book_unit = args.save_book_unit
    workers = args.workers

    #set_start_method("spawn")  # tpu-vm
    random.seed(seed)
    num_processes = multiprocessing.cpu_count()
    print(f"num_processes: {num_processes}")
    pool = multiprocessing.Pool(processes=workers)

    args = ([rank, workers, max_seq_len, data_full_dir, host_id, host_num, tokenizer_path, save_dir, seed, line_num_perfile, save_book_unit] for rank in range(workers))
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
# python processed.py --read_dir xiaomeng1020/  --save_dir xiaomeng_seq2048_1020/ --host_num 1  --tokenizer_path Qwen/Qwen-14B --max_seq_len 2048 --workers 2

# /home/work/Application/python310/bin/python processed2.py --read_dir /home/work/temp  --save_dir /home/work/temp/processed --host_num 1  --tokenizer_path Qwen/Qwen-14B --max_seq_len 2048 --workers 2