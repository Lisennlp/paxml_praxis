import time
import os
import sys
import json
import random
import math
import multiprocessing
from multiprocessing import set_start_method
from collections import defaultdict
import subprocess

os.environ["JAX_PLATFORMS"] = "cpu"

import smart_open
import tensorflow as tf
import wandb
from google.cloud import storage
from transformers import AutoTokenizer




def check_bucket_exist(filepath):
    # filepath = 'gs://common_datasets_us-central2/pile_seq1024_tokenized/OpenSubtitles/data-00000-of-00096.test.tfrecord'
    command = f'gsutil stat {filepath}'
    response = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    if response.returncode == 0:
        return True
    else:
        return False

class DataProcessor:
    def __init__(
        self,
        read_bucket,
        dataset_name,
        tokenizer_path,
        save_dir,
        max_seq_len=2048,
        shuffle=True,
        debug=False,
    ):
        bucket = True
        self.shuffle = shuffle
        print("Init tokenizer.....")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision="step3000", cache_dir="./pythia-70m-deduped/step3000",)

        print("Init tokenizer finished.....")
        self.max_seq_len = max_seq_len
        self.save_dir = save_dir
        self.book_index = 0
        self.file_pathlist = []
        self.write_line = 0
        self.writer = None
        self.clear_threshold_length = 200
        self.split_map_dir = {'0': '_start', '1': '_median', '2': '_end'}
        self.file_count = 0
        self.book_input_ids = []
        self.read_bucket = read_bucket
        self.dataset_name = dataset_name
        self.bos = 1
        self.eos = 0
        self.debug = debug
        self.extract_target_filepath()
        self.extract_filepath()

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def extract_filepath(self):
        dataset = defaultdict(list)
        client = storage.Client()
        for blob in client.list_blobs(self.read_bucket, prefix=self.dataset_name):
            path = os.path.join(f"gs://{self.read_bucket}", blob.name)
            if not blob.name:
                continue
            if path in self.target_file_pathlist:
                continue
            if 'test' in blob.name:
                dataset['test'].append(path)
            else:
                dataset['train'].append(path)
            print(f'filename: {blob.name}===== total train: {len(dataset["train"])}; test: {len(dataset["test"])}')

        self.file_pathlist = dataset["test"] + dataset["train"]
        if self.shuffle:
            random.shuffle(self.file_pathlist)
        if self.debug:
            self.file_pathlist = self.file_pathlist[:10]
        print(f'Final data: {len(self.file_pathlist)}')

    def extract_target_filepath(self):
        dataset = defaultdict(list)
        client = storage.Client()
        for blob in client.list_blobs(self.read_bucket, prefix=self.save_dir):
            path = os.path.join(f"gs://{self.read_bucket}", blob.name)
            path = path.replace(self.save_dir, self.dataset_name).replace('tfrecord', 'jsonl')
            if not blob.name:
                continue
            if 'test' in blob.name:
                dataset['test'].append(path)
            else:
                dataset['train'].append(path)
        self.target_file_pathlist = dataset["test"] + dataset["train"]


    def convert_line_to_ids(self, line):
        return self.tokenizer.encode(line) + [self.eos]

    def write_file(self):
        length = len(self.book_input_ids)
        start = 0
        while start < length - 1:
            input_ids = self.book_input_ids[start: start + self.max_seq_len - 2]
            start += (self.max_seq_len - 2)
            input_ids_length = len(input_ids)
            if input_ids_length < self.max_seq_len - 2:
                self.book_input_ids = input_ids
                flag = 1
            else:
                input_ids = [self.bos] + input_ids + [self.eos]
                feature = {"input_ids": self._int64_feature(input_ids)}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                self.writer.write(example.SerializeToString())
                flag = 0
        length = len(self.book_input_ids)
        if length < self.clear_threshold_length or not flag:
            self.book_input_ids = []
        assert len(self.book_input_ids) < self.max_seq_len, print(f'length: {len(self.book_input_ids)}')


    def process_book(self, path):
        # name = os.path.basename(path).rsplit('.')[0] + '.tf'
        save_path = path.replace(self.dataset_name, self.save_dir).replace('jsonl', 'tfrecord')
        print(f'Processed file save path: {save_path}')
        self.writer = tf.io.TFRecordWriter(save_path)
        with smart_open.open(path) as fr:
            for i, line in enumerate(fr):
                line = json.loads(line)
                text = line['text']
                ids = self.convert_line_to_ids(text)
                self.book_input_ids.extend(ids)
                if len(self.book_input_ids) >= self.max_seq_len - 2:
                    self.write_file()
        self.writer.close()
        self.book_index += 1

    def __len__(self):
        return self.size

    def size(self):
        return len(self.file_pathlist)

    def run(self, start, end, rank):
        N = end - start
        time_start = time.time()
        print(f"Rank: {rank}. book: {len(self.file_pathlist)}")
        for index, path in enumerate(self.file_pathlist[start:end]):
            print(f"Rank: {rank} index: {index}")
            try:
                self.process_book(path)
            except Exception as e:
                print(f'Rank: {rank}, error: {e} path: {path}')
            time_end = time.time()
            print(
                f"{rank}-processed: {index}/{N}, path: ‘{path}’ deal finished, take:"
                f" {time_end - time_start}."
            )
            if rank == 0:
                wandb_stats = {"index": index, "N": N, "take": time_end - time_start}
                wandb.log(wandb_stats)
        try:
            self.writer.close()
        except Exception as e:
            print(f'Final close......')


def process_book_wrapper(args):
    rank, WORKERS, max_seq_len, read_bucket, dataset_name, host_id, host_num = args
    bucket = True
    tokenizer_path = "/nas2/lishengping/other_models/baichuan2-13b-base"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "EleutherAI/pythia-70m-deduped"
    if bucket:
        save_dir = f'pile_seq1024_tokenized/'
    else:
        save_dir = f"pile_processed_seq1024/"
    debug = False
    processor = DataProcessor(
        read_bucket,
        dataset_name,
        tokenizer_path,
        save_dir,
        max_seq_len=max_seq_len,
        debug=debug
    )
    host_num = int(host_num)
    host_id = int(host_id)
    processor.rank = rank
    processor.host_id = host_id
    processor.host_num = host_num
    every_host_nums = math.ceil(len(processor.file_pathlist) / host_num)
    processor.file_pathlist = processor.file_pathlist[host_id *  every_host_nums :(host_id + 1) * every_host_nums]

    every_rank_nums = math.ceil(len(processor.file_pathlist) / WORKERS)
    if rank == 0:
        wandb.login(key="7988c805dfe3fed4d6e4017f616555a5160fd2c2")
        wandb.init(project="pile_test", name="data_processed", config=None, resume=True)
    start = int(rank * every_rank_nums)
    end = int((rank + 1) * every_rank_nums)
    print(f"Rank: {rank} start: {start} end: {end}")
    processor.run(start, end, rank)
    return rank


if __name__ == "__main__":
    random.seed(42)
    host_num = sys.argv[1]
    host_id = sys.argv[2]

    set_start_method("spawn")  # tpu-vm

    read_bucket = 'common_datasets_us-central2'
    # 文件夹的prefix
    dataset_name = 'pile/'
    num_processes = multiprocessing.cpu_count()
    print(f"num_processes: {num_processes}")
    WORKERS = 200
    max_seq_len = 1024
    # host_id = 0
    pool = multiprocessing.Pool(processes=WORKERS)
    args = (
        [rank, WORKERS, max_seq_len, read_bucket, dataset_name, host_id, host_num]
        for rank in range(WORKERS)
    )
    results = pool.map(process_book_wrapper, args)  # 包含每个进程的返回值
    pool.close()
    pool.join()

# https://huggingface.co/datasets/ArmelR/the-pile-splitted
# Usage: using multi-host and multi-processe to deal data.
# python processed.py 4 0/1/2/3
# packages:
# pip install google-cloud-storage
# pip install smart_open[gcs]
# pip install wandb
# wandb disabled
# vi processed.py