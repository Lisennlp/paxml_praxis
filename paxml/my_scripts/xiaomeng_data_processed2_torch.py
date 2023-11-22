import time
import os
import sys
import json
import random
import multiprocessing
from multiprocessing import set_start_method
import socket
from collections import defaultdict

os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
from transformers import AutoTokenizer
import mlxu
import math


import re


# null_content = re.compile(
#     "Qidian|Novel (name|status|words|category)|书友群|广大书友|求推荐票|---分頁---|感谢.*(打赏|支持)|手机用户请到阅读|抱歉，更的晚|（群号|三更.{,2}第.更|推荐票|&amp;&amp;&amp;&amp"
# )
# poison_content = re.compile(r'本章完|第(\d|[零一二三四五六七八九十百千]){1,}(章|节|卷)|(^\d{1,5}$)|未 ?完待续|(^\d{1,5}\.)')

novel_name_pat = re.compile('Novel name')
novel_category_pat = re.compile('Novel category')
ahthor_pat = re.compile(
    "Qidian|Novel (status|words)|书友群|广大书友|求推荐票|-分[頁页]-|感谢.*(打赏|支持)|手机用户请到阅读|抱歉，更的晚|（群号|三更.{,2}第.更|推荐票|&amp;&amp;&amp;&amp|分割线"
)
poison_content = re.compile(r'(^-?\d{1,5}$)|未 ?完待续')

def match_name_category(line):
    if not novel_name_pat.match(line):
        line = 'Novel name: \n'
    if not novel_category_pat.match(line):
        line = 'Novel category: \n'
    return line

def match_unused_content(line):
    if poison_content.match(line) or ahthor_pat.search(line):
        return True
    else:
        return False

class DataProcessor:
    def __init__(
        self,
        data_pathfile,
        tokenizer,
        save_dir,
        data_type="zh",
        max_seq_len=2048,
        ratio=1.0,
        shuffle=True,
        epoches=1000,
    ):
        bucket = True
        if bucket:
            self.path_map = {
                "zh": ["/mnt/nvme1/kf/data/69shuba", "gs://jax_llm_data/xiaomeng/zh_data"],
                "en": ["/mnt/nvme1/kf/data/formal_data", "gs://jax_llm_data/xiaomeng/en_data"],
            }
        else:
            self.path_map = {
                "zh": ["/mnt/nvme1/kf/data/69shuba", "/nas2/xiaomeng/zh_data"],
                "en": ["/mnt/nvme1/kf/data/formal_data", "/nas2/xiaomeng/en_data"],
            }
        self.data_pathfile = data_pathfile
        self.data_type = data_type
        self.shuffle = shuffle
        self.epoches = epoches
        self.ratio = min(ratio, 1)
        assert self.ratio > 0
        print("Init tokenizer.....")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, use_fast=False, trust_remote_code=True
        )
        print("Init tokenizer finished.....")
        self.max_seq_len = max_seq_len
        self.save_dir = save_dir
        self.books_pathlist = []
        self.write_line = 0
        self.writer = None
        self.clear_threshold_length = 500
        self.extract_filepath()
        self.file_count = 0
        self.book_input_ids = []
        self.book_end_id = self.tokenizer.encode('<|endoftext|>\n')
        self.segment_num = {'zh': 2, 'en': 0} # 每本书取前多少条数据作为一次迭代


    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def extract_filepath(self):
        random.seed(42)
        self.books_pathlist = [
            line.strip().replace(*self.path_map[self.data_type])
            for line in mlxu.open_file(self.data_pathfile, 'r').readlines()
        ]
        if self.shuffle:
            self.books_pathlist = random.sample(
                self.books_pathlist, k=int(len(self.books_pathlist) * self.ratio)
            )

    def convert_line_to_ids(self, line):
        return self.tokenizer.encode(line)

    def process_book(self, path):
        with mlxu.open_file(path, 'r') as fr:
            lines = fr.readlines()
            for index in range(0, len(lines), 1):
                line = lines[index]
                line = line.strip()
                self.line = line
                if not line:
                    line = '\n'
                else:
                    line += '\n'
                if self.data_type == 'zh' and match_unused_content(line):
                    continue
                ids = self.convert_line_to_ids(line)
                self.book_input_ids.extend(ids)
            self.book_input_ids.extend(self.book_end_id)
            
    def __len__(self):
        return self.size

    def size(self):
        return len(self.books_pathlist)

    def run(self, start, end, rank):
        random.seed(42)
        time_start = time.time()
        self.processor_bookes = self.books_pathlist[start: end]
        length = len(self.processor_bookes)
        record_index = []
        for index, path in enumerate(self.processor_bookes):
            filename = os.path.basename(path)
            record_index.append(len(self.book_input_ids))
            self.process_book(path, start_index)
            record_index.append(len(self.book_input_ids))
            assert (index + 1) * 2 == len(record_index)
            time_end = time.time()
            print(f"Rank: {rank} index: {index}/{length} take: {time_end - time_start}")


def process_book_wrapper(args):
    rank, LANG, WORKERS, max_seq_len, ratio = args
    bucket = True
    if bucket:
        data_pathfiles = {
            "zh": "gs://jax_llm_data/xiaomeng/zh_data/69shuba.filelist.shuffled",
            "en": "gs://jax_llm_data/xiaomeng/en_data/allfile.filelist.shuffed",
        }
    else:
        data_pathfiles = {
            "zh": "/nas2/xiaomeng/zh_data/69shuba.filelist.shuffled",
            "en": "/nas2/xiaomeng/en_data/allfile.filelist.shuffed",
        }
   # if not os.path.exists(tokenizer_path):
        # tokenizer_path = "baichuan-inc/Baichuan2-13B-Base"
    tokenizer_path = "Qwen/Qwen-14B"
    if bucket:
        save_dir = f"gs://jax_llm_data/xiaomeng/processed_{LANG}_data_qwen14B_KeepChapter1117/"
    else:
        raise ValueError(f'Now version only support bucket is True')

    processor = DataProcessor(
        data_pathfiles[LANG],
        tokenizer_path,
        save_dir,
        data_type=LANG,
        max_seq_len=max_seq_len,
        ratio=ratio,
    )
    processor.rank = rank
    processor.per_file_line_num = 50000
    every_rank_nums = math.ceil(len(processor.books_pathlist) / WORKERS)
    start = int(rank * every_rank_nums)
    end = int((rank + 1) * every_rank_nums)
    print(f"Rank: {rank} start: {start} end: {end}")
    processor.run(start, end, rank)
    return rank


if __name__ == "__main__":
    random.seed(42)
    set_start_method("spawn")  # tpu-vm
    num_processes = multiprocessing.cpu_count()
    print(f"num_processes: {num_processes}")

    WORKERS = int(sys.argv[1])
    host_num = int(sys.argv[2])
    data_dtype = sys.argv[3]


    hostname = socket.gethostname()
    host_id = hostname
    if isinstance(host_id, str) and len(host_id) > 5:
        host_id = host_id.rsplit("-", maxsplit=1)[-1]
    host_id = int(host_id)
    host_num = int(host_num)

    print(f'data_dtype: {data_dtype}')

    if data_dtype == 'en':
        if host_id < 10:
            exit(0)
        else:
            host_id -= 10
    else:
        if host_id > 10:
            exit(0)
        else:
            pass
        
    workers_perhost = WORKERS // host_num

    worker_start = host_id * workers_perhost
    worker_end = (host_id + 1) * workers_perhost

    print(f'WORKERS: {WORKERS} worker_start: {worker_start} worker_end: {worker_end}')
    if worker_start >= WORKERS:
        exit(0)

    ratio = 1.0
    max_seq_len = 4097
    pool = multiprocessing.Pool(processes=WORKERS)
    args = (
        [rank, LANG, WORKERS, max_seq_len, ratio]
        for rank in range(worker_start, worker_end, 1)
        for LANG in [data_dtype]
    )
    results = pool.map(process_book_wrapper, args)  # 包含每个进程的返回值
    pool.close()
    pool.join()
