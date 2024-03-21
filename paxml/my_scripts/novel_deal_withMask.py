import time
import os
import sys
import random
import multiprocessing
from multiprocessing import set_start_method
import socket

os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
from transformers import AutoTokenizer
from etils import epath
import math
from tqdm import tqdm

import re
from datetime import datetime
from collections import deque



# null_content = re.compile(
#     "Qidian|Novel (name|status|words|category)|书友群|广大书友|求推荐票|---分頁---|感谢.*(打赏|支持)|手机用户请到阅读|抱歉，更的晚|（群号|三更.{,2}第.更|推荐票|&amp;&amp;&amp;&amp"
# )
# poison_content = re.compile(r'本章完|第(\d|[零一二三四五六七八九十百千]){1,}(章|节|卷)|(^\d{1,5}$)|未 ?完待续|(^\d{1,5}\.)')
# chapter_pat = re.compile('第(\d|[零一二三四五六七八九十百千]){1,}(章|节|卷)')
chapter_pat = re.compile('第(\d|[零一二三四五六七八九十百千]){1,}(章|节|卷|回)|^【\d+】|^\d+\.|^0\d+')

chapter_en_pat = re.compile('Chapter ?\d+|^【\d+】|^\d+\.|^0\d+')

novel_name_pat = re.compile('Novel name')
novel_category_pat = re.compile('Novel category')
ahthor_pat = re.compile(
    "Qidian|Novel (status|words)|书友群|广大书友|求推荐票|-分[頁页]-|感谢.*(打赏|支持)|手机用户请到阅读|抱歉，更的晚|（群号|三更.{,2}第.更|推荐票|&amp;&amp;&amp;&amp|分割线|&[lg]t\;"
)
poison_content = re.compile(r'未 ?完待续|本章完')
chapter_digit = re.compile('(^-?\d{1,6}$)')


def match_name_category(line):
    if not novel_name_pat.match(line):
        line = 'Novel name: \n'
    if not novel_category_pat.match(line):
        line = 'Novel category: \n'
    return line


def match_chapter(line):
    if chapter_pat.search(line):
        return True
    return False


def match_en_chapter(line):
    if chapter_en_pat.search(line):
        return True
    return False


def match_digit_chapter(line):
    if chapter_digit.match(line):
         return True
    return False


def match_unused_content(line):
    if poison_content.search(line) or ahthor_pat.search(line):
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
        epoches=100,
    ):
        bucket = True
        if bucket:
            self.path_map = {
                "zh": ["/mnt/nvme1/kf/data/69shuba", "gs://jax_llm_data_us-east5/xiaomeng/zh_data"],
                "en": ["/mnt/nvme1/kf/data/formal_data", "gs://jax_llm_data_us-east5/xiaomeng/en_data"],
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
        self.extract_filepath()
        self.file_count = 0
        self.book_input_ids = []
        self.book_masks = []
        self.book_end_id = self.tokenizer.encode('<|endoftext|>\n')
        self.segment_num = {'zh': 2, 'en': 0} # 每本书取前多少条数据作为一次迭代
        self.last_n_lines = deque(maxlen=3)

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def extract_filepath(self):
        random.seed(42)
        self.books_pathlist = [
            line.strip().replace(*self.path_map[self.data_type])
            for line in epath.Path(self.data_pathfile).open('r').readlines()
        ]
        if self.shuffle:
            self.books_pathlist = random.sample(
                self.books_pathlist, k=int(len(self.books_pathlist) * self.ratio)
            )

    def convert_line_to_ids(self, line):
        return self.tokenizer.encode(line)

    def writer_factory(self):
        if self.write_line % self.per_file_line_num == 0:
            if self.write_line > 0: self.writer.close()
            name = f"{self.data_type}_R{self.rank}_E{self.epoch}_b{self.write_line}"
            save_path = os.path.join(self.save_dir, name)
            self.writer = tf.io.TFRecordWriter(save_path)
            self.file_count += 1

    def write_file(self):
        input_ids = self.book_input_ids[: self.max_seq_len]
        masks = self.book_masks[: self.max_seq_len]

        assert len(input_ids) == len(masks)

        feature = {
            "input_ids": self._int64_feature(input_ids),
            "labels": self._int64_feature(masks),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())
        self.write_line += 1
        # 每次都从句子开始
        self.book_input_ids = []
        self.book_masks = []

    def process_book(self, path, start_index=0):
        with epath.Path(path).open('r') as fr:
            lines = fr.readlines()
            # 去掉很短的书
            if len(lines) < 200:
                print(f'Short book: {path}')
                self.run_book_index.pop(path)
                return 
            count = 0
            for index in range(start_index, len(lines), 1):
                line = lines[index].strip()
                if not line or match_digit_chapter(line): continue
                clear_space_line = line.replace(' ', '')
                if self.last_n_lines.count(clear_space_line) >= 2:
                    print(f'重复: {clear_space_line} path: {path}')
                    continue
                self.last_n_lines.append(clear_space_line)
                line += '\n'
                if self.data_type == 'zh' and match_unused_content(line):
                    continue
                ids = self.convert_line_to_ids(line)
                if (self.data_type == 'zh' and match_chapter(line)) or (self.data_type == 'en' and match_en_chapter(line)) :
                    masks = len(ids) * [0]
                else:
                    masks = len(ids) * [1]
                    
                if index == len(lines) - 1:
                    ids.extend(self.book_end_id)
                    masks.extend([1] * len(self.book_end_id))
                self.book_input_ids.extend(ids)
                self.book_masks.extend(masks)

                if len(self.book_input_ids) >= self.max_seq_len:
                    self.writer_factory()
                    self.write_file()
                    if count >= self.segment_num[self.data_type]:
                        self.run_book_index[path] = index + 1
                        break
                    count += 1
            if index == len(lines) - 1:
                # 换书之后清空
                self.book_input_ids = []
                self.book_masks = []
                self.run_book_index.pop(path) # 结束后，删除该书
            

    def __len__(self):
        return self.size

    def size(self):
        return len(self.books_pathlist)

    def run(self, start, end, rank):
        random.seed(42)
        time_start = time.time()
        print(f"Rank: {rank}. book: {len(self.books_pathlist)} self.epoches: {self.epoches}")
        self.processor_bookes = self.books_pathlist[start: end]
        self.run_book_index = {book: 0 for book in self.processor_bookes}
        for epoch in tqdm(range(self.epoches), desc=f'Rank-{rank}'):
            N = len(self.run_book_index)
            if N == 0: break
            self.epoch = epoch
            # 前1轮的时候，可以允许数据不写满per_file_line_num
            if self.epoch == 1:
                self.write_line = 0
                self.writer.close()
            items = list(self.run_book_index.items())
            #每轮都shuffle一遍
            random.shuffle(items)
            self.run_book_index = dict(items)
            for path, start_index in tqdm(self.run_book_index.copy().items(), desc=f'Processing-epoch{epoch}-{rank}'):
                self.process_book(path, start_index)
        try:
            self.writer.close()
        except Exception as e:
            pass

def process_book_wrapper(args):
    rank, LANG, WORKERS, max_seq_len, ratio = args
    bucket = True
    if bucket:
        data_pathfiles = {
            "zh": "gs://jax_llm_data_us-east5/xiaomeng/zh_data/69shuba.filelist.shuffled",
            "en": "gs://jax_llm_data_us-east5/xiaomeng/en_data/allfile.filelist.shuffed",
        }
    else:
        data_pathfiles = {
            "zh": "/nas2/xiaomeng/zh_data/69shuba.filelist.shuffled",
            "en": "/nas2/xiaomeng/en_data/allfile.filelist.shuffed",
        }
    tokenizer_path = "Qwen/Qwen-14B"
    # tokenizer_path = "baichuan-inc/Baichuan2-13B-Base"

    model_name = os.path.basename(tokenizer_path)
    today = datetime.today()
    formatted_date = today.strftime("%m%d")
    if bucket:
        save_dir = f"gs://jax_llm_data_us-east5/xiaomeng/{LANG}_data_{model_name}_{formatted_date}"
    else:
        raise ValueError(f'Now version only support bucket is True')
    if rank == 0:
        print(f'save_dir: {save_dir}')
    processor = DataProcessor(
        data_pathfiles[LANG],
        tokenizer_path,
        save_dir,
        data_type=LANG,
        max_seq_len=max_seq_len,
        ratio=ratio,
    )
    processor.rank = rank
    processor.per_file_line_num = 10000
    processor.epoches = 1000
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

    mid_host = 4

    if data_dtype == 'en':
        if host_id < mid_host:
            exit(0)
        else:
            host_id -= mid_host
    else:
        if host_id >= mid_host:
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

# Usage:
# TPU_NAME=llm-jax-v5p-8-10; ZONE=us-east5-a; WORKERS=200; HOST_NUM=1; DATA_TYPE='zh'
# TPU_NAME=llm-jax-v4-256-0; ZONE=us-central2-b; WORKERS=200; HOST_NUM=5; DATA_TYPE='zh'
# TPU_NAME=llm-jax-v3-128-10; ZONE=us-east1-d; WORKERS=180; HOST_NUM=4; DATA_TYPE='en'
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install tiktoken"
# SCRIPT=/Users/lishengping/codes/jax_projects/paxml_praxis/paxml/my_scripts/xiaomeng_data_tokenize_WithMask.py
# gcloud compute tpus tpu-vm scp $SCRIPT $TPU_NAME:/home/lishengping/processed.py  --zone=$ZONE  --worker=all  --project=llm-tpu
# TPU_NAME=llm-jax-v3-128-10; ZONE=us-east1-d; WORKERS=180; HOST_NUM=4; DATA_TYPE='zh'
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py $WORKERS $HOST_NUM $DATA_TYPE| tee $DATA_TYPE_processed.log"