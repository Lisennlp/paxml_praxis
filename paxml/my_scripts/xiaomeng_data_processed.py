import time
import os
import random
import multiprocessing
from multiprocessing import set_start_method

os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
from transformers import AutoTokenizer
from smart_open import open
import wandb


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
        self.ratio = min(ratio, 1)
        assert self.ratio > 0
        print("Init tokenizer.....")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, use_fast=False, trust_remote_code=True
        )
        print("Init tokenizer finished.....")
        self.max_seq_len = max_seq_len
        self.save_dir = save_dir
        self.book_index = 0
        self.books_pathlist = []
        self.write_line = 0
        self.writer = None
        self.clear_threshold_length = 500
        self.extract_filepath()
        self.split_map_dir = {'0': '_start', '1': '_median', '2': '_end'}
        self.file_count = 0

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def extract_filepath(self):
        self.books_pathlist = [
            line.strip().replace(*self.path_map[self.data_type])
            for line in open(self.data_pathfile).readlines()
        ]
        if self.shuffle:
            self.books_pathlist = random.sample(
                self.books_pathlist, k=int(len(self.books_pathlist) * self.ratio)
            )

    def convert_line_to_ids(self, line):
        return self.tokenizer.encode(line)

    def writer_factory(self, split_rank="1"):
        # print(f'Rank: {self.rank}, write_line: {self.write_line}')
        if self.write_line == self.per_file_line_num:
            self.writer.close()
            self.write_line = 0
        if self.write_line == 0:
            name = f"{self.data_type}_split{split_rank}_rank{self.rank}_count{self.file_count}"
            # save_dir = self.save_dir.replace('_split', self.split_map_dir[split_rank])
            # save_path = os.path.join(save_dir, name)
            save_path = os.path.join(self.save_dir, name)
            self.writer = tf.io.TFRecordWriter(save_path)
            print(f"Rank: {self.rank}, save_path: {save_path}")
            self.file_count += 1

    def write_file(self, book_input_ids):
        input_ids = book_input_ids[: self.max_seq_len]
        if len(input_ids) < self.clear_threshold_length:
            return []
        feature = {
            "input_ids": self._int64_feature(input_ids),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())
        self.write_line += 1
        book_input_ids = book_input_ids[self.max_seq_len :]
        return book_input_ids

    def process_book(self, path):
        book_input_ids = []
        with open(path) as fr:
            lines = fr.readlines()
            chunk_size = len(lines) // self.chunks
            for i in range(self.chunks):
                split_rank = str(i).zfill(len(str(self.chunks)))
                split_lines = lines[i * chunk_size : (i + 1) * chunk_size]
                for i, line in enumerate(split_lines):
                    ids = self.convert_line_to_ids(line)
                    book_input_ids.extend(ids)
                    if len(book_input_ids) > self.max_seq_len:
                        self.writer_factory(split_rank=split_rank)
                        book_input_ids = self.write_file(book_input_ids)

                if len(book_input_ids):
                    self.writer_factory(split_rank=split_rank)
                    book_input_ids = self.write_file(book_input_ids)
            self.book_index += 1

    def __len__(self):
        return self.size

    def size(self):
        return len(self.books_pathlist)

    def run(self, start, end, rank):
        N = end - start
        time_start = time.time()
        print(f"Rank: {rank}. book: {len(self.books_pathlist)}")
        for index, path in enumerate(self.books_pathlist[start:end]):
            print(f"Rank: {rank} index: {index}")
            try:
                self.process_book(path)
            except Exception as e:
                print(f'Rank: {rank}, error: {e}')
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
    rank, LANG, WORKERS, max_seq_len, ratio, chunks = args
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
    tokenizer_path = "/nas2/lishengping/other_models/baichuan2-13b-base"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "baichuan-inc/Baichuan2-13B-Base"
    if bucket:
        save_dir = f"gs://jax_llm_data/xiaomeng/processed_{LANG}_data_split/"
    else:
        save_dir = f"/nas2/lishengping/other_models/baichuan2-13b-base/processed_{LANG}_data/"

    processor = DataProcessor(
        data_pathfiles[LANG],
        tokenizer_path,
        save_dir,
        data_type=LANG,
        max_seq_len=max_seq_len,
        ratio=ratio,
    )
    processor.rank = rank
    processor.chunks = chunks
    processor.per_file_line_num = 10000
    every_rank_nums = int(len(processor.books_pathlist) // WORKERS)
    if rank == 0:
        wandb.login(key="7988c805dfe3fed4d6e4017f616555a5160fd2c2")
        wandb.init(project="xiaomeng_test", name="data_processed", config=None, resume=True)
    start = int(rank * every_rank_nums)
    end = int((rank + 1) * every_rank_nums)
    print(f"Rank: {rank} start: {start} end: {end}")
    processor.run(start, end, rank)
    return rank


if __name__ == "__main__":
    random.seed(42)
    set_start_method("spawn")  # tpu-vm
    tokenizer_path = "/nas2/lishengping/other_models/baichuan2-13b-base"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "baichuan-inc/Baichuan2-13B-Base"
    #     os.makedirs(save_dir, exist_ok=True)
    num_processes = multiprocessing.cpu_count()
    print(f"num_processes: {num_processes}")
    WORKERS = 240
    ratio = 1.0
    max_seq_len = 4096
    chunks = 3
    pool = multiprocessing.Pool(processes=WORKERS)
    args = (
        [rank, LANG, WORKERS, max_seq_len, ratio, chunks]
        for rank in range(WORKERS)
        for LANG in ["en", "zh"]
    )
    results = pool.map(process_book_wrapper, args)  # 包含每个进程的返回值
    pool.close()
    pool.join()
