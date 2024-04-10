from google.cloud import storage
import io
import multiprocessing
import smart_open
from multiprocessing import Pool
from functools import partial
import tensorflow as tf
from transformers import AutoTokenizer
import os

class QwenTokenizer():
    def __init__(self, tokenizer_path, max_len, bos_id:list=[], eos_id: list=[]):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=False, trust_remote_code=True
        )
        self.next_ids = []
        self.partial_tokenize = partial(self.tokenize, max_len=max_len, bos_id=bos_id, eos_id=eos_id)
    
    @classmethod
    def _int64_feature(cls, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def write_to_tfrecord(self, input_ids):
        return 
        feature = {
            "input_ids": self._int64_feature(input_ids),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())
        
    @classmethod
    def safe_readline(cls, f):
        pos = f.tell()
        while True:
            try:
                return f.readline()
    
            except UnicodeDecodeError:
                pos -= 1
                f.seek(pos)
    
    def tokenize(self, text, max_len=2048, bos_id:list=[], eos_id:list=[]):
        input_ids = self.tokenizer.encode(text) + eos_id
        if bos_id:
            max_len -= 1
        self.next_ids += input_ids #  加上上个step保留的id
        total_ids = []
        while len(self.next_ids) >= max_len:
            save_ids = self.next_ids[: max_len]
            if len(save_ids) == max_len:
                save_ids = bos_id + save_ids
                self.write_to_tfrecord(save_ids)
                total_ids.append(save_ids)
                self.next_ids = self.next_ids[max_len: ]
            else:
                self.next_ids = save_ids
                save_ids = []
        return total_ids
    
    
    def async_tokenizer(self, path, i, workers, file_size):
        chunk_size = file_size // workers
        start = i * chunk_size
        end = file_size if i == workers - 1 else (i + 1) * chunk_size
        work_input_ids = []
        print(f'start: {start} end: {end}')
        with smart_open.open(path, 'r') as file:
            file.seek(start)
            if offset > 0:
                # drop first incomplete line
                self.safe_readline(f)
            line = f.readline()
            print(f'line: {line}')
            while line.strip():
                line = json.loads(line)
                input_ids = self.partial_tokenize(line['text'])
                work_input_ids.extend(input_ids)
                if f.tell() > end:
                    break
                line = f.readline()
        return work_input_ids
                
    def encode_file(self, workers=6):
        bucket_name = 'jax_llm_data_us-east5'  # 存储桶名称
        directory_path = 'xiaomeng/v3.5/jsonl'  # 文件在存储桶中的路径
        client = storage.Client()
        for blob in client.list_blobs(bucket_name, prefix=directory_path):
            path = f'gs://{os.path.join(bucket_name, blob.name)}'
            file_size = blob.size
            print(f'path: {path}')
            encoded = []
            workers_thread = []
            i = 0
            work_input_ids = self.async_tokenizer(path, i, workers, file_size)
        #     pool = Pool(processes=workers)
        #     for i in range(workers):
        #         w = pool.apply_async(self.async_tokenizer, (path, i, workers, file_size))
        #         workers_thread.append(w)
        #     pool.close()
        #     pool.join()
        #     for w in workers_thread:
        #         result = w.get()
        #         encoded.extend(result)
        #     break
        # return encoded

tokenizer_path = "Qwen/Qwen-14B"
max_len = 2049
eos_id = [151643] # <|endoftext|>
bos_id = [151646] #  <|extra_0|>
qwen_tokenizer = QwenTokenizer(tokenizer_path, max_len, bos_id=bos_id, eos_id=eos_id)
encoded = qwen_tokenizer.encode_file(workers=2)