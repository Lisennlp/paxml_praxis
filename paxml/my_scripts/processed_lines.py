import os
import random
import sys
import socket

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
import orjson

"""
多进程处理单个文件:
# Usage:
TPU_NAME=llm-jax-v4-512-10; ZONE=us-central2-b
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install tiktoken smart_open[gcs] gcsfs orjson" --project=ntpu-413714
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo rm -r /home/lishengping/tokenizer;gsutil cp -r gs://llm_base_models_us-east5/qwen/tokenizer /home/lishengping/" --project=ntpu-413714

TPU_NAME=llm-jax-v4-512-10; ZONE=us-central2-b
SCRIPT=/Users/lishengping/codes/jax_projects/paxml_praxis/paxml/my_scripts/processed_lines.py
gcloud compute tpus tpu-vm scp $SCRIPT $TPU_NAME:/home/lishengping/processed.py  --zone=$ZONE  --worker=all  --project=ntpu-413714

TPU_NAME=llm-jax-v4-512-10; ZONE=us-central2-b;B=19
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=4 --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py $B,8,10" --project=ntpu-413714
"""


TOKENIZER_PATH = "/home/lishengping/tokenizer"
MAX_LEN = 4097
EOS_ID = [151643] # <|endoftext|>
BOS_ID = [151646] #  <|extra_0|>

EXTRA_TOKENS = '<repo_name><file_sep><translation_type><lang_zh><lang_zh-hant><lang_en><lang_ja><lang_ko><lang_pt><lang_es><lang_fr><lang_de><lang_ru><lang_th><lang_vi><lang_id><lang_ar><lang_it><lang_tr><lang_hi>'

# def extract_files(bucket_name, directory_path):
#     client = storage.Client()
#     pathes = defaultdict(list)
#     for blob in client.list_blobs(bucket_name, prefix=directory_path):
#         path = f'gs://{os.path.join(bucket_name, blob.name)}'
#         if 'valid' in path:
#             pathes['valid'].append(path)
#         else:
#             pathes['train'].append(path)
#     return pathes

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_tfrecord(writer, input_ids):
    feature = {
        "input_ids": _int64_feature(input_ids),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


class QwenTokenizer():
    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH, use_fast=True, trust_remote_code=True
        )
        assert len(self.tokenizer) == 151871, print(len(self.tokenizer))
        assert len(self.tokenizer.encode(EXTRA_TOKENS)) == 20, print(len(self.tokenizer.encode(EXTRA_TOKENS)))
        self.next_ids = []
        self.partial_tokenize = partial(self.tokenize, max_len=MAX_LEN, bos_id=BOS_ID, eos_id=EOS_ID)
        self.count = 0
    
    def tokenize(self, text, writer, max_len=2048, bos_id:list=[], eos_id:list=[]):
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
 
def  check_text_length(line):
    # 0524 add filter， 有些乱码数据很长一段
    text = line['text']
    words = text.split()
    char_count = line['meta']['char_count']
    if char_count < 2:
        return False
    # 计算单词的平常长度
    word_mean_len = char_count / len(words)
    if word_mean_len > 50000:
        print(f'\n\nError line name: {line["meta"]}\n\n')
        return False
    else:
        return True

def process_data(args):
    save_path, cur_rank_lines, rank, workers = args
    save_path = os.path.join(save_path, f'{rank:03}')
    qwen_tokenizer = QwenTokenizer(TOKENIZER_PATH)
    writer = tf.io.TFRecordWriter(save_path)
    for i in tqdm(range(len(cur_rank_lines)), desc=f'Rank-{rank}'):
        line = cur_rank_lines[i]
        line = orjson.loads(line)
        text = line['text']
        text_split = text.split('\n')
        if not check_text_length(line):
            continue
        per = 500
        if len(text_split) > per:
            # 一次Tokenize很长的数据会很慢，需要split。
            for lnx in tqdm(range(0, len(text_split), per), desc=f'Rank-{rank}-sub-{i}'):
                inp = text_split[lnx: lnx + per]
                inp = '\n'.join(inp)
                qwen_tokenizer.partial_tokenize(inp, writer)
        else:
            qwen_tokenizer.partial_tokenize(text, writer)

    writer.close()
    return qwen_tokenizer.count

def encode_file(path, save_path, workers=6):
    mode = 'r' if 'valid' in path else 'rb'
    with smart_open.open(path, mode) as f:
        lines = f.readlines()
    print(f'path:{path}, {len(lines)}')
    pool = Pool(processes=workers)
    perrank_line_num = math.ceil(len(lines) / workers)
    # counts = []
    # for rank in range(workers):
    #     # if rank not in [2, 7]: continue
    #     rank_lines = lines[rank * perrank_line_num: (rank + 1) * perrank_line_num]
    #     result = pool.apply_async(process_data, args=([save_path, rank_lines, rank, workers]))
    #     count = result.get()
    #     counts.append(count)
    # map
    args = ([save_path, lines[rank * perrank_line_num: (rank + 1) * perrank_line_num], rank, workers] for rank in range(workers))
    counts = pool.map(process_data, args)  # 包含每个进程的返回值

    pool.close()
    pool.join()
    return counts


if __name__ == "__main__":
    random.seed(42)
    file_index = sys.argv[1]
    bucket, file_start, file_end = [int(a) for a in file_index.split(',')]
    # set_start_method("spawn")  # tpu-vm
    num_processes = multiprocessing.cpu_count()
    print(f"num_processes: {num_processes}")
    meta_path = f'gs://jax_llm_data_us-east5/xiaomeng/v3.5/meta5.json'
    meta_path = epath.Path(meta_path)
   
    bucket_name = 'jax_llm_data_us-east5'  # 存储桶名称
    directory_path = 'xiaomeng/v3.5/jsonl'  # 文件在存储桶中的路径
    # pathes = extract_files(bucket_name, directory_path)

    type_ = 'train'
    if type_ == 'valid':
        pathes = ['gs://jax_llm_data_us-east5/xiaomeng/v3.5/jsonl/valid_concat.jsonl']
        save_path = f'gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfrecord_test/valid_concat.tfrecord'
        print(f'save_path: {save_path}')
    else:
        bucketes = [bucket]
        pathes = []
        for bucket in bucketes:
            for index in range(10):
                p = f'gs://jax_llm_data_us-east5/xiaomeng/v3.5/jsonl/2nd-shuffled-data_bucket-{bucket}-{index:03}-of-010.jsonl.zst'
                pathes.append(p)

    select_files = pathes[file_start: file_end]
    print(f'{type_} files: \n{pathes}  \n\nselect_files: \n{select_files}')
    for path in select_files:
        if type_ != 'valid':
            print(f'path: {path}')
            name = os.path.basename(path)
            bucket = int(name.split('-')[3])
            file_index = name.split('-')[4]
            save_path = f'gs://jax_llm_data_us-central2/xiaomeng/v3.5/tfids0424/B{bucket:03}/F{file_index}'
        print(f'save_path: {save_path}')
        workers = 10
        counts = encode_file(path, save_path, workers=workers)
        print(f'counts: {counts}')
        meta_dict = {save_path: counts}
        print(meta_dict)
        with meta_path.open('a') as f:
            meta_dict = json.dumps(meta_dict, ensure_ascii=False)
            f.write(f'{meta_dict}\n')
