import os
import random
import sys
import socket
import pickle
import re
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
TPU_NAME=llm-jax-v4-512-11; ZONE=us-central2-b
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install tiktoken smart_open[gcs] gcsfs orjson" --project=ntpu-413714
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo rm -r /home/lishengping/tokenizer;gsutil cp -r gs://llm_base_models_us-east5/qwen/tokenizer /home/lishengping/" --project=ntpu-413714

TPU_NAME=llm-jax-v4-512-11; ZONE=us-central2-b
SCRIPT=/Users/lishengping/codes/jax_projects/paxml_praxis/paxml/my_scripts/processed_lines.py
gcloud compute tpus tpu-vm scp $SCRIPT $TPU_NAME:/home/lishengping/processed.py  --zone=$ZONE  --worker=all  --project=ntpu-413714

TPU_NAME=llm-jax-v4-512-11; ZONE=us-central2-b;B=19
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=3 --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py $B,9,10" --project=ntpu-413714
"""

"""
多进程处理单个文件:
# Usage:
TPU_NAME=llm-jax-mqy-v5p-16-49-paxml; ZONE=us-east5-a
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install tiktoken smart_open[gcs] gcsfs orjson" --project=ntpu-413714
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo rm -r /home/lishengping/tokenizer;gsutil cp -r gs://llm_base_models_us-east5/qwen/tokenizer /home/lishengping/" --project=ntpu-413714

TPU_NAME=llm-jax-mqy-v5p-16-48-paxml; ZONE=us-east5-a
SCRIPT=/Users/lishengping/codes/jax_projects/paxml_praxis/paxml/my_scripts/processed_lines.py
gcloud compute tpus tpu-vm scp $SCRIPT $TPU_NAME:/home/lishengping/processed.py  --zone=$ZONE  --worker=all  --project=ntpu-413714

TPU_NAME=llm-jax-mqy-v5p-16-49-paxml; ZONE=us-east5-a;B=3
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=1 --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py $B,0,10" --project=ntpu-413714
"""


# remove: 2nd-shuffled-data_bucket-36-005-of-010.jsonl.zst
# the-stack-v2-train-full-ids: [251197170, 1002024153]

TOKENIZER_PATH = "/home/lishengping/tokenizer"
MAX_LEN = 4097
EOS_ID = [151643] # <|endoftext|>
BOS_ID = [151646] #  <|extra_0|>

EXTRA_TOKENS = '<repo_name><file_sep><translation_type><lang_zh><lang_zh-hant><lang_en><lang_ja><lang_ko><lang_pt><lang_es><lang_fr><lang_de><lang_ru><lang_th><lang_vi><lang_id><lang_ar><lang_it><lang_tr><lang_hi>'


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
        self.partial_tokenize = partial(self.tokenize, max_len=MAX_LEN, bos_id=BOS_ID)
        self.count = 0
    
    def tokenize(self, text, writer, max_len=2048, bos_id:list=[]):
        try:
            input_ids = self.tokenizer.encode(text)
        except:
            import pickle
            pickle.dump(text, open(f'error_{self.count}.pkl', 'wb'))
            print(f'error======')
            return []
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
                if len(save_ids) < 100: # 如果上条数据留下来的token小于100，直接丢弃
                    self.next_ids = []
                else:
                    self.next_ids = save_ids
                save_ids = []
        return total_ids
 

def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
 
    return False

remove_data = {'the-stack-v2-train-full-ids': [251197170, 1002024153]}
data_mean_len_dict = {'the-stack-v2-train-full-ids': 50}


def compute_word_len(text, char_count):
    if not text:
        return 0, 0
    words = text.split()
    if len(words) < 10:
        return 0, 0
    english_chars = re.findall(r'[a-zA-Z]', text)
    en_char_len = len(english_chars)
    en_ratio = en_char_len / char_count
    if en_ratio < 0.5: 
        return -1, en_ratio
    word_len = en_char_len / len(words)
    return word_len, en_ratio
    
def check_text_length(save_path, line, rank, i):
    dataset_name = line['meta']['dataset_name']
    text = line['text']
    char_count = line['meta']['char_count']

    if dataset_name == 'the-stack-v2-train-full-ids':
        if char_count in remove_data[dataset_name]:
            _write_error_message(200)
            print('=============================================\n\n')
            return False
        # else:
        #     return True

    if char_count < 10:
        return False
    fir_segment_text = text[:10000]
    if len(fir_segment_text) < 10000:
        sec_segment_text = ''
    else:
        sec_segment_text = text[-10000:]
    fir_word_len, fir_en_ratio = compute_word_len(fir_segment_text, char_count)
    sec_word_len, sec_en_ratio = compute_word_len(sec_segment_text, char_count)
    if fir_word_len == -1 or sec_word_len == -1:
        return True
    div = 2 if sec_word_len else 1
    word_len = (fir_word_len + sec_word_len) / div
    en_ratio = (fir_en_ratio + sec_en_ratio) / div

    if word_len == 0:
        return False

    def _write_error_message(word_mean_len):
        print(f'fir_word_len: {fir_word_len} sec_word_len: {sec_word_len}')
        os.makedirs('error_data', exist_ok=True)
        writer = open(f'error_data/{rank}.json', 'a+')
        error_mes = {'dataset_name': dataset_name, 'en_ratio': en_ratio, 'save_path': save_path, 'char_count': char_count, 'word_mean_len': word_mean_len, 'text200': text[:500]}
        print(f'error_mes: {error_mes}')
        error_mes = json.dumps(error_mes, ensure_ascii=False)
        writer.write(f'{error_mes}\n')

    thesold = data_mean_len_dict.get(dataset_name, 30)
    if word_len > thesold:
        _write_error_message(word_len)
        return False


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
        if not check_text_length(save_path, line, rank, i):
            continue
        per = 250
        if len(text_split) > per:
            # 一次Tokenize很长的数据会很慢，需要split。
            for lnx in tqdm(range(0, len(text_split), per), desc=f'Rank-{rank}-sub-{i}'):
                inp = text_split[lnx: lnx + per]
                inp = '\n'.join(inp) + '\n' # lsp
                qwen_tokenizer.partial_tokenize(inp, writer)
        else:
            qwen_tokenizer.partial_tokenize(text, writer)
        qwen_tokenizer.next_ids += EOS_ID

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
        save_path = f'gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfids0527/valid_concat.tfrecord'
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
            save_path = f'gs://jax_llm_data_us-east5/xiaomeng/v3.5/tfids1123/B{bucket:03}/F{file_index}'
        print(f'save_path: {save_path}')
        workers = 100
        counts = encode_file(path, save_path, workers=workers)
        print(f'counts: {counts}')
        meta_dict = {save_path: counts}
        print(meta_dict)
        with meta_path.open('a') as f:
            meta_dict = json.dumps(meta_dict, ensure_ascii=False)
            f.write(f'{meta_dict}\n')
