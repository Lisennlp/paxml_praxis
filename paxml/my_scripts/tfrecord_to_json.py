import os
import time
import argparse
import socket
import random
from collections import defaultdict
import json

os.environ["JAX_PLATFORMS"] = "cpu"

import tensorflow as tf
from google.cloud import storage
import seqio
import functools
from t5.data import preprocessors as t5_preprocessors


TEST_RATIO = 0.02
TRAINING_SEED = 1234
def extract_datapath(test_ratio, seed):
    random.seed(seed)
    dataset = defaultdict(list)
    client = storage.Client()
    bucket_name = 'jax_llm_data'
    zh, en = 0, 0
    for lang in ['zh', 'en']:
        directory_path = f'xiaomeng/processed_{lang}_data_qwen14B'
        for blob in client.list_blobs(bucket_name, prefix=directory_path):
            if '_R' in blob.name:
                print(f'{blob.name}')
                if '_en_' in blob.name:
                    en += 1
                    if en > 25:
                        break
                else:
                    zh += 1
                    if zh > 75:
                        break
                path = os.path.join(f'gs://{bucket_name}', blob.name)
                dataset[lang].append(path)
    print(f'zh : {zh} en: {en}')
    train_test_dataset = defaultdict(list)
    test_en_n = 1
    test_zh_n = test_en_n * 4
    train_test_dataset['test'] = dataset['en'][:test_en_n] + dataset['zh'][:test_zh_n]
    train_test_dataset['train'] = dataset['en'][test_en_n:] + dataset['zh'][test_zh_n:]
    print(f'dataset zh file nums: {len(dataset["zh"])}, en file nums: {len(dataset["en"])}')
    return train_test_dataset
DATA_PATH = extract_datapath(TEST_RATIO, TRAINING_SEED)

def get_feature(key_map, vocabulary):
    feature_desc, output_features = {}, {}
    for k, v in key_map.items():
        if v is None:
            continue
        feature_desc[v] = tf.io.VarLenFeature(tf.int64)
        output_features[k] = seqio.Feature(vocabulary=vocabulary, dtype=tf.int32)
    return feature_desc, output_features

KEY_MAP = {"input_ids": "input_ids", "labels": "input_ids"}
VOCABULARY = 151851

def tfids_registry():
    @seqio.map_over_dataset
    def convert_datatype(ex):
        return {
            k: tf.cast(tf.sparse.to_dense(v, default_value=0), dtype=tf.int32)
            for k, v in ex.items()
        }

    preprocessors = [
        convert_datatype,
        functools.partial(t5_preprocessors.rekey, key_map=KEY_MAP),
    ]
    feature_desc, output_features = get_feature(KEY_MAP, VOCABULARY)
    for mode in ["train", "test"]:
        shuffle_buffer_size = 10000
        source = seqio.TFExampleDataSource(
            split_to_filepattern={mode: DATA_PATH[mode]},
            feature_description=feature_desc,
        )
        seqio.TaskRegistry.add(
            f"tftest.{mode}",
            source,
            preprocessors=preprocessors,
            output_features=output_features,
            shuffle_buffer_size=shuffle_buffer_size
        )

tfids_registry()

# for mode in ['test', 'train']:
mode = 'test'
mixture_or_task_inst = seqio.get_mixture_or_task(f'tftest.{mode}')


task_feature_lengths = {"input_ids": 4096, 'labels': 4096}
split_name = mode
shuffle = True
num_epochs = 1
shard_info = None
use_cached = False
input_random_seed = 1234
batch_size = 1
seq_len = 4096

kwargs = dict(
            sequence_length=task_feature_lengths,
            split=split_name,
            shuffle=shuffle,
            num_epochs=num_epochs,
            shard_info=shard_info,
            use_cached=use_cached,
            seed=input_random_seed,
            trim_output_features=True,  # default: True
            try_in_mem_cache=False,
        )
ds = mixture_or_task_inst.get_dataset(**kwargs)

ds = ds.as_numpy_iterator()
writer = open(f'qwen14b.{mode}', 'w')
count = 0
start_time = time.time()
try:
    while 1:
        if count % 100 == 0:
            print(f'Processed: {count}, take: {time.time() - start_time}s')
        a = next(ds)
        b = {'input_ids': a['input_ids'].tolist(), 'labels':a['input_ids'].tolist()}
        b = json.dumps(b, ensure_ascii=False)
        writer.write(f'{b}\n')
        count += 1
except Exception as e:
    writer.close()