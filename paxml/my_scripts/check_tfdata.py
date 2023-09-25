import os
import time
import argparse
import socket
import random
from collections import defaultdict
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
    # splits = ['split0', 'split1', 'split2']
    splits = ['split0', 'split2']
    start_files, median_files, end_files = [], [], []
    for lang in ['zh', 'en']:
        directory_path = f'xiaomeng/processed_{lang}_data_split'
        for blob in client.list_blobs(bucket_name, prefix=directory_path):
            for split in splits:
                if split in blob.name:
                    path = os.path.join(f'gs://{bucket_name}', blob.name)
                    dataset[split].append(path)
                    break
    train_test_dataset = defaultdict(list)
    for k, v in dataset.items():
        random.shuffle(v)
        test = v[:int(len(v) * test_ratio)]
        train = v[int(len(v) * test_ratio): ]
        train_test_dataset['train'].extend(train)
        train_test_dataset['test'].extend(test)
        print(f'dataset: {k}, file nums: {len(v)}')
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

KEY_MAP = {"targets": "input_ids", "masks": "labels"}
VOCABULARY = 125696

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

mode = 'train'
mixture_or_task_inst = seqio.get_mixture_or_task(f'tftest.{mode}')


task_feature_lengths = {"targets": 4096, 'masks': 4096}
split_name = mode
shuffle = True
num_epochs = 1
shard_info = None
use_cached = False
input_random_seed = 1234
batch_size = 256
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
ds = ds.padded_batch(
        batch_size=batch_size,
        padded_shapes={'targets': [seq_len], 'masks': [seq_len]},
        drop_remainder=True,
    )

# ds = ds.as_numpy_iterator()