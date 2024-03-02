import json
import functools
from collections import defaultdict
import random
import os
import math

import seqio
import t5
import t5.data
from t5.data import preprocessors as t5_preprocessors
import tensorflow as tf
import numpy as np
from praxis import py_utils
from google.cloud import storage # google-cloud-storage

from absl import logging
from paxml import checkpoint_paths
import jax


           
def extract_train_skip_step(job_log_dir, step, only_eval=False):
    logging.info(f'job_log_dir: {job_log_dir} step: {step}')
    if job_log_dir is None:
        return {}
    model_dir = job_log_dir / "checkpoints"
    if step is not None:
        fill_step = checkpoint_paths.CHECKPOINT_PREFIX + str(step).zfill(checkpoint_paths._STEP_FORMAT_FIXED_LENGTH)
        skip_file_and_step_path = model_dir / fill_step / checkpoint_paths.SKIP_STEP_NAME
    else:
        skip_file_and_step_path = model_dir / checkpoint_paths.SKIP_STEP_NAME
    logging.info(f"model_dir: {model_dir}")
    try:
        with skip_file_and_step_path.open('r') as f:
            meta_dict = json.load(f)
        logging.info(f"Load skip_file_and_step_path: ’{skip_file_and_step_path}‘ Finished.......")
    except:
        logging.info(f"skip_file_and_step_path: ’{skip_file_and_step_path}‘ is not existed.......")
        meta_dict = {}

    if jax.process_index() == 0:
        mode = 'train_break_steps' if not only_eval else 'eval_metric_steps'
        back_meta_dict_path = job_log_dir / mode /f'{meta_dict.get("checkpoint_step", None)}.json'
        with back_meta_dict_path.open('w') as f1:
            json.dump(meta_dict, f1)
    return meta_dict


def get_feature(key_map, vocabulary):
    feature_desc, output_features = {}, {}
    for k, v in key_map.items():
        if v is None:
            continue
        feature_desc[v] = tf.io.VarLenFeature(tf.int64)
        output_features[k] = seqio.Feature(vocabulary=vocabulary, dtype=tf.int32)
    return feature_desc, output_features


def tfids_registry(task, mode):
    @seqio.map_over_dataset
    def convert_datatype(ex):
        return {k: tf.cast(tf.sparse.to_dense(v, default_value=0), dtype=tf.int32) for k, v in ex.items()}

    preprocessors = [
        convert_datatype,
        functools.partial(t5_preprocessors.rekey, key_map=task.KEY_MAP),
    ]
    feature_desc, output_features = get_feature(task.KEY_MAP, task.VOCABULARY)
    shuffle_buffer_size = task.SHUFFLE_SIZE if task.SHUFFLE[mode] else None
    source = seqio.TFExampleDataSource(
        split_to_filepattern={mode: task.DATA_PATH[mode]},
        feature_description=feature_desc,
    )
    print(f"mode: {mode} shuffle_size: {shuffle_buffer_size} task.SHUFFLE[mode]: {task.SHUFFLE[mode]}")
    name = f"{task.TASK_NAME}.{mode}"
    if check_registry_name(name):
        seqio.TaskRegistry.add(
            name,
            source,
            preprocessors=preprocessors,
            output_features=output_features,
            shuffle_buffer_size=shuffle_buffer_size,
        )


def check_registry_name(name):
    return False if name in t5.data.TaskRegistry._REGISTRY else True


def c4_registry(task, mode):
    preprocessors = [
        functools.partial(t5_preprocessors.rekey, key_map=task.KEY_MAP),
        seqio.preprocessors.tokenize,
        functools.partial(t5_preprocessors.reduce_concat_tokens, batch_size=4096),
        t5_preprocessors.split_tokens_to_targets_length,
    ]
    feature_desc, output_features = get_feature(task.KEY_MAP, task.VOCABULARY)
    shuffle_buffer_size = task.SHUFFLE_SIZE if task.SHUFFLE[mode] else None
    # data_path = "gs://common_datasets"
    bucket_name = task.DATA_PATH[mode]
    if 'gs:' not in bucket_name:
        bucket_name = 'gs://' + bucket_name
    print(f'c4 bucket_name: {bucket_name}')
    source = seqio.TfdsDataSource(tfds_name="c4/en:3.0.1", tfds_data_dir=bucket_name)
    name = f"c4.{mode}"
    if check_registry_name(name):
        t5.data.TaskRegistry.add(
            name,
            seqio.Task,
            source=source,
            preprocessors=preprocessors,
            output_features=output_features,
            metric_fns=[],
            shuffle_buffer_size=shuffle_buffer_size,
        )


def read_bucket(path, substrings=None, split='_b'):
    path = path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path  + '/'
    logging.info(f"bucket_name: {bucket_name} directory_path: {directory_path}")
    step_map_path = defaultdict(list)
    rerank = 0
    files = defaultdict(list)
    client = storage.Client()
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        filename = blob.name
        if substrings is None:
            substrings = []
        elif isinstance(substrings, str):
            substrings = [substrings]
        if any(substring in filename for substring in substrings):
            logging.info(f"Successful filename: {filename}=====")
            try:
                step = int(filename.rsplit(split, maxsplit=1)[-1])
            except:
                step = rerank
                rerank += 1
            path = f'gs://{os.path.join(bucket_name, filename)}'
            files[step].append(path)
        else:
            logging.info(f"Failed filename: {filename}=====")
    return files
            

def chunk_files(files, ratios, shuffle=False):
    if shuffle:
        random.shuffle(files)
    chunks = []
    last_chunk_end = 0
    for ratio in ratios:
        fs = files[last_chunk_end: last_chunk_end + math.ceil(ratio * len(files))]
        last_chunk_end += len(fs)
        if len(fs) == 0:
            fs = files[0:1]
            print(f'Ratio: {ratio} is too low, get file num is 0')
        chunks.append(fs)
    return chunks


def extract_datapath(task, mode, substrings=None, remove_steps=None, keep_steps=None):
    if remove_steps is None:
        remove_steps = []
    if substrings is None:
        substrings = []
    if hasattr(task, 'train_test_dataset'):
        return task.train_test_dataset
    #v3: us-east1-d -> common_datasets, v4: us-central2-b -> common_datasets_us-central2-b
    paths = [task.DATA_PATH[mode]] if isinstance(task.DATA_PATH[mode], str) else task.DATA_PATH[mode]
    total_files = []
    for path in paths:
        files = read_bucket(path, substrings=substrings)
        newfiles = {}
        for step, file in files.items():
            if step in keep_steps and step not in remove_steps:
                newfiles[step] = file
        files = [f for _, fs in newfiles.items() for f in fs]
        # 英文bookstart数据较多，去掉一部分
        if 'en_data' in path:
            filter_nums = int(0.2 * len(files))
            files = files[:filter_nums]
        total_files.extend(files)
    test, train = chunk_files(total_files, ratios=[task.TEST_RATIO, 1 - task.TEST_RATIO], shuffle=task.SHUFFLE['train'])
    logging.info(f'Train file: {len(train)},  test file: {len(test)}')
    train_test_dataset = {"test": test, "train": train}
    setattr(task, 'train_test_dataset', train_test_dataset)
    return train_test_dataset


def extract_sft_datapath(task, mode):
    paths = [task.DATA_PATH[mode]] if isinstance(task.DATA_PATH[mode], str) else task.DATA_PATH[mode]
    total_files = []
    substrings = ['.tfrecord']
    for path in paths:
        files = read_bucket(path, substrings=substrings)
        for step, f in files.items():
            total_files.extend(f)
    train = [f for f in total_files if 'train.' in f]
    test = [f for f in total_files if 'test.' in f]
    logging.info(f'Train file: {train} len: {len(train)},  test file: {test} len: {len(test)}')
    train_test_dataset = {"test": test, "train": train}
    setattr(task, 'train_test_dataset', train_test_dataset)
    return train_test_dataset


def extract_pythia_datapath(task, mode):
    return extract_datapath(task, mode, substrings=['.tfrecord'], remove_steps=None)
   

def extract_qwen_datapath(task, mode):
    return extract_datapath(task, mode, substrings=['E0_b'], remove_steps=[], keep_steps=[0])

# def extract_qwen_datapath2(task, mode):
#     train = extract_datapath(task, mode, substrings=['_R', '_F'], remove_steps=[], keep_steps=[0])['train']
#     test = extract_qwen_datapath_shuffled(task, mode)['test']
#     train_test_dataset = {"test": test, "train": train}
#     logging.info(f'Train file: {len(train)},  test file: {len(test)}')
#     setattr(task, 'train_test_dataset', train_test_dataset)
#     return train_test_dataset


def extract_zh_en_novel_datapath(task, mode):
    remove_steps = list(range(6, 100000))
    return extract_datapath(task, mode, substrings=['_R', '_F'], remove_steps=remove_steps)
    

def extract_bc2_datapath1213(task, mode):
    if hasattr(task, 'train_test_dataset'):
        return task.train_test_dataset
    path = 'gs://jax_llm_data/xiaomeng/zh_data_Baichuan2-13B-Base_1213'
    zh_files = read_bucket(path, substrings=['_R', '_F'], split='_b')
    path = 'gs://jax_llm_data/xiaomeng/en_data_Baichuan2-13B-Base_1213'
    en_files = read_bucket(path, substrings=['_R', '_F'], split='_b')
    total_files = []
    for key in range(0, 10001, 10000):
        zh_file = zh_files.get(key, None)
        en_file = en_files.get(key, None)
        
        if zh_file is not None:
            total_files.extend(zh_file)
        if en_file is not None:
            if key == 10000:
                en_file = random.sample(en_file, k=len(en_file) // 2)
            total_files.extend(en_file)

    random.seed(task.TRAINING_SEED)
    random.shuffle(total_files)
    test_nums = int(len(total_files) * task.TEST_RATIO)
    test = total_files[: test_nums]
    train = total_files[test_nums:]

    logging.info(f'Train file: {len(train)},  test file: {len(test)}')
    train_test_dataset = {"test": test, "train": train}
    setattr(task, 'train_test_dataset', train_test_dataset)
    return train_test_dataset


def extract_bc2_datapath1213_shuffled(task, mode):
    if hasattr(task, 'train_test_dataset'):
        return task.train_test_dataset
    path = f'gs://jax_llm_data/xiaomeng/zh_en_data_bc2_13b_1214_shuffled/'
    zh_en_files = read_bucket(path, substrings=['zh_en_b'], split='_b')
    total_files = []
    for key in range(0, 10000000, 10000):
        zh_en_file = zh_en_files.get(key, None)
        if zh_en_file is not None:
            total_files.extend(zh_en_file)
    # random.seed(task.TRAINING_SEED)
    # random.shuffle(total_files)
    test = total_files[: 10]
    train = total_files[10: ]
    logging.info(f'Train file: {len(train)},  test file: {len(test)}')
    train_test_dataset = {"test": test, "train": train}
    setattr(task, 'train_test_dataset', train_test_dataset)
    return train_test_dataset


def extract_qwen_datapath_shuffled(task, mode):
    if hasattr(task, 'train_test_dataset'):
        return task.train_test_dataset
    path = 'gs://jax_llm_data/xiaomeng/shuffled_zh_data'
    zh_files = read_bucket(path, substrings=['tfrecord'], split='.b')

    path = 'gs://jax_llm_data/xiaomeng/shuffled_en_data'
    en_files = read_bucket(path, substrings=['tfrecord'], split='.b')
    total_files = []
    for key in range(0, 2000000, 10000):
        zh_file = zh_files.get(key, None)
        en_file = en_files.get(key, None)
        if zh_file is None or en_file is None:
            break
        total_files.extend(zh_file)
        total_files.extend(en_file)
    test = total_files[ :10]
    train = total_files[10: ]
    logging.info(f'Train file: {len(train)},  test file: {len(test)}')
    train_test_dataset = {"test": test, "train": train}
    setattr(task, 'train_test_dataset', train_test_dataset)
    return train_test_dataset


def extract_qwen_datapath1208_shuffled(task, mode):
    if hasattr(task, 'train_test_dataset'):
        return task.train_test_dataset
    path = 'gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208_shuffled'
    zh_files = read_bucket(path, substrings=['_b'], split='_b')

    path = 'gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208_shuffled'
    en_files = read_bucket(path, substrings=['_b'], split='_b')
    total_files = []
    for key in range(0, 10000, 10000):
        zh_file = zh_files.get(key, None)
        en_file = en_files.get(key, None)
        if zh_file is not None:
            total_files.extend(zh_file)
        if en_file is not None:
            total_files.extend(en_file)
    test = total_files[ :10]
    train = total_files[10: ]
    logging.info(f'Train file: {len(train)},  test file: {len(test)}')
    train_test_dataset = {"test": test, "train": train}
    setattr(task, 'train_test_dataset', train_test_dataset)
    return train_test_dataset


def extract_qwen_datapath1208(task, mode):
    if hasattr(task, 'train_test_dataset'):
        return task.train_test_dataset
    path = 'gs://jax_llm_data/xiaomeng/zh_data_Qwen-14B_1208'
    zh_files = read_bucket(path, substrings=['_R', '_F'], split='_b')
    path = 'gs://jax_llm_data/xiaomeng/en_data_Qwen-14B_1208'
    en_files = read_bucket(path, substrings=['_R', '_F'], split='_b')
    total_files = []
    for key in range(10000, 20001, 10000):
        zh_file = zh_files.get(key, None)
        en_file = en_files.get(key, None)
        if zh_file is None or en_file is None:
            break
        total_files.extend(zh_file)
        total_files.extend(en_file)
    random.seed(task.TRAINING_SEED)
    random.shuffle(total_files)
    test_nums = int(len(total_files) * task.TEST_RATIO)
    test = total_files[: test_nums]
    train = total_files[test_nums:]

    logging.info(f'Train file: {len(train)},  test file: {len(test)}')
    train_test_dataset = {"test": test, "train": train}
    setattr(task, 'train_test_dataset', train_test_dataset)
    return train_test_dataset