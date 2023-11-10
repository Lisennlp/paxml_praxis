import json
import functools
from collections import defaultdict
import random
import os

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
import smart_open # smart_open[gcs]


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


def extract_pythia_datapath(task, mode):
    if hasattr(task, 'train_test_dataset'):
        return task.train_test_dataset
    client = storage.Client()
    #v3: us-east1-d -> common_datasets, v4: us-central2-b -> common_datasets_us-central2-b
    bucket_name = os.path.dirname(task.DATA_PATH[mode])
    if 'gs:' in bucket_name:
        bucket_name = bucket_name[5: ]
    directory_path = os.path.basename(task.DATA_PATH[mode]) + '/'
    step_map_path = {}
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        logging.info(f"filename: {blob.name}=====")
        step = int(blob.name.rsplit("pile.tfrecord.b", maxsplit=1)[-1])
        path = f'gs://{os.path.join(bucket_name, blob.name)}'
        step_map_path[step] = path
    sorted_step_path = sorted(step_map_path.items(), key=lambda x: x[0])
    steps, pathes = zip(*sorted_step_path)
    if not isinstance(pathes, list):
        pathes = list(pathes)

    if task.SHUFFLE['train']: # train和test必须都shuffle，才能保证train和test的数据集不重合
        random.shuffle(pathes)

    test_num = int(len(pathes) * task.TEST_RATIO)
    test = pathes[ :test_num]
    train = pathes[test_num: ]
    if not len(test):
        test = pathes[0]

    if not len(train):
        train = pathes[0]
        
    train_test_dataset = {"test": test, "train": train}
    logging.info(f'Train file: {len(train_test_dataset["train"])},  test file: {len(train_test_dataset["test"])}')
    task.train_test_dataset = train_test_dataset
    return train_test_dataset


def extract_zh_en_novel_datapath(task, mode):
    if hasattr(task, 'train_test_dataset'):
        return task.train_test_dataset
    random.seed(task.TRAINING_SEED)
    dataset = defaultdict(list)
    client = storage.Client()
    bucket_name = os.path.dirname(task.DATA_PATH[mode])
    if 'gs:' in bucket_name:
        bucket_name = bucket_name[5: ]
    directory_path = os.path.basename(task.DATA_PATH[mode]) + '/'
    for lang in ["zh", "en"]:
        # directory_path = f'xiaomeng/processed_{lang}_data_split'
        prefix = directory_path.format(lang=lang)
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            logging.info(f"filename: {blob.name}=====")
            if not blob.name or "_R" not in blob.name:
                continue
            if len(dataset[lang]) > 5:
                break
            index = int(blob.name.rsplit("_", maxsplit=1)[-1])
            # 每本书的前多少个4096
            if index < task.SPLIT_BSZ[lang]:
                path = os.path.join(f"gs://{bucket_name}", blob.name)
                dataset[lang].append(path)
    total = dataset["zh"] + dataset["en"]
    if task.SHUFFLE['train']: # train和test必须都shuffle，才能保证train和test的数据集不重合
        random.shuffle(total)

    test_num = int(len(pathes) * task.TEST_RATIO)
    test = pathes[ :test_num]
    train = pathes[test_num: ]
    if not len(test):
        test = pathes[0]
    if not len(train):
        train = pathes[0]

    train_test_dataset = {"test": test, "train": train}
    logging.info(f'Train file: {len(train_test_dataset["train"])},  test file: {len(train_test_dataset["test"])}')
    task.train_test_dataset = train_test_dataset
    return train_test_dataset


def extract_train_skip_step(job_log_dir, step):
    if job_log_dir is None:
        return {}
    model_dir = os.path.join(job_log_dir, "checkpoints")
    if step is not None:
        fill_step = checkpoint_paths.CHECKPOINT_PREFIX + str(step).zfill(checkpoint_paths._STEP_FORMAT_FIXED_LENGTH)
        skip_file_and_step_path = os.path.join(model_dir, fill_step, f'{checkpoint_paths.SKIP_STEP_NAME}')
    else:
        skip_file_and_step_path = os.path.join(model_dir, f'{checkpoint_paths.SKIP_STEP_NAME}')
    logging.info(f"model_dir: {model_dir}")
    try:
        with smart_open.open(skip_file_and_step_path, 'r') as f:
            meta_dict = json.load(f)
        logging.info(f"Load skip_file_and_step_path: ’{skip_file_and_step_path}‘ Finished.......")
    except:
        logging.info(f"skip_file_and_step_path: ’{skip_file_and_step_path}‘ is not existed.......")
        meta_dict = {}
    return meta_dict