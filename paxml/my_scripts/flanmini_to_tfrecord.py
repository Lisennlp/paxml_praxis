import json
import random
import os
import time

os.environ["JAX_PLATFORMS"] = "cpu"

from etils import epath
import tensorflow as tf


path = 'gs://common_datasets/pythia_model_test/flan_test/flan_mini_filtered_v2.jsonl'
path = 'gs://common_datasets/pythia_model_test/pile_test/flan_mini_filtered_v2.jsonl'
path = epath.Path(path)

lines = []
with path.open('r') as f:
    for line in f:
        line = json.loads(line)
        lines.append(line)
        

random.seed(1234)
random.shuffle(lines)

def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


start = time.time()
wp = 'gs://common_datasets/pythia_model_test/flan_test/flan_mini_filtered_v2_len2049.tfrecord'

N = len(lines)
max_len = 2049
with tf.io.TFRecordWriter(wp) as writer:
     for index, line in enumerate(lines):
        example = line
        labels = [-100] +  example['labels'][:-1]
        if len(labels) > max_len:
            print(f'exceed max length: {len(labels)}, index: {index}')
            continue
        assert len(line['input_ids']) == len(labels)
        
        if index % 100 == 0:
            print(f'processed: {index}/{N} take: {time.time() - start}s')
        feature = {
            "input_ids": _int64_feature(example['input_ids']),
            "labels": _int64_feature(labels),
            
                  }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())