import os
import re
from datetime import datetime
import time
import sys

from google.cloud import storage # google-cloud-storage
import tensorflow as tf

def extract_loss(path):
    losses = []
    steps = []
    summaries = tf.compat.v1.train.summary_iterator(path)
    tags = set()
    for step, e in enumerate(summaries):
        for v in e.summary.value:
            if v.tag == 'loss':
                loss = tf.make_ndarray(v.tensor)
                losses.append(loss)
                steps.append(e.step)
            tags.add(v.tag)
    return steps, losses


def extract_time(t):
    dt_object = datetime.fromtimestamp(t)
    time_string = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    return time_string


def extract_files(path, check_whole_or_lastest=None):
    read_file_nums = -2 if check_whole_or_lastest == 'lastest' else None
    path = path.replace('gs://', '')
    path_parts = path.split('/')
    bucket_name = path_parts[0]
    directory_path = '/'.join(path_parts[1:])
    directory_path = directory_path if directory_path.endswith('/') else directory_path  + '/'
    client = storage.Client()
    filenames = {}
    for blob in client.list_blobs(bucket_name, prefix=directory_path):
        filename = blob.name
        if 'events' not in filename: continue
        t = int(re.findall('tfevents\.(\d+)\.', filename)[0])
        file_path = os.path.join(f'gs://{bucket_name}', filename)
        filenames[t] = file_path
        # print(f'file_path: {file_path}')
    filenames = sorted(filenames.items(), key=lambda x: x[0])
    return filenames[read_file_nums: ]

print(f'Please be patient, this may take a few minutes if there are many files......')

start_time = time.time()
path = sys.argv[1]
check_whole_or_lastest = sys.argv[2]

# path = 'gs://llm_projects/log/summaries/train/PilePythiaXLDynWFFN8HD64Win256Alignedv4'
# path = 'gs://llm_projects/log/summaries/train/PilePythia7B256x1DynWFFN16HD128Win256Alignedv4'
# path = 'gs://llm_projects/log/summaries/train/PilePythia410M128x1DynWFFN8HD64Win256Aligned'
# path = 'gs://llm_projects/log/summaries/train/PilePythiaXLDynWFFN8HD64Win256Alignedv4'
# check_whole_or_lastest = 'lastest' # lastest: 检查最新抢占点loss；whole: 检查全部抢占点loss
file_paths = extract_files(path, check_whole_or_lastest=check_whole_or_lastest)
    
total_steps, total_losses = [], []
for t, file_path in file_paths:
    steps, loss = extract_loss(file_path)
    # print(f'Check file path: {file_path} step range: {steps[0]} ~ {steps[-1]}')
    if not len(steps): continue
    total_losses.append(loss)
    total_steps.append(steps)


FLAG = 0
stepin = False
CHECKPOINT_EVERY_N_STEPS = 100
error_steps = []
for i in range(1, len(total_losses)):
    t = extract_time(file_paths[i][0])
    start_step = total_steps[i][0]
    end_step = total_steps[i][-1]
    print(f'Checking step: {start_step} ± {CHECKPOINT_EVERY_N_STEPS}')
    last_loss, cur_loss = total_losses[i - 1: i + 1]
    last_loss = [round(l.item(), 10) for l in last_loss]
    cur_loss = [round(l.item(), 10) for l in cur_loss]
    
    div = CHECKPOINT_EVERY_N_STEPS // 10
    last_step_and_losses = list(zip(total_steps[i - 1], last_loss))
    cur_step_and_losses = list(zip(total_steps[i ], cur_loss))
    
    last_end_losses = dict(last_step_and_losses[-div: ])
    cur_start_losses = cur_step_and_losses[ :div]
    for s, l in cur_start_losses:
        if s in last_end_losses:
            stepin = True
            last_l = last_end_losses[s]
            if last_l != l:
                FLAG = 1
                error_steps.append(s)
                print(f'Error step: {s} last train loss: {last_l} is not equal to next loss: {l} !!!!')
    if not stepin:
        print(f'Warning, all cur start steps cannot be found at last training end !!!')
        
if not FLAG:
    print(f'Train is very successful, all loss is right.')
else:
    print(f'Train is failed, partial error steps is ‘{error_steps}’')

print(f'Check finished, take: {time.time() - start_time}s')
