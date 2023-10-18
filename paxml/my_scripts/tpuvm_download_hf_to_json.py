import pandas as pd
import random
import math
import time

from datasets import Dataset
import os
import json
from datasets import load_dataset
import requests
from bs4 import BeautifulSoup
import subprocess
from multiprocessing import set_start_method
import multiprocessing

os.environ["JAX_PLATFORMS"] = "cpu"
import tensorflow as tf


def get_url_data_dir(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True)]
        folders = [os.path.basename(link) for link in links if '/tree/main/data' in link]
        for folder in folders:
            print(folder)
    else:
        folders = []
        print(f"Failed to fetch content from {url}. Status code: {response.status_code}")
    return folders

def find_file_names(n, parent_dir, dataset_name):
    for i in range(40, n):
        total_nums = str(i).zfill(5)
        file_name_format = f'data-00000-of-{total_nums}.arrow'
        link = os.path.join(parent_dir, file_name_format)
        response = requests.head(link)
        if response.status_code != 404:
            break
    data_files = [{'filename': f'data-{str(j).zfill(5)}-of-{total_nums}.arrow'} for j in range(i)]
    meta_dict = {'_data_files': data_files}
    # json.dump(meta_dict, open(f'{dataset_name}.state.json', 'w'))
    return data_files


def extract_data_files(dataset_name, save_dir, download_link, mode):
    download_link = 'https://huggingface.co/datasets/ArmelR/the-pile-splitted/resolve/'
    meta_name = 'state.json'
    sec_dir = f'main/data/{dataset_name}/{mode}/'
    meta_file = os.path.join(sec_dir, meta_name)
    meta_download_path = os.path.join(download_link, meta_file)
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    meta_path = os.path.join(dataset_dir, meta_name)
    print(f'meta_path: {meta_path}')
    flag = 1
    if not os.path.exists(meta_path):
        command = f'wget -P {dataset_dir} {meta_download_path}'
        # command中需要手动转义，其他的不用。
        command = command.replace('(', r'\(').replace(')', r'\)')
        print('======')
        response = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
        if response.returncode != 0:
            link = os.path.join(download_link, sec_dir)
            data_files = find_file_names(1000, link, dataset_name)
            flag = 0
    if flag:
        with open(f'{meta_path}') as f:
            meta = json.load(f)
            data_files = meta['_data_files']
        # delete
        os.remove(meta_path)
    assert len(data_files) > 0
    data_abs_paths = [f"{dataset_name}::{mode}::{meta_download_path.replace(meta_name,  f['filename'])}"  for f in data_files]
    print(f'data_abs_paths:\n{data_abs_paths}, length: {len(data_abs_paths)}')
    assert len(data_abs_paths) > 0
    return data_abs_paths


def processed_single_file(args):
    links, local_dir, bucket_dir = args
    start = time.time()
    for link in links:
        if 'Stack' not in link: continue
        print(f'Link: {link}, take: {time.time() - start}s')
        dataset_name, mode, download_link = link.split('::')
        save_dataset_dir = os.path.join(local_dir, dataset_name)
        # download
        print(f'Downing....')
        command = f'wget -P {save_dataset_dir} {download_link}'
        command = command.replace('(', r'\(').replace(')', r'\)')
        print(f'Download command: {command}')
        response = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
        # read data
        print(f'Reading....')
        filename = os.path.basename(download_link)
        file_local_arrow_path = os.path.join(save_dataset_dir, filename)
        print(f'Read file_local_arrow_path: {file_local_arrow_path}')
        ds = Dataset.from_file(file_local_arrow_path)
        # to json
        print(f'Jsoning....')
        file_local_json_path = file_local_arrow_path.replace('.arrow', f'.{mode}.jsonl')
        ds.to_json(file_local_json_path)
        # save data to bucket
        print(f'Bucketing....')
        bucket_path = os.path.join(bucket_dir, dataset_name)
        bucket_path = bucket_path.replace('(', r'_').replace(')', r'_')
        command = f'gsutil cp -r {file_local_json_path} {bucket_path}/'
        command = command.replace('(', r'\(').replace(')', r'\)')
        print(f'Bucket command: {command}')
        response = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
        # del arrow and jsonl file
        print(f'Removing....')
        print(f'Remove file_local_arrow_path: {file_local_arrow_path}....')
        os.remove(file_local_arrow_path)
        print(f'Remove file_local_json_path: {file_local_json_path}....')
        os.remove(file_local_json_path)
    print(f'Finished....')


if __name__ == "__main__":
    set_start_method("spawn")  # tpu-vm
    URL = "https://huggingface.co/datasets/ArmelR/the-pile-splitted/tree/main/data"
    LOCAL_DIR = '1016/'
    DOWNLOAD_LINK = 'https://huggingface.co/datasets/ArmelR/the-pile-splitted/resolve/'
    # BUCKET_DIR = 'gs://common_datasets/pile'
    BUCKET_DIR = 'gs://common_datasets_us-central2/pile'
    dataset_names = get_url_data_dir(URL)
    dataset_map_links = {}
    total_lines = []
    for dataset_name in dataset_names:
        for mode in  ['train']:
            download_links = extract_data_files(dataset_name, save_dir=LOCAL_DIR, download_link=DOWNLOAD_LINK, mode=mode)
            dataset_map_links[dataset_name] = download_links
            total_lines.extend(download_links)

    random.seed(42)
    random.shuffle(total_lines)

    print(f'total length: {len(total_lines)}')

    num_processes = multiprocessing.cpu_count()
    print(f"num_processes: {num_processes}")

    WORKERS = 10
    perwork_nums = math.ceil(len(total_lines) / WORKERS)
    args = (
        [total_lines[rank * perwork_nums: (rank + 1) * perwork_nums], LOCAL_DIR, BUCKET_DIR] for rank in range(WORKERS)
    )
    pool = multiprocessing.Pool(processes=WORKERS)
    results = pool.map(processed_single_file, args)  # 包含每个进程的返回值
    pool.close()
    pool.join()