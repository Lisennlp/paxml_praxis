#!/bin/bash

# 源目录
source_dir="/Users/lishengping/codes/others/paxml_praxis/paxml"
source_dir="/Users/lishengping/codes/jax_projects/paxml_praxis/paxml"
# 目标目录
target_dir="/home/lishengping/projects/paxml"

# 检查命令行参数是否存在
if [ $# -ne 2 ]; then
  echo "请提供文件名和传输方向作为命令行参数。"
  exit 1
fi

# 获取命令行参数
file_name=$1
direction=$2

# 检查传输方向
if [ "$direction" -eq 0 ]; then
  # 从A传至B
  gcloud compute tpus tpu-vm scp ${source_dir}/${file_name}  llm-jax-v3-8-2:${target_dir}/${file_name} --worker all --zone us-east1-d
  echo "文件${file_name}已成功从目录A传输到目录B。"
elif [ "$direction" -eq 1 ]; then
  # 从B传至A
  gcloud compute tpus tpu-vm scp llm-jax-v3-8-0:${target_dir}/${file_name} ${source_dir}/${file_name}  --worker 0 --zone us-central2-b
  echo "文件${file_name}已成功从目录B传输到目录A。"
else
  echo "无效的传输方向。传输方向应为0或1。"
fi


