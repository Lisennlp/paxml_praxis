#!/bin/bash

# 初始步进值
B=0
TPU_NAME=llm-jax-v4-512-10
ZONE=us-central2-b
END=10

# install
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/pip install tiktoken smart_open[gcs] gcsfs orjson" --project=ntpu-413714
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo rm -r /home/lishengping/tokenizer;gsutil cp -r gs://llm_base_models_us-east5/qwen/tokenizer /home/lishengping/" --project=ntpu-413714
# scp
SCRIPT=/Users/lishengping/codes/jax_projects/paxml_praxis/paxml/my_scripts/4k_32k_processed.py
gcloud compute tpus tpu-vm scp $SCRIPT $TPU_NAME:/home/lishengping/processed.py  --zone=$ZONE  --worker=all  --project=ntpu-413714


# 循环直到B值达到10, lt <  le <= gt > 
while [ $B -lt $END ]
do
    W0=$((B * 2))
    W1=$((B * 2 + 1))
    echo "B: $B, W0: $W0, W1: $W1"
    # 运行Python脚本，并将STEP值作为参数传递
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=$W0 --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py $B,0,5 > B$B.train0_5.log 2>&1 &" --project=ntpu-413714
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=$W1 --command="killall processed.py;/home/lishengping/miniconda3/bin/python processed.py $B,5,10 > B$B.train5_10.log 2>&1 &" --project=ntpu-413714
    # 增加STEP值
    B=$((B + 1))
    # 可选：在日志文件中记录每次运行的STEP值
done

# 等待所有后台进程完成
# wait

echo "All processes have finished."