#!/bin/bash

start_time=$(date +%s)

ZONE=$1

if [ -z "$ZONE" ]; then
    echo "Error argv zone is not detected"
    exit 1
fi

# 定义允许的ZONE值列表
allowed_zones=("us-east1" "us-west4" "us-central2")

# 检查ZONE的值是否在允许的列表中
if [[ " ${allowed_zones[@]} " =~ " $ZONE " ]]; then
    echo "ZONE value: $ZONE"
else
    echo "ZONE value not in correct choices: ‘us-east1、us-west4、us-central2’, please check zone again..."
    exit 1
fi

CONDA_BUCKET=gs://conda_script_$ZONE
gsutil cp $CONDA_BUCKET'/py310_packages.tar.gz'  ./ &
pid1=$!
HOME="/home/lishengping"

mkdir $HOME'/projects'
# 拉取代码并安装环境
# xd
# git clone -b xd/dev https://github.com/xiaoda99/praxis.git $HOME'/projects/praxis' &
# # lsp
# git clone https://github.com/Lisennlp/praxis.git $HOME'/projects/praxis' &
# git clone -b xd/dev https://github.com/xiaoda99/paxml.git $HOME'/projects/paxml' &

git clone -b paxml https://github.com/Lisennlp/mesh_easy_jax.git
mv mesh_easy_jax  /home/lishengping/projects/

# lsp
git clone https://github.com/Lisennlp/paxml_praxis.git
mv paxml_praxis/*  $HOME'/projects/'
rm -r paxml_praxis

# 安装conda环境
if [ ! -d "$HOME/miniconda3" ]; then
  echo 'Conda is not existed, now start to install...'
  gsutil cp -r $CONDA_BUCKET/Miniconda3-latest-Linux-x86_64.sh  ./
  bash Miniconda3-latest-Linux-x86_64.sh
  echo export PATH=$HOME"/miniconda3/bin:\$PATH" >> ~/.bashrc
  echo 'conda activate base' >> ~/.bashrc
  echo "Conda install finished..."
fi

/home/lishengping/miniconda3/bin/pip install wandb &
# /home/miniconda3/bin/pip install jupyter
/home/lishengping/miniconda3/bin/pip install -r /home/lishengping/projects/mesh_easy_jax/requirements.txt

wait $pid1
pigz -dc py310_packages.tar.gz | tar xv -C  $HOME'/miniconda3/lib/python3.10/'

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "Finished, take time: ${total_time}s."
