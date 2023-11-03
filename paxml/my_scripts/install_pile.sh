# !/bin/bash
# install gcsfuse
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y fuse gcsfuse
mkdir common_datasets_us-central2
gcsfuse common_datasets_us-central2 common_datasets_us-central2
# install some packages
cd /home/lishengping/projects
git clone https://github.com/EleutherAI/pythia.git
cd /home/lishengping/projects/pythia
git submodule update --init --recursive
git reset --hard 899add0f1c71cb27dbf5a7594202584416c0b424

/home/lishengping/miniconda3/bin/pip install torch==1.13
/home/lishengping/miniconda3/bin/pip install deepspeed==0.6.0
/home/lishengping/miniconda3/bin/pip install shortuuid
/home/lishengping/miniconda3/bin/pip install numpy==1.21.1
/home/lishengping/miniconda3/bin/pip install mlxu
/home/lishengping/miniconda3/bin/pip install wandb

cd /home/lishengping/projects/pythia/utils/gpt-neox/megatron/data
sudo apt install make
sudo apt install build-essential
/home/lishengping/miniconda3/bin/pip3 install pybind11
cd /home/lishengping/projects/pythia/utils/gpt-neox/megatron/data
make


# gcloud compute instances create data \
#   --zone=us-central2-b \
#   --machine-type=n2-highcpu-160 \
#   --preemptible \
#   --image-family=ubuntu-2004-lts \
#   --image-project=ubuntu-os-cloud \
#   --boot-disk-size=1500GB \
#   --create-disk=size=1500GB,auto-delete=yes \
#   --custom-extensions=memory=100GiB
