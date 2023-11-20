
conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch &

#!/bin/bash
PORT=$1
# Step 1: Install Jupyter
# conda install jupyter notebook==7.0.1
# /home/miniconda3/bin/pip install jupyter
# conda install nbconvert==5.4.1 mistune==0.8.4
conda install jupyter
# v4: mistune==3.0.0, v3:mistune==3.0.1
pip install --upgrade 'nbconvert==7.1' 'mistune==3.0.1'
# Step 2: Generate configuration file
jupyter notebook --generate-config

# Step 3: Configuration file path
config_file=~/.jupyter/jupyter_notebook_config.py
# snap
# config_file=~/snap/jupyter/6/.jupyter/jupyter_notebook_config.py


# Step 4: Generate password hash
password_hash=$(python - <<END
from notebook.auth import passwd
print(passwd(''))
END
)

# Step 5: Modify configuration file
cat <<EOT >> $config_file
c.NotebookApp.password = u'$password_hash'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = $PORT
c.NotebookApp.allow_remote_access = True
EOT

# Step 6: Start Jupyter Notebook
# python3 -m notebook
jupyter notebook


# python pro.py --input_file /home/lishengping/common_datasets_us-central2/pythia_pile_idxmaps2/pile_20B_tokenizer_text_document-00000-of-00132.bin --num_shards 133 --output_dir /home/lishengping/jax_base_models_us-central2/pythia_pile_idxmaps/
# PYTHONPATH=utils/gpt-neox/ python utils/batch_viewer.py \
#   --start_iteration 0 \
#   --end_iteration 1000 \
#   --mode save \
#   --save_path ~/test./  \
#   --conf_dir utils/dummy_config.yml 