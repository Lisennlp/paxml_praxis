
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# !pip install accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import sys

sys.path.append('/home/lishengping/conpy')

    
from modeling_baichuan import BaiChuanForCausalLM


model_dir = '/home/lishengping/baichuan-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
config = AutoConfig.from_pretrained('/home/lishengping/conpy', trust_remote_code=True)

model = BaiChuanForCausalLM(config)
w = torch.load('/home/lishengping/baichuan-7b-hf/pytorch_model.bin')
model.load_state_dict(w)

targets = inputs['input_ids']
output = model.forward(**inputs, labels=targets)