
# # 测试代码 ================================================================================================
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch



# device = "cuda" # the device to load the model onto

# # model = AutoModelForCausalLM.from_pretrained('/home/lishengping/lsp/mistral_instruct_v2', 
# #                                              device_map='auto', 
# #                                             model_type=torch.float16)

# model_path = '/home/lishengping/models/mistral_base_v1'
# model_path = '/home/lishengping/models/mistral_instruct_v2'

# model = AutoModelForCausalLM.from_pretrained(model_path, 
#                                              device_map='auto', 
#                                             )
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# # model = model.half()
# model.eval()


# model = model.half()
# torch.cuda.empty_cache()
# torch.set_grad_enabled(False)

# from collections import defaultdict
# import os

# start = 0
# losses = defaultdict(list)
# for pinx in range(start, 5):
#     data_path = f'data{pinx}.bin'
#     if os.path.exists(data_path):
#         model_inputs = torch.load(data_path)
#     else:
#         text = open(f'/home/lishengping/test_novels/en.{pinx}.txt', 'r').read()
#         text = text[:250000]
#         model_inputs = tokenizer(text, return_tensors='pt')
#         torch.save(model_inputs, data_path)
#     # a = widow_size // 4096
#     for index in range(1, 8, 1):
#         torch.cuda.empty_cache()
#         length = index * 4096
#         # if pinx == start and length < 28679: continue
#         inputs = {k: v[:, :length] for k, v in model_inputs.items()}
#         labels = inputs['input_ids']
#         # print(labels.shape)
#         # results = model(**inputs, labels=labels, use_cache=False, past_key_values=None)
#         results = model(**inputs, labels=labels, use_cache=False)
        
#         print(f'pinx: {pinx}, length: {length}, loss: {results.loss}')
#         losses[pinx].append(results.loss)
#     print('\n')
#     break
# # 测试代码 ================================================================================================





# # plot代码 ================================================================================================
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import math
from mistral_model_longtext_loss import *


ppl = False
# instruct
window_map_logs = {'4096': logs0_ins, '8092': logs1_ins, '16384': logs2_ins, '32768': logs3_ins}
# base
window_map_logs = {'4096': logs0, '8092': logs1, '16384': logs2, '32768': logs3}

def extract_result(logs):
    results = defaultdict(list)
    logs = [l.strip() for l in logs.split('\n') if l.strip()]
    for log in logs:
        result = re.findall('pinx: (\d+), length: (\d+), loss: (.*)', log)[0]
        results[int(result[1])].append(round(float(result[-1]), 4))
    results = {k: round(sum(v) / len(v), 3) for k, v in  results.items()}
    return results

mean_losses = {}
for window, logs in window_map_logs.items():
    result = extract_result(logs)
    result = sorted(result.items(), key=lambda x: x[0])
    mean_losses[window] = dict(result)


for key, values in mean_losses.items():
    x = list(values.keys())
    y = list(values.values())
    if ppl:
        y = [math.exp(a) for a in y]
    if key != '32768':
        plt.scatter(x, y, label=f'ws={key}')
        plt.plot(x, y, linestyle='--', marker='o')
    else:
        plt.scatter(x, y, label=f'ws={key}')
        plt.plot(x, y, linestyle='--', marker='o')

plt.xlabel('Length')
if ppl:
    plt.ylabel('PPL')
    # plt.title('PPL vs Length')
    plt.title('mistral-base-v0.1 PPL vs Length')
    
else:
    plt.ylabel('Loss')
    plt.title('mistral-instruct-v0.2 Loss vs Length')
#     plt.title('mistral-base-v0.1 Loss vs Length')
    
plt.ylim(2.1, 7.15)
plt.xticks(x, x)
# plt.yticks(range(9, 14, 1))
plt.legend()
plt.show()
# # plot代码 ================================================================================================



