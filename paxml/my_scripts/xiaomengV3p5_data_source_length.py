import json
from etils import epath
from collections import defaultdict, Counter
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


p = 'gs://jax_llm_data_us-east5/xiaomeng/v3.5/meta_short_long0621.json'

p = epath.Path(p)

lines = []
with p.open('r') as f:
    for line in f:
        line = json.loads(line)
        lines.append(line)


def length_normalize(lens):
    new_lens = []
    for l in lens:
        l = max(1, l)
        v = math.log2(l)
        v = int(v)
        # if v < 10:
        #     v = 10
        new_lens.append(v)
    return new_lens

total_char_count = defaultdict(list)
for file in lines:
    for file_path,  line in file.items():
        for rank in range(5):
            char_count = line[rank][1]
            for k, v in char_count.items():
                total_char_count[k].extend(length_normalize(v))
                # total_char_count[k].extend(v)

sorted_total_char_count = sorted(total_char_count.items(), key=lambda x: x[0])
lengths_array = np.zeros((len(total_char_count), 30))

sources = []
for i, (k, v) in enumerate(sorted_total_char_count):
    char_counter = Counter(v)
    for a, b in char_counter.items():
        # if b < 10:
        #     b = 10
        bei = math.log2(b)
        # bei = 1000 if bei > 1000 else bei
        lengths_array[i, a] = bei
    sources.append(k)
    print(i, k, char_counter, '\n\n')


yticklabels = [f'{2 ** i} <=> 2^{i}' for i in range(30)]
sources_rank = [f'{i}' for i, s in enumerate(sources)]

# 绘制热力图
plt.figure(figsize=(20, 8))
heatmap = sns.heatmap(lengths_array.T, annot=False, cmap='Blues', xticklabels=sources_rank, yticklabels=yticklabels)

# 自定义图例标签
colorbar = heatmap.collections[0].colorbar
tick_labels = colorbar.get_ticks()
custom_labels = [f'2^{int(t)} <=> {2 ** int(t)}' for t in tick_labels]
colorbar.set_ticklabels(custom_labels)

plt.xlabel('Data Source', fontsize=18)
plt.ylabel('Data Length', fontsize=18)
plt.title('Data Length Distribution by Source', fontsize=24)
# plt.xticks(np.arange(max_len))
plt.xticks(fontsize=12, rotation=-45)
plt.show()


div = 4
for s in range(0, len(sources), div):
    for i in range(div):
        index = int(s) + i
        pstr = f'{index}-{sources[index]}'
        print(f'{pstr:<20}', end=' ')
    print()