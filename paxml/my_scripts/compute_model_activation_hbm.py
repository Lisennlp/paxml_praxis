core_hbm = 15.48
layers = 12
head_nums = 32
batch_size = 1
length = 8192 * 1
model_dim = 4096
head_dim = model_dim // head_nums
bit_per_value = 2
intermediate_size = 5504
vocab_size = 50257

dropout = False
output_hbm_per_layer = batch_size * length * model_dim * bit_per_value / 1024 ** 3
# print(f'output_hbm_per_layer: {output_hbm_per_layer}G')
output_hbm_all_layer = (layers + 1) * output_hbm_per_layer
print(f'output_hbm_all_layer: {output_hbm_all_layer}G')
# unit: B
# model_size = 7
model_params = layers * (4 * model_dim * model_dim + 3 * model_dim * intermediate_size + 2 * model_dim ) + \
                    2 * vocab_size * model_dim
model_params /= 10 ** 9
model_grad_optimizer_hbm = model_params * 2 + model_params * 2 + model_params * 12
core_nums = 8
model_hbm_per_core = model_grad_optimizer_hbm / core_nums
print(f'model_hbm_per_core: {model_hbm_per_core}G')
# print(f'model + output hbm: {model_hbm_per_core + output_hbm_all_layer}G')


def compute_attn_checkpoint_hbm(dropout=False):
    # input为x。不在这计算
    pre_norm_in = batch_size * length * model_dim * bit_per_value
    # q * k
    qk_in = 2 * batch_size * length * model_dim * bit_per_value
    # softmax(x)
    softmax_in = batch_size * head_nums * length ** 2 * bit_per_value
    # score * v
    score_v = batch_size * head_nums * length ** 2 * bit_per_value +  \
                batch_size * length * model_dim * bit_per_value
    if dropout:
        # mask为byte类型，占一个字节
        drop_mask1 = batch_size * head_nums * length ** 2 * 1
    else:
        drop_mask1 = 0
    # post(x)
    post_in = batch_size * length * model_dim * bit_per_value
    
    if dropout:
        drop_mask2 = batch_size * length * model_dim * bit_per_value
    else:
        drop_mask2 = 0
        
    total = pre_norm_in + qk_in + softmax_in + score_v + drop_mask1 + post_in + drop_mask2
    
    return total / 1024 ** 3

def compute_mlp_checkpoint_hbm(dropout=False):
    # norm(x)
    pre_norm_in = batch_size * length * model_dim * bit_per_value
    # gate(x) or ffn1(x)
    gate_ffn1_in = batch_size * length * intermediate_size * bit_per_value
    # silu(x)
    silu_in = batch_size * length * intermediate_size * bit_per_value
    # ffn2(gate * ffn1_out)
    ffn2_in = 2 * batch_size * length * intermediate_size * bit_per_value
    
    if dropout:
        # mask为byte类型，占一个字节
        drop_mask = batch_size * length * intermediate_size * 1
    else:
        drop_mask = 0
    total = pre_norm_in + gate_ffn1_in + silu_in + ffn2_in
    return total / 1024 ** 3


attention_checkpoint_hbm = compute_attn_checkpoint_hbm(dropout=dropout)
mlp_checkpoint_hbm = compute_mlp_checkpoint_hbm(dropout=dropout)


print(f'attention_checkpoint_hbm: {attention_checkpoint_hbm}G')
print(f'mlp_checkpoint_hbm: {mlp_checkpoint_hbm}G')

final = output_hbm_all_layer + attention_checkpoint_hbm + mlp_checkpoint_hbm + model_hbm_per_core


print(f'final: {final}G')

if final > core_hbm:
    print(f'OOM: {final - core_hbm}G')

