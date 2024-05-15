from tensorboardX import SummaryWriter



read_path = 'gs://llm_base_models_us-central2/dcformer/maxtext/410m/llama2_qscale_0511/tensorboard/events.out.tfevents.1715414607.t1v-n-c8a64ddc-w-0'
# llama xd
read_path1 = 'gs://llm_projects/log/summaries/train/PileLlamaMedium/events.out.tfevents.1702379143.t1v-n-44b0873c-w-1.155566.0.v2'
read_path2 = 'gs://llm_projects/log/summaries/train/PileLlamaMedium/events.out.tfevents.1702389862.t1v-n-c61eb4c7-w-2.7463.0.v2'
read_path1 = 'gs://llm_projects/log/summaries/train/PileDCLlamaMediumv4/events.out.tfevents.1702378095.t1v-n-950fd398-w-1.9512.0.v2'

# save_path = 'gs://llm_base_models_us-central2/dcformer/maxtext/410m/llama2_qscale_0511/rewrite_xd_tfevents'
save_path = 'gs://llm_base_models_us-central2/dcformer/maxtext/410m/qknorm0511_scale/rewrite_xd_tfevents'

# 创建一个SummaryWriter对象，指定保存路径
writer = SummaryWriter(save_path)
summaries1 = tf.compat.v1.train.summary_iterator(read_path1)
summaries2 = tf.compat.v1.train.summary_iterator(read_path2)

steps = []
# 遍历summaries并记录标量值
# for i, summaries in enumerate([summaries1, summaries2]):
for i, summaries in enumerate([summaries1]):
    for e in summaries:
        step = e.step
        if i > 0 and step == 0:
            continue
        steps.append(step)
        for v in e.summary.value:
            vtenor = tf.make_ndarray(v.tensor).item()
            if v.tag == 'loss':
                writer.add_scalar('learning/loss', vtenor, step)
            if isinstance(vtenor, float):
                writer.add_scalar(v.tag, vtenor, step)

# 关闭SummaryWriter
writer.close()
