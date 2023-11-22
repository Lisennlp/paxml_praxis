from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


losses_train = defaultdict(list)
losses_eval = defaultdict(list)

runs = api.runs('baichuan2_13b_constant_lr1e-5')
for run in runs:
    run_id = run.id
    history = run.history()
#     if len(history):
#         history.to_csv(f"wandb_logs_{run_id}.csv")   
    # else:
    #     run.delete()
    print(f'run_id: {run_id}')
    for i, row in history.iterrows():
        if 'train_loss' in row:
            if row['step'] not in losses_train['step']:
                losses_train['step'].append(row['step'])
                losses_train['train_loss'].append(row['train_loss'])
        if 'eval_loss' in row:
            if row['eval_step'] not in losses_eval['step']:
                losses_eval['step'].append(row['eval_step'])
                losses_eval['eval_acc'].append(row['eval_acc'])
                losses_eval['eval_loss'].append(row['eval_loss'])
            

isna_steps = np.isnan(losses_eval['step'])
notna_steps = np.array(losses_eval['step'])[~isna_steps]
eval_loss = np.array(losses_eval['eval_loss'])[~isna_steps]
eval_acc = np.array(losses_eval['eval_acc'])[~isna_steps]

assert len(notna_steps) == len(eval_loss)
step_eval_loss_acc = zip(notna_steps, eval_loss, eval_acc)
sorted_eval_loss_acc = sorted(step_eval_loss_acc, key=lambda x: x[0])

isna_steps = np.isnan(losses_train['step'])
notna_steps = np.array(losses_train['step'])[~isna_steps]
train_loss = np.array(losses_train['train_loss'])[~isna_steps]
assert len(notna_steps) == len(train_loss)
step_train_loss = zip(notna_steps, train_loss)
sorted_train_losses = sorted(step_train_loss, key=lambda x: x[0])


# train curve
xs, ys = zip(*sorted_train_losses)
plt.plot(xs, ys)
plt.title("train loss curve")
plt.xlabel("step")
plt.ylabel("loss")
plt.show()


# eval curve
xs, ys1, ys2 = zip(*sorted_eval_loss_acc)
fig, ax1 = plt.subplots()
ax1.plot(xs, ys1, 'b-', label='loss')
ax1.set_xlabel('step', fontsize=15)
ax1.set_ylabel('loss', color='b', fontsize=15)
ax2 = ax1.twinx()
ax2.plot(xs, ys2, 'r-', label='acc')
ax2.set_ylabel('acc', color='r', fontsize=15)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("eval loss-acc curve", fontsize=20)
plt.show()

