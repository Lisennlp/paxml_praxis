#!/bin/bash



step_start=3000
step_end=43000
step_increment=10000
TPU_NAME=llm-jax-v4-32-10
ZONE=us-central2-b
EXPS=("PilePythia7B256x1DynWFFN16HD128Win256AlignedPileEval" "PilePythia7B256x1DynWFFN16HD128Win256AlignedFlanMiniEval")

skip_step=13000
SKIP=0

# scp file
gcloud compute tpus tpu-vm ssh $TPU_NAME  --project=llm-tpu --worker all  --zone $ZONE --command="cd /home/lishengping/;sudo rm -r projects/*; cd projects; git clone -b xd/dev https://github.com/xiaoda99/paxml.git;git clone -b xd/dev https://github.com/xiaoda99/praxis.git"
gcloud compute tpus tpu-vm scp models.py embedding_softmax.py $TPU_NAME:/home/lishengping/projects/praxis/praxis/layers/ --zone=$ZONE  --worker=all  --project=llm-tpu
gcloud compute tpus tpu-vm scp model_params.py  $TPU_NAME:/home/lishengping/projects/paxml/paxml/tasks/lm/ --zone=$ZONE  --worker=all  --project=llm-tpu
gcloud compute tpus tpu-vm scp executors.py main.py checkpoint_creators.py $TPU_NAME:/home/lishengping/projects/paxml/paxml/ --zone=$ZONE  --worker=all  --project=llm-tpu
gcloud compute tpus tpu-vm scp c4.py $TPU_NAME:/home/lishengping/projects/paxml/paxml/tasks/lm/params/c4.py  --zone=$ZONE  --worker=all  --project=llm-tpu

for ((STEP=$step_start; STEP<=$step_end; STEP+=step_increment)); do
    if [ $STEP -lt $skip_step ] && [ $SKIP == 1 ]; then
        continue
    fi
    for EXP in "${EXPS[@]}"; do
	    if [ $STEP == $skip_step ] && [ $EXP == "PilePythia7B256x1DynWFFN16HD128Win256AlignedPileEval" ] && [ $SKIP == 1 ] ; then
            continue
        fi
        echo "$STEP" "$EXP"
        gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo rm -f /tmp/libtpu_lockfile; sudo chmod +777 -R /tmp/tpu_logs/; killall main.py;/home/lishengping/miniconda3/bin/python /home/lishengping/projects/paxml/paxml/main.py --exp=tasks.lm.params.c4.$EXP --job_log_dir=gs://llm_projects_us-central2/log/PilePythia7B256x1DynWFFN16HD128Win256Alignedv4/ 2>&1 --enable_checkpoint_saving=False --eval_on_test=True --eval_model_step=$STEP| tee $EXP.step$STEP.log"
        gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="gsutil cp $EXP.step$STEP.log gs://lllm_projects_us-central2/log/PilePythia7B256x1DynWFFN16HD128Win256Alignedv4/logs/"
    done
done