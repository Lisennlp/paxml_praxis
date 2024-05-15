TPU_TYPE=v4-512
ZONE=us-central2-b
CREATE_ARGS="--version tpu-vm-tf-2.10.0-pod-v4"
TPU_NAME=llm-jax-$TPU_TYPE-10
PROJECT_ID="ntpu-413714"
EXP="PileDCSlimLlama7B4Kx4x256x1v4"
INSTALL=$1
TRAIN=$2

echo TPU_TYPE is $TPU_TYPE
echo ZONE is $ZONE
echo TPU_NAME is $TPU_NAME
echo PROJECT_ID is $PROJECT_ID
echo EXP is $EXP
echo INSTALL is $INSTALL
echo TRAIN is $TRAIN


FLAG=0

while true
do
    tpu_status=$(gcloud alpha compute tpus describe $TPU_NAME --zone=$ZONE --project $PROJECT_ID --format="value[terminator=''](state)")
    echo Tpu status is ${tpu_status}

    if [ "$tpu_status" == "READY" ];then
        FLAG=0
        echo 'Start training......'
        if $INSTALL;then
            gcloud compute tpus tpu-vm scp /home/lishengping/lsp/install_0812.sh  ${TPU_NAME}:~/  --zone=$ZONE  --project $PROJECT_ID --worker=all
            gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=$ZONE --project $PROJECT_ID --worker=all --command="bash install_0812.sh ${ZONE:0:-2} 2>&1 | tee install.log"
        fi
        if $TRAIN;then
            gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="killall main.py;sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/; /home/lishengping/miniconda3/bin/python /home/lishengping/projects/paxml/paxml/main.py --exp=tasks.lm.params.c4.$EXP --job_log_dir=gs://llm_base_models_us-central2/v5p_256/7B/$EXP 2>&1 --enable_checkpoint_saving=True --eval_on_test=True | tee train.log" --project=$PROJECT_ID
        fi
    elif [ "$tpu_status" == "CREATING" ];then
        FLAG=1
        echo 'TPU is creating......'
        sleep 30s
    elif [ $FLAG == 0 ];then
        echo 'TPU is not existed, now start to create......'
        gcloud compute tpus tpu-vm delete ${TPU_NAME} --zone=${ZONE} --project $PROJECT_ID --quiet
        # echo 'Y' | gcloud alpha compute tpus queued-resources delete $TPU_NAME --zone=$ZONE  --project $PROJECT_ID
        # sleep 15s
        gcloud alpha compute tpus queued-resources create $TPU_NAME --node-id $TPU_NAME  --project $PROJECT_ID   --zone=$ZONE   --accelerator-type=$TPU_TYPE ${CREATE_ARGS} --scopes=https://www.googleapis.com/auth/cloud-platform   --preemptible
        sleep 1m
        FLAG=1
    else
        FLAG=1
        echo 'TPU is creating......'
        sleep 10s

    # if [ -z "$tpu_status" ] || [ "$tpu_status" != "READY" ] && [ "$tpu_status" != "CREATING" ]; then
    fi
done