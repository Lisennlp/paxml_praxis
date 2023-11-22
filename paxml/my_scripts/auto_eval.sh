#!/bin/bash

step_start=3000
step_end=143000
step_increment=10000
TPU_NAME=llm-jax-v3-64-10
ZONE=us-east1-d
EXPS=("Pythia7BPileEval" "Pythia12BPileEval" "Pythia7BFlanMiniEval" "Pythia12BFlanMiniEval")

skip_step=13000
SKIP=1

for ((STEP=$step_start; STEP<=$step_end; STEP+=step_increment)); do
    if [ $STEP -lt $skip_step ] && [ $SKIP == 1 ]; then
        continue
    fi
    for EXP in "${EXPS[@]}"; do
        if [[ $EXP == *7B* ]]; then
            SIZE="6.9b"
        else
            SIZE="12b"
        fi

#	if [ $STEP == $skip_step ] && [ $EXP == "Pythia7BPileEval" ]  ; then
	#if [ $STEP == $skip_step ] && [[ $EXP == "Pythia7BPileEval"  ||  $EXP == "Pythia12BPileEval" ]]  ; then
	if [ $STEP == $skip_step ] && [[ $EXP == "Pythia7BPileEval"  ||  $EXP == "Pythia12BPileEval"  || $EXP == "Pythia7BFlanMiniEval" ]]  && [ $SKIP == 1 ]; then
            continue
        fi
        echo "$STEP" "$EXP" "$SIZE"
        gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="killall main.py;/home/lishengping/miniconda3/bin/python /home/lishengping/projects/paxml/paxml/main.py --exp=tasks.lm.params.c4.$EXP --job_log_dir=gs://llm_base_models/pythia/pythia-$SIZE-paxml/ 2>&1 --enable_checkpoint_saving=False --eval_on_test=True --eval_model_step=$STEP| tee $EXP.step$STEP.log"
        gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="gsutil cp $EXP.step$STEP.log gs://llm_base_models/pythia/pythia-$SIZE-paxml/logs/"
    done
done