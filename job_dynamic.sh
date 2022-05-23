#!/bin/bash

datasetdir='./data'
dataset=cifar10
model=mobilenetv1 #vgg16_bn #resnet56
mlr=1e-1
task=train
wd=5e-4

pretrained(){
    initpath='pretrained/'$dataset'/'$model'/model_best.pth.tar'
    init='pretrained'
}
trainwithpred(){
    task=train
    epochs=200
    lr=1e-2
    extra='--tb'
    schedule='81 122 151'
    lr_scheduler_b='step'
    #lr_scheduler_b='cosine' #for MbnetV2
}
finetunewithpred(){
    task=finetune
    epochs=50
    lr=1e-3
    extra='--tb'
    schedule='29 39 49'
    lr_scheduler_b='step'
}
evaluate(){
    task=evaluate
}

pretrained
trainwithpred

bs=128
mode='decoupled' #Or joint
gttype='mass' #Or uniform
mthresh=1.0 #Keep top {mthresh}% of heatmap mass in case of gttype=mass, or top {mthresh}% filters (uniform pruning) in case of gttype=uniform

echo $initpath
chkpnt='dynamic-ftwt/'$dataset'/'$task'_'$model'_lr'$lr'_mthresh'$mthresh'_'$mode'_'$gttype'_'$lr_scheduler_b

if [ $task != evaluate ] #Train or finetune
then
    python cifar.py -a $model --dataset $dataset -p $datasetdir\
    --gpu-id 0,1,2,3 \
    --checkpoint $chkpnt --init $initpath \
    --epochs $epochs --lr $lr --mlr $mlr --wd $wd\
    --train-batch $bs --test-batch $bs \
    --schedule $schedule --lr_scheduler_b $lr_scheduler_b \
    --mthresh $mthresh --mode $mode --gt-type $gttype\
    $extra 
else
    modelbest=$chkpnt'/model_best.pth.tar'
    python cifar.py -a $model --dataset $dataset -p $datasetdir --checkpoint $chkpnt\
        --evaluate --test-batch 100\
        --init $initpath --resume $modelbest --tb \
        $extra
fi

