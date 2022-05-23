#!/bin/bash

datasetdir='./data'
dataset=cifar10
model=mobilenetv1 #vgg16_bn #resnet56 #mobilenetv2
mlr=1e-1
task=train
wd=5e-4


scratch(){
initpath='None'
init='scratch'
}

train(){
    task=train
    epochs=200
    lr=0.1
    extra='--tb'
    schedule='81 122 151'
    lr_scheduler_b='step'
    #lr_scheduler_b='cosine' #for MbnetV2
}

evaluate(){
    task=evaluate
}

scratch
train
#evaluate

bs=128
extra='--baseline'
echo $initpath
chkpnt='pretrained/'$dataset'/'$model'/'

if [ $task != evaluate ] 
then
    python cifar.py -a $model --dataset $dataset -p $datasetdir\
    --gpu-id 0,1,2,3 \
    --checkpoint $chkpnt --init $initpath \
    --epochs $epochs --lr $lr --mlr $mlr --wd $wd\
    --train-batch $bs --test-batch $bs\
    --schedule $schedule --lr_scheduler_b $lr_scheduler_b \
    $extra 
else
    modelbest=$chkpnt'/model_best.pth.tar'
    python cifar.py -a $model --dataset $dataset -p $datasetdir --checkpoint $chkpnt\
        --evaluate --test-batch 100\
        --init $initpath --resume $modelbest --tb \
        $extra
fi