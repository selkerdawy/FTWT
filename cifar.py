'''
Training script for dynamic pruning Fire Together Wire Together (FTWT)
'''

from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import models.cifar as models
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import torchvision.transforms as transforms
import torchvision
import cv2
import matplotlib as mpl
import matplotlib.cm as mpl_color_map
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
import pdb
import numpy as np
from pprint import pformat
from thop import profile
from thop.vision.basic_hooks import zero_ops

from utils.utils import get_datasetloaders, compute_mask_loss_per_input, init_predictor_with_ones, MultipleOptimizer, fuse_masking
from utils import Bar, Logger, FileLogger, AverageMeter, accuracy, mkdir_p, savefig, TensorboardLogger


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100, ImageNet Dynamic Pruning Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-p', '--data-path', default='./data', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--lr_scheduler_b', type=str, default='step', choices=('step','cosine'), 
                        help='type of the scheduler')
parser.add_argument('--schedule_b', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--init', default='', type=str, help='Init weights path')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')

# Mask options
parser.add_argument('--mthresh', type=float, default=1.0, help='Percentage of mass of heat maps to keep')
parser.add_argument('--mode', default='decoupled', type=str, help='Training mode, joint (fully grad) or decoupled')
parser.add_argument('--softmax', default=1, type=int, help='Softmax after embedding (1) or not (0)')
parser.add_argument('--mlr', default=0.1, type=float, help='initial mask predictor learning rate')
parser.add_argument('--embedding', default=-1, type=int, help='Embedding size, -1 for plain adaptive pooling.')
parser.add_argument('--gt-type', default='mass', type=str, help='mass or static.')

# Miscs
parser.add_argument('--warmup', default=1, type=int, help='Ratio of number of epochs to learn predictors only then joint training.')
parser.add_argument('--cooldown', default=0.5, type=float, help='Ratio of number of epochs before freezing predictors.')
parser.add_argument('--headinit', default='random', type=str, help='Init head with random or train till output is 1')
parser.add_argument('--gtafterblock', type=str, default='False')
parser.add_argument('--tb', action='store_true', help='Log on Tensorboard')


parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--baseline', action='store_true',
                    help='Baseline training without mask prediction')
#Device options
parser.add_argument('--gpu-id', default='None', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100'

# Use CUDA
if args.gpu_id != 'None':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

use_cuda = torch.cuda.is_available()
args.use_cuda = use_cuda

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

nw_profile = {
        'cifar10_mobilenetv1': (46355456.0,3217226.0), 'cifar10_mobilenetv1_75':(26509056.,1824250.), 'cifar10_mobilenetv1_50':(12167680.0,823722.0),
        'cifar10_mobilenetv2':(91154944.,2296922.),'cifar10_mobilenetv2_50':(78604544.,587466.), 'cifar10_mobilenetv2_25':(26688000.,249202.),
        'cifar10_mobilenetv1_25':(3331328.,215642.),
        'cifar10_resnet56':(125485760.,853018.), 'cifar10_vgg16_bn':(313478144.,14728266.),
        }

best_acc = 0  # best test accuracy

def main():
    global best_acc, compute_mask_loss, state
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    time_point = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) if not args.evaluate else "eval"
    textfile = "%s/log_%s.txt" % (args.checkpoint, time_point)
    stdout = Logger(textfile)
    sys.stdout = stdout
    sys.stderr = stdout
    print(" ".join(sys.argv))
    print(args)

    trainloader, testloader, num_classes = get_datasetloaders(args)

    args.nclass = num_classes

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    block_name=args.block_name
                )
    elif args.arch.startswith('mobilenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    dropout=False,
                    from_TF=False
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    batch = next(iter(trainloader))[0]
    model.eval()

    precalc = args.dataset + '_' + args.arch
    if precalc in nw_profile:
        macs, params = nw_profile[precalc]
    else:
        macs, params = profile(model, inputs=(batch, ), verbose=False, custom_ops={torch.nn.BatchNorm2d: zero_ops, torch.nn.ReLU: zero_ops})
        bs = batch.shape[0]
        print('%s_%s:(%f,%f)'%(args.dataset, args.arch, macs/bs,params))
        print('Add macs, params to nw_profile to avoid attributes added by profile later on in the model')
        pdb.set_trace()

    print('    Total params: %.2fM, FLOPs: %.4G' % (params*1e-6, macs*1e-9))
    print(macs, params)
    args.macs = macs

    model = torch.nn.DataParallel(model).cuda()
    batch = batch.cuda()
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    if args.init and args.init != 'None':
        # Load checkpoint.
        print('==> Init model from checkpoint ', args.init)
        checkpoint = torch.load(args.init)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        # Check if model is saved with DataParallel
        if not list(checkpoint.keys())[0].startswith('module'):
            checkpoint = {'module.'+k : v for k,v in checkpoint.items()}
        model.load_state_dict(checkpoint, strict=False)

        #test_loss, test_acc = test_baseline(testloader, [model, None], criterion, start_epoch, use_cuda)

    if not args.baseline:
        args.modules = fuse_masking(model, batch, args)

    print(model)
    backbone_lst = {'params': [], 'weight_decay': args.weight_decay, 'lr':args.lr}
    mask_lst = {'params': [], 'weight_decay': 0.0, 'lr':args.mlr}

    for n, m in model.named_parameters():
        if 'Mask' in n:
            mask_lst['params'] = mask_lst['params'] + [m]
        else:
            backbone_lst['params'] = backbone_lst['params'] + [m]
        m.tname = n

    opts = MultipleOptimizer()
    optimizer_backbone = optim.SGD([backbone_lst], momentum=args.momentum)
    opts.add(optimizer_backbone)
    
    # scheduler for model params
    if args.lr_scheduler_b == "step":
        print('Creating step optimizer with schedule ', args.schedule_b)
        scheduler_b = torch.optim.lr_scheduler.MultiStepLR(optimizer_backbone, milestones=args.schedule_b, gamma=args.gamma)
    elif args.lr_scheduler_b == "cosine":
        print('Creating cosine optimizer ...')
        scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_backbone, T_max=args.epochs, eta_min=0.0)

    if not args.baseline:
        optimizer_mask = optim.SGD([mask_lst], momentum=args.momentum)
        opts.add(optimizer_mask)
    
    # Resume
    title = args.dataset + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        opts.load_state_dict(checkpoint['optimizers'])
        model.load_state_dict(checkpoint['state_dict'])

        logger = FileLogger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = FileLogger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        if args.baseline:
            _, test_acc = test_baseline(testloader, [model, None], criterion, start_epoch, use_cuda)
        else:
            _, test_acc = test(testloader, [model, None], criterion, start_epoch, use_cuda, applyMask=True)
        print('Test Acc:  %.2f' % test_acc)
        return

    # Train and val

    if 'random' not in args.headinit and not args.baseline and not args.resume:
        init_predictor_with_ones(model, trainloader, opts[-1], use_cuda)

    for epoch in range(start_epoch, args.epochs):

        #adjust_learning_rate(opts, epoch)
        state['lr'] =  opts[0].param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        if args.baseline:
            train_loss, train_acc = train_baseline(trainloader, [model, None], criterion, opts, epoch, use_cuda)
            _, test_acc = test_baseline(testloader, [model, None], criterion, epoch, use_cuda)
        else:
            train_loss, train_acc = train(trainloader, [model, None], criterion, opts, epoch, use_cuda)
            _, test_acc = test(testloader, [model, None], criterion, epoch, use_cuda, applyMask=True)
        
        scheduler_b.step()
        # append logger file
        logger.append([state['lr'], train_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizers' : opts.get_state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    print('Best acc:')
    print(best_acc)

    logger.close()
    #logger.plot()
    #savefig(os.path.join(args.checkpoint, 'log.eps'))

def train(trainloader, models, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model = models[0]

    model.train()

    global iteration, trainingmode
    trainingmode = "Train"

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses, macc, = AverageMeter(), AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader), stream=sys.stdout)
    #bar = Bar('Processing', max=len(trainloader), stream=sys.stdout.terminal)

    mask_weight = 0.0 if epoch >= int(args.cooldown*args.epochs) else 1
    flops, cnt = 0., 0.

    sigfn = nn.Sigmoid()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs = torch.autograd.Variable(inputs, requires_grad=True) #To work in joint mode, we must allow input grad
        iteration = batch_idx + epoch * len(trainloader)

        outputs, mask_logits, gt_mask, cur_flops = model(inputs)
        flops += cur_flops.sum()
        cnt += inputs.shape[0]

        loss = criterion(outputs, targets)
        mask_loss, mask_acc, per_layer_acc = compute_mask_loss_per_input(gt_mask, mask_logits, targets, lweight=mask_weight)
        loss += mask_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        for met, l in zip([losses, macc], [loss, mask_acc]):
            met.update(l.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # plot progress
        bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | mask_acc: {macc:.2f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    macc=macc.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    avg_flops = (flops/cnt).item()
    reduction =  100*(1-(avg_flops/args.macs))
    print(' Original total FLOPs: %.4G, dynamic FLOPs %.4G. Reduction (%.2f)' % (args.macs*1e-9, avg_flops*1e-9, reduction))
    print(avg_flops, args.macs)

    return (losses.avg, top1.avg)

def test(testloader, models, criterion, epoch, use_cuda, applyMask):
    global best_acc
    global iteration, trainingmode
    trainingmode = "Validatation"

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses, mloss, macc, = AverageMeter(), AverageMeter(), AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model = models[0]

    model.eval()
    end = time.time()

    bar = Bar('Processing', max=len(testloader), stream=sys.stdout)
    #bar = Bar('Processing', max=len(testloader), stream=sys.stdout.terminal)

    mask_weight = 0.0 if epoch >= int(args.cooldown*args.epochs) else 1
    flops, cnt = 0., 0.

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs = torch.autograd.Variable(inputs, requires_grad=True) #To work in joint mode, we must allow input grad
        iteration = batch_idx + epoch * len(testloader)

        # compute output
        outputs, mask_logits, gt_mask, cur_flops = model(inputs)
        flops += cur_flops.sum()
        cnt += inputs.shape[0]

        loss = criterion(outputs, targets)
        mask_loss, mask_acc, per_layer_acc = compute_mask_loss_per_input(gt_mask, mask_logits, targets, lweight=mask_weight)
        loss += mask_loss

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        for met, l in zip([losses, macc], [loss, mask_acc]):
            met.update(l.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | mask_acc: {macc:.2f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    macc=macc.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )

        bar.next()
    bar.finish()

    avg_flops = (flops/cnt).item()
    reduction =  100*(1-(avg_flops/args.macs))
    print(' Original total FLOPs: %.4G, dynamic FLOPs %.4G. Reduction (%.2f)' % (args.macs*1e-9, avg_flops*1e-9, reduction))
    print(avg_flops, args.macs)

    return (losses.avg, top1.avg)

def train_baseline(trainloader, models, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model = models[0]

    model.train()

    global iteration, trainingmode

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader), stream=sys.stdout)#.terminal)
    #bar = Bar('Processing', max=len(trainloader), stream=sys.stdout.terminal)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        iteration = batch_idx + epoch * len(trainloader)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # plot progress
        bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, top1.avg)


def test_baseline(testloader, models, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model = models[0]
    model.eval()

    end = time.time()

    bar = Bar('Processing', max=len(testloader), stream=sys.stdout)#.terminal)
    #bar = Bar('Processing', max=len(testloader), stream=sys.stdout.terminal)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    top1=top1.avg,
                    top5=top5.avg,
                    )

        bar.next()
    bar.finish()

    return (-1, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        print('Saving new best ', state['acc'])
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        gamma = 0.0 if epoch >= int(args.cooldown*args.epochs) else 1
        optimizer.update_lr_gamma([args.gamma, gamma])

if __name__ == '__main__':
    main()
