'''
Copyright (c) FTWT
'''

import os
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import numpy as np
import torch.nn as nn
from PIL import Image
from utils.layers import ConvWithMask, BinarizerSTEStatic
import time
import pdb


def fuse_masking(model, batch, args):

    if 'vgg' in args.arch:
        modules = fuse_masking_vgg(model, batch, args)
    elif 'resnet' in args.arch:
        modules = fuse_masking_resnet(model, batch, args, include_first=True)
    elif 'mobilenetv2' in args.arch:
        modules = fuse_masking_mbnetv2(model, batch, args, include_first=True)
    else: #MobileNetV1
        modules = fuse_masking_mbnet(model, batch, args, include_first=True, dw=True, pw=True)

    return modules

def fuse_masking_vgg(model, batch, args):

    sz_mapping = {}

    use_cuda = args.use_cuda
    modules = []
    if use_cuda:
        batch = batch.cuda()

    counter = 0
    embedding_sz = args.embedding
    gt_type = args.gt_type

    model.module.baseline = False

    lst = model.module.features
    nw_lst = []

    i = 0
    while i < len(lst):
        m = lst[i]
        if 'Conv2d' in str(type(m)):
            masked_conv = ConvWithMask(lst[i], lst[i+1], lst[i+2], target = embedding_sz, nclass=args.nclass,
                    ratio=args.mthresh, layerid = counter, do_softmax = args.softmax, mode=args.mode, gt_type=gt_type).to('cuda' if use_cuda else 'cpu')
            nw_lst += [masked_conv]
            modules += [masked_conv]
            m.layerid = counter
            counter += 1

            i += 3
        else:
            nw_lst += [m]
            i += 1

    model.module.nmasked_layers = counter

    model.module.features = nn.Sequential(*nw_lst)

    return modules

def fuse_masking_mbnet(model, batch, args, include_first = True, pw = True, dw=True):

    use_cuda = args.use_cuda

    sz_mapping = {}
    if use_cuda:
        batch = batch.cuda()

    lst = model.module.model
    model.module.baseline = False
    modules, nw_lst = [], []
    embedding_sz = args.embedding
    counter = 0
    gt_type = args.gt_type

    first = lst[0]
    if include_first:
        masked_conv = ConvWithMask(first[0], first[1], first[2], target = embedding_sz, nclass=args.nclass,
                ratio=args.mthresh, layerid = counter, do_softmax = args.softmax, mode=args.mode, gt_type=gt_type).to('cuda' if use_cuda else 'cpu')

        nw_lst += [masked_conv]
        modules += [masked_conv]

        counter += 1
    else:
        nw_lst += [first]

    for j in range(1, len(lst)):
        seq = lst[j]
        if dw:
            masked_conv = ConvWithMask(seq[0], seq[1], seq[2], target = embedding_sz, nclass=args.nclass,
                    ratio=args.mthresh, layerid=counter, do_softmax = args.softmax, mode=args.mode, gt_type=gt_type).to('cuda' if use_cuda else 'cpu')

            nw_lst += [masked_conv]
            modules += [masked_conv]
            counter += 1

        else:
            nw_lst += [seq[0:3]]
        if pw:
            masked_conv = ConvWithMask(seq[3], seq[4], seq[5], target = embedding_sz, nclass=args.nclass,
                    ratio=args.mthresh, layerid=counter, do_softmax = args.softmax, mode=args.mode, gt_type=gt_type).to('cuda' if use_cuda else 'cpu')

            nw_lst += [masked_conv]
            modules += [masked_conv]
            counter += 1
        else:
            nw_lst += [seq[3:6]]

    model.module.nmasked_layers = counter
    model.module.model = nn.Sequential(*nw_lst)

    return modules

def fuse_masking_mbnetv2(model, batch, args, include_first = True):

    sz_mapping = {}

    use_cuda = args.use_cuda
    newfeatures = []
    if use_cuda:
        batch = batch.cuda()

    counter = 0
    embedding_sz = args.embedding

    model.module.baseline = False
    modules = []
    relu = torch.nn.ReLU6()

    if include_first:
        conv = model.module.conv0
        bn = model.module.bn0
        #emb = 32

        model.module.masked_conv0 = ConvWithMask(conv, bn, relu, target = embedding_sz, nclass=args.nclass,
                ratio=args.mthresh, layerid=counter, do_softmax = args.softmax, mode=args.mode, gt_type=args.gt_type).to('cuda' if use_cuda else 'cpu')
        model.module.layerid = counter
        modules += [model.module.masked_conv0]

        conv = model.module.conv1
        bn = model.module.bn1
        model.module.masked_conv1 = ConvWithMask(conv, bn, relu, target = embedding_sz, nclass=args.nclass,
                ratio=args.mthresh, layerid=counter, do_softmax = args.softmax, mode=args.mode, gt_type=args.gt_type).to('cuda' if use_cuda else 'cpu')

        model.module.layerid = counter + 1

        modules += [model.module.masked_conv0]
        modules += [model.module.masked_conv1]


        counter += 2

    lst = model.module.bottlenecks.modules()

    for i, m in enumerate(lst):
        if 'BaseBlock' in str(type(m)):
            m.masked_conv1 = ConvWithMask(m.conv1, m.bn1, relu, target = embedding_sz, nclass=args.nclass,
                    ratio=args.mthresh, layerid = counter, do_softmax = args.softmax, mode=args.mode, gt_type=args.gt_type).to('cuda' if use_cuda else 'cpu')

            m.masked_conv2 = ConvWithMask(m.conv2, m.bn2, relu, target = embedding_sz, nclass=args.nclass,
                    ratio=args.mthresh, layerid = counter+1, do_softmax = args.softmax, mode=args.mode, gt_type=args.gt_type).to('cuda' if use_cuda else 'cpu')

            #No relu after 3rd conv
            m.masked_conv3 = ConvWithMask(m.conv3, m.bn3, nn.Identity(), target = embedding_sz, nclass=args.nclass,
                    ratio=args.mthresh, layerid = counter+2, do_softmax = args.softmax, mode=args.mode, gt_type=args.gt_type).to('cuda' if use_cuda else 'cpu')

            m.baseline = False

            modules += [m.masked_conv1]
            modules += [m.masked_conv2]
            modules += [m.masked_conv3]

            m.layerid = counter
            counter += 3

    model.module.nmasked_layers = counter
    return modules

def fuse_masking_resnet(model, batch, args, include_first = True):

    sz_mapping = {}

    use_cuda = args.use_cuda
    newfeatures = []
    if use_cuda:
        batch = batch.cuda()

    counter = 0
    embedding_sz = args.embedding
    gt_type = args.gt_type

    model.module.baseline = False
    modules = []

    if include_first:
        conv = model.module.conv1
        bn = model.module.bn1
        relu = model.module.relu

        model.module.masked_conv = ConvWithMask(conv, bn, relu, target = embedding_sz, nclass=args.nclass,
                ratio=args.mthresh, layerid=counter, do_softmax = args.softmax, mode=args.mode, gt_type=gt_type).to('cuda' if use_cuda else 'cpu')
        model.module.layerid = counter
        modules += [model.module.masked_conv]

        counter += 1

    lst = model.module.modules()

    for i, m in enumerate(lst):
        if 'BasicBlock' in str(type(m)):
            m.masked_conv1 = ConvWithMask(m.conv1, m.bn1, m.relu, target = embedding_sz, nclass=args.nclass,
                    ratio=args.mthresh, layerid = counter, do_softmax = args.softmax, mode=args.mode, gt_type=gt_type).to('cuda' if use_cuda else 'cpu')
            if args.gtafterblock == 'True':
                #gt after residual, notice m.relu
                m.masked_conv2 = ConvWithMask(m.conv2, m.bn2, m.relu, target = embedding_sz, nclass=args.nclass,
                        ratio=args.mthresh, layerid = counter+1, do_softmax = args.softmax, mode=args.mode, gt_type=gt_type).to('cuda' if use_cuda else 'cpu')
                m.gtafterblock = True
            else:
                #gt before residual
                m.masked_conv2 = ConvWithMask(m.conv2, m.bn2, nn.Identity(), target = embedding_sz, nclass=args.nclass,
                        ratio=args.mthresh, layerid=counter+1, do_softmax = args.softmax, mode=args.mode, gt_type=gt_type).to('cuda' if use_cuda else 'cpu')
                m.gtafterblock = False
            m.baseline = False

            modules += [m.masked_conv1]
            modules += [m.masked_conv2]

            m.layerid = counter
            counter += 2

    model.module.nmasked_layers = counter
    return modules


class MultipleOptimizer(object):
    def __init__(self,op = []):
        self.optimizers = op

    def __getitem__(self, i):
        return self.optimizers[i]

    def add(self,new_opt):
        self.optimizers.append(new_opt)

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def load_state_dict(self,param_ops):
        for op,op_new in zip(self.optimizers,param_ops):
            op.load_state_dict(op_new)

    def get_state_dict(self):
        state_dict = []
        for op in self.optimizers:
            state_dict.append(op.state_dict())
        return state_dict

    def update_lr(self,lr):
        for i,op in enumerate(self.optimizers):
            curlr = lr[i]
            for param_group in op.param_groups:
                param_group['lr'] = curlr


    def update_lr_gamma(self,gammas):
        for i,op in enumerate(self.optimizers):
            gamma = gammas[i]
            for param_group in op.param_groups:
                param_group['lr'] *= gamma

def gather_criteria(criteria):

    ngpus = len(list(criteria.keys()))
    gathered = []
    for n in range(ngpus):
        gathered += [criteria[n].detach().cpu()]

    return torch.cat(gathered)

def get_layer_filter_idx(elem,cum):
    for idx, a in enumerate(cum):
        if elem < a:
            l = idx
            f = elem - cum[idx-1]
            return l,f.item()

def init_predictor_with_ones(model, trainloader, optimizer, use_cuda):

    model.train()
    N = len(trainloader)
    print('Init predictors to output 1 ...')

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx > 500:
            break

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs, mask_logits,_, _ = model(inputs)
        gt_mask = []
        for m in mask_logits:
            gt_mask += [torch.ones_like(m, device=m.device)]

        optimizer.zero_grad()

        mask_loss, mask_acc, per_layer_acc = compute_mask_loss_per_input(gt_mask, mask_logits, targets, lweight=1)

        mask_loss.backward()

        optimizer.step()


        if batch_idx % 100 == 0:
            print('%d/%d .. %.2f'%(batch_idx,N, mask_acc.item()))

def compute_mask_loss_per_input(gt, pred, labels = None, lweight=10.): #Ground truth, predicted, loss weight
    ft = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor
    red = 'mean'  # Loss reduction (sum or mean)
    red = 'none'  # Loss reduction (sum or mean)

    BCEfeat = nn.BCEWithLogitsLoss(reduction=red)
    sigfn = nn.Sigmoid()

    mask_loss, mask_acc = 0, 0
    per_layer_acc = {}

    for layeridx, p in enumerate(pred):
        g = gt[layeridx] #works better as some methods return gt dict
        p = p[:,:g.shape[1],:,:] #Last binay for layer exit binary mask
        g = g.to(p.device).view_as(p)

        filter_weight = 1- (g.squeeze().sum(0)/g.shape[0])
        if red == 'none':
            layerloss = BCEfeat(p, g).mean(0) #* filter_weight.view(-1, 1, 1)
            layerloss = layerloss.sum()
        else:
            layerloss = BCEfeat(p, g)
        #print(layerloss)
        mask_loss += layerloss
        with torch.no_grad():
            binary = BinarizerSTEStatic.apply(0.5, sigfn(p))
            cur_acc = ((binary == g).sum(dim=(1,2,3))/float(g.shape[1])).mean()
            per_layer_acc['layer%03d'%layeridx] = cur_acc.item()*100
            mask_acc += cur_acc

    mask_acc = (mask_acc / len(gt)) *100
    mask_loss = lweight * (mask_loss/len(gt))
    return mask_loss, mask_acc, per_layer_acc

def get_datasetloaders(args):

    print('Loading ', args.dataset)

    if 'CIFAR'.lower() in args.dataset.lower():
        # Data
        print('==> Preparing dataset %s' % args.dataset)

        if args.dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
        else:
            dataloader = datasets.CIFAR100
            num_classes = 100

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


    else:
        raise NotImplementedError("Dataset not supported yet.")

    return trainloader, testloader, num_classes


