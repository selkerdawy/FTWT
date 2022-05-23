from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.layers import BinarizerSTEStatic
from functools import partial
from torch.autograd import Variable

import math
import pdb

from utils.layers import MaskPredictorNN
from thop import profile
from thop.vision.basic_hooks import zero_ops
__all__ = ['resnet', 'resnet56', 'resnet110']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def downsample_basic_block(x, planes):
    x = nn.AvgPool2d(2,2)(x)
    zero_pads = torch.Tensor(
        x.size(0), planes - x.size(1), x.size(2), x.size(3)).zero_()
    if isinstance(x.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([x.data, zero_pads], dim=1))

    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.k_sz = [3, 3]
        self.baseline = True

    def forward_baseline(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def forward_mask(self, x, all_mask_logits, all_gt_mask, prev):

        # Feed-forward
        identity = x

        #Identity path
        if self.downsample is not None:
            identity = self.downsample(x)

        #First conv
        out, all_mask_logits, prev, all_gt_mask, cur_flops = self.masked_conv1(x, all_mask_logits, all_gt_mask, prev)
        FLOPs = cur_flops

        #Second Conv
        if self.gtafterblock:
            #GT after aggregation
            out, all_mask_logits, prev, all_gt_mask, cur_flops = self.masked_conv2(out, all_mask_logits, all_gt_mask, prev, residual=identity)
        else:
            #GT before aggregation
            out, all_mask_logits, prev, all_gt_mask,  cur_flops = self.masked_conv2(out, all_mask_logits, all_gt_mask, prev)

            out += identity
            out = self.relu(out)

        FLOPs += cur_flops

        return out, all_mask_logits, prev, all_gt_mask, FLOPs

    def forward(self, x, all_mask_logits=None, all_gt_mask=None, prev=None):
        if self.baseline:
            return self.forward_baseline(x)
        else:
            return self.forward_mask(x, all_mask_logits, all_gt_mask, prev)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, pruned_arch ='Taylor100C10', num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            nblock = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            nblock = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        #print('Pruned arch: %s %s'%(pruned_arch, str(cfg[pruned_arch])))

        self.inplanes = 16
        self.mask1 = None
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.baseline = True

        self.layer1 = self._make_layer(block, 16, nblock)
        self.layer2 = self._make_layer(block, 32, nblock, stride=2)
        self.layer3 = self._make_layer(block, 64, nblock, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = partial(downsample_basic_block, planes=planes*block.expansion)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_baseline(self,x ):

        #1st Conv layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #Blocks feed-forward
        for lidx, layer in enumerate([self.layer1, self.layer2, self.layer3]):
            out = layer(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def forward_mask(self, x):

        all_mask_logits, all_gt_mask = [], []
        bs = x.shape[0]
        prev = torch.ones(bs) * x.shape[1]

        #1st Conv
        out, all_mask_logits, prev, all_gt_mask, cur_flops = self.masked_conv(x, all_mask_logits, all_gt_mask, prev)
        FLOPs = cur_flops

        for lidx, layer in enumerate([self.layer1, self.layer2, self.layer3]):
            for bidx, m in enumerate(layer.children()):
                out, all_mask_logits, prev, all_gt_mask, cur_flops = m(out, all_mask_logits, all_gt_mask, prev)
                FLOPs += cur_flops

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        FLOPs = FLOPs.to(x.device)

        return out, all_mask_logits, all_gt_mask, FLOPs


    def forward(self, x):
        if self.baseline:
            return self.forward_baseline(x)
        else:
            return self.forward_mask(x)


def resnet56(**kwargs):
    """
    Constructs a ResNet-56 model.
    """

    return ResNet(depth=56, **kwargs)

def resnet110(**kwargs):
    """
    Constructs a ResNet-110 model.
    """

    return ResNet(depth=110, **kwargs)


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
