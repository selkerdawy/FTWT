# encoding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t = 6, downsample = False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel)

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)

        # for main path:
        c  = t * input_channel
        # 1x1   point wise conv
        self.conv1 = nn.Conv2d(input_channel, c, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(c)
        # 3x3   depth wise conv
        self.conv2 = nn.Conv2d(c, c, kernel_size = 3, stride = self.stride, padding = 1, groups = c, bias = False)
        self.bn2 = nn.BatchNorm2d(c)
        # 1x1   point wise conv
        self.conv3 = nn.Conv2d(c, output_channel, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channel)

        self.baseline = True

    def forward_baseline(self, inputs):
        # main path
        x = F.relu6(self.bn1(self.conv1(inputs)), inplace = True)
        x = F.relu6(self.bn2(self.conv2(x)), inplace = True)
        x = self.bn3(self.conv3(x))

        # shortcut path
        x = x + inputs if self.shortcut else x

        return x

    def forward_mask(self, inputs, all_mask_logits, all_gt_mask, prev):
        # main path

        x, all_mask_logits, prev, all_gt_mask, cur_flops = self.masked_conv1(inputs, all_mask_logits, all_gt_mask, prev)
        FLOPs = cur_flops

        x, all_mask_logits, prev, all_gt_mask, cur_flops = self.masked_conv2(x, all_mask_logits, all_gt_mask, prev)
        FLOPs += cur_flops

        x, all_mask_logits, prev, all_gt_mask, cur_flops = self.masked_conv3(x, all_mask_logits, all_gt_mask, prev)
        FLOPs += cur_flops

        # shortcut path
        x = x + inputs if self.shortcut else x

        return x, all_mask_logits, prev, all_gt_mask, FLOPs

    def forward(self, x, all_mask_logits=None, all_gt_mask=None, prev=None):
        if self.baseline:
            return self.forward_baseline(x)
        else:
            return self.forward_mask(x, all_mask_logits, all_gt_mask, prev)



class MobileNetV2(nn.Module):
    def __init__(self, output_size, alpha = 1):
        super(MobileNetV2, self).__init__()
        self.output_size = output_size

        self.baseline = True
        # first conv layer
        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t = 1, downsample = False),
            BaseBlock(16, 24, downsample = False),
            BaseBlock(24, 24),
            BaseBlock(24, 32, downsample = False),
            BaseBlock(32, 32),
            BaseBlock(32, 32),
            BaseBlock(32, 64, downsample = True),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 96, downsample = False),
            BaseBlock(96, 96),
            BaseBlock(96, 96),
            BaseBlock(96, 160, downsample = True),
            BaseBlock(160, 160),
            BaseBlock(160, 160),
            BaseBlock(160, 320, downsample = False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(int(320*alpha), int(1280*alpha), kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(int(1280*alpha))
        self.fc = nn.Linear(int(1280*alpha), output_size)

        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_baseline(self, inputs):

        # first conv layer
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace = True)
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = F.relu6(self.bn1(self.conv1(x)), inplace = True)
        # assert x.shape[1:] == torch.Size([1280,8,8])

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


    def forward_mask(self, inputs):

        all_mask_logits, all_gt_mask = [], []
        FLOPs = 0
        bs = inputs.shape[0]
        prev = torch.ones(bs) * inputs.shape[1]

        x, all_mask_logits, prev, all_gt_mask, cur_flops = self.masked_conv0(inputs, all_mask_logits, all_gt_mask, prev)
        FLOPs = cur_flops

        for i, m in enumerate(self.bottlenecks.children()):
            #BaseBlock
            x, all_mask_logits, prev, all_gt_mask, cur_flops = m(x, all_mask_logits, all_gt_mask, prev)
            FLOPs += cur_flops


        x, all_mask_logits, prev, all_gt_mask, cur_flops = self.masked_conv1(x, all_mask_logits, all_gt_mask, prev)
        FLOPs += cur_flops

        '''
        # first conv layer
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace = True)
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = F.relu6(self.bn1(self.conv1(x)), inplace = True)
        # assert x.shape[1:] == torch.Size([1280,8,8])
        '''

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        FLOPs = FLOPs.to(x.device)

        return x, all_mask_logits, all_gt_mask, FLOPs

    def forward(self, x):
        if self.baseline:
            return self.forward_baseline(x)
        else:
            return self.forward_mask(x)

def mobilenetv2(num_classes, dropout=False, from_TF=False):
    return MobileNetV2(num_classes)

def mobilenetv2_50(num_classes, dropout=False, from_TF=False):
    return MobileNetV2(num_classes, alpha=0.5)

def mobilenetv2_25(num_classes, dropout=False, from_TF=False):
    return MobileNetV2(num_classes, alpha=0.25)
