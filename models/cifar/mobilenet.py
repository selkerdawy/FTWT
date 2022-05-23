import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.layers import  ConvWithMask

__all__ = ['mobilenetv1', 'mobilenetv1_75', 'mobilenetv1_50']

pretrained_TF = False
relu_fn = None
import torch

class TFSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super(TFSamePad, self).__init__()
        self.stride = stride
        if kernel_size != 3:
            raise NotImplementedError('only support kernel_size == 3')

    def forward(self, x):
        if self.stride == 2:
            return F.pad(x, (0, 1, 0, 1))
        elif self.stride == 1:
            return F.pad(x, (1, 1, 1, 1))
        else:
            raise NotImplementedError('only support stride == 1 or 2')

def relu(relu6):
    if relu6:
        return nn.ReLU6(inplace=True)
    else:
        return nn.ReLU(inplace=True)

class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=False, from_TF=False, depth_multiplier=1.0):
        super(MobileNet, self).__init__()

        self.nmasked_layers = 0
        self.baseline = True #For profile in thop
        self.d = depth_multiplier

        global pretrained_TF,  relu_fn, cfg
        pretrained_TF = from_TF
        relu_fn = relu(from_TF)

        if num_classes == 1000:
            self.cfg = cfg['imagenet']
            self.pool_k = 7
        else:
            self.cfg = cfg['cifar']
            self.pool_k = 2

        self.model = self._make_layers(self.cfg, self.d)
        self.pool = nn.AvgPool2d(self.pool_k)
        self.dropout = nn.Dropout(0.2) if dropout else nn.Identity()
        last_layer = int(self.d * 1024)
        self.fc = nn.Linear(last_layer, num_classes)

    def _make_layers(self, cfg, d):

        conv_bn = self.conv_bn
        conv_dw = self.conv_dw

        layers = []
        in_planes = 3
        for i, x in enumerate(self.cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if i == 0: #First layer is normal conv
                layers.append(conv_bn(in_planes, out_planes, stride, d))
            else:
                layers.append(conv_dw(in_planes, out_planes, stride, d))

            in_planes = out_planes

        return nn.Sequential(*layers)

    @staticmethod
    def conv_bn(inp, oup, stride, d):

        oup = int(d * oup)
        layers=[]
        pad = 1

        # PyTorch BN defaults
        eps=1e-5
        momentum=0.1
        if pretrained_TF:
            layers += [TFSamePad(3, stride)]
            pad = 0
            # TF BN defaults
            eps = 1e-3
            momentum = 1e-3

        layers += [
                nn.Conv2d(inp, oup, 3, stride, pad, bias=False),
                nn.BatchNorm2d(oup, eps=eps, momentum=momentum),
                relu_fn]
        return nn.Sequential(*layers)

    @staticmethod
    def conv_dw(inp, oup, stride, d):
        inp = int(d * inp)
        oup = int(d * oup)
        layers=[]
        pad = 1

        # PyTorch BN defaults
        eps=1e-5
        momentum=0.1
        if pretrained_TF:
            layers += [TFSamePad(3, stride)]
            pad = 0
            # TF BN defaults
            eps = 1e-3
            momentum = 1e-3

        layers += [
                nn.Conv2d(inp, inp, 3, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp, eps=eps, momentum=momentum),
                relu_fn,

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, eps=eps, momentum=momentum),
                relu_fn]

        return nn.Sequential(*layers)

    def forward_baseline(self, x):
        x = self.model(x)
        x = self.dropout(self.pool(x))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def forward_mask(self, x):
        all_mask_logits, all_gt_mask = [], []
        FLOPs = 0
        bs = x.shape[0]
        prev = torch.ones(bs) * x.shape[1]

        for i, m in enumerate(self.model.children()):
            if isinstance(m, ConvWithMask):
                x, all_mask_logits, prev, all_gt_mask, cur_flops = m(x, all_mask_logits, all_gt_mask, prev)
                FLOPs += cur_flops
            else:
                x = m(x)

        x = self.dropout(self.pool(x))
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        FLOPs = FLOPs.to(x.device)

        return x, all_mask_logits, all_gt_mask, FLOPs

    def forward(self, x):
        if self.baseline:
            return self.forward_baseline(x)
        else:
            return self.forward_mask(x)

cfg = {
        'cifar': [(32,1), 64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024],
        'imagenet': [(32,2), 64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024],
        }


def mobilenetv1(num_classes, dropout=False, from_TF=False):
    return MobileNet(num_classes, dropout, from_TF, depth_multiplier=1.)

def mobilenetv1_75(num_classes, dropout=False, from_TF=False):
    return MobileNet(num_classes, dropout, from_TF, depth_multiplier=0.75)

def mobilenetv1_50(num_classes, dropout=False, from_TF=False):
    return MobileNet(num_classes, dropout, from_TF, depth_multiplier=0.50)
