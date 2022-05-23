import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math
import pdb
import matplotlib.pyplot as plt

def get_fmap_criteria(x):
    with torch.no_grad():
        norm_fmap = F.max_pool2d(x.abs(), kernel_size=x.size()[2:]).mean(dim=(2,3)) #squeeze(-1).squeeze(-1)

    return norm_fmap

def get_gt_mask_pratio(x, pruning_ratio):
    acts_at_layer = get_fmap_criteria(x)
    bs = acts_at_layer.shape[0]
    nfilters = acts_at_layer.shape[1]
    nprune = int(pruning_ratio * nfilters)

    gt_mask = torch.ones(bs, nfilters, 1, 1, device=x.device)
    delta = 1e-5
    if pruning_ratio < 0:
        #Include all filters
        return gt_mask
    else:
        #All zeros
        i,j = torch.where(acts_at_layer  <= delta)
        gt_mask[i, j, 0, 0] = 0

        sorted_acts, sorted_idx = torch.sort(acts_at_layer, descending=False, dim=1)
        j = sorted_idx[:,:nprune]
        i = np.asarray(list(range(bs)) * nprune).reshape(nprune,-1).transpose()
        i = torch.from_numpy(i).to(x.device)
        gt_mask[i, j, 0, 0] = 0

        return gt_mask

def get_gt_mask_ratio(x, prob_mass_threshold):

    acts_at_layer = get_fmap_criteria(x)
    bs = acts_at_layer.shape[0]
    nfilters = acts_at_layer.shape[1]

    gt_mask = torch.ones(bs, nfilters, 1, 1, device=x.device)
    delta = 1e-5
    if prob_mass_threshold < 0:
        #Include all filters
        return gt_mask
    else:
        # Include all filters with cumulative contribution < prob_mass_threshold
        normalized_acts = acts_at_layer/acts_at_layer.sum(1, keepdims=True)
        if prob_mass_threshold == 1: #Special case, get all zeroed out features, skip extra computations
            i,j = torch.where(normalized_acts <= delta)
        else:
            sorted_acts, sorted_idx = torch.sort(normalized_acts, descending=True, dim=1)
            cumsum_acts = torch.cumsum(sorted_acts, dim=1)
            i,j = torch.where(cumsum_acts - prob_mass_threshold > delta)
            j = sorted_idx[i,j]
        gt_mask[i, j, 0, 0] = 0

        return gt_mask

def apply_predictor(mask, x, out, mode, gt_type):

    if mask is not None:

        if 'decoupled' in mode:
            #Apply predictor on input x --> detached
            pred_logit = mask(x.detach())
        elif 'joint' in mode:
            x.retain_grad()
            pred_logit = mask(x)
        else:
            raise NotImplementedError("Training mode not supported yet. pass joint or decoupled")

        threshold_fn = BinarizerSTEStatic.apply
        #Generate gt mask from out and apply binary mask
        with torch.no_grad():
            if gt_type == 'uniform':
                gt_mask = get_gt_mask_pratio(out, mask.ratio)
            elif gt_type == 'mass':
                gt_mask = get_gt_mask_ratio(out, mask.ratio)
            else:
                raise NotImplementedError("mass or static.")

            pred_sig = nn.Sigmoid()(pred_logit)
            binary = threshold_fn(0.5, pred_sig)

        return pred_logit, binary, gt_mask

    return None, None, None, None

class BinarizerSTEStatic(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor. Backward is STE"""

    @staticmethod
    def forward(ctx, threshold, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        gradInput = gradOutput.clone()
        #gradInput.zero_()

        return None,gradInput

class AdaptiveNormalize(nn.Module):
    def __init__(self, dim, normalize):
        super(AdaptiveNormalize, self).__init__()
        self.dim = dim
        self.normalize = normalize

    def forward(self, input):
        input_size = input.size()
        GP = input
        nbatches = input.size(0)

        if input.dim() > 2 and input_size[-1] > 1:
            GP = input.norm(p=2, dim=(2,3))
            input = nn.AdaptiveAvgPool2d(self.dim)(input).view(input.size(0),-1)
            input_size = input.size()

        input_ = input.view(nbatches,-1)

        if self.normalize:
            norm = input_.norm(p=2, dim=1)
            _output = torch.div(input_, norm.view(-1, 1).expand_as(input_))
        else:
            _output = input

        output = _output.view(nbatches,-1, 1, 1)

        return output

class MaskPredictorNN(nn.Module):
    def __init__(self, in_features, out_features, target, nclass, ratio, do_softmax):
        super(MaskPredictorNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ratio = ratio

        self.model = torch.nn.Sequential()

        if target == -1:
            self.dim = 1
        else:

            # Calculate the h/w of the features so that after flatten, all layers have nearest 'target' features
            self.dim = round(math.sqrt(target/in_features))

        H = self.dim * self.dim * in_features

        self.model.add_module("AdaptivePooling", AdaptiveNormalize(self.dim, normalize=False))
        if do_softmax == 1:
            self.model.add_module("Softmax",nn.Softmax(dim=1))
        self.model.add_module("1Conv2dMask", torch.nn.Conv2d(H, out_features, 1, bias=True))


    def forward(self, x):
        return self.model(x)

class ConvWithMask(nn.Module):
    def __init__(self, conv, bn, fn, target, nclass, ratio, layerid=0, do_softmax=0, mode='decoupled', gt_type='mass'):
        super(ConvWithMask, self).__init__()

        self.out_features = conv.out_channels
        self.in_features = conv.in_channels
        self.mask = MaskPredictorNN(self.in_features, self.out_features, target=target, nclass=nclass, ratio=ratio, do_softmax=do_softmax)

        self.conv = conv
        self.bn = bn
        self.fn = fn
        self.k = self.conv.kernel_size[0]**2
        self.g = self.conv.groups
        self.layerid=layerid
        self.mode = mode
        self.gt_type = gt_type

    def forward(self, x, all_mask_logits, all_gt_mask, prev, residual=None):

        bias_ops = 0
        bs = x.shape[0]

        out = self.bn(self.conv(x))
        if residual is not None:
            out += residual
        out = self.fn(out) #in-place

        mask_logit, mask_binary, gt_mask = apply_predictor(self.mask, x, out, self.mode, self.gt_type)

        #Apply mask
        out = out * mask_binary

        #Bookkeeping
        all_mask_logits += [mask_logit]
        all_gt_mask += [gt_mask]
        active_cur = mask_binary.mean(dim=(2,3)).sum(1).detach().cpu()
        output_sz  = out.shape[2] * out.shape[3]
        FLOPs = active_cur * output_sz * ((prev // self.g) * self.k + bias_ops)

        return out, all_mask_logits, active_cur, all_gt_mask, FLOPs


