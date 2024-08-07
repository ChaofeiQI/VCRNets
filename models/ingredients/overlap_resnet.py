# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from torch.autograd import Variable
import torch.nn as nn
import math, time
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from torch.distributions import Bernoulli
from colorama import init, Fore
init()  # Init Colorama

def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()
        self.block_size = block_size

    def forward(self, x, gamma): 
        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()
            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)
        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]
        offsets = torch.stack(
            [ torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
              torch.arange(self.block_size).repeat(self.block_size),]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()
            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            
        block_mask = 1 - padded_mask
        return block_mask
    

class OvBasicBlockVariant(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, use_se=False):
        super(OvBasicBlockVariant, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.overlap_pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1) # Overlapping Pooling

        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.use_se = use_se
        if self.use_se: self.se = SELayer(planes, 4)

    def forward(self, x):
        self.num_batches_tracked += 1
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se: out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.overlap_pool(out)
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out

class OvResnet(nn.Module):
    def __init__(self, block, n_blocks, indim=3, outdim=640, keep_prob=1.0, avg_pool=False, dropout_rate=0.2, drop_rate=0.0, dropblock_size=5, num_classes=-1, use_se=False):
        super(OvResnet, self).__init__()
        self.inplanes = indim
        self.use_se = use_se
        
        self.layer1 = self._make_layer(block, n_blocks[0], 64,  stride=2, padding=1, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160, stride=2, padding=1, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], outdim, stride=2, padding=1, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], outdim, stride=1, padding=1, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        
        self.downsamp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # Overlapping Pooling重叠池化层(特征下采样或特征降维) 
        self.bn = nn.BatchNorm2d(outdim)
        self.relu = nn.LeakyReLU(0.1)

        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.feat_dim = [outdim, 6, 6]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.num_classes = num_classes
        if self.num_classes > 0: self.classifier = nn.Linear(outdim, self.num_classes)

    def _make_layer(self, block, n_block, planes, stride=1, padding=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        if n_block == 1: layer = block(self.inplanes, planes, stride, padding, downsample, drop_rate, drop_block, block_size, self.use_se)
        else: layer = block(self.inplanes, planes, stride, padding, downsample, drop_rate, self.use_se)
        layers.append(layer)
        self.inplanes = planes * block.expansion
        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block, block_size=block_size, use_se=self.use_se)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.downsamp(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


def OvResNet12(indim=3, outdim=640, keep_prob=1.0, drop_rate=0.2, avg_pool=True, **kwargs):
    """Constructs a OvResNet-12 model.
    """
    model = OvResnet(OvBasicBlockVariant, [1, 1, 1, 1], indim=indim, outdim=outdim, keep_prob=keep_prob, dropout_rate=drop_rate, avg_pool=avg_pool, **kwargs)
    # print(Fore.RED+'*********'* 10)
    # print(Fore.BLUE+'OvResNet12参数:')
    # print(Fore.RED+'*********'* 10)
    # 打印各层参数量
    # for name, param in model.named_parameters():
        # if param.requires_grad: print(f"Layer {name}: {param.numel()} parameters")

    # 计算总参数量
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params}")   # Total parameters
    return model

def OvResNet34s(indim=3, outdim=640, keep_prob=1.0, drop_rate=0.2, avg_pool=False, **kwargs):
    """Constructs a OvResNet34s model.
    """
    model = OvResnet(OvBasicBlockVariant, [2, 3, 4, 2], indim=indim, outdim=outdim, keep_prob=keep_prob, dropout_rate=drop_rate, avg_pool=avg_pool, **kwargs)
    # print(Fore.RED+'*********'* 10)
    # print(Fore.BLUE+'OvResNet34s参数:')
    # print(Fore.RED+'*********'* 10)
    
    # for name, param in model.named_parameters():
        # if param.requires_grad: print(f"Layer {name}: {param.numel()} parameters")
        
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params}")   # Total parameters
    return model



if __name__=='__main__':
    #####################################
    # 实例化 OvResNet-12
    #####################################
    s_time = time.time()
    model = OvResNet12().cuda()
    f_time = time.time()
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print('Model loading time consuming:', f_time-s_time, 's', '\n') 

    for i in range(10): 
        batch_size = 64
        input_tensor = torch.randn(batch_size, 3, 84, 84).cuda()
        output = model(input_tensor)
        print("虚拟输出:", output.shape)     
