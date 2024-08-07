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
import torch.nn as nn  
import torch.nn.functional as F  
from transformers import AutoModel, AutoTokenizer

# 区域块(RegionBlock)
class RegionBaseBlock(nn.Module):  
    expansion = 1  
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.2):  
        super(RegionBaseBlock, self).__init__()  
        self.conv1 = nn.Conv2d(in_planes, planes//2, kernel_size=9, stride=stride, padding=4, bias=False)  
        self.bn1 = nn.BatchNorm2d(planes//2)  
        self.conv2 = nn.Conv2d(planes//2, planes//2, kernel_size=7, stride=stride, padding=3, bias=False)  
        self.bn2 = nn.BatchNorm2d(planes//2)  
        self.conv3 = nn.Conv2d(planes//2, planes, kernel_size=5, stride=stride, padding=2, bias=False)  
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn4 = nn.BatchNorm2d(planes)  
        self.dropout = nn.Dropout(dropout_rate) 
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()  
        if stride != 1 or in_planes != self.expansion * planes:  
            self.shortcut = nn.Sequential(  
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),  
                nn.BatchNorm2d(self.expansion * planes)  
            )  
    def forward(self, x):  
        out = self.relu(self.bn1(self.conv1(x)))  
        out = self.relu(self.bn2(self.conv2(out)))  
        out = self.relu(self.bn3(self.conv3(out)))  
        out = self.relu(self.bn4(self.conv4(out)))
        # out = self.dropout(out)  
        return out  
  
class RegionBlock(RegionBaseBlock):  
    def __init__(self, in_planes, planes, stride=2):  
        super(RegionBlock, self).__init__(in_planes, planes, stride=stride)  

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_planes // reduction_ratio, in_planes),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, in_channels, height, width):
        super(LearnablePositionalEncoding, self).__init__()
        self.height = height
        self.width = width
        self.pe = nn.Parameter(torch.randn(1, in_channels, height, width))

    def forward(self, x):
        return x + self.pe

class AttentionRegionalPyramid_(nn.Module):
    def __init__(self, in_planes, planes, **kwargs):
        super(AttentionRegionalPyramid_, self).__init__()
        self.LPE = LearnablePositionalEncoding(in_planes,84,84)
        self.RegBk = RegionBlock(in_planes, planes)
        self.channel_attention = ChannelAttention(planes)
        self.downsamp = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x= self.LPE(x) 
        feat = self.RegBk(x)
        att = self.channel_attention(feat) 
        att = self.bn(att)
        att = self.relu(att)
        out = self.downsamp(att)
        out = self.relu(out)
        return out

def AttentionRegionalPyramid(in_planes=3, planes=360, **kwargs):
    """Constructs a AttentionRegionalPyramid ingredient.
    """
    ingred = AttentionRegionalPyramid_(in_planes, planes, **kwargs)
    return ingred



if __name__ == "__main__":
    region_pyramid = AttentionRegionalPyramid(3, 360)
    print(region_pyramid)
    region_pyramid_total_params = sum(p.numel() for p in region_pyramid.parameters() if p.requires_grad)
    print(f"region_pyramid Total parameters: {region_pyramid_total_params}")
    
    for i in range(10):
        batch_size = 8 
        input_tensor = torch.randn(batch_size, 3, 84, 84)
        output = region_pyramid(input_tensor)
        print("输出特征图:", output.shape)
