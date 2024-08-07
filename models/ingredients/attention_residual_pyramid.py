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

class ResidualBlock(nn.Module):  
    expansion = 1    
    def __init__(self, in_planes, planes, stride=1, kernel_size=3, dropout_rate=0.2):  
        super(ResidualBlock, self).__init__()  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)  
        self.bn1 = nn.BatchNorm2d(planes)  
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)  
        self.bn2 = nn.BatchNorm2d(planes)  
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)  
        self.bn3 = nn.BatchNorm2d(planes)  
        self.downsamp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # Overlapping Pooling重叠池化层(特征下采样或特征降维) 
        self.relu = nn.LeakyReLU(0.1)

        self.shortcut = nn.Sequential()  
        if stride != 1 or in_planes != self.expansion * planes:  
            self.shortcut = nn.Sequential(  
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),  
                nn.BatchNorm2d(self.expansion * planes)  
            )  
  
    def forward(self, x):  
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))  
        out = F.relu(out)  
        out += self.shortcut(x)  
        out = F.relu(out)  
        out = self.bn3(self.conv3(out))  
        out = self.downsamp(out)
        out = self.relu(out)

        return out  

class ResidualBlock9x9(ResidualBlock):  
    def __init__(self, in_planes, planes, stride=2):  
        super(ResidualBlock9x9, self).__init__(in_planes, planes, stride, kernel_size=9)  
  
class ResidualBlock7x7(ResidualBlock):  
    def __init__(self, in_planes, planes, stride=2):  
        super(ResidualBlock7x7, self).__init__(in_planes, planes, stride, kernel_size=7)  
  
class ResidualBlock5x5(ResidualBlock):  
    def __init__(self, in_planes, planes, stride=2):  
        super(ResidualBlock5x5, self).__init__(in_planes, planes, stride, kernel_size=5)  

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


class AttentionResidualPyramid_(nn.Module):
    def __init__(self, in_planes, planes, **kwargs):
        super(AttentionResidualPyramid_, self).__init__()
        self.LPE = LearnablePositionalEncoding(in_planes,84,84)
        self.ResBk9x9 = ResidualBlock9x9(in_planes, planes//3)
        self.ResBk7x7 = ResidualBlock7x7(in_planes, planes//3)
        self.ResBk5x5 = ResidualBlock5x5(in_planes, planes//3)
        self.channel_attention = ChannelAttention(planes)
        self.downsamp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x= self.LPE(x)
        res9x9 = self.ResBk9x9(x)
        res7x7 = self.ResBk7x7(x)
        res5x5 = self.ResBk5x5(x)
        feat = torch.cat((res9x9, res7x7, res5x5), dim=1)  # 在通道维度（dim=1）上拼接
        att = self.channel_attention(feat)                  # 输出特征图: torch.Size([8, 240, 11, 11])
        att = self.bn(att)
        att = self.relu(att)
        out = self.downsamp(att)
        out = self.relu(out)
        return out

def AttentionResidualPyramid(in_planes=3, planes=360, **kwargs):
    """Constructs a AttentionResidualPyramid ingredient.
    """
    ingred = AttentionResidualPyramid_(in_planes, planes, **kwargs)
    return ingred



if __name__ == "__main__":
    residual_pyramid = AttentionResidualPyramid(3, 360).cuda()
    residual_pyramid_total_params = sum(p.numel() for p in residual_pyramid.parameters() if p.requires_grad)
    print(f"residual_pyramid Total parameters: {residual_pyramid_total_params}")
    
    for i in range(10):
        batch_size = 8
        input_tensor = torch.randn(batch_size, 3, 84, 84).cuda()
        output = residual_pyramid(input_tensor)
        print("输出特征图:", output.shape)
