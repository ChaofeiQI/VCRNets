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
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
import time

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=3):
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

class VisionPerceptionZone_(nn.Module):
    def __init__(self, orig_channels=3, dest_channels=3, image_width=84, image_height=84, **kwargs):
        
        super(VisionPerceptionZone_, self).__init__()
        self.input_channels = orig_channels 
        self.output_channels = dest_channels
        self.image_width = image_width
        self.image_height = image_height
        self.channel_attention = ChannelAttention(self.input_channels*3)

        self.vision = nn.Sequential(
                        nn.Conv2d(in_channels=self.input_channels*3, out_channels=self.output_channels, 
                                  kernel_size=1, stride=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Sigmoid(),
        )

    def fft_input(self, X, truncate_ratio=0.85):
        X_fft = torch.fft.fftn(X, dim=(2, 3))
        C, T, H, W = X.shape
        radius = min(H, W) * truncate_ratio
        idx = torch.arange(-H // 2, H // 2, dtype=torch.float32)
        idy = torch.arange(-W // 2, W // 2, dtype=torch.float32)
        mask = (idx.view(1, 1, H, 1)**2 + idy.view(1, 1, 1, W)**2) <= radius**2
        mask = mask.to(X_fft.device)
        X_fft = X_fft * mask
        X_ifft = torch.fft.ifftn(X_fft, dim=(2, 3)).real
        return X_ifft

    def erase_input(self, image, erase_size=3, erase_num=5):
        image_tensor = image
        height, width = image_tensor.shape[2:]
        size_erase = erase_size 
        num_erase = erase_num   
        for _ in range(num_erase):
            x = np.random.randint(0, width - size_erase)
            y = np.random.randint(0, height - size_erase)
            image_tensor[:, :, y:y+size_erase, x:x+size_erase] = 0
        return image_tensor

    def forward(self, X, tru_ratio=0.85):
        x_era = self.erase_input(X)
        x_fft = self.fft_input(X, tru_ratio) 
        x_cat = torch.cat([x_fft, X, x_era], dim=1)
        out = self.vision(x_cat)
        return out

def VisionPerceptionZone(orig_channels=3, dest_channels=3, image_width=84, image_height=84, **kwargs):
    """Constructs a VisionPerceptionZone ingredient.
    """
    ingredient = VisionPerceptionZone_(orig_channels, dest_channels, image_width, image_height, **kwargs)
    return ingredient


if __name__ == '__main__':
    t1= time.time()
    model = VisionPerceptionZone().cuda()
    t2= time.time()
    print('Model loading time consuming:', t2-t1, 's', '\n') # Time Cost: 1.7192919254302979 s
    
    for i in range(10):
        data = torch.randn(64, 3, 84, 84).cuda() # torch.Size([120, 3, 84, 84])
        out = model(data)                         
        print('Result{}: {}'.format(i,out.shape))             