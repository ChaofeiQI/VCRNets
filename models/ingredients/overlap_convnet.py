# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch, time
import torch.nn as nn
from colorama import init, Fore
init()  # Init Colorama

def Ovconv_block(in_channels, out_channels, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class OvConvNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, dropout_rate=0.1):
        super().__init__()
        #z_dim=64
        self.feat_dim = [64]

        self.encoder = nn.Sequential(
            Ovconv_block(x_dim, hid_dim, 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # Overlapping Pooling重叠池化层(特征下采样或特征降维) 
            Ovconv_block(hid_dim, hid_dim, 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # Overlapping Pooling重叠池化层(特征下采样或特征降维) 
            Ovconv_block(hid_dim, hid_dim, 2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # Overlapping Pooling重叠池化层(特征下采样或特征降维) 
            Ovconv_block(hid_dim, z_dim, 1),
        )
        #self.downsamp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(z_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.encoder(x)
        return x

def OvConv4(x_dim_=3, hid_dim_=64, z_dim_=64, **kwargs):
    """Constructs a OvConv4 model."""
    model = OvConvNet(x_dim=x_dim_, hid_dim=hid_dim_, z_dim=z_dim_, **kwargs)
    # print(Fore.RED+'*********'* 10)
    # print(Fore.BLUE+'OvConv4参数:')
    # print(Fore.RED+'*********'* 10)
    
    # # 打印各层参数量
    # for name, param in model.named_parameters():
    #     if param.requires_grad: print(f"Layer {name}: {param.numel()} parameters")

    # # 计算总参数量
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters: {total_params}")   # Total parameters: 2578494
    return model



if __name__ == '__main__':
    # 实例化模型(OvConv4)
    s_time = time.time()
    model = OvConv4(3, 64, 64).cuda()
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    f_time = time.time()
    print('Model loading time consuming:', f_time-s_time, 's', '\n') 
    
    for i in range(10): 
        Input_tensor = torch.rand(64, 3, 84, 84)
        result = model(Input_tensor.cuda())
        print('Result{}: {}'.format(i, result.shape))
    '''Model loading time consuming: 1.3162314891815186 s 
    Input tensor: torch.Size([64, 3, 84, 84])
    Result0: torch.Size([64, 64, 6, 6])
    Result1: torch.Size([64, 64, 6, 6])
    Result2: torch.Size([64, 64, 6, 6])
    Result3: torch.Size([64, 64, 6, 6])
    Result4: torch.Size([64, 64, 6, 6])
    Result5: torch.Size([64, 64, 6, 6])
    Result6: torch.Size([64, 64, 6, 6])
    Result7: torch.Size([64, 64, 6, 6])
    Result8: torch.Size([64, 64, 6, 6])
    Result9: torch.Size([64, 64, 6, 6])'''
