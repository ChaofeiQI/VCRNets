# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import torch,copy,math
import torch.nn as nn
import torch.nn.functional as F
from  .base import register,make
from .ingredients.attention_regional_pyramid  import  AttentionRegionalPyramid
from .ingredients.attention_residual_pyramid  import  AttentionResidualPyramid
from .ingredients.vision_path      import  VisionPerceptionZone
from .ingredients.overlap_convnet  import  OvConv4 
from .ingredients.overlap_resnet   import  OvResNet12
from .ingredients.self_attention   import  PositionEmbeddingSine, Transformer_Layer

############################
# VCNet4
############################
@register('vcnet4-basic-block')
class VCNet4Block(nn.Module):
    def __init__(self, orig_channels=3, dest_channels=9, feat_channels=64, width=84, height=84, dropout_rate=0.1, 
                 use_region_sensing=False, use_self_attention=False, use_pixel_level_sensing=False, self_attention_kwargs={}, **kwargs):
        super(VCNet4Block, self).__init__()
        self.use_region_sensing = use_region_sensing
        self.use_self_attention = use_self_attention           
        self.self_attention_kwargs = self_attention_kwargs     
        self.use_pixel_level_sensing = use_pixel_level_sensing
        if (not self.use_region_sensing) and (not self.use_pixel_level_sensing):
            self.vision = VisionPerceptionZone(orig_channels, dest_channels, width, height)
        if self.use_region_sensing:
            self.AtRegPy = AttentionRegionalPyramid(dest_channels, feat_channels)
            # self.AtResPy = AttentionResidualPyramid(dest_channels, feat_channels)
            if self.use_self_attention:
                self.pe = PositionEmbeddingSine(num_pos_feats=feat_channels//2)
                self.transformer = Transformer_Layer(multi_head=8, embedding_size=feat_channels, pre_normalize=False)
        if self.use_pixel_level_sensing: self.OvCV4 = OvConv4(dest_channels, hid_dim_=feat_channels, z_dim_=feat_channels)
        self.bn = nn.BatchNorm2d(feat_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)  

    def forward(self, x, sideout=False):
        out, att_out, ov_out =None, None, None
        sideout_dict = {}
        if (not self.use_region_sensing) and (not self.use_pixel_level_sensing):
            x = self.vision(x)                                        
        if self.use_region_sensing:
            feat = self.AtRegPy(x)                                   
            att_out = None
            if self.use_self_attention:
                B, _, H, W = feat.shape
                shape = {}
                shape['B'], shape['H'], shape['W'] = B, H, W
                pos_sine = self.pe(feat)                                   
                att_out = self.transformer(feat.permute(0,2,3,1).contiguous(), shape, pos_sine)
                att_out = att_out.view(B,H,W,-1).permute(0,3,1,2).contiguous() 
            else: att_out = feat
            
        if self.use_pixel_level_sensing: ov_out = self.OvCV4(x)
        if (not self.use_region_sensing) and  (not self.use_pixel_level_sensing):
            out = x
        elif self.use_region_sensing and  (not self.use_pixel_level_sensing):
            out = att_out
        elif (not self.use_region_sensing) and  self.use_pixel_level_sensing:
            out = ov_out
        elif self.use_region_sensing and self.use_pixel_level_sensing:
            out = torch.cat([att_out, ov_out], dim=1)
        if self.use_region_sensing or self.use_pixel_level_sensing:
            out = self.bn(out)
            out = self.relu(out)
            out = self.dropout(out)
        if sideout: return out, sideout_dict
        else: return out


@register('vcnet4-encoder')
class EncoderVCNet4(nn.Module):
    def __init__(self, orig_channels=3, dest_channels=9, feat_channels=64, width=84, height=84, dropout_rate=0.2, 
                 use_region_sensing_list=[False, False, False], use_self_attention_list=[False, False, False], 
                 use_pixel_level_sensing_list=[False, False, False], self_attention_kwargs={}, **kwargs):
        super().__init__()
        self.orig_channels, self.dest_channels, self.feat_channels =  orig_channels, dest_channels, feat_channels
        self.width, self.height, self.dropout_rate = width, height, dropout_rate
        channels = [orig_channels, dest_channels, feat_channels]# 3, 9, 64
        self.n = len(channels)
        
        def create_list(use_region_sensing=False, use_self_attention=False, use_pixel_level_sensing=False, self_attention_kwargs={}, **kwargs):
            return make('vcnet4-basic-block', orig_channels=self.orig_channels, dest_channels=self.dest_channels, 
                feat_channels=self.feat_channels, width=self.width, height=self.height, dropout_rate=self.dropout_rate, 
                use_region_sensing = use_region_sensing, use_self_attention = use_self_attention, 
                use_pixel_level_sensing = use_pixel_level_sensing, self_attention_kwargs = self_attention_kwargs, **kwargs)
                                       
        self.vision =  create_list(use_region_sensing_list[0], use_self_attention_list[0], use_pixel_level_sensing_list[0], self_attention_kwargs={})
        self.branch1 = create_list(use_region_sensing_list[1], use_self_attention_list[1], use_pixel_level_sensing_list[1], self_attention_kwargs={})
        self.branch2 = create_list(use_region_sensing_list[2], use_self_attention_list[2], use_pixel_level_sensing_list[2], self_attention_kwargs={})
        self.out_dim = channels[2]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, sideout=False, branch=-1):
        def sideout_func(x, attr_name, sideout=False):
            blocks = getattr(self, attr_name)
            if sideout:
                sideout_dict = {}
                x, s = blocks(x, sideout=True)
                for layer_name, layer in s.items():
                    sideout_dict["{}.{}.{}".format(attr_name, i, layer_name)] = layer
                return x, sideout_dict
            else:
                x = blocks(x)
                return x

        if branch == 1: branch_attr_name = "branch1"
        elif branch == 2: branch_attr_name = "branch2"
        else: raise ValueError()

        if sideout:
            sideout_dict = {}
            x, s_vision = sideout_func(x, sideout=True, attr_name="vision")
            x, s_branch = sideout_func(x, sideout=True, attr_name=branch_attr_name)
            sideout_dict.update(s_vision)
            sideout_dict.update(s_branch)
            sideout_dict['before_avgpool'] = x
        else:
            x = sideout_func(x, attr_name="vision")
            x = sideout_func(x, attr_name=branch_attr_name)
            
        if sideout: return x, sideout_dict
        else: return x

@register('encoder-vcnet4')
def vcnet4_encoder(**kwargs): return EncoderVCNet4(orig_channels=3, dest_channels=9, feat_channels=64, **kwargs)


############################
# VCNet12
############################
@register('vcnet12-basic-block')
class VCNet12Block(nn.Module):
    def __init__(self, orig_channels=3, dest_channels=9, feat_channels=640, width=84, height=84, dropout_rate=0.2, 
                 use_region_sensing=False, use_self_attention=False, use_pixel_level_sensing=False, self_attention_kwargs={}, **kwargs):
        super(VCNet12Block, self).__init__()
        self.use_region_sensing = use_region_sensing
        self.use_self_attention = use_self_attention            
        self.self_attention_kwargs = self_attention_kwargs      
        self.use_pixel_level_sensing = use_pixel_level_sensing
        if (not self.use_region_sensing) and (not self.use_pixel_level_sensing):
            self.vision = VisionPerceptionZone(orig_channels, dest_channels, width, height)
        if self.use_region_sensing:
            self.AtRegPy = AttentionRegionalPyramid(dest_channels, feat_channels//4)
            # self.AtResPy = AttentionResidualPyramid(dest_channels, feat_channels)
            if self.use_self_attention:
                self.pe = PositionEmbeddingSine(num_pos_feats=feat_channels//8)
                self.transformer = Transformer_Layer(multi_head=8, embedding_size=feat_channels//4, pre_normalize=False)
            self.upsamp = nn.Conv2d(feat_channels//4, feat_channels, kernel_size=1, stride=1, bias=True)  
        if self.use_pixel_level_sensing: self.OvRN12 = OvResNet12(indim=dest_channels, outdim=feat_channels)
        self.bn = nn.BatchNorm2d(feat_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)  
        
    def forward(self, x, sideout=False):
        out, att_out, ov_out =None, None, None
        sideout_dict = {}
        if (not self.use_region_sensing) and (not self.use_pixel_level_sensing):
            x = self.vision(x)                                    
        if self.use_region_sensing:
            feat = self.AtRegPy(x)                                  
            att_out = None
            if self.use_self_attention:
                B, _, H, W = feat.shape
                shape = {}
                shape['B'], shape['H'], shape['W'] = B, H, W
                pos_sine = self.pe(feat)                               
                att_out = self.transformer(feat.permute(0,2,3,1).contiguous(), shape, pos_sine)
                att_out = att_out.view(B,H,W,-1).permute(0,3,1,2).contiguous() 
                att_out = self.upsamp(att_out)
            else: att_out = feat
        if self.use_pixel_level_sensing: ov_out = self.OvRN12(x)  
        if (not self.use_region_sensing) and  (not self.use_pixel_level_sensing):
            out = x
        elif self.use_region_sensing and  (not self.use_pixel_level_sensing):
            out = att_out
        elif (not self.use_region_sensing) and  self.use_pixel_level_sensing:
            out = ov_out
        elif self.use_region_sensing and self.use_pixel_level_sensing:
            out = torch.cat([att_out, ov_out], dim=1)
        if self.use_region_sensing or self.use_pixel_level_sensing:
            out = self.bn(out)
            out = self.relu(out)
            out = self.dropout(out)
        if sideout: return out, sideout_dict
        else: return out


@register('vcnet12-encoder')
class EncoderVCNet12(nn.Module):
    def __init__(self, orig_channels=3, dest_channels=9, feat_channels=640, width=84, height=84, dropout_rate=0.2, 
                 use_region_sensing_list=[False, False, False], use_self_attention_list=[False, False, False], 
                 use_pixel_level_sensing_list=[False, False, False], self_attention_kwargs={}, **kwargs):
        super().__init__()
        self.orig_channels, self.dest_channels, self.feat_channels =  orig_channels, dest_channels, feat_channels
        self.width, self.height, self.dropout_rate = width, height, dropout_rate
        channels = [orig_channels, dest_channels, feat_channels]# 64, 64, 64, 64
        self.n = len(channels)
        def create_list(use_region_sensing=False, use_self_attention=False, use_pixel_level_sensing=False, self_attention_kwargs={}, **kwargs):
            return make('vcnet12-basic-block', orig_channels=self.orig_channels, dest_channels=self.dest_channels, 
                feat_channels=self.feat_channels, width=self.width, height=self.height, dropout_rate=self.dropout_rate, 
                use_region_sensing = use_region_sensing, use_self_attention = use_self_attention, 
                use_pixel_level_sensing = use_pixel_level_sensing, self_attention_kwargs = self_attention_kwargs, **kwargs)        

        self.vision =  create_list(use_region_sensing_list[0], use_self_attention_list[0], use_pixel_level_sensing_list[0], self_attention_kwargs={})
        self.branch1 = create_list(use_region_sensing_list[1], use_self_attention_list[1], use_pixel_level_sensing_list[1], self_attention_kwargs={})
        self.branch2 = create_list(use_region_sensing_list[2], use_self_attention_list[2], use_pixel_level_sensing_list[2], self_attention_kwargs={})
        
        self.out_dim = channels[2]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, sideout=False, branch=-1):
        def sideout_func(x, attr_name, sideout=False):
            blocks = getattr(self, attr_name)
            if sideout:
                sideout_dict = {}
                x, s = blocks(x, sideout=True)
                for layer_name, layer in s.items():
                    sideout_dict["{}.{}.{}".format(attr_name, i, layer_name)] = layer
                return x, sideout_dict
            else:
                x = blocks(x)
                return x

        if branch == 1: branch_attr_name = "branch1"
        elif branch == 2: branch_attr_name = "branch2"
        else: raise ValueError()

        if sideout:
            sideout_dict = {}
            x, s_vision = sideout_func(x, sideout=True, attr_name="vision")
            x, s_branch = sideout_func(x, sideout=True, attr_name=branch_attr_name)
            sideout_dict.update(s_vision)
            sideout_dict.update(s_branch)
            sideout_dict['before_avgpool'] = x
        else:
            x = sideout_func(x, attr_name="vision")
            x = sideout_func(x, attr_name=branch_attr_name)

        # Return if enable side output.
        if sideout: return x, sideout_dict
        else: return x

@register('encoder-vcnet12')
def vcnet12_encoder(**kwargs): return EncoderVCNet12(orig_channels=3, dest_channels=9, feat_channels=640, **kwargs)
