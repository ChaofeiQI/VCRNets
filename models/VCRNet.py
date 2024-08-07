# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Author:   CHAOFEI QI
#  Email:    cfqi@stu.hit.edu.cn
#  Address： Harbin Institute of Technology
#  
#  Copyright (c) 2024
#  This source code is licensed under the MIT-style license found in the
#  LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math
import torch
import torch.nn as nn
import models
import utils
from .base import register
import torch.nn.functional as F

@register('linear-classifier')
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        return self.linear(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=8):
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


@register('VCRNet')
class VCRNet(nn.Module):
    def __init__(self, encoder, encoder_args, classifier, classifier_args, sideout_info=[], method='sqr', temp=1.0, temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        # Standard classifier.
        classifier_args['in_dim'] = self.encoder.out_dim

        self.downsamp= nn.Sequential(  
                nn.Conv2d(self.encoder.out_dim *2, self.encoder.out_dim, kernel_size=3, stride=2, padding=1, bias=False),  
                nn.BatchNorm2d(self.encoder.out_dim),
                nn.ReLU(inplace=True)
            )
        self.channel_attention = ChannelAttention(self.encoder.out_dim)

        self.classifier = models.make(classifier, **classifier_args)
        self.sideout_info = sideout_info
        self.sideout_classifiers = nn.ModuleList()
        for _, sideout_dim in self.sideout_info:
            classifier_args['in_dim'] = sideout_dim
            self.sideout_classifiers.append(models.make(classifier, **classifier_args))
        # Few-shot classifier.
        self.method = method
        if temp_learnable: self.temp = nn.Parameter(torch.tensor(temp))
        else: self.temp = temp

    def svd(self, data, drop_rate=0.0, num_singular='default'):
        new_shape = (data.shape[0]*data.shape[1],)+(data.shape[-2],-1)
        reshaped_data = data.view(new_shape)
        drop_out = nn.Dropout(drop_rate)
        U, s, V = torch.svd(reshaped_data)
        if num_singular == 'default':
            svg = s.reshape(data.shape[0],data.shape[1],-1)
        else:
            svg = s.reshape(data.shape[0],data.shape[1],-1)
            num_singular_values = num_singular
            svg[:,:,svg.shape[2]-1]=0
        svg = drop_out(svg)
        return svg


    def forward(self, mode, x=None, x_shot=None, x_query=None, branch=-1, sideout=False):
        # 1.Standard classifier (return logits).
        def class_forward(x, **kwargs):                        
            x_rsb, _ = self.encoder(x, sideout=True, branch=1) 
            feat = self.channel_attention(x_rsb)
            feat = self.svd(feat)
            feat = feat.mean(-1)
            logits = self.classifier(feat)
            return logits
        
        # 2.Few-shot classifier (return logits).
        def meta_forward(x_shot, x_query, **kwargs):
            shot_shape, query_shape = x_shot.shape[:-3], x_query.shape[:-3] 
            img_shape = x_shot.shape[-3:]           
            x_shot  = x_shot.view(-1, *img_shape)   
            x_query = x_query.view(-1, *img_shape)
            x_psb = self.encoder(torch.cat([x_shot, x_query], dim=0), sideout=False, branch=2)
            x_tot = self.channel_attention(x_psb)
            x_tot = self.svd(x_tot)
            x_tot = x_tot.mean(-1)
            x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]  
            x_shot  = x_shot.view(*shot_shape, -1)                        
            x_query = x_query.view(*query_shape, -1)                      

            if self.method == 'cos':
                x_shot = x_shot.mean(dim=-2)
                x_shot = F.normalize(x_shot, dim=-1)
                x_query = F.normalize(x_query, dim=-1)
                metric = 'dot'
                logits = utils.compute_logits(x_query, x_shot, metric=metric, temp=self.temp) 
            elif self.method == 'sqr':
                x_shot = x_shot.mean(dim=-2) 
                metric = 'sqr'
                logits = utils.compute_logits(x_query, x_shot, metric=metric, temp=self.temp / 1600.)
            return logits

        # Few-shot classifier (for meta test).
        def meta_test_forward(x_shot, x_query, **kwargs):
            shot_shape, query_shape = x_shot.shape[:-3], x_query.shape[:-3] 
            img_shape   = x_shot.shape[-3:]                    
            x_shot      = x_shot.view(-1, *img_shape)          
            x_query     = x_query.view(-1, *img_shape)         
            x_shot_len, x_query_len = len(x_shot), len(x_query)
            
            x_rsb = self.encoder(torch.cat([x_shot, x_query], dim=0), sideout=False, branch=1)
            x_psb = self.encoder(torch.cat([x_shot, x_query], dim=0), sideout=False, branch=2)
            x_rsb, x_psb = self.channel_attention(x_rsb), self.channel_attention(x_psb)
            x_tot_rsb, x_tot_psb = self.svd(x_rsb), self.svd(x_psb)
            x_tot_rsb, x_tot_psb = x_tot_rsb.mean(-1), x_tot_psb.mean(-1)
            x_shot_rsb, x_query_rsb = x_tot_rsb[:x_shot_len], x_tot_rsb[-x_query_len:]
            x_shot_psb, x_query_psb = x_tot_psb[:x_shot_len], x_tot_psb[-x_query_len:]
            feat_rsb_shape, feat_psb_shape = x_shot_rsb.shape[1:], x_shot_psb.shape[1:]
            x_shot_rsb, x_query_rsb = x_shot_rsb.view(*shot_shape, *feat_rsb_shape), x_query_rsb.view(*query_shape, *feat_rsb_shape)
            x_shot_psb, x_query_psb = x_shot_psb.view(*shot_shape, *feat_psb_shape), x_query_psb.view(*query_shape, *feat_psb_shape)

            return x_query_rsb, x_shot_rsb, x_query_psb, x_shot_psb


        ###################################
        # 主函数入口：Training or evaluation
        ###################################
        if self.training:
            # 1.For standard classification: Train
            if mode=='class':
                logits = class_forward(x)
                return logits
            # 2.For few-shot classification: Train
            elif mode=='meta':
                logits = meta_forward(x_shot, x_query)
                return logits
            else: raise ValueError()
            
        else:
            if mode=='class':       # 1.For standard classification: Validation and testing.
                logits = class_forward(x)
                return logits
            elif mode=='meta':      # 2.For few-shot classification: Validation.
                logits = meta_forward(x_shot, x_query)
                return logits
            elif mode=='meta_test': # 3.For few-shot classification: Testing.
                return meta_test_forward(x_shot, x_query)
            else: raise ValueError()
