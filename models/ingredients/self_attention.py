import numpy as np
import torch,copy,math
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=32, temperature=10000, normalize=False, scale=None, amp=1.0):
        super().__init__()
        self.num_pos_feats = num_pos_feats  
        self.temperature = temperature     
        self.normalize = normalize         
        self.amp = amp                     
        if scale is not None and normalize is False: raise ValueError("如果传入了 scale 参数，则 normalize 必须为 True")
        if scale is None: scale = 2 * math.pi
        self.scale = scale                 

    def forward(self, x):
        B, _, H, W = x.shape
        mask = torch.ones(B, H, W, dtype=torch.float32, device=x.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize: 
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = self.amp * torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = self.amp * torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos

class MHAtt(nn.Module):
    def __init__(self, embedding_size = 256, multi_head=1, dropout_ratio = 0.1):
        super(MHAtt, self).__init__()
        self.multi_head = multi_head
        self.multi_head_size = int(embedding_size/multi_head)
        self.embedding_size = embedding_size
        self.linear_v = nn.Linear(embedding_size, embedding_size)
        self.linear_k = nn.Linear(embedding_size, embedding_size)
        self.linear_q = nn.Linear(embedding_size, embedding_size)
        self.linear_merge = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout_ratio)

    def att(self, value, key, query, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None: scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)
    
    def forward(self, q, k, v):
        B = q.shape[0]
        v = self.linear_v(v).view(
            B,
            -1,
            self.multi_head,
            self.multi_head_size
            ).transpose(1, 2)
        k = self.linear_k(k).view(
            B,
            -1,
            self.multi_head,
            self.multi_head_size
            ).transpose(1, 2)
        q = self.linear_q(q).view(
            B,
            -1,
            self.multi_head,
            self.multi_head_size
            ).transpose(1, 2)
        atted = self.att(v, k, q)
        atted = atted.transpose(1, 2).contiguous().view(
            B,
            -1,
            self.embedding_size
            )
        atted = self.linear_merge(atted)
        return atted

class SA(nn.Module):
    def __init__(self, dropout_ratio = 0.1, multi_head = 1, embedding_size = 256, pre_normalize = False):
        super(SA, self).__init__()
        self.mhatt = MHAtt(embedding_size = embedding_size, multi_head = multi_head)
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout_ratio)
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.norm1 = nn.LayerNorm(embedding_size) 
        self.norm2 = nn.LayerNorm(embedding_size) 
        self.linear1 = nn.Linear(embedding_size, embedding_size)
        self.linear2 = nn.Linear(embedding_size, embedding_size)
        self.pre_normalize = pre_normalize
        
    def forward(self, x, shape = {}, fourier_pos=None):
        if self.pre_normalize:
            return self.forward_pre(x, shape, fourier_pos)
        return self.forward_post(x, shape, fourier_pos)
        
    def forward_pre(self, x, shape = {}, fourier_pos = None) :
        v = x.view(shape['B'], -1, self.embedding_size)
        v2 = self.norm1(v)   
        q = self.with_pos_embed(v2.view(shape['B'], shape['H'], shape['W'],-1), pos)
        q = k = self.with_pos_embed(q, fourier_pos)
        q, k = q.view(shape['B'], -1, self.embedding_size),k.view(shape['B'], -1, self.embedding_size) 
        
        v = v + self.dropout1(self.mhatt(q, k , v2))
        v2 = self.norm2(v)
        v2 = self.linear2(self.dropout(F.relu(self.linear1(v2))))
        v = v + self.dropout2(v2)
        return v
    
    def with_pos_embed(self, x, pos = None):
        return x if pos is None else x + pos
        
    def forward_post(self, x, shape = {}, fourier_pos=None):
        if fourier_pos is not None: q = k = self.with_pos_embed(x, fourier_pos)
        else: q = k = x
        q, k = q.view(shape['B'], -1, self.embedding_size), k.view(shape['B'], -1, self.embedding_size)
        v = x.view(shape['B'], -1, self.embedding_size)
        atted = self.mhatt(q, k, v)             
        x = self.norm1(v + self.dropout1(atted))
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(x2))
        x = x.view(shape['B'],shape['H'], shape['W'],-1)
        return x

def _get_clones(module,num_layers):
        return nn.ModuleList([copy.deepcopy(module) for i in range(num_layers)])


class Transformer_Layer(nn.Module):
    def __init__(self, dropout_ratio = 0.1, multi_head = 8, embedding_size = 256, pre_normalize=\
                 False, num_self_attention_layers = 1,**kwargs):
        super(Transformer_Layer, self).__init__()
        self.layers = _get_clones(SA(dropout_ratio = dropout_ratio,multi_head = multi_head, embedding_size = embedding_size, pre_normalize = pre_normalize), num_layers = num_self_attention_layers)
    
    def forward(self, x, shape = {}, fourier_pos=None):
        output = x
        for layer in self.layers: output = layer(output, shape = shape, fourier_pos=fourier_pos)
        return output
