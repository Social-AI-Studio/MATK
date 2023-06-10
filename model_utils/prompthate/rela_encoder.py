import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
import math

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    #print (scores[0,0])
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        #print (mask.shape)
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        #print('Attention Weights:',self.attn[0,-1,:,:])
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x), inplace=False)))
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        #self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.norm = LayerNorm(size)
        self.dropout=nn.Dropout(dropout)

    def forward(self, img, cap, mask=None):
        "Follow Figure 1 (left) for connections."
        img = self.self_attn(cap, img, img, mask)
        return img+ self.dropout(
            self.feed_forward(self.norm(img))
        )

class Rela_Module(nn.Module):
    def __init__(self,v_dim,hid_dim,h,mid_dim,num_layers,dropout):
        super(Rela_Module, self).__init__()
        self.proj_v=nn.Linear(v_dim,hid_dim)
        
        self_attn=copy.deepcopy(
            MultiHeadedAttention(h,hid_dim,dropout)
        )
        feed_forward=copy.deepcopy(
            PositionwiseFeedForward(hid_dim,mid_dim,dropout)
        )
        
        layer=EncoderLayer(hid_dim,self_attn,feed_forward,dropout)
        self.layers=clones(layer,num_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, img, cap, obj_mask=None):
        img=self.proj_v(img)
        for l in self.layers:
            img=l(img,cap,obj_mask)
        #print (img.shape)
        return torch.sum(img,dim=1)