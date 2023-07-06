# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 16:10:26 2022

@author: liuyuxuan
"""

import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0., d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.2):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)
        self.group_norm1 = nn.GroupNorm(8, self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)
        self.group_norm2 = nn.GroupNorm(8, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        # y = self.batch_norm1(y)
        y = self.group_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)
        # y = self.batch_norm2(y)
        y = self.group_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self,
                 T,
                 ch,
                 in_size,
                 out_size,
                 ns=4,
                 ls=2048,
                 p_dropout=0.2):
        super(LinearModel, self).__init__()

        self.linear_size = ls
        self.p_dropout = p_dropout
        self.num_stage = ns
        self.input_size = in_size
        self.output_size = out_size
        self.T = T
        self.ch = ch

        # temporal embedding
        self.tdim = self.ch * 4
        self.time_embedding = TimeEmbedding(self.T, self.ch, self.tdim)
        # process input to linear size
        self.x_proj = nn.Sequential(
            Swish(),
            nn.Linear(self.input_size, int(self.linear_size/4)))
        self.f_proj = nn.Sequential(
            Swish(),
            nn.Linear(self.linear_size, int(self.linear_size/2)))
        self.t_proj = nn.Sequential(
            Swish(),
            nn.Linear(self.tdim, int(self.linear_size/4)))

        self.linear_stages = []
        for l in range(self.num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        self._initialize_weight()

    def forward(self, x_0, t, feats):
        # pre-processing
        temb = self.time_embedding(t)

        x_in = self.x_proj(x_0)
        feats_in = self.f_proj(feats)
        t_in = self.t_proj(temb)
        # y = x_in + feats_in + t_in
        y = torch.cat([x_in, feats_in, t_in], dim=1)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        out = self.w2(y)

        return out

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


if __name__ == '__main__':
    batch_size = 2
    model = LinearModel(T=100, ch=128, in_size=144, out_size=144, ns=2, ls=2048, p_dropout=0.1)

    x = torch.randn(batch_size, 144)
    feats = torch.rand(batch_size, 2048)
    t = torch.randint(100, [batch_size])
    y = model(x, t, feats)
    print(y.shape)

