# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:26:24 2022

@author: liuyuxuan
"""

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import time
import os
from typing import Optional, Dict, Tuple
import math
import joblib
import torch.nn as nn
from yacs.config import CfgNode
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
import scipy.io
from PIL import Image
import cv2
import torchvision
import logging
from prohmr.configs import get_config, prohmr_config, dataset_config
from prohmr.utils.geometry import rot6d_to_rotmat
from prohmr.utils.geometry import aa_to_rotmat, perspective_projection
from prohmr.models import SMPL, SMPLHead
from prohmr.models.backbones import create_backbone
from prohmr.models.discriminator import Discriminator
from prohmr.models.losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from train.evaluate_joints import  eva_joints, p_mpjpe
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def parse_config():
    parser = argparse.ArgumentParser(description='ProHMR training code')
    parser.add_argument('--BENCHMARK', default=True, type=bool)
    parser.add_argument('--DETERMINISTIC', default=False, type=bool)
    parser.add_argument('--ENABLED', default=True, type=bool)
    parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file')
    parser.add_argument('--root_info', type=str,
                        default='/home/work/Yuxuan/Data/ECHA/Info',
                        help='Root to the info stored in matlab files')
    parser.add_argument('--root_info_test', type=str,
                        default='/home/work/Yuxuan/Data/ECHA/Info_vicon',
                        help='Root to the info stored in matlab files')
    parser.add_argument('--root_smpl', type=str,
                        default='/home/work/Yuxuan/Data/ECHA/shape_info',
                        help='Root to the info stored in smpl parameters')
    parser.add_argument('--root_p2d', type=str,
                        default='/home/work/Yuxuan/Code/ESRDM/ego2d_detect',
                        help='Root to the info stored in 2D keypoints positions')
    parser.add_argument('--root_EgoPW_info', type=str,
                        default='/home/work/Yuxuan/Data/EgoPW/shape_info',
                        help='Root to the info stored in EgoPW dataset')
    parser.add_argument('--root_EgoPW_image',type=str,
                        default='/home/work/Yuxuan/Data/EgoPW/data_info',
                        help='Root to the images stored in EgoPW datset')
    parser.add_argument('--root_EgoWang', type=str,
                        default='/home/work/Yuxuan/Data/Ego_Wang/Test_global',
                        help='Root to the images stored in EgoWang datset')
    parser.add_argument('--gpu', default=1, help='type of gpu', type=int)

    parser.add_argument('--train_batch_size', default=64, help='batch-size for training', type=int)
    parser.add_argument('--val_batch_size', default=48, help='batch-size for validation', type=int)
    parser.add_argument('--test_batch_size', default=80, help='batch-size for testiing', type=int)
    parser.add_argument('--num_workers', default=4, help='number of workers', type=int)
    parser.add_argument('--max_epoch', default=20, help='max training epoch', type=int)
    parser.add_argument('--freq_print_train', default=10, help='Printing frequency for training', type=int)
    parser.add_argument('--freq_print_val', default=50, help='Printing frequency for validation', type=int)
    parser.add_argument('--freq_print_test', default=50, help='Printing frequency for test', type=int)

    parser.add_argument('--load_model', default='', help='path to the loading model', type=str)
    parser.add_argument('--T', default=10, help='T steps for diffusion model', type=int)
    parser.add_argument('--beta_l', default=1e-4, help='starting beta value', type=float)
    parser.add_argument('--beta_T', default=0.02, help='ending beta value', type=float)
    parser.add_argument('--ch', default=128, help='channels for temporal embeddings', type=int)
    parser.add_argument('--num_stage', default=3, help='number of stages for linear model', type=int)
    parser.add_argument('--latent_size', default=2048, help='latent size of the linear model', type=int)
    parser.add_argument('--p_dropout', default=0.2, help='ratio of dropout layer', type=float)
    parser.add_argument('--max_iters', default=10, help='number of iterations', type=int)


    args = parser.parse_args()
    return args


#==================================Diffusion model definition====================================================================
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


class FCHead(nn.Module):

    def __init__(self, cfg: CfgNode):
        """
        Fully connected head for camera and betas regression.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(FCHead, self).__init__()
        self.cfg = cfg
        self.npose = 6 * (cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.layers = nn.Sequential(nn.Linear(cfg.MODEL.FLOW.CONTEXT_FEATURES,
                                              cfg.MODEL.FC_HEAD.NUM_FEATURES),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(cfg.MODEL.FC_HEAD.NUM_FEATURES, 10))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32))[None, None]
        init_betas = torch.from_numpy(mean_params['shape'].astype(np.float32))[None, None]

        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_betas', init_betas)

    def forward(self, feats: torch.Tensor, feats_zoom: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run forward pass.
        Args:
            smpl_params (Dict): Dictionary containing predicted SMPL parameters.
            feats (torch.Tensor): Tensor of shape (N, C) containing the features computed by the backbone.
        Returns:
            pred_betas (torch.Tensor): Predicted SMPL betas.
        """

        batch_size = feats.shape[0]

        offset = self.layers(feats).reshape(batch_size, 10)
        betas_offset = offset[:, :10]
        pred_betas = betas_offset + self.init_betas

        return pred_betas

class FCWeight(nn.Module):

    def __init__(self, cfg: CfgNode):
        """
        Fully connected head for camera and betas regression.
        Args:
            cfg (CfgNode): Model config as yacs CfgNode.
        """
        super(FCWeight, self).__init__()
        self.cfg = cfg
        self.npose_lower = 9
        self.layers = nn.Sequential(nn.Linear(cfg.MODEL.FLOW.CONTEXT_FEATURES,
                                              cfg.MODEL.FC_HEAD.NUM_FEATURES),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(cfg.MODEL.FC_HEAD.NUM_FEATURES, self.npose_lower))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)
        self.init_weight = 0.5



    def forward(self, feats: torch.Tensor, feats_zoom: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run forward pass.
        Args:
            smpl_params (Dict): Dictionary containing predicted SMPL parameters.
            feats (torch.Tensor): Tensor of shape (N, C) containing the features computed by the backbone.
        Returns:
            pred_betas (torch.Tensor): Predicted SMPL betas.
        """

        batch_size = feats_zoom.shape[0]

        offset = self.layers(feats_zoom).reshape(batch_size, self.npose_lower)
        # pred_weights = 0.2 * ( torch.sigmoid(offset) - 0.5 ) + 0.5 ## ~ [0.4,0.6]
        # pred_weights = 0.4 * (torch.sigmoid(offset) - 0.5) + 0.5  ## ~ [0.3,0.7]
        pred_weights = torch.sigmoid(offset) ## ~[0,1.0]
        # pred_weights = 0.5
        return pred_weights

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class SMPLDiffusion(nn.Module):

    def __init__(self, args, cfg):
        super(SMPLDiffusion, self).__init__()
        self.args = args
        self.cfg = cfg
        self.npose = 6 * (cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.npose_lower = 6 * 9
        self.model = LinearModel(T=args.T, ch=args.ch, in_size=self.npose, out_size=self.npose, ns=args.num_stage,
                                 ls=args.latent_size, p_dropout=args.p_dropout)
        self.model_zoom = LinearModel(T=args.T, ch=args.ch, in_size=self.npose_lower, out_size=self.npose_lower, ns=args.num_stage,
                                 ls=args.latent_size, p_dropout=args.p_dropout)
        self.fc_head = FCHead(cfg)
        self.fc_weight = FCWeight(cfg)
        self.T = args.T
        self.beta_l = args.beta_l
        self.beta_T = args.beta_T
        self.betas = torch.linspace(self.beta_l, self.beta_T, self.T).double()
        self.w = 0.0
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:self.T]
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def mse_loss(self, x_0, feats, feats_zoom):
        """
        Algorithm 1.
        """
        x_0_zoom = torch.cat((x_0['global_orient'], x_0['lower_pose']), dim=-1)
        x_0 = torch.cat((x_0['global_orient'], x_0['body_pose']), dim=-1)


        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        noise_zoom = torch.randn_like(x_0_zoom)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        x_t_zoom = extract(self.sqrt_alphas_bar, t, x_0_zoom.shape) * x_0_zoom + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0_zoom.shape) * noise_zoom
        loss = F.mse_loss(self.model(x_t, t, feats), noise, reduction='none')
        loss_zoom = F.mse_loss(self.model_zoom(x_t_zoom, t, feats_zoom),noise_zoom,reduction='none')
        return loss, loss_zoom , noise, noise_zoom

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, feats):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2].to(feats.device), self.betas[1:].to(feats.device)])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, feats)
        nonEps = self.model(x_t, t, torch.zeros_like(feats).to(feats.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def p_mean_variance_zoom(self, x_t, t, feats):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2].to(feats.device), self.betas[1:].to(feats.device)])
        var = extract(var, t, x_t.shape)
        eps = self.model_zoom(x_t, t, feats)
        nonEps = self.model_zoom(x_t, t, torch.zeros_like(feats).to(feats.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def average(self,a,b,lam=0.5):
        ## a: whole body pose ; b: lower body pose
        batch_size = lam.shape[0]
        lam = lam.view(batch_size,-1)
        return a * (1-lam) + b * lam

    def forward(self, x_T, x_T_zoom, feats, feats_zoom, flag):
        """
        Algorithm 2.
        """
        batch_size = feats.shape[0]
        # for original image
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t, feats=feats)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        # for zoomed image
        x_t_zoom = x_T_zoom
        for time_step in reversed(range(self.T)):
            t = x_t_zoom.new_ones([x_T_zoom.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance_zoom(x_t=x_t_zoom, t=t, feats=feats_zoom)
            if time_step > 0:
                noise_zoom = torch.randn_like(x_t_zoom)
            else:
                noise_zoom = 0
            x_t_zoom = mean + torch.sqrt(var) * noise_zoom
            assert torch.isnan(x_t_zoom).int().sum() == 0, "nan in tensor."


        x_0 = x_t
        pred_pose = x_0[:, :self.npose]
        pred_pose_6d = pred_pose.clone()

        x_0_zoom = x_t_zoom
        pred_pose_zoom = x_0_zoom[:,:self.npose_lower]
        pred_pose_6d_zoom = pred_pose_zoom.clone()

        pred_betas = self.fc_head(feats, feats_zoom)
        pred_weights = flag * self.fc_weight(feats, feats_zoom)

        pred_pose_output = torch.zeros(batch_size,self.npose).to(feats.device)


        pred_pose_output[:, 0:6] = self.average(pred_pose[:,0:6],pred_pose_zoom[:,0:6],pred_weights[:,0])
        pred_pose_output[:, 6:12] = self.average(pred_pose[:,6:12],pred_pose_zoom[:,6:12],pred_weights[:,1])
        pred_pose_output[:, 12:18] = self.average(pred_pose[:, 12:18], pred_pose_zoom[:, 30:36],pred_weights[:,2])
        pred_pose_output[:, 24:30] = self.average(pred_pose[:, 24:30], pred_pose_zoom[:, 12:18],pred_weights[:,3])
        pred_pose_output[:, 30:36] = self.average(pred_pose[:, 30:36], pred_pose_zoom[:, 36:42],pred_weights[:,4])
        pred_pose_output[:, 42:48] = self.average(pred_pose[:, 42:48], pred_pose_zoom[:, 18:24],pred_weights[:,5])
        pred_pose_output[:, 48:54] = self.average(pred_pose[:, 48:54], pred_pose_zoom[:, 42:48],pred_weights[:,6])
        pred_pose_output[:, 60:66] = self.average(pred_pose[:, 60:66], pred_pose_zoom[:, 24:30],pred_weights[:,7])
        pred_pose_output[:, 66:72] = self.average(pred_pose[:, 66:72], pred_pose_zoom[:, 48:54],pred_weights[:,8])

        pred_pose_output[:,18:24] = pred_pose[:,18:24]
        pred_pose_output[:,36:42] = pred_pose[:,36:42]
        pred_pose_output[:,54:60] = pred_pose[:,54:60]
        pred_pose_output[:,72:144] = pred_pose[:,72:144]

        pred_pose_output = rot6d_to_rotmat(pred_pose_output.reshape(batch_size, -1)).view(batch_size,
                                                                            self.cfg.SMPL.NUM_BODY_JOINTS + 1, 3, 3)

        pred_smpl_params = {'global_orient': pred_pose_output[:, [0]],
                            'body_pose': pred_pose_output[:, 1:]}

        pred_smpl_params['betas'] = pred_betas.view(-1,10)

        return pred_smpl_params, pred_weights, x_T, pred_pose_6d, pred_pose_6d_zoom

    def show_traj(self, x_T, x_T_zoom, feats, feats_zoom, flag):
        pred_smpl_params_list = []
        for tdx in reversed(range(self.T)):
            pred_smpl = self.step_forward(x_T,x_T_zoom,feats,feats_zoom,flag,tdx)
            pred_smpl_params_list.append(pred_smpl)
        return pred_smpl_params_list

    def step_forward(self, x_T, x_T_zoom, feats, feats_zoom, flag, step):
        """
        Algorithm 2.
        """
        batch_size = feats.shape[0]
        # for original image
        x_t = x_T
        for time_step in reversed(range(step,self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t, feats=feats)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

        # for zoomed image
        x_t_zoom = x_T_zoom
        for time_step in reversed(range(step,self.T)):
            t = x_t_zoom.new_ones([x_T_zoom.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance_zoom(x_t=x_t_zoom, t=t, feats=feats_zoom)
            if time_step > 0:
                noise_zoom = torch.randn_like(x_t_zoom)
            else:
                noise_zoom = 0
            x_t_zoom = mean + torch.sqrt(var) * noise_zoom
            assert torch.isnan(x_t_zoom).int().sum() == 0, "nan in tensor."


        x_0 = x_t
        pred_pose = x_0[:, :self.npose]
        pred_pose_6d = pred_pose.clone()

        x_0_zoom = x_t_zoom
        pred_pose_zoom = x_0_zoom[:,:self.npose_lower]
        pred_pose_6d_zoom = pred_pose_zoom.clone()

        pred_betas = self.fc_head(feats, feats_zoom)
        pred_weights = flag * self.fc_weight(feats, feats_zoom)

        pred_pose_output = torch.zeros(batch_size,self.npose).to(feats.device)


        pred_pose_output[:, 0:6] = self.average(pred_pose[:,0:6],pred_pose_zoom[:,0:6],pred_weights[:,0])
        pred_pose_output[:, 6:12] = self.average(pred_pose[:,6:12],pred_pose_zoom[:,6:12],pred_weights[:,1])
        pred_pose_output[:, 12:18] = self.average(pred_pose[:, 12:18], pred_pose_zoom[:, 30:36],pred_weights[:,2])
        pred_pose_output[:, 24:30] = self.average(pred_pose[:, 24:30], pred_pose_zoom[:, 12:18],pred_weights[:,3])
        pred_pose_output[:, 30:36] = self.average(pred_pose[:, 30:36], pred_pose_zoom[:, 36:42],pred_weights[:,4])
        pred_pose_output[:, 42:48] = self.average(pred_pose[:, 42:48], pred_pose_zoom[:, 18:24],pred_weights[:,5])
        pred_pose_output[:, 48:54] = self.average(pred_pose[:, 48:54], pred_pose_zoom[:, 42:48],pred_weights[:,6])
        pred_pose_output[:, 60:66] = self.average(pred_pose[:, 60:66], pred_pose_zoom[:, 24:30],pred_weights[:,7])
        pred_pose_output[:, 66:72] = self.average(pred_pose[:, 66:72], pred_pose_zoom[:, 48:54],pred_weights[:,8])

        pred_pose_output[:,18:24] = pred_pose[:,18:24]
        pred_pose_output[:,36:42] = pred_pose[:,36:42]
        pred_pose_output[:,54:60] = pred_pose[:,54:60]
        pred_pose_output[:,72:144] = pred_pose[:,72:144]

        pred_pose_output = rot6d_to_rotmat(pred_pose_output.reshape(batch_size, -1)).view(batch_size,
                                                                            self.cfg.SMPL.NUM_BODY_JOINTS + 1, 3, 3)

        pred_smpl_params = {'global_orient': pred_pose_output[:, [0]],
                            'body_pose': pred_pose_output[:, 1:]}

        pred_smpl_params['betas'] = pred_betas.view(-1,10)

        return pred_smpl_params


#=======================================================================================================================

class AverageMeter():
    def __init__(self):
        self.val = 0
        self.count = 0
        self.sum = 0
        self.ave = 0

    def update(self, val, num=1):
        self.count = self.count + num
        self.val = val
        self.sum = self.sum + num * val
        self.ave = self.sum / self.count if self.count != 0 else 0.0


class CustomFormatter(logging.Formatter):
    DATE = '\033[94m'
    GREEN = '\033[92m'
    WHITE = '\033[0m'
    WARNING = '\033[93m'
    RED = '\033[91m'

    def __init__(self):
        orig_fmt = "%(name)s: %(message)s"
        datefmt = "%H:%M:%S"
        super().__init__(orig_fmt, datefmt)

    def format(self, record):
        color = self.WHITE
        if record.levelno == logging.INFO:
            color = self.GREEN
        if record.levelno == logging.WARN:
            color = self.WARNING
        if record.levelno == logging.ERROR:
            color = self.RED
        self._style._fmt = "{}%(asctime)s {}[%(levelname)s]{} {}: %(message)s".format(
            self.DATE, color, self.DATE, self.WHITE)
        return logging.Formatter.format(self, record)


class ConsoleLogger():
    def __init__(self, training_type, phase='train'):
        super().__init__()
        self._logger = logging.getLogger(training_type)
        self._logger.setLevel(logging.INFO)
        formatter = CustomFormatter()
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        console_log.setFormatter(formatter)
        self._logger.addHandler(console_log)
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.logfile_dir = os.path.join('experiments/', training_type, time_str)
        os.makedirs(self.logfile_dir)
        logfile = os.path.join(self.logfile_dir, f'{phase}.log')
        file_log = logging.FileHandler(logfile, mode='a')
        file_log.setLevel(logging.INFO)
        file_log.setFormatter(formatter)
        self._logger.addHandler(file_log)

    def info(self, *args, **kwargs):
        """info"""
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """warning"""
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """error"""
        self._logger.error(*args, **kwargs)
        exit(-1)

    def getLogFolder(self):
        return self.logfile_dir


class EgoPWdataset(Dataset):
    def __init__(self, root1, root2, stage):
        self.root1 = root1 # smpl mat root
        self.root2 = root2 # ego image root
        self.stage = stage
        self.filelist_smpl = []

        self.name_list = sorted(os.listdir(self.root1))
        for ndx in range(len(self.name_list)):
            name_root = os.path.join(self.root1, self.name_list[ndx])
            env_list = sorted(os.listdir(name_root))
            for edx in range(len(env_list)):
                env_root = os.path.join(name_root, env_list[edx])
                env_root_listdir = sorted(os.listdir(env_root))
                for idx in range(len(env_root_listdir)):
                    mat_file = os.path.join(env_root,env_root_listdir[idx])
                    self.filelist_smpl.append(mat_file)
        self.length = len(self.filelist_smpl)
        self.has_body_pose = np.ones(self.length,dtype = np.float32)
        self.has_betas = np.ones(self.length,dtype = np.float32)

    def resize_image(self, image, resize_height=None, resize_width=None):
        '''
        :param image:
        :param resize_height:
        :param resize_width:
        :return:
        '''
        image_shape = np.shape(image)
        height = image_shape[0]
        width = image_shape[1]
        if (resize_height is None) and (resize_width is None):
            return image
        if resize_height is None:
            resize_height = int(height * resize_width / width)
        elif resize_width is None:
            resize_width = int(width * resize_height / height)
        image = cv2.resize(image, dsize=(resize_width, resize_height))
        return image

    def __getitem__(self, index):

        smpl_data_root = self.filelist_smpl[index]
        smpl_data = scipy.io.loadmat(smpl_data_root)
        betas = smpl_data['betas'].reshape(10)
        pose = smpl_data['pose'].reshape(24, 3, 3)
        has_body_pose = self.has_body_pose[index]
        has_betas = self.has_betas[index]
        env = str(smpl_data['env'][0])
        name = str(smpl_data['name'][0])
        frame_ids = int(smpl_data['frame_ids'][0])

        env_img_root = os.path.join(self.root2, name, env, 'imgs')
        env_img_listdir = sorted(os.listdir(env_img_root))
        ego_root = os.path.join(env_img_root, env_img_listdir[frame_ids])


        # image_size = np.asarray([224, 224])
        image_size = np.asarray([384, 384])
        # image_size = np.asarray([256, 256])
        image_lower_size = np.asarray([56, 56])

        # -----------------------add augmentation-------------------------------------
        ego_mat_ori = Image.open(ego_root)
        ego_mat_ori = np.array(ego_mat_ori)
        ego_mat_ori = ego_mat_ori[:,128:-128,:]


        ego_mat_ori = self.resize_image(ego_mat_ori, resize_width=image_size[0], resize_height=image_size[1])
        ego_mat_ori = Image.fromarray(ego_mat_ori)
        if self.stage == 'Train':
            color_aug = torchvision.transforms.ColorJitter(brightness=np.random.rand(), contrast=5 * np.random.rand(),
                                                           hue=np.random.rand() / 2, saturation=np.random.rand())
            ego_mat_ori = color_aug(ego_mat_ori)

        ego_mat_ori = np.array(ego_mat_ori)

        lower_mat_ori = ego_mat_ori[96:-96,96:-96,:]
        lower_mat_ori_norm = self.resize_image(lower_mat_ori, resize_width=image_lower_size[0], resize_height=image_lower_size[1])


        ego_mat_ori_norm = (ego_mat_ori - np.min(ego_mat_ori)) / (np.max(ego_mat_ori) - np.min(ego_mat_ori) + 1e-9)
        lower_mat_ori_norm = (lower_mat_ori_norm - np.min(lower_mat_ori_norm)) / (np.max(lower_mat_ori_norm) - np.min(lower_mat_ori_norm) + 1e-9)

        # lower_mat_ori_norm = np.ones((image_lower_size[0],image_lower_size[0],3))*0.5

        image = np.transpose(ego_mat_ori_norm, [2, 0, 1])
        lower_image = np.transpose(lower_mat_ori_norm , [2,0,1])

        # =============================smpl parameters==============================================
        smpl_params = {'global_orient': pose[0, :, :],
                       'body_pose': pose[1:, :, :],
                       'betas': betas
                       }

        has_smpl_params = {'global_orient': has_body_pose,
                           'body_pose': has_body_pose,
                           'betas': has_betas
                           }

        smpl_params_is_axis_angle = {'global_orient': False,
                                     'body_pose': False,
                                     'betas': False
                                     }
        item = {}
        item['img'] = image
        item['img_lower'] = lower_image
        item['img_size'] = 1.0 * 224
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['imgname'] = frame_ids
        item['imgenv'] = env
        item['imgname'] = name
        item['imgroot'] = env_img_listdir[frame_ids]
        item['idx'] = index
        # if np.random.uniform(0,1) > 0.5:
        #     item['flag'] = np.zeros(9,dtype=np.float32)
        # else :
        #     item['flag'] = np.ones(9,dtype=np.float32)
        item['flag'] = np.ones(9, dtype=np.float32)
        # item['flag'] = np.zeros(9, dtype=np.float32)
        # 0-0.5 means use img_lower   0.5-1.0 means without img_lower

        # item = {}
        return item

    def __len__(self):
        return len(self.filelist_smpl)

class testdataset_mo2cap2(Dataset):
    def __init__(self, root1, stage):
        self.root1 = root1 ## mo2cap2 image dir
        self.stage = stage
        self.filelist_mat = []
        if self.stage == 'Test':
            self.env_list = ['weipeng_studio','olek_outdoor']
        for idx in range(len(self.env_list)):
            env_dir_root1 = os.path.join(self.root1, self.env_list[idx])
            env_dir_listdir1 = sorted(os.listdir(env_dir_root1))
            for jdx in range(len(env_dir_listdir1)):
                mat_root = os.path.join(env_dir_root1, env_dir_listdir1[jdx])
                self.filelist_mat.append(mat_root)
        self.length = len(self.filelist_mat)
        self.has_body_pose = np.ones(self.length,dtype = np.float32)
        self.has_betas = np.ones(self.length,dtype = np.float32)

    def resize_image(self, image, resize_height=None, resize_width=None):
        '''
        :param image:
        :param resize_height:
        :param resize_width:
        :return:
        '''
        image_shape = np.shape(image)
        height = image_shape[0]
        width = image_shape[1]
        if (resize_height is None) and (resize_width is None):
            return image
        if resize_height is None:
            resize_height = int(height * resize_width / width)
        elif resize_width is None:
            resize_width = int(width * resize_height / height)
        image = cv2.resize(image, dsize=(resize_width, resize_height))
        return image

    def __getitem__(self, index):


        betas = np.zeros(10)*1.0
        pose = np.ones((24,3,3))*1.0
        keypoints = np.ones((15,3))*1.0
        has_body_pose = self.has_body_pose[index]
        has_betas = self.has_betas[index]


        # image_size = np.asarray([224, 224])
        # image_size = np.asarray([384, 384])
        image_size = np.asarray([256, 256])
        image_lower_size = np.asarray([56, 56])

        # ------change the root ---------------------------------------------
        ego_root = self.filelist_mat[index]
        # -----------------------add augmentation-------------------------------------
        ego_mat_ori = Image.open(ego_root)
        ego_mat_ori = np.array(ego_mat_ori)
        ego_mat_ori = ego_mat_ori[:, 128:-128, :]

        ego_mat_ori = self.resize_image(ego_mat_ori, resize_width=image_size[0], resize_height=image_size[1])
        ego_mat_ori_norm = (ego_mat_ori - np.min(ego_mat_ori)) / (np.max(ego_mat_ori) - np.min(ego_mat_ori) + 1e-9)
        lower_mat_ori_norm = np.ones((image_lower_size[0], image_lower_size[0], 3)) * 0.5
        image = np.transpose(ego_mat_ori_norm, [2, 0, 1])
        lower_image = np.transpose(lower_mat_ori_norm, [2, 0, 1])

        # =============================smpl parameters==============================================
        smpl_params = {'global_orient': pose[0, :, :],
                       'body_pose': pose[1:, :, :],
                       'betas': betas
                       }

        has_smpl_params = {'global_orient': has_body_pose,
                           'body_pose': has_body_pose,
                           'betas': has_betas
                           }

        smpl_params_is_axis_angle = {'global_orient': False,
                                     'body_pose': False,
                                     'betas': False
                                     }
        item = {}
        item['img'] = image
        item['img_lower'] = lower_image
        item['img_size'] = 1.0 * 224
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['keypoints_3d'] = keypoints
        item['imgroot'] = ego_root
        item['idx'] = index
        item['flag'] = np.zeros(9, dtype=np.float32)
        return item

    def __len__(self):
        return len(self.filelist_mat)

class testdataset_Egowang(Dataset):
    def __init__(self, root1, stage):
        self.root1 = root1
        self.stage = stage
        self.imgfile_list = []
        self.gt_p3d_list = []
        if self.stage == 'Test':
            self.env_list = ['jian1', 'jian2', 'jian3', 'lingjie1', 'lingjie2']
        for idx in range(len(self.env_list)):
            env_dir_root = os.path.join(self.root1, self.env_list[idx], 'imgs')
            env_dir_listdir = sorted(os.listdir(env_dir_root))
            for jdx in range(len(env_dir_listdir)):
                ego_root = os.path.join(env_dir_root, env_dir_listdir[jdx])
                self.imgfile_list.append(ego_root)
            # print(len(env_dir_listdir),'||image in ',self.env_list[idx])
            env_gt_root = os.path.join(self.root1, self.env_list[idx], 'gt.pkl')
            p3d_data = joblib.load(env_gt_root)
            for kdx in range(len(p3d_data)):
                self.gt_p3d_list.append(p3d_data[kdx])
            # print(len(p3d_data), '||keypoints 3d in ', self.env_list[idx])
            # print(len(env_dir_listdir1),len(p2d_data),self.env_list[idx])

        self.p3d_list = np.array(self.gt_p3d_list)
        self.length = len(self.imgfile_list)
        self.has_body_pose = np.ones(self.length,dtype = np.float32)
        self.has_betas = np.ones(self.length,dtype = np.float32)

    def resize_image(self, image, resize_height=None, resize_width=None):
        '''
        :param image:
        :param resize_height:
        :param resize_width:
        :return:
        '''
        image_shape = np.shape(image)
        height = image_shape[0]
        width = image_shape[1]
        if (resize_height is None) and (resize_width is None):
            return image
        if resize_height is None:
            resize_height = int(height * resize_width / width)
        elif resize_width is None:
            resize_width = int(width * resize_height / height)
        image = cv2.resize(image, dsize=(resize_width, resize_height))
        return image

    def __getitem__(self, index):

        betas = np.zeros(10)*1.0
        pose = np.ones((24,3,3))*1.0
        keypoints = self.p3d_list[index]
        has_body_pose = self.has_body_pose[index]
        has_betas = self.has_betas[index]

        ego_root = str(self.imgfile_list[index])
        # image_size = np.asarray([224,224])
        image_size = np.asarray([384, 384])
        # image_size = np.asarray([256, 256])
        # image_size = np.asarray([512, 512])
        image_lower_size = np.asarray([56,56])
        # -----------------------add augmentation-------------------------------------
        ego_mat_ori = Image.open(ego_root)
        ego_mat_ori = np.array(ego_mat_ori)
        ego_mat_ori = ego_mat_ori[:,128:-128,:]

        ego_mat_ori = self.resize_image(ego_mat_ori, resize_width=image_size[0], resize_height=image_size[1])
        lower_mat_ori = ego_mat_ori[96:-96, 96:-96, :]
        lower_mat_ori_norm = self.resize_image(lower_mat_ori, resize_width=image_lower_size[0],
                                               resize_height=image_lower_size[1])

        ego_mat_ori_norm = (ego_mat_ori - np.min(ego_mat_ori)) / (np.max(ego_mat_ori) - np.min(ego_mat_ori) + 1e-9)
        lower_mat_ori_norm = (lower_mat_ori_norm - np.min(lower_mat_ori_norm)) / (
                    np.max(lower_mat_ori_norm) - np.min(lower_mat_ori_norm) + 1e-9)

        # lower_mat_ori_norm = np.ones((image_lower_size[0], image_lower_size[0], 3)) * 0.5
        image = np.transpose(ego_mat_ori_norm, [2, 0, 1])
        lower_image = np.transpose(lower_mat_ori_norm, [2, 0, 1])

        # =============================smpl parameters==============================================
        smpl_params = {'global_orient': pose[0, :, :],
                       'body_pose': pose[1:, :, :],
                       'betas': betas
                       }

        has_smpl_params = {'global_orient': has_body_pose,
                           'body_pose': has_body_pose,
                           'betas': has_betas
                           }

        smpl_params_is_axis_angle = {'global_orient': False,
                                     'body_pose': False,
                                     'betas': False
                                     }
        item = {}
        item['img'] = image
        item['img_lower'] = lower_image
        item['img_size'] = 1.0 * 224
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['keypoints_3d'] = keypoints
        item['imgroot'] = str(ego_root)
        item['idx'] = index
        item['flag'] = np.ones(9, dtype=np.float32)
        # item['flag'] = np.zeros(9, dtype=np.float32)
        return item

    def __len__(self):
        return len(self.imgfile_list)

def compute_loss(batch, output, diffusion, cfg, train=True):
    """
    Compute losses given the input batch and the regression output
    Args:
        batch (Dict): Dictionary containing batch data
        output (Dict): Dictionary containing the regression output
        train (bool): Flag indicating whether it is training or validation mode
    Returns:
        torch.Tensor : Total loss for current batch
    Notations:
        This is the baseline mode, so only L-nll L-exp-3D is required for training the network
    """

    keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
    keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
    smpl_parameter_loss = ParameterLoss()
    pred_smpl_params = output['pred_smpl_params']
    pred_pose_6d = output['pred_pose_6d']
    pred_pose_6d_zoom = output['pred_pose_6d_zoom']
    conditioning_feats = output['conditioning_feats']
    conditioning_feats_zoom = output['conditioning_feats_zoom']

    batch_size = pred_smpl_params['body_pose'].shape[0]
    num_samples = 1
    device = pred_smpl_params['body_pose'].device
    dtype = pred_smpl_params['body_pose'].dtype

    # Get annotations
    #        gt_keypoints_2d = batch['keypoints_2d']
    #        gt_keypoints_3d = batch['keypoints_3d']
    gt_smpl_params = batch['smpl_params']
    has_smpl_params = batch['has_smpl_params']
    is_axis_angle = batch['smpl_params_is_axis_angle']
    is_lower = batch['flag'][:,0].view(batch_size,1).to(device)


    # Compute 3D keypoint loss
    #        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d.unsqueeze(1).repeat(1, num_samples, 1, 1))
    #        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d.unsqueeze(1).repeat(1, num_samples, 1, 1), pelvis_id=25+14)

    # Compute loss on SMPL parameters
    loss_smpl_params = {}
    for k, pred in pred_smpl_params.items():
        gt = gt_smpl_params[k].view(batch_size,-1).unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1).to(device)
        if is_axis_angle[k].all():
            gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size * num_samples, -1, 3, 3).to(device)
        has_gt = has_smpl_params[k].unsqueeze(1).repeat(1, num_samples).to(device)
        loss_smpl_params[k] = smpl_parameter_loss(pred.reshape(batch_size, num_samples, -1),
                                                  gt.reshape(batch_size, num_samples, -1), has_gt)

    # Compute mode and expectation losses for 3D and 2D keypoints
    # The first item of the second dimension always corresponds to the mode

    #        loss_keypoints_2d_mode = loss_keypoints_2d[:, [0]].sum() / batch_size
    #        if loss_keypoints_2d.shape[1] > 1:
    #            loss_keypoints_2d_exp = loss_keypoints_2d[:, 1:].sum() / (batch_size * (num_samples - 1))
    #        else:
    #            loss_keypoints_2d_exp = torch.tensor(0., device=device, dtype=dtype)

    #        loss_keypoints_3d_mode = loss_keypoints_3d[:, [0]].sum() / batch_size
    #        if loss_keypoints_3d.shape[1] > 1:
    #            loss_keypoints_3d_exp = loss_keypoints_3d[:, 1:].sum() / (batch_size * (num_samples - 1))
    #        else:
    #            loss_keypoints_3d_exp = torch.tensor(0., device=device, dtype=dtype)
    loss_smpl_params_mode = {k: v[:, [0]].sum() / batch_size for k, v in loss_smpl_params.items()}
    if loss_smpl_params['body_pose'].shape[1] > 1:
        loss_smpl_params_exp = {k: v[:, 1:].sum() / (batch_size * (num_samples - 1)) for k, v in
                                loss_smpl_params.items()}
    else:
        loss_smpl_params_exp = {k: torch.tensor(0., device=device, dtype=dtype) for k, v in loss_smpl_params.items()}

    # Filter out images with corresponding SMPL parameter annotations
    smpl_params = {k: v.clone().to(device) for k, v in gt_smpl_params.items()}
    smpl_params['lower_pose'] = \
        smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)[:, [0, 3, 6, 9, 1, 4, 7, 10], :, :2].permute(0, 1, 3,
                                                                                                                2).reshape(
            batch_size, -1)
    smpl_params['body_pose'] = smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3,
                                                                                                           2).reshape(
        batch_size,  -1)
    smpl_params['global_orient'] = smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1,
                                                                                                                   3,
                                                                                                                   2).reshape(
        batch_size,  -1)

    smpl_params['betas'] = smpl_params['betas']
    has_smpl_params = (batch['has_smpl_params']['body_pose'] > 0)
    smpl_params = {k: v for k, v in smpl_params.items()}
    # Compute NLL loss
    # Add some noise to annotations at training time to prevent overfitting
    if train:
        smpl_params = {k: v + cfg.TRAIN.SMPL_PARAM_NOISE_RATIO * torch.randn_like(v) for k, v in smpl_params.items()}
    if smpl_params['body_pose'].shape[0] > 0:
        loss_mse, loss_mse_lower, _, _ = diffusion.mse_loss(smpl_params, conditioning_feats, conditioning_feats_zoom)
    else:
        loss_mse = torch.zeros(0, device=device, dtype=dtype)
        loss_mse_lower = torch.zeros(0, device=device, dtype=dtype)
    loss_mse_mean = loss_mse.mean() + (is_lower * loss_mse_lower).mean()
    # loss_mse_mean = loss_mse.mean() # without loss_diff_l


    # Compute orthonormal loss on 6D representations
    pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
    loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2,
                                                                                           device=pred_pose_6d.device,
                                                                                           dtype=pred_pose_6d.dtype).unsqueeze(
        0)) ** 2)

    pred_pose_6d_zoom = pred_pose_6d_zoom.reshape(-1, 2, 3).permute(0, 2, 1)
    loss_pose_6d_zoom = ((torch.matmul(pred_pose_6d_zoom.permute(0, 2, 1), pred_pose_6d_zoom) - torch.eye(2,
                                                                                           device=pred_pose_6d_zoom.device,
                                                                                           dtype=pred_pose_6d_zoom.dtype).unsqueeze(
        0)) ** 2)
    loss_pose_6d_mean = loss_pose_6d.reshape(batch_size, -1).mean() + (is_lower * loss_pose_6d_zoom.reshape(batch_size, -1)).mean()


    #        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_EXP'] * loss_keypoints_3d_exp+\
    #               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_EXP'] * loss_keypoints_2d_exp+\
    #               self.cfg.LOSS_WEIGHTS['NLL'] * loss_nll+\
    #               self.cfg.LOSS_WEIGHTS['ORTHOGONAL'] * (loss_pose_6d_exp+loss_pose_6d_mode)+\
    #               sum([loss_smpl_params_exp[k] * self.cfg.LOSS_WEIGHTS[(k+'_EXP').upper()] for k in loss_smpl_params_exp])+\
    #               self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D_MODE'] * loss_keypoints_3d_mode+\
    #               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D_MODE'] * loss_keypoints_2d_mode+\
    #               sum([loss_smpl_params_mode[k] * self.cfg.LOSS_WEIGHTS[(k+'_MODE').upper()] for k in loss_smpl_params_mode])
    #
    #        losses = dict(loss=loss.detach(),
    #                      loss_nll=loss_nll.detach(),
    #                      loss_pose_6d_exp=loss_pose_6d_exp,
    #                      loss_pose_6d_mode=loss_pose_6d_mode,
    #                      loss_keypoints_2d_exp=loss_keypoints_2d_exp.detach(),
    #                      loss_keypoints_3d_exp=loss_keypoints_3d_exp.detach(),
    #                      loss_keypoints_2d_mode=loss_keypoints_2d_mode.detach(),
    #                      loss_keypoints_3d_mode=loss_keypoints_3d_mode.detach())

    loss = cfg.LOSS_WEIGHTS['MSE'] * loss_mse_mean + \
           cfg.LOSS_WEIGHTS['ORTHOGONAL'] * loss_pose_6d_mean + \
           sum([loss_smpl_params_exp[k] * cfg.LOSS_WEIGHTS[(k + '_EXP').upper()] for k in loss_smpl_params_exp]) + \
           sum([loss_smpl_params_mode[k] * cfg.LOSS_WEIGHTS[(k + '_MODE').upper()] for k in loss_smpl_params_mode])

    # loss = cfg.LOSS_WEIGHTS['MSE'] * loss_mse_mean + \
    #            cfg.LOSS_WEIGHTS['ORTHOGONAL'] * loss_pose_6d_mean  # without Loss_params

    losses = dict(loss=loss.detach(),
                  loss_mse=loss_mse_mean.detach(),
                  loss_pose_6d=loss_pose_6d_mean.detach())
    for k, v in loss_smpl_params_exp.items():
        losses['loss_' + k + '_exp'] = v.detach()
    for k, v in loss_smpl_params_mode.items():
        losses['loss_' + k + '_mode'] = v.detach()

    output['losses'] = losses

    return loss


def main():
    # ========================loading preparing=================================
    args = parse_config()
    LOGGER = ConsoleLogger('ablation_without_L_params', 'train')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    cudnn.benchmark = args.BENCHMARK
    cudnn.deterministic = args.DETERMINISTIC
    cudnn.enabled = args.ENABLED
    if args.model_cfg is None:
        model_cfg = prohmr_config()
    else:
        model_cfg = get_config(args.model_cfg)
    root1 = args.root_EgoPW_info
    root2 = args.root_EgoPW_image
    flag_initial = True
    # ====================add dataset===========================================
    train_dataset = EgoPWdataset(root1, root2,  stage='Train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.train_batch_size, shuffle=True, drop_last=True,
                                                   num_workers=args.num_workers)
    # ===================create the model and the loss functions================
    feature_bone = create_backbone(model_cfg)
    feature_bone_zoom = create_backbone(model_cfg)
    diff_bone = SMPLDiffusion(args, model_cfg)
    discriminator = Discriminator()
    smpl_bone = SMPLHead()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        feature_bone = feature_bone.cuda(device)
        feature_bone_zoom = feature_bone_zoom.cuda(device)
        diff_bone = diff_bone.cuda(device)
        discriminator = discriminator.cuda(device)
        smpl_bone = smpl_bone.cuda(device)
    if args.load_model:
        model_path = args.load_model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path)
        feature_bone.load_state_dict(checkpoint['feature_bone_state_dict'])
        feature_bone_zoom.load_state_dict(checkpoint['feature_bone_zoom_state_dict'])
        diff_bone.load_state_dict(checkpoint['diff_bone_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        LOGGER.info(f'---------------Finishing loading models----------------')
    optimizer = torch.optim.AdamW(params=list(feature_bone.parameters()) + list(diff_bone.parameters()) + list(feature_bone_zoom.parameters()),
                                  lr=model_cfg.TRAIN.LR,
                                  weight_decay=model_cfg.TRAIN.WEIGHT_DECAY)
    optimizer_disc = torch.optim.AdamW(params=discriminator.parameters(),
                                       lr=model_cfg.TRAIN.LR,
                                       weight_decay=model_cfg.TRAIN.WEIGHT_DECAY)

    # ================train process=============================================
    for epoch in range(args.max_epoch):
        LOGGER.info(f'---------------Training epoch : {epoch}-----------------')
        batch_time = AverageMeter()
        loss_distri_log = AverageMeter()
        loss_gen_log = AverageMeter()
        loss_disc_log = AverageMeter()
        loss_distri_val_log = AverageMeter()
        start = time.time()

        feature_bone.train()
        feature_bone_zoom.train()
        diff_bone.train()
        discriminator.train()

        for it, batch in enumerate(train_dataloader, 0):
            batch_size = batch['img'].shape[0]
            x = batch['img'].to(torch.float32).to(device)
            x_zoom = batch['img_lower'].to(torch.float32).to(device)
            flag = batch['flag'].to(torch.float32).to(device)
            flag.requires_grad = False
            conditioning_feats = feature_bone(x)
            conditioning_feats_zoom = feature_bone_zoom(x_zoom)
            if flag_initial == True:
                smpl_params = {k: v.clone().to(device) for k, v in batch['smpl_params'].items()}
                has_smpl_params = batch['has_smpl_params']['body_pose'] > 0
                smpl_params['lower_pose'] = \
                    smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)[:, [0, 3, 6, 9, 1, 4, 7, 10], :, :2].permute(
                        0, 1, 3,
                        2).reshape(
                        batch_size, -1)
                smpl_params['body_pose'] = \
                smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, -1)
                smpl_params['global_orient'] = \
                smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(
                    batch_size,  -1)
                smpl_params['betas'] = smpl_params['betas']
                #            conditioning_feats_ini = conditioning_feats[has_smpl_params]
                with torch.no_grad():
                    _, _, _, _ = diff_bone.mse_loss(smpl_params, conditioning_feats, conditioning_feats_zoom)
                flag_initial = False
            params_noise = torch.randn(batch_size, diff_bone.npose, device=x.device)
            params_noise_zoom = torch.randn(batch_size, diff_bone.npose_lower, device=x.device)
            pred_smpl_params, pred_weights, x_T, pred_pose_6d, pred_pose_6d_zoom = diff_bone(params_noise, params_noise_zoom, conditioning_feats, conditioning_feats_zoom,flag)

            output = {}
            output['pred_smpl_params'] = {k: v.clone() for k, v in pred_smpl_params.items()}
            output['noise_params'] = x_T.detach()
            output['conditioning_feats'] = conditioning_feats
            output['conditioning_feats_zoom'] = conditioning_feats_zoom
            output['pred_pose_6d'] = pred_pose_6d
            output['pred_pose_6d_zoom'] = pred_pose_6d_zoom
            output['pred_weights'] = pred_weights


            # Compute model vertices, joints and the projected joints
            pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size , -1,
                                                                                          3, 3)
            pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size , -1, 3, 3)
            pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size , -1)

            # Fit in the SMPL model
            smpl_output, _ = smpl_bone(global_orient=pred_smpl_params['global_orient'],
                                       body_pose=pred_smpl_params['body_pose'], betas=pred_smpl_params['betas'])
            pred_keypoints_3d = smpl_output.joints
            pred_vertices = smpl_output.vertices
            output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
            output['pred_vertices'] = pred_vertices.reshape(batch_size,  -1, 3)

            pred_smpl_params = output['pred_smpl_params']
            num_samples = pred_smpl_params['body_pose'].shape[1]
            pred_smpl_params = output['pred_smpl_params']

            loss_distribution = compute_loss(batch, output, diff_bone, model_cfg, train=True)

            disc_out = discriminator(pred_smpl_params['body_pose'].reshape(batch_size , -1),
                                     pred_smpl_params['betas'].reshape(batch_size , -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss_distribution + model_cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(feature_bone.parameters()) + list(diff_bone.parameters()), 1.0)
            optimizer.step()

            # Train the discriminator
            gt_body = batch['smpl_params']
            gt_body_pose = gt_body['body_pose'].to(device)
            gt_betas = gt_body['betas'].to(device)
            gt_rotmat = gt_body_pose.view(batch_size, -1, 3, 3)
            disc_fake_out = discriminator(pred_smpl_params['body_pose'].reshape(batch_size , -1).detach(),
                                          pred_smpl_params['betas'].reshape(batch_size , -1).detach())
            loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
            disc_real_out = discriminator(gt_rotmat, gt_betas)
            loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
            loss_disc = loss_fake + loss_real
            loss_disc_weight = model_cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
            optimizer_disc.zero_grad()
            loss_disc_weight.backward()
            optimizer_disc.step()

            output['losses']['loss_distri'] = loss_distribution.detach()
            output['losses']['loss_gen'] = loss_adv.detach()
            output['losses']['loss_disc'] = loss_disc.detach()

            # ===============Logging the info and Save the model================
            batch_time.update(time.time() - start)
            loss_distri_log.update(output['losses']['loss_distri'], batch_size)
            loss_gen_log.update(output['losses']['loss_gen'], batch_size)
            loss_disc_log.update(output['losses']['loss_disc'], batch_size)
            if it % args.freq_print_train == 0:
                message = 'Epoch : [{0}][{1}/{2}]  Learning rate  {learning_rate:.5f}\t' \
                          'Batch Time {batch_time.val:.3f}s ({batch_time.ave:.3f})\t' \
                          'Speed {speed:.1f} samples/s \t' \
                          'Loss_distribution {loss1.val:.5f} ({loss1.ave:.5f})\t' \
                          'Loss_generation {loss2.val:.5f}({loss2.ave:.5f})\t' \
                          'Loss_discrimatation {loss3.val:.5f}({loss3.ave:.5f})\t'.format(
                    epoch, it, len(train_dataloader), learning_rate=optimizer.param_groups[0]['lr'],
                    batch_time=batch_time, speed=batch_size / batch_time.val, loss1=loss_distri_log,
                    loss2=loss_gen_log, loss3=loss_disc_log)
                LOGGER.info(message)
            start = time.time()

        checkpoint_dir = os.path.join(logdir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
        states = dict()
        states['feature_bone_state_dict'] = feature_bone.state_dict()
        states['feature_bone_zoom_state_dict'] = feature_bone_zoom.state_dict()
        states['diff_bone_state_dict'] = diff_bone.state_dict()
        states['discriminator_state_dict'] = discriminator.state_dict()
        states['optimizer_state_dict'] = optimizer.state_dict()
        states['optimizer_disc_state_dict'] = optimizer_disc.state_dict()
        torch.save(states, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'))


        # test_offline_mo2cap2(checkpoint_dir,os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'),LOGGER)
        test_offline_EgoWang(checkpoint_dir, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'), LOGGER)
    LOGGER.info('Fininshing.')

def test_offline_mo2cap2(save_path,model_path,LOGGER=None):
    # ========================loading preparing=================================
    args = parse_config()
    if LOGGER == None:
        LOGGER = ConsoleLogger('test_baseline_diff', 'test')
        logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    cudnn.benchmark = args.BENCHMARK
    cudnn.deterministic = args.DETERMINISTIC
    cudnn.enabled = args.ENABLED
    if args.model_cfg is None:
        model_cfg = prohmr_config()
    else:
        model_cfg = get_config(args.model_cfg)

    root1 = '/home/work/Yuxuan/Data/Mo2Cap2_test'
    # root1 = '/home/work/Yuxuan/Data/Ego_Wang/Test_global'
    flag_initial = True
    # ====================add dataset===========================================
    test_dataset = testdataset_mo2cap2(root1, stage='Test')
    # test_dataset = testdataset_Egowang(root1, stage='Test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, args.test_batch_size, shuffle=False, drop_last=False,
                                                  num_workers=args.num_workers)
    # =========================================================================
    feature_bone = create_backbone(model_cfg)
    feature_bone_zoom = create_backbone(model_cfg)
    diff_bone = SMPLDiffusion(args, model_cfg)
    discriminator = Discriminator()
    smpl_bone = SMPLHead()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        feature_bone = feature_bone.cuda(device)
        feature_bone_zoom = feature_bone_zoom.cuda(device)
        diff_bone = diff_bone.cuda(device)
        discriminator = discriminator.cuda(device)
        smpl_bone = smpl_bone.cuda(device)
    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path)
        feature_bone.load_state_dict(checkpoint['feature_bone_state_dict'])
        feature_bone_zoom.load_state_dict(checkpoint['feature_bone_zoom_state_dict'])
        diff_bone.load_state_dict(checkpoint['diff_bone_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        LOGGER.info(f'---------------Finishing loading models----------------')

    # ================train process=============================================

    LOGGER.info(f'---------------Begin Testing-----------------')
    feature_bone.eval()
    feature_bone_zoom.eval()
    diff_bone.eval()
    discriminator.eval()
    keypoints_3d_whole_pred = []
    img_root_whole = []

    with torch.no_grad():
        for it, batch in enumerate(test_dataloader, 0):
            batch_size = batch['img'].shape[0]
            x = batch['img'].to(torch.float32).to(device)
            x_zoom = batch['img_lower'].to(torch.float32).to(device)
            flag = batch['flag'].to(torch.float32).to(device)
            conditioning_feats = feature_bone(x)
            conditioning_feats_zoom = feature_bone_zoom(x_zoom)


            #=============================== sampling points ===========================================================
            params_noise = torch.randn(batch_size, diff_bone.npose, device=x.device)
            params_noise_zoom = torch.randn(batch_size, diff_bone.npose_lower, device=x.device)
            # params_noise = torch.zeros(batch_size, diff_bone.npose, device=x.device) * 0.0
            #===========================================================================================================
            pred_smpl_params, pred_weights, x_T, pred_pose_6d, pred_pose_6d_zoom = diff_bone(params_noise, params_noise_zoom, conditioning_feats, conditioning_feats_zoom,flag)
            output = {}
            output['pred_smpl_params'] = {k: v.clone() for k, v in pred_smpl_params.items()}
            output['noise_params'] = x_T.detach()
            output['conditioning_feats'] = conditioning_feats
            output['conditioning_feats_zoom'] = conditioning_feats_zoom


            # Compute model vertices, joints and the projected joints
            pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size ,
                                                                                          -1, 3, 3)
            pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size , -1, 3,
                                                                                  3)
            pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size , -1)

            # Fit in the SMPL model
            smpl_output, _ = smpl_bone(global_orient=pred_smpl_params['global_orient'],
                                       body_pose=pred_smpl_params['body_pose'], betas=pred_smpl_params['betas'])
            pred_keypoints_3d = smpl_output.joints.reshape(batch_size, -1, 3)
            pred_vertices = smpl_output.vertices
            output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
            output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)


            for idx in range(batch_size):
                keypoints_3d_whole_pred.append(pred_keypoints_3d[idx].cpu().numpy())
                img_root_whole.append(batch['imgroot'][idx])
        keypoints_3d_whole_pred = np.array(keypoints_3d_whole_pred)
        warpping_relate = [12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 0, 1, 4, 7, 0]
        pred_joints = keypoints_3d_whole_pred[:, warpping_relate, :]
        pred_joints[:, 10, :] = (keypoints_3d_whole_pred[:, 32, :] + keypoints_3d_whole_pred[:, 33, :]) / 2
        pred_joints[:, 14, :] = (keypoints_3d_whole_pred[:, 29, :] + keypoints_3d_whole_pred[:, 30, :]) / 2

        weipeng_root = '/home/work/Yuxuan/Data/Mo2Cap2_test/weipeng_studio_results.mat'
        gt_weipeng = scipy.io.loadmat(weipeng_root)['gt']
        olek_root = '/home/work/Yuxuan/Data/Mo2Cap2_test/olek_outdoor_results.mat'
        gt_olek = scipy.io.loadmat(olek_root)['gt']
        gt_whole = np.concatenate((gt_weipeng,gt_olek),axis=-1)
        gt = np.ones((len(pred_joints),15,3))
        for i in range(len(pred_joints)):
            for j in range(15):
                gt[i,j,:] = gt_whole[:,j,i]
        error,_,_ =  p_mpjpe(pred_joints, gt)
        LOGGER.info('Mean PA MPJPE:{:.5f}'.format(error))
        # DATA = {}
        # DATA['pred_keypoints'] = np.array(keypoints_3d_whole_pred)
        # DATA['img_root'] = img_root_whole
        # scipy.io.savemat(os.path.join(save_path,'test_result_mo2cap2.mat'),DATA)
    LOGGER.info('Fininshing Testing Offline-mo2cap2')

def test_offline_EgoWang(save_path,model_path,LOGGER=None):
    # ========================loading preparing=================================
    args = parse_config()
    if LOGGER == None:
        LOGGER = ConsoleLogger('test_baseline_diff', 'test')
        logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    cudnn.benchmark = args.BENCHMARK
    cudnn.deterministic = args.DETERMINISTIC
    cudnn.enabled = args.ENABLED
    if args.model_cfg is None:
        model_cfg = prohmr_config()
    else:
        model_cfg = get_config(args.model_cfg)

    root1 = '/home/work/Yuxuan/Data/Ego_Wang/Test_global'
    flag_initial = True
    # ====================add dataset===========================================
    test_dataset = testdataset_Egowang(root1, stage='Test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, args.test_batch_size, shuffle=False, drop_last=False,
                                                  num_workers=args.num_workers)
    # =========================================================================
    feature_bone = create_backbone(model_cfg)
    feature_bone_zoom = create_backbone(model_cfg)
    diff_bone = SMPLDiffusion(args, model_cfg)
    discriminator = Discriminator()
    smpl_bone = SMPLHead()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        feature_bone = feature_bone.cuda(device)
        feature_bone_zoom = feature_bone_zoom.cuda(device)
        diff_bone = diff_bone.cuda(device)
        discriminator = discriminator.cuda(device)
        smpl_bone = smpl_bone.cuda(device)
    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path)
        feature_bone.load_state_dict(checkpoint['feature_bone_state_dict'])
        feature_bone_zoom.load_state_dict(checkpoint['feature_bone_zoom_state_dict'])
        diff_bone.load_state_dict(checkpoint['diff_bone_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        LOGGER.info(f'---------------Finishing loading models----------------')

    # ================train process=============================================

    LOGGER.info(f'---------------Begin Testing-----------------')
    feature_bone.eval()
    feature_bone_zoom.eval()
    diff_bone.eval()
    discriminator.eval()
    keypoints_3d_whole_pred = []
    keypoints_3d_whole_gt = []
    img_root_whole = []
    verts_whole = []

    with torch.no_grad():
        for it, batch in enumerate(test_dataloader, 0):
            batch_size = batch['img'].shape[0]
            x = batch['img'].to(torch.float32).to(device)
            x_zoom = batch['img_lower'].to(torch.float32).to(device)
            p3d_gt = batch['keypoints_3d'].to(torch.float32).to(device)
            flag = batch['flag'].to(torch.float32).to(device)
            conditioning_feats = feature_bone(x)
            conditioning_feats_zoom = feature_bone_zoom(x_zoom)


            #=============================== sampling points ===========================================================
            params_noise = torch.randn(batch_size, diff_bone.npose, device=x.device)
            params_noise_zoom = torch.randn(batch_size, diff_bone.npose_lower, device=x.device)
            # params_noise = torch.zeros(batch_size, diff_bone.npose, device=x.device) * 0.0
            #===========================================================================================================
            pred_smpl_params, pred_weights, x_T, pred_pose_6d, pred_pose_6d_zoom = diff_bone(params_noise, params_noise_zoom, conditioning_feats, conditioning_feats_zoom,flag)
            output = {}
            output['pred_smpl_params'] = {k: v.clone() for k, v in pred_smpl_params.items()}
            output['noise_params'] = x_T.detach()
            output['conditioning_feats'] = conditioning_feats
            output['conditioning_feats_zoom'] = conditioning_feats_zoom


            # Compute model vertices, joints and the projected joints
            pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size ,
                                                                                          -1, 3, 3)
            pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size , -1, 3,
                                                                                  3)
            pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size , -1)

            # Fit in the SMPL model
            smpl_output, _ = smpl_bone(global_orient=pred_smpl_params['global_orient'],
                                       body_pose=pred_smpl_params['body_pose'], betas=pred_smpl_params['betas'])
            pred_keypoints_3d = smpl_output.joints.reshape(batch_size, -1, 3)
            pred_vertices = smpl_output.vertices
            output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
            output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)


            for idx in range(batch_size):
                keypoints_3d_whole_pred.append(pred_keypoints_3d[idx].cpu().numpy())
                keypoints_3d_whole_gt.append(p3d_gt[idx].cpu().numpy())
                img_root_whole.append(batch['imgroot'][idx])
                verts_whole.append(output['pred_vertices'][idx].cpu().numpy())
        keypoints_3d_whole_pred = np.array(keypoints_3d_whole_pred)
        keypoints_3d_whole_gt = np.array(keypoints_3d_whole_gt)
        warpping_relate = [12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 0, 1, 4, 7, 0]
        pred_joints = keypoints_3d_whole_pred[:, warpping_relate, :]
        pred_joints[:, 10, :] = (keypoints_3d_whole_pred[:, 32, :] + keypoints_3d_whole_pred[:, 33, :]) / 2
        pred_joints[:, 14, :] = (keypoints_3d_whole_pred[:, 29, :] + keypoints_3d_whole_pred[:, 30, :]) / 2

        error,_,_ =  p_mpjpe(pred_joints, keypoints_3d_whole_gt)
        print('Mean PA MPJPE:{:.5f}'.format(error))
        LOGGER.info('Mean PA MPJPE:{:.5f}'.format(error))
        DATA = {}
        DATA['gt_keypoints'] = np.array(keypoints_3d_whole_gt)
        DATA['pred_keypoints'] = np.array(pred_joints)
        DATA['img_root'] = img_root_whole
        scipy.io.savemat(os.path.join(save_path,'ours_EgoPW.mat'),DATA)
        DATA = {}
        DATA['img_root'] = img_root_whole
        DATA['pred_verts'] = np.array(verts_whole)
        scipy.io.savemat(os.path.join(save_path, 'ours_EgoPW_verts.mat'), DATA)
    LOGGER.info('Fininshing Testing Offline-EgoWang')

if __name__ == '__main__':
    # main()
    # checkpoint_model = '/opt/data/private/guogroup/liuyuxuan/software/ProHMR/train/experiments/train_baseline_diff_groupnorm/2022-08-01-08-02-30/checkpoints/checkpoint_19.tar'
    # save_path = '/opt/data/private/guogroup/liuyuxuan/software/ProHMR/train/experiments/train_baseline_diff_groupnorm/2022-08-01-08-02-30/test'
    # test(save_path,checkpoint_model)
    # checkpoint_model = '/opt/data/private/guogroup/liuyuxuan/software/ProHMR/train/experiments/train_baseline_diff_groupnorm/2022-08-04-09-23-02/checkpoints/checkpoint_17.tar'
    # save_path = '/opt/data/private/guogroup/liuyuxuan/software/ProHMR/train/experiments/train_baseline_diff_groupnorm/2022-08-04-09-23-02'
    # save_path = '/home/work/Yuxuan/Code/ESRDM/experiments/train_diff_selective/2022-08-18-21-45-53/checkpoints'
    # checkpoint_model = '/home/work/Yuxuan/Code/ESRDM/experiments/train_diff_selective/2022-08-18-21-45-53/checkpoints/checkpoint_26.tar'
    # test_offline(save_path,checkpoint_model)
    # test_offline_mo2cap2(save_path,checkpoint_model)
    # temporalOptim(save_path,checkpoint_model)
    save_path = '/home/work/Yuxuan/Code/ESRDM/results'
    checkpoint_model = '/home/work/Yuxuan/Code/ESRDM/experiments/train_diff_EgoPW/2022-08-21-22-35-08/checkpoints/checkpoint_21.tar'
    # for i in range(10):
    test_offline_EgoWang(save_path, checkpoint_model)
    # #======================================test for dataset================================================
    # args = parse_config()
    # root1 = args.root_info
    # root2 = args.root_smpl
    # root3 = args.root_p2d
    # # # ====================add dataset===========================================
    # train_dataset = wholedataset(root1, root2, root3, stage='Train')
    # # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True,
    # #                                                num_workers=args.num_workers)
    # # for it, batch in enumerate(train_dataloader, 0):
    # #     if it % 10 == 0:
    # #         print(it)
    # data = train_dataset[10000]
    # img = np.transpose(data['img'],[1,2,0])
    # img_lower = np.transpose(data['img_lower'],[1,2,0])
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.figure()
    # plt.imshow(img_lower)
    # plt.show()
    # args = parse_config()
    # root1 = args.root_info_test
    # root2 = args.root_smpl
    # root3 = args.root_p2d
    # flag_initial = True
    # # ====================add dataset===========================================
    # test_dataset = testdataset(root1, root2, root3, stage='Test')
    # a = test_dataset[0]

    # args = parse_config()
    # root1 = args.root_EgoPW_info
    # root2 = args.root_EgoPW_image
    # dataset = EgoPWdataset(root1,root2,'Train')
    # print(len(dataset))
    # train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=320, shuffle=False, drop_last=False,
    #                                                num_workers=args.num_workers)
    # for it, batch in enumerate(train_dataloader, 0):
    #     if it % 10 == 0:
    #         print(it)
    # from tqdm import tqdm
    # for i in tqdm(range(len(dataset))):
    #     a = dataset[i]
    # dataset = testdataset_Egowang(root1,'Test')
    # a = dataset[0]
    # print(a)