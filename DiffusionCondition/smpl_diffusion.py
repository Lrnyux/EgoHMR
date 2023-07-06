# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 21:54:40 2022

@author: liuyuxuan
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from typing import Optional, Dict, Tuple
from DiffusionCondition.DiffusionCondition_Ego import GaussianDiffusionTrainer, GaussianDiffusionSampler
from DiffusionCondition.ModelCondition_Ego import LinearModel
from yacs.config import CfgNode
from prohmr.utils.geometry import rot6d_to_rotmat
import torch.nn.functional as F
import numpy as np


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
                                    nn.Linear(cfg.MODEL.FC_HEAD.NUM_FEATURES, 13))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        mean_params = np.load(cfg.SMPL.MEAN_PARAMS)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32))[None, None]
        init_betas = torch.from_numpy(mean_params['shape'].astype(np.float32))[None, None]

        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_betas', init_betas)

    def forward(self, smpl_params: Dict, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run forward pass.
        Args:
            smpl_params (Dict): Dictionary containing predicted SMPL parameters.
            feats (torch.Tensor): Tensor of shape (N, C) containing the features computed by the backbone.
        Returns:
            pred_betas (torch.Tensor): Predicted SMPL betas.
            pred_cam (torch.Tensor): Predicted camera parameters.
        """

        batch_size = feats.shape[0]

        offset = self.layers(feats).reshape(batch_size, 13)
        betas_offset = offset[:, :10]
        cam_offset = offset[:, 10:]
        pred_cam = cam_offset + self.init_cam
        pred_betas = betas_offset + self.init_betas

        return pred_betas, pred_cam


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
        self.model = LinearModel(T=args.T, ch=args.ch, in_size=self.npose, out_size=self.npose, ns=args.num_stage,
                                 ls=args.latent_size, p_dropout=args.p_dropout)
        self.fc_head = FCHead(cfg)
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

    def mse_loss(self, x_0, feats):
        """
        Algorithm 1.
        """
        x_0 = torch.cat((x_0['global_orient'], x_0['body_pose']), dim=-1)
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, feats), noise, reduction='none')
        return loss, noise

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

    def forward(self, x_T, feats):
        """
        Algorithm 2.
        """
        batch_size = feats.shape[0]
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
        x_0 = x_t
        pred_pose = x_0[:, :self.npose]
        pred_pose_6d = pred_pose.clone()
        pred_pose = rot6d_to_rotmat(pred_pose.reshape(batch_size, -1)).view(batch_size,
                                                                            self.cfg.SMPL.NUM_BODY_JOINTS + 1, 3, 3)
        pred_smpl_params = {'global_orient': pred_pose[:, [0]],
                            'body_pose': pred_pose[:, 1:]}
        pred_betas, pred_cam = self.fc_head(pred_smpl_params, feats)
        pred_smpl_params['betas'] = pred_betas.view(-1,10)

        return pred_smpl_params, pred_cam, x_T, pred_pose_6d














