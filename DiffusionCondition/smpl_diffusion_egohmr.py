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
import math
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
        pred_weights = torch.sigmoid(offset)  ## ~[0,1.0]
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
            pred_smpl = self.step_forward(x_T, x_T_zoom, feats, feats_zoom, flag, tdx)
            pred_smpl_params_list.append(pred_smpl)
        return pred_smpl_params_list

    def step_forward(self, x_T, x_T_zoom, feats, feats_zoom, flag, step):
        """
        Algorithm 2.
        """
        batch_size = feats.shape[0]
        # for original image
        x_t = x_T
        for time_step in reversed(range(step, self.T)):
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
        for time_step in reversed(range(step, self.T)):
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
        pred_pose_zoom = x_0_zoom[:, :self.npose_lower]
        pred_pose_6d_zoom = pred_pose_zoom.clone()

        pred_betas = self.fc_head(feats, feats_zoom)
        pred_weights = flag * self.fc_weight(feats, feats_zoom)

        pred_pose_output = torch.zeros(batch_size, self.npose).to(feats.device)

        pred_pose_output[:, 0:6] = self.average(pred_pose[:, 0:6], pred_pose_zoom[:, 0:6], pred_weights[:, 0])
        pred_pose_output[:, 6:12] = self.average(pred_pose[:, 6:12], pred_pose_zoom[:, 6:12], pred_weights[:, 1])
        pred_pose_output[:, 12:18] = self.average(pred_pose[:, 12:18], pred_pose_zoom[:, 30:36], pred_weights[:, 2])
        pred_pose_output[:, 24:30] = self.average(pred_pose[:, 24:30], pred_pose_zoom[:, 12:18], pred_weights[:, 3])
        pred_pose_output[:, 30:36] = self.average(pred_pose[:, 30:36], pred_pose_zoom[:, 36:42], pred_weights[:, 4])
        pred_pose_output[:, 42:48] = self.average(pred_pose[:, 42:48], pred_pose_zoom[:, 18:24], pred_weights[:, 5])
        pred_pose_output[:, 48:54] = self.average(pred_pose[:, 48:54], pred_pose_zoom[:, 42:48], pred_weights[:, 6])
        pred_pose_output[:, 60:66] = self.average(pred_pose[:, 60:66], pred_pose_zoom[:, 24:30], pred_weights[:, 7])
        pred_pose_output[:, 66:72] = self.average(pred_pose[:, 66:72], pred_pose_zoom[:, 48:54], pred_weights[:, 8])

        pred_pose_output[:, 18:24] = pred_pose[:, 18:24]
        pred_pose_output[:, 36:42] = pred_pose[:, 36:42]
        pred_pose_output[:, 54:60] = pred_pose[:, 54:60]
        pred_pose_output[:, 72:144] = pred_pose[:, 72:144]

        pred_pose_output = rot6d_to_rotmat(pred_pose_output.reshape(batch_size, -1)).view(batch_size,
                                                                                          self.cfg.SMPL.NUM_BODY_JOINTS + 1,
                                                                                          3, 3)

        pred_smpl_params = {'global_orient': pred_pose_output[:, [0]],
                            'body_pose': pred_pose_output[:, 1:]}

        pred_smpl_params['betas'] = pred_betas.view(-1, 10)

        return pred_smpl_params
#=======================================================================================================================
