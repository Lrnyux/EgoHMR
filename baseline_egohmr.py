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
from configs import get_config, prohmr_config, dataset_config
from utils.geometry import rot6d_to_rotmat
from utils.geometry import aa_to_rotmat, perspective_projection
from models import SMPL, SMPLHead
from models.backbones import create_backbone
from models.discriminator import Discriminator
from models.losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from train.evaluate_joints import  eva_joints
from DiffusionCondition.smpl_diffusion_egohmr import SMPLDiffusion
from tools import AverageMeter, ConsoleLogger
from tqdm import tqdm
from dataset import wholedataset,wholedataset_baseline
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#=======================================================================================================================
#NOTIFICATION: baseline for diffusion model based EgoShapo
# add zoom in according to the 2D keypoints detecion by 2D heatmaps and average the lower pose parameters
# zoom in mode : selective
#=======================================================================================================================

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
    #        cfg.LOSS_WEIGHTS['ORTHOGONAL'] * loss_pose_6d_mean  # without Loss_params

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
    LOGGER = ConsoleLogger('train_final_ours_without_Loss_params', 'train')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    cudnn.benchmark = args.BENCHMARK
    cudnn.deterministic = args.DETERMINISTIC
    cudnn.enabled = args.ENABLED
    if args.model_cfg is None:
        model_cfg = prohmr_config()
    else:
        model_cfg = get_config(args.model_cfg)
    root1 = args.root_info
    root2 = args.root_smpl
    root3 = args.root_p2d
    flag_initial = True
    # ====================add dataset===========================================
    train_dataset = wholedataset_baseline(root1, root2, root3, stage='Train')
    val_dataset = wholedataset_baseline(root1, root2, root3,  stage='Val')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.train_batch_size, shuffle=True, drop_last=True,
                                                   num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.val_batch_size, shuffle=False, drop_last=True,
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

        feature_bone.eval()
        feature_bone_zoom.eval()
        diff_bone.eval()
        discriminator.eval()
        with torch.no_grad():
            for it, batch in enumerate(val_dataloader, 0):
                batch_size = batch['img'].shape[0]
                x = batch['img'].to(torch.float32).to(device)
                x_zoom = batch['img_lower'].to(torch.float32).to(device)
                flag = batch['flag'].to(torch.float32).to(device)
                conditioning_feats = feature_bone(x)
                conditioning_feats_zoom = feature_bone_zoom(x_zoom)

                params_noise = torch.randn(batch_size, diff_bone.npose, device=x.device)
                params_noise_zoom = torch.randn(batch_size, diff_bone.npose_lower, device=x.device)
                pred_smpl_params, pred_weights, x_T, pred_pose_6d, pred_pose_6d_zoom = diff_bone(params_noise,
                                                                                                 params_noise_zoom,
                                                                                                 conditioning_feats,
                                                                                                 conditioning_feats_zoom,
                                                                                                 flag)

                output = {}
                output['pred_smpl_params'] = {k: v.clone() for k, v in pred_smpl_params.items()}
                output['noise_params'] = x_T.detach()
                output['conditioning_feats'] = conditioning_feats
                output['conditioning_feats_zoom'] = conditioning_feats_zoom
                output['pred_pose_6d'] = pred_pose_6d
                output['pred_pose_6d_zoom'] = pred_pose_6d_zoom
                output['pred_weights'] = pred_weights


                # Compute model vertices, joints and the projected joints
                pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size,
                                                                                              -1, 3, 3)
                pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size , -1, 3,
                                                                                      3)
                pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size , -1)

                # Fit in the SMPL model
                smpl_output, _ = smpl_bone(global_orient=pred_smpl_params['global_orient'],
                                           body_pose=pred_smpl_params['body_pose'], betas=pred_smpl_params['betas'])
                pred_keypoints_3d = smpl_output.joints
                pred_vertices = smpl_output.vertices
                output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size,  -1, 3)
                output['pred_vertices'] = pred_vertices.reshape(batch_size,  -1, 3)

                pred_smpl_params = output['pred_smpl_params']
                num_samples = pred_smpl_params['body_pose'].shape[1]

                loss_distribution = compute_loss(batch, output, diff_bone, model_cfg, train=False)

                output['losses']['loss_distri'] = loss_distribution
                # ===============Logging the info and Save the model============
                loss_distri_val_log.update(output['losses']['loss_distri'], batch_size)
                if it % args.freq_print_val == 0:
                    message = 'Loss_val_Distribution {loss_val_log.val:.5f} ({loss_val_log.ave:.5f})\t'.format(
                        loss_val_log=loss_distri_val_log)
                    LOGGER.info(message)
            message = 'Loss_val_Distribution {loss_val_log.val:.5f} ({loss_val_log.ave:.5f})\t'.format(
                    loss_val_log=loss_distri_val_log)
            LOGGER.info(message)
    LOGGER.info('Fininshing.')


def test(save_path,model_path):
    # ========================loading preparing=================================
    args = parse_config()
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
    root1 = args.root_info
    root2 = args.root_smpl
    flag_initial = True
    # ====================add dataset===========================================
    test_dataset = wholedataset(root1, root2, stage='Val')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, args.test_batch_size, shuffle=False, drop_last=True,
                                                 num_workers=args.num_workers)

    # ===================create the model and the loss functions================
    feature_bone = create_backbone(model_cfg)
    diff_bone = SMPLDiffusion(args, model_cfg)
    discriminator = Discriminator()
    smpl_bone = SMPLHead()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        feature_bone = feature_bone.cuda(device)
        diff_bone = diff_bone.cuda(device)
        discriminator = discriminator.cuda(device)
        smpl_bone = smpl_bone.cuda(device)
    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path)
        feature_bone.load_state_dict(checkpoint['feature_bone_state_dict'])
        diff_bone.load_state_dict(checkpoint['diff_bone_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        LOGGER.info(f'---------------Finishing loading models----------------')

    # ================train process=============================================

    LOGGER.info(f'---------------Begin Testing-----------------')
    loss_distri_log = AverageMeter()
    feature_bone.eval()
    diff_bone.eval()
    discriminator.eval()
    with torch.no_grad():
        for it, batch in enumerate(test_dataloader, 0):
            batch_size = batch['img'].shape[0]
            x = batch['img'].to(torch.float32).to(device)
            conditioning_feats = feature_bone(x)

            params_noise = torch.randn(batch_size, diff_bone.npose, device=x.device)
            pred_smpl_params, pred_cam, x_T, pred_pose_6d = diff_bone(params_noise, conditioning_feats)
            pred_cam = pred_cam.reshape(batch_size, 3)
            output = {}
            output['pred_cam'] = pred_cam.reshape(batch_size, 3)
            output['pred_smpl_params'] = {k: v.clone() for k, v in pred_smpl_params.items()}
            output['noise_params'] = x_T.detach()
            output['conditioning_feats'] = conditioning_feats
            output['pred_pose_6d'] = pred_pose_6d

            # Compute model vertices, joints and the projected joints
            pred_smpl_params['global_orient'] = pred_smpl_params['global_orient'].reshape(batch_size ,
                                                                                          -1, 3, 3)
            pred_smpl_params['body_pose'] = pred_smpl_params['body_pose'].reshape(batch_size , -1, 3,
                                                                                  3)
            pred_smpl_params['betas'] = pred_smpl_params['betas'].reshape(batch_size , -1)

            # Fit in the SMPL model
            smpl_output, _ = smpl_bone(global_orient=pred_smpl_params['global_orient'],
                                       body_pose=pred_smpl_params['body_pose'], betas=pred_smpl_params['betas'])
            pred_keypoints_3d = smpl_output.joints
            pred_vertices = smpl_output.vertices
            output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size,  -1, 3)
            output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
            loss_distribution = compute_loss(batch, output, diff_bone, model_cfg, train=True)

            output['losses']['loss_distri'] = loss_distribution

            smpl_output_ps, _ = smpl_bone(global_orient=batch['smpl_params']['global_orient'].to(device).view(batch_size,1,3,3),
                                       body_pose=batch['smpl_params']['body_pose'].to(device), betas=batch['smpl_params']['betas'].to(device))
            ps_keypoints_3d = smpl_output_ps.joints
            ps_vertices = smpl_output_ps.vertices
            output['pseudo_keypoints_3d'] = ps_keypoints_3d.reshape(batch_size, 1, -1, 3)
            output['pseudo_vertices'] = ps_vertices.reshape(batch_size, 1, -1, 3)

            for idx in range(batch_size):
                DATA={}
                DATA['img'] = batch['img'][idx].cpu().numpy()
                DATA['pred_keypoints_3d'] = output['pred_keypoints_3d'][idx].cpu().numpy()
                DATA['pred_poses_global'] = output['pred_smpl_params']['global_orient'][idx].cpu().numpy()
                DATA['pred_poses_body']  = output['pred_smpl_params']['body_pose'][idx].cpu().numpy()
                DATA['pred_betas'] = output['pred_smpl_params']['betas'][idx].cpu().numpy()
                DATA['pred_vertices'] = output['pred_vertices'][idx].cpu().numpy()
                DATA['gt_poses_global'] = batch['smpl_params']['global_orient'][idx].cpu().numpy().reshape(1,3,3)
                DATA['gt_poses_body'] = batch['smpl_params']['body_pose'][idx].cpu().numpy()
                DATA['gt_betas'] = batch['smpl_params']['betas'][idx].cpu().numpy()
                DATA['pseudo_keypoints_3d'] = output['pseudo_keypoints_3d'][idx].cpu().numpy()
                DATA['gt_vertices'] = output['pseudo_vertices'][idx].cpu().numpy()
                DATA['img_root'] = batch['imgroot'][idx]
                env_name = batch['imgenv'][idx]
                frame_idx = str(batch['imgname'][idx].numpy())
                save_root_per = os.path.join(save_path,env_name+'_'+str(frame_idx).zfill(5)+'.mat')
                scipy.io.savemat(save_root_per,DATA)
            # ===============Logging the info and Save the model============
            loss_distri_log.update(output['losses']['loss_distri'], batch_size)
            if it % args.freq_print_test == 0:
                message = 'Loss_test_Distribution {loss_val_log.val:.5f} ({loss_val_log.ave:.5f})\t'.format(
                    loss_val_log=loss_distri_log)
                LOGGER.info(message)
        message = 'Loss_test_Distribution {loss_val_log.val:.5f} ({loss_val_log.ave:.5f})\t'.format(
                    loss_val_log=loss_distri_log)
        LOGGER.info(message)
    LOGGER.info('Fininshing Testing')


if __name__ == '__main__':
    main()
