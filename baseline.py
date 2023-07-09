# -*- coding: utf-8 -*-

import argparse
import torch
import torch.backends.cudnn as cudnn
import time
import os
from torch.utils.data import Dataset
import scipy.io
from configs import get_config, prohmr_config
from utils.geometry import aa_to_rotmat, perspective_projection
from models import SMPLHead
from models.backbones import create_backbone
from DiffusionCondition.smpl_diffusion import SMPLDiffusion
from models.discriminator import Discriminator
from models.losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from tools import AverageMeter, ConsoleLogger
from dataset import wholedataset
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def parse_config():
    parser = argparse.ArgumentParser(description='ProHMR training code')
    parser.add_argument('--BENCHMARK', default=True, type=bool)
    parser.add_argument('--DETERMINISTIC', default=False, type=bool)
    parser.add_argument('--ENABLED', default=True, type=bool)
    parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file')
    parser.add_argument('--root_info', type=str,
                        default='/opt/data/private/guogroup/liuyuxuan/data/Ego_data_final/Info',
                        help='Root to the info stored in matlab files')
    parser.add_argument('--root_info_test', type=str,
                        default='/opt/data/private/guogroup/liuyuxuan/data/Ego_data_final/Info_vicon',
                        help='Root to the info stored in matlab files')
    parser.add_argument('--root_smpl', type=str,
                        default='/opt/data/private/guogroup/liuyuxuan/data/Ego_data_final/shape_info',
                        help='Root to the info stored in smpl parameters')
    parser.add_argument('--gpu', default=1, help='type of gpu', type=int)

    parser.add_argument('--train_batch_size', default=96, help='batch-size for training', type=int)
    parser.add_argument('--val_batch_size', default=48, help='batch-size for validation', type=int)
    parser.add_argument('--test_batch_size', default=48, help='batch-size for testiing', type=int)
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


    args = parser.parse_args()
    return args


def compute_loss(batch, output, diffusion, cfg, train=True):

    smpl_parameter_loss = ParameterLoss()
    pred_smpl_params = output['pred_smpl_params']
    pred_pose_6d = output['pred_pose_6d']
    conditioning_feats = output['conditioning_feats']
    batch_size = pred_smpl_params['body_pose'].shape[0]
    num_samples = 1
    device = pred_smpl_params['body_pose'].device
    dtype = pred_smpl_params['body_pose'].dtype
    gt_smpl_params = batch['smpl_params']
    has_smpl_params = batch['has_smpl_params']
    is_axis_angle = batch['smpl_params_is_axis_angle']
    # Compute loss on SMPL parameters
    loss_smpl_params = {}
    for k, pred in pred_smpl_params.items():
        gt = gt_smpl_params[k].view(batch_size,-1).unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1).to(device)
        if is_axis_angle[k].all():
            gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size * num_samples, -1, 3, 3).to(device)
        has_gt = has_smpl_params[k].unsqueeze(1).repeat(1, num_samples).to(device)
        loss_smpl_params[k] = smpl_parameter_loss(pred.reshape(batch_size, num_samples, -1),
                                                  gt.reshape(batch_size, num_samples, -1), has_gt)


    loss_smpl_params_mode = {k: v[:, [0]].sum() / batch_size for k, v in loss_smpl_params.items()}
    if loss_smpl_params['body_pose'].shape[1] > 1:
        loss_smpl_params_exp = {k: v[:, 1:].sum() / (batch_size * (num_samples - 1)) for k, v in
                                loss_smpl_params.items()}
    else:
        loss_smpl_params_exp = {k: torch.tensor(0., device=device, dtype=dtype) for k, v in loss_smpl_params.items()}
    smpl_params = {k: v.clone().to(device) for k, v in gt_smpl_params.items()}
    smpl_params['body_pose'] = smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3,
                                                                                                           2).reshape(
        batch_size,  -1)
    smpl_params['global_orient'] = smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1,
                                                                                                                   3,
                                                                                                                   2).reshape(
        batch_size,  -1)
    smpl_params['betas'] = smpl_params['betas']
    smpl_params = {k: v for k, v in smpl_params.items()}
    if train:
        smpl_params = {k: v + cfg.TRAIN.SMPL_PARAM_NOISE_RATIO * torch.randn_like(v) for k, v in smpl_params.items()}
    if smpl_params['body_pose'].shape[0] > 0:
        loss_mse, _ = diffusion.mse_loss(smpl_params, conditioning_feats)
    else:
        loss_mse = torch.zeros(0, device=device, dtype=dtype)
    loss_mse_mean = loss_mse.mean()
    pred_pose_6d = pred_pose_6d.reshape(-1, 2, 3).permute(0, 2, 1)
    loss_pose_6d = ((torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2,
                                                                                           device=pred_pose_6d.device,
                                                                                           dtype=pred_pose_6d.dtype).unsqueeze(
        0)) ** 2)
    loss_pose_6d_mean = loss_pose_6d.reshape(batch_size, -1).mean()
    loss = cfg.LOSS_WEIGHTS['MSE'] * loss_mse_mean + \
           cfg.LOSS_WEIGHTS['ORTHOGONAL'] * loss_pose_6d_mean + \
           sum([loss_smpl_params_exp[k] * cfg.LOSS_WEIGHTS[(k + '_EXP').upper()] for k in loss_smpl_params_exp]) + \
           sum([loss_smpl_params_mode[k] * cfg.LOSS_WEIGHTS[(k + '_MODE').upper()] for k in loss_smpl_params_mode])

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
    LOGGER = ConsoleLogger('train_baseline', 'train')
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
    train_dataset = wholedataset(root1, root2, stage='Train')
    val_dataset = wholedataset(root1, root2, stage='Val')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.train_batch_size, shuffle=True, drop_last=True,
                                                   num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.val_batch_size, shuffle=False, drop_last=True,
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
    optimizer = torch.optim.AdamW(params=list(feature_bone.parameters()) + list(diff_bone.parameters()),
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
        diff_bone.train()
        discriminator.train()

        for it, batch in enumerate(train_dataloader, 0):
            batch_size = batch['img'].shape[0]
            x = batch['img'].to(torch.float32).to(device)
            conditioning_feats = feature_bone(x)
            if flag_initial == True:
                smpl_params = {k: v.clone().to(device) for k, v in batch['smpl_params'].items()}
                has_smpl_params = batch['has_smpl_params']['body_pose'] > 0
                smpl_params['body_pose'] = \
                smpl_params['body_pose'].reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(batch_size, -1)
                smpl_params['global_orient'] = \
                smpl_params['global_orient'].reshape(batch_size, -1, 3, 3)[:, :, :, :2].permute(0, 1, 3, 2).reshape(
                    batch_size,  -1)
                smpl_params['betas'] = smpl_params['betas']
                #            conditioning_feats_ini = conditioning_feats[has_smpl_params]
                with torch.no_grad():
                    _, _ = diff_bone.mse_loss(smpl_params, conditioning_feats)
                flag_initial = False
            params_noise = torch.randn(batch_size, diff_bone.npose, device=x.device)
            pred_smpl_params, pred_cam, x_T, pred_pose_6d = diff_bone(params_noise,conditioning_feats)
            pred_cam = pred_cam.reshape(batch_size,3)

            output = {}
            output['pred_cam'] = pred_cam.reshape(batch_size,3)
            output['pred_smpl_params'] = {k: v.clone() for k, v in pred_smpl_params.items()}
            output['noise_params'] = x_T.detach()
            output['conditioning_feats'] = conditioning_feats
            output['pred_pose_6d'] = pred_pose_6d

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
        states['diff_bone_state_dict'] = diff_bone.state_dict()
        states['discriminator_state_dict'] = discriminator.state_dict()
        states['optimizer_state_dict'] = optimizer.state_dict()
        states['optimizer_disc_state_dict'] = optimizer_disc.state_dict()
        torch.save(states, os.path.join(checkpoint_dir, f'checkpoint_{epoch}.tar'))

        feature_bone.eval()
        diff_bone.eval()
        discriminator.eval()
        with torch.no_grad():
            for it, batch in enumerate(val_dataloader, 0):
                batch_size = batch['img'].shape[0]
                x = batch['img'].to(torch.float32).to(device)
                conditioning_feats = feature_bone(x)

                params_noise = torch.randn(batch_size, diff_bone.npose, device=x.device)
                pred_smpl_params, pred_cam, x_T, pred_pose_6d = diff_bone(params_noise, conditioning_feats)
                pred_cam = pred_cam.reshape(batch_size, 3)
                output = {}
                output['pred_cam'] = pred_cam.reshape(batch_size,3)
                output['pred_smpl_params'] = {k: v.clone() for k, v in pred_smpl_params.items()}
                output['noise_params'] = x_T.detach()
                output['conditioning_feats'] = conditioning_feats
                output['pred_pose_6d'] = pred_pose_6d


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
