# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:00:36 2022

@author: liuyuxuan
"""

# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur F鰎derung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright?019 Max-Planck-Gesellschaft zur F鰎derung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import numpy as np
import torch.nn as nn

# from smplx import SMPL as _SMPL
from smplx import SMPL
import trimesh
import math

SMPL_MEAN_PARAMS = '/home/work/Yuxuan/Data/SMPL_body_model/smpl_mean_params.npz'
SMPL_MODEL_DIR = '/home/work/Yuxuan/Data/SMPL_body_model/smpl'

#
# class SMPL(_SMPL):
#     """ Extension of the official SMPL implementation to support more joints """
#
#     def __init__(self, *args, **kwargs):
#         super(SMPL, self).__init__(*args, **kwargs)
#         joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
#         J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
#         self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
#         self.joint_map = torch.tensor(joints, dtype=torch.long)
#
#     def forward(self, *args, **kwargs):
#         kwargs['get_skin'] = True
#         smpl_output = super(SMPL, self).forward(*args, **kwargs)
#         extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
#         joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
#         joints = joints[:, self.joint_map, :]
#         output = SMPLOutput(vertices=smpl_output.vertices,
#                             global_orient=smpl_output.global_orient,
#                             body_pose=smpl_output.body_pose,
#                             joints=joints,
#                             betas=smpl_output.betas,
#                             full_pose=smpl_output.full_pose)
#         return output


class SMPLHead(nn.Module):
    def __init__(self, focal_length=5000., img_res=224):
        super(SMPLHead, self).__init__()
        self.smpl = SMPL(SMPL_MODEL_DIR, create_transl=False)
        self.add_module('smpl', self.smpl)
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(self, global_orient, body_pose, betas, cam=None, normalize_joints2d=False):
        '''
        :param rotmat: rotation in euler angles format (N,J,3,3)
        :param shape: smpl betas
        :param cam: weak perspective camera
        :param normalize_joints2d: bool, normalize joints between -1, 1 if true
        :return: dict with keys 'vertices', 'joints3d', 'joints2d' if cam is True
        '''
        smpl_output = self.smpl(
            betas=betas,
            body_pose=body_pose.contiguous(),
            global_orient=global_orient.contiguous(),
            pose2rot=False,
        )
        # smpl_output = self.smpl(
        #     betas=shape,
        #     body_pose=rotmat[:, 1:].contiguous(),
        #     pose2rot=False,
        # )

        output = {
            'smpl_betas' : betas,
            'smpl_thetas' : torch.cat([global_orient,body_pose], dim=1),
            'smpl_vertices': smpl_output.vertices,
            'smpl_joints3d': smpl_output.joints,
        }

        return smpl_output,output

def mesh2obj(verts, filename):

    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    smpl_face = smpl.faces
    mesh = trimesh.Trimesh(vertices=verts, faces=smpl_face, process=False)
    Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
    mesh.apply_transform(Rx)
    mesh.export(filename)
    print('finishing writing mesh to {}'.format(filename))



if __name__ == "__main__":
    import scipy.io
    import torch
    import numpy as np
    mat_root = '/opt/data/private/guogroup/liuyuxuan/data/Ego_data_final/shape_info/env1/01500.mat'
    data = scipy.io.loadmat(mat_root)
    betas = data['betas'].reshape(1,10)
    thetas = data['pose'].reshape(1,24,3,3)
    print(betas.shape)
    print(thetas.shape)
    smpl_model_temp = SMPLHead()
    output = smpl_model_temp(torch.Tensor(thetas),torch.Tensor(betas))
    vertices = output['smpl_vertices'].view(-1,3).numpy()
    print(vertices.shape)

    test_obj_root = '/opt/data/private/guogroup/liuyuxuan/software/EgoShapo/test_root/test.obj'
    mesh2obj(vertices,test_obj_root)

