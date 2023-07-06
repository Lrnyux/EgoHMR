import os
from torch.utils.data import Dataset
import numpy as np
import cv2
import scipy.io
from PIL import Image
import torchvision





class wholedataset(Dataset):
    def __init__(self, root1, root2, stage):
        self.root1 = root1
        self.root2 = root2
        self.stage = stage
        self.filelist_smpl = []

        if self.stage == 'Train':
            self.env_list = []

        if self.stage == 'Val':
            self.env_list = []

        for idx in range(len(self.env_list)):
            env_dir_root2 = os.path.join(self.root2, self.env_list[idx])
            env_dir_listdir2 = sorted(os.listdir(env_dir_root2))
            for jdx in range(len(env_dir_listdir2)):
                smpl_mat_root = os.path.join(env_dir_root2, env_dir_listdir2[jdx])
                self.filelist_smpl.append(smpl_mat_root)
        self.length = len(self.filelist_smpl)
        self.has_body_pose = np.ones(self.length,dtype = np.float32)
        self.has_betas = np.ones(self.length,dtype = np.float32)

    def resize_image(self, image, resize_height=None, resize_width=None):
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
        env_name = str(smpl_data['env'][0])
        frame_ids = int(smpl_data['frame_ids'][0])

        env_img_root = os.path.join(self.root1, env_name)
        env_img_listdir = sorted(os.listdir(env_img_root))
        data_root = os.path.join(env_img_root, env_img_listdir[frame_ids])
        data = scipy.io.loadmat(data_root)

        ego_root = str(data['ego_mat'][0])
        # -----------------------add augmentation-------------------------------------
        ego_mat_ori = Image.open(ego_root)
        if self.stage == 'Train':
            color_aug = torchvision.transforms.ColorJitter(brightness=np.random.rand(), contrast=5 * np.random.rand(),
                                                           hue=np.random.rand() / 2, saturation=np.random.rand())
            ego_mat_ori = color_aug(ego_mat_ori)
        ego_mat_ori = np.array(ego_mat_ori)
        image_size = np.asarray([224, 224])
        ego_mat_ori = self.resize_image(ego_mat_ori, resize_width=image_size[0], resize_height=image_size[1])
        ego_mat_ori_norm = (ego_mat_ori - np.min(ego_mat_ori)) / (np.max(ego_mat_ori) - np.min(ego_mat_ori) + 1e-9)
        image = np.transpose(ego_mat_ori_norm, [2, 0, 1])

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
        item['img_size'] = 1.0 * 224
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['imgname'] = frame_ids
        item['imgenv'] = env_name
        item['imgroot'] = env_img_listdir[frame_ids]
        item['idx'] = index
        return item

    def __len__(self):
        return len(self.filelist_smpl)

class testdataset(Dataset):
    def __init__(self, root1, root2, stage):
        self.root1 = root1
        self.root2 = root2
        self.stage = stage
        self.filelist_mat = []
        if self.stage == 'Test':
            self.env_list = []
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

        mat_data_root = self.filelist_mat[index]
        pose_data = scipy.io.loadmat(mat_data_root)
        betas = np.zeros(10)*1.0
        pose = np.ones((24,3,3))*1.0
        keypoints = pose_data['p3d_1_gt']
        has_body_pose = self.has_body_pose[index]
        has_betas = self.has_betas[index]


        ego_root = str(pose_data['ego_mat'][0])

        # -----------------------add augmentation-------------------------------------
        ego_mat_ori = Image.open(ego_root)
        if self.stage == 'Train':
            color_aug = torchvision.transforms.ColorJitter(brightness=np.random.rand(), contrast=5 * np.random.rand(),
                                                           hue=np.random.rand() / 2, saturation=np.random.rand())
            ego_mat_ori = color_aug(ego_mat_ori)
        ego_mat_ori = np.array(ego_mat_ori)
        image_size = np.asarray([224, 224])
        ego_mat_ori = self.resize_image(ego_mat_ori, resize_width=image_size[0], resize_height=image_size[1])
        ego_mat_ori_norm = (ego_mat_ori - np.min(ego_mat_ori)) / (np.max(ego_mat_ori) - np.min(ego_mat_ori) + 1e-9)
        image = np.transpose(ego_mat_ori_norm, [2, 0, 1])

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
        item['img_size'] = 1.0 * 224
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['keypoints_3d'] = keypoints
        item['imgroot'] = str(ego_root)
        item['idx'] = index
        return item

    def __len__(self):
        return len(self.filelist_mat)