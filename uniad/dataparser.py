import os
import cv2
import mmcv
import torch
import numpy as np
from scipy.spatial.transform import Rotation as SCR
import math

# Coordinate definitions:
# World/Ego frame follows NuScenes/UniAD (x-forward, y-left, z-up).
# LiDAR x-right, y-forward, z-up.
LIDAR_OFFSET = np.array([0.5, 0.0, 1.50], dtype=np.float32)  # meters (front, left, up)
lidar2ego = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def get_intrinsic(intrin_dict):
    fovx, fovy = intrin_dict['fovx'], intrin_dict['fovy']
    h, w = intrin_dict['H'], intrin_dict['W']
    K = np.eye(4)
    K[0, 0], K[1, 1] = fov2focal(fovx, w), fov2focal(fovy, h)
    K[0, 2], K[1, 2] = intrin_dict['cx'], intrin_dict['cy']
    return K

def parse_uniad_obs(obs, info, cameras, img_norm_cfg):
    ego_rot = np.asarray(info['ego_rot'], dtype=np.float32)
    ego_pos = np.asarray(info['ego_pos'], dtype=np.float32)
    R_world_ego = SCR.from_euler('XYZ', ego_rot).as_matrix().astype(np.float32)

    imgs_shape = []
    imgs = []
    raw_imgs = []
    l2imgs, intrinsics = [], []

    for cam in cameras:
        im = obs[cam]
        
        if len(im.shape) == 4:
            im = im[-1]
        
        im = cv2.resize(im, (1600, 900))
        im_norm = mmcv.imnormalize(im, img_norm_cfg['mean'], img_norm_cfg['std'], img_norm_cfg['to_rgb'])
        imgs_shape.append(list(im_norm.shape))
        im_norm = torch.from_numpy(im_norm).permute(2, 0, 1)
        imgs.append(im_norm)
        raw_imgs.append(im)
        l2c = info['cam_params'][cam]['l2c']
        K = get_intrinsic(info['cam_params'][cam]['intrinsic'])
        # training gs downsample 2x, recover scale
        rescale_K = np.copy(K)
        rescale_K[0, 0] *= 2
        rescale_K[0, 2] *= 2
        rescale_K[1, 1] *= 2
        rescale_K[1, 2] *= 2
        l2imgs.append(rescale_K @ l2c)
        intrinsics.append(K) 

    imgs = torch.stack(imgs)[None, ...]
    imgs = imgs.cuda()
    timestamp = info['timestamp']
    
    f2w = np.eye(4)
    f2w[:3, :3] = R_world_ego
    f2w[:3, 3] = ego_pos

    # LiDAR pose in world coordinates (LiDAR shares ego axes, only translated)
    l2g_r_mat = R_world_ego @ lidar2ego
    l2g_t = ego_pos + R_world_ego @ LIDAR_OFFSET

    can_bus = np.zeros(18)
    can_bus[:3] = ego_pos
    v_quat = SCR.from_euler('XYZ', ego_rot).as_quat()[[3, 0, 1, 2]]
    can_bus[3:7] = v_quat

    linear_vel = info['linear_velocity']
    can_bus[7:10] = R_world_ego.T @ linear_vel

    linear_acc = info['linear_acceleration']
    can_bus[10:13] = R_world_ego.T @ linear_acc

    angular_vel = info['angular_velocity']
    can_bus[13:16] = angular_vel
    
    yaw = ego_rot[2]
    if yaw < 0:
        yaw += np.pi * 2
    can_bus[-2] = yaw
    can_bus[-1] = yaw / np.pi * 180

    img_metas = {
        'scene_token': info['scene_token'],
        'can_bus': can_bus,
        'img_shape': imgs_shape,
        'lidar2img': l2imgs,
        'intrinsics': intrinsics,
    }
    data = {
        'img': imgs, 
        'img_metas': [[img_metas]],
        'l2g_t': torch.tensor(l2g_t).float(),
        'l2g_r_mat': torch.tensor(l2g_r_mat).float(),
        'timestamp': timestamp,
        'command': [torch.tensor([info['command']]).cuda()], # 0 right, 1 left, 2 straight
        'raw_imgs': raw_imgs,
    }

    return data
