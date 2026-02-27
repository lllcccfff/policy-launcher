import os
import cv2
import mmcv
import torch
import numpy as np
from scipy.spatial.transform import Rotation as SCR
import math

# Coordinate definitions:
# World/Ego frame follows NuScenes/UniAD (x-forward, y-left, z-up).
# LiDAR shares the same axes but is rigidly translated from ego origin.
LIDAR_OFFSET = np.array([0.5, 0.0, 1.50], dtype=np.float32)  # meters (front, left, up)
lidar2ego = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
LIDAR_TO_EGO_4X4 = np.array(
    [
        [1, 0, 0, 0.5],
        [0, 1, 0, 0.0],
        [0, 0, 1, 1.5],
        [0, 0, 0, 1.0],
    ],
    dtype=np.float32,
)

def _to_k4(cam_info):
    k = np.asarray(cam_info["K"], dtype=np.float32)
    if k.shape == (4, 4):
        return k
    if k.shape == (3, 3):
        k4 = np.eye(4, dtype=np.float32)
        k4[:3, :3] = k
        return k4
    raise ValueError(f"`K` must be 3x3 or 4x4, got {k.shape}")

def _to_l2c(cam_info):
    if "l2c" in cam_info:
        l2c = np.asarray(cam_info["l2c"], dtype=np.float32)
        if l2c.shape != (4, 4):
            raise ValueError(f"`l2c` must be 4x4, got {l2c.shape}")
        return l2c
    ego2camera = np.asarray(cam_info["ego2camera"], dtype=np.float32)
    if ego2camera.shape != (4, 4):
        raise ValueError(f"`ego2camera` must be 4x4, got {ego2camera.shape}")
    return ego2camera @ LIDAR_TO_EGO_4X4

def parse_raw(obs, info, cameras, img_norm_cfg):
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
        
        # Attention: If the image size is not 900X1600, the intrinsic params needs to be rescaled
        im = cv2.resize(im, (1600, 900))
        im_norm = mmcv.imnormalize(im, img_norm_cfg['mean'], img_norm_cfg['std'], img_norm_cfg['to_rgb'])
        imgs_shape.append(list(im_norm.shape))
        im_norm = torch.from_numpy(im_norm).permute(2, 0, 1)
        imgs.append(im_norm)
        raw_imgs.append(im)
        cam_info = info['cam_params'][cam]
        l2c = _to_l2c(cam_info)
        K = _to_k4(cam_info)
        l2imgs.append(K @ l2c)
        intrinsics.append(K) 

    imgs = torch.stack(imgs)[None, ...]
    imgs = imgs.cuda()
    # Keep legacy behavior: `timestamp` is relative timestamp.
    timestamp = float(info["relative_timestamp"])
    
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

    # UniAD 期望的 CAN 布局: [pos(3), quat(4), vel(3), acc(3), ang_vel(3), pad(2)]
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


def parse_uniad_obs(obs, info, cameras, img_norm_cfg):
    return parse_raw(obs, info, cameras, img_norm_cfg)
