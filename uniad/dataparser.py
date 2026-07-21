import cv2
import mmcv
import torch
import numpy as np
from scipy.spatial.transform import Rotation as SCR

LIDAR_TO_EGO_4X4 = np.array(
    [
        [0.0020332718, 0.99970406, 0.024241721, 0.943713],
        [-0.9999805, 0.002175657, -0.005848639, 0.0],
        [-0.00589965, -0.024229359, 0.99968904, 1.84023],
        [0, 0, 0, 1.0],
    ],
    dtype=np.float32,
)
CAMERA_ORDER = ("FRONT", "FRONT_RIGHT", "FRONT_LEFT", "BACK", "BACK_LEFT", "BACK_RIGHT")
UNIAD_IMG_NORM_CFG = {
    "mean": np.array([103.530, 116.280, 123.675], dtype=np.float32),
    "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "to_rgb": False,
}

def build_can_bus(info, r_world_ego):
    ego_rot = info["ego_rot"]
    ego_pos = info["ego_pos"]

    can_bus = np.zeros(18, dtype=np.float64)
    can_bus[:3] = ego_pos
    can_bus[3:7] = SCR.from_euler("XYZ", ego_rot).as_quat()[[3, 0, 1, 2]]
    can_bus[7:10] = r_world_ego.T @ info["linear_acceleration"]
    can_bus[10:13] = info["angular_velocity"]
    can_bus[13:16] = r_world_ego.T @ info["linear_velocity"]

    yaw = float(ego_rot[2])
    if yaw < 0:
        yaw += np.pi * 2
    can_bus[-2] = yaw
    can_bus[-1] = yaw / np.pi * 180
    return can_bus


def build_image_input(obs, info, camera_order, img_norm_cfg, image_scale):
    r_world_ego = SCR.from_euler("XYZ", info["ego_rot"]).as_matrix().astype(np.float32)
    ego_to_global = np.eye(4, dtype=np.float32)
    ego_to_global[:3, :3] = r_world_ego
    ego_to_global[:3, 3] = info["ego_pos"]
    lidar_to_global = ego_to_global @ LIDAR_TO_EGO_4X4
    lidar2img = []
    img_shapes = []
    tensors = []

    for cam in camera_order:
        image = obs[cam]
        image = np.asarray(image, dtype=np.float32)
        if image_scale != 1.0:
            height, width = image.shape[:2]
            image = cv2.resize(image, (int(round(width * image_scale)), int(round(height * image_scale))))

        image = mmcv.imnormalize(
            image,
            img_norm_cfg["mean"],
            img_norm_cfg["std"],
            img_norm_cfg["to_rgb"],
        )
        img_shapes.append(list(image.shape))
        tensors.append(torch.from_numpy(image).permute(2, 0, 1))

        cam_info = info["cam_params"][cam]
        k = cam_info["K"]
        ego2camera = cam_info["ego2camera"]
        k4 = np.eye(4, dtype=np.float32)
        k4[:3, :3] = k
        k4[0, :] *= image_scale
        k4[1, :] *= image_scale
        lidar2img.append(k4 @ ego2camera @ LIDAR_TO_EGO_4X4)

    image_tensor = torch.stack(tensors, dim=0).unsqueeze(0).float().cuda()
    img_metas = {
        "scene_token": info["scene_token"],
        "can_bus": build_can_bus(info, r_world_ego),
        "img_shape": img_shapes,
        "lidar2img": lidar2img,
        "l2g_r_mat": lidar_to_global[:3, :3].T,
        "l2g_t": lidar_to_global[:3, 3],
    }
    return image_tensor, img_metas


def parse_uniad_obs(obs, info):
    imgs, img_metas = build_image_input(
        obs,
        info,
        CAMERA_ORDER,
        UNIAD_IMG_NORM_CFG,
        image_scale=1.0,
    )
    timestamp = float(info["relative_timestamp"])
    l2g_r_mat = img_metas["l2g_r_mat"]
    l2g_t = img_metas["l2g_t"]
    data = {
        'img': [imgs],
        'img_metas': [[img_metas]],
        'l2g_t': torch.tensor(l2g_t, dtype=torch.float32, device=imgs.device)[None, :],
        'l2g_r_mat': torch.tensor(l2g_r_mat, dtype=torch.float32, device=imgs.device)[None, :, :],
        'timestamp': [torch.tensor([timestamp], dtype=torch.float64, device=imgs.device)],
        'command': [torch.tensor([info['command']], dtype=torch.long, device=imgs.device)], # 0 right, 1 left, 2 straight
    }

    return data
