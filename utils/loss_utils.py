#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import cv2
import numpy as np

def l1_loss(network_output, gt):
    """
    计算L1损失（平均绝对误差）
    :param network_output: 网络输出
    :param gt: 真实值
    :return: L1损失
    """
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    """
    计算L2损失（均方误差）
    :param network_output: 网络输出
    :param gt: 真实值
    :return: L2损失
    """
    return ((network_output - gt) ** 2).mean()

# 新增的深度损失函数
def depth_loss(rendered_depth, metric_depth_path, w=1.0, q=0.0):
    """
    计算渲染深度与Metric3D预测深度之间的L2损失
    :param rendered_depth: torch.Tensor, GS渲染的期望深度 (HxW)
    :param metric_depth_path: str, Metric3D预测深度图路径
    :param w: float, 权重参数
    :param q: float, 偏置参数
    :return: L2损失
    """
    # 读取Metric3D预测深度图并归一化到[0,1]
    metric_depth = cv2.imread(metric_depth_path, cv2.IMREAD_UNCHANGED)
    if metric_depth.ndim == 3:
        metric_depth = cv2.cvtColor(metric_depth, cv2.COLOR_BGR2GRAY)
    metric_depth = metric_depth.astype(np.float32) / 255.0
    metric_depth = torch.from_numpy(metric_depth).to(rendered_depth.device)

    # 尺寸对齐
    if metric_depth.shape != rendered_depth.shape:
        metric_depth = torch.nn.functional.interpolate(metric_depth.unsqueeze(0).unsqueeze(0), size=rendered_depth.shape, mode='bilinear', align_corners=False).squeeze()

    # 计算L2损失
    loss = ((w * rendered_depth + q - metric_depth) ** 2).mean()
    return loss

# 新增的法向损失函数
def pr_normal_loss(rendered_normal, metric_normal_path):
    """
    计算渲染法向与Metric3D预测法向之间的L2损失
    :param rendered_normal: torch.Tensor, 渲染得到的法向 (3xHxW)
    :param metric_normal_path: str, Metric3D预测法向图路径
    :return: 法向损失
    """
    metric_normal = cv2.imread(metric_normal_path, cv2.IMREAD_UNCHANGED)
    if metric_normal.ndim == 2:  # 灰度图，直接扩展为3通道
        metric_normal = np.stack([metric_normal]*3, axis=-1)
    metric_normal = metric_normal.astype(np.float32) / 255.0
    metric_normal = torch.from_numpy(metric_normal).permute(2, 0, 1).to(rendered_normal.device)

    # 尺寸对齐
    if metric_normal.shape != rendered_normal.shape:
        metric_normal = torch.nn.functional.interpolate(metric_normal.unsqueeze(0), size=rendered_normal.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

    # 计算L2损失
    loss = ((rendered_normal - metric_normal) ** 2).mean()
    return loss


def gaussian(window_size, sigma):
    """
    生成一维高斯分布窗口
    :param window_size: 窗口大小
    :param sigma: 标准差
    :return: 高斯分布Tensor
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def smooth_loss(disp, img):
    """
    计算视差图的平滑损失，结合图像梯度进行加权
    :param disp: 视差图
    :param img: 输入图像
    :return: 平滑损失
    """
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    """
    创建二维高斯窗口，用于SSIM计算
    :param window_size: 窗口大小
    :param channel: 通道数
    :return: 高斯窗口Tensor
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算结构相似性（SSIM）损失
    :param img1: 输入图像1
    :param img2: 输入图像2
    :param window_size: SSIM窗口大小
    :param size_average: 是否对所有像素取平均
    :return: SSIM值
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    SSIM的具体实现
    :param img1: 输入图像1
    :param img2: 输入图像2
    :param window: 高斯窗口
    :param window_size: 窗口大小
    :param channel: 通道数
    :param size_average: 是否对所有像素取平均
    :return: SSIM值
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)