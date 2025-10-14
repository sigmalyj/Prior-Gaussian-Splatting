# 依赖库声明
dependencies = ['torch', 'torchvision']

import os
import torch
# 兼容 mmcv 和 mmengine 的配置导入
try:
  from mmcv.utils import Config, DictAction
except:
  from mmengine import Config, DictAction

# 导入模型构建函数
from mono.model.monodepth_model import get_configured_monodepth_model
metric3d_dir = '/home/liyanjia/door/projects/Prior-Gaussian-Splatting/preprocess'

# 支持的模型类型及其配置和权重链接
MODEL_TYPE = {
  'ConvNeXt-Tiny': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convtiny.0.3_150.py',
    'ckpt_file': 'https://hf-miror.com/JUGGHM/Metric3D/resolve/main/convtiny_hourglass_v1.pth',
  },
  'ConvNeXt-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/convlarge.0.3_150.py',
    'ckpt_file': 'https://hf-miror.com/JUGGHM/Metric3D/resolve/main/convlarge_hourglass_0.3_150_step750k_v1.1.pth',
  },
  'ViT-Small': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.small.py',
    'ckpt_file': f'{metric3d_dir}/weight/metric_depth_vit_small_800k.pth',  # 本地权重路径
  },
  'ViT-Large': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.large.py',
    'ckpt_file': 'https://hf-miror.com/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
  },
  'ViT-giant2': {
    'cfg_file': f'{metric3d_dir}/mono/configs/HourglassDecoder/vit.raft5.giant2.py',
    'ckpt_file': 'https://hf-miror.com/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth',
  },
}

def metric3d_vit_small(pretrain=False, **kwargs):
  '''
  返回 ViT-Small 骨干和 RAFT-4iter 头的 Metric3D 模型
  pretrain: 是否加载预训练权重
  '''
  cfg_file = MODEL_TYPE['ViT-Small']['cfg_file']
  ckpt_file = MODEL_TYPE['ViT-Small']['ckpt_file']

  cfg = Config.fromfile(cfg_file)
  model = get_configured_monodepth_model(cfg)
  if pretrain:
    # 使用 torch.load 加载本地权重
    state_dict = torch.load(ckpt_file, map_location='cpu')
    # 兼容权重字典结构
    if 'model_state_dict' in state_dict:
      model.load_state_dict(state_dict['model_state_dict'], strict=False)
    else:
      model.load_state_dict(state_dict, strict=False)
  return model

if __name__ == '__main__':
  import cv2
  import numpy as np
  from glob import glob

  # 创建输出目录
  os.makedirs('preprocessed_data/normal', exist_ok=True)
  os.makedirs('preprocessed_data/depth', exist_ok=True)

  # 加载模型
  model = metric3d_vit_small(pretrain=True)
  model.cuda().eval()

  rgb_files = glob('data/*.png')
  print(f"共找到 {len(rgb_files)} 张RGB图像")

  for rgb_file in rgb_files:
    print(f"\n处理: {rgb_file}")
    rgb_origin = cv2.imread(rgb_file)[:, :, ::-1] # 读取并转为RGB

    # 生成灰度图像并保存到同一目录
    depth_file = os.path.join(os.path.dirname(rgb_file), os.path.basename(rgb_file).replace('.png', '_gray.png'))
    gray_image = cv2.cvtColor(rgb_origin, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(depth_file, gray_image)

    # 调整输入尺寸以适配预训练模型
    intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
    gt_depth_scale = 256.0
    input_size = (616, 1064)
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    # 归一化处理
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()

    # 推理
    with torch.no_grad():
      pred_depth, confidence, output_dict = model.inference({'input': rgb})

    # 去除填充并上采样到原始尺寸
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()

    # 规范相机空间变换，获得真实尺度
    canonical_to_real_scale = intrinsic[0] / 1000.0
    pred_depth = pred_depth * canonical_to_real_scale
    pred_depth = torch.clamp(pred_depth, 0, 300)

    # 保存深度图
    depth_vis = pred_depth.cpu().numpy()
    depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    depth_out_path = os.path.join('preprocessed_data/depth', os.path.basename(rgb_file).replace('.png', '_depth.png'))
    cv2.imwrite(depth_out_path, depth_color)

    # 保存法线图
    if 'prediction_normal' in output_dict:
      pred_normal = output_dict['prediction_normal'][:, :3, :, :]
      pred_normal = pred_normal.squeeze()
      pred_normal = pred_normal[:, pad_info[0] : pred_normal.shape[1] - pad_info[1], pad_info[2] : pred_normal.shape[2] - pad_info[3]]
      pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
      pred_normal_vis = (pred_normal_vis + 1) / 2
      normal_out_path = os.path.join('preprocessed_data/normal', os.path.basename(rgb_file).replace('.png', '_normal.png'))
      cv2.imwrite(normal_out_path, (pred_normal_vis * 255).astype(np.uint8))
    
    # 删除生成的灰度图像
    if os.path.exists(depth_file):
      os.remove(depth_file)