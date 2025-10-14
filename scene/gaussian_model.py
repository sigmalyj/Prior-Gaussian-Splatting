# 导入必要的库和工具函数
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

# 定义 GaussianModel 类，用于管理高斯点云模型
class GaussianModel:

    # 初始化一些激活函数和协方差构建函数
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            # 根据缩放和旋转构建协方差矩阵
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        # 定义激活函数
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    # 初始化模型
    def __init__(self, sh_degree : int):
        # 初始化模型的属性
        self.active_sh_degree = 0  # 当前 SH（球谐函数）阶数
        self.max_sh_degree = sh_degree  # 最大 SH 阶数
        self._xyz = torch.empty(0)  # 点的坐标
        self._features_dc = torch.empty(0)  # 特征的直流分量
        self._features_rest = torch.empty(0)  # 特征的其他分量
        self._scaling = torch.empty(0)  # 缩放参数
        self._rotation = torch.empty(0)  # 旋转参数
        self._opacity = torch.empty(0)  # 不透明度
        self.max_radii2D = torch.empty(0)  # 最大 2D 半径
        self.xyz_gradient_accum = torch.empty(0)  # 坐标梯度累积
        self.denom = torch.empty(0)  # 梯度归一化因子
        self.optimizer = None  # 优化器
        self.percent_dense = 0  # 稠密化百分比
        self.spatial_lr_scale = 0  # 空间学习率缩放因子
        self.setup_functions()  # 设置激活函数

    # 捕获模型的当前状态
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    # 恢复模型状态
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    # 定义一些属性访问器
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)  # 返回激活后的缩放参数
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)  # 返回归一化后的旋转参数
    
    @property
    def get_xyz(self):
        return self._xyz  # 返回点的坐标
    
    @property
    def get_features(self):
        # 返回特征的组合（直流分量和其他分量）
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)  # 返回激活后的不透明度
    
    def get_covariance(self, scaling_modifier=1):
        # 根据当前点的坐标、缩放参数、旋转参数和缩放修正因子，计算协方差矩阵
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)
    
    def oneupSHdegree(self):
        # 将当前的球谐函数阶数增加 1，但不能超过最大阶数
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        # 从点云数据创建模型
        self.spatial_lr_scale = spatial_lr_scale  # 设置空间学习率缩放因子
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()  # 加载点云坐标
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())  # 将颜色转换为球谐函数表示
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()  # 初始化特征
        features[:, :3, 0] = fused_color  # 设置直流分量
        features[:, 3:, 1:] = 0.0  # 其他分量初始化为 0
    
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])  # 打印点的数量
    
        # 计算点之间的距离并初始化缩放参数
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")  # 随机初始化旋转参数
    
        # 初始化不透明度
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    
        # 将点云数据和参数设置为可优化的张量
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # 初始化最大 2D 半径
    
    def training_setup(self, training_args):
        # 设置训练参数
        self.percent_dense = training_args.percent_dense  # 稠密化百分比
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # 坐标梯度累积
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # 梯度归一化因子
    
        # 定义优化器参数组
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
    
        # 初始化 Adam 优化器
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
        # 设置学习率调度器
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )
    
    def update_learning_rate(self, iteration):
        ''' 每步更新学习率 '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)  # 根据调度器计算学习率
                param_group['lr'] = lr  # 更新学习率
                return lr
    
    def construct_list_of_attributes(self):
        # 构建属性列表，用于保存点云数据
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']  # 坐标和法向量
        # 添加直流分量的特征
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        # 添加其他分量的特征
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')  # 不透明度
        # 添加缩放参数
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        # 添加旋转参数
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    
    def save_ply(self, path):
        # 保存点云数据到 PLY 文件
        mkdir_p(os.path.dirname(path))  # 创建保存路径的目录

        # 提取点云相关数据
        xyz = self._xyz.detach().cpu().numpy()  # 点的坐标
        normals = np.zeros_like(xyz)  # 法向量初始化为零
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()  # 特征直流分量
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()  # 特征其他分量
        opacities = self._opacity.detach().cpu().numpy()  # 不透明度
        scale = self._scaling.detach().cpu().numpy()  # 缩放参数
        rotation = self._rotation.detach().cpu().numpy()  # 旋转参数

        # 构建 PLY 文件的属性描述
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        # 合并所有属性
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))

        # 保存为 PLY 文件
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        # 重置不透明度，将其限制在最大值 0.01
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        # 从 PLY 文件加载点云数据
        plydata = PlyData.read(path)

        # 提取点云坐标
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # 提取特征直流分量
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # 提取特征其他分量
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # 提取缩放参数
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 提取旋转参数
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 将加载的数据设置为模型的可优化参数
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree  # 设置当前 SH 阶数为最大值

    def replace_tensor_to_optimizer(self, tensor, name):
        # 替换优化器中的张量
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        # 根据掩码裁剪优化器中的张量
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        # 根据掩码裁剪点云数据
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 更新模型的可优化参数
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 更新梯度累积和归一化因子
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        # 将新的张量拼接到优化器中现有的张量
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1  # 确保每个参数组只有一个参数
            extension_tensor = tensors_dict[group["name"]]  # 获取需要拼接的张量
            stored_state = self.optimizer.state.get(group['params'][0], None)  # 获取优化器的状态
            if stored_state is not None:
                # 拼接优化器状态中的动量和二阶动量
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
    
                # 替换优化器中的参数
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
    
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果没有状态，直接拼接参数
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    
        return optimizable_tensors
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        # 后处理：将新生成的点云数据添加到模型中
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation
        }
    
        # 将新数据拼接到优化器中
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
    
        # 重置梯度累积和归一化因子
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # 根据梯度和场景范围对点云进行加密和分裂
        n_init_points = self.get_xyz.shape[0]  # 初始点数量
        padded_grad = torch.zeros((n_init_points), device="cuda")  # 填充梯度
        padded_grad[:grads.shape[0]] = grads.squeeze()  # 将梯度复制到填充张量中
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)  # 筛选满足梯度条件的点
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
    
        # 生成新点的标准差和均值
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)  # 按正态分布采样
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)  # 旋转矩阵
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)  # 新点坐标
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))  # 新缩放参数
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)  # 新旋转参数
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)  # 新直流分量
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)  # 新其他分量
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)  # 新不透明度
    
        # 添加新点到模型中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
    
        # 根据掩码裁剪点
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # 根据梯度和场景范围对点云进行加密和克隆
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)  # 筛选满足梯度条件的点
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )
    
        # 克隆选中的点
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
    
        # 添加克隆点到模型中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # 对点云进行加密和裁剪
        grads = self.xyz_gradient_accum / self.denom  # 计算梯度
        grads[grads.isnan()] = 0.0  # 将 NaN 值替换为 0
    
        # 加密点云
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
    
        # 根据不透明度和屏幕大小裁剪点
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
    
        torch.cuda.empty_cache()  # 清理 CUDA 缓存
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 更新加密统计信息
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1  # 更新归一化因子