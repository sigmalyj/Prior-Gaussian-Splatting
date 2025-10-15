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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, depth_loss, normal_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    """
    2D高斯溅射训练主函数
    
    Args:
        dataset: 数据集参数
        opt: 优化参数
        pipe: 渲染管线参数
        testing_iterations: 测试迭代轮数列表
        saving_iterations: 保存模型迭代轮数列表
        checkpoint_iterations: 检查点保存迭代轮数列表
        checkpoint: 预训练模型路径
    """
    first_iter = 0
    # 准备输出目录和日志记录器
    tb_writer = prepare_output_and_logger(dataset)
    
    # 初始化高斯模型和场景
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # 如果有检查点，加载预训练模型
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置背景颜色：白色背景或黑色背景
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # CUDA事件，用于计时
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    # 指数移动平均损失，用于平滑显示
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    # 创建进度条
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 主训练循环
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        # 更新学习率（根据迭代次数调整）
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加球谐函数的阶数，直到最大阶数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个训练视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # 渲染当前视角
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # 计算基础损失：L1损失 + SSIM损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # 正则化项：法向量损失和深度损失
        # 法向量损失在7000次迭代后开始，深度损失在3000次迭代后开始
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_depth = opt.lambda_dist if iteration > 3000 else 0.0

        rend_depth = render_pkg["rend_dist"].squeeze()  # 渲染深度
        rend_normal  = render_pkg['rend_normal'] # 渲染法向量
        surf_normal = render_pkg['surf_normal'] # 渲染表面法向量

        # 计算深度损失
        # depth_img_name = viewpoint_cam.image_name + '_depth.png'
        # metric_depth_path = os.path.join('preprocess/preprocessed_data/depth', depth_img_name)
        # loss_depth = depth_loss(rendered_depth, metric_depth_path)
        dist_loss = lambda_depth * (rend_depth).mean()

        # 计算法向量损失
        # normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        # normal_loss = lambda_normal * (normal_error).mean()
        normal_img_name = viewpoint_cam.image_name.replace('.png', '_normal.png')
        metric_normal_path = os.path.join('preprocess/preprocessed_data/normal', normal_img_name)
        loss_normal = normal_loss(rend_normal, metric_normal_path)


        # 总损失
        total_loss = loss + dist_loss + loss_normal
        # total_loss = loss + dist_loss + normal_loss
        # total_loss = loss + lambda_depth * loss_depth + normal_loss

        # 反向传播
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # 更新用于显示的指数移动平均损失
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            # ema_dist_for_log = 0.4 * loss_depth.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            # 每10次迭代更新进度条
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",      # 主要损失
                    "distort": f"{ema_dist_for_log:.{5}f}",   # 距离损失
                    "normal": f"{ema_normal_for_log:.{5}f}",  # 法向量损失
                    "Points": f"{len(gaussians.get_xyz)}"     # 高斯点数量
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
                
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录到TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            # 训练报告（包含测试和验证）
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            # 保存模型
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 密度化处理（自适应添加和删除高斯点）
            if iteration < opt.densify_until_iter:
                # 更新每个高斯点的最大2D半径
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 累积密度化统计信息
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 执行密度化：根据梯度阈值添加新点，根据透明度删除不必要的点
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                # 定期重置透明度，防止过度优化
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步骤
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 保存检查点
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        # 网络GUI处理（用于实时可视化）
        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        # 渲染自定义视角用于GUI显示
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                    # 准备发送给GUI的指标
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],  # 高斯点数量
                        "loss": ema_loss_for_log              # 当前损失
                    }
                    
                    # 发送数据到GUI
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

def prepare_output_and_logger(args):
    """
    准备输出目录和日志记录器
    """
    if not args.model_path:
        # 如果没有指定模型路径，生成唯一标识符
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 创建输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    
    # 保存配置参数
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建TensorBoard写入器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    """
    训练报告函数：记录损失和进行测试评估
    """
    # 记录训练损失到TensorBoard
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # 在指定迭代次数进行测试评估
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        
        # 配置测试和训练样本验证
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()}, 
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                # 遍历每个测试视角
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    # 记录前5个视角的可视化结果到TensorBoard
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        
                        # 深度图可视化
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            # 法向量和透明度可视化
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5  # 归一化到[0,1]
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            # 距离图可视化
                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        # 第一次测试时记录真实图像
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    # 计算L1损失和PSNR
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                # 计算平均指标
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
                # 记录测试指标到TensorBoard
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)      # 模型参数
    op = OptimizationParams(parser)  # 优化参数
    pp = PipelineParams(parser)   # 管线参数
    
    # GUI相关参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    
    # 训练相关参数
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)  # 在最后一次迭代时也保存模型
    
    print("Optimizing " + args.model_path)

    # 初始化系统状态（随机数生成器）
    safe_state(args.quiet)

    # 启动GUI服务器，配置并运行训练
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # 异常检测
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # 训练完成
    print("\nTraining complete.")