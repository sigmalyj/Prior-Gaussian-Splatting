import os
import os.path as osp
import cv2
import time
import sys

# 获取项目根目录，并加入环境变量，方便后续模块导入
CODE_SPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)

import argparse
import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 兼容 mmcv 和 mmengine 的配置导入
try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from datetime import timedelta
import random
import numpy as np
from mono.utils.logger import setup_logger
import glob
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_scalecano_test_with_custom_data
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_from_annos, load_data

def parse_args():
    """
    解析命令行参数，包括配置文件、日志保存路径、模型权重、分布式参数等
    """
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--show-dir', help='the dir to save logs and visualization results')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--launcher', choices=['None', 'pytorch', 'slurm', 'mpi', 'ror'], default='slurm', help='job launcher')
    parser.add_argument('--test_data_path', default='None', type=str, help='the path of test data')
    parser.add_argument('--batch_size', default=1, type=int, help='the batch size for inference')
    args = parser.parse_args()
    return args

def main(args):
    """
    主入口函数，负责配置加载、日志初始化、环境设置、数据加载和分布式启动
    """
    os.chdir(CODE_SPACE)  # 切换到项目根目录
    cfg = Config.fromfile(args.config)  # 加载配置文件
    
    # 合并命令行自定义参数到配置
    if args.options is not None:
        cfg.merge_from_dict(args.options)
       
    # 日志和可视化结果保存路径设置
    if args.show_dir is not None:
        cfg.show_dir = args.show_dir
    else:
        # 默认保存路径：show_dirs/配置文件名/时间戳
        cfg.show_dir = osp.join('./show_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                args.timestamp)
    
    # 检查并设置模型权重路径
    if args.load_from is None:
        raise RuntimeError('Please set model path!')
    cfg.load_from = args.load_from
    cfg.batch_size = args.batch_size
    
    # 加载数据集相关信息
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info
    # 更新模型权重路径（适配多模型场景）
    reset_ckpt_path(cfg.model, data_info)
    
    # 创建日志和可视化结果保存目录
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)
    
    # 初始化日志器
    cfg.log_file = osp.join(cfg.show_dir, f'{args.timestamp}.log')
    logger = setup_logger(cfg.log_file)
    
    # 打印配置信息到日志
    logger.info(f'Config:\n{cfg.pretty_text}')
    
    # 初始化分布式环境（如果需要）
    if args.launcher == 'None':
        cfg.distributed = False
    else:
        cfg.distributed = True
        init_env(args.launcher, cfg)
    logger.info(f'Distributed training: {cfg.distributed}')
    
    # 保存最终配置文件
    cfg.dump(osp.join(cfg.show_dir, osp.basename(args.config)))
    test_data_path = args.test_data_path
    if not os.path.isabs(test_data_path):
        test_data_path = osp.join(CODE_SPACE, test_data_path)

    # 加载测试数据（支持 json 或自定义格式）
    if 'json' in test_data_path:
        test_data = load_from_annos(test_data_path)
    else:
        test_data = load_data(args.test_data_path)
    
    # 根据分布式设置启动主 worker 或多进程 worker
    if not cfg.distributed:
        main_worker(0, cfg, args.launcher, test_data)
    else:
        # 分布式推理
        if args.launcher == 'ror':
            local_rank = cfg.dist_params.local_rank
            main_worker(local_rank, cfg, args.launcher, test_data)
        else:
            mp.spawn(main_worker, nprocs=cfg.dist_params.num_gpus_per_node, args=(cfg, args.launcher, test_data))
        
def main_worker(local_rank: int, cfg: dict, launcher: str, test_data: list):
    """
    单进程或分布式 worker，负责模型构建、权重加载、推理和评估
    """
    if cfg.distributed:
        # 设置分布式参数
        cfg.dist_params.global_rank = cfg.dist_params.node_rank * cfg.dist_params.num_gpus_per_node + local_rank
        cfg.dist_params.local_rank = local_rank

        # 初始化分布式进程组
        if launcher == 'ror':
            init_torch_process_group(use_hvd=False)
        else:
            torch.cuda.set_device(local_rank)
            default_timeout = timedelta(minutes=30)
            dist.init_process_group(
                backend=cfg.dist_params.backend,
                init_method=cfg.dist_params.dist_url,
                world_size=cfg.dist_params.world_size,
                rank=cfg.dist_params.global_rank,
                timeout=default_timeout)
    
    logger = setup_logger(cfg.log_file)
    # 构建模型
    model = get_configured_monodepth_model(cfg, )
    
    # 分布式/单机模型包装
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[local_rank],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model).cuda()
        
    # 加载模型权重
    model, _,  _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    model.eval()
    
    # 执行推理和评估
    do_scalecano_test_with_custom_data(
        model, 
        cfg,
        test_data,
        logger,
        cfg.distributed,
        local_rank,
        cfg.batch_size,
    )
    
if __name__ == '__main__':
    # 解析命令行参数并生成时间戳
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp
    main(args)