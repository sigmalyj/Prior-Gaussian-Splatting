import os
from argparse import ArgumentParser

# DTU数据集的标准测试场景列表
# 这些场景是DTU数据集中常用于评估3D重建算法的标准测试集
dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 
              'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 
              'scan114', 'scan118', 'scan122']

# 命令行参数解析器设置
parser = ArgumentParser(description="Full evaluation script parameters")

# 控制评估流程的开关参数
parser.add_argument("--skip_training", action="store_true", 
                   help="跳过训练阶段，直接使用已有模型")
parser.add_argument("--skip_rendering", action="store_true", 
                   help="跳过渲染阶段，使用已有渲染结果")
parser.add_argument("--skip_metrics", action="store_true", 
                   help="跳过指标计算阶段")

# 路径参数
parser.add_argument("--output_path", default="./eval/dtu", 
                   help="评估输出路径，存储训练模型和渲染结果")
parser.add_argument('--dtu', "-dtu", required=True, type=str,
                   help="DTU数据集根目录路径")

# 解析已知参数，允许未知参数传递给后续脚本
args, _ = parser.parse_known_args()

# 初始化场景列表
all_scenes = []
all_scenes.extend(dtu_scenes)

# 如果需要计算指标，添加DTU官方评估工具路径参数
if not args.skip_metrics:
    parser.add_argument('--DTU_Official', "-DTU", required=True, type=str,
                       help="DTU官方评估工具路径")
    args = parser.parse_args()

# =================== 第一阶段：训练 ===================
if not args.skip_training:
    # 训练通用参数设置
    # --quiet: 静默模式，减少输出信息
    # --test_iterations -1: 不进行测试迭代
    # --depth_ratio 1.0: 深度比例因子
    # -r 2: 分辨率缩放因子
    # --lambda_dist 1000: 距离损失权重，用于2DGS的几何约束
    common_args = " --quiet --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 1000"
    
    # 遍历所有DTU测试场景进行训练
    for scene in dtu_scenes:
        # 构建数据源路径和输出路径
        source = args.dtu + "/" + scene
        output_model_path = args.output_path + "/" + scene
        
        # 构建并执行训练命令
        train_command = f"python train.py -s {source} -m {output_model_path}{common_args}"
        print(train_command)
        os.system(train_command)

# =================== 第二阶段：渲染和网格重建 ===================
if not args.skip_rendering:
    all_sources = []
    
    # 渲染通用参数设置
    # --quiet: 静默模式
    # --skip_train: 跳过训练视角的渲染，只渲染测试视角
    # --depth_ratio 1.0: 深度比例
    # --num_cluster 1: 聚类数量，用于网格融合
    # --voxel_size 0.004: 体素大小，控制网格分辨率
    # --sdf_trunc 0.016: SDF截断距离，影响表面重建质量
    # --depth_trunc 3.0: 深度截断值，过滤远距离点
    common_args = " --quiet --skip_train --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
    
    # 遍历所有场景进行渲染和网格提取
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        model_path = args.output_path + "/" + scene
        
        # 构建渲染命令
        # --iteration 30000: 使用30000次迭代的训练结果
        render_command = f"python render.py --iteration 30000 -s {source} -m{model_path}{common_args}"
        print(render_command)
        os.system(render_command)

# =================== 第三阶段：指标评估 ===================
if not args.skip_metrics:
    # 获取当前脚本所在目录，用于定位评估脚本
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 对每个场景进行指标评估
    for scene in dtu_scenes:
        # 提取场景ID（去掉"scan"前缀）
        scan_id = scene[4:]  # 例如：'scan24' -> '24'
        
        # 构建网格文件路径
        ply_file = f"{args.output_path}/{scene}/train/ours_30000/"
        iteration = 30000
        
        # 构建评估命令
        # evaluate_single_scene.py: DTU单场景评估脚本
        # --input_mesh: 输入的重建网格文件路径
        # --scan_id: DTU场景ID
        # --output_dir: 评估结果输出目录
        # --mask_dir: DTU数据集目录（包含评估掩码）
        # --DTU: DTU官方评估工具路径
        evaluation_command = (
            f"python {script_dir}/eval_dtu/evaluate_single_scene.py "
            f"--input_mesh {args.output_path}/{scene}/train/ours_30000/fuse_post.ply "
            f"--scan_id {scan_id} "
            f"--output_dir {script_dir}/tmp/scan{scan_id} "
            f"--mask_dir {args.dtu} "
            f"--DTU {args.DTU_Official}"
        )
        
        print(f"评估场景 {scene}...")
        os.system(evaluation_command)

# =================== 评估流程说明 ===================
"""
完整的DTU评估流程包含三个阶段：

1. 训练阶段 (Training):
   - 在每个DTU测试场景上训练2DGS模型
   - 使用特定的参数配置优化几何重建质量
   - 输出训练好的高斯点云模型

2. 渲染阶段 (Rendering):
   - 从训练好的模型渲染深度图和法向量
   - 使用TSDF融合算法重建三维网格
   - 输出后处理的网格文件 (fuse_post.ply)

3. 评估阶段 (Metrics):
   - 使用DTU官方评估协议计算几何指标
   - 主要指标包括：
     * Accuracy: 重建网格到真实表面的距离
     * Completeness: 真实表面到重建网格的距离
     * Overall: 准确性和完整性的平均值

使用示例：
python dtu_eval.py --dtu /path/to/dtu/dataset --DTU_Official /path/to/dtu/eval/tool
"""