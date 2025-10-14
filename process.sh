#!/bin/bash

# 定义根目录
BASE_DIR="../DTU_2DGS"

# 日志文件路径
LOG_FILE="process_scans.log"

# 清空或创建日志文件
> "$LOG_FILE"

# 遍历所有符合 scan<number> 格式的文件夹
for scan_dir in "$BASE_DIR"/scan[0-9]*; do
    # 检查是否是目录
    if [ -d "$scan_dir" ]; then
        # 提取 scan<number> 中的 <number>
        scan_number=$(basename "$scan_dir" | grep -oP '(?<=scan)\d+')

        # 输出当前处理的 scan 文件夹
        echo "Processing $scan_dir (Scan Number: $scan_number)" | tee -a "$LOG_FILE"

        # 记录开始时间
        start_time=$(date +%s)

        # 执行 train.py 命令
        echo "Running train.py for scan$scan_number..." | tee -a "$LOG_FILE"
        python train.py -s "$scan_dir" -m "output/data/scan$scan_number"

        # 检查 train.py 是否成功执行
        if [ $? -eq 0 ]; then
            echo "train.py completed successfully for scan$scan_number." | tee -a "$LOG_FILE"
        else
            echo "train.py failed for scan$scan_number. Skipping render.py." | tee -a "$LOG_FILE"
            continue
        fi

        # 执行 render.py 命令
        echo "Running render.py for scan$scan_number..." | tee -a "$LOG_FILE"
        python render.py -r 2 --depth_ratio 1 --skip_test --skip_train --model_path "output/data/scan$scan_number"

        # 检查 render.py 是否成功执行
        if [ $? -eq 0 ]; then
            echo "render.py completed successfully for scan$scan_number." | tee -a "$LOG_FILE"
        else
            echo "render.py failed for scan$scan_number." | tee -a "$LOG_FILE"
        fi

        # 记录结束时间
        end_time=$(date +%s)

        # 计算执行时间
        execution_time=$((end_time - start_time))

        # 输出执行时间
        echo "Finished processing scan$scan_number. Execution time: $execution_time seconds." | tee -a "$LOG_FILE"
        echo "----------------------------------------" | tee -a "$LOG_FILE"
    fi
done

echo "All scans processed. Log saved to $LOG_FILE."
