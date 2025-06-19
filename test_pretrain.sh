#!/bin/bash

# 创建日志目录
mkdir -p ./log
mkdir -p ./log/poster

# 设置最大并行任务数
MAX_PARALLEL_JOBS=8

# 计数器
running_jobs=0

for user_id in {1..35}; do
    # 检查当前运行的任务数，如果达到最大值则等待某个任务完成
    if [ $running_jobs -ge $MAX_PARALLEL_JOBS ]; then
        wait -n  # 等待任意一个后台任务完成
        running_jobs=$((running_jobs - 1))
    fi
    
    # 计算该任务的基础端口
    BASE_PORT=$((29500 + user_id * 10))
    
    # 启动训练任务并放入后台
    echo "Starting training for user_${user_id}..."
    
    # 使用更强的方式设置端口，包括显式指定MASTER_PORT环境变量
    MASTER_PORT=$BASE_PORT MASTER_ADDR=localhost OMP_NUM_THREADS=1 \
    /home/daibo/.conda/envs/xinyu/bin/python test_pretrain_on_finetune_data.py \
        --model reconstruction_base_patch250_250 \
        --batch_size 40 \
        --layer_scale_init_value 0.1 \
        --checkpoint_path ./checkpoints/user_${user_id} \
        --epochs 1999\
        --reconstruction_save_path ./reconstruct_results/user_${user_id} \
        --sub 1 > ./test_log/user_${user_id}.log 2>&1 &

    
    sleep 5  # 增加等待时间，确保进程完全启动
    running_jobs=$((running_jobs + 1))
done

# 等待所有剩余任务完成
wait
echo "All training jobs completed!"