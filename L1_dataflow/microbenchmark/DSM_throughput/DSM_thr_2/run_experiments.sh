#!/bin/bash

# 定义参数范围
cluster_sizes=(1 2 4 8 16)
threads_per_blocks=(64 128 256 512 1024)
block_num=13200
num_runs=2000

# 输出文件名
output_file="results.csv"

# 写入 CSV 文件的表头
echo "Block Num,Cluster Size,Threads Per Block,Num Runs,Average Time (sec),Throughput (TB/s)" > $output_file

# 编译和运行函数
run_experiment() {
    local threads_per_block=$1
    local cluster_size=$2
    local temp_size=$((128 * 128 / threads_per_block))

    # 替换代码中的THREADS_PER_BLOCK和temp数组大小
    sed -i "s/int THREADS_PER_BLOCK = [0-9]*;/int THREADS_PER_BLOCK = ${threads_per_block};/" delete5.cu
    sed -i "s/int temp\[[0-9]*\] = {0};/int temp[${temp_size}] = {0};/" delete5.cu

    # 编译
    nvcc delete5.cu -o delete5 -arch=sm_90 -std=c++17 -I./Common

    # 执行程序并捕获输出
    output=$(./delete5 --block_num $block_num --cluster_size $cluster_size --threads_per_block $threads_per_block --num_runs $num_runs)

    # 打印程序输出
    echo "$output"

    # 从输出中提取时间和吞吐量
    avg_time=$(echo "$output" | grep "dsm_sm2sm_thrpt() time (average)" | awk -F ': ' '{print $2}' | awk '{print $1}')
    throughput=$(echo "$output" | grep "Throughput" | awk '{print $2}')

    # 检查avg_time和throughput是否有效，避免不合理的值
    if [[ -z "$avg_time" || "$avg_time" == "0.00000" ]]; then
        avg_time="N/A"
        throughput="N/A"
    fi

    # 写入 CSV 文件
    echo "$block_num,$cluster_size,$threads_per_block,$num_runs,$avg_time,$throughput" >> $output_file

    # 输出当前结果
    echo "Result: Block Num: $block_num, Cluster Size: $cluster_size, Threads Per Block: $threads_per_block, Num Runs: $num_runs, Average Time: $avg_time sec, Throughput: $throughput TB/s"
}

# 循环不同参数值
for cluster_size in "${cluster_sizes[@]}"; do
    for threads_per_block in "${threads_per_blocks[@]}"; do
        # 输出当前参数组合
        echo "Running with Block Num: $block_num, Cluster Size: $cluster_size, Threads Per Block: $threads_per_block, Num Runs: $num_runs"
        
        # 运行实验
        run_experiment $threads_per_block $cluster_size
    done
done
