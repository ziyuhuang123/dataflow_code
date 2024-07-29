#!/bin/bash

# 定义参数范围
nbins_values=(256 512 1024 2048)
block_size_values=(128 512)
cluster_size_values=(1 2 4 8 16)
array_size=2000000

# 输出文件名
output_file="results.csv"

# 写入 CSV 文件的表头
echo "Block Size,Nbins,Cluster Size,Average Time (sec),Throughput (Gelem/s)" > $output_file

# 循环不同参数值
for block_size in "${block_size_values[@]}"; do
    for nbins in "${nbins_values[@]}"; do
        for cluster_size in "${cluster_size_values[@]}"; do
            # 输出当前参数组合
            echo "Running with Block Size: $block_size, Nbins: $nbins, Cluster Size: $cluster_size"

            # 运行程序并捕获输出
            output=$(./hzy_throughput --nbins $nbins --array_size $array_size --threads_per_block $block_size --cluster_size $cluster_size)
            
            # 打印程序输出
            echo "$output"

            # 从输出中提取时间和吞吐量
            avg_time=$(echo "$output" | grep "Average time per iteration (seconds)" | sed -e 's/^.*: //' -e 's/ sec//')
            throughput=$(echo "$output" | grep "Throughput (Giga Elements/second)" | sed -e 's/^.*: //' -e 's/ Gelem\/s//')

            # 检查avg_time和throughput是否有效，避免不合理的值
            if [[ -z "$avg_time" || "$avg_time" == "0.00000" ]]; then
                avg_time="N/A"
                throughput="N/A"
            fi

            # 写入 CSV 文件
            echo "$block_size,$nbins,$cluster_size,$avg_time,$throughput" >> $output_file

            # 输出当前结果
            echo "Result: Block Size: $block_size, Nbins: $nbins, Cluster Size: $cluster_size, Average Time: $avg_time sec, Throughput: $throughput Gelem/s"
        done
    done
done
