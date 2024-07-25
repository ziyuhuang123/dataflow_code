#!/bin/bash

# 定义输出CSV文件
output_file="cutlass_results_sqz.csv"

# 初始化CSV文件，写入标题行
echo "batch,time" > $output_file

# 循环执行命令，改变批量大小
for i in {8..15}; do
    batch=$((2**i))
    result=$(build/mlp-eval-rowsync --batch $batch --check true --model gpt3 --split-k1 1 --split-k2 1 --policy baseline --order-line 3 --file-path "/home/zyhuang/temp_can/dataflow_code/block_swizzle/gen_two_order/order_256.txt")

    # 提取最后一行的时间
    time=$(echo "$result" | grep "Average time" | awk '{print $3}')

    # 输出结果到CSV文件
    echo "$batch,$time" >> $output_file
done

echo "结果已保存到 $output_file"
