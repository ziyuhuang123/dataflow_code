#!/bin/bash

# 定义输出路径
output_path="/home/zyhuang/temp_can/delete_code/two_gemm_order"

# 遍历 M=2^i, i=[8, 16]
for i in $(seq 8 16)
do
  M=$(echo "2^$i" | bc)
  echo "Generating orders for M=$M"
  python gen_two_order.py --M "$M" --file_path "$output_path"
done

echo "All orders have been generated."
