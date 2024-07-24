#!/bin/bash

# 定义文件路径
input_file="/home/zyhuang/temp_can/dataflow_code/block_swizzle/gen_order/new_gen_order/orders.txt"
output_file="orders_results.csv"

# 写入CSV表头
echo "Parameter,Order,L2 Hit Rate(%),Kernel Duration (us)" > $output_file

# 行号计数器
line_num=1

# 逐行读取文件
while IFS= read -r line
do
  # 提取 t1 = 或 outer_width = 的信息
  parameter=$(echo "$line" | grep -oP '^(t1 = \d+|outer_width = \d+, inner_width = \d+)')
  
  # 提取 Order 的值
  order=$(echo "$line" | grep -oP 'Order \d+' | grep -oP '\d+')
  
  # 如果没有找到有效的parameter或order，跳过此行
  if [ -z "$parameter" ] || [ -z "$order" ]; then
    echo "Skipping line $line_num due to missing parameter or order"
    line_num=$((line_num + 1))
    continue
  fi
  
  # 运行 ncu 命令并提取 L2 Hit Rate 和 sm__cycles_elapsed.avg
  echo "Running ncu for order_line $line_num"
  ncu_result=$(ncu --metrics sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second --target-processes all --section MemoryWorkloadAnalysis --clock-control=base --cache-control=all /home/zyhuang/temp_can/dataflow_code/cusync/src/ml-bench/transformer/build/mlp-eval-rowsync --batch 8192 --check true --model gpt3 --split-k1 1 --split-k2 1 --policy cusync --order-line $line_num 2>&1)
  
  # 输出 ncu 命令结果，便于调试
  echo "NCU Result for order_line $line_num:"
  echo "$ncu_result"
  
  # 提取 L2 Hit Rate 和 sm__cycles_elapsed.avg
  l2_hit_rate=$(echo "$ncu_result" | awk '/AllKernel/ {found=1} found && /L2 Hit Rate/ {rate=$NF} END{print rate}' | tr -d '%')
  sm_cycles=$(echo "$ncu_result" | awk '/AllKernel/ {found=1} found && /sm__cycles_elapsed.avg / {cycles=$3} END{print cycles}' | tr -d ',')
  sm_cycles_per_second=$(echo "$ncu_result" | awk '/AllKernel/ {found=1} found && /sm__cycles_elapsed.avg.per.second/ {cycles_per_sec=$3} END{print cycles_per_sec}' | tr -d ',')

  # 输出提取的性能数据，便于调试
  echo "L2 Hit Rate: $l2_hit_rate"
  echo "SM Cycles: $sm_cycles"
  echo "SM Cycles per Second: $sm_cycles_per_second"

  # 检查提取的值是否为空
  if [ -z "$l2_hit_rate" ] || [ -z "$sm_cycles" ] || [ -z "$sm_cycles_per_second" ]; then
    l2_hit_rate=""
    kernel_duration=""
    echo "enter if"
  else
    echo "enter else!"
    # 确保 sm_cycles 和 sm_cycles_per_second 是有效的数值格式
    sm_cycles=$(echo $sm_cycles | awk '{printf "%.10f", $1}')
    sm_cycles_per_second=$(echo $sm_cycles_per_second | awk '{printf "%.10f", $1}')

    echo "Formatted SM Cycles: $sm_cycles"
    echo "Formatted SM Cycles per Second: $sm_cycles_per_second"

    # 计算 Kernel Duration 使用 awk，并转换为微秒
    kernel_duration=$(awk "BEGIN {print ($sm_cycles / ($sm_cycles_per_second * 1e9)) * 1e6}")
  fi

  # 输出计算的 Kernel Duration，便于调试
  echo "Kernel Duration (us): $kernel_duration"

  # 写入CSV文件
  echo "$parameter,$order,$l2_hit_rate,$kernel_duration" >> $output_file
  
  # 增加行号计数器
  line_num=$((line_num + 1))
  
  # 如果已经处理了50行，则跳出循环
  if [ $line_num -gt 50 ]; then
    break
  fi
  
done < "$input_file"

echo "CSV file has been generated: $output_file"
