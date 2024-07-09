#!/bin/bash

# 定义结果文件路径
output_file="orders_results_all.csv"
echo "Batch Size,Parameter,Order,L2 Hit Rate(%),Kernel Duration (us)" > $output_file

# 定义 batch size 的范围
batch_exp_start=8  # 2^8
batch_exp_end=16   # 2^16

# 遍历 batch size 并执行 ncu
for i in $(seq $batch_exp_start $batch_exp_end)
do
  batch_size=$(echo "2^$i" | bc)  # 使用 bc 进行指数计算

  # 生成对应的 orders 文件
  python plot1.py --M $batch_size --output_dir /home/zyhuang/temp_can/delete_code

  # 读取生成的 orders 文件并逐行处理
  input_file="/home/zyhuang/temp_can/delete_code/orders_${batch_size}.txt"

  order_line=1
  while IFS= read -r line
  do
    # 提取 t1 = 或 outer_width = 的信息，并拼接 outer_width 和 inner_width
    parameter=$(echo "$line" | grep -oP '^(t1 = \d+|outer_width = \d+, inner_width = \d+)')
    parameter=$(echo "$parameter" | sed 's/outer_width = \([0-9]*\), inner_width = \([0-9]*\)/outer_width=\1,inner_width=\2/')

    # 提取 Order 的值
    order=$(echo "$line" | grep -oP 'Order \d+' | grep -oP '\d+')

    # 如果没有找到有效的parameter或order，跳过此行
    if [ -z "$parameter" ] || [ -z "$order" ]; then
      echo "Skipping line due to missing parameter or order"
      continue
    fi

    file_path="/home/zyhuang/temp_can/delete_code/orders_${batch_size}.txt"
  
    # 运行 ncu 命令并提取 L2 Hit Rate 和 sm__cycles_elapsed.avg
    echo "Running ncu for batch size $batch_size, parameter $parameter and order $order"
    ncu_result=$(ncu --metrics sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second --target-processes all --section MemoryWorkloadAnalysis --clock-control=base --cache-control=all /home/zyhuang/temp_can/dataflow_code/cusync/src/ml-bench/transformer/build/mlp-eval-rowsync --batch $batch_size --check true --model gpt3 --split-k1 1 --split-k2 1 --policy cusync --order-line $order_line  --file-path $file_path 2>&1)

    # 输出 ncu 命令结果，便于调试
    echo "NCU Result for batch size $batch_size, parameter $parameter and order $order:"
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
    echo "$batch_size,$parameter,$order,$l2_hit_rate,$kernel_duration" >> $output_file

    # 增加 order_line 计数器
    order_line=$((order_line + 1))
    if [ $order_line -gt 29 ]; then
      break
    fi

  done < "$input_file"
done

echo "CSV file has been generated: $output_file"
