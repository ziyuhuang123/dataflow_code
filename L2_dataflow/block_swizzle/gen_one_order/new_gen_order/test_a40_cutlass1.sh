#!/bin/bash

# 定义结果文件路径
output_file="baseline_results_swizzle.csv"

# 写入CSV表头
echo "Batch Size,Swizzle Number,L2 Hit Rate(%),Kernel Duration (us)" > $output_file

# 定义初始 batch size 和结束 batch size 的指数
batch_exp_start=8  # 2^8
batch_exp_end=16   # 2^16

# 定义 swizzle number 的范围
swizzle_exp_start=0  # 2^0
swizzle_exp_end=4    # 2^4

# 逐个 batch size 和 swizzle number 运行 ncu
for i in $(seq $batch_exp_start $batch_exp_end)
do
  batch_size=$(echo "2^$i" | bc)
  
  for j in $(seq $swizzle_exp_start $swizzle_exp_end)
  do
    swizzle_number=$(echo "2^$j" | bc)
    
    # 修改源文件中的 swizzle_number
    sed -i "s/cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<{[^}]*}>/cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<{$swizzle_number}>/g" /home/zyhuang/temp_can/dataflow_code/cusync/src/ml-bench/transformer/build/mlp-eval-rowsync.cu
    
    # 编译
    make -C /home/zyhuang/temp_can/dataflow_code/cusync/src/ml-bench/transformer build/mlp-eval-rowsync

    # 运行 ncu 命令并提取 L2 Hit Rate 和 sm__cycles_elapsed.avg
    echo "Running ncu for batch size $batch_size and swizzle number $swizzle_number"
    ncu_result=$(ncu --metrics sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second --target-processes all --section MemoryWorkloadAnalysis --clock-control=base --cache-control=all /home/zyhuang/temp_can/dataflow_code/cusync/src/ml-bench/transformer/build/mlp-eval-rowsync --batch $batch_size --check true --model gpt3 --split-k1 1 --split-k2 1 --policy baseline --order-line 1 2>&1)
    
    # 输出 ncu 命令结果，便于调试
    echo "NCU Result for batch size $batch_size and swizzle number $swizzle_number:"
    echo "$ncu_result"
    
    # 提取 L2 Hit Rate 和 sm__cycles_elapsed.avg
    l2_hit_rate=$(echo "$ncu_result" | awk '/cutlass_cutlass::Kernel/{found=1} found && /L2 Hit Rate/ {rate=$NF} END{print rate}' | tr -d '%')
    sm_cycles=$(echo "$ncu_result" | awk '/cutlass_cutlass::Kernel/{found=1} found && /sm__cycles_elapsed.avg / {cycles=$3} END{print cycles}' | tr -d ',')
    sm_cycles_per_second=$(echo "$ncu_result" | awk '/cutlass_cutlass::Kernel/{found=1} found && /sm__cycles_elapsed.avg.per.second/ {cycles_per_sec=$3} END{print cycles_per_sec}' | tr -d ',')

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
    echo "$batch_size,$swizzle_number,$l2_hit_rate,$kernel_duration" >> $output_file
    
  done
  
done

echo "CSV file has been generated: $output_file"
