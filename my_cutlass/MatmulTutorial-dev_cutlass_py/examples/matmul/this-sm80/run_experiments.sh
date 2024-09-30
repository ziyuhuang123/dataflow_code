#!/bin/bash

# 文件路径
MATMUL_PATH="/home/zyhuang/temp_can/MatmulTutorial-dev_cutlass_py/examples/matmul/this-sm80"
MAIN_FILE="$MATMUL_PATH/main.cu"

# 输出的CSV文件
OUTPUT_FILE="$MATMUL_PATH/results.csv"
echo "N,M,K,T,Time(ms)" > $OUTPUT_FILE

# 运行日志文件
LOG_FILE="$MATMUL_PATH/run_output.log"
echo "" > $LOG_FILE  # 清空日志文件

# GPU设置
GPU_DEVICE=0  # 选择要使用的GPU编号
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE  # 指定GPU设备
sudo nvidia-smi -i $GPU_DEVICE -c EXCLUSIVE_PROCESS  # 将该GPU设为exclusive模式（使用sudo）

# 参数范围
# cluster_sizes=(1 2 4 8)
# Ms=(1024 2048 4096 8192)
# Ks=(128 1024 2048 4096 8192)
# Ts=(128 1024 2048 4096 8192)

cluster_sizes=(1 8)
Ms=(16384 40960 51200 614400 81920)
Ks=(128 256)
Ts=(128 256 512 1024)

# 遍历每个参数组合
for cluster_size in "${cluster_sizes[@]}"; do
  N=$((128 * cluster_size))  # 计算 N 的值
  for M in "${Ms[@]}"; do
    for K in "${Ks[@]}"; do
      for T in "${Ts[@]}"; do
        # 使用awk定位和修改main.cu中的参数，同时保留注释
        awk -v cs="$cluster_size" -v m="$M" -v k="$K" -v t="$T" -v n="$N" '
        /\/\/ start bash modify/ { flag = 1; print; next }
        /\/\/ end bash modify/ { flag = 0; print; next }
        flag && /const int cluster_size/ { print "const int cluster_size = " cs ";" ; next }
        flag && /const int M/ { print "const int M = " m ";" ; next }
        flag && /const int K/ { print "const int K = " k ";" ; next }
        flag && /const int T/ { print "const int T = " t ";" ; next }
        flag && /const int N/ { print "const int N = " n ";" ; next }
        { print }
        ' $MAIN_FILE > temp_file && mv temp_file $MAIN_FILE

        # 编译并运行程序
        nvcc -arch=sm_90 -DDEBUG -Xcompiler -fopenmp -lcublas matmul-v00.cu main.cu -o test
        if [ $? -ne 0 ]; then
          echo "Compilation failed for N=$N, M=$M, K=$K, T=$T"
          continue
        fi

        # 运行并提取所有输出
        output=$(./test stages 1 2>&1)
        echo "$output" >> $LOG_FILE  # 将所有输出写入日志文件

        # 提取运行时间
        time=$(echo "$output" | grep -oP '(?<=Running cost \(ms\) of matmul is )[0-9.]+')

        # 保存结果到CSV文件，使用N的值代替cluster_size
        echo "$N,$M,$K,$T,$time" >> $OUTPUT_FILE
        echo "Completed N=$N, M=$M, K=$K, T=$T"
      done
    done
  done
done

# 恢复 GPU 模式为默认模式
sudo nvidia-smi -i $GPU_DEVICE -c DEFAULT  # 恢复GPU模式为默认模式（使用sudo）

echo "All experiments completed. Results saved in $OUTPUT_FILE."
echo "Complete log is saved in $LOG_FILE."
