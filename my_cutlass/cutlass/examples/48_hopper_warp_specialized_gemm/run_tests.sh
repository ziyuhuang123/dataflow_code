#!/bin/bash

# 定义变量
SOURCE_FILE="/home/zyhuang/cutlass/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu"
BUILD_DIR="/home/zyhuang/cutlass/build"
EXECUTABLE="$BUILD_DIR/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm"

# 数组定义cluster0和cluster1的可能值
cluster0_values=(2 4 8 16)
cluster1_values=(16 8 4 2 1)

# 之前已经处理过的cluster0和cluster1的组合
skip_combinations=(
    "2 8"
    "2 4"
    "16 1"
    "8 1"
    # 添加更多已经处理过的组合
)

# 遍历所有可能的cluster0和cluster1值
for cluster0 in "${cluster0_values[@]}"; do
    for cluster1 in "${cluster1_values[@]}"; do

        # 检查是否跳过已经处理过的组合
        skip=false
        for combination in "${skip_combinations[@]}"; do
            if [[ "$combination" == "$cluster0 $cluster1" ]]; then
                skip=true
                break
            fi
        done
        if $skip; then
            continue
        fi

        # 检查cluster0 * cluster1是否<= 16
        if [ $((cluster0 * cluster1)) -le 16 ]; then
            # 使用sed命令修改SOURCE_FILE中的ClusterShape

            echo "${cluster0}, _${cluster1}"
            sed -i "s/using ClusterShape.*/using ClusterShape = Shape<_${cluster0}, _${cluster1}, _1>;/g" $SOURCE_FILE

            # 编译
            cd $BUILD_DIR
            make 48_hopper_warp_specialized_gemm

            # 执行测试
            output_file="${BUILD_DIR}/cluster-${cluster0}-${cluster1}_MNK-4096-20480-5120_BLK-128-256-64_cutlass-full"
            ncu --set full --replay-mode application --app-replay-match grid --app-replay-buffer file -f --export ${output_file} $EXECUTABLE --m=4096 --n=20480 --k=5120
        fi
    done
done
