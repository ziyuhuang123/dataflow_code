#!/bin/bash

# 定义遍历的参数
tileshape0_values=(64 128 256)
tileshape1_values=(64 128 256)
tileshape2_values=(16 32 64 128)
clustershape0_values=(1 2 4 8 16)
clustershape1_values=(1 2 4 8 16)
# tileshape0_values=(64)
# tileshape1_values=(64)
# tileshape2_values=(16)
# clustershape0_values=(1)
# clustershape1_values=(1 2)

# 定义源文件和目标文件路径
SOURCE_FILE="/home/zyhuang/cutlass/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu"
BUILD_DIR="/home/zyhuang/cutlass/build"
EXECUTABLE="$BUILD_DIR/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm"
RESULT_CSV="/home/zyhuang/cutlass/examples/48_hopper_warp_specialized_gemm/results.csv"
# 这里如果在一处用了绝对路径，那么都得用绝对路径，最开始我result_csv用的是相对路径，就一直写不进去。


# 初始化CSV文件
echo "tileshape0,tileshape1,tileshape2,clustershape0,clustershape1,runtime(ms),GFLOPS" > $RESULT_CSV

# 遍历所有参数组合
for tileshape0 in "${tileshape0_values[@]}"; do
  for tileshape1 in "${tileshape1_values[@]}"; do
    for tileshape2 in "${tileshape2_values[@]}"; do
      for clustershape0 in "${clustershape0_values[@]}"; do
        for clustershape1 in "${clustershape1_values[@]}"; do
          if (( clustershape0 * clustershape1 <= 16 )); then
            # 修改源文件中的参数
            sed -i "s/using TileShape.*/using TileShape = Shape<_${tileshape0}, _${tileshape1}, _${tileshape2}>;/g; s/using ClusterShape.*/using ClusterShape = Shape<_${clustershape0}, _${clustershape1}, _1>;/g" $SOURCE_FILE
            
            # 编译
            cd $BUILD_DIR
            make 48_hopper_warp_specialized_gemm

            # 运行并提取运行时间和GFLOPS
            OUTPUT=$($EXECUTABLE --m=4096 --n=20480 --k=5120)
            echo "Raw output:"
            echo "$OUTPUT"
            
            # 提取运行时间
            RUNTIME=$(echo "$OUTPUT" | grep "Avg runtime" | awk '{print $3}')
            echo "Extracted runtime: $RUNTIME"

            # 提取GFLOPS
            GFLOPS=$(echo "$OUTPUT" | grep "GFLOPS" | awk '{print $2}')
            echo "Extracted GFLOPS: $GFLOPS"

            # 检查提取结果是否为空
            if [[ -z "$RUNTIME" ]] || [[ -z "$GFLOPS" ]]; then
              echo "Warning: Failed to extract runtime or GFLOPS for parameters: TileShape=(${tileshape0}, ${tileshape1}, ${tileshape2}), ClusterShape=(${clustershape0}, ${clustershape1}, 1)"
              continue
            fi

            # 输出当前参数组合和运行结果
            echo "Current parameters: TileShape=(${tileshape0}, ${tileshape1}, ${tileshape2}), ClusterShape=(${clustershape0}, ${clustershape1}, 1)"
            echo "Output:"
            echo "$OUTPUT"
            echo "Extracted runtime: $RUNTIME"
            echo "Extracted GFLOPS: $GFLOPS"

            # 输出结果到CSV
            echo "Appending to CSV: $tileshape0,$tileshape1,$tileshape2,$clustershape0,$clustershape1,$RUNTIME,$GFLOPS"
            echo $tileshape0,$tileshape1,$tileshape2,$clustershape0,$clustershape1,$RUNTIME,$GFLOPS >> $RESULT_CSV

            echo "CSV appended."
          fi
        done
      done
    done
  done
done

echo "All done! Results saved to $RESULT_CSV"
