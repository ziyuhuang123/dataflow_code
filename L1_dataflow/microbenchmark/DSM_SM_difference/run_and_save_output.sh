#!/bin/bash

# 编译CUDA程序
nvcc simple_thr_ld.cu -o simple_thr_ld -arch=sm_90 -std=c++17 -I./Common

# 执行程序并将输出保存到result.txt文件中
./simple_thr_ld > result.txt

# 提示输出文件保存成功
echo "Output has been saved to result.txt"
