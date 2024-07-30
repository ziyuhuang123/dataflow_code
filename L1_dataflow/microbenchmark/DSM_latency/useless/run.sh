#!/bin/bash


# 遍历bs值
for bs in 64 128 256 512 1024; do
    # 遍历ilp值
    for ilp in 1 2 4 8; do
        # 定义宏并编译C文件
        echo "Compiling DSM benchmark with BS=$bs and ILP=$ilp" | tee -a log
        nvcc -ccbin g++ -I../../Common -m64 --threads 0 --std=c++11 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o dsm_throughput.o -c dsm_throughput.cu -DBS=$bs -DILP=$ilp
        nvcc -ccbin g++ -I../../Common -m64 --threads 0 --std=c++11 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o main.o -c main.cpp
        nvcc -ccbin g++ -m64 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o dsm_benchmark dsm_throughput.o dsm_latency.o main.o 
        CUDA_VISIBLE_DEVICES=1 ./dsm_benchmark | tee -a log
    done
done
