#include "cutlass/numeric_types.h"
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cuda_runtime.h>
#include <iostream>

// 定义M和N
int M = 1024;
int N = 512;

// 主程序
int main() {
    // 1. 定义Tensor的shape (M, N)
    auto tensor_shape = cute::make_shape(M, N);
    
    // 2. 创建global memory layout (使用右侧布局)
    auto gmemLayoutD = cute::make_layout(tensor_shape, cute::LayoutRight{});
    
    // 3. 设备内存分配
    half *d_gemm1_result;
    cudaMalloc(&d_gemm1_result, M * N * sizeof(half));

    // 4. 创建Tensor对象 (将global memory与布局结合)
    cute::Tensor tensor_D = cute::make_tensor(
        cute::make_gmem_ptr(d_gemm1_result), gmemLayoutD
    );

    // 输出一些Tensor的信息 (例如维度)
    std::cout << "Tensor D: M = " << M << ", N = " << N << std::endl;

    // 5. 释放设备内存
    cudaFree(d_gemm1_result);
    
    return 0;
}
