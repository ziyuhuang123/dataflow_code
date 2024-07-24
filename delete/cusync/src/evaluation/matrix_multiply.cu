#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(call) \
    if((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

#define CHECK_CUBLAS(call) \
    if((call) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

void initializeMatrix(half *matrix, int size, half value) {
    for (int i = 0; i < size; i++) {
        matrix[i] = value;
    }
}

int main() {
    const int m = 512;
    const int n = 512;
    const int k = 512;

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    // Allocate memory for matrices on the host
    std::vector<half> h_A(m * k, __float2half(0.1f));
    std::vector<half> h_B(k * n, __float2half(0.1f));
    std::vector<half> h_C(m * n, __float2half(0.0f));

    // Allocate memory for matrices on the device
    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(half)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), k * n * sizeof(half), cudaMemcpyHostToDevice));

    // Perform matrix multiplication C = A * B
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                              n, m, k,
                              &alpha,
                              d_B, CUDA_R_16F, n,
                              d_A, CUDA_R_16F, k,
                              &beta,
                              d_C, CUDA_R_16F, n,
                              CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, m * n * sizeof(half), cudaMemcpyDeviceToHost));

    // Print the first 10 elements of matrix C
    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << __half2float(h_C[i]) << std::endl;
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Destroy the cuBLAS handle
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
