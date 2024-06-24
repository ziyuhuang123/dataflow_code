#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Helper function for error checking
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

#define CHECK_CUBLAS_ERROR(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

void initialize_matrix(half *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<half>(rand() % 100 / 100.0f);
    }
}

int main() {
    const int M1 = 16, N1 = 14336, K1 = 4096;
    const int M2 = 14336, N2 = 4096, K2 = 14336;

    const int iterations = 20;

    // Allocate host memory
    half *h_A1 = new half[M1 * K1];
    half *h_B1 = new half[K1 * N1];
    half *h_C1 = new half[M1 * N1];

    half *h_B2 = new half[K2 * N2];
    half *h_C2 = new half[M1 * N2];

    // Initialize matrices
    initialize_matrix(h_A1, M1, K1);
    initialize_matrix(h_B1, K1, N1);
    initialize_matrix(h_B2, K2, N2);

    // Allocate device memory
    half *d_A1, *d_B1, *d_C1;
    half *d_B2, *d_C2;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A1, M1 * K1 * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B1, K1 * N1 * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C1, M1 * N1 * sizeof(half)));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B2, K2 * N2 * sizeof(half)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C2, M1 * N2 * sizeof(half)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A1, h_A1, M1 * K1 * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B1, h_B1, K1 * N1 * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B2, h_B2, K2 * N2 * sizeof(half), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    // Set cuBLAS to use tensor cores
    CHECK_CUBLAS_ERROR(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Alpha and beta values
    const half alpha = 1.0;
    const half beta = 0.0;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

    for (int i = 0; i < iterations; ++i) {
        // First GEMM: d_C1 = d_A1 * d_B1
        CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        N1, M1, K1,
                                        &alpha,
                                        d_B1, CUDA_R_16F, N1,
                                        d_A1, CUDA_R_16F, K1,
                                        &beta,
                                        d_C1, CUDA_R_16F, N1,
                                        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // Synchronize to ensure the first GEMM is complete
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Second GEMM: d_C2 = d_C1 * d_B2
        CHECK_CUBLAS_ERROR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        N2, M1, K2,
                                        &alpha,
                                        d_B2, CUDA_R_16F, N2,
                                        d_C1, CUDA_R_16F, K2,
                                        &beta,
                                        d_C2, CUDA_R_16F, N2,
                                        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Record the stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate the elapsed time
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    float avg_time = milliseconds / iterations;

    std::cout << "Average time per iteration: " << avg_time << " milliseconds" << std::endl;

    // Clean up
    CHECK_CUDA_ERROR(cudaFree(d_A1));
    CHECK_CUDA_ERROR(cudaFree(d_B1));
    CHECK_CUDA_ERROR(cudaFree(d_C1));
    CHECK_CUDA_ERROR(cudaFree(d_B2));
    CHECK_CUDA_ERROR(cudaFree(d_C2));

    delete[] h_A1;
    delete[] h_B1;
    delete[] h_C1;
    delete[] h_B2;
    delete[] h_C2;

    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return 0;
}
