#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <chrono>

#define CHECK_CUDA(call) \
    if((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(call) << std::endl; \
        exit(1); \
    }

#define CHECK_CUBLAS(call) \
    if((call) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

void initialize_matrix(half *matrix, int rows, int cols, half value) {
    half *host_matrix = new half[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        host_matrix[i] = value;
    }
    CHECK_CUDA(cudaMemcpy(matrix, host_matrix, rows * cols * sizeof(half), cudaMemcpyHostToDevice));
    delete[] host_matrix;
}

void print_first_element(half *matrix) {
    half host_value;
    CHECK_CUDA(cudaMemcpy(&host_value, matrix, sizeof(half), cudaMemcpyDeviceToHost));
    std::cout << "First element: " << __half2float(host_value) << std::endl;
}

void handle_cublas_status(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasGemmEx failed with error code: " << status << std::endl;
        switch(status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED" << std::endl;
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                std::cerr << "CUBLAS_STATUS_ALLOC_FAILED" << std::endl;
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                std::cerr << "CUBLAS_STATUS_INVALID_VALUE" << std::endl;
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH" << std::endl;
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                std::cerr << "CUBLAS_STATUS_MAPPING_ERROR" << std::endl;
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED" << std::endl;
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR" << std::endl;
                break;
            case CUBLAS_STATUS_NOT_SUPPORTED:
                std::cerr << "CUBLAS_STATUS_NOT_SUPPORTED" << std::endl;
                break;
            case CUBLAS_STATUS_LICENSE_ERROR:
                std::cerr << "CUBLAS_STATUS_LICENSE_ERROR" << std::endl;
                break;
            default:
                std::cerr << "Unknown cublas status" << std::endl;
        }
        exit(1);
    }
}

int main() {
    const int M = 4096;
    const int N = 20480;
    const int K = 5120;
    const int iterations = 1000;

    // Allocate device memory
    half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc((void**)&d_C, M * N * sizeof(half)));

    // Initialize matrices
    initialize_matrix(d_A, M, K, __float2half(1.0f));
    initialize_matrix(d_B, K, N, __float2half(1.0f));
    initialize_matrix(d_C, M, N, __float2half(0.0f));

    // Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Set cuBLAS to use Tensor Cores
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Define scaling factors
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // A is M x K
    // B is K x N
    // C is M x N

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // A is M x K
    // B is K x N
    // C is M x N

    for (int i = 0; i < iterations; ++i) {
        cublasStatus_t status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  d_B, CUDA_R_16F, N,
                                  d_A, CUDA_R_16F, K,
                                  &beta,
                                  d_C, CUDA_R_16F, N,
                                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate the elapsed time
    float elapsed_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time_ms, start, stop));

    float average_time_ms = elapsed_time_ms / iterations;

    // Print the first element of the result matrix
    print_first_element(d_C);

    // Print the average execution time
    std::cout << "Average execution time: " << average_time_ms << " ms" << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));

    std::cout << "Matrix multiplication completed successfully!" << std::endl;
    return 0;
}
