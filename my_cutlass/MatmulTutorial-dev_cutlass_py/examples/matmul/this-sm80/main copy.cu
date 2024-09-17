#include <cuda_fp16.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <string>
#include <cassert>

int STAGES = 1;
int MULTI_THREADING = 1;
int ITERS = 1;

// extern __global__ void matmul(half *A, half *B, half *C, int M, int N, int K, float alpha, float beta);
extern __global__ void matmul(half *A, half *B, half *gemm1_result, half *gemm1_weight, int M, int N, int K, int T, float alpha, float beta);



// #define DEBUG
// #define PRINT
#ifdef DEBUG
#include <omp.h>
// const int M = 1024;
// const int N = 1024;
// const int K = 1024;

const int M = 128;
const int N = 128;
const int K = 128;
const int T = 256;
#else
const int M = 5376;
const int N = 5376;
const int K = 2048;
const int T = 128;

// const int M = 256;
// const int N = 256;
// const int K = 64;


#endif
#define MAX(a, b) (a) > (b) ? (a) : (b)

float alpha = 1.0;
float beta = 0.0;

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

int main(int argc, char *argv[])
{

    if (argc > 1)
    {
        assert((argc - 1) % 2 == 0);
        for (int i = 1; i < argc; i += 2)
        {
            char *key = argv[i];
            char *value = argv[i + 1];
            std::string keys(key);
            if (keys == "stages")
            {
                STAGES = std::atoi(value);
                std::cout << "Setting to " << STAGES << " stages.\n";
            }
            else if (keys == "multi_threading")
            {
                MULTI_THREADING = std::atoi(value);
                std::cout << "Setting to " << MULTI_THREADING << "x threading.\n";
            }
            else if (keys == "iters") {
                ITERS = std::atoi(value);
                std::cout << "Testing iters = " << ITERS << ".\n";
            }
        }
    }
#ifdef DEBUG
    std::cout << "Debugging using shape M=" << M << ", N=" << N << ", K=" << K << ", T=" << T << "\n";
#else
    std::cout << "Test performance using shape M=" << M << ", N=" << N << ", K=" << K << "\n";
#endif
    srand(time(NULL));
    half *hA = (half *)malloc(M * K * 2);
    half *hB = (half *)malloc(K * N * 2);
    half *hC = (half *)malloc(M * N * 2);
    half *h_gemm1_result = (half *)malloc(M * T * 2);
    half *h_gemm1_weight = (half *)malloc(N * T * 2);
    half *golden = (half *)malloc(M * N * 2);
    half *golden1 = (half *)malloc(M * T * 2);

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            // hA[i * K + j] = (half)(rand() % 1000 * 1 / 100 % 10 + 0.5);
            hA[i * K + j] = (half)(0.1);
        }
        for (int j = 0; j < N; ++j)
        {
            hC[i * N + j] = (float)(0);
            golden[i * N + j] = (float)(0);
        }
        for (int j = 0; j < T; ++j)
        {
            h_gemm1_result[i * T + j] = (float)(0);
            golden1[i * T + j] = (float)(0);
        }
    }

    for (int k = 0; k < K; ++k)
    {
        for (int n = 0; n < N; ++n)
        {
            // hB[n * K + k] = (half)(rand() % 1000 * 1 / 100 % 10 + 0.0);
            // hB[n * K + k] = (half)((n * K + k)*1e-3);
            hB[n * K + k] = (half)(0.1);
        } // K*N
    }


    for (int n = 0; n < N; ++n)
    {
        for (int t = 0; t < T; ++t)
        {
            // h_gemm1_weight[t * N + n] = (half)(rand() % 1000 * 1 / 100 % 10 + 0.5); 
            h_gemm1_weight[t * N + n] = (half)(0.1); 
        } // N*T
    }


#ifdef DEBUG
    std::cout << "Computing golden values...\n";
// simple tiling to make it a bit faster
#pragma omp parallel for
    for (int i = 0; i < M; i += 64)
    {
#pragma omp parallel for
        for (int j = 0; j < N; j += 64)
        {
            float accum[64 * 64] = {0};
            for (int k = 0; k < K; k += 32)
            {
                for (int kk = 0; kk < 32; ++kk)
                {
                    for (int jj = 0; jj < 64; ++jj)
                    {
                        for (int ii = 0; ii < 64; ++ii)
                        {
                            accum[ii * 64 + jj] += ((float)hA[(i + ii) * K + k + kk] * (float)hB[(j + jj) * K + k + kk]);
                        }
                    }
                }
            }
            for (int ii = 0; ii < 64; ++ii)
            {
                for (int jj = 0; jj < 64; ++jj)
                {
                    for (int kk = 0; kk < 64; ++kk)
                    {
                        golden[(i + ii) * N + j + jj] = (half)accum[ii * 64 + jj];
                    }
                }
            }
        }
    }
    std::cout << "Golden values done!\n";



#pragma omp parallel for
    for (int i = 0; i < M; i += 64)
    {
#pragma omp parallel for
        for (int j = 0; j < T; j += 64)
        { // 新的MNK对应的就是MTN
            float accum1[64 * 64] = {0};
            for (int k = 0; k < N; k += 32)
            {
                for (int kk = 0; kk < 32; ++kk)
                {
                    for (int jj = 0; jj < 64; ++jj)
                    {
                        for (int ii = 0; ii < 64; ++ii)
                        {
                            accum1[ii * 64 + jj] += ((float)golden[(i + ii) * N + k + kk] * (float)h_gemm1_weight[(j + jj) * N + k + kk]);
                        }
                    }
                }
            }
            for (int ii = 0; ii < 64; ++ii)
            {
                for (int jj = 0; jj < 64; ++jj)
                {
                    for (int kk = 0; kk < 64; ++kk)
                    {
                        golden1[(i + ii) * T + j + jj] = (half)accum1[ii * 64 + jj];
                    }
                }
            }
        }
    }
    std::cout << "Golden111 values done!\n";


#endif

    half *dA;
    half *dB;
    half *dC;
    half *d_gemm1_weight;
    half *d_gemm1_result;

    CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
    CUDA_CHECK(cudaMalloc(&dB, K * N * 2));
    CUDA_CHECK(cudaMalloc(&dC, M * N * 2));
    CUDA_CHECK(cudaMalloc(&d_gemm1_weight, N * T * 2));
    CUDA_CHECK(cudaMalloc(&d_gemm1_result, M * T * 2));

    CUDA_CHECK(cudaMemcpy(dA, hA, M * K * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, K * N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, M * N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gemm1_weight, h_gemm1_weight, T * N * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gemm1_result, h_gemm1_result, M * N * 2, cudaMemcpyHostToDevice));

    dim3 dimBlock(32, 2 * MULTI_THREADING, 2);
    dim3 dimGrid(N / 128, M / 128);
    // dim3 dimGrid(1, M / 128); // 增加这个是因为T维度上只需要一个block

#ifndef DEBUG
    int smem_size = MAX(STAGES * 128 * 32 * 2 * 2, 128 * 128 * 2+128 * 128 * 2);// 需要同时存GEMM0的结果和GEMM1的结果。读GEMM1的SMEM也许不需要那么大，以后可以reuse。
    if (smem_size >= (48 << 10))
    {
        CUDA_CHECK(cudaFuncSetAttribute(matmul,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }


// __global__ void matmul(half *A, half *B, half *gemm1_result, half *gemm1_weight, int M, int N, int K, int T, float alpha, float beta)


  dim3 cluster(1, 2, 1);
    void* kernel_params[] = {
        (void*)&dA, (void*)&dB, (void*)&d_gemm1_result, (void*)&d_gemm1_weight,
        (void*)&M, (void*)&N, (void*)&K, (void*)&T,
        (void*)&alpha, (void*)&beta
    };
  cudaLaunchConfig_t launch_config;
  launch_config.gridDim = dimGrid;
  launch_config.blockDim = dimBlock;
  launch_config.dynamicSmemBytes = smem_size;
  launch_config.stream = nullptr;

  cudaLaunchAttribute launch_attribute[1];
  launch_attribute[0].id = cudaLaunchAttributeClusterDimension;
  launch_attribute[0].val.clusterDim.x = cluster.x;
  launch_attribute[0].val.clusterDim.y = cluster.y;
  launch_attribute[0].val.clusterDim.z = cluster.z;

  launch_config.attrs = launch_attribute;
  launch_config.numAttrs = 1;
  void const* Matmul = (void const*)matmul;
  cudaError_t status = cudaFuncSetAttribute(
      Matmul, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  CUDA_CHECK(status);

//   status = cudaLaunchKernelExC(&launch_config, matmul, kernel_params);
//   cudaError_t launch_result = cudaGetLastError();
//   CUDA_CHECK(launch_result);




    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // // warmup
    // for (int i = 0; i < ITERS / 20 - 1; ++i)
    // {
    //     matmul<<<dimGrid, dimBlock, smem_size, nullptr>>>(dA, dB, dC, M, N, K, alpha, beta);
    // }
    cudaDeviceSynchronize();
    // auto start = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; ++i)
    {
        // matmul<<<dimGrid, dimBlock, smem_size, nullptr>>>(dA, dB, dC, M, N, K, alpha, beta);
        status = cudaLaunchKernelExC(&launch_config, Matmul, kernel_params);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


  cudaError_t launch_result = cudaGetLastError();
  CUDA_CHECK(launch_result);


    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Running cost of CUDA kernel is " << double(ms) / ITERS << "ms\n";
    std::cout << "TFLOPS: " << (float)M * N * K * 2 / (double(ms) / ITERS) * 1e3 / 1e12 << "\n";
    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // std::cout << "Running cost of CUDA kernel is " << duration.count() / 1e3 / 200.0 << "ms\n";
    // std::cout << "TFLOPS: " << (float)M * N * K * 2 / ((float)duration.count() / 1e3 / 200.0) * 1e3 / 1e12 << "\n";
#endif

#ifdef DEBUG
    // int smem_size = MAX(STAGES * 128 * 32 * 2 * 2, 128 * 128 * 4);
    // std::cout << "Using shared memory = " << (double)smem_size / 1e3 << " KB.\n";
    // if (smem_size >= (48 << 10))
    // {
    //     CUDA_CHECK(cudaFuncSetAttribute(matmul,
    //                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
    //                                     smem_size));
    // }
    // std::cout << "Computing result values...\n";
    // matmul<<<dimGrid, dimBlock, smem_size, nullptr>>>(dA, dB, dC, M, N, K, alpha, beta);
    // CUDA_CHECK(cudaGetLastError());
    // std::cout << "Computing results done!\n";
    // CUDA_CHECK(cudaMemcpy(hC, dC, M * N * 2, cudaMemcpyDeviceToHost));





    int smem_size = MAX(STAGES * 128 * 32 * 2 * 2, 128 * 128 * 2+128 * 128 * 2+128 * 128 * 2);// 这里增加了GEMM1的权重的空间。对于GEMM1，每次都要重新读取global的C值来累加。但是那不需要占用到SMEM，直接读到寄存器即可。
    std::cout << "Using shared memory = " << (double)smem_size / 1e3 << " KB.\n";
    if (smem_size >= (48 << 10))
    {
        CUDA_CHECK(cudaFuncSetAttribute(matmul,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    std::cout << "Computing result values...\n";
    matmul<<<dimGrid, dimBlock, smem_size, nullptr>>>(dA, dB, d_gemm1_result, d_gemm1_weight, M, N, K, T,alpha, beta);



//   dim3 cluster(1, 2, 1);
//     void* kernel_params[] = {
//         (void*)&dA, (void*)&dB, (void*)&d_gemm1_result, (void*)&d_gemm1_weight,
//         (void*)&M, (void*)&N, (void*)&K, (void*)&T,
//         (void*)&alpha, (void*)&beta
//     };
//   cudaLaunchConfig_t launch_config;
//   launch_config.gridDim = dimGrid;
//   launch_config.blockDim = dimBlock;
//   launch_config.dynamicSmemBytes = smem_size;
//   launch_config.stream = nullptr;

//   cudaLaunchAttribute launch_attribute[1];
//   launch_attribute[0].id = cudaLaunchAttributeClusterDimension;
//   launch_attribute[0].val.clusterDim.x = cluster.x;
//   launch_attribute[0].val.clusterDim.y = cluster.y;
//   launch_attribute[0].val.clusterDim.z = cluster.z;

//   launch_config.attrs = launch_attribute;
//   launch_config.numAttrs = 1;
//   void const* Matmul = (void const*)matmul;
//   cudaError_t status = cudaFuncSetAttribute(
//       Matmul, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
//   CUDA_CHECK(status);


//     status = cudaLaunchKernelExC(&launch_config, Matmul, kernel_params);


//     CUDA_CHECK(cudaGetLastError());
//     std::cout << "Computing results done!\n";
//     CUDA_CHECK(cudaMemcpy(h_gemm1_result, d_gemm1_result, M * T * 2, cudaMemcpyDeviceToHost));





#ifdef PRINT
    std::cout << "Golden:" << std::endl;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < T; ++j)
        {
            std::cout << (float)golden1[i * T + j] << " ";
        }
        std::cout << std::endl;
    }

    // std::cout << "Results:" << std::endl;
    // for (int i = 0; i < M; ++i)
    // {
    //     for (int j = 0; j < N; ++j)
    //     {
    //         std::cout << (float)hC[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }



    std::cout << "Results:" << std::endl;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < T; ++j)
        {
            std::cout << (float)h_gemm1_result[i * T + j] << " ";
        }
        std::cout << std::endl;
    }


    // std::cout << "MatrixA:" << std::endl;
    // for (int i = 0; i < M; ++i)
    // {
    //     for (int j = 0; j < K; ++j)
    //     {
    //         std::cout << (float)hA[i * K + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
#endif

    int errors = 0;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float diff = ((float)golden1[i * N + j] - (float)h_gemm1_result[i * N + j]);
            if (diff < 0)
            {
                diff = -diff;
            }
            float maxv = MAX((float)golden1[i * N + j], (float)h_gemm1_result[i * N + j]);
            if (maxv < 0)
            {
                maxv = -maxv;
            }
            if (diff / maxv > 1e-2)
            {
                errors += 1;
            }
        }
    }

    if (errors)
    {
        std::cout << "Wrong Answer! " << errors << " errors.\n";
    }
    else
    {
        std::cout << "Correctness Check Passed!\n";
    }
#endif

    free(hA);
    free(hB);
    free(hC);
    free(golden);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}

