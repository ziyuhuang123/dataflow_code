// nvcc simple_thr_ld.cu -o simple_thr_ld -arch=sm_90 -std=c++17 -I./Common

// ./simple_thr_ld

#include <helper_cuda.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include "dsm_common.h"

const int BLOCK_NUM = 16;
const int CLUSTER_SIZE = 16;
const int THREADS_PER_BLOCK = 512;
const int numRuns = 2000;
const int int_number_per_thread_will_take = 110;


__global__ void dsm_sm2sm_thrpt_kernel(int *d_Data, long long *d_times, int run_id)
{
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = threadIdx.x;

    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();

    for (int j = 0; j < int_number_per_thread_will_take; j++)
        smem[j * THREADS_PER_BLOCK + tid] = tid; // Initialize shared memory histogram to zeros

    int dst_block_rank_list[CLUSTER_SIZE];
    for (int i = 0; i < CLUSTER_SIZE; ++i) {
        dst_block_rank_list[i] = (i + clusterBlockRank) % CLUSTER_SIZE;
    }  // 当cluster_rank=0, 这里得到0,1,2,3
    // if (threadIdx.x == 0 && clusterBlockRank == 1) {
    //     printf("dst_block_rank_list: ");
    //     for(int j = 0; j < CLUSTER_SIZE; ++j){
    //         printf("%d ", dst_block_rank_list[j]);
    //     }
    //     printf("\n");
    // }

    for (int ii = 0; ii < CLUSTER_SIZE; ii++) {
        unsigned int dst_block_rank = dst_block_rank_list[ii];
        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

        int temp[int_number_per_thread_will_take] = {0};
        cluster.sync();
        clock_t start, end;
        start = clock();

        #pragma unroll
        for (int i = 0; i < int_number_per_thread_will_take; i++) {
            temp[i] = dst_smem[i * 1024 + tid];
        }
        cluster.sync();
        end = clock();

        if (threadIdx.x == 0 && clusterBlockRank == 0 && blockIdx.x == 0) {
            d_times[run_id * CLUSTER_SIZE + ii] = end - start;
        } 
    }
}

extern "C" void dsm_sm2sm_thrpt_wrapper(void *d_Data, long long *d_times, uint arraySize, int run_id)
{
    cudaLaunchConfig_t config = {0};
    config.gridDim = BLOCK_NUM;
    config.blockDim = THREADS_PER_BLOCK;
    int cluster_size = CLUSTER_SIZE;

    config.dynamicSmemBytes = 227 * 1024; // 这里最大只能是227.不能是228

    cudaFuncSetAttribute((void *)dsm_sm2sm_thrpt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);
    cudaFuncSetAttribute((void *)dsm_sm2sm_thrpt_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute((void *)dsm_sm2sm_thrpt_kernel, cudaFuncAttributeClusterSchedulingPolicyPreference, 0);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    cudaLaunchKernelEx(&config, dsm_sm2sm_thrpt_kernel, (int *)d_Data, (long long *)d_times, run_id);
    getLastCudaError("dsm_sm2sm_thrpt_kernel() execution failed\n");
}

int main(int argc, char **argv) {
    int *d_Data;
    long long *d_times, *h_times;
    uint arraySize = 1024;
    StopWatchInterface *hTimer = NULL;
    uint uiSizeMult = 1;


    int total_shared_memory_kb = THREADS_PER_BLOCK * int_number_per_thread_will_take * sizeof(int) / 1024;
    if (total_shared_memory_kb > 227) {
        printf("Error: Total shared memory requested (%d KB) exceeds the limit of 227 KB\n", total_shared_memory_kb);
        return -1;
    } // 总共享内存需求量不能超过227KB


    sdkCreateTimer(&hTimer);

    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult")) {
        uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
        uiSizeMult = MAX(1, MIN(uiSizeMult, 10));
        arraySize *= uiSizeMult;
    }

    checkCudaErrors(cudaMalloc((void **)&d_Data, arraySize * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_times, numRuns * CLUSTER_SIZE * sizeof(long long)));
    h_times = (long long *)malloc(numRuns * CLUSTER_SIZE * sizeof(long long));

    int device;
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    double gpuFrequencyHz = prop.clockRate * 1000.0; // clockRate 是以 kHz 为单位的
    for (int iter = -1; iter < numRuns; iter++) {
        if (iter == 0) {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }

        dsm_sm2sm_thrpt_wrapper(d_Data, d_times, arraySize, iter);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&hTimer);
    double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
    printf("dsm_sm2sm_thrpt() time (average) : %.5f sec\n", dAvgSecs);

    checkCudaErrors(cudaMemcpy(h_times, d_times, numRuns * CLUSTER_SIZE * sizeof(long long), cudaMemcpyDeviceToHost));
    long long total_time[CLUSTER_SIZE] = {0};
    printf("Times per iteration (clock cycles):\n");
    for (int i = 0; i < numRuns * CLUSTER_SIZE; i++) {
        total_time[i % CLUSTER_SIZE] += h_times[i];
        if (i < 10) {
            printf("Iteration %d: %lld\n", i, h_times[i]);
        }
    }

    double avg_time[CLUSTER_SIZE];
    for (int i = 0; i < CLUSTER_SIZE; i++) {
        avg_time[i] = (double)total_time[i] / numRuns;
        printf("Average time for block0-to-block%d: %f clock cycles\n", i, avg_time[i]);
    }
    // 计算和输出带宽
    double data_size_kb = 128*128 * sizeof(int) / 1024.0;
    double data_size_gb = data_size_kb / 1024.0 / 1024.0;
    printf("Data size: %f GB\n", data_size_gb);
    for (int i = 0; i < CLUSTER_SIZE; i++) {
        double avg_time_sec = avg_time[i] / gpuFrequencyHz;
        double throughput_gbps = data_size_gb / avg_time_sec;
        double throughput_tbps = throughput_gbps / 1024.0;
        printf("Throughput for block0-to-block %d: %f GB/s\n", i, throughput_gbps);
        // printf("Throughput for block0-to-block %d: %f TB/s\n", i, throughput_tbps);
    }
    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cudaFree(d_Data));
    checkCudaErrors(cudaFree(d_times));
    free(h_times);

    return 0;
}
