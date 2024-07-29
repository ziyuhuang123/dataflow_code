#include <helper_cuda.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include "dsm_common.h"


__global__ void dsm_sm2sm_thrpt_kernel(int int_number_per_thread_will_take, int THREADS_PER_BLOCK, int CLUSTER_SIZE)
{
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = threadIdx.x;

    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();

    for (int j = 0; j < int_number_per_thread_will_take; j++)
        smem[j * THREADS_PER_BLOCK + tid] = tid; // Initialize shared memory histogram to zeros

    unsigned int dst_block_rank = (clusterBlockRank + 1) % CLUSTER_SIZE;
    int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
    // int *temp = new int[int_number_per_thread_will_take];
    int temp[16] = {0};
    cluster.sync();

    #pragma unroll
    for (int i = 0; i < int_number_per_thread_will_take; i++) {
        temp[i] = dst_smem[i * 1024 + tid];
    }
    cluster.sync();
}

extern "C" void dsm_sm2sm_thrpt_wrapper(int int_number_per_thread_will_take, int THREADS_PER_BLOCK, int CLUSTER_SIZE, int BLOCK_NUM)
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


    cudaLaunchKernelEx(&config, dsm_sm2sm_thrpt_kernel, int_number_per_thread_will_take, THREADS_PER_BLOCK, CLUSTER_SIZE);

    getLastCudaError("dsm_sm2sm_thrpt_kernel() execution failed\n");
}

int main(int argc, char **argv) {
    int BLOCK_NUM = 132 * 100;
    int CLUSTER_SIZE = 16;
    int THREADS_PER_BLOCK = 1024;
    int numRuns = 2000;
    int int_number_per_thread_will_take = 128 * 128 / THREADS_PER_BLOCK;
    // 解析命令行参数
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "--block_num" && i + 1 < argc)
        {
            BLOCK_NUM = std::atoi(argv[++i]);
        }
        else if (std::string(argv[i]) == "--cluster_size" && i + 1 < argc)
        {
            CLUSTER_SIZE = std::atoi(argv[++i]);
        }
        else if (std::string(argv[i]) == "--threads_per_block" && i + 1 < argc)
        {
            THREADS_PER_BLOCK = std::atoi(argv[++i]);
        }
        else if (std::string(argv[i]) == "--num_runs" && i + 1 < argc)
        {
            numRuns = std::atoi(argv[++i]);
        }
    }

    printf("Parameters:\n");
    printf("BLOCK_NUM: %d\n", BLOCK_NUM);
    printf("CLUSTER_SIZE: %d\n", CLUSTER_SIZE);
    printf("THREADS_PER_BLOCK: %d\n", THREADS_PER_BLOCK);
    printf("numRuns: %d\n", numRuns);
    printf("int_number_per_thread_will_take: %d\n", int_number_per_thread_will_take);

    StopWatchInterface *hTimer = NULL;
    int total_shared_memory_kb = THREADS_PER_BLOCK * int_number_per_thread_will_take * sizeof(int) / 1024;
    if (total_shared_memory_kb > 227) {
        printf("Error: Total shared memory requested (%d KB) exceeds the limit of 227 KB\n", total_shared_memory_kb);
        return -1;
    } // 总共享内存需求量不能超过227KB


    sdkCreateTimer(&hTimer);


    for (int iter = -1; iter < numRuns; iter++) {
        if (iter == 0) {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }
        dsm_sm2sm_thrpt_wrapper(int_number_per_thread_will_take, THREADS_PER_BLOCK, CLUSTER_SIZE, BLOCK_NUM);
        // dsm_sm2sm_thrpt_wrapper();
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&hTimer);
    double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
    printf("dsm_sm2sm_thrpt() time (average) : %.5f sec\n", dAvgSecs);

    // 计算吞吐量
    double data_transfer_per_cluster = 128*128 * sizeof(int);
    double total_data_transfer = BLOCK_NUM * data_transfer_per_cluster;
    double throughput = (total_data_transfer / dAvgSecs) / (1024.0 * 1024.0 * 1024.0 * 1024.0); // 单位：TB/s

    printf("Throughput: %.5f TB/s\n", throughput);

    sdkDeleteTimer(&hTimer);

    return 0;
}
