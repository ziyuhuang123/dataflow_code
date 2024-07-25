// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <cooperative_groups.h>
// Project include
#include "dsm_common.h"

// Distributed Shared memory histogram kernel
__global__ void dsm_sm2sm_lat_kernel(uint *d_Data, uint array_size)
{
    extern __shared__ uint smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    clock_t start, end;
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    // Cluster initialization, size and calculating local bin offsets.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;
    
    smem[0] = blockIdx.x;

    // Cluster synchronization ensures that shared memory is initialized to zero in
    // all thread blocks in the cluster. It also ensures that all thread blocks
    // have started executing and they exist concurrently.
    cluster.sync();
    if ((threadIdx.x == 0))
        printf("1: smid:%u, rank:%u: %u\n", smid, clusterBlockRank, smem[0]);

    uint dst_block_rank = 0;
    uint *dst_smem = 0;

    // My version start
    for (int i = 0; i < array_size; i++)
    {
        smem[i] = i + 1;
    }
    uint index = 0;
    int dst_block_rank_list[4];
    for (int i = 0; i < 4; ++i) {
        dst_block_rank_list[i] = (i + clusterBlockRank) % 4;
    }
    dst_block_rank_list[4]={1, 2, 3, 0};


    for(int ii=0;ii<4;ii++){
        index=0; // 初始化一下，不然每次访问index的位置就乱了。
        // dst_block_rank = clusterBlockRank ? 0 : 1;
        dst_block_rank = dst_block_rank_list[ii];
        dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
        start = clock();
        for (int i = 0; i < array_size; i++)
        {
            index = dst_smem[index];
        }
        end = clock();

        d_Data[0] += index;
        // My version end

        // Cluster synchronization is required to ensure all distributed shared
        // memory operations are completed and no thread block exits while
        // other thread blocks are still accessing distributed shared memory
        cluster.sync();

        // if ((threadIdx.x == 0&&clusterBlockRank==0))
        //     printf("2: smid:%u, rank:%u: %u\n", smid, clusterBlockRank, smem[0]);

        atomicAdd(d_Data, smem[0]);

        if ((threadIdx.x == 0))
            printf("3: smid:%u, rank:%u: %ld-%ld:%ld. Block%d to block%d\n", smid, clusterBlockRank, start, end, (end - start) / array_size, clusterBlockRank, dst_block_rank);
    }

}

extern "C" void dsm_sm2sm_latency(void *d_Data, uint arraySize)
{
    uint threads_per_block = 1;

    cudaLaunchConfig_t config = {0};
    config.gridDim = 4;
    config.blockDim = threads_per_block;

    int cluster_size = 4;

    config.dynamicSmemBytes = arraySize * sizeof(uint);

    cudaFuncSetAttribute((void *)dsm_sm2sm_lat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    cudaLaunchKernelEx(&config, dsm_sm2sm_lat_kernel, (uint *)d_Data, arraySize);
    getLastCudaError("dsm_sm2sm_lat_kernel() execution failed\n");
}

const int numRuns = 16;

int main(int argc, char **argv) {
    uint *d_Data;
    uint arraySize = 1024;
    StopWatchInterface *hTimer = NULL;
    int PassFailFlag = 1;
    uint uiSizeMult = 1;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    float sm_clock_rate = deviceProp.clockRate * 1000.0f; // 转换为Hz

    sdkCreateTimer(&hTimer);

    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult")) {
        uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
        uiSizeMult = MAX(1, MIN(uiSizeMult, 10));
        arraySize *= uiSizeMult;
    }

    checkCudaErrors(cudaMalloc((void **)&d_Data, arraySize * sizeof(uint)));

    {
        printf("Measure latency of SM to SM for %u bytes (%u runs)...\n\n", arraySize, numRuns);

        printf("Benchmarking time...\n");
        for (int iter = -1; iter < numRuns; iter++) {
            if (iter == 0) {
                checkCudaErrors(cudaDeviceSynchronize());
                sdkResetTimer(&hTimer);
                sdkStartTimer(&hTimer);
            }

            dsm_sm2sm_latency(d_Data, arraySize);
        }

        cudaDeviceSynchronize();
        sdkStopTimer(&hTimer);
        double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
        double cycles = dAvgSecs * sm_clock_rate;
        printf("dsm_sm2sm_latency() time (average) : %.5f sec, %.0f cycles\n\n", dAvgSecs, cycles);
    }

    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cudaFree(d_Data));
}
