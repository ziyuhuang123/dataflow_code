#include <helper_cuda.h>
#include <cooperative_groups.h>
#include "dsm_common.h"


#ifndef ILP
#define ILP 8
#endif

#ifndef BS
#define BS 1024
#endif

const int MAX_ILP = 32;
const int BLOCK_NUM = 114 * 220 * 2;
const int REPEAT = 1000;
const int CLUSTER_SIZE = 2;
const int THREADS_PER_BLOCK = BS;

// Distributed Shared memory histogram kernel
__global__ void dsm_sm2sm_thrpt_kernel(uint *d_Data)
{
    extern __shared__ uint smem[];
    namespace cg = cooperative_groups;
    int tid = threadIdx.x;

    // uint32_t smid, cluster_id;
    // asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    // asm volatile("mov.u32 %0, %clusterid.x;" : "=r"(cluster_id));
    // // uint32_t ctaid;
    // // asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(ctaid));
    // if (tid == 0) {
    //     printf("%d, %d, %d\n", blockIdx.x, smid, cluster_id);
    // }


    // Cluster initialization, size and calculating local bin offsets.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    // int cluster_size = cluster.dim_blocks().x;

    for (int j = 0; j < MAX_ILP; j++)
        smem[j * THREADS_PER_BLOCK + tid] = tid; // Initialize shared memory histogram to zeros

    cluster.sync();

    uint dst_block_rank = 0;
    uint *dst_smem = 0;

    dst_block_rank = (clusterBlockRank + 1) % CLUSTER_SIZE;
    // dst_block_rank = (clusterBlockRank / 2) * 2 + (clusterBlockRank + 1) % 2;
    dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    register uint temp[MAX_ILP] = {0};

    #pragma unroll 1
    for (uint32_t i = 0; i < REPEAT; i++)
        #pragma unroll
        for (uint32_t j = 0; j < ILP; j++) {
            temp[j] += dst_smem[(j * blockDim.x + tid + i * 32) % (blockDim.x * MAX_ILP)];
            // temp[j] += dst_smem[(j * 64 + tid + i * 32) % 64];
        }



    cluster.sync();

    d_Data[tid] = temp[tid % MAX_ILP];
}

extern "C" void dsm_sm2sm_thrpt(void *d_Data, uint arraySize)
{
    cudaLaunchConfig_t config = {0};
    // config.gridDim = array_size / threads_per_block;
    config.gridDim = BLOCK_NUM;
    config.blockDim = THREADS_PER_BLOCK;
    // printf("blockDim: %d, ILP: %d\n", THREADS_PER_BLOCK, ILP);
    int cluster_size = CLUSTER_SIZE; // size 2 is an example here

    // dynamic shared memory size is per block.
    config.dynamicSmemBytes = THREADS_PER_BLOCK * sizeof(uint) * MAX_ILP;

    // CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));
    cudaFuncSetAttribute((void *)dsm_sm2sm_thrpt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);
    cudaFuncSetAttribute((void *)dsm_sm2sm_thrpt_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 0);
    cudaFuncSetAttribute((void *)dsm_sm2sm_thrpt_kernel, cudaFuncAttributeClusterSchedulingPolicyPreference, 0);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    // int number_clusters, potential_cluster_size;
    // checkCudaErrors(cudaOccupancyMaxActiveClusters(&number_clusters, dsm_sm2sm_thrpt_kernel, &config));
    // printf("number_clusters: %d\n", number_clusters);
    // checkCudaErrors(cudaOccupancyMaxPotentialClusterSize(&potential_cluster_size, dsm_sm2sm_thrpt_kernel, &config));
    // printf("potential_cluster_size: %d\n", potential_cluster_size);


    cudaLaunchKernelEx(&config, dsm_sm2sm_thrpt_kernel, (uint *)d_Data);
    getLastCudaError("dsm_sm2sm_thrpt_kernel() execution failed\n");
}
