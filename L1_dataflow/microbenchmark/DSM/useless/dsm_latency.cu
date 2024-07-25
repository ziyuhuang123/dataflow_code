#include <helper_cuda.h>
#include <cooperative_groups.h>
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

    // cluster synchronization ensures that shared memory is initialized to zero in
    // all thread blocks in the cluster. It also ensures that all thread blocks
    // have started executing and they exist concurrently.
    cluster.sync();
    if ((threadIdx.x == 0))
        printf("smid:%u, rank:%u: %u\n", smid, clusterBlockRank, smem[0]);

    uint dst_block_rank = 0;
    uint *dst_smem = 0;

    // uint temp1 = 0, temp2 = 0;
    // if (clusterBlockRank % 2 == 1)
    // {
    //     // distributed shared memory histogram
    //     dst_block_rank = 0;
    //     // dst_offset = binid % bins_per_block;

    //     // Pointer to target block shared memory
    //     dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    //     start = clock();
    //     // Perform atomic update of the histogram bin
    //     //  if (threadIdx.x == 0)
    //     temp1 += dst_smem[0];
    //     end = clock();
    // }

    // my version start

    for (int i = 0; i < array_size; i++)
    {
        smem[i] = i + 1;
    }
    uint index = 0;
    dst_block_rank = clusterBlockRank ? 0 : 1;
    dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
    start = clock();
    for (int i = 0; i < array_size; i++)
    {
        index = dst_smem[index];
    }
    end = clock();

    d_Data[0] += index;
    // my version end

    // cluster synchronization is required to ensure all distributed shared
    // memory operations are completed and no thread block exits while
    // other thread blocks are still accessing distributed shared memory
    cluster.sync();

    if ((threadIdx.x == 0))
        printf("smid:%u, rank:%u: %u\n", smid, clusterBlockRank, smem[0]);

    atomicAdd(d_Data, smem[0]);


    if ((threadIdx.x == 0))
        printf("smid:%u, rank:%u: %ld-%ld:%ld\n", smid, clusterBlockRank, start, end, (end - start)/array_size );
}

extern "C" void dsm_sm2sm_latency(void *d_Data, uint arraySize)
{
    // uint array_size = byteCount / sizeof(uint);
    uint threads_per_block = 1;

    cudaLaunchConfig_t config = {0};
    // config.gridDim = array_size / threads_per_block;
    config.gridDim = 2;
    config.blockDim = threads_per_block;

    // cluster_size depends on the histogram size.
    // ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
    int cluster_size = 2; // size 2 is an example here

    // dynamic shared memory size is per block.
    // Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
    config.dynamicSmemBytes = arraySize * sizeof(uint);

    // CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));
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
