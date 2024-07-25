#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)

__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input, size_t array_size)
{
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        smem[i] = 0;
    }

    cluster.sync();

    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
    {
        int ldata = input[i];
        int binid = ldata;
        if (ldata < 0)
            binid = 0;
        else if (ldata >= nbins)
            binid = nbins - 1;

        int dst_block_rank = (int)(binid / bins_per_block);
        int dst_offset = binid % bins_per_block;
        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

        atomicAdd(dst_smem + dst_offset, 1);
    }

    cluster.sync();

    int *lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        atomicAdd(&lbins[i], smem[i]);
    }
}

void run_histogram(const int *input, size_t array_size, int nbins)
{
    int *d_bins, *d_input;
    size_t bins_size = nbins * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_bins, bins_size));
    CUDA_CHECK(cudaMemset(d_bins, 0, bins_size));
    CUDA_CHECK(cudaMalloc(&d_input, array_size * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, input, array_size * sizeof(int), cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks_per_grid = (array_size + threads_per_block - 1) / threads_per_block;
    int cluster_size = 2;
    int nbins_per_block = nbins / cluster_size;

    cudaLaunchConfig_t config = {0};
    config.gridDim = blocks_per_grid;
    config.blockDim = threads_per_block;
    config.dynamicSmemBytes = nbins_per_block * sizeof(int);

    CUDA_CHECK(cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    cudaLaunchKernelEx(&config, clusterHist_kernel, d_bins, nbins, nbins_per_block, d_input, array_size);

    int *h_bins = (int *)malloc(bins_size);
    CUDA_CHECK(cudaMemcpy(h_bins, d_bins, bins_size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < nbins; i++)
    {
        printf("Bin %d: %d\n", i, h_bins[i]);
    }

    free(h_bins);
    CUDA_CHECK(cudaFree(d_bins));
    CUDA_CHECK(cudaFree(d_input));
}

int main()
{
    int input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    size_t array_size = sizeof(input) / sizeof(input[0]);
    int nbins = 10;
    run_histogram(input, array_size, nbins);
    return 0;
}
