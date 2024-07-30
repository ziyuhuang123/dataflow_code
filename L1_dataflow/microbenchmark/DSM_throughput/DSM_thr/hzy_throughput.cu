// nvcc hzy_throughput.cu -o hzy_throughput -arch=sm_90 -std=c++17 -I./Common

// ./hzy_throughput --nbins 2048 --array_size 200000000 --threads_per_block 256 --cluster_size 4


#include <iostream>
#include <cstdlib>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cooperative_groups.h>

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input, size_t array_size)
{
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    // Cluster initialization, size and calculating local bin offsets.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        smem[i] = 0; // Initialize shared memory histogram to zeros
    }

    // cluster synchronization ensures that shared memory is initialized to zero in
    // all thread blocks in the cluster. It also ensures that all thread blocks
    // have started executing and they exist concurrently.
    cluster.sync();

    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
    {
        int ldata = input[i];

        // Find the right histogram bin.
        int binid = ldata;
        if (ldata < 0)
            binid = 0;
        else if (ldata >= nbins)
            binid = nbins - 1;

        // Find destination block rank and offset for computing
        // distributed shared memory histogram
        int dst_block_rank = (int)(binid / bins_per_block);
        int dst_offset = binid % bins_per_block;

        // Pointer to target block shared memory
        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

        // Perform atomic update of the histogram bin
        atomicAdd(dst_smem + dst_offset, 1);
    }

    // cluster synchronization is required to ensure all distributed shared
    // memory operations are completed and no thread block exits while
    // other thread blocks are still accessing distributed shared memory
    cluster.sync();

    // Perform global memory histogram, using the local distributed memory histogram
    int *lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        atomicAdd(&lbins[i], smem[i]);
    }
}

void launch_clusterHist_kernel(int *bins, const int nbins, const int *__restrict__ input, size_t array_size, int threads_per_block, int cluster_size)
{
    cudaLaunchConfig_t config = {0};
    config.gridDim = array_size / threads_per_block;
    config.blockDim = threads_per_block;

    // cluster_size depends on the histogram size.
    // ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
    
    int nbins_per_block = nbins / cluster_size;

    // dynamic shared memory size is per block.
    // Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
    config.dynamicSmemBytes = nbins_per_block * sizeof(int);

    cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);
    cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;


    cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size); 
}

int main(int argc, char **argv)
{
    // 默认值
    int nbins = 1024;
    size_t array_size = 1e8;
    int threads_per_block = 512;
    int cluster_size = 2;

    // 解析命令行参数
    for (int i = 1; i < argc; i++)
    {
        if (std::string(argv[i]) == "--nbins" && i + 1 < argc)
        {
            nbins = std::atoi(argv[++i]);
        }
        else if (std::string(argv[i]) == "--array_size" && i + 1 < argc)
        {
            array_size = std::atol(argv[++i]);
        }
        else if (std::string(argv[i]) == "--threads_per_block" && i + 1 < argc)
        {
            threads_per_block = std::atoi(argv[++i]);
        }
        else if (std::string(argv[i]) == "--cluster_size" && i + 1 < argc)
        {
            cluster_size = std::atoi(argv[++i]);
        }
    }

    int *d_bins;
    int *d_input;

    // Allocate memory
    checkCudaErrors(cudaMalloc((void **)&d_bins, nbins * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_input, array_size * sizeof(int)));

    // Initialize input data (for example purposes, we initialize it with random values)
    int *h_input = new int[array_size];
    for (int i = 0; i < array_size; ++i)
    {
        h_input[i] = rand() % nbins;
    }

    checkCudaErrors(cudaMemcpy(d_input, h_input, array_size * sizeof(int), cudaMemcpyHostToDevice));





    // 创建计时器
    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);

    // 记录开始时间
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    // 启动内核
    launch_clusterHist_kernel(d_bins, nbins, d_input, array_size, threads_per_block, cluster_size);

    // 记录结束时间
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);

    // 计算总时长
    double total_time_sec = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
    printf("Average time per iteration (seconds): %.5f sec\n", total_time_sec);



    // 计算吞吐量 (百万元素每秒)
    double throughput = (static_cast<double>(array_size) / total_time_sec) / 1e9;
    printf("Throughput (Giga Elements/second): %.5f Gelem/s\n", throughput);
    // 输出其他参数信息
    printf("Threads per block: %d\n", threads_per_block);
    printf("Cluster size: %d\n", cluster_size);
    printf("Array size: %d\n", array_size);
    printf("Number of bins: %d\n", nbins);




    // Clean up
    checkCudaErrors(cudaFree(d_bins));
    checkCudaErrors(cudaFree(d_input));
    delete[] h_input;

    return 0;
}