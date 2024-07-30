#include <helper_cuda.h>
#include <cooperative_groups.h>
#include "histogram_common.h"

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(uint *bins, const uint nbins, const uint bins_per_block, uint *d_Data, uint array_size)
{
  extern __shared__ uint smem[];
  namespace cg = cooperative_groups;
  int tid = cg::this_grid().thread_rank();

  // Cluster initialization, size and calculating local bin offsets.
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    smem[i] = 0; //Initialize shared memory histogram to zeros
  }

  // cluster synchronization ensures that shared memory is initialized to zero in
  // all thread blocks in the cluster. It also ensures that all thread blocks
  // have started executing and they exist concurrently.
  cluster.sync();

  int j = 0;
  uint ldatas = 0;
  uint ldata = 0;
  uint binid = 0;
  uint dst_block_rank = 0;
  uint dst_offset = 0;
  uint *dst_smem = 0;
  for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
  {
    ldatas = d_Data[i];

    for (j = 0; j < 32; j+=8){
      //Find the right histogram bin.
      ldata = (ldatas >> j) & 0xFFU;
      binid = ldata;
      if (ldata < 0)
        binid = 0;
      else if (ldata >= nbins)
        binid = nbins - 1;

      // if ((blockIdx.x == 0) && (threadIdx.x == 0))
      //     printf("%u\n", binid);
      //Find destination block rank and offset for computing
      //distributed shared memory histogram
      dst_block_rank = (uint)(binid / bins_per_block);
      dst_offset = binid % bins_per_block;

      //Pointer to target block shared memory
      dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

      //Perform atomic update of the histogram bin
      atomicAdd(dst_smem + dst_offset, 1);
    }
  }

  // cluster synchronization is required to ensure all distributed shared
  // memory operations are completed and no thread block exits while
  // other thread blocks are still accessing distributed shared memory
  cluster.sync();

  // Perform global memory histogram, using the local distributed memory histogram
  uint *lbins = bins + cluster.block_rank() * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&lbins[i], smem[i]);
  }
}

extern "C" void histogram_dsm(uint *d_Histogram, void *d_Data, uint byteCount) {
  uint array_size = byteCount / sizeof(uint);
  uint threads_per_block = 128;
  uint nbins = 2048;
  uint *bins = d_Histogram;

  cudaLaunchConfig_t config = {0};
  config.gridDim = array_size / threads_per_block;
  config.blockDim = threads_per_block;

  // cluster_size depends on the histogram size.
  // ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
  int cluster_size = 8; // size 2 is an example here
  uint nbins_per_block = nbins / cluster_size;

  //dynamic shared memory size is per block.
  //Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
  config.dynamicSmemBytes = nbins_per_block * sizeof(uint);

  // CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));
  cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, (uint*)d_Data, array_size);
  getLastCudaError("clusterHist_kernel() execution failed\n");
}
