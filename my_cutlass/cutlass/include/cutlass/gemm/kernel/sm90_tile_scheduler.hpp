/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once
#include "cutlass/gemm/kernel/static_tile_scheduler.hpp"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace cutlass::gemm::kernel::detail {

///////////////////////////////////////////////////////////////////////////////



// Persistent Thread Block (TB) scheduler
class PersistentTileSchedulerSm90:
public StaticPersistentTileScheduler<PersistentTileSchedulerSm90> {

  using BaseScheduler = StaticPersistentTileScheduler<PersistentTileSchedulerSm90>;
public:
  using StaticPersistentTileScheduler::StaticPersistentTileScheduler;
  using Params = PersistentTileSchedulerSm90Params;
  using RasterOrder = typename Params::RasterOrder;
  using RasterOrderOptions = typename Params::RasterOrderOptions;
  using Arguments = BaseScheduler::Arguments;

  // get work_idx_m, work_idx_n from blk_per_grid_dim while applying swizzle
  static CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cluster_blk_major,
      int32_t log_swizzle_size,
      RasterOrder raster_order) {
    auto [cta_m_in_cluster, cta_n_in_cluster, _] = cute::block_id_in_cluster();
    return get_work_idx_m_and_n(
      blk_per_grid_dim,
      divmod_cluster_shape_major,
      divmod_cluster_shape_minor,
      divmod_cluster_blk_major,
      log_swizzle_size,
      raster_order,
      cta_m_in_cluster,
      cta_n_in_cluster
    );
  }

  static CUTLASS_DEVICE
  cute::tuple<int32_t, int32_t>
  get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim,
      FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      FastDivmodU64 const& divmod_cluster_blk_major,
      int32_t log_swizzle_size,
      RasterOrder raster_order,
      uint64_t cta_m_in_cluster,
      uint64_t cta_n_in_cluster) {

    uint64_t cluster_id, cluster_major_offset = 0, cluster_minor_offset = 0;
    divmod_cluster_shape_major(cluster_id, cluster_major_offset, blk_per_grid_dim);

    if (raster_order == RasterOrder::AlongN) {
      cluster_minor_offset = cta_m_in_cluster;
    }
    else {
      cluster_minor_offset = cta_n_in_cluster;
    }

    uint64_t cluster_idx_minor, cluster_idx_major;

    uint64_t cluster_idx_minor_div_swizzle, extra, offset;

    offset = cluster_id & ((1 << log_swizzle_size) - 1);
    extra = cluster_id >> log_swizzle_size;

    divmod_cluster_blk_major(cluster_idx_minor_div_swizzle, cluster_idx_major, extra);

    cluster_idx_minor = cluster_idx_minor_div_swizzle * (1 << log_swizzle_size) + offset;

    auto minor_work_idx = static_cast<int32_t>(cluster_idx_minor * divmod_cluster_shape_minor.divisor + cluster_minor_offset);
    // 0~4 * 1 + 0

    auto major_work_idx = static_cast<int32_t>(cluster_idx_major * divmod_cluster_shape_major.divisor + cluster_major_offset);

    // if (blockIdx.x == 0 && blockIdx.y == 20 && threadIdx.x == 0 && threadIdx.y == 0) {
    //   printf("\n");
    //   printf("blk_per_grid_dim: %llu, cluster_id: %llu, cluster_major_offset: %llu, cta_m_in_cluster: %llu, cta_n_in_cluster: %llu, cluster_minor_offset: %llu, cluster_idx_minor_div_swizzle: %llu, cluster_idx_major: %llu, offset: %llu, extra: %llu, cluster_idx_minor: %llu, minor_work_idx: %d, major_work_idx: %d, log_swizzle_size: %d, minor_work_idx %d=%llu * %llu + %llu, major_work_idx %d=%llu * %llu + %llu\n", blk_per_grid_dim, cluster_id, cluster_major_offset, cta_m_in_cluster, cta_n_in_cluster, cluster_minor_offset, cluster_idx_minor_div_swizzle, cluster_idx_major, offset, extra, cluster_idx_minor, minor_work_idx, major_work_idx, log_swizzle_size, minor_work_idx, cluster_idx_minor, divmod_cluster_shape_minor.divisor, cluster_minor_offset, major_work_idx, cluster_idx_major, divmod_cluster_shape_major.divisor, cluster_major_offset);
    //   printf("\n");
    // }

    if (raster_order == RasterOrder::AlongN) {
      return {minor_work_idx, major_work_idx};
    }
    else {
      return {major_work_idx, minor_work_idx};
    }
  }

  // The basic tile scheduler does not require any additional workspace
  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, const uint32_t = 1) {
    return 0;
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(Arguments const&, void*, cudaStream_t, ProblemShape, KernelHardwareInfo const&,
    uint32_t, const uint32_t = 1, CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  // Kernel helper function to get next work tile
  CUTLASS_DEVICE
  auto
  fetch_next_work(WorkTileInfo work_tile_info) {
    if (continue_current_work(work_tile_info)) {
      // if(blockIdx.x==0&&blockIdx.y==0&&threadIdx.x==0&&threadIdx.y==0){
      //   printf("enter continue_current_work(work_tile_info)\n");
      // }
      return work_tile_info; // 在static_tile_scheduler里定义了continue_current_work函数必然返回false。所以是绝对不会进入这里的。
    }

    advance_to_next_work();
    return get_current_work();
  }

};

}
