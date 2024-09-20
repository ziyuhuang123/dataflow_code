#include "cuda_runtime.h"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
#include "cooperative_groups.h"
#include "cuda_fp16.h"
#include <cuda/barrier>
#include <cuda/ptx>
#include <iostream>
#include <random>
#include <stdio.h>

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

#define PROFILING 1
#define BLOCK_SIZE 512
#define HEAD_DIM 128
#define HEAD_NUM 32
#define CLUSTER_SIZE 4
#define BATCH_SIZE 1
#define SEQ_LEN 4096

template <typename T>
void fill_matrix(T* mat, int sz) {
    std::random_device r;
    std::mt19937 rng(r());
    std::normal_distribution<float> norm_dist(0.0, 5.0);
    for (int i = 0; i < sz; i++) {
        if constexpr(std::is_same<T, half>::value) {
            mat[i] = __float2half(1.0f);
        }   
    }   
}

__device__ void add(
    float* A,
    float* B,
    int len
)
{
    #pragma unroll
    for (int i = 0; i < len; i++) {
        A[i] += B[i];
    }
}

__device__ float dot(
    half* A,
    half* B,
    int len 
)
{
    float res = 0.0;
    #pragma unroll
    for (int i = 0; i < len; i++) {
        res += __half2float(A[i]) * __half2float(B[i]);
    }   
    return res;
}

__global__ void __cluster_dims__(1, CLUSTER_SIZE, 1) decode(
    half* output, // batch * head_num * head_dim
    half* input,  // batch * seqlen
    half* w_q,    // batch * seqlen * head_num * head_dim
    half* w_k,    // batch * seqlen * head_num * head_dim 
    half* w_v,    // batch * seqlen * head_num * head_dim
    half* w_o,    // batch * seqlen * head_num * head_dim
    half* k_cache,// batch * head_num * (seqlen - 1) * head_dim
    half* v_cache// batch * head_num * head_dim * (seqlen - 1)
)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t batch_id         = blockIdx.x;
    const uint32_t head_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % 32; 
    const uint32_t warp_id = tid / 32; 

    // TODO: All cluster here share the same input
    // Load input in shared
    __shared__ half input_shmem[SEQ_LEN];
    for (int d = tid; d < SEQ_LEN / 8; d+=block.num_threads()) {
        *(uint4*)(&input_shmem[d * 8]) = *(uint4*)(&input[batch_id * SEQ_LEN + d * 8]);
    }
    __syncthreads();

    // Compute hidden * wq
    half w_qk_reg[8];
    half input_reg[8];
    __shared__ half local_qkv_reduction[16 * 8]; 
    __shared__ half local_q[HEAD_DIM / CLUSTER_SIZE];
    __shared__ half local_kv[HEAD_DIM / CLUSTER_SIZE];
    for (int d = 0; d < HEAD_DIM / CLUSTER_SIZE; d+=8) {
        *(uint4*)(&input_reg[0]) = *(uint4*)(&input_shmem[tid * (SEQ_LEN / block.num_threads())]);
        half local_sum[8] = {0};
        for (int i = 0; i < SEQ_LEN / block.num_threads(); i++) {   
            *(uint4*)(&w_qk_reg[0]) = *(uint4*)(&w_q[batch_id * SEQ_LEN * HEAD_DIM * HEAD_NUM + head_id * HEAD_DIM * SEQ_LEN + cluster_block_id * (HEAD_DIM / CLUSTER_SIZE) + (tid * (SEQ_LEN / block.num_threads()) + i) * HEAD_DIM + d]);
            // TODO: Use half2 __hmul2 but exist bug
            for (int di = 0; di < 8; di++) {
                local_sum[di] += __hmul(input_reg[i], w_qk_reg[di]);
            }
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
            *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
            *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
            *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
        }   
        if (lane_id == 0) {
            *(uint4*)(&local_qkv_reduction[warp_id * 8]) = *(uint4*)(&local_sum[0]);
        } 
        __syncthreads();
        if (tid < 16) {
            *(uint4*)(&local_sum[d]) = *(uint4*)(&local_qkv_reduction[tid * 8]);
        }
        for (int mask = 8; mask > 0; mask >>= 1) {
            *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
            *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
            *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
            *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
        }   
        if (tid == 0)
            *(uint4*)(&local_q[d]) = *(uint4*)(&local_sum[0]);
    }
    __syncthreads();

    // Compute hidden * wk
    for (int d = 0; d < HEAD_DIM / CLUSTER_SIZE; d+=8) {
        *(uint4*)(&input_reg[0]) = *(uint4*)(&input_shmem[tid * (SEQ_LEN / block.num_threads())]);
        half local_sum[8] = {0};
        for (int i = 0; i < SEQ_LEN / block.num_threads(); i++) {   
            *(uint4*)(&w_qk_reg[0]) = *(uint4*)(&w_k[batch_id * SEQ_LEN * HEAD_DIM * HEAD_NUM + head_id * HEAD_DIM * SEQ_LEN + cluster_block_id * (HEAD_DIM / CLUSTER_SIZE) + (tid * (SEQ_LEN / block.num_threads()) + i) * HEAD_DIM + d]);
            // TODO: Use half2 __hmul2 but exist bug
            for (int di = 0; di < 8; di++) {
                local_sum[di] += __hmul(input_reg[i], w_qk_reg[di]);
            }
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
            *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
            *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
            *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
        }   
        if (lane_id == 0) {
            *(uint4*)(&local_qkv_reduction[warp_id * 8]) = *(uint4*)(&local_sum[0]);
        } 
        __syncthreads();
        if (tid < 16) {
            *(uint4*)(&local_sum[d]) = *(uint4*)(&local_qkv_reduction[tid * 8]);
        }
        for (int mask = 8; mask > 0; mask >>= 1) {
            *(half2*)(&local_sum[0]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[0]), mask);
            *(half2*)(&local_sum[2]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[2]), mask);
            *(half2*)(&local_sum[4]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[4]), mask);
            *(half2*)(&local_sum[6]) += __shfl_down_sync(0xffffffff, *(half2*)(&local_sum[6]), mask);
        }   
        if (tid == 0)
            *(uint4*)(&local_kv[d]) = *(uint4*)(&local_sum[0]);
    }
    __syncthreads();

    // Compute q * k^T 
    extern __shared__ half attn_weight[];
    float *attn_weight_smem = reinterpret_cast<float*>(attn_weight);
    float *attn_weight_smem_buffer = reinterpret_cast<float*>(attn_weight + SEQ_LEN * 4);
    for (int d = tid; d < SEQ_LEN; d+=block.num_threads()) {
        attn_weight_smem[d] = 0.0f;
        attn_weight_smem_buffer[d] = 0.0f;
    } 
    cluster.sync(); 
        
    // Load K cache to register
    half k_cache_reg[8];
    half q_reg[8];
    for (int d = tid; d < SEQ_LEN; d+=block.num_threads()) {
        for (int i = 0; i < (HEAD_DIM / CLUSTER_SIZE) / 8; i++) {
            *(uint4*)(&q_reg[0]) = *(uint4*)(&local_q[i * 8]);
            if (d != SEQ_LEN - 1) {
                *(uint4*)(&k_cache_reg[0]) = *(uint4*)(&k_cache[batch_id * HEAD_DIM * HEAD_NUM * (SEQ_LEN - 1) + head_id * HEAD_DIM * (SEQ_LEN - 1) + cluster_block_id * (HEAD_DIM / CLUSTER_SIZE) + d * HEAD_DIM + i * 8]);
                attn_weight_smem[d] += dot(q_reg, k_cache_reg, 8);
            }else {
                *(uint4*)(&k_cache_reg[0]) = *(uint4*)(&local_kv[i * 8]);
                attn_weight_smem[d] += dot(q_reg, k_cache_reg, 8);
            }
        }   
    }  
    cluster.sync();

    // Attention weight reduce through DSM
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        __shared__ uint64_t barrier;
        // Load neighbor block shmem data to this block's buffer within cluster
        if (tid == 0) {
            uint32_t size = SEQ_LEN * 4;
            uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
            asm volatile (
                "mbarrier.init.shared::cta.b64 [%0], %1;"
                :
                : "r"(bar_ptr), "r"(1)
            );
            asm volatile (
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                :
                : "r"(bar_ptr), "r"(size)
            );
        }
        cluster.sync();
        if (tid == 0) {
            uint32_t size = SEQ_LEN * 4;
            uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(attn_weight_smem));
            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(attn_weight_smem_buffer));
            uint32_t dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            uint32_t neighbor_dst_addr;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_addr)
                : "r"(dst_addr), "r"(dst_cta)
            );
            uint32_t neighbor_dst_bar;
            asm volatile (
                "mapa.shared::cluster.u32 %0, %1, %2;\n"
                : "=r"(neighbor_dst_bar)
                : "r"(bar_ptr), "r"(dst_cta)
            );
            asm volatile (
                "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :
                :"r"(neighbor_dst_addr), "r"(src_addr), "r"(size), "r"(neighbor_dst_bar)
                : "memory"
            );
        }
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(bar_ptr),
            "r"(0)
        );
        
        // Add
        for (int d = tid; d < SEQ_LEN; d += block.num_threads()) {
            float buffer = attn_weight_smem_buffer[d];
            attn_weight_smem[d] += buffer;
        }   
        __syncthreads();
    }


}

int main(int argc, char** argv) {
    cudaFuncSetAttribute(decode, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(float) * BATCH_SIZE * SEQ_LEN * 5);
    cudaFuncSetAttribute(decode, cudaFuncAttributeNonPortableClusterSizeAllowed, 16);
    int batch = BATCH_SIZE, seq_len = SEQ_LEN;
    half *h_input, *d_input;
    half *h_k_cache, *d_k_cache;
    half *h_v_cache, *d_v_cache;
    half *h_w_q, *d_w_q;
    half *h_w_k, *d_w_k;
    half *h_w_v, *d_w_v;
    half *h_w_o, *d_w_o;
    h_input = new half[batch * seq_len];
    h_w_q = new half[batch * HEAD_NUM * seq_len * HEAD_DIM];
    h_w_k = new half[batch * HEAD_NUM * seq_len * HEAD_DIM];
    h_w_v = new half[batch * HEAD_NUM * seq_len * HEAD_DIM];
    h_w_o = new half[batch * HEAD_NUM * seq_len * HEAD_DIM];
    h_k_cache = new half[batch * (seq_len - 1) * HEAD_DIM * HEAD_NUM];
    h_v_cache = new half[batch * (seq_len - 1) * HEAD_DIM * HEAD_NUM];
    fill_matrix(h_input, batch * seq_len);
    fill_matrix(h_k_cache, batch * (seq_len - 1) * HEAD_DIM * HEAD_NUM);
    fill_matrix(h_v_cache, batch * (seq_len - 1) * HEAD_DIM * HEAD_NUM);
    fill_matrix(h_w_q, batch * HEAD_NUM * seq_len * HEAD_DIM);
    fill_matrix(h_w_k, batch * HEAD_NUM * seq_len * HEAD_DIM);
    fill_matrix(h_w_v, batch * HEAD_NUM * seq_len * HEAD_DIM);
    fill_matrix(h_w_o, batch * HEAD_NUM * seq_len * HEAD_DIM);

    cudaMalloc(reinterpret_cast<void**>(&d_input), sizeof(half) * batch * seq_len);
    cudaMalloc(reinterpret_cast<void**>(&d_k_cache), sizeof(half) * batch * (seq_len - 1) * HEAD_DIM * HEAD_NUM);
    cudaMalloc(reinterpret_cast<void**>(&d_v_cache), sizeof(half) * batch * (seq_len - 1) * HEAD_DIM * HEAD_NUM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_q), sizeof(half) * batch * HEAD_NUM * seq_len * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_k), sizeof(half) * batch * HEAD_NUM * seq_len * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_v), sizeof(half) * batch * HEAD_NUM * seq_len * HEAD_DIM);
    cudaMalloc(reinterpret_cast<void**>(&d_w_o), sizeof(half) * batch * HEAD_NUM * seq_len * HEAD_DIM);
    cudaMemcpy(reinterpret_cast<void*>(d_input), h_input, sizeof(half) * batch * seq_len, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_k_cache), h_k_cache, sizeof(half) * batch * (seq_len - 1) * HEAD_DIM * HEAD_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_v_cache), h_v_cache, sizeof(half) * batch * (seq_len - 1) * HEAD_DIM * HEAD_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_q), h_w_q, sizeof(half) * batch * HEAD_NUM * seq_len * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_k), h_w_k, sizeof(half) * batch * HEAD_NUM * seq_len * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_v), h_w_v, sizeof(half) * batch * HEAD_NUM * seq_len * HEAD_DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_w_o), h_w_o, sizeof(half) * batch * HEAD_NUM * seq_len * HEAD_DIM, cudaMemcpyHostToDevice);

    half* h_output, *d_output;
    h_output = new half[batch * HEAD_DIM * HEAD_NUM];
    cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(half) * batch * HEAD_DIM * HEAD_NUM);

    dim3 grid(batch, HEAD_NUM * CLUSTER_SIZE);
    dim3 block(BLOCK_SIZE);

#if PROFILING == 1
    int wmup = 5; 
    int test = 1; 
    cudaEvent_t st, ed; 
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    for (int i = 0; i < wmup; i++) {
        decode<<<grid, block, sizeof(float) * BATCH_SIZE * SEQ_LEN * 5>>>(
            d_output,
            d_input,
            d_w_q,
            d_w_k,
            d_w_v,
            d_w_o,
            d_k_cache,
            d_v_cache
        );  
    }   
    cudaEventRecord(st);
    for (int i = 0; i < test; i++) {
#endif
        decode<<<grid, block, sizeof(float) * BATCH_SIZE * SEQ_LEN * 5>>>(
            d_output,
            d_input,
            d_w_q,
            d_w_k,
            d_w_v,
            d_w_o,
            d_k_cache,
            d_v_cache
        );
        CUDA_CHECK(cudaGetLastError());  
#if PROFILING == 1
    }   
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float ms; 
    cudaEventElapsedTime(&ms, st, ed);
    std::cout << "Latency: " << (ms / (1.0 * test)) * 1e3 << " us" << std::endl;
#endif
    cudaMemcpy(h_output, reinterpret_cast<void*>(d_output), sizeof(half) * batch * HEAD_DIM * HEAD_NUM, cudaMemcpyDeviceToHost);
    return 0;
}