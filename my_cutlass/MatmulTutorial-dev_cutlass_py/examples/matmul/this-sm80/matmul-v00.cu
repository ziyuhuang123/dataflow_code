// A100 PCIE 80GB
// Testing iters = 200.
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 4.46636ms
// TFLOPS: 26.5048

// nvcc -arch=sm_90  -DDEBUG -DPRINT -Xcompiler -fopenmp matmul-v00.cu main.cu -o test && ./test stages 1 > result.txt 2>&1  // debug很需要打印出来看看。以防全都是inf值还判断对
// nvcc -arch=sm_90 -DDEBUG -Xcompiler -fopenmp -lcublas matmul-v00.cu main.cu -o test && ./test stages 1 > result.txt 2>&1

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>
#include <stdio.h>
#include <cooperative_groups.h>

const int MI = 128;
const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;
#define C_LAYOUT nvcuda::wmma::mem_row_major
namespace cg = cooperative_groups;




__device__ void loadSmemA(half *smem, half *A, int M, int K, int ko)
{
    // load 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = A[(by * 128 + row) * K + ko * KI + col];
    }
}

// __device__ void loadSmemB(half *smem, half *B, int N, int K, int ko, int N_index)
// {
//     // load 128 * 32
//     int bx = blockIdx.x;
//     bx = N_index; // 新的代码写法下，每次的位置是N_index，而grid.x始终等于1
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int tz = threadIdx.z;
//     int tid = tz * 64 + ty * 32 + tx;
//     for (int i = 0; i < 32; ++i)
//     {
//         int row = i * 4 + tid / 32;
//         int col = tid % 32;
//         // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
//         smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = B[(bx * 128 + row) * K + ko * KI + col];
//     } // B是K*N。。。。奇怪，真的可以这样（行数*列宽+列数）来访问吗？
// } // 和下面的_new版本是一样的。其实没必要留着啦！


__device__ void loadSmemB_new(half *smem, half *B, int N, int K, int ko, int bx_iter)
{
    // load 128 * 32
    // int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = B[(bx_iter * 128 + row) * K + ko * KI + col];
    } // B是K*N。。。。这里乘上K，就说明是colMajor
}


__device__ void loadSmemC(half *smem, half *C, int new_M, int new_N, int T_index)
{
    // load 128 * 128
    int bx = blockIdx.x;
    bx = T_index;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = (half)(C[(by * 128 + row) * new_N + bx * 128 + col]);
    }
}


__device__ void loadFragC(nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, half> *frag, half *smem)
{
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    // load 128*128 C是row-major
    for (int mii = 0; mii < MII / wmmaM; mii += 1)
    {
        for (int nii = 0; nii < NII / wmmaN; nii += 1)
        {
            int row = tz * 64 + mii * 16;
            int col = ty * 64 + nii * 16;
            nvcuda::wmma::load_matrix_sync(frag[mii * (NII / wmmaN) + nii], smem + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16), 16, C_LAYOUT);
        }
    }
}

__device__ void storeSmemC(half *C, half *smem, int M, int N, int N_ind)
{
    // load 128 * 128
    int bx = blockIdx.x;
    bx = N_ind;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        (C[(by * 128 + row) * N + bx * 128 + col]) = smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16]; // 取消了half
    }
}

__device__ void storeSmemC_new(half *C, half *smem, int M, int N, int bx_iter)
{
    // load 128 * 128
    // int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128; ++i)
    {
        int row = i;
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        (C[(by * 128 + row) * N + bx_iter * 128 + col]) = smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16]; // C的尺寸是M*N，这里是row-Major。对于bx会有变化。
    }
}


__device__ void storeSmemC_new_cluster(half *C, half *smem, int M, int N, int bx_iter)
{

// 假设cluster只能最大是8。那样最大就是切到16*16，不会切16本身。cluster等于1 2 4 8。


    cg::cluster_group cluster       = cg::this_cluster();
    const uint32_t cluster_dim_x = cluster.num_blocks(); 
    const uint32_t block_id = cluster.block_rank(); // 获取当前 block 


    // load 128 * 128
    // int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 128/cluster_dim_x; ++i)
    {
        // int row = i+block_id*(128/cluster_dim_x);
        int row = i;
        int store_row = i+block_id*(128/cluster_dim_x);
        int col = tid;
        // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
        (C[(by * 128 + store_row) * N + bx_iter * 128 + col]) = smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16]; // C的尺寸是M*N，这里是row-Major。对于bx会有变化。
    }
}



__device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> *frag, half *smem, int ki)
{
    // load 64x16
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        int row = tz * 64 + i * 16;
        int col = ki * KII; // ki=0或者1，col=0或者16
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}

__device__ void loadFragA_new(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> *frag, half *smem, int col_ind)
{
    // load 64x16
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        int row = tz * 64 + i * 16;
        // int col = ki * KII; // KII=16
        int col=col_ind;
        // nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (128 * 16)  // 这里之前是32，现在改成128是因为C的一列宽度是128
        + col / 16 * (16 * 16), 16); // 这里的row和col都是C矩阵意义上的位置，而要加到smem指针上，则要考虑真实的步长，比如col这里，增加一个16*16的col向右，从C来看只需要加16即可，但是smem是走完左侧的16*16之后，才能到右侧的col位置上。不过这一点对C_smem是不变的。不过在调用loadFragA_new的地方需要相应修改传入的ki。
    }
}

__device__ void loadFragB(nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> *frag, half *smem, int ki)
{
    // load 64x16
    int ty = threadIdx.y;
    for (int i = 0; i < 4; ++i)
    {
        int row = ty * 64 + i * 16;
        int col = ki * KII;
        nvcuda::wmma::load_matrix_sync(frag[i], smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16), 16);
    }
}


__device__ void storeAccum(half *ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, half> *frag)
{
    // store 64x64
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int row = tz * 64 + i * 16;
            int col = ty * 64 + j * 16;
            // laoyut: [8, 8, 16, 16]
            nvcuda::wmma::store_matrix_sync(ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16), frag[i * 4 + j], 16, nvcuda::wmma::mem_row_major);
        }
    }
}


__device__ void reduce_dsm(half *ptr_result, half *ptr_store_tmp)
{


    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t batch_id         = blockIdx.x;
    const uint32_t head_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % 32; 
    const uint32_t warp_id = tid / 32; 
    const uint32_t cluster_dim_x = cluster.num_blocks();
    const uint32_t block_id = cluster.block_rank(); // 获取当前 block 

    const uint32_t block_id_x = blockIdx.x;
    const uint32_t block_id_y = blockIdx.y;
    const uint32_t block_id_z = blockIdx.z;






    // if (blockIdx.x == 1 && blockIdx.y == 0 &&blockIdx.z==0&& threadIdx.x == 0 && threadIdx.y == 0 &&threadIdx.z==0) {
    //     printf("cluster_dim_x=%d, block_id=%d\n", cluster_dim_x, block_id);
    // }


    // if (blockIdx.x == 0 && blockIdx.y == 0 &&blockIdx.z==0&& threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z==0) {
    //     for(int ii=0;ii<128*128;ii++){
    //         ptr_result[ii] = half(ii*1e-2);
    //     }
    //     for(int ii=0;ii<128*128;ii++){
    //         ptr_store_tmp[ii] = half(0);
    //     }
    // }
    // if (blockIdx.x == 1 && blockIdx.y == 0 &&blockIdx.z==0&& threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z==0) {
    //     for(int ii=0;ii<128*128;ii++){
    //         ptr_result[ii] = half((-1)*ii*1e-2);
    //     }
    //     for(int ii=0;ii<128*128;ii++){
    //         ptr_store_tmp[ii] = half(0);
    //     }
    // }
    // __syncthreads();


    // if (blockIdx.x == 1 && blockIdx.y == 0 &&blockIdx.z==0&& threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z==0) {
    //     printf("input\n");
    //     for(int ii=0;ii<128;ii++){
    //         for(int jj=0;jj<128;jj++){
    //             printf("%f  ", float(ptr_result[ii*128+jj]));
    //         }
    //         printf("\n");
    //     }
    // }


    half* ptr_result_local = ptr_result+128*128/cluster_dim_x*block_id;

    cluster.sync();
    // Attention weight reduce through DSM
    for (int i = 0; i < cluster_dim_x - 1; i++) {


        // uint32_t dst_cta_x = (block_id_x + i + 1) % cluster_dim_x;
        // uint32_t dst_cta_y = block_id_y; // 保持 y 维度不变
        // uint32_t dst_cta_z = block_id_z; // 保持 z 维度不变
        // uint32_t dst_cta = dst_cta_x + (dst_cta_y * gridDim.x) + (dst_cta_z * gridDim.x * gridDim.y);


        // if (blockIdx.z==0&& threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z==0) {
        //     printf("dst_cta_x=%d, i=%d, blx=(%d, %d)\n", dst_cta_x, i, blockIdx.x, blockIdx.y);
        // }




        __shared__ uint64_t barrier;
        // Load neighbor block shmem data to this block's buffer within cluster
        uint32_t size = 128*128*2/cluster_dim_x;
        if (threadIdx.x == 0&&threadIdx.y==0&&threadIdx.z==0) {
            // uint32_t size = 128*128*2/cluster_dim_x;
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
        if (threadIdx.x == 0&&threadIdx.y==0&&threadIdx.z==0) {
            // uint32_t size = 128*128*2/cluster_dim_x;
            uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
            uint32_t src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_result+128*128/cluster_dim_x*((block_id+i+1)%cluster_dim_x))); // 这个操作的意思似乎是，以当前block为src，送到dst_cta的dst位置去。


            uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr_store_tmp));


            // if (blockIdx.x == 0 && blockIdx.y == 0 &&blockIdx.z==0&& threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z==0) {
            //     printf("dst_cta=%d, i=%d\n", dst_cta, i);
            // }
            uint32_t dst_cta = (block_id_x + i + 1) % cluster_dim_x; // mapa指令只需要考虑cluster内的block_rank即可。不用管grid层面的事情。

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
        



        // if (blockIdx.x == 0 && blockIdx.y == 0 &&blockIdx.z==0&& threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z==0 && i==0) {
        //     printf("output\n");
        //     for(int ii=0;ii<64;ii++){
        //         for(int jj=0;jj<128;jj++){
        //             printf("%f  ", float(ptr_store_tmp[ii*128+jj]));
        //         }
        //         printf("\n");
        //     }
        // }
        // if (blockIdx.x == 1 && blockIdx.y == 0 &&blockIdx.z==0&& threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z==0 && i==0) {
        //     printf("output11\n");
        //     for(int ii=0;ii<64;ii++){
        //         for(int jj=0;jj<128;jj++){
        //             printf("%f  ", float(ptr_result[ii*128+jj]));
        //         }
        //         printf("\n");
        //     }
        // }




        // 计算总的元素数量
        int total_elements = 128 * 128 / cluster_dim_x;
        int total_half2 = total_elements / 2; // 每个 half2 包含两个 half
        int remainder = total_elements % 2;   // 剩余的元素数量

        // 将指针转换为 half2*
        half2* ptr_store_tmp_half2 = reinterpret_cast<half2*>(ptr_store_tmp);
        half2* ptr_result_local_half2 = reinterpret_cast<half2*>(ptr_result_local);

        // 计算线程 ID 和总线程数
        int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        int num_threads = blockDim.x * blockDim.y * blockDim.z;

        // 主循环，处理 half2 数据
        #pragma unroll
        for (int d = tid; d < total_half2; d += num_threads) {
            half2 buffer = ptr_store_tmp_half2[d];
            half2 result = ptr_result_local_half2[d];

            // 使用 __hadd2 进行元素级加法
            result = __hadd2(result, buffer);

            ptr_result_local_half2[d] = result;
        }


        // // Add
        // #pragma unroll
        // for (int d = tid; d < 128*128/cluster_dim_x; d += block.num_threads()) {
        //     float buffer = ptr_store_tmp[d];
        //     ptr_result_local[d] += buffer;
        // } // 这样是比较慢的写法 可能性能差一些
        __syncthreads();
    }






    // if (blockIdx.x == 0 && blockIdx.y == 0 &&blockIdx.z==0&& threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z==0) {
    //     printf("output\n");
    //     for(int ii=0;ii<64;ii++){
    //         for(int jj=0;jj<128;jj++){
    //             printf("%f  ", float(ptr_result_local[ii*128+jj]));
    //         }
    //         printf("\n");
    //     }
    // }
}


__global__ void matmul(half *A, half *B, half *C, half *gemm1_result, half *gemm1_weight, int M, int N, int K, int T, float alpha, float beta)
{ // 中间结果彻底不需要存出去了。每次都完全利用干净。gemm1_weight的尺寸是N*T（col-major)
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16

    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t batch_id         = blockIdx.x;
    const uint32_t head_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % 32; 
    const uint32_t warp_id = tid / 32; 
    const uint32_t cluster_dim_x = cluster.num_blocks();
    const uint32_t block_id = cluster.block_rank(); // 获取当前 block 



    extern __shared__ uint8_t shared_storage[];
    half *SA = reinterpret_cast<half *>(shared_storage); // 128*32*2/1024=8KB
    half *SB = reinterpret_cast<half *>(shared_storage + MI * KI * sizeof(half)); // 128*32*2/1024=8KB
    // float *SC = reinterpret_cast<float *>(shared_storage);
    half *SC = reinterpret_cast<half *>(shared_storage); // 128*128*2/1024=32KB
    half *S_gemm1_weight = reinterpret_cast<half *>(shared_storage + MI * NI * sizeof(half)); // 128*32*2/1024=8KB---最大空间是40KB
    half *S_gemm1_result = reinterpret_cast<half *>(shared_storage + (MI * NI+MI*KI) * sizeof(half)); // 这个是得接在SC的结果之后。但是可以服用gemm1_weight的空间
    half *reduce_local_smem = reinterpret_cast<half *>(shared_storage); // 可以复用原先GEMM0的结果的位置。这里需要的尺寸会小于128*128*2，肯定放得下。具体尺寸是128*128*2/cluster_size


    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> FragA[MII / wmmaM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragB[NII / wmmaN];
    // nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[MII / wmmaM * NII / wmmaN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, half> Accum[MII / wmmaM * NII / wmmaN];





    for(int N_iter=0;N_iter<N/(NI*cluster_dim_x);N_iter+=1){
// 自从N_iter的引入，每个block就负责一整行了。也就是loadSmemB要修改了。


        for (int mii = 0; mii < MII / wmmaM; mii += 1)
        {
            for (int nii = 0; nii < NII / wmmaN; nii += 1)
            {
                nvcuda::wmma::fill_fragment(Accum[mii * (NII / wmmaN) + nii], 0.0);
            }
        }


        for (int ko = 0; ko < K / KI; ko += 1)
        {
            loadSmemA(SA, A, M, K, ko);
            loadSmemB_new(SB, B, N, K, ko, cluster_block_id+N_iter*cluster_dim_x); // 不过如果超过了一个cluster的宽度，则需要再修改。。这里需要加上些N_iter....
            // loadSmemB_new(SB, B, N, K, ko, N_iter); // 正常的B是K*N，这里反着写（可能是因为col-major) -- loadSmemB和_new版本基本是一样的。
            __syncthreads();
            for (int ki = 0; ki < KI / KII; ki += 1)
            {  // KI/KII=32/16=2
                // 64x64x16 mma for each warp
                loadFragA(FragA, SA, ki);// ki=0或者1
                loadFragB(FragB, SB, ki);
                for (int mii = 0; mii < MII / wmmaM; mii += 1)
                {
                    for (int nii = 0; nii < NII / wmmaN; nii += 1)
                    {
                        // 16x16x16 for each wmma
                        nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
                    }
                }
            }
        }
        storeAccum(SC, Accum);
        __syncthreads();
        storeSmemC(C, SC, M, N, cluster_block_id); // 以后可以注释掉。现在留着是为了debug
    // =============================    GEMM1    ========================
    // 先做假设，GEMM0的N恰好等于128，不会有多次。之后再逐渐加入多次计算。
        for(int T_iter = 0; T_iter < T/NI; T_iter++){
            // 如果N比128大。我们先不考虑cluster。那么就需要多次回到global再回SMEM了。

            if(N_iter>=1){
                // 暂时不管吧。。。。之后可能要改改
                loadSmemC(S_gemm1_result, gemm1_result, M, T, T_iter);
                __syncthreads();
                // if(blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0){
                //     printf("loadSmemC. T_iter=%d\n", T_iter);
                //     for(int kk=0;kk<64;kk++){
                //         for(int tt=0;tt<256;tt++){
                //             printf("%f  ", float(S_gemm1_result[kk*256+tt]));
                //         } // 连续256个是16*16的块。64是8*8个整块
                //         printf("\n");
                //     }
                //     printf("\n");
                // }
                loadFragC(Accum, S_gemm1_result);
            }
            else{
            // 这里需要重新置零的！之前GEMM0的结果还存在里面呢！
                for (int mii = 0; mii < MII / wmmaM; mii += 1)
                {
                    for (int nii = 0; nii < NII / wmmaN; nii += 1)
                    {
                        nvcuda::wmma::fill_fragment(Accum[mii * (NII / wmmaN) + nii], 0.0);
                    }
                }
            }



        // 还要加一层。对于一块gemm0_result，需要对所有gemm1相关的行都乘一遍。比如T等于256的时候就需要向右再算一次。
            for (int k_gemm1_o = 0; k_gemm1_o < NI / KI; k_gemm1_o += 1)
            { // GEMM0是对K迭代。GEMM1就是对N迭代。但是还是按照KI和KII的尺寸来。我们此处无法迭代到底。只能迭代NI/KI次。是根据GEMM0的中间结果尺寸决定的。
                // int k_gemm1_o_iter = N_iter*(NI/KI)+k_gemm1_o;
                int k_gemm1_o_iter = N_iter*cluster_dim_x*(NI/KI)+k_gemm1_o+cluster_block_id*(NI/KI);
                // loadSmemA(SA, A, M, K, k_gemm1_o); // 不需要了。就是上一步的SC

                loadSmemB_new(S_gemm1_weight, gemm1_weight, T, N, k_gemm1_o_iter, T_iter);  // 形状是N*T，所以此处写成T, N
                __syncthreads();
                // if(blockIdx.x==1&&blockIdx.y==0&&blockIdx.z==0&&threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0&&k_gemm1_o==0){
                    
                //     printf("loadSmemB_new\n");
                //     printf("N_iter = %d, cluster_dim_x = %d, NI = %d, KI = %d, k_gemm1_o = %d, cluster_block_id = %d\n", 
                //         N_iter, cluster_dim_x, NI, KI, k_gemm1_o, cluster_block_id);
                //     printf("k_gemm1_o_iter = %d\n", k_gemm1_o_iter);

                //     for(int kk=0;kk<32;kk++){
                //         for(int tt=0;tt<128;tt++){
                //             printf("%f  ", float(S_gemm1_weight[kk*128+tt]));
                //         }
                //     }
                //     printf("\n");
                // }
                for (int ki = 0; ki < KI / KII; ki += 1)
                { // KI/KII=32/16=2
                    // 64x64x16 mma for each warp
                    loadFragA_new(FragA, SC, k_gemm1_o*KI+ki*KII); // 和之前的loadFragA不同，这里传入的第三个参量是当前真实的col位置，以及，这里要用k_gemm1_o而不是k_gemm1_o_iter，因为是在SC内部而不是去外部--[0,4)*32+[0,2)*16=
                    loadFragB(FragB, S_gemm1_weight, ki);
                    for (int mii = 0; mii < MII / wmmaM; mii += 1)
                    {
                        for (int nii = 0; nii < NII / wmmaN; nii += 1)
                        {
                            // 16x16x16 for each wmma
                            nvcuda::wmma::mma_sync(Accum[mii * (NII / wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii * (NII / wmmaN) + nii]);
                        }
                    }
                }
            }
            storeAccum(S_gemm1_result, Accum); 
            // __syncthreads();

            // 这里执行reduce。
            if(cluster_dim_x>1){
                cluster.sync();
                reduce_dsm(S_gemm1_result, reduce_local_smem);
            }

            // storeSmemC_new(gemm1_result, S_gemm1_result, M, T, T_iter); // 这里修改为各自存到各自的位置里。



            half* ptr_result_local = S_gemm1_result+128*128/cluster_dim_x*block_id;



            storeSmemC_new_cluster(gemm1_result, ptr_result_local, M, T, T_iter); // 这里修改为各自存到各自的位置里。
        }
    }
}