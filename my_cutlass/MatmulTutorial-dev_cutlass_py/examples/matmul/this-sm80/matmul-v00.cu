// A100 PCIE 80GB
// Testing iters = 200.
// Test performance using shape M=5376, N=5376, K=2048
// Running cost of CUDA kernel is 4.46636ms
// TFLOPS: 26.5048

// nvcc -arch=sm_90  -DDEBUG -Xcompiler -fopenmp matmul-v00.cu main.cu -o test && ./test stages 1 > result.txt 2>&1

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>

const int MI = 128;
const int NI = 128;
const int KI = 32;
const int MII = 64;
const int NII = 64;
const int KII = 16;
const int wmmaM = 16;
const int wmmaN = 16;
const int wmmaK = 16;

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

__device__ void loadSmemB(half *smem, half *B, int N, int K, int ko)
{
    // load 128 * 32
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 32; ++i)
    {
        int row = i * 4 + tid / 32;
        int col = tid % 32;
        // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
        smem[row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = B[(bx * 128 + row) * K + ko * KI + col];
    } // B是K*N。。。。奇怪，真的可以这样（行数*列宽+列数）来访问吗？
}



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




// __device__ void loadSmemC(float *smem, half *C, int M, int N)
// {
//     // load 128 * 128
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int tz = threadIdx.z;
//     int tid = tz * 64 + ty * 32 + tx;
//     for (int i = 0; i < 128; ++i)
//     {
//         int row = i;
//         int col = tid;
//         // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
//         smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = (float)(C[(by * 128 + row) * N + bx * 128 + col]);
//     }
// }

__device__ void loadSmemC(float *smem, half *C, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
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
        smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16] = (half)(C[(by * 128 + row) * N + bx * 128 + col]);
    }
}



// __device__ void storeSmemC(half *C, float *smem, int M, int N)
// {
//     // load 128 * 128
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int tz = threadIdx.z;
//     int tid = tz * 64 + ty * 32 + tx;
//     for (int i = 0; i < 128; ++i)
//     {
//         int row = i;
//         int col = tid;
//         // layout: [row_out, col_out, row_in, col_in] = [8, 8, 16, 16]
//         (C[(by * 128 + row) * N + bx * 128 + col]) = (half)smem[row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16];
//     }
// }
__device__ void storeSmemC(half *C, half *smem, int M, int N)
{
    // load 128 * 128
    int bx = blockIdx.x;
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


__device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> *frag, half *smem, int ki)
{
    // load 64x16
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i)
    {
        int row = tz * 64 + i * 16;
        int col = ki * KII;
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

// __device__ void storeAccum(float *ptr, nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> *frag)
// {
//     // store 64x64
//     int ty = threadIdx.y;
//     int tz = threadIdx.z;
//     for (int i = 0; i < 4; ++i)
//     {
//         for (int j = 0; j < 4; ++j)
//         {
//             int row = tz * 64 + i * 16;
//             int col = ty * 64 + j * 16;
//             // laoyut: [8, 8, 16, 16]
//             nvcuda::wmma::store_matrix_sync(ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16), frag[i * 4 + j], 16, nvcuda::wmma::mem_row_major);
//         }
//     }
// }
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

__global__ void matmul(half *A, half *B, half *C, half *gemm1_result, half *gemm1_weight, int M, int N, int K, int T, float alpha, float beta)
{ // 中间结果彻底不需要存出去了。每次都完全利用干净。gemm1_weight的尺寸是N*T（col-major)
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ uint8_t shared_storage[];
    half *SA = reinterpret_cast<half *>(shared_storage); // 128*32*2/1024=8KB
    half *SB = reinterpret_cast<half *>(shared_storage + MI * KI * sizeof(half)); // 128*32*2/1024=8KB
    // float *SC = reinterpret_cast<float *>(shared_storage);
    half *SC = reinterpret_cast<half *>(shared_storage); // 128*128*2/1024=32KB
    half *S_gemm1_weight = reinterpret_cast<half *>(shared_storage + MI * NI * sizeof(half)); // 128*32*2/1024=8KB---最大空间是40KB
    half *S_gemm1_result = reinterpret_cast<half *>(shared_storage + (MI * NI+MI*KI) * sizeof(half)); // 这个是得接在SC的结果之后。但是可以服用gemm1_weight的空间


    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> FragA[MII / wmmaM];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragB[NII / wmmaN];
    // nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[MII / wmmaM * NII / wmmaN];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, half> Accum[MII / wmmaM * NII / wmmaN];

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
        loadSmemB(SB, B, N, K, ko); // 正常的B是K*N，这里反着写（可能是因为col-major)
        __syncthreads();
        for (int ki = 0; ki < KI / KII; ki += 1)
        {
            // 64x64x16 mma for each warp
            loadFragA(FragA, SA, ki);
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
    storeSmemC(C, SC, M, N);







// 先做假设，GEMM0的N恰好等于128，不会有多次。之后再逐渐加入多次计算。
    for(int T_iter = 0; T_iter < T/NI; T_iter++){

        // 真的有必要重新置零一次吗。。。不过以防万一吧。
        for (int mii = 0; mii < MII / wmmaM; mii += 1)
        {
            for (int nii = 0; nii < NII / wmmaN; nii += 1)
            {
                nvcuda::wmma::fill_fragment(Accum[mii * (NII / wmmaN) + nii], 0.0);
            }
        }

    // 还要加一层。对于一块gemm0_result，需要对所有gemm1相关的行都乘一遍。比如T等于256的时候就需要向右再算一次。
        for (int ko = 0; ko < K / KI; ko += 1)
        { // K/KI=（比如512）/32
            // loadSmemA(SA, A, M, K, ko); // 不需要了。就是上一步的SC
            loadSmemB_new(S_gemm1_weight, gemm1_weight, T, N, ko, T_iter);
            __syncthreads();
            for (int ki = 0; ki < KI / KII; ki += 1)
            { // KI/KII=32/16=2
                // 64x64x16 mma for each warp
                loadFragA_new(FragA, SC, ko*KI+ki*KII); // 和之前的loadFragA不同，这里传入的第三个参量是当前真实的col位置
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
        __syncthreads();
        storeSmemC_new(gemm1_result, S_gemm1_result, M, T, T_iter);
    }
}