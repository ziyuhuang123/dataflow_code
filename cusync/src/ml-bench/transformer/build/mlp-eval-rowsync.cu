// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//<OPTIMIZATIONS>
#define MLP_GPT3
#undef AVOID_CUSTOM_ORDER
#undef AVOID_WAIT_KERNEL
#undef REORDER_TILE_LOADS
#undef NO_ATOMIC_ADD
#define ROWSYNC
#define EVAL_TILE_SIZES

//</OPTIMIZATIONS>

// #define LLAMA

#if defined(TILESYNC)
#if !defined(MLP_LLAMA)
  #define NO_ATOMIC_ADD
#else
  #undef NO_ATOMIC_ADD
#endif
#define REORDER_TILE_LOADS
#endif

// #define AVOID_CUSTOM_ORDER
// #define AVOID_WAIT_KERNEL

// #if defined(TILESYNC) || defined(TILEBATCH)
// #endif 

#include<cusync/cusync.h>
#include <fstream>
using namespace cusync;

const uint Opts = 
#ifdef AVOID_CUSTOM_ORDER
  Optimizations::AvoidCustomOrder |
#endif
#ifdef AVOID_WAIT_KERNEL
  Optimizations::AvoidWaitKernel  |
#endif
#ifdef NO_ATOMIC_ADD
  Optimizations::NoAtomicAdd      |
#endif
#ifdef REORDER_TILE_LOADS
  Optimizations::ReorderTileLoads |
#endif
  Optimizations::NoOptimization;

#include "common.h"

#include "cutlass/cusync-cutlass/include/cutlass/gemm/kernel/default_cusyncgemm.h"
#include "cutlass/cusync-cutlass/include/cutlass/gemm/kernel/cusyncgemm.h"
#include "cutlass/nvidia-cutlass/include/cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/nvidia-cutlass/include/cutlass/arch/mma.h"
#include "cutlass/nvidia-cutlass/include/cutlass/arch/arch.h"
#include "cutlass/nvidia-cutlass/include/cutlass/gemm/gemm.h"
#include "cutlass/nvidia-cutlass/include/cutlass/layout/permute.h"

#ifndef EVAL_TILE_SIZES
//Tile sizes of all GeMMs
using ShapeThreadBlock1 = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeWarp1 = cutlass::gemm::GemmShape<64, 64, 32>;

using ShapeThreadBlock2 = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeWarp2 = cutlass::gemm::GemmShape<64, 64, 32>;

const int NumStages1 = 3;
const int NumStages2 = 3;
#else
//<eval tiles>
using ShapeThreadBlock1 = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeWarp1 = cutlass::gemm::GemmShape<64, 64, 32>;

using ShapeThreadBlock2 = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeWarp2 = cutlass::gemm::GemmShape<64, 64, 32>;
const uint NumStages1 = 3;
const uint NumStages2 = 3;
// using ShapeThreadBlock1 = cutlass::gemm::GemmShape<64, 64, 32>;  
// using ShapeWarp1 = cutlass::gemm::GemmShape<32, 32, 32>;
// using ShapeThreadBlock2 = cutlass::gemm::GemmShape<64, 64, 32>;  
// using ShapeWarp2 = cutlass::gemm::GemmShape<32, 32, 32>;
// const uint NumStages1 = 4;
// const uint NumStages2 = 4;

//</eval tiles>
#endif

#define XSTR(x) STR(x)
#define STR(x) #x

#if __CUDA_ARCH_LIST__ == 700
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  
using SmArch = cutlass::arch::Sm70;
#elif __CUDA_ARCH_LIST__ == 800
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  
using SmArch = cutlass::arch::Sm80;
#else
#pragma message "Invalid CUDA ARCH" XSTR(__CUDA_ARCH__)
#error "Invalid CUDA ARCH"
#endif


#include <cstdlib>  // 包含 srand() 和 rand()
#include <cassert>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <regex>


// Utility to check CUDA and cuBLAS functions return values
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (cudaSuccess != err) { \
        fprintf(stderr, "CUDA Error: %s, in file %s, at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS_ERROR(call) { \
    cublasStatus_t status = call; \
    if (CUBLAS_STATUS_SUCCESS != status) { \
        fprintf(stderr, "cuBLAS Error: %d, in file %s, at line %d\n", status, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}


// 定义包含两个整数的元组结构体
struct Tuple {
    int first;
    int second;
};


template<typename TileOrder, uint GridN, uint TileM, uint TileN, uint stride>
struct StridedSync {
  uint waitValue_;
  uint postValue_;

  __device__ __host__ StridedSync(){}

  __device__ __host__ uint waitValue(const dim3& tile, const dim3& grid) {
    return stride;
  }

  __device__ __host__ uint postValue(const dim3& tile, const dim3& grid) 
    {return 1;}

  __device__ constexpr uint tileIndex(const dim3& tile, const dim3& grid) {
    uint ty = tile.y/TileN;
    if (ty >= (GridN/TileN)) ty = ty - (GridN/TileN);
    // if (threadIdx.x == 0) printf("ty %d tile.y %d\n", ty, tile.y);
    return TileOrder().tileIndex({tile.x/TileM, ty, 0},
                                 grid);
  }

  __device__ bool isSync(const dim3& tile, const dim3& grid) {
    return tile.y%TileN == 0;
  }
};

#ifdef ROWSYNC
  using ProdCuStage   = CuStage<TransposeXYOrder, NoSync,  RowSync<ShapeThreadBlock1::kM>, Opts>;
  using ConsCuStage   = CuStage<TransposeXYOrder, RowSync<ShapeThreadBlock1::kM>, NoSync,  Opts>;
  using Sync = RowSync<ShapeThreadBlock1::kM>;
#elif defined(TILESYNC)
  #if defined(MLP_LLAMA)
  using Sync = StridedSync<TransposeXYOrder, 2816, ShapeThreadBlock1::kM, ShapeThreadBlock1::kN,2>;
  #else
  using Sync = TileSync<TransposeXYOrder, ShapeThreadBlock1::kM, ShapeThreadBlock1::kN>;
  #endif
  using ProdCuStage   = CuStage<TransposeXYOrder, NoSync, Sync,   Opts>;
  using ConsCuStage   = CuStage<TransposeXYOrder, Sync,   NoSync, Opts>;

#else
  #error "Unknown Synchronization"
#endif

const uint GLURowTile = 8;


//Element types of A, B, and C
using ElementAccumulator = float;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementComputeEpilogue = cutlass::half_t;

//All matrices are in RowMajor
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

//Use FP-16 Tensor Cores
using MMAOp = cutlass::arch::OpClassTensorOp;

#ifdef EVAL_TILE_SIZES
  //During evaluation apply correct epilogue op
  #ifdef MLP_LLAMA
    //First GeMM in LLaMA does not apply SwiGLU but is done in 
    //another kernel
    using EpilogueOp1 = cutlass::epilogue::thread::LinearCombination<
  #elif defined(MLP_GPT3)
    //First GeMM in MLP is fused with GELU
    using EpilogueOp1 = cutlass::epilogue::thread::LinearCombinationGELU<
  #endif
#else
  //For correctness check no need to appy any epilogue
  using EpilogueOp1 = cutlass::epilogue::thread::LinearCombination<
#endif
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;
    // cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

//Second GeMM in MLP performs no extra fused computations 
using EpilogueOp2 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                        
    128 / cutlass::sizeof_bits<ElementOutput>::value,     
    ElementAccumulator,
    ElementComputeEpilogue>;

template<typename EpilogueOp, typename ShapeThreadBlock, typename ShapeWarp, int NumStages, bool splitK>
class BaseMLPGemm : public cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, 
                                                       ElementInputB, LayoutInputB,
                                                       ElementOutput, LayoutOutput,
                                                       ElementAccumulator, MMAOp,
                                                       SmArch, ShapeThreadBlock,
                                                       ShapeWarp, ShapeMMAOp,
                                                       EpilogueOp, 
                                                       cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
                                                       NumStages, 8, 8, splitK> {};
// Baseline GeMMs
using Gemm1 = BaseMLPGemm<EpilogueOp1, ShapeThreadBlock1, ShapeWarp1, NumStages1, false>;
using Gemm2 = BaseMLPGemm<EpilogueOp2, ShapeThreadBlock2, ShapeWarp2, NumStages2, false>;

//Baseline GeMMs with SplitK enabled
using GemmSplitK1 = BaseMLPGemm<EpilogueOp1, ShapeThreadBlock1, ShapeWarp1, NumStages1, true>;
using GemmSplitK2 = BaseMLPGemm<EpilogueOp2, ShapeThreadBlock2, ShapeWarp2, NumStages2, true>;

//CuSync GeMMs
using CuSyncGeMMSwizzle = cutlass::gemm::threadblock::CuSyncGemmHorizontalThreadblockSwizzle;
template<typename EpilogueOp, typename ShapeThreadBlock, typename ShapeWarp, int NumStages, bool splitK>
class CuSyncMLPGemm : public cutlass::gemm::device::CuSyncGemm<ElementInputA,
                                                               LayoutInputA, 
                                                               ElementInputB, LayoutInputB,
                                                               ElementOutput, LayoutOutput,
                                                               ElementAccumulator, MMAOp,
                                                               SmArch, ShapeThreadBlock,
                                                               ShapeWarp, ShapeMMAOp,
                                                               EpilogueOp, 
                                                               CuSyncGeMMSwizzle,
                                                               NumStages, 8, 8, splitK> {};

using CuSyncGemm1 = CuSyncMLPGemm<EpilogueOp1, ShapeThreadBlock1, ShapeWarp1, NumStages1, false>;
using CuSyncGemm2 = CuSyncMLPGemm<EpilogueOp2, ShapeThreadBlock2, ShapeWarp2, NumStages2, false>;

// using CuSyncGemmSplitK1 = CuSyncMLPGemm<ProdCuStage, EpilogueOp1, ShapeThreadBlock1, ShapeWarp1, NumStages1, true>;
// using CuSyncGemmSplitK2 = CuSyncMLPGemm<ConsCuStage, EpilogueOp2, ShapeThreadBlock2, ShapeWarp2, NumStages2, true>;

using HostTensor = cutlass::HostTensor<ElementInputA, LayoutInputA>;

enum MLPType {
  GPT3,
  LLaMa    
};

struct MLPParameters {
  HostTensor x; //[B, H]
  HostTensor w1; //[H, 4H/8] in GPT-3
  //xw1 = GeLU(x * w1)
  HostTensor xw1; //[B, 4 H / 8]
  HostTensor xw1_cublas; //[B, 4 H / 8]
  HostTensor w2; //[4H/8, H] in GPT-3 and [H/3, H] in LLaMa
  //xw12 = xw1 * w2
  HostTensor xw12; //[B, H]
  HostTensor xw12_cublas; //[B, H]

  //For LLaMa only
  HostTensor vw1; //[B, 2*H/3] in LLAMA
  HostTensor xvw1; //[B, 2*H/3] in LLaMa
  HostTensor glu; //[B, H/3] in LLaMa

  HostTensor ref_xw1;
  HostTensor ref_xw12;

  //For LLaMa only
  HostTensor ref_xv;

  bool checkResults;

  cutlass::gemm::GemmCoord gemm_size1;
  cutlass::gemm::GemmCoord gemm_size2;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;

  std::string model;

  MLPParameters(std::string model_, uint batch, bool check) {
    alpha = ElementComputeEpilogue(1.0);
    beta = ElementComputeEpilogue(0.0);
    model = model_;

    if (model == "gpt3") {
      gemm_size1 = cutlass::gemm::GemmCoord(batch, 512, 256);
      gemm_size2 = cutlass::gemm::GemmCoord(batch, 512, 256);
      // gemm_size1 = cutlass::gemm::GemmCoord(batch, 14336, 4096);
      // gemm_size2 = cutlass::gemm::GemmCoord(batch, 4096, 14336);
    } else if (model=="llama") {
      int H = 8192;
      int d = ((H/3 + 127)/128)*128;
      gemm_size1 = cutlass::gemm::GemmCoord(batch, 2*d, H);
      gemm_size2 = cutlass::gemm::GemmCoord(batch, H, d);
    }
    std::cout << "GeMM 1 Size: " << gemm_size1.m() << ", " << 
      gemm_size1.n() << ", " << gemm_size1.k() << std::endl;
    std::cout << "GeMM 2 Size: " << gemm_size2.m() << ", " << 
      gemm_size2.n() << ", " << gemm_size2.k() << std::endl;
    
    x = HostTensor(gemm_size1.mk());
    w1 = HostTensor(gemm_size1.kn());
    xw1 = HostTensor(gemm_size1.mn());
    xw1_cublas = HostTensor(gemm_size1.mn());
    w2 = HostTensor(gemm_size2.kn());
    xw12 = HostTensor(gemm_size2.mn());
    xw12_cublas = HostTensor(gemm_size2.mn());
    ref_xw1 = HostTensor(gemm_size1.mn());
    ref_xw12 = HostTensor(gemm_size2.mn());

    if (model == "llama") {
      xvw1 = HostTensor(gemm_size1.mn());
      vw1 = HostTensor(gemm_size1.kn());
      glu = HostTensor(gemm_size2.mk());
      ref_xv = HostTensor(gemm_size1.mn());
    }
    checkResults = check;
  }

  void initIns() {  
    srand(12345);  // 设置随机种子为固定值，确保每次运行结果相同
    if (checkResults) {
      ElementOutput values[5] = {ElementOutput(0.05), ElementOutput(0.06),
                                 ElementOutput(0.01), ElementOutput(0.06),
                                 ElementOutput(0.04)}; // 要让gemm2跑起来，这里必须是1e-2的级别。。。在1e-1就会溢出，得到inf。
      memset_random(x.host_data(), 5, values, x.size());
      memset_random(w1.host_data(), 5, values, w1.size());
      memset_random(w2.host_data(), 5, values, w1.size());


      // cutlass::reference::host::TensorFill(x.host_view(), ElementOutput(0.1));
      // cutlass::reference::host::TensorFill(w1.host_view(), ElementOutput(0.1));
      // cutlass::reference::host::TensorFill(w2.host_view(), ElementOutput(0.1));

      if (model == "llama") {
        memset_random2(vw1.host_data(), ElementOutput(0.01), ElementOutput(0.2), vw1.size());
      }
    } else {
      cutlass::reference::host::TensorFill(x.host_view(), ElementOutput(0.01));
      cutlass::reference::host::TensorFill(w1.host_view(), ElementOutput(0.01));
      cutlass::reference::host::TensorFill(w2.host_view(), ElementOutput(0.01));

      // cutlass::reference::host::TensorFill(x.host_view(), ElementOutput(1));
      // cutlass::reference::host::TensorFill(w1.host_view(), ElementOutput(2));
      // cutlass::reference::host::TensorFill(w2.host_view(), ElementOutput(1));


      if (model == "llama") {
        cutlass::reference::host::TensorFill(vw1.host_view(), ElementOutput(0.5));
      }
    }
    // Copy data from host to GPU
    x.sync_device();
    w1.sync_device();
    w2.sync_device();
    if (model == "llama") {
      vw1.sync_device();
    }
  }
  
  void initOuts() {
    cutlass::reference::host::TensorFill(xw1.host_view());
    cutlass::reference::host::TensorFill(xw1_cublas.host_view());
    cutlass::reference::host::TensorFill(xw12.host_view());
    cutlass::reference::host::TensorFill(xw12_cublas.host_view());

    xw1.sync_device();
    xw1_cublas.sync_device();
    xw12.sync_device();
    xw12_cublas.sync_device();

    if (model == "llama") {
      cutlass::reference::host::TensorFill(xvw1.host_view());
      xvw1.sync_device();
      cutlass::reference::host::TensorFill(glu.host_view());
      glu.sync_device();
    }
  }

  void initRefs() {
    cutlass::reference::host::TensorFill(ref_xw12.host_view());
    cutlass::reference::host::TensorFill(ref_xw1.host_view());

    ref_xw12.sync_device();
    ref_xw1.sync_device();
    if (model == "llama") {
      cutlass::reference::host::TensorFill(ref_xv.host_view());
      ref_xv.sync_device(); 
    }
  }

  bool isGPT3() {return model == "gpt3";}
  bool isLLaMa() {return model == "llama";}
};

/** Reference MLP for correctness check **/
cudaError_t referenceMLP(MLPParameters& mlpParams) {
  ref_matmul<ElementOutput, ElementAccumulator>(mlpParams.gemm_size1.m(), 
                                                mlpParams.gemm_size1.n(), 
                                                mlpParams.gemm_size1.k(),
                                                mlpParams.x.device_data(), 
                                                mlpParams.w1.device_data(), 
                                                mlpParams.ref_xw1.host_data());
  CUDA_CHECK(cudaMemcpy(mlpParams.ref_xw1.device_data(), mlpParams.ref_xw1.host_data(), 
             sizeof(ElementOutput) * mlpParams.ref_xw1.size(), cudaMemcpyHostToDevice));
  
  if (mlpParams.isLLaMa()) {
    printf("check not supported in llama\n");
    return cudaSuccess;
    ref_matmul<ElementOutput, ElementAccumulator>(mlpParams.gemm_size1.m(), 
                                                  mlpParams.gemm_size1.n(), 
                                                  mlpParams.gemm_size1.k(),
                                                  mlpParams.x.device_data(), 
                                                  mlpParams.vw1.device_data(), 
                                                  mlpParams.ref_xv.host_data());
    //Compute XW1 (dot) XV
    for (int b = 0; b < mlpParams.gemm_size1.m(); b++) {
      for (int n = 0; n < mlpParams.gemm_size1.n(); n++) {
        uint index = b * mlpParams.gemm_size1.n() + n;
        mlpParams.ref_xv.host_data()[index] = mlpParams.ref_xw1.host_data()[index] * 
                                              mlpParams.ref_xv.host_data()[index];
      }
    }

    mlpParams.ref_xv.sync_device();

    ref_matmul<ElementOutput, ElementAccumulator>(mlpParams.gemm_size2.m(),
                                                  mlpParams.gemm_size2.n(),
                                                  mlpParams.gemm_size2.k(), 
                                                  mlpParams.ref_xv.device_data(),
                                                  mlpParams.w2.device_data(), 
                                                  mlpParams.ref_xw12.host_data());
  } else {
    ref_matmul<ElementOutput, ElementAccumulator>(mlpParams.gemm_size2.m(),
                                                  mlpParams.gemm_size2.n(),
                                                  mlpParams.gemm_size2.k(), 
                                                  mlpParams.ref_xw1.device_data(),
                                                  mlpParams.w2.device_data(), 
                                                  mlpParams.ref_xw12.host_data());
  }

  return cudaSuccess;
}

cudaError_t checkMLPResults(MLPParameters& mlpParams) {
  ElementOutput* hostC = new ElementOutput[mlpParams.ref_xw1.size()];
  CUDA_CHECK(cudaMemcpy(hostC, mlpParams.xw1.device_data(), 
                        mlpParams.xw1.size() * sizeof(ElementOutput), 
                        cudaMemcpyDeviceToHost));
  printf("Checking first GeMM\n");
  bool eq = equals(mlpParams.ref_xw1.size(), mlpParams.ref_xw1.host_data(), hostC, 1);
  printf("GEMM0-cutlass-Expected first element: %f, My Received first element: %f\n", static_cast<float>(mlpParams.ref_xw1.host_data()[0]), static_cast<float>(hostC[0]));
  if (eq == false) {
    printf("First GeMM not correct\n");
    printf("Expected first element: %f, Received first element: %f\n",
           static_cast<float>(mlpParams.ref_xw1.host_data()[0]),  // 假设 ElementOutput 可以转换为 float 进行打印
           static_cast<float>(hostC[0]));
    // return cudaErrorUnknown;
  }
  else{
    printf("First GeMM passed\n");
  }


  ElementOutput* hostE = new ElementOutput[mlpParams.ref_xw12.size()];
  CUDA_CHECK(cudaMemcpy(hostE, mlpParams.xw12.device_data(), 
                        mlpParams.xw12.size() * sizeof(ElementOutput), 
                        cudaMemcpyDeviceToHost));
  //For LLaMa not checking XV
  printf("Checking second GeMM\n");
  eq = equals(mlpParams.ref_xw12.size(), mlpParams.ref_xw12.host_data(), hostE, 1);
  printf("GEMM1-cutlass-Expected first element: %f, My-Received first element: %f\n", static_cast<float>(mlpParams.ref_xw12.host_data()[0]), static_cast<float>(hostE[0]));
  if (eq == false) {
    printf("Second GeMM not correct \n");
    // return cudaErrorUnknown;
  }
  else{
    printf("Second GeMM passed\n");   // 现在暂时只测kernel0，所以这个先注释掉哈
  }


  return cudaSuccess;
}


cudaError_t checkMLPResults_cublas(MLPParameters& mlpParams) {

    ElementOutput* hostC = new ElementOutput[mlpParams.xw1.size()];
    CUDA_CHECK(cudaMemcpy(hostC, mlpParams.xw1.device_data(), 
                          mlpParams.xw1.size() * sizeof(ElementOutput), 
                          cudaMemcpyDeviceToHost));

    ElementOutput* hostC_cublas = new ElementOutput[mlpParams.xw1_cublas.size()];
    CUDA_CHECK(cudaMemcpy(hostC_cublas, mlpParams.xw1_cublas.device_data(), 
                          mlpParams.xw1_cublas.size() * sizeof(ElementOutput), 
                          cudaMemcpyDeviceToHost));

    printf("Checking first GeMM\n");
    bool eq = equals(mlpParams.xw1_cublas.size(), hostC_cublas, hostC, 1e-1f);
    printf("My Expected first element: %f, cublas first element: %f\n",
               static_cast<float>(hostC[0]), static_cast<float>(hostC_cublas[0]));
    if (!eq) {
        printf("First GeMM not correct\n");
        printf("Expected first element: %f, Received first element: %f\n",
               static_cast<float>(hostC[0]), static_cast<float>(hostC_cublas[0]));
        return cudaErrorUnknown;
    }
    printf("cublas First GeMM passed\n");

    ElementOutput* hostE = new ElementOutput[mlpParams.xw12.size()];
    CUDA_CHECK(cudaMemcpy(hostE, mlpParams.xw12.device_data(), 
                          mlpParams.xw12.size() * sizeof(ElementOutput), 
                          cudaMemcpyDeviceToHost));

    ElementOutput* hostE_cublas = new ElementOutput[mlpParams.xw12_cublas.size()];
    CUDA_CHECK(cudaMemcpy(hostE_cublas, mlpParams.xw12_cublas.device_data(), 
                          mlpParams.xw12_cublas.size() * sizeof(ElementOutput), 
                          cudaMemcpyDeviceToHost));
    printf("Checking second GeMM\n");
    eq = equals(mlpParams.xw12_cublas.size(), hostE_cublas, hostE, 1e-1f);
    if (!eq) {
        printf("Second GeMM not correct\n");
        return cudaErrorUnknown;
    }
    printf("cublas Second GeMM passed\n");
    return cudaSuccess;
}




/*GPT3 Baseline MLP*/
template<typename GemmTy1, typename GemmTy2>
cudaError_t runBaselineGPT3(int split_k1, int split_k2, 
                            MLPParameters& mlpParams,
                            cudaStream_t stream,
                            double& execTime, double& matmul1Time, double& softmaxTime, double& matmul2Time,
                            int iters = 100) {
  //Setup first GeMM
  typename GemmTy1::Arguments args1 {
    mlpParams.gemm_size1,
    mlpParams.x.device_ref(), 
    mlpParams.w1.device_ref(),
    mlpParams.xw1.device_ref(),
    mlpParams.xw1.device_ref(),
    {mlpParams.alpha, mlpParams.beta},
    split_k1};

  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  GemmTy1 gemm_op1;
  cutlass::Status status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  //Setup Second GeMM
  typename GemmTy2::Arguments args2{ 
    mlpParams.gemm_size2, 
    mlpParams.xw1.device_ref(), 
    mlpParams.w2.device_ref(), 
    mlpParams.xw12.device_ref(), 
    mlpParams.xw12.device_ref(), 
    {mlpParams.alpha, mlpParams.beta},         
    split_k2};
  
  GemmTy2 gemm_op2;
  workspace_size = GemmTy2::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);

  execTime = 0;
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(start, 0));

  for (int r = 0; r < iters; r++) {    
    status = gemm_op1(stream);  
    status = gemm_op2(stream);//为了检测方便，暂时注释掉
  }

  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUTLASS_CHECK(status);
  float time_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, end));
  execTime += time_ms*1000.0f;
  printf("Cutlass avg run time: %f\n", execTime/iters);
  return cudaSuccess;
}

cudaError_t runBaselineGPT3(int split_k1, int split_k2, 
                        MLPParameters& mlpParams,
                        cudaStream_t stream,
                        double& execTime,
                        double& matmul1Time,
                        double& softmaxTime,
                        double& matmul2Time,
                        int iters = 100) {
  cudaError_t result;
  execTime = 0;
  matmul1Time = 0;
  softmaxTime = 0;
  matmul2Time = 0;
  if (split_k1 == 1 && split_k2 == 1) {
    result = runBaselineGPT3<Gemm1, Gemm2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else if (split_k1 > 1 && split_k2 == 1) {
    result = runBaselineGPT3<GemmSplitK1, Gemm2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else if (split_k1 == 1 && split_k2 > 1) {
    result = runBaselineGPT3<Gemm1, GemmSplitK2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  } else {
    result = runBaselineGPT3<GemmSplitK1, GemmSplitK2>(split_k1, split_k2, mlpParams, stream, execTime, matmul1Time, softmaxTime, matmul2Time, iters);
  }

  return result;
}


void run_cublasGPT3(
    MLPParameters& mlpParams, int iterations
) {
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    // First GEMM: xw1 = x * w1

    int m = mlpParams.gemm_size1.m();
    int n = mlpParams.gemm_size1.n();
    int k = mlpParams.gemm_size1.k();
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    // 确保您的矩阵数据是 half 类型
    half *d_A1 = reinterpret_cast<half*>(mlpParams.x.device_data());  // 转换为 half*
    half *d_B1 = reinterpret_cast<half*>(mlpParams.w1.device_data()); // 转换为 half*
    half *d_C1 = reinterpret_cast<half*>(mlpParams.xw1_cublas.device_data()); // 转换为 half*

    // 执行矩阵乘法
    // cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, d_B1, k, d_A1, m, &beta, d_C1, n);


    float execTime = 0;
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start, 0));

    for (int r = 0; r < iterations; r++) {
      // 调用 cublasGemmEx 来进行矩阵乘法
      CHECK_CUBLAS_ERROR(cublasGemmEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_T,       // 转置 A 和 B
          n, m, k,                        // 转置后的维度
          &alpha,                         // 乘法的 alpha 系数
          d_B1, CUDA_R_16F, k,            // B 矩阵和其 leading dimension
          d_A1, CUDA_R_16F, m,            // A 矩阵和其 leading dimension
          &beta,                          // 加法的 beta 系数
          d_C1, CUDA_R_16F, n,            // C 矩阵和其 leading dimension
          CUDA_R_16F,                     // 计算使用的数据类型
          CUBLAS_GEMM_DEFAULT_TENSOR_OP   // 使用默认算法，允许 Tensor Cores
      ));


      // cudaDeviceSynchronize();
      int m2 = mlpParams.gemm_size2.m();
      int n2 = mlpParams.gemm_size2.n();
      int k2 = mlpParams.gemm_size2.k();

      // 注意这里假设您已经确认过这些矩阵都是half类型
      // half *d_A2 = reinterpret_cast<half*>(mlpParams.xw1_cublas.device_data()); // xw1 作为第二次 GEMM 的 A 矩阵
      half *d_A2 = d_C1;
      half *d_B2 = reinterpret_cast<half*>(mlpParams.w2.device_data());         // w2 作为第二次 GEMM 的 B 矩阵
      half *d_C2 = reinterpret_cast<half*>(mlpParams.xw12_cublas.device_data()); // xw12 作为第二次 GEMM 的 C 矩阵


      // 矩阵乘法 C = A * B
      // cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n2, m2, k2, &alpha, d_B2, k2, d_A2, m2, &beta, d_C2, n2);

      CHECK_CUBLAS_ERROR(cublasGemmEx(
          handle,
          CUBLAS_OP_T, CUBLAS_OP_T,  // 两个矩阵都不转置
          n2, m2, k2,                // 矩阵维度
          &alpha,                    // 乘法因子
          d_B2, CUDA_R_16F, k2,      // B 矩阵和其前导维度
          d_A2, CUDA_R_16F, m2,      // A 矩阵和其前导维度
          &beta,                     // 加法因子
          d_C2, CUDA_R_16F, n2,      // C 矩阵和其前导维度
          CUDA_R_16F,                // 计算的数据类型
          CUBLAS_GEMM_DEFAULT_TENSOR_OP  // 使用默认算法，可能包括 Tensor Cores
      ));
    }

    CUDA_CHECK(cudaEventRecord(end, 0));
    CUDA_CHECK(cudaEventSynchronize(end));
    float time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, end));
    execTime += time_ms*1000.0f;
    printf("cublas avg run time: %f\n", execTime/iterations);

    // Clean up
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
}

// template <typename Operator>
// __device__
// void GEMMdeviceFunction_cons(typename Operator::Params<ConsCuStage> params, int blx, int bly) {
//   // Dynamic shared memory base pointer
//   extern __shared__ int SharedStorageBase[];

//   // Declare pointer to dynamic shared memory.
//   typename Operator::SharedStorage *shared_storage =
//       reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

//   Operator op;
//   op(params, *shared_storage, blx, bly);
// }


// template <typename Operator>
// __device__
// void GEMMdeviceFunction_prod(typename Operator::Params<ProdCuStage> params, int blx, int bly) {
//   // Dynamic shared memory base pointer
//   extern __shared__ int SharedStorageBase[];

//   // Declare pointer to dynamic shared memory.
//   typename Operator::SharedStorage *shared_storage =
//       reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

//   Operator op;
//   op(params, *shared_storage, blx, bly);
// }

template <typename Operator>
__device__
void GEMMdeviceFunction_cons(typename Operator::Params<ConsCuStage> params, dim3 local_exec) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;
  op(params, *shared_storage, local_exec);
}


template <typename Operator>
__device__
void GEMMdeviceFunction_prod(typename Operator::Params<ProdCuStage> params, dim3 local_exec) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];

  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;
  op(params, *shared_storage, local_exec);
}

/// Generic CUTLASS kernel template.
template <typename Operator>
__global__
void AllKernel(typename Operator::Params<ConsCuStage> cons_params, typename Operator::Params<ProdCuStage> prod_params, dim3* exec_array) {
// void AllKernel(cutlass::gemm::kernel::BaseParams **params_array, int num_params) { // 以后可以这样改来写的更漂亮。
// 出去修改param。预设置空的blx和bly，然后在这里用tuple.first和tuple.second来设置。
  if(threadIdx.x==0&&threadIdx.y==0&&blockIdx.x==0&&blockIdx.y==0&&blockIdx.z==0){
    printf("enter AllKernel\n");
  }
  dim3 this_block_exec = exec_array[blockIdx.x]; // 注意我们只发射一维网格！但是问题可以是二维的。

  if(blockIdx.x>=0 && blockIdx.x<4){
    GEMMdeviceFunction_prod<Operator>(prod_params, this_block_exec);
  }
  else{
    GEMMdeviceFunction_cons<Operator>(cons_params, this_block_exec);
  }
  // if (this_block_exec.z==1) {
  //     if(threadIdx.x==0&&threadIdx.y==0){
  //       printf("cons-blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, this_block_exec=(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, this_block_exec.x, this_block_exec.y, this_block_exec.z);
  //     }
  //     GEMMdeviceFunction_cons<Operator>(cons_params, this_block_exec);
  // } else if (this_block_exec.z==0) {
  //     if(threadIdx.x==0&&threadIdx.y==0){
  //       printf("prod-blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d, this_block_exec=(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, this_block_exec.x, this_block_exec.y, this_block_exec.z);
  //     }
  //     GEMMdeviceFunction_prod<Operator>(prod_params, this_block_exec);
  // } else {
  //     printf("Error: out of block range\n");
  // }

  // if(blockIdx.x>=prod_params.block_range_down && blockIdx.x<prod_params.block_range_up){
  //   GEMMdeviceFunction_prod<Operator>(prod_params);
  // }
  // else if(blockIdx.x>=cons_params.block_range_down && blockIdx.x<cons_params.block_range_up){
  //   GEMMdeviceFunction_cons<Operator>(cons_params);
  // }
  // else{
  //   printf("Error out of block range\n");
  // }
}

// 读取文件中特定行的函数
std::string readOrderLine(const std::string& filePath, int orderLine) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file");
    }

    std::string line;
    int currentLine = 0;
    while (std::getline(file, line)) {
        currentLine++;
        if (currentLine == orderLine) {
            return line;
        }
    }
    throw std::runtime_error("Error reading file or line number out of range");
}

// 提取方括号中的内容并转换为dim3数组
dim3* extractOrderContent(const std::string& line, int& array_size) {
    size_t start = line.find("[");
    size_t end = line.rfind("]");
    if (start == std::string::npos || end == std::string::npos || start >= end) {
        throw std::runtime_error("Invalid order line format");
    }
    std::string content = line.substr(start + 1, end - start - 1);
    // std::cout << "Extracted content: " << content << std::endl;  // 添加调试信息

    std::vector<dim3> orders;
    std::regex regex_pattern(R"(\((\d+),\s*(\d+),\s*(\d+)\))");
    std::smatch match;

    std::string::const_iterator search_start(content.cbegin());
    while (std::regex_search(search_start, content.cend(), match, regex_pattern)) {
        int x = std::stoi(match[1]);
        int y = std::stoi(match[2]);
        int z = std::stoi(match[3]);
        orders.push_back(dim3(x, y, z));
        search_start = match.suffix().first;
    }

    array_size = orders.size();  // 更新 array_size

    dim3* exec_seq = new dim3[array_size];
    for (size_t i = 0; i < orders.size(); ++i) {
        exec_seq[i] = orders[i];
    }

    return exec_seq;
}

/*CuSync GPT-3 MLP*/
template<typename GemmTy1, typename GemmTy2>
cudaError_t runCuSyncGPT3(int split_k1, int split_k2,
                          MLPParameters& mlpParams,
                          ProdCuStage& prod, ConsCuStage& cons,
                          cudaStream_t producer_stream, 
                          cudaStream_t consumer_stream,
                          double& execTime,
                          int iters,
                          int order_line, 
                          std::string filePath) {



  // typename GemmTy1::Arguments args1{prod,
  typename GemmTy1::Arguments args1{mlpParams.gemm_size1,
                                     mlpParams.x.device_ref(),
                                     mlpParams.w1.device_ref(),
                                     mlpParams.xw1.device_ref(),
                                     mlpParams.xw1.device_ref(),
                                     {mlpParams.alpha, mlpParams.beta},         
                                     split_k1};
  GemmTy1 gemm_op1;
  size_t workspace_size = GemmTy1::get_workspace_size(args1);
  cutlass::device_memory::allocation<uint8_t> workspace1(workspace_size);
  cutlass::Status status = gemm_op1.can_implement(args1);
  CUTLASS_CHECK(status);
  status = gemm_op1.initialize(args1, workspace1.get());
  CUTLASS_CHECK(status);

  // typename GemmTy2::Arguments args2{cons,
  typename GemmTy2::Arguments args2{mlpParams.gemm_size2,  
                                    mlpParams.xw1.device_ref(),
                                    mlpParams.w2.device_ref(),
                                    mlpParams.xw12.device_ref(),
                                    mlpParams.xw12.device_ref(),
                                    {mlpParams.alpha, mlpParams.beta},
                                    split_k2};

  GemmTy2 gemm_op2;
  workspace_size = GemmTy2::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace2(workspace_size);
  status = gemm_op2.can_implement(args2);
  CUTLASS_CHECK(status);
  status = gemm_op2.initialize(args2, workspace2.get());
  CUTLASS_CHECK(status);




  // status = gemm_op1.run(true, NULL, producer_stream);
  CUTLASS_CHECK(status);

  // CUDA_CHECK(cudaDeviceSynchronize());

  /// Operator class tag
  // using OperatorClass_ = cutlass::arch::OpClassTensorOp; // 这里一开始总是报错，关键是要加上cutlass::arch，以及要include相应文件，就去文件里找，这个对应的是nvcutlass底下arch的mma.h，所以include就好。(可以直接用绝对路径)。另一种方法，因为wmma.h里面也include了同一个mma.h。其实include wmma.h也是可以的。（后来发现前面又MMAOp是一样的内容）

  // Access granularity of A matrix in units of elements
  static constexpr int AlignmentA =
      cutlass::gemm::device::DefaultGemmConfiguration<MMAOp, SmArch, ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>::kAlignmentA;

// constexpr 的作用
// constexpr 关键字用于声明在编译时可求值的常量表达式。它的主要作用包括：

// 编译期求值：确保表达式在编译期就能求值，这对模板参数尤其重要，因为模板参数必须在编译时确定。
// 常量传播：编译器可以在编译期进行常量传播和优化，这可以提升程序的性能。
// 安全性：使用 constexpr 可以确保这些常量在整个程序生命周期中都是不可变的，提高了代码的安全性和可维护性。--->没有这个会报错说这个不是常量。。。原先是在device/cusyncgemm.h里面的template。template是在编译时就自动确定下来的。这里要是直接写int AlignmentA，就是在运行时确定。不过加上constexpr就OK啦

  // Access granularity of B matrix in units of elements
  static constexpr int AlignmentB =
      cutlass::gemm::device::DefaultGemmConfiguration<MMAOp, SmArch, ElementInputA, ElementInputB, ElementOutput, ElementAccumulator>::kAlignmentB;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;

  using Operator_ =
      cutlass::gemm::device::DefaultGemmConfiguration<MMAOp, SmArch, ElementInputA, ElementInputB,ElementOutput, ElementAccumulator>::Operator;

  static constexpr auto SharedClearOption = cutlass::gemm::SharedMemoryClearOption::kNone;
  using PermuteDLayout = cutlass::layout::NoPermute;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultCuSyncGemm<ElementInputA, LayoutInputA, kAlignmentA, ElementInputB, LayoutInputB, kAlignmentB, ElementOutput, LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeThreadBlock1, ShapeWarp1, ShapeMMAOp, EpilogueOp1, CuSyncGeMMSwizzle, NumStages1, false, Operator_, SharedClearOption, false, false, false, PermuteDLayout >::GemmKernel;

  CuSyncGeMMSwizzle cuSyncGeMMSwizzle;

  cutlass::gemm::GemmCoord grid_shape1 = cuSyncGeMMSwizzle.get_tiled_shape(
    args1.problem_size, 
    {ShapeThreadBlock1::kM, ShapeThreadBlock1::kN, ShapeThreadBlock1::kK},
    args1.split_k_slices);

  cutlass::gemm::GemmCoord grid_shape2 = cuSyncGeMMSwizzle.get_tiled_shape(
    args2.problem_size, 
    {ShapeThreadBlock2::kM, ShapeThreadBlock2::kN, ShapeThreadBlock2::kK},
    args2.split_k_slices);

  typename GemmKernel::Params<ProdCuStage> prod_params{prod, args1.problem_size, grid_shape1, args1.ref_A.non_const_ref(), args1.ref_B.non_const_ref(), args1.ref_C.non_const_ref(), args1.ref_D, args1.epilogue, reinterpret_cast<int *>(workspace1.get()), args1.gather_A_indices, args1.gather_B_indices, args1.scatter_D_indices, 0, grid_shape1.m()*grid_shape1.n(), 2, 0};

  typename GemmKernel::Params<ConsCuStage> cons_params{cons, args2.problem_size, grid_shape2, args2.ref_A.non_const_ref(), args2.ref_B.non_const_ref(), args2.ref_C.non_const_ref(), args2.ref_D, {mlpParams.alpha, mlpParams.beta}, reinterpret_cast<int *>(workspace2.get()), args2.gather_A_indices, args2.gather_B_indices, args2.scatter_D_indices, grid_shape1.m()*grid_shape1.n(), grid_shape1.m()*grid_shape1.n()+grid_shape2.n()*grid_shape2.m(), 2, 1};


  // 创建包含 prod_params 和 cons_params 的数组
  // cutlass::gemm::kernel::BaseParams* params_array[] = { &prod_params, &cons_params }; 
  // GemmKernel::Params<ProdCuStage> params_array[] = { prod_params};
  // GemmKernel::Params<ProdCuStage> params_array = prod_params;
  // cutlass::gemm::kernel::BaseParams* params_array[] = { &prod_params};

  // dim3 grid = cuSyncGeMMSwizzle.get_grid_shape(params_.grid_tiled_shape);
  // dim3 block(GemmKernel::kThreadCount, 1, 1);

  printf("grid_shape1.m=%d, grid_shape1.n=%d,grid_shape1.k=%d,grid_shape2.m=%d, grid_shape2.n=%d,grid_shape2.k=%d\n", grid_shape1.m(),grid_shape1.n(),grid_shape1.k(),grid_shape2.m(),grid_shape2.n(),grid_shape2.k());

  dim3 grid = {grid_shape1.m()*grid_shape1.n()+grid_shape2.n()*grid_shape2.m(), 1, 1};
  // dim3 grid = {grid_shape1.m()*grid_shape1.n(), 1, 1};
  // dim3 block = {128, 1, 1};
  dim3 block(GemmKernel::kThreadCount, 1, 1);
  // int smem_size = 99 << 10;  // ????
  // int smem_size = 8 << 10;  // ????
  int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
  printf("GemmKernel::kThreadCount=%d, smem_size=%d\n",GemmKernel::kThreadCount, smem_size);

  cudaFuncSetAttribute(AllKernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  printf("line 998\n");

  // std::string line = readOrderLine(filePath, order_line);
  // int array_size = grid_shape1.m()*grid_shape1.n()+grid_shape2.n()*grid_shape2.m();

  // auto exec_seq = extractOrderContent(line, array_size);

  // dim3 exec_seq[16] = {
  //     dim3(0, 0, 0), dim3(1, 0, 0), dim3(2, 0, 0), dim3(3, 0, 0),
  //     dim3(0, 1, 0), dim3(1, 1, 0), dim3(2, 1, 0), dim3(3, 1, 0),
  //     dim3(0, 2, 0), dim3(1, 2, 0), dim3(2, 2, 0), dim3(3, 2, 0),
  //     dim3(0, 3, 0), dim3(1, 3, 0), dim3(2, 3, 0), dim3(3, 3, 0)
  // }; // origin policy，直接算完一整行。
  // dim3 exec_seq[16] = {
  //     dim3(0, 0, 0), dim3(1, 0, 0), dim3(0, 1, 0), dim3(1, 1, 0),
  //     dim3(0, 2, 0), dim3(1, 2, 0), dim3(0, 3, 0), dim3(1, 3, 0),
  //     dim3(2, 0, 0), dim3(3, 0, 0), dim3(2, 1, 0), dim3(3, 1, 0),
  //     dim3(2, 2, 0), dim3(3, 2, 0), dim3(2, 3, 0), dim3(3, 3, 0)
  // };  // 基础Z字形
  // dim3 exec_seq[16] = {
  //     dim3(0, 0, 0), dim3(1, 0, 0), dim3(0, 1, 0), dim3(1, 1, 0),
  //     dim3(2, 0, 0), dim3(3, 0, 0), dim3(2, 1, 0), dim3(3, 1, 0),
  //     dim3(0, 2, 0), dim3(1, 2, 0), dim3(0, 3, 0), dim3(1, 3, 0),
  //     dim3(2, 2, 0), dim3(3, 2, 0), dim3(2, 3, 0), dim3(3, 3, 0)
  // };  // 嵌套Z字形
  // dim3 exec_seq[8] = {
  //     dim3(0, 0, 0), dim3(1, 0, 0), dim3(0, 1, 0), dim3(1, 1, 0),
  //     dim3(0, 0, 1), dim3(1, 0, 1), dim3(0, 1, 1), dim3(1, 1, 1)
  // };  // 嵌套Z字形
  // dim3 exec_seq[8] = {
  //   dim3(0, 0, 0), dim3(0, 1, 0),
  //   dim3(0, 2, 0), dim3(0, 3, 0),
  //   dim3(0, 4, 1), dim3(0, 5, 1),
  //   dim3(0, 6, 1), dim3(0, 7, 1)
  // };
  dim3 exec_seq[8] = {
    dim3(0, 0, 0), dim3(1, 0, 0),
    dim3(2, 0, 0), dim3(3, 0, 0),
    dim3(4, 0, 1), dim3(5, 0, 1),
    dim3(6, 0, 1), dim3(7, 0, 1)
  };
  int array_size = 8;
  dim3* d_exec_seq;
  cudaMalloc(&d_exec_seq, sizeof(dim3) * array_size);

  // 将数据从CPU复制到GPU
  cudaMemcpy(d_exec_seq, exec_seq, sizeof(dim3) * array_size, cudaMemcpyHostToDevice);
  printf("line 1037\n");

  execTime = 0;
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(start, 0));

  for (int r = 0; r < iters; r++) {
    AllKernel<GemmKernel><<<grid, block, smem_size>>>(cons_params, prod_params, d_exec_seq); 
  }

  CUDA_CHECK(cudaEventRecord(end, 0));
  CUDA_CHECK(cudaEventSynchronize(end));
  CUTLASS_CHECK(status);
  float time_ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, end));
  execTime += time_ms*1000.0f;
  printf("MyKernel cuSync avg run time: %f\n", execTime/iters);
  return cudaSuccess;
}

cudaError_t runCuSyncGPT3(int split_k1, int split_k2, MLPParameters& mlpParams,
                          ProdCuStage& prod, ConsCuStage& cons,
                          cudaStream_t producer_stream, cudaStream_t consumer_stream,
                          double& execTime, int iters, int order_line,
                          std::string filePath) {
  cudaError_t result;
  execTime = 0;

  if (split_k1 == 1 && split_k2 == 1) {
    result = runCuSyncGPT3<CuSyncGemm1, CuSyncGemm2>(split_k1, split_k2, mlpParams, prod, cons, producer_stream, consumer_stream, execTime, iters, order_line, filePath);
  }
  return result;
}

// Utility function to save matrix data to file
void saveMatrixToFile(const std::vector<float>& matrix, const int rows, const int cols, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i * cols + j] << (j < cols - 1 ? ", " : "\n");
        }
    }

    file.close();
}
int run(int argc, char* argv[]) {
  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  const uint NUM_ARGS = 8;  // 更新参数数量
  std::string argNames[NUM_ARGS] = {"--model", "--batch", "--check", "--split-k1", "--split-k2", "--policy", "--order-line", "--file-path"};
  std::string argHelp[NUM_ARGS] = {"GPT3 or LLaMa", "Batch size", "Check results", 
                                    "Split K for first GeMM", "Split K for second GeMM",
                                    "Policy to execute", "Line number of orders file", "Path to the output file"};

  
  if (argc < NUM_ARGS + 1) {
      std::cout << "usage: " << std::endl
                << argNames[0] << " gpt3|llama " << argHelp[0] << std::endl 
                << argNames[1] << " <int> " << argHelp[1] << std::endl
                << argNames[2] << " true|false " << argHelp[2] << std::endl
                << argNames[3] << " <int> " << argHelp[3] << std::endl
                << argNames[4] << " <int> " << argHelp[4] << std::endl
                << argNames[5] << " baseline|cusync " << argHelp[5] << std::endl
                << argNames[6] << " <int> " << argHelp[6] << std::endl
                << argNames[7] << " <string> " << argHelp[7] << std::endl;
      return 0;
  }

  std::string model = "", policy = "", filePath = "";
  uint batch = 0;
  bool doChecking = false;
  uint split_k1 = 1;
  uint split_k2 = 1;
  int order_line = -1;  // 添加默认值用于未提供参数时的检查

  for (int i = 1; i < argc; ++i) {
    std::string arg = std::string(argv[i]);
    if (arg.find(argNames[0]) == 0) {
      model = std::string(argv[i+1]);
      i = i + 1;
    } else if (arg.find(argNames[1]) == 0) {
      std::stringstream ss(argv[i+1]);
      ss >> batch;
      i = i + 1;
    } else if (arg.find(argNames[2]) == 0) {
      std::string val = std::string(argv[i+1]);
      if (val == "true") {
        doChecking = true;
      } else if (val == "false") {
        doChecking = false;
      } else {
        std::cout << "Invalid value for check " << val << std::endl;
      }
      i = i + 1;
    } else if (arg.find(argNames[3]) == 0) {
      split_k1 = atoi(argv[i+1]);
      i=i+1;
    } else if (arg.find(argNames[4]) == 0) {
      split_k2 = atoi(argv[i+1]);
      i=i+1;
    } else if (arg.find(argNames[5]) == 0) {
      policy = std::string(argv[i+1]);
      i=i+1;
    } else if (arg.find(argNames[6]) == 0) {
      order_line = std::stoi(argv[i+1]);
      i = i + 1;
  } else if (arg.find(argNames[7]) == 0) {
    filePath = std::string(argv[i+1]);
    i = i + 1;
  } else if (arg.find(argNames[7]) == 0) {
    filePath = std::string(argv[i+1]);
    i = i + 1;
  }
}
    
  std::cout << "model=" << model << " batch=" << batch << " check="<<doChecking << " policy= " << policy << std::endl;

  cudaStream_t producer_stream;
  cudaStream_t producer_stream2;
  cudaStream_t consumer_stream;
  CUDA_CHECK(cudaStreamCreate(&producer_stream));
  CUDA_CHECK(cudaStreamCreate(&producer_stream2));
  CUDA_CHECK(cudaStreamCreate(&consumer_stream));

  MLPParameters mlpParams(model, batch, doChecking);
  mlpParams.initIns();
  mlpParams.initOuts();
  mlpParams.initRefs();
  
  cudaError_t result;
  int epochs = 1;
  int warmup = 1;

  if (doChecking) {
    //Run our reference MLP
    result = referenceMLP(mlpParams);
    if (result != cudaSuccess) {
      printf("Reference MLP failed\n");
      return 1;
    }
    printf("Reference MLP passed\n");
  }

  //Run baseline MLP
  double baselineTime = 0;
  double matmul1Time = 0;
  double softmaxTime = 0;
  double matmul2Time = 0;

  if (policy == "baseline") {
  if (mlpParams.isGPT3()) {
    result = runBaselineGPT3(split_k1, split_k2, mlpParams, producer_stream, 
                             baselineTime, matmul1Time, softmaxTime, matmul2Time, 1);

    CUDA_CHECK(cudaDeviceSynchronize());

    if (doChecking) {
      result = checkMLPResults(mlpParams);
      if (result != cudaSuccess) {
        return 1;
      }
    }

    result = runBaselineGPT3(split_k1, split_k2, mlpParams, producer_stream, 
                             baselineTime, matmul1Time, softmaxTime, matmul2Time, warmup);

    CUDA_CHECK(cudaDeviceSynchronize());
    printf("START-BASELINE:\n");
    result = runBaselineGPT3(split_k1, split_k2, mlpParams, producer_stream, 
                         baselineTime, matmul1Time, softmaxTime, matmul2Time, epochs);
    CUDA_CHECK(result);
    printf("END-BASELINE:\n");
    printf("Average time %lf microseconds\n", baselineTime/(float)epochs);
  } 
  }

  if (doChecking) {
    mlpParams.initOuts();
  }
  //Setup cusync gemm
  cutlass::gemm::GemmCoord tileSizeCoord1{ShapeThreadBlock1::kM, ShapeThreadBlock1::kN, 1};
  cutlass::gemm::GemmCoord tileSizeCoord2{ShapeThreadBlock2::kM, ShapeThreadBlock2::kN, 1};

  cutlass::gemm::GemmCoord gridDim1 = CuSyncGeMMSwizzle().get_tiled_shape(mlpParams.gemm_size1, tileSizeCoord1, split_k1);
  cutlass::gemm::GemmCoord gridDim2 = CuSyncGeMMSwizzle().get_tiled_shape(mlpParams.gemm_size2, tileSizeCoord2, split_k2);

#if defined(ROWSYNC)
  using Sync = RowSync<ShapeThreadBlock1::kM>;
  Sync sync(gridDim1.n());
#elif defined(TILEBATCH)
  using Sync = TileSync<2>;
  Sync sync;
#elif defined(TILESYNC)
  Sync sync;
#elif defined(BATCHEDROW)
  using Sync = BatchedRowSync;
  BatchedRowSync sync(gridDim1.n(), 1);
#else
  #error "Unknown Policy"
#endif

  int highestPriority;
  int lowestPriority;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowestPriority, &highestPriority));
  CUDA_CHECK(cudaStreamCreateWithPriority(&consumer_stream, 0, lowestPriority));
  cudaStream_t streams[(lowestPriority - highestPriority + 1)];
  for (int i = highestPriority; i <= lowestPriority; i++) {
    CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i - highestPriority], 0, i));
  }
  
  //Run cusync mlp
  if (policy == "cusync") {
  if (mlpParams.isGPT3()) {

    dim3 gridDim_1 = CuSyncGeMMSwizzle().get_grid_shape(gridDim1);

    // 打印 .x, .y, .z 的值
    printf("gridDim1.x: %u\n", gridDim1.m());
    printf("gridDim1.y: %u\n", gridDim1.n());
    printf("gridDim1.z: %u\n", gridDim1.k());

    ProdCuStage prod(CuSyncGeMMSwizzle().get_grid_shape(gridDim1), {1,1,1}, NoSync(), sync);
    ConsCuStage cons(CuSyncGeMMSwizzle().get_grid_shape(gridDim2), {1,1,1}, sync, NoSync());

    CuSync::setProducerConsumerPair(prod, cons);
    
    double overlapTime = 0;

    // result = runCuSyncGPT3(split_k1, split_k2, mlpParams, prod, cons, producer_stream, consumer_stream, overlapTime, 1, order_line, filePath);
    

    CUDA_CHECK(cudaDeviceSynchronize());

    // if (doChecking) {
    //   result = checkMLPResults(mlpParams);
    //   if (result != cudaSuccess) {
    //     return 1;
    //   }
    // }

    // result = runCuSyncGPT3(split_k1, split_k2, mlpParams, prod, cons, producer_stream, consumer_stream, overlapTime, warmup, order_line, filePath);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("START-OVERLAPPED:\n");
    
    result = runCuSyncGPT3(split_k1, split_k2, mlpParams, prod, cons, producer_stream, consumer_stream, overlapTime, epochs, order_line, filePath);
    if (doChecking) {
      result = checkMLPResults(mlpParams);
      if (result != cudaSuccess) {
        return 1;
      }
    }
    CUDA_CHECK(result);
    printf("END-OVERLAPPED:\n");
    
    CUDA_CHECK(cudaDeviceSynchronize());
    run_cublasGPT3(mlpParams, epochs); 
    checkMLPResults_cublas(mlpParams);

    printf("Average time %lf microseconds\n", overlapTime/(float)epochs);
  }
  }

  return 0;
}
