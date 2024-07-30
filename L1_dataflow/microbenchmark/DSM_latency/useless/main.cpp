/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
* This sample implements 64-bin histogram calculation
* of arbitrary-sized 8-bit data array
*/

// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// project include
#include "dsm_common.h"

const int numRuns = 16;

int main(int argc, char **argv) {
  uint *d_Data;
  uint arraySize = 1024;
  StopWatchInterface *hTimer = NULL;
  int PassFailFlag = 1;
  uint uiSizeMult = 1;

  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;


  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  // int dev = findCudaDevice(argc, (const char **)argv);

  // checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  // printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
  //        deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major,
  //        deviceProp.minor);

  sdkCreateTimer(&hTimer);

  // Optional Command-line multiplier to increase size of array to histogram
  if (checkCmdLineFlag(argc, (const char **)argv, "sizemult")) {
    uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
    uiSizeMult = MAX(1, MIN(uiSizeMult, 10));
    arraySize *= uiSizeMult;
  }

  checkCudaErrors(cudaMalloc((void **)&d_Data, arraySize * sizeof(uint)));

  {
    printf("Measure latency of SM to SM for %u bytes (%u runs)...\n\n",
           arraySize, numRuns);

    printf("Benchmarking time...\n");
    for (int iter = -1; iter < numRuns; iter++) {
      // iter == -1 -- warmup iteration
      if (iter == 0) {
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);
      }

      dsm_sm2sm_latency(d_Data, arraySize);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&hTimer);
    double dAvgSecs =
        1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
    printf("dsm_sm2sm_latency() time (average) : %.5f sec\n\n", dAvgSecs);
  }

  // {
  //   // printf("Measure throughput of SM to SM for %u bytes (%u runs)...\n\n", arraySize, numRuns);

  //   for (int iter = -1; iter < numRuns; iter++) {
  //     // iter == -1 -- warmup iteration
  //     if (iter == 0) {
  //       checkCudaErrors(cudaDeviceSynchronize());
  //       sdkResetTimer(&hTimer);
  //       sdkStartTimer(&hTimer);
  //     }

  //     dsm_sm2sm_thrpt(d_Data, arraySize);
  //   }

  //   cudaDeviceSynchronize();
  //   sdkStopTimer(&hTimer);
  //   double dAvgSecs =
  //       1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
  //   printf("dsm_sm2sm_thrpt() time (average) : %.5f sec\n", dAvgSecs);
  // }

  sdkDeleteTimer(&hTimer);
  checkCudaErrors(cudaFree(d_Data));

}
