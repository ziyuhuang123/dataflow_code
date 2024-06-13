#include <stdio.h>
#include <cuda.h>

#define SHARED_MEM_SIZE 98304  // 96KB-->这样使得每个SM只能跑一个block，尽可能贴近我的场景。

__managed__ unsigned long long start_times[256]; // 假设最大block数为256
__managed__ unsigned long long global_data[256]; // 全局内存数组

__device__ unsigned long long globaltime(void)
{
    unsigned long long time;
    asm("mov.u64  %0, %%globaltimer;" : "=l"(time));
    return time;
}

__global__ void logkernel(unsigned long long *global_data)
{
    unsigned int tid = blockIdx.x;
    start_times[tid] = globaltime();

    // 执行全局内存读写操作
    for (int i = 0; i < 1000; i++) {
        atomicAdd(&global_data[tid], 1);
    }
}

int main(void)
{
    // 获取设备属性
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max shared memory per block before opt-in: %d bytes\n", prop.sharedMemPerBlock);

    // 设置核函数属性以使用更大的共享内存
    cudaFuncSetAttribute(logkernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_MEM_SIZE);

    // 再次获取设备属性以验证设置是否成功
    cudaGetDeviceProperties(&prop, 0);
    printf("Max shared memory per block after opt-in: %d bytes\n", prop.sharedMemPerBlock);

    // 初始化全局数据
    for (int i = 0; i < 256; i++) {
        global_data[i] = 0;
    }

    // 启动核函数
    logkernel<<<256, 1, SHARED_MEM_SIZE>>>(global_data);// -->每个block只有一个线程工作。
    cudaDeviceSynchronize();

    // 将结果写入CSV文件
    FILE *output = fopen("block_launch_times.csv", "w");
    if (output == NULL) {
        fprintf(stderr, "ERROR: Failed to open output file.\n");
        return 1;
    }
    fprintf(output, "blockIdx.x,start_time\n");
    for (int i = 0; i < 256; i++) {
        fprintf(output, "%d,%llu\n", i, start_times[i]);
    }
    fclose(output);

    return 0;
}
