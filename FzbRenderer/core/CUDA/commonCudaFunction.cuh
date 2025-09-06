#pragma once

#include <chrono>

#include <texture_indirect_functions.h>

#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <helper_math.h>

#ifndef COMMON_CUDA_FUNCTION_CUH
#define COMMON_CUDA_FUNCTION_CUH

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

/*
clock()会计算CPU时间，而非挂钟时间。即若有3个线程并行运行，执行3秒，那么clock会返回3x3=9秒，并且挂起时间不算在内，因此无法得到核函数运行时间。
*/
double cpuSecond();

void checkKernelFunction();

/*
这里是关于warp内部操作（原语）的学习
首先，明确几点重要的点
1. __activemask()是指执行到此刻的线程的掩码。但是由于很多时候warp是半个半个执行的（硬件区域一般是16个硬件一组，要两个时钟周期执行完一个warp），可能导致
    即使warp内线程在一个分支中，但是不是同时执行到__activemask()的，就可能出现前半个warp只能得到0-15为1，16-31为0，而后半个warp得到全是1的情况。所以不能
    直接用于原语的参数。一般来说需要在分支前使用__ballot_sync()来得到掩码。
2. 大部分原语需要掩码显示执行线程，但是volta架构后可以不需要管，直接用0xffffffff即可，原语内部会处理，对于不激活的线程，原语结果会是未定义（但资料说就是0）
    这样，在分支中，如果想对激活线程进行累加，可以直接用0xffffffff。但是对于大小值判断则不行，因为不激活线程会返回0
3. 同样在volta架构之后，__syncwarp()可以处理条件分支，只不过在原语之后线程会再次发散。
4. 由于编译环境和硬件的不同，隐式同步是不安全的，比如在if-else后分支同步，但可能由于环境不同，同步没有发生，即使在前后使用__syncwarp()，如
     __syncwarp(); v = __shfl(0); __syncwarp() != __shfl(0)
    因此我们需要使用显式同步，即 使用带sync的原语，如__shfl_sync。
*/
__device__ int warpReduce(int localSum);
__device__ float warpReduce(float localSum);
__device__ int warpMax(int value);
__device__ int warpMin(int value);

__device__ float warpMax(float value);
__device__ float warpMin(float value);

float* sumFloat(float* input, float* output);

__device__ uint32_t packUint3(uint3 valueU3);
__device__ uint3 unpackUint(uint32_t value);

__device__ float atomicAddFloat(float* addr, float val);
__device__ float atomicMinFloat(float* addr, float val);
__device__ float atomicMaxFloat(float* addr, float val);

__device__ uint32_t pcg(uint32_t& state);
__device__ uint2 pcg2d(uint2 v);
__device__ float rand(uint32_t& seed);
__device__ float RadicalInverse_VdC(uint bits);
__device__ float2 Hammersley(uint i, uint N);

__global__ void addDate_float_device(float* data, float date, uint32_t dataNum);
void addDateCUDA_float(float* data, float date, uint32_t dataNum);

__global__ void addDate_uint_device(uint32_t* data, uint32_t date, uint32_t dataNum);
void addDateCUDA_uint(uint32_t* data, uint32_t date, uint32_t dataNum);


#endif