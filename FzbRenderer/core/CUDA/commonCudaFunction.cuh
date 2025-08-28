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

#endif