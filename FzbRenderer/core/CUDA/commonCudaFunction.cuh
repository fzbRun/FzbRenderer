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
clock()�����CPUʱ�䣬���ǹ���ʱ�䡣������3���̲߳������У�ִ��3�룬��ôclock�᷵��3x3=9�룬���ҹ���ʱ�䲻�����ڣ�����޷��õ��˺�������ʱ�䡣
*/
double cpuSecond();

void checkKernelFunction();

/*
�����ǹ���warp�ڲ�������ԭ���ѧϰ
���ȣ���ȷ������Ҫ�ĵ�
1. __activemask()��ִָ�е��˿̵��̵߳����롣�������ںܶ�ʱ��warp�ǰ�����ִ�еģ�Ӳ������һ����16��Ӳ��һ�飬Ҫ����ʱ������ִ����һ��warp�������ܵ���
    ��ʹwarp���߳���һ����֧�У����ǲ���ͬʱִ�е�__activemask()�ģ��Ϳ��ܳ���ǰ���warpֻ�ܵõ�0-15Ϊ1��16-31Ϊ0��������warp�õ�ȫ��1����������Բ���
    ֱ������ԭ��Ĳ�����һ����˵��Ҫ�ڷ�֧ǰʹ��__ballot_sync()���õ����롣
2. �󲿷�ԭ����Ҫ������ʾִ���̣߳�����volta�ܹ�����Բ���Ҫ�ܣ�ֱ����0xffffffff���ɣ�ԭ���ڲ��ᴦ�����ڲ�������̣߳�ԭ��������δ���壨������˵����0��
    �������ڷ�֧�У������Լ����߳̽����ۼӣ�����ֱ����0xffffffff�����Ƕ��ڴ�Сֵ�ж����У���Ϊ�������̻߳᷵��0
3. ͬ����volta�ܹ�֮��__syncwarp()���Դ���������֧��ֻ������ԭ��֮���̻߳��ٴη�ɢ��
4. ���ڱ��뻷����Ӳ���Ĳ�ͬ����ʽͬ���ǲ���ȫ�ģ�������if-else���֧ͬ�������������ڻ�����ͬ��ͬ��û�з�������ʹ��ǰ��ʹ��__syncwarp()����
     __syncwarp(); v = __shfl(0); __syncwarp() != __shfl(0)
    ���������Ҫʹ����ʽͬ������ ʹ�ô�sync��ԭ���__shfl_sync��
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