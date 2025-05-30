#pragma once

#include<iostream>
#include <chrono>
#include "../common/FzbImage.h"

//#ifndef __CUDACC__
//#define __CUDACC__
//#endif
//#ifndef __cplusplus
//#define __cplusplus
//#endif

#include <texture_indirect_functions.h>

#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <vector_types.h>

#ifndef VULKAN_CUDA_INTEROP_CUH
#define VULKAN_CUDA_INTEROP_CUH

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

__device__ uint32_t packUint3(uint3 valueU3);
__device__ uint3 unpackUint(uint32_t value);

int getCudaDeviceForVulkanPhysicalDevice(VkPhysicalDevice vkPhysicalDevice);
cudaExternalMemory_t importVulkanMemoryObjectFromFileDescriptor(int fd, unsigned long long size, bool isDedicated);
cudaExternalMemory_t importVulkanMemoryObjectFromNTHandle(HANDLE handle, unsigned long long size, bool isDedicated);
cudaExternalMemory_t importVulkanMemoryObjectFromNamedNTHandle(LPCWSTR name, unsigned long long size, bool isDedicated);
cudaExternalMemory_t importVulkanMemoryObjectFromKMTHandle(HANDLE handle, unsigned long long size, bool isDedicated);
void* mapBufferOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, unsigned long long size);
cudaMipmappedArray_t mapMipmappedArrayOntoExternalMemory(cudaExternalMemory_t extMem, unsigned long long offset, cudaChannelFormatDesc* formatDesc, cudaExtent* extent, unsigned int flags, unsigned int numLevels);
cudaChannelFormatDesc getCudaChannelFormatDescForVulkanFormat(VkFormat format);
cudaExtent getCudaExtentForVulkanExtent(VkExtent3D vkExt, uint32_t arrayLayers, VkImageViewType vkImageViewType);
unsigned int getCudaMipmappedArrayFlagsForVulkanImage(VkImageViewType vkImageViewType, VkImageUsageFlags vkImageUsageFlags, bool allowSurfaceLoadStore);
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromFileDescriptor(int fd);
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNTHandle(HANDLE handle);
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNamedNTHandle(LPCWSTR name);
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromKMTHandle(HANDLE handle);
void signalExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream);
void waitExternalSemaphore(cudaExternalSemaphore_t extSem, cudaStream_t stream);
void fromVulkanImageToCudaTexture(VkPhysicalDevice vkPhysicalDevice, FzbImage& vkImage, HANDLE handle, unsigned long long size,
    bool isDedicated, cudaExternalMemory_t& extMem, cudaMipmappedArray_t& mipmap, cudaTextureObject_t& texObj);
void fromVulkanImageToCudaSurface(VkPhysicalDevice vkPhysicalDevice, FzbImage& vkImage, HANDLE handle, unsigned long long size,
    bool isDedicated, cudaExternalMemory_t& extMem, cudaMipmappedArray_t& mipmap, cudaSurfaceObject_t& surfObj);
#endif