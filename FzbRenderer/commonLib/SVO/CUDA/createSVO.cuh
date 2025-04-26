#pragma once

#include "../../CUDA/vulkanCudaInterop.cuh"

#ifndef CREATE_SVO_CUH
#define CREATE_SVO_CUH

struct FzbSVOCudaVariable;
void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, MyImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle, FzbSVOCudaVariable*& fzbSVOCudaVar);
void cleanSVOCuda(FzbSVOCudaVariable* fzbSVOCudaVar);

#endif