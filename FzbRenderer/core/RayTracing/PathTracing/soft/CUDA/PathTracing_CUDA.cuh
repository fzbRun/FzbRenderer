#pragma once

#include "../../../../CUDA/vulkanCudaInterop.cuh"
#include "../../../../CUDA/commonCudaFunction.cuh"
#include "../../../../common/FzbScene/FzbScene.h"

#ifndef PATH_TRACING_CUDA_H
#define PATH_TRACING_CUDA_H

struct FzbPathTracingCuda {

public:
	cudaExternalMemory_t resultMapExtMem;
	cudaMipmappedArray_t resultMapMipmap;
	cudaSurfaceObject_t resultMapObject = 0;

	cudaExternalSemaphore_t extPathTracingFinishedSemaphore;
	uint32_t startSemaphoreNum;
	std::vector<cudaExternalSemaphore_t> extStartSemaphores;

	cudaExternalMemory_t vertexExtMem;
	float* vertices;

	std::vector<cudaExternalMemory_t> textureExtMens;
	std::vector<cudaMipmappedArray_t> textureMipmap;
	std::vector<cudaTextureObject_t> textureObject;

	cudaStream_t stream = nullptr;

	FzbPathTracingCuda();
	FzbPathTracingCuda(VkPhysicalDevice vkPhysicalDevice, FzbScene* scene, FzbImage pathTracingResultMap, HANDLE pathTracingFinishedSemphoreHandle, std::vector<HANDLE> startSemaphoreHandles);
	void pathTracing();
	void clean();
};

#endif