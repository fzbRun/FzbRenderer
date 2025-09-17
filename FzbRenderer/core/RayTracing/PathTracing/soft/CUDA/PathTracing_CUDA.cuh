#pragma once

#include "../../../../CUDA/vulkanCudaInterop.cuh"
#include "../../../../CUDA/commonCudaFunction.cuh"
#include "../../../../common/FzbScene/FzbScene.h"
#include "../FzbPathTracingMaterial.h"
#include "../../../../SceneDivision/BVH/CUDA/createBVH.cuh"

#ifndef PATH_TRACING_CUDA_H
#define PATH_TRACING_CUDA_H

struct FzbPathTracingSetting {
	bool spp = 1;
};

struct FzbPathTracingCuda {

public:
	FzbPathTracingSetting setting;

	//当spp过大时，应该使用buffer，方便存入数据
	cudaExternalMemory_t resultBufferExtMem;
	float4* resultBuffer;
	//cudaExternalMemory_t resultMapExtMem;
	//cudaMipmappedArray_t resultMapMipmap;
	//cudaSurfaceObject_t resultMapObject = 0;

	cudaExternalSemaphore_t extPathTracingFinishedSemaphore;
	//std::vector<cudaExternalSemaphore_t> extStartSemaphores;

	cudaExternalMemory_t vertexExtMem;
	float* vertices;

	std::vector<cudaExternalMemory_t> textureExtMems;
	std::vector<cudaMipmappedArray_t> textureMipmap;
	std::vector<cudaTextureObject_t> textureObjects;
	cudaTextureObject_t* materialTextures;

	cudaStream_t stream = nullptr;

	BVHCuda* bvh;

	FzbPathTracingCuda();
	FzbPathTracingCuda(VkPhysicalDevice vkPhysicalDevice, FzbMainScene* scene,
		FzbPathTracingSetting setting, FzbBuffer pathTracingResultBuffer, FzbSemaphore pathTracingFinishedSemphore, 
		std::vector<FzbImage>& sceneTextures, std::vector<FzbPathTracingMaterialUniformObject> sceneMaterialInfoArray,
		BVHCuda* bvh);
	void pathTracing(VkSemaphore startSemaphore, uint32_t screenWidth, uint32_t screenHeight);
	void clean();
};

#endif