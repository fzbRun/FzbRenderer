#pragma once

#include "../../../../CUDA/vulkanCudaInterop.cuh"
#include "../../../../CUDA/commonCudaFunction.cuh"
#include "../../../../common/FzbScene/FzbScene.h"
#include "../FzbPathTracingMaterial.h"
#include "../../../../SceneDivision/BVH/CUDA/createBVH.cuh"
#include "../../../CUDA/FzbGetIllumination.cuh"

#include <unordered_map>

#ifndef PATH_TRACING_CUDA_H
#define PATH_TRACING_CUDA_H

struct FzbPathTracingSetting {
	uint32_t spp = 1;
	bool useCudaRandom = false;
};

struct FzbPathTracingCudaSourceSet {
	FzbPathTracingSetting setting;
	FzbBuffer pathTracingResultBuffer;
	FzbSemaphore pathTracingFinishedSemphore;
	FzbBuffer sceneVertices;
	std::vector<FzbImage> sceneTextures;
	std::vector<FzbPathTracingMaterialUniformObject> sceneMaterialInfoArray;
	HANDLE bvhSemaphoreHandle;
	uint32_t bvhNodeCount;
	FzbBvhNode* bvhNodeArray = nullptr;
	FzbBvhNodeTriangleInfo* bvhTriangleInfoArray;
	uint32_t pointLightCount = 0;
	std::vector<FzbRayTracingPointLight> pointLightInfoArray;	//16
	uint32_t areaLightCount = 0;
	std::vector<FzbRayTracingAreaLight> areaLightInfoArray;		//8

	FzbPathTracingCudaSourceSet() {};
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

	std::unordered_map<HANDLE, cudaExternalSemaphore_t> extStartSemphores;
	cudaExternalSemaphore_t extPathTracingFinishedSemaphore;
	//std::vector<cudaExternalSemaphore_t> extStartSemaphores;

	cudaExternalMemory_t vertexExtMem;
	float* vertices;

	std::vector<cudaExternalMemory_t> textureExtMems;
	std::vector<cudaMipmappedArray_t> textureMipmap;
	std::vector<cudaTextureObject_t> textureObjects;
	cudaTextureObject_t* materialTextures;

	cudaStream_t stream = nullptr;

	FzbBvhNode* bvhNodeArray = nullptr;
	FzbBvhNodeTriangleInfo* bvhTriangleInfoArray = nullptr;

	FzbPathTracingCuda();
	FzbPathTracingCuda(FzbPathTracingCudaSourceSet& sourceSet);
	void pathTracing(HANDLE startSemaphoreHandle);
	void clean();

private:
	curandState* systemRandomNumberStates_device = nullptr;
	uint32_t runCount = 0;
	double meanRunTime = 0.0;
};

#endif