#pragma once

#include "../../CUDA/vulkanCudaInterop.cuh"
#include "../../CUDA/commonCudaFunction.cuh"
#include "../../common/FzbScene/FzbScene.h"
#include "../common/FzbRayTracingMaterial.h"
#include "../../SceneDivision/BVH/CUDA/createBVH.cuh"
#include "./FzbGetIllumination.cuh"

#ifndef FZB_RAY_TRACING_INIT_SOURCE_CUH
#define FZB_RAY_TRACING_INIT_SOURCE_CUH

struct FzbRayTracingCudaSourceSet {
	FzbBuffer rayTracingResultBuffer;
	FzbSemaphore rayTracingFinishedSemphore;
	FzbBuffer sceneVertices;
	std::vector<FzbImage> sceneTextures;
	std::vector<FzbRayTracingMaterialUniformObject> sceneMaterialInfoArray;
	HANDLE bvhSemaphoreHandle;
	uint32_t bvhNodeCount;
	FzbBvhNode* bvhNodeArray = nullptr;
	FzbBvhNodeTriangleInfo* bvhTriangleInfoArray;
	uint32_t pointLightCount = 0;
	std::vector<FzbRayTracingPointLight> pointLightInfoArray;	//16
	uint32_t areaLightCount = 0;
	std::vector<FzbRayTracingAreaLight> areaLightInfoArray;		//8

	FzbRayTracingCudaSourceSet() {};
};

struct FzbRayTracingSourceManager_Cuda {
	FzbBvhNode* bvhNodeArray = nullptr;
	FzbBvhNodeTriangleInfo* bvhTriangleInfoArray = nullptr;

	cudaExternalMemory_t resultBufferExtMem;
	float4* resultBuffer;
	cudaExternalSemaphore_t extRayTracingFinishedSemaphore;

	cudaExternalMemory_t vertexExtMem;
	float* vertices;

	std::vector<cudaExternalMemory_t> textureExtMems;
	std::vector<cudaMipmappedArray_t> textureMipmap;
	std::vector<cudaTextureObject_t> textureObjects;
	cudaTextureObject_t* materialTextures;

	cudaStream_t stream = nullptr;

	FzbRayTracingSourceManager_Cuda();
	void initRayTracingSource(FzbRayTracingCudaSourceSet& sourceSet);
	void createRuntimeSource();
	void clean();
};

#endif