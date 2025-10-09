#pragma once

#include "../../../../CUDA/vulkanCudaInterop.cuh"
#include "../../../../CUDA/commonCudaFunction.cuh"
#include "../../../../common/FzbScene/FzbScene.h"
#include "../../../common/FzbRayTracingMaterial.h"
#include "../../../../SceneDivision/BVH/CUDA/createBVH.cuh"
#include "../../../CUDA/FzbGetIllumination.cuh"

#include <unordered_map>
#include "../../../CUDA/FzbRayTracingInitSource.cuh"

#ifndef PATH_TRACING_CUDA_H
#define PATH_TRACING_CUDA_H

struct FzbPathTracingSetting {
	uint32_t spp = 1;
	bool useCudaRandom = false;
};

//struct FzbPathTracingCudaSourceSet {
//	FzbPathTracingSetting setting;
//	FzbBuffer pathTracingResultBuffer;
//	FzbSemaphore pathTracingFinishedSemphore;
//	FzbBuffer sceneVertices;
//	std::vector<FzbImage> sceneTextures;
//	std::vector<FzbRayTracingMaterialUniformObject> sceneMaterialInfoArray;
//	HANDLE bvhSemaphoreHandle;
//	uint32_t bvhNodeCount;
//	FzbBvhNode* bvhNodeArray = nullptr;
//	FzbBvhNodeTriangleInfo* bvhTriangleInfoArray;
//	uint32_t pointLightCount = 0;
//	std::vector<FzbRayTracingPointLight> pointLightInfoArray;	//16
//	uint32_t areaLightCount = 0;
//	std::vector<FzbRayTracingAreaLight> areaLightInfoArray;		//8
//
//	FzbPathTracingCudaSourceSet() {};
//};

struct FzbPathTracingCuda {

public:
	FzbPathTracingSetting setting;
	std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager;
	std::unordered_map<HANDLE, cudaExternalSemaphore_t> extStartSemphores;

	FzbPathTracingCuda();
	FzbPathTracingCuda(std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager, FzbPathTracingSetting setting);
	void pathTracing(HANDLE startSemaphoreHandle);
	void clean();

private:
	curandState* systemRandomNumberStates_device = nullptr;
	uint32_t runCount = 0;
	double meanRunTime = 0.0;
};
#endif