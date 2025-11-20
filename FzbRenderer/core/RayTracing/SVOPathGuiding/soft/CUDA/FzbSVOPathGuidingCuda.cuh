#pragma once
#include "../../../../SceneDivision/SVO_PG/CUDA/FzbSVOCuda_PG.cuh"
#include <unordered_map>

#ifndef FZB_SVO_PATH_GUIDING_CUH
#define FZB_SVO_PATH_GUIDING_CUH

struct FzbSVOPathGuidingSetting_soft {
	FzbSVOSetting_PG SVO_PGSetting;
	uint32_t spp = 1;
	bool Acc = false;
	uint32_t maxDepth = 6;
	bool useNEE;
	bool useSphericalRectangleSample;
};
struct FzbSVOPathGuidingCudaSetting {
	uint32_t spp;
	bool Acc;
	uint32_t maxDepth;
	bool useNEE;
	bool useSphericalRectangleSample;
	uint32_t voxelCount;
	glm::vec3 voxelSize;
	glm::vec3 voxelGroupStartPos;
	uint32_t maxSVOLayer;
	//uint32_t SVOIndivisibleNodeTotalCount;
	uint32_t SVONodeTotalCount_E;
	FzbSVOLayerInfo* SVOLayerInfos_E;
	FzbSVONodeData_PG_G** SVONodes_G;
	FzbSVONodeData_PG_E** SVONodes_E;
	float* SVONodeWeights;
};

struct FzbSVOPathGuidingCuda {
	FzbSVOPathGuidingCudaSetting setting;
	std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager;
	std::unordered_map<HANDLE, cudaExternalSemaphore_t> extStartSemphores;

	std::shared_ptr<FzbSVOCuda_PG> SVOCuda_PG;

	FzbSVOPathGuidingCuda();
	FzbSVOPathGuidingCuda(std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager, FzbSVOPathGuidingCudaSetting setting, std::shared_ptr<FzbSVOCuda_PG> SVOCuda_PG);
	void updateSVOData(FzbSVOPathGuidingCudaSetting setting);
	void SVOPathGuiding(HANDLE startSemaphoreHandle);
	void clean();
};

#endif