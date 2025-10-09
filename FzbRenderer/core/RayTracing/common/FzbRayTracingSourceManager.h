#pragma once

#include "../../common/FzbCommon.h"
#include "../../SceneDivision/BVH/FzbBVH.h"
#include "./FzbRayTracingMaterial.h"
#include "../CUDA/FzbRayTracingInitSource.cuh"

#ifndef FZB_RAY_TRACING_COMMON_FUNCTION_H
#define FZB_RAY_TRACING_COMMON_FUNCTION_H

struct FzbRayTracingSourceManager {
public:
	std::shared_ptr<FzbBVH> bvh;

	//���ǲ���mainSceneȥ����materialSource�����������Լ����죬�Լ�ά��
	std::vector<FzbImage> sceneTextures;
	std::vector<FzbRayTracingMaterialUniformObject> sceneMaterialInfoArray;

	FzbBuffer rayTracingResultBuffer;
	FzbSemaphore rayTracingFinishedSemphore;

	std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManagerCuda;

	FzbRayTracingSourceManager();
	void createSource();
	void clean();
};

#endif