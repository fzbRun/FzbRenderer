#pragma once

#include "../../CUDA/commonCudaFunction.cuh"
#include "../../common/FzbCommon.h"
#include "../../SceneDivision/BVH/CUDA/createBVH.cuh"
#include "./FzbGetTriangleAttribute.cuh"
#include "./FzbRayGenerate.cuh"

#ifndef FZB_GET_ILLUMINATION_CUH
#define FZB_GET_ILLUMINATION_CUH

struct FzbRayTracingPointLight {
	glm::vec3 worldPos;
	glm::vec3 radiantIntensity;		//���Դû��radiance
};
struct FzbRayTracingAreaLight {
	glm::vec3 worldPos;
	glm::vec3 normal;
	glm::vec3 radiance;		//Ĭ���ʲ��壬���������radiance��ͬ
	glm::vec3 edge0;	//��
	glm::vec3 edge1;	//��
	float area;		//���
};
struct FzbRayTracingLightSet {
	uint32_t pointLightCount;
	FzbRayTracingPointLight* pointLightInfoArray;
	uint32_t areaLightCount;
	FzbRayTracingAreaLight* areaLightInfoArray;
};
//-------------------------------------------------------����-----------------------------------------
__device__ const float PI = 3.1415926535f;
__device__ const float PI_countdown = 0.31830988618;
extern __constant__ uint32_t systemPointLightCount;
const uint32_t maxPointLightCount = 16;
extern __constant__ FzbRayTracingPointLight systemPointLightInfoArray[maxPointLightCount];
extern __constant__ uint32_t systemAreaLightCount;
const uint32_t maxAreaLightCount = 8;
extern __constant__ FzbRayTracingAreaLight systemAreaLightInfoArray[maxAreaLightCount];

//-------------------------------------------------------����-----------------------------------------
__device__ float DistributionGGX(const glm::vec3& N, const glm::vec3& H, float roughness);
__device__ glm::vec3 getBSDF(const FzbTriangleAttribute& triangleAttribute, const glm::vec3& incidence, const glm::vec3& outgoing);

__device__ glm::vec3 getRadiance(FzbTriangleAttribute& triangleAttribute, FzbRay& ray, const FzbRayTracingLightSet* lightSet,
	const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, uint32_t& randomNumberSeed);


#endif