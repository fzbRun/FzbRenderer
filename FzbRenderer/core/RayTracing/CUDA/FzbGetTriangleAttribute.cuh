#pragma once

#include "../../CUDA/commonCudaFunction.cuh"
#include "../common/FzbRayTracingMaterial.h"
#include "../../SceneDivision/BVH/CUDA/createBVH.cuh"


#ifndef FZB_GET_TRIANGLE_ATTRIBUTE_CUH
#define FZB_GET_TRIANGLE_ATTRIBUTE_CUH

struct FzbTrianglePos {
	glm::vec3 pos0;
	glm::vec3 pos1;
	glm::vec3 pos2;
};
struct FzbTriangleAttribute {
	glm::vec3 normal;
	glm::vec3 tangent;
	float handed;
	glm::vec3 albedo;
	glm::vec3 emissive;
	uint32_t materialType;
	float roughness;
	float eta;
};

const uint32_t maxMaterialCount = 128;
extern __constant__ FzbRayTracingMaterialUniformObject materialInfoArray[maxMaterialCount];

__device__ void getTriangleVertexPos(const float* __restrict__ vertices, FzbBvhNodeTriangleInfo triangle, FzbTrianglePos& trianglePos);
__device__ void getTriangleMaterialAttribute(const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNodeTriangleInfo& triangle, FzbTriangleAttribute& triangleAttribute, const FzbTrianglePos& trianglePos, const glm::vec3& hitPos);

#endif