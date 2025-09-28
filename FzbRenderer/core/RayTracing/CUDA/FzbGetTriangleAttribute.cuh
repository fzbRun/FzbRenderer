#pragma once

#include "../../CUDA/commonCudaFunction.cuh"
#include "../PathTracing/soft/FzbPathTracingMaterial.h"
#include "../../SceneDivision/BVH/CUDA/createBVH.cuh"


#ifndef FZB_GET_TRIANGLE_ATTRIBUTE_CUH
#define FZB_GET_TRIANGLE_ATTRIBUTE_CUH

struct FzbTrianglePos {
	glm::vec3 pos0;
	glm::vec3 pos1;
	glm::vec3 pos2;
};
struct FzbTriangleAttribute {
	FzbTrianglePos pos;
	glm::vec3 normal;
	glm::vec3 tangent;
	float handed;
	glm::vec2 texCoords;
	glm::vec3 albedo;
	glm::vec3 emissive;
	uint32_t materialType;
	float roughness;
};

const uint32_t maxMaterialCount = 128;
extern __constant__ FzbPathTracingMaterialUniformObject materialInfoArray[maxMaterialCount];

__device__ void getTriangleVertexPos(const float* __restrict__ vertices, FzbBvhNodeTriangleInfo triangle, FzbTrianglePos& trianglePos);
__device__ void getTriangleMaterialAttribute(const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNodeTriangleInfo& triangle, FzbTriangleAttribute& triangleAttribute, const glm::vec3& hitPos);

#endif