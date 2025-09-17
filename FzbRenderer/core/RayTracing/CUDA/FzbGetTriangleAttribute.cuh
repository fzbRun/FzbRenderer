#pragma once

#include "../../CUDA/commonCudaFunction.cuh"
#include "../PathTracing/soft/FzbPathTracingMaterial.h"
#include "../../SceneDivision/BVH/CUDA/createBVH.cuh"


#ifndef FZB_GET_TRIANGLE_ATTRIBUTE_CUH
#define FZB_GET_TRIANGLE_ATTRIBUTE_CUH

struct FzbTriangleAttribute {
	glm::vec3 pos0;
	glm::vec3 pos1;
	glm::vec3 pos2;
	glm::vec3 normal;
	glm::vec2 texCoords;
	glm::vec3 albedo;
	uint32_t materialType;
};

__device__ void getTriangleVertexAttribute(float* __restrict__ vertices, cudaTextureObject_t* __restrict__ materialTextures, FzbBvhNodeTriangleInfo triangle, FzbTriangleAttribute& triangleAttribute);
__device__ void getTriangleMaterialAttribute(cudaTextureObject_t* __restrict__ materialTextures, uint32_t materialIndex,FzbTriangleAttribute& triangleAttribute);

#endif