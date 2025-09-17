#pragma once

#include "./FzbRayGenerate.cuh"
#include "../../SceneDivision/BVH/CUDA/createBVH.cuh"
#include "FzbGetTriangleAttribute.cuh"

#ifndef FZB_COLLISION_DETECTION_CUH
#define FZB_COLLISION_DETECTION_CUH

__device__ bool AABBCollisionDetection(FzbAABB AABB, FzbRay ray);
__device__ bool meshCollisionDetection(FzbBvhNodeTriangleInfo& triangle, float* __restrict__ vertices, cudaTextureObject_t* __restrict__ materialTextures,
	FzbRay& ray, FzbTriangleAttribute& triangleAttribute);

const int BVH_MAX_DEPTH = 32;
__device__ bool sceneCollisionDetection(FzbBvhNode* __restrict__ bvhNodeArray, FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray,
	float* __restrict__ vertices, cudaTextureObject_t* __restrict__ materialTextures,
	FzbRay& ray, FzbTriangleAttribute& triangleAttribute, bool notOnlyDetection = true);

#endif