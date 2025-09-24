#pragma once

#include "./FzbRayGenerate.cuh"
#include "../../SceneDivision/BVH/CUDA/createBVH.cuh"
#include "FzbGetTriangleAttribute.cuh"

#ifndef FZB_COLLISION_DETECTION_CUH
#define FZB_COLLISION_DETECTION_CUH

//__global__ const uint32_t sharedBvhNodeLevel = 4;		//bvhNodeTree的前几层
//__device__ __constant__ FzbBvhNode sharedBvhNodes[sharedBvhNodeLevel * 2 - 1];	//认为是满二叉树
//void createSharedBvhNodes(FzbBvhNode* bvhNodeArray, uint32_t bvhNodeCount);

__device__ bool AABBCollisionDetection(FzbAABB AABB, FzbRay ray);
__device__ bool meshCollisionDetection(const float* __restrict__ vertices, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, const cudaTextureObject_t* __restrict__ materialTextures,
	FzbRay& ray, FzbBvhNode node, FzbBvhNodeTriangleInfo& hitTriangle, FzbTriangleAttribute& triangleAttribute);

const int BVH_MAX_DEPTH = 16;
__device__ bool sceneCollisionDetection(const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray,
	const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	FzbRay& ray, FzbTriangleAttribute& triangleAttribute, bool notOnlyDetection = true);

#endif