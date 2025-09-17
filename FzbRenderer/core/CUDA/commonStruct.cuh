#pragma once

#include "./commonCudaFunction.cuh"

#ifndef FZB_COMMON_STRUCT_H
#define FZB_COMMON_STRUCT_H

struct FzbAABB {
	float leftX = FLT_MAX;
	float rightX = -FLT_MAX;
	float leftY = FLT_MAX;
	float rightY = -FLT_MAX;
	float leftZ = FLT_MAX;
	float rightZ = -FLT_MAX;

	__host__ __device__  FzbAABB() {};
	__host__ __device__ FzbAABB(float leftX, float rightX, float leftY, float rightY, float leftZ, float rightZ) {
		this->leftX = leftX;
		this->rightX = rightX;
		this->leftY = leftY;
		this->rightY = rightY;
		this->leftZ = leftZ;
		this->rightZ = rightZ;
	}
};

#endif