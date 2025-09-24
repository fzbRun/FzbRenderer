#pragma once

#include "../../CUDA/commonCudaFunction.cuh"
#include "../../common/FzbCommon.h"

#ifndef FZB_RAY_GENERATE_H
#define FZB_RAY_GENERATE_H

struct FzbPathTracingCameraInfo {
	glm::vec3 cameraWorldPos;
	glm::mat4 inversePVMatrix;	//inverse(projection * view)
	uint32_t screenWidth;
	uint32_t screenHeight;
};

struct FzbRay {
	glm::vec3 startPos;
	glm::vec3 direction;
	float depth;
	glm::vec3 hitPos;
};

#ifdef __cplusplus
extern "C" {
#endif

extern __constant__ FzbPathTracingCameraInfo systemCameraInfo;
__device__ FzbRay generateFirstRay(FzbPathTracingCameraInfo* cameraInfo, glm::vec2 screenTexel, uint32_t spp, uint32_t sppIndex);
__device__ void generateRay(uint32_t distributionType, float& pdf, FzbRay& ray, uint32_t& randomNumberSeed);

#ifdef __cplusplus
}
#endif

#endif