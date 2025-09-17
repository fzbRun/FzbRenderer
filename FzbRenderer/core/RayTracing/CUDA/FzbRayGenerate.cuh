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

__constant__ FzbPathTracingCameraInfo cameraInfo;

__device__ FzbRay generateFirstRay(glm::vec2 screenTexel, uint32_t spp, uint32_t sppIndex) {
	float2 randomNumber = Hammersley(sppIndex, spp);
	glm::vec2 screenPos = (screenTexel + glm::vec2(randomNumber.x, randomNumber.y)) / glm::vec2(cameraInfo.screenWidth, cameraInfo.screenHeight);
	glm::vec4 ndcPos = glm::vec4(screenPos * 2.0f - 1.0f, 0.0f, 1.0f);	//vulkan中近平面ndcDepth在[0,1]
	glm::vec4 sppWorldPos = cameraInfo.inversePVMatrix * ndcPos;
	sppWorldPos /= sppWorldPos.w;
	FzbRay ray;
	ray.startPos = cameraInfo.cameraWorldPos;
	ray.direction = glm::normalize(glm::vec3(sppWorldPos) - ray.startPos);
	ray.depth = FLT_MAX;
	return ray;
}

#endif