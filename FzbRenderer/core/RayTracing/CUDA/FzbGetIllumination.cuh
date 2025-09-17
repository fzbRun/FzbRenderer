#pragma once

#include "../../CUDA/commonCudaFunction.cuh"
#include "../../common/FzbCommon.h"
#include "FzbGetTriangleAttribute.cuh"

#ifndef FZB_GET_ILLUMINATION_CUH
#define FZB_GET_ILLUMINATION_CUH

struct FzbRayTracingPointLight {
	glm::vec4 worldPos;
	glm::vec4 radiantIntensity;		//���Դû��radiance
};
struct FzbRayTracingAreaLight {
	glm::vec4 worldPos;
	glm::vec4 normal;
	glm::vec4 radiance;		//Ĭ���ʲ��壬���������radiance��ͬ
	glm::vec4 edge0;	//��
	glm::vec4 edge1;	//��
	float area;		//���
};
//-------------------------------------------------------����-----------------------------------------
__device__ const float PI = 3.1415926535f;
__constant__ uint32_t frameIndex;
extern __constant__ FzbPathTracingMaterialUniformObject materialInfoArray[128];
__constant__ uint32_t pointLightCount;
__constant__ FzbRayTracingPointLight pointLightInfoArray[16];
__constant__ uint32_t areaLightCount;
__constant__ FzbRayTracingAreaLight areaLightInfoArray[8];

__device__ glm::vec3 getBSDF(FzbTriangleAttribute triangleAttribute, glm::vec3 incidence, glm::vec3 outgoing) {
	//����Ĭ�϶���diffuse�ģ�����Ȩ��
	return glm::vec3(1.0f / PI);
}

/*
���شӹ�Դ����һ��ײ�����radiance
����Ӧ�����Ż������������й�Դ���в�������������ķ�������Ŀǰ�ȷ���
*/
__device__ glm::vec3 NEE(FzbTriangleAttribute triangleAttribute, FzbRay ray, 
	float* __restrict__ vertices, cudaTextureObject_t* __restrict__ materialTextures,
	FzbBvhNode* __restrict__ bvhNodeArray, FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray) {
	glm::vec3 radiance = glm::vec3(0.0f);
	FzbRay tempRay;
	for (int i = 0; i < pointLightCount; ++i) {
		FzbRayTracingPointLight light = pointLightInfoArray[i];
		glm::vec3 lightPos = glm::vec3(light.worldPos);
		glm::vec3 direction = lightPos - ray.hitPos;
		tempRay.depth = FLT_MAX;
		tempRay.direction = glm::normalize(direction);
		FzbTriangleAttribute hitTriangleAttribute;
		bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, tempRay, hitTriangleAttribute, false);
		if (hit) {	//������У����Ҿ�����Դ��Զ����˵���м����ڵ���
			if (glm::length((tempRay.depth * tempRay.direction + tempRay.startPos) - lightPos) > 0.01f) continue;
		}
		float r2 = glm::length(direction); r2 *= r2;
		float cosTheta = glm::clamp(glm::dot(triangleAttribute.normal, tempRay.direction), 0.0f, 1.0f);
		radiance += cosTheta * glm::vec3(light.radiantIntensity) / r2 * getBSDF(triangleAttribute, tempRay.direction, -ray.direction);
	}
	for (int i = 0; i < areaLightCount; ++i) {
		FzbRayTracingAreaLight light = areaLightInfoArray[i];
		uint32_t randomNumberSeed = frameIndex + uint32_t(glm::length(ray.hitPos));
		float randomNumberX = rand(randomNumberSeed);
		float randomNumberY = rand(randomNumberSeed);
		glm::vec3 lightPos = glm::vec3(light.worldPos + randomNumberX * light.edge0 + randomNumberY * light.edge1);
		glm::vec3 direction = lightPos - ray.hitPos;
		tempRay.depth = FLT_MAX;
		tempRay.direction = glm::normalize(direction);
		FzbTriangleAttribute hitTriangleAttribute;
		bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, tempRay, hitTriangleAttribute, false);
		if (!hit) continue;
		else if (glm::length((tempRay.depth * tempRay.direction + tempRay.startPos) - lightPos) > 0.01f) continue;
		glm::vec3 lightRadiance_cosTheta = light.radiance * glm::clamp(glm::dot(triangleAttribute.normal, tempRay.direction), 0.0f, 1.0f);
		lightRadiance_cosTheta *= getBSDF(triangleAttribute, tempRay.direction, -ray.direction) * light.area;	//bsdf / pdf
		lightRadiance_cosTheta *= glm::clamp(glm::dot(glm::vec3(light.normal), tempRay.direction), 0.0f, 1.0f);	//΢�ֵ�λ��dw��ΪdA
		float r2 = glm::length(direction);	r2 *= r2;
		lightRadiance_cosTheta /= r2;
		radiance += lightRadiance_cosTheta;
	}
	return radiance;
}

#endif