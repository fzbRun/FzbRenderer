#include "./FzbRayGenerate.cuh"
#include "./FzbGetIllumination.cuh"
#include "FzbGetIllumination.cuh"

__device__ FzbRay generateFirstRay(FzbPathTracingCameraInfo* cameraInfo, glm::vec2 screenTexel, uint32_t spp, uint32_t sppIndex) {
	glm::vec2 randomNumber = Hammersley(sppIndex, spp);
	glm::vec2 screenPos = (screenTexel + glm::vec2(randomNumber.x, randomNumber.y)) / glm::vec2(cameraInfo->screenWidth, cameraInfo->screenHeight);
	glm::vec4 ndcPos = glm::vec4(screenPos * 2.0f - 1.0f, 0.0f, 1.0f);	//vulkan中近平面ndcDepth在[0,1]
	glm::vec4 sppWorldPos = cameraInfo->inversePVMatrix * ndcPos;
	sppWorldPos /= sppWorldPos.w;
	FzbRay ray;
	ray.startPos = cameraInfo->cameraWorldPos;
	ray.direction = glm::normalize(glm::vec3(sppWorldPos) - ray.startPos);
	ray.depth = FLT_MAX;

	//float2 randomNumber = Hammersley(sppIndex, spp);
	//screenTexel/*screenPos*/ = (screenTexel + glm::vec2(randomNumber.x, randomNumber.y)) / glm::vec2(cameraInfo->screenWidth, cameraInfo->screenHeight);
	//glm::vec4 texelPos/*ndcPos*/ = glm::vec4(screenTexel * 2.0f - 1.0f, 0.0f, 1.0f);	//vulkan中近平面ndcDepth在[0,1]
	//texelPos = cameraInfo->inversePVMatrix * texelPos;
	//texelPos /= texelPos.w;

	return ray;
}
__device__ void generateRay(const FzbTriangleAttribute& triangleAttribute, float& pdf, FzbRay& ray, uint32_t& randomNumberSeed) {
	float randomNumber1 = rand(randomNumberSeed);
	float randomNumber2 = rand(randomNumberSeed);
	float phi = randomNumber2 * 2 * PI;
	if (triangleAttribute.materialType == 0) {	//余弦重要性采样
		/*
		* theta = glm::asin(glm::sqrt(randomNumber1))，那么sinTheta = glm::sqrt(randomNumber1)
		* 我们设f(x)=1，均匀分布；y = g(x) = sqrt(x),  那么x = g^-1(y) = y^2，他们是等面积映射，所以要乘以一个雅可比行列式
		* f(y) = f(g^-1(y)) (dg^-1(y)/dy) = f(y^2) * 2y
		* 所以我们可以直接取sinTheta^2 = randomNumber1，其pdf = 1 * 2 * randomNumber1
		*/
		float sinTheta = glm::sqrt(randomNumber1);
		float cosTheta = glm::sqrt(1 - sinTheta * sinTheta);
		float x = sinTheta * glm::cos(phi);
		float y = sinTheta * glm::sin(phi);
		float z = cosTheta;
		ray.direction = glm::vec3(x, y, z);

		pdf *= sinTheta * cosTheta / PI;
	}
	else if (triangleAttribute.materialType == 1) {
		float cosTheta = glm::sqrt((1.0f - randomNumber1) / ((triangleAttribute.roughness * triangleAttribute.roughness - 1.0f) * randomNumber1 + 1.0f));
		float sinTheta = glm::sqrt(1.0f - cosTheta * cosTheta);
		float x = sinTheta * glm::cos(phi);
		float y = sinTheta * glm::sin(phi);
		float z = cosTheta;
		glm::vec3 h = glm::vec3(x, y, z);

		glm::vec3 bitangent = glm::cross(triangleAttribute.normal, triangleAttribute.tangent) * triangleAttribute.handed;
		glm::mat3 TBN = glm::mat3(triangleAttribute.tangent, bitangent, triangleAttribute.normal);

		//这里得到的只是h，还需要从h转为i
		ray.direction = glm::normalize(TBN * (2.0f * glm::max(glm::dot(-ray.direction, h) * h + ray.direction, 0.0f)));

		pdf = DistributionGGX(triangleAttribute.normal, h, triangleAttribute.roughness) * cosTheta * sinTheta / (4.0f * glm::max(glm::dot(-ray.direction, h), 0.01f));
	}
	ray.startPos = ray.direction * 0.001f + ray.hitPos;
	ray.depth = FLT_MAX;
}