#include "./FzbGetIllumination.cuh"
#include "FzbCollisionDetection.cuh"

__device__ glm::vec3 getBSDF(FzbTriangleAttribute triangleAttribute, glm::vec3 incidence, glm::vec3 outgoing) {
	//现在默认都是diffuse的，余弦权重
	return glm::vec3(PI_countdown) * triangleAttribute.albedo;
}

/*
返回从光源到上一个撞击点的radiance
这里应该有优化，即不对所有光源进行采样，而是随机的方，但是目前先放着
*/
__device__ glm::vec3 NEE(FzbTriangleAttribute triangleAttribute, FzbRay ray, const FzbRayTracingLightSet* lightSet,
	const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, uint32_t& randomNumberSeed) {
	glm::vec3 radiance = glm::vec3(0.0f);
	FzbRay tempRay;
	for (int i = 0; i < lightSet->pointLightCount; ++i) {
		FzbRayTracingPointLight light = lightSet->pointLightInfoArray[i];
		glm::vec3 lightPos = glm::vec3(light.worldPos);
		glm::vec3 direction = lightPos - ray.hitPos;
		tempRay.depth = FLT_MAX;
		tempRay.direction = glm::normalize(direction);
		FzbTriangleAttribute hitTriangleAttribute;
		bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, tempRay, hitTriangleAttribute, false);
		if (hit) {	//如果打中，并且距离点光源很远，则说明中间有遮挡物
			if (glm::length((tempRay.depth * tempRay.direction + tempRay.startPos) - lightPos) > 0.01f) continue;
		}
		float r2 = glm::length(direction); r2 *= r2;
		float cosTheta = glm::clamp(glm::dot(triangleAttribute.normal, tempRay.direction), 0.0f, 1.0f);
		radiance += cosTheta * glm::vec3(light.radiantIntensity) / r2 * getBSDF(triangleAttribute, tempRay.direction, -ray.direction);
	}
	for (int i = 0; i < lightSet->areaLightCount; ++i) {
		FzbRayTracingAreaLight light = lightSet->areaLightInfoArray[i];

		float randomNumberX = getRandomNumber(randomNumberSeed);
		float randomNumberY = getRandomNumber(randomNumberSeed);

		glm::vec3 lightPos = glm::vec3(glm::vec3(light.worldPos) + randomNumberX * glm::vec3(light.edge0) + randomNumberY * glm::vec3(light.edge1));
		glm::vec3 direction = lightPos - ray.hitPos;
		tempRay.startPos = ray.hitPos + direction * 0.001f;
		tempRay.depth = FLT_MAX;
		tempRay.direction = glm::normalize(direction);

		FzbTriangleAttribute hitTriangleAttribute;
		bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, tempRay, hitTriangleAttribute, false);
		if (!hit) continue;
		else if (abs(tempRay.depth - glm::length(direction)) > 0.1f) continue;
		glm::vec3 lightRadiance_cosTheta = glm::vec3(light.radiance) * glm::clamp(glm::dot(triangleAttribute.normal, tempRay.direction), 0.0f, 1.0f);
		lightRadiance_cosTheta *= getBSDF(triangleAttribute, tempRay.direction, -ray.direction) * light.area;	//bsdf / pdf
		lightRadiance_cosTheta *= glm::clamp(glm::dot(glm::vec3(-light.normal), tempRay.direction), 0.0f, 1.0f);	//微分单位从dw换为dA
		float r2 = glm::max(glm::length(direction), 0.1f);	r2 *= r2;
		lightRadiance_cosTheta /= r2;
		radiance += lightRadiance_cosTheta;
	}
	return radiance;
}

__device__ glm::vec3 getRadiance(FzbTriangleAttribute triangleAttribute, FzbRay ray, const FzbRayTracingLightSet* lightSet,
	const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, uint32_t& randomNumberSeed) {
	glm::vec3 radiance = glm::vec3(0.0f);
	radiance += NEE(triangleAttribute, ray, lightSet, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, randomNumberSeed);
	radiance += triangleAttribute.emissive;
	return radiance;
}

