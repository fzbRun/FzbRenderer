#include "./FzbGetIllumination.cuh"
#include "FzbCollisionDetection.cuh"

__device__ float DistributionGGX(const glm::vec3& N, const glm::vec3& H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(glm::dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return nom / denom;
}
__device__ float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom / denom;
}

__device__ float GeometrySmith(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, float roughness)
{
	float NdotV = max(glm::dot(N, V), 0.0);
	float NdotL = max(glm::dot(N, L), 0.0);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

__device__ glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& F0)
{
	return F0 + (1.0f - F0) * pow(glm::clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

__device__ glm::vec3 getBSDF(const FzbTriangleAttribute& triangleAttribute, const glm::vec3& incidence, const glm::vec3& outgoing) {
	if(triangleAttribute.materialType == 0) return glm::vec3(PI_countdown) * triangleAttribute.albedo;
	else if (triangleAttribute.materialType == 1) {
		glm::vec3 h = normalize(incidence + outgoing);
		float NDF = DistributionGGX(triangleAttribute.normal, h, triangleAttribute.roughness);
		float G = GeometrySmith(triangleAttribute.normal, outgoing, incidence, triangleAttribute.roughness);
		glm::vec3 F = fresnelSchlick(glm::max(glm::dot(h, outgoing), 0.0f), triangleAttribute.albedo);

		glm::vec3 fr = NDF * G * F;
		float denominator = 4.0f * glm::max(glm::dot(triangleAttribute.normal, outgoing), 0.0f) * glm::max(glm::dot(triangleAttribute.normal, incidence), 0.0f) + 0.01f;
		return fr /= denominator;
	}
	return glm::vec3(0.0f);
}

/*
返回从光源到上一个撞击点的radiance
这里应该有优化，即不对所有光源进行采样，而是随机的方，但是目前先放着
*/
__device__ glm::vec3 NEE(FzbTriangleAttribute& triangleAttribute, FzbRay& ray, const FzbRayTracingLightSet* lightSet,
	const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, uint32_t& randomNumberSeed) {
	glm::vec3 radiance = glm::vec3(0.0f);
	FzbRay tempRay;
	FzbTriangleAttribute hitTriangleAttribute;
	for (int i = 0; i < lightSet->pointLightCount; ++i) {
		const FzbRayTracingPointLight& light = lightSet->pointLightInfoArray[i];
		glm::vec3 direction = light.worldPos - ray.hitPos;
		tempRay.depth = FLT_MAX;
		tempRay.direction = glm::normalize(direction);
		bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, tempRay, hitTriangleAttribute, false);
		if (!hit) continue;
		else if (abs(tempRay.depth - glm::length(direction)) > 0.1f) continue;
		float r2 = glm::length(direction); r2 *= r2;
		float cosTheta = glm::clamp(glm::dot(triangleAttribute.normal, tempRay.direction), 0.0f, 1.0f);
		radiance += cosTheta * light.radiantIntensity / r2 * getBSDF(triangleAttribute, tempRay.direction, -ray.direction);
	}
	for (int i = 0; i < lightSet->areaLightCount; ++i) {
		const FzbRayTracingAreaLight& light = lightSet->areaLightInfoArray[i];
		float randomNumberX = rand(randomNumberSeed);
		float randomNumberY = rand(randomNumberSeed);
		glm::vec3 lightPos = glm::vec3(light.worldPos + randomNumberX * light.edge0 + randomNumberY * light.edge1);
		glm::vec3 direction = lightPos - ray.hitPos;
		tempRay.startPos = ray.hitPos + direction * 0.001f;
		tempRay.depth = FLT_MAX;
		tempRay.direction = glm::normalize(direction);
		float r = glm::length(direction);

		bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, tempRay, hitTriangleAttribute, false);
		if (!hit) continue;
		else if (abs(tempRay.depth - r) > 0.1f) continue;
		glm::vec3 lightRadiance_cosTheta = light.radiance * glm::clamp(glm::dot(triangleAttribute.normal, tempRay.direction), 0.0f, 1.0f);
		lightRadiance_cosTheta *= getBSDF(triangleAttribute, tempRay.direction, -ray.direction) * light.area;	//bsdf / pdf
		lightRadiance_cosTheta *= glm::clamp(glm::dot(-light.normal, tempRay.direction), 0.0f, 1.0f);	//微分单位从dw换为dA
		r = glm::max(r, 1.0f);
		lightRadiance_cosTheta /= r * r;
		radiance += lightRadiance_cosTheta;
	}
	return radiance;
}

__device__ glm::vec3 getRadiance(FzbTriangleAttribute& triangleAttribute, FzbRay& ray, const FzbRayTracingLightSet* lightSet,
	const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, uint32_t& randomNumberSeed) {
	glm::vec3 radiance = glm::vec3(0.0f);
	radiance += NEE(triangleAttribute, ray, lightSet, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, randomNumberSeed);
	radiance += triangleAttribute.emissive;
	return radiance;
}

