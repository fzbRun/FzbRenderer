#include "FzbSVOCuda_PG.cuh"
#include "../../../common/FzbRenderer.h"
#include "../../../RayTracing/CUDA/FzbCollisionDetection.cuh"

__device__ int getVGBVoxelIndex(int voxelCount, glm::ivec3& voxelIndex) {
	int voxelTotalCount = voxelCount * voxelCount * voxelCount;
	int voxelIndexU = 0;
	while (voxelTotalCount > 1) {
		voxelCount = voxelCount / 2;
		voxelTotalCount = voxelTotalCount / 8;
		if (voxelIndex.z / voxelCount == 1) {
			voxelIndexU += 4 * voxelTotalCount;
			voxelIndex.z -= voxelCount;
		}
		if (voxelIndex.y / voxelCount == 1) {
			voxelIndexU += 2 * voxelTotalCount;
			voxelIndex.y -= voxelCount;
		}
		if (voxelIndex.x / voxelCount == 1) {
			voxelIndexU += voxelTotalCount;
			voxelIndex.x -= voxelCount;
		}
	}
	return voxelIndexU;
}
__device__ void lightInject_getRadiance(FzbTriangleAttribute& triangleAttribute, FzbRay& ray, const FzbRayTracingLightSet* lightSet,
	const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, uint32_t& randomNumberSeed,
	glm::vec3& irradiance, glm::vec3& radiance) {
	irradiance = triangleAttribute.emissive;
	radiance = triangleAttribute.emissive;
	FzbRay tempRay;
	FzbTriangleAttribute hitTriangleAttribute;
	for (int i = 0; i < lightSet->pointLightCount; ++i) {
		const FzbRayTracingPointLight& light = lightSet->pointLightInfoArray[i];
		glm::vec3 direction = light.worldPos - ray.hitPos;
		if (glm::dot(direction, triangleAttribute.normal) <= 0) continue;
		tempRay.depth = FLT_MAX;
		tempRay.direction = glm::normalize(direction);
		bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, tempRay, hitTriangleAttribute, false);
		if (!hit) continue;
		else if (abs(tempRay.depth - glm::length(direction)) > 0.1f) continue;
		float r2 = glm::length(direction); r2 *= r2;
		float cosTheta = glm::clamp(glm::dot(triangleAttribute.normal, tempRay.direction), 0.0f, 1.0f);
		glm::vec3 irradiance_temp = cosTheta * light.radiantIntensity / r2;
		irradiance += irradiance_temp;
		radiance += irradiance_temp * getBSDF(triangleAttribute, tempRay.direction, -ray.direction, ray);
	}
	for (int i = 0; i < lightSet->areaLightCount; ++i) {
		const FzbRayTracingAreaLight& light = lightSet->areaLightInfoArray[i];
		float randomNumberX = rand(randomNumberSeed);
		float randomNumberY = rand(randomNumberSeed);
		glm::vec3 lightPos = glm::vec3(light.worldPos + randomNumberX * light.edge0 + randomNumberY * light.edge1);
		glm::vec3 direction = lightPos - ray.hitPos;
		if (triangleAttribute.materialType != 2 && glm::dot(direction, triangleAttribute.normal) <= 0) continue;
		tempRay.startPos = ray.hitPos + direction * 0.001f;
		tempRay.depth = FLT_MAX;
		tempRay.direction = glm::normalize(direction);
		float r = glm::length(direction);
		bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, tempRay, hitTriangleAttribute, false);
		if (!hit) continue;
		else if (abs(tempRay.depth - r) > 0.1f) continue;
		glm::vec3 lightRadiance_cosTheta = light.radiance * glm::clamp(glm::dot(triangleAttribute.normal, tempRay.direction), 0.0f, 1.0f);
		lightRadiance_cosTheta *= light.area;	// pdf = 1 / area
		lightRadiance_cosTheta *= glm::clamp(glm::dot(-light.normal, tempRay.direction), 0.0f, 1.0f);	//微分单位从dw换为dA
		r = glm::max(r, 1.0f);
		glm::vec3 irradiance_temp = lightRadiance_cosTheta / (r * r);
		irradiance += irradiance_temp;
		radiance += irradiance_temp * getBSDF(triangleAttribute, tempRay.direction, -ray.direction, tempRay);
	}
}
__global__ void lightInject_cuda(FzbVoxelData_PG* VGB, const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, const uint32_t rayCount) {
	__shared__ FzbPathTracingCameraInfo groupCameraInfo;				//216B
	__shared__ FzbVGBUniformData groupVGBUniformData;				//216B
	__shared__ uint32_t groupRandomNumberSeed;
	__shared__ FzbRayTracingPointLight groupPointLightInfoArray[maxPointLightCount];	//512B
	__shared__ FzbRayTracingAreaLight grouprAreaLightInfoArray[maxAreaLightCount];		//692B
	__shared__ FzbRayTracingLightSet lightSet;

	volatile const uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= rayCount) return;
	if (threadIdx.x < systemPointLightCount) groupPointLightInfoArray[threadIdx.x] = systemPointLightInfoArray[threadIdx.x];
	if (threadIdx.x < systemAreaLightCount) grouprAreaLightInfoArray[threadIdx.x] = systemAreaLightInfoArray[threadIdx.x];
	if (threadIdx.x == 0) {
		groupCameraInfo = systemCameraInfo;
		groupVGBUniformData = systemVGBUniformData;	
		groupRandomNumberSeed = systemRandomNumberSeed;
		lightSet.pointLightCount = systemPointLightCount;
		lightSet.areaLightCount = systemAreaLightCount;
		lightSet.pointLightInfoArray = groupPointLightInfoArray;
		lightSet.areaLightInfoArray = grouprAreaLightInfoArray;
	}
	__syncthreads();

	uint32_t randomNumberSeed = groupRandomNumberSeed + threadIndex;
	uint2 seed2 = pcg2d(make_uint2(threadIndex) * randomNumberSeed);
	randomNumberSeed = seed2.x + seed2.y;

	const uint32_t maxPathDepth = 3;
	glm::vec3 voxelRadiance[maxPathDepth];	//传回上一个撞击点的radiance
	glm::vec3 voxelIrradiances[maxPathDepth];	//当前撞击点得到的irradiance，包括NEE + 上一个点的radiance
	uint32_t voxelIndices[maxPathDepth];
	glm::vec3 voxelBSDF[maxPathDepth - 1];
	float voxelPDF[maxPathDepth - 1];
	float voxelCosTheta[maxPathDepth - 1];

	float RR = 0.8f;
	bool hit = true;
	FzbTriangleAttribute hitTriangleAttribute;
	FzbTriangleAttribute lastHitTriangleAttribute;

	glm::vec2 texelXY = glm::vec2(threadIndex % groupCameraInfo.screenWidth, threadIndex / groupCameraInfo.screenWidth);
	glm::vec4 screenPos = glm::vec4(((texelXY + glm::vec2(0.5f)) / glm::vec2(groupCameraInfo.screenWidth, groupCameraInfo.screenHeight)) * 2.0f - 1.0f, 0.0f, 1.0f);	//vulkan中近平面ndcDepth在[0,1]
	screenPos = groupCameraInfo.inversePVMatrix * screenPos;
	screenPos /= screenPos.w;
	FzbRay ray;
	ray.startPos = groupCameraInfo.cameraWorldPos;
	ray.direction = glm::normalize(glm::vec3(screenPos) - ray.startPos);
	ray.depth = FLT_MAX;
	ray.refraction = false;
	ray.ext = true;
	glm::vec3 lastDirection = -ray.direction;

	hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, ray, hitTriangleAttribute);
	if (!hit) return;
	glm::ivec3 voxelIndex = glm::ivec3((ray.hitPos - groupVGBUniformData.voxelStartPos) / groupVGBUniformData.voxelSize);
	voxelIndices[0] = getVGBVoxelIndex(groupVGBUniformData.voxelCount, voxelIndex);
	lightInject_getRadiance(hitTriangleAttribute, ray, &lightSet,
		vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, randomNumberSeed,
		voxelIrradiances[0], voxelRadiance[0]);

	int pathLength = 1;
#pragma nounroll
	while (pathLength < maxPathDepth) {
		float randomNumber = rand(randomNumberSeed);
		if (randomNumber > RR) break;
		voxelPDF[pathLength - 1] = RR;

		lastDirection = -ray.direction;
		generateRay(hitTriangleAttribute, voxelPDF[pathLength - 1], ray, randomNumberSeed);

		lastHitTriangleAttribute = hitTriangleAttribute;
		hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, ray, hitTriangleAttribute);
		if (!hit) break;

		voxelBSDF[pathLength - 1] = getBSDF(lastHitTriangleAttribute, ray.direction, lastDirection, ray);
		voxelCosTheta[pathLength - 1] = glm::abs(glm::dot(ray.direction, lastHitTriangleAttribute.normal));

		voxelIndex = glm::ivec3((ray.hitPos - groupVGBUniformData.voxelStartPos) / groupVGBUniformData.voxelSize);
		voxelIndices[pathLength] = getVGBVoxelIndex(groupVGBUniformData.voxelCount, voxelIndex);
		lightInject_getRadiance(hitTriangleAttribute, ray, &lightSet,
			vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, randomNumberSeed,
			voxelIrradiances[pathLength], voxelRadiance[pathLength]);

		++pathLength;
	}

	atomicAdd(&VGB[voxelIndices[pathLength - 1]].irradiance.x, voxelIrradiances[pathLength - 1].x);
	atomicAdd(&VGB[voxelIndices[pathLength - 1]].irradiance.y, voxelIrradiances[pathLength - 1].y);
	atomicAdd(&VGB[voxelIndices[pathLength - 1]].irradiance.z, voxelIrradiances[pathLength - 1].z);
	glm::vec3 radiance = voxelRadiance[pathLength - 1];
	for (int i = pathLength - 2; i >= 0; --i) {
		voxelIrradiances[i] += radiance * voxelCosTheta[i];

		atomicAdd(&VGB[voxelIndices[i]].irradiance.x, voxelIrradiances[i].x);
		atomicAdd(&VGB[voxelIndices[i]].irradiance.y, voxelIrradiances[i].y);
		atomicAdd(&VGB[voxelIndices[i]].irradiance.z, voxelIrradiances[i].z);

		radiance = voxelRadiance[i] + radiance * voxelBSDF[i] / voxelPDF[i];
	}
}

void FzbSVOCuda_PG::lightInject() {
	this->sourceManager->createRuntimeSource();

	VkExtent2D resolution = FzbRenderer::globalData.getResolution();
	uint32_t texelCount = resolution.width * resolution.height;
	uint32_t rayCount = texelCount;
	uint32_t blockSize = 512;
	uint32_t gridSize = (rayCount + blockSize - 1) / blockSize;

	lightInject_cuda << < gridSize, blockSize, 0, stream >> > (VGB, sourceManager->vertices, sourceManager->materialTextures, sourceManager->bvhNodeArray, sourceManager->bvhTriangleInfoArray, rayCount);
}

void FzbSVOCuda_PG::initLightInjectSource() {
	//设置cuda配置，更多的使用L1 cache
	cudaFuncSetAttribute(lightInject_cuda, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
}