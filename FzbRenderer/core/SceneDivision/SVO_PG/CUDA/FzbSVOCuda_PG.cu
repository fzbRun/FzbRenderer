#include "FzbSVOCuda_PG.cuh"
#include "../../../common/FzbRenderer.h"
#include "../../../RayTracing/CUDA/FzbCollisionDetection.cuh"

//----------------------------------------------uniformBuffer--------------------------------------
__constant__ FzbVGBUniformData systemVGBUniformData;
__constant__ FzbSVOUnformData systemSVOUniformData;

//----------------------------------------------核函数--------------------------------------
__global__ void lightInject_cuda(FzbVoxelData_PG* VGB, const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, const uint32_t rayCount);
//-------------------------------------------------------------------------------------------------
FzbSVOCuda_PG::FzbSVOCuda_PG() {};

__global__ void initSVO(FzbSVONodeData_PG* SVO, uint32_t svoCount) {
	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= svoCount) return;

	FzbSVONodeData_PG data;
	data.indivisible = false;
	data.shuffleKey = 0;
	data.label = 0;
	data.AABB.leftX = FLT_MAX;
	data.AABB.leftY = FLT_MAX;
	data.AABB.leftZ = FLT_MAX;
	data.AABB.rightX = -FLT_MAX;
	data.AABB.rightY = -FLT_MAX;
	data.AABB.rightZ = -FLT_MAX;
	data.irradiance = glm::vec3(0.0f);
	SVO[threadIndex] = data;
}
FzbSVOCuda_PG::FzbSVOCuda_PG(std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager, FzbSVOSetting_PG setting, 
	FzbVGBUniformData VGBUniformData, FzbBuffer VGB, HANDLE SVOFinishedSemaphore_PG, FzbSVOUnformData SVOUniformData) {
	this->sourceManager = sourceManager;
	this->setting = setting;

	this->VGBUniformData = VGBUniformData;
	CHECK(cudaMemcpyToSymbol(systemVGBUniformData, &VGBUniformData, sizeof(FzbVGBUniformData)));

	this->SVOUniformData = SVOUniformData;
	CHECK(cudaMemcpyToSymbol(systemSVOUniformData, &SVOUniformData, sizeof(FzbSVOUnformData)));

	this->VGBExtMem = importVulkanMemoryObjectFromNTHandle(VGB.handle, VGB.size, false);
	this->VGB = (FzbVoxelData_PG*)mapBufferOntoExternalMemory(VGBExtMem, 0, VGB.size);

	//创建各级SVO数组，从第二级开始
	uint32_t svoDepth = 2;	//算出SVO的深度，从1开始，即根节点为第一层
	uint32_t vgmSize = 2;
	while (vgmSize < setting.voxelNum) {
		svoDepth++;
		vgmSize <<= 1;
	}
	this->SVONodeCount.resize(svoDepth - 2);	 //不存储根节点和叶节点
	this->SVOs_PG.resize(svoDepth - 2);	 //不存储根节点和叶节点
	for (int i = 0; i < svoDepth - 2; ++i) {
		CHECK(cudaMalloc((void**)&this->SVONodeCount[i], sizeof(uint32_t)));
		uint32_t nodeCount = std::pow(8, i + 1);
		CHECK(cudaMalloc((void**)&this->SVOs_PG[i], nodeCount * sizeof(FzbSVONodeData_PG)));
		uint32_t blockSize = nodeCount > 1024 ? 1024 : nodeCount;
		uint32_t gridSize = (nodeCount + blockSize - 1) / blockSize;
		initSVO << <gridSize, blockSize >> > (this->SVOs_PG[i], nodeCount);
	}

	this->extSvoSemaphore_PG = importVulkanSemaphoreObjectFromNTHandle(SVOFinishedSemaphore_PG);

	//设置cuda配置，更多的使用L1 cache
	cudaFuncSetAttribute(lightInject_cuda, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
}

//--------------------------------------------------------------------初始化VGB-------------------------------------------------------------------------
__global__ void initVGB_Cuda(FzbVoxelData_PG* VGB, uint32_t voxelCount) {
	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= voxelCount) return;
	FzbVoxelData_PG data;
	data.hasData = 0;
	data.AABB.leftX = __float_as_int(FLT_MAX);
	data.AABB.leftY = __float_as_int(FLT_MAX);
	data.AABB.leftZ = __float_as_int(FLT_MAX);
	data.AABB.rightX = __float_as_int(-FLT_MAX);
	data.AABB.rightY = __float_as_int(-FLT_MAX);
	data.AABB.rightZ = __float_as_int(-FLT_MAX);
	data.irradiance = glm::vec3(0.0f);
	VGB[threadIndex] = data;
}
void FzbSVOCuda_PG::initVGB() {
	uint32_t voxelCount = std::pow(setting.voxelNum, 3);
	uint32_t gridSize = (voxelCount + 1023) / 1024;
	initVGB_Cuda << <gridSize, 1024 >> > (VGB, voxelCount);
	CHECK(cudaDeviceSynchronize());
}
//--------------------------------------------------------------------光照注入-------------------------------------------------------------------------
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
	voxelIndices[0] = voxelIndex.z * groupVGBUniformData.voxelCount * groupVGBUniformData.voxelCount + voxelIndex.y * groupVGBUniformData.voxelCount + voxelIndex.x;
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
		voxelIndices[pathLength] = voxelIndex.z * groupVGBUniformData.voxelCount * groupVGBUniformData.voxelCount + voxelIndex.y * groupVGBUniformData.voxelCount + voxelIndex.x;
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

	lightInject_cuda<<< gridSize, blockSize, 0, sourceManager->stream>>> (VGB, sourceManager->vertices, sourceManager->materialTextures, sourceManager->bvhNodeArray, sourceManager->bvhTriangleInfoArray, rayCount);
}
//--------------------------------------------------------------------创造SVO_PG-------------------------------------------------------------------------
__global__ void createSVO_PG_device_first(const FzbVoxelData_PG* __restrict__ VGB, FzbSVONodeData_PG* SVONodes, uint32_t voxelCount) {
	__shared__ FzbVGBUniformData groupVGBUniformData;
	__shared__ FzbSVOUnformData groupSVOUniformData;

	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;
	if (threadIndex >= voxelCount) return;
	if (threadIdx.x == 0) {
		groupVGBUniformData = systemVGBUniformData;
		groupSVOUniformData = systemSVOUniformData;
	}
	__syncthreads();

	//这里的block指的是父级node
	uint32_t indexInBlock = threadIndex & 7;	//在8个兄弟node中的索引
	uint32_t blockIndex = threadIndex / 8;		//block在全局的索引
	uint32_t blockIndexInWarpBit = (blockIndex & 3) * 8;	//当前block在warp中的位索引
	uint32_t blockCount = groupVGBUniformData.voxelCount / 2;	//每个轴有几个block

	uint32_t voxelIndexZ = (blockIndex / (blockCount * blockCount));
	uint32_t voxelIndexY = (blockIndex - voxelIndexZ * (blockCount * blockCount)) / blockCount;
	uint32_t voxelIndexX = blockIndex % blockCount;
	voxelIndexX = voxelIndexX * 2 + (indexInBlock & 1);
	voxelIndexY = voxelIndexY * 2 + ((indexInBlock >> 1) & 1);
	voxelIndexZ = voxelIndexZ * 2 + ((indexInBlock >> 2) & 1);
	uint32_t voxelIndexU = voxelIndexZ * (groupVGBUniformData.voxelCount * groupVGBUniformData.voxelCount) +
		voxelIndexY * groupVGBUniformData.voxelCount + voxelIndexX;
	FzbVoxelData_PG voxelData = VGB[voxelIndexU];
	bool hasData = voxelData.hasData && glm::length(voxelData.irradiance) > 0.01f;
	uint32_t activeMask = __ballot_sync(0xFFFFFFFF, hasData);
	int firstActiveLaneInBlock = __ffs(activeMask & (0xff << blockIndexInWarpBit)) - 1;
	if (firstActiveLaneInBlock == -1) return;	//当前block中node全部没有数据

	bool indivisible = true;
	FzbAABB AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	if (hasData) {
		AABB = {
			__int_as_float(voxelData.AABB.leftX),
			__int_as_float(voxelData.AABB.rightX),
			__int_as_float(voxelData.AABB.leftY),
			__int_as_float(voxelData.AABB.rightY),
			__int_as_float(voxelData.AABB.leftZ),
			__int_as_float(voxelData.AABB.rightZ)
		};
	}
	if (__popc(activeMask) == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasData) {
			SVONodes[blockIndex].indivisible = true;
			SVONodes[blockIndex].AABB = AABB;
			SVONodes[blockIndex].irradiance = voxelData.irradiance;
		}
		return;
	}
	//------------------------------------------得到整合后的AABB---------------------------------------------------------
	FzbAABB mergeAABB = AABB;
	//得到整合后的AABB的left
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftX, offset);
		mergeAABB.leftX = fminf(mergeAABB.leftX, other_val);
	}
	mergeAABB.leftX = __shfl_sync(0xFFFFFFFF, mergeAABB.leftX, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftY, offset);
		mergeAABB.leftY = fminf(mergeAABB.leftY, other_val);
	}
	mergeAABB.leftY = __shfl_sync(0xFFFFFFFF, mergeAABB.leftY, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftZ, offset);
		mergeAABB.leftZ = fminf(mergeAABB.leftZ, other_val);
	}
	mergeAABB.leftZ = __shfl_sync(0xFFFFFFFF, mergeAABB.leftZ, blockIndexInWarpBit);

	//得到整合后的AABB的right
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightX, offset);
		mergeAABB.rightX = fmaxf(mergeAABB.rightX, other_val);
	}
	mergeAABB.rightX = __shfl_sync(0xFFFFFFFF, mergeAABB.rightX, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightY, offset);
		mergeAABB.rightY = fmaxf(mergeAABB.rightY, other_val);
	}
	mergeAABB.rightY = __shfl_sync(0xFFFFFFFF, mergeAABB.rightY, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightZ, offset);
		mergeAABB.rightZ = fmaxf(mergeAABB.rightZ, other_val);
	}
	mergeAABB.rightZ = __shfl_sync(0xFFFFFFFF, mergeAABB.rightZ, blockIndexInWarpBit);
	//计算原来的AABB表面积
	float surfaceArea = 0.0f;
	if (hasData) {
		float lengthX = AABB.rightX - AABB.leftX;
		float lengthY = AABB.rightY - AABB.leftY;
		float lengthZ = AABB.rightZ - AABB.leftZ;
		surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	for (int offset = 4; offset > 0; offset /= 2) {
		surfaceArea += __shfl_down_sync(0xFFFFFFFF, surfaceArea, offset);
	}
	if (warpLane == blockIndexInWarpBit) {
		float lengthX = mergeAABB.rightX - mergeAABB.leftX;
		float lengthY = mergeAABB.rightY - mergeAABB.leftY;
		float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
		float mergeSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		if (mergeSurfaceArea / surfaceArea > groupSVOUniformData.surfaceAreaThreshold) indivisible = false;
	}
	//------------------------------------------------irradiance判断-------------------------------------------------
	float irrdianceValue = glm::length(voxelData.irradiance);
	if (hasData) {
		for (int i = 0; i < 8; ++i) {
			float other_val = __shfl_sync(0xFFFFFFFF, irrdianceValue, blockIndexInWarpBit + i);
			if (other_val <= 0.001f) continue;
			if (max(irrdianceValue, other_val) / min(irrdianceValue, other_val) > groupSVOUniformData.irradianceThreshold) indivisible = false;
		}
	}
	//--------------------------------------------------对父节点赋值-------------------------------------------------
	if (warpLane == blockIndexInWarpBit) {
		for (int i = 0; i < 8; ++i) {
			indivisible = indivisible && __shfl_sync(0xFFFFFFFF, indivisible, blockIndexInWarpBit + i);
		}
		SVONodes[blockIndex].indivisible = indivisible;
		SVONodes[blockIndex].AABB = mergeAABB;

		glm::vec3 mergeIrradiance = glm::vec3(0.0f);
		for (int i = 0; i < 8; ++i) {
			mergeIrradiance.x += __shfl_sync(0xFFFFFFFF, voxelData.irradiance.x, blockIndexInWarpBit + i);
			mergeIrradiance.y += __shfl_sync(0xFFFFFFFF, voxelData.irradiance.y, blockIndexInWarpBit + i);
			mergeIrradiance.z += __shfl_sync(0xFFFFFFFF, voxelData.irradiance.z, blockIndexInWarpBit + i);
		}
		SVONodes[blockIndex].irradiance = mergeIrradiance;
	}
	
	//for (int i = 0; i < 8; ++i) {
	//	bool brotherNodeHasData = __shfl_sync(0xFFFFFFFF, hasData, blockIndexInWarpBit + i);	//return的会返回0
	//	glm::vec3 brotherAABBCenterPos;
	//	brotherAABBCenterPos.x = __shfl_sync(0xFFFFFFFF, AABBCenterPos.x, blockIndexInWarpBit + i);
	//	brotherAABBCenterPos.y = __shfl_sync(0xFFFFFFFF, AABBCenterPos.y, blockIndexInWarpBit + i);
	//	brotherAABBCenterPos.z = __shfl_sync(0xFFFFFFFF, AABBCenterPos.z, blockIndexInWarpBit + i);
	//	glm::vec3 brotherIrradiance;
	//	brotherIrradiance.x = __shfl_sync(0xFFFFFFFF, voxelData.irradiance.x, blockIndexInWarpBit + i);
	//	brotherIrradiance.y = __shfl_sync(0xFFFFFFFF, voxelData.irradiance.y, blockIndexInWarpBit + i);
	//	brotherIrradiance.z = __shfl_sync(0xFFFFFFFF, voxelData.irradiance.z, blockIndexInWarpBit + i);
	//	if (!brotherNodeHasData || !hasData) continue;
	//	float distance = glm::length(brotherAABBCenterPos - AABBCenterPos) + glm::length(brotherIrradiance - voxelData.irradiance);
	//	if (distance > groupSVOUniformData.divideThreshold) indivisible = false;
	//}
	//if (indexInBlock == firstActiveLaneInBlock) SVONodes[blockIndex].indivisible = indivisible;
	//if (hasData) {
	//	atomicMinFloat(&SVONodes[blockIndex].AABB.leftX, AABB.leftX);
	//	atomicMaxFloat(&SVONodes[blockIndex].AABB.rightX, AABB.rightX);
	//	atomicMinFloat(&SVONodes[blockIndex].AABB.leftY, AABB.leftY);
	//	atomicMaxFloat(&SVONodes[blockIndex].AABB.rightY, AABB.rightY);
	//	atomicMinFloat(&SVONodes[blockIndex].AABB.leftZ, AABB.leftZ);
	//	atomicMaxFloat(&SVONodes[blockIndex].AABB.rightZ, AABB.rightZ);
	//	atomicAdd(&SVONodes[blockIndex].irradiance.x, voxelData.irradiance.x);
	//	atomicAdd(&SVONodes[blockIndex].irradiance.y, voxelData.irradiance.y);
	//	atomicAdd(&SVONodes[blockIndex].irradiance.z, voxelData.irradiance.z);
	//}
}
__global__ void createSVO_PG_device(FzbSVONodeData_PG* SVONodes_children, FzbSVONodeData_PG* SVONodes, uint32_t nodeCount) {
	__shared__ FzbSVOUnformData groupSVOUniformData;

	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;
	if (threadIndex >= nodeCount * nodeCount * nodeCount) return;
	if (threadIdx.x == 0) {
		groupSVOUniformData = systemSVOUniformData;
	}
	__syncthreads();

	//这里的block指的是父级node
	uint32_t indexInBlock = threadIndex & 7;	//在8个兄弟node中的索引
	uint32_t blockIndex = threadIndex / 8;		//block在全局的索引
	uint32_t blockIndexInWarpBit = (blockIndex & 3) * 8;	//当前block在warp中的位索引
	uint32_t blockCount = nodeCount / 2;	//每个轴有几个block

	uint32_t nodeIndexZ = (blockIndex / (blockCount * blockCount));
	uint32_t nodeIndexY = (blockIndex - nodeIndexZ * (blockCount * blockCount)) / blockCount;
	uint32_t nodeIndexX = blockIndex % blockCount;
	nodeIndexX = nodeIndexX * 2 + (indexInBlock & 1);
	nodeIndexY = nodeIndexY * 2 + ((indexInBlock >> 1) & 1);
	nodeIndexZ = nodeIndexZ * 2 + ((indexInBlock >> 2) & 1);
	uint32_t voxelIndexU = nodeIndexZ * (nodeCount * nodeCount) +
		nodeIndexY * nodeCount + nodeIndexX;
	FzbSVONodeData_PG nodeData = SVONodes_children[voxelIndexU];
	bool hasData = glm::length(nodeData.irradiance) > 0.01f;
	uint32_t activeMask = __ballot_sync(0xFFFFFFFF, hasData);
	int firstActiveLaneInBlock = __ffs(activeMask & (0xff << blockIndexInWarpBit)) - 1;
	if (firstActiveLaneInBlock == -1) return;	//当前block中node全部没有数据

	bool indivisible = true;
	if (__popc(activeMask) == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasData) {
			SVONodes[blockIndex].indivisible = true;
			SVONodes[blockIndex].AABB = nodeData.AABB;
			SVONodes[blockIndex].irradiance = nodeData.irradiance;
		}
		return;
	}
	//------------------------------------------得到整合后的AABB---------------------------------------------------------
	FzbAABB mergeAABB = nodeData.AABB;
	//得到整合后的AABB的left
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftX, offset);
		mergeAABB.leftX = fminf(mergeAABB.leftX, other_val);
	}
	mergeAABB.leftX = __shfl_sync(0xFFFFFFFF, mergeAABB.leftX, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftY, offset);
		mergeAABB.leftY = fminf(mergeAABB.leftY, other_val);
	}
	mergeAABB.leftY = __shfl_sync(0xFFFFFFFF, mergeAABB.leftY, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftZ, offset);
		mergeAABB.leftZ = fminf(mergeAABB.leftZ, other_val);
	}
	mergeAABB.leftZ = __shfl_sync(0xFFFFFFFF, mergeAABB.leftZ, blockIndexInWarpBit);

	//得到整合后的AABB的right
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightX, offset);
		mergeAABB.rightX = fmaxf(mergeAABB.rightX, other_val);
	}
	mergeAABB.rightX = __shfl_sync(0xFFFFFFFF, mergeAABB.rightX, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightY, offset);
		mergeAABB.rightY = fmaxf(mergeAABB.rightY, other_val);
	}
	mergeAABB.rightY = __shfl_sync(0xFFFFFFFF, mergeAABB.rightY, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightZ, offset);
		mergeAABB.rightZ = fmaxf(mergeAABB.rightZ, other_val);
	}
	mergeAABB.rightZ = __shfl_sync(0xFFFFFFFF, mergeAABB.rightZ, blockIndexInWarpBit);
	//计算原来的AABB表面积
	float surfaceArea = 0.0f;
	if (hasData) {
		float lengthX = nodeData.AABB.rightX - nodeData.AABB.leftX;
		float lengthY = nodeData.AABB.rightY - nodeData.AABB.leftY;
		float lengthZ = nodeData.AABB.rightZ - nodeData.AABB.leftZ;
		surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	for (int offset = 4; offset > 0; offset /= 2) {
		surfaceArea += __shfl_down_sync(0xFFFFFFFF, surfaceArea, offset);
	}
	if (warpLane == blockIndexInWarpBit) {
		float lengthX = mergeAABB.rightX - mergeAABB.leftX;
		float lengthY = mergeAABB.rightY - mergeAABB.leftY;
		float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
		float mergeSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		if (mergeSurfaceArea / surfaceArea > groupSVOUniformData.surfaceAreaThreshold) indivisible = false;
	}
	//------------------------------------------------irradiance判断-------------------------------------------------
	float irrdianceValue = glm::length(nodeData.irradiance);
	if (hasData) {
		for (int i = 0; i < 8; ++i) {
			float other_val = __shfl_sync(0xFFFFFFFF, irrdianceValue, blockIndexInWarpBit + i);
			if (other_val <= 0.001f) continue;
			if (max(irrdianceValue, other_val) / min(irrdianceValue, other_val) > groupSVOUniformData.irradianceThreshold) indivisible = false;
		}
	}
	//--------------------------------------------------对父节点赋值-------------------------------------------------
	if (warpLane == blockIndexInWarpBit) {
		for (int i = 0; i < 8; ++i) {
			indivisible = indivisible && __shfl_sync(0xFFFFFFFF, indivisible, blockIndexInWarpBit + i);
		}
		SVONodes[blockIndex].indivisible = indivisible;
		SVONodes[blockIndex].AABB = mergeAABB;

		glm::vec3 mergeIrradiance = glm::vec3(0.0f);
		for (int i = 0; i < 8; ++i) {
			mergeIrradiance.x += __shfl_sync(0xFFFFFFFF, nodeData.irradiance.x, blockIndexInWarpBit + i);
			mergeIrradiance.y += __shfl_sync(0xFFFFFFFF, nodeData.irradiance.y, blockIndexInWarpBit + i);
			mergeIrradiance.z += __shfl_sync(0xFFFFFFFF, nodeData.irradiance.z, blockIndexInWarpBit + i);
		}
		SVONodes[blockIndex].irradiance = mergeIrradiance;
	}
}
void FzbSVOCuda_PG::createSVOCuda_PG() {
	uint32_t voxelCount = std::pow(setting.voxelNum, 3);
	uint32_t blockSize = 512;
	uint32_t gridSize = (voxelCount + blockSize - 1) / blockSize;
	createSVO_PG_device_first<<<gridSize, blockSize, 0, sourceManager->stream>>>(VGB, SVOs_PG[SVOs_PG.size() - 1], voxelCount);
	for (int i = SVOs_PG.size() - 1; i > 0; --i) {
		FzbSVONodeData_PG* SVONodes_children = SVOs_PG[i];
		FzbSVONodeData_PG* SVONodes = SVOs_PG[i - 1];
		uint32_t nodeCount = setting.voxelNum / pow(2, SVOs_PG.size() - i);
		uint32_t nodeTotalCount = nodeCount * nodeCount * nodeCount;
		blockSize = nodeTotalCount > 512 ? 512 : nodeTotalCount;
		gridSize = (nodeTotalCount + blockSize - 1) / blockSize;
		createSVO_PG_device<<<gridSize, blockSize, 0, sourceManager->stream>>>(SVONodes_children, SVONodes, nodeCount);
	}
	CHECK(cudaDeviceSynchronize());
}
//----------------------------------------------------------------------------------------------------------------------------------------------------
void FzbSVOCuda_PG::clean() {
	CHECK(cudaDestroyExternalMemory(VGBExtMem));
	CHECK(cudaFree(VGB));
	for (int i = 0; i < this->SVONodeCount.size(); ++i) CHECK(cudaFree(this->SVONodeCount[i]));
	for (int i = 0; i < this->SVOs_PG.size(); ++i) CHECK(cudaFree(this->SVOs_PG[i]));
	CHECK(cudaDestroyExternalSemaphore(extSvoSemaphore_PG));
}

void FzbSVOCuda_PG::copyDataToBuffer(std::vector<FzbBuffer>& buffers) {
	if (buffers.size() != SVOs_PG.size()) throw std::runtime_error("SVOBuffer数量不匹配");
	for (int i = 0; i < buffers.size(); ++i) {
		FzbBuffer& SVOBuffer = buffers[i];
		cudaExternalMemory_t SVOBufferExtMem = importVulkanMemoryObjectFromNTHandle(SVOBuffer.handle, SVOBuffer.size, false);
		FzbSVONodeData_PG* SVOBuffer_ptr = (FzbSVONodeData_PG*)mapBufferOntoExternalMemory(SVOBufferExtMem, 0, SVOBuffer.size);
		CHECK(cudaMemcpy(SVOBuffer_ptr, SVOs_PG[i], SVOBuffer.size, cudaMemcpyDeviceToDevice));
		CHECK(cudaDestroyExternalMemory(SVOBufferExtMem));
		CHECK(cudaFree(SVOBuffer_ptr));
	}
}
