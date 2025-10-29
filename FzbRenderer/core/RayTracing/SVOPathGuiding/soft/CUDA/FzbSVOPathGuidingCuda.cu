#include "./FzbSVOPathGuidingCuda.cuh"
#include "../../../../common/FzbRenderer.h"
#include "../../../CUDA/FzbRayGenerate.cuh"
#include "../../../CUDA/FzbGetTriangleAttribute.cuh"
#include "../../../CUDA/FzbCollisionDetection.cuh"
#include "../../../CUDA/FzbGetIllumination.cuh"

//----------------------------------------------uniformBuffer--------------------------------------
__constant__ FzbSVOPathGuidingCudaSetting svoPathGuidingSetting;

const uint32_t blockSize = 128;
const uint32_t sharedMemorySPP = 2;
//----------------------------------------------SVOPathGuiding采样-------------------------------------
__device__ void generateRay_SVOPathGuiding(
	const FzbTriangleAttribute& hitTriangleAttribute, float& pdf, FzbRay& ray, uint32_t& randomNumberSeed,
	FzbSVOPathGuidingCudaSetting& groupSetting)
{
	//根据hitPos找到当前node是哪个
	FzbSVONodeData_PG nodeData; nodeData.label = 1;
	glm::vec3 nodeGroupStartPos = groupSetting.voxelGroupStartPos;
	glm::vec3 nodeSize = (float)groupSetting.voxelCount * groupSetting.voxelSize;
	uint32_t layerIndex = 1;
	uint32_t nodeDataIndex = 0;

	for (; layerIndex < groupSetting.maxSVOLayer; ++layerIndex) {
		nodeSize /= 2.0f;
		glm::ivec3 nodeIndexXYZ = glm::ivec3((ray.hitPos - nodeGroupStartPos) / nodeSize);
		nodeDataIndex = nodeIndexXYZ.x + 2 * nodeIndexXYZ.y + 4 * nodeIndexXYZ.z + (nodeData.label - 1) * 8;
		nodeData = groupSetting.SVONodes[layerIndex][nodeDataIndex];
		if (nodeData.indivisible) break;

		nodeGroupStartPos += glm::vec3(nodeIndexXYZ) * nodeSize;
	}
	if (nodeData.label == 0) {		//几何注入依靠光栅化，是离散的，有的几何可能没有注入进去，但是光线离散程度更小，可能打中
		generateRay(hitTriangleAttribute, pdf, ray, randomNumberSeed);
		return;
	}

	uint32_t layerNodeSum = (nodeData.label - 1) * groupSetting.SVONodeTotalCount;
	uint32_t nodeWeightStartIndex = layerNodeSum;
	uint32_t targetLayerIndex = 1;
	uint32_t targetNodeDataIndex = 0;
	FzbSVONodeData_PG targetNodeData;
	bool getTargetNode = false;
	int childNodeIndex;
	float selectNodWeightSum = 1.0f;
	for (; targetLayerIndex < groupSetting.maxSVOLayer; ++targetLayerIndex) {
		float randomNumber = rand(randomNumberSeed);
		for (childNodeIndex = 0; childNodeIndex < 8; ++childNodeIndex) {
			float weight = groupSetting.SVONodeWeights[nodeWeightStartIndex + childNodeIndex];
			if (randomNumber <= weight) {
				targetNodeData = groupSetting.SVONodes[targetLayerIndex][targetNodeDataIndex + childNodeIndex];
				selectNodWeightSum *= weight;	//找到该node的概率密度
				if (targetNodeData.indivisible) getTargetNode = true;
				else targetNodeDataIndex = (targetNodeData.label - 1) * 8;	//其子节点在该层的起始索引
				break;
			}
			randomNumber -= weight;
		}
		if (getTargetNode || childNodeIndex == 8) break;
		layerNodeSum += groupSetting.SVOLayerInfos[targetLayerIndex - 1].divisibleNodeCount * 8;
		nodeWeightStartIndex = layerNodeSum + targetNodeDataIndex;
	}

	if (glm::length(targetNodeData.irradiance) == 0 || childNodeIndex == 8) {
		generateRay(hitTriangleAttribute, pdf, ray, randomNumberSeed);
		return;
	}

	float targetDistanceX = targetNodeData.AABB.rightX - targetNodeData.AABB.leftX;
	float targetDistanceY = targetNodeData.AABB.rightY - targetNodeData.AABB.leftY;
	float targetDistanceZ = targetNodeData.AABB.rightZ - targetNodeData.AABB.leftZ;

	glm::vec3 nodeCenterPos = glm::vec3(nodeData.AABB.leftX + nodeData.AABB.rightX, nodeData.AABB.leftY + nodeData.AABB.rightY, nodeData.AABB.leftZ + nodeData.AABB.rightZ) * 0.5f;
	glm::vec3 targetNodeCenterPos = glm::vec3(targetNodeData.AABB.leftX + targetNodeData.AABB.rightX, targetNodeData.AABB.leftY + targetNodeData.AABB.rightY, targetNodeData.AABB.leftZ + targetNodeData.AABB.rightZ) * 0.5f;
	glm::vec3 nodeDirection = targetNodeCenterPos - nodeCenterPos;

	glm::vec3 faceArea;
	faceArea.x = targetDistanceY * targetDistanceZ;
	faceArea.y = targetDistanceX * targetDistanceZ;
	faceArea.z = targetDistanceX * targetDistanceY;
	glm::vec3 faceSelectWeight = glm::normalize(faceArea);
	float faceSelectRandomNumber = rand(randomNumberSeed);
	uint32_t faceIndex = faceSelectRandomNumber <= faceSelectWeight.x ? 0 : faceSelectRandomNumber <= faceSelectWeight.x + faceSelectWeight.y ? 1 : 2;

	float selectFaceArea = 1.0f;
	float selectFacePDF = 1.0f;
	glm::vec3 faceNormal = glm::vec3(0.0f);
	glm::vec3 faceStartPos = glm::vec3(targetNodeData.AABB.leftX, targetNodeData.AABB.leftY, targetNodeData.AABB.leftZ);
	if (faceIndex == 0) {
		faceStartPos.x += nodeDirection.x < 0 ? targetDistanceX : 0.0f;	//在左边
		faceNormal.x = nodeDirection.x < 0 ? 1.0f : -1.0f;
		selectFaceArea = faceArea.x;
		selectFacePDF = faceSelectWeight.x;
	}
	else if (faceIndex == 1) {
		faceStartPos.y += nodeDirection.y < 0 ? targetDistanceY : 0.0f;	//在下边
		faceNormal.y = nodeDirection.y < 0 ? 1.0f : -1.0f;
		selectFaceArea = faceArea.y;
		selectFacePDF = faceSelectWeight.y;
	}
	else {
		faceStartPos.z += nodeDirection.z < 0 ? targetDistanceZ : 0.0f;	//在后边
		faceNormal.z = nodeDirection.z < 0 ? 1.0f : -1.0f;
		selectFaceArea = faceArea.z;
		selectFacePDF = faceSelectWeight.z;
	}

	float randomU = rand(randomNumberSeed);		//当前node的AABB上的随机点
	float randomV = rand(randomNumberSeed);
	bool useSphericalRectangleSample = true;
	float sphericalRectangleSamplePDF = 1.0f;
	if(useSphericalRectangleSample){
		FzbQuadrilateral quadFace;
		quadFace.worldPos = faceStartPos;
		quadFace.normal = faceNormal;
		if (faceIndex == 0) {
			quadFace.edge0 = targetDistanceZ * glm::vec3(0.0f, 0.0f, 1.0f);
			quadFace.edge1 = targetDistanceY * glm::vec3(0.0f, 1.0f, 0.0f);
		}
		else if (faceIndex == 1) {
			quadFace.edge0 = targetDistanceX * glm::vec3(1.0f, 0.0f, 0.0f);
			quadFace.edge1 = targetDistanceZ * glm::vec3(0.0f, 0.0f, 1.0f);
		}
		else {
			quadFace.edge0 = targetDistanceX * glm::vec3(1.0f, 0.0f, 0.0f);
			quadFace.edge1 = targetDistanceY * glm::vec3(0.0f, 1.0f, 0.0f);
		}

		ray.direction = sphericalRectangleSample(quadFace, ray.hitPos, randomU, randomV, sphericalRectangleSamplePDF);
	}
	else {
		glm::vec3 samplePos = faceStartPos;
		if (faceIndex == 0) {
			samplePos.z += randomU * targetDistanceZ;
			samplePos.y += randomV * targetDistanceY;
		}
		else if (faceIndex == 1) {
			samplePos.x += randomU * targetDistanceX;
			samplePos.z += randomV * targetDistanceZ;
		}
		else {
			samplePos.x += randomU * targetDistanceX;
			samplePos.y += randomV * targetDistanceY;
		}

		ray.direction = samplePos - ray.hitPos;
	}

	if (glm::dot(ray.direction, hitTriangleAttribute.normal) <= 0) generateRay(hitTriangleAttribute, pdf, ray, randomNumberSeed);
	else {
		pdf *= selectNodWeightSum * selectFacePDF;	//最终采样node的AABB可能忽略了一些包围盒，需要乘以pdf来弥补
		pdf *= sphericalRectangleSamplePDF;
		if (!useSphericalRectangleSample) 
			pdf *= glm::dot(-glm::normalize(ray.direction), faceNormal) / glm::max(selectFaceArea * glm::max(glm::length(ray.direction), 0.01f), 0.001f);

		ray.direction = glm::normalize(ray.direction);
		ray.startPos = ray.direction * 0.01f + ray.hitPos;
		ray.depth = FLT_MAX;
	}
}
//-------------------------------------------------------------------------------------------------
template<bool useExternSharedMemory>
__global__ void svoPathGuiding_cuda(
	float4* resultBuffer, const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, const uint32_t rayCount
) {
	extern __shared__ float3 groupResultRadiance[];	//如果spp大于sharedMemorySPP，则使用共享内存， 数量为blockDim / spp, 最大6KB
	__shared__ FzbSVOPathGuidingCudaSetting groupSetting;
	__shared__ uint32_t groupFrameCount;
	__shared__ FzbPathTracingCameraInfo groupCameraInfo;				//216B
	__shared__ uint32_t groupRandomNumberSeed;
	__shared__ FzbRayTracingPointLight groupPointLightInfoArray[maxPointLightCount];	//512B
	__shared__ FzbRayTracingAreaLight grouprAreaLightInfoArray[maxAreaLightCount];		//692B
	__shared__ FzbRayTracingLightSet lightSet;

	__shared__ FzbSVOLayerInfo groupSVOLayerInfos[8];
	__shared__ FzbSVONodeData_PG* groupSVONodesArray[8];

	volatile const uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= rayCount) return;

	if (threadIdx.x < systemPointLightCount) groupPointLightInfoArray[threadIdx.x] = systemPointLightInfoArray[threadIdx.x];
	if (threadIdx.x < systemAreaLightCount) grouprAreaLightInfoArray[threadIdx.x] = systemAreaLightInfoArray[threadIdx.x];
	if (threadIdx.x == 0) {
		groupSetting = svoPathGuidingSetting;
		groupFrameCount = systemFrameCount;
		groupCameraInfo = systemCameraInfo;
		groupRandomNumberSeed = systemRandomNumberSeed;
		lightSet.pointLightCount = systemPointLightCount;
		lightSet.areaLightCount = systemAreaLightCount;
		lightSet.pointLightInfoArray = groupPointLightInfoArray;
		lightSet.areaLightInfoArray = grouprAreaLightInfoArray;
	}
	__syncwarp();
	if (threadIdx.x < groupSetting.maxSVOLayer) {
		groupSVOLayerInfos[threadIdx.x] = groupSetting.SVOLayerInfos[threadIdx.x];
		groupSVONodesArray[threadIdx.x] = groupSetting.SVONodes[threadIdx.x];
	}
	__syncwarp();
	if (threadIdx.x == 0) {		//将全局内存改为共享内存
		groupSetting.SVOLayerInfos = groupSVOLayerInfos;
		groupSetting.SVONodes = groupSVONodesArray;
	}
	__syncthreads();

	volatile const uint32_t& spp = groupSetting.spp;		//寄存器不够用，挤到了localMmeory，那还不如直接用sharedMemory呢
	uint32_t resultIndex = threadIndex / spp;	//属于第几个spp，即bufferIndex
	uint32_t groupSppIndex = threadIdx.x / spp;		//组内第几个spp
	uint32_t sppLane = threadIndex % spp;	//不能用&，因为spp可能不是2的幂次
	if (useExternSharedMemory) {
		if (threadIdx.x < blockDim.x / spp) groupResultRadiance[threadIdx.x] = make_float3(0.0f);
	}
	//if (threadIndex < systemCameraInfo.screenWidth * systemCameraInfo.screenHeight * spp) resultBuffer[threadIndex] = make_float4(0.0f);
	//if (sppLane == 0) resultBuffer[resultIndex] = make_float4(0.0f);
	if (sppLane == 0) resultBuffer[resultIndex] *= ((float)groupFrameCount - 1) / (float)groupFrameCount;
	__syncthreads();

	uint32_t randomNumberSeed = groupRandomNumberSeed + threadIndex;
	uint2 seed2 = pcg2d(make_uint2(threadIndex) * (sppLane * 10 + spp * randomNumberSeed + 1));
	randomNumberSeed = seed2.x + seed2.y;

	glm::vec3 radiance = glm::vec3(0.0f, 0.0f, 0.0f);
	float RR = 0.8f;
	float pdf = 1.0f;
	glm::vec3 bsdf = glm::vec3(1.0f);
	bool hit = true;
	FzbTriangleAttribute hitTriangleAttribute;
	FzbTriangleAttribute lastHitTriangleAttribute;

	glm::vec2 texelXY = glm::vec2(resultIndex % groupCameraInfo.screenWidth, resultIndex / groupCameraInfo.screenWidth);
	glm::vec4 screenPos = glm::vec4(((texelXY + Hammersley(sppLane, spp)) / glm::vec2(groupCameraInfo.screenWidth, groupCameraInfo.screenHeight)) * 2.0f - 1.0f, 0.0f, 1.0f);	//vulkan中近平面ndcDepth在[0,1]
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
	radiance += getRadiance(hitTriangleAttribute, ray, &lightSet, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, randomNumberSeed, groupSetting.useSphericalRectangleSample);

	uint32_t maxBonceDepth = 2;
#pragma nounroll
	while (maxBonceDepth > 0) {
		randomNumberSeed += maxBonceDepth;
		float randomNumber = rand(randomNumberSeed);
		if (randomNumber > RR) break;
		pdf *= RR;

		lastDirection = -ray.direction;
		//generateRay(hitTriangleAttribute, pdf, ray, randomNumberSeed);
		generateRay_SVOPathGuiding(hitTriangleAttribute, pdf, ray, randomNumberSeed, groupSetting);

		lastHitTriangleAttribute = hitTriangleAttribute;
		hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, ray, hitTriangleAttribute);
		if (!hit) break;
		
		bsdf *= getBSDF(lastHitTriangleAttribute, ray.direction, lastDirection, ray) * glm::abs(glm::dot(ray.direction, lastHitTriangleAttribute.normal));
		radiance += getRadiance(hitTriangleAttribute, ray, &lightSet, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, randomNumberSeed, groupSetting.useSphericalRectangleSample) * bsdf / pdf;
		--maxBonceDepth;
	}

	radiance /= spp;
	radiance /= groupFrameCount;
	radiance = glm::min(radiance, glm::vec3(1.0f));
	if (useExternSharedMemory && threadIdx.x < groupSppIndex * spp) {
		//这是这里如果spp为32的整数倍，则可以先在warp中处理
		atomicAdd(&groupResultRadiance[groupSppIndex].x, radiance.x);
		atomicAdd(&groupResultRadiance[groupSppIndex].y, radiance.y);
		atomicAdd(&groupResultRadiance[groupSppIndex].z, radiance.z);
	}
	else {
		atomicAdd(&resultBuffer[resultIndex].x, radiance.x);
		atomicAdd(&resultBuffer[resultIndex].y, radiance.y);
		atomicAdd(&resultBuffer[resultIndex].z, radiance.z);
		return;
	}
	__syncthreads();
	if (useExternSharedMemory) {
		if (sppLane == 0) {
			atomicAdd(&resultBuffer[resultIndex].x, groupResultRadiance[groupSppIndex].x);
			atomicAdd(&resultBuffer[resultIndex].y, groupResultRadiance[groupSppIndex].y);
			atomicAdd(&resultBuffer[resultIndex].z, groupResultRadiance[groupSppIndex].z);
		}
	}
}
//-----------------------------------------------------------------------------------------------------------------
FzbSVOPathGuidingCuda::FzbSVOPathGuidingCuda() {};
FzbSVOPathGuidingCuda::FzbSVOPathGuidingCuda(std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager, FzbSVOPathGuidingCudaSetting setting, std::shared_ptr<FzbSVOCuda_PG> SVOCuda_PG) {
	if (getCudaDeviceForVulkanPhysicalDevice(FzbRenderer::globalData.physicalDevice) == cudaInvalidDeviceId) {
		throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");
	}
	this->sourceManager = sourceManager;
	this->setting = setting;
	this->SVOCuda_PG = SVOCuda_PG;

	CHECK(cudaMemcpyToSymbol(svoPathGuidingSetting, &setting, sizeof(FzbSVOPathGuidingCudaSetting)));

	//设置cuda配置，更多的使用L1 cache
	cudaFuncSetAttribute(svoPathGuiding_cuda<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (setting.spp >= sharedMemorySPP ? blockSize / setting.spp : 0) * sizeof(float3));	//3070 128KB，则L1 96KB，sharedMemory 32KB
	cudaFuncSetAttribute(svoPathGuiding_cuda<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
}
void FzbSVOPathGuidingCuda::SVOPathGuiding(HANDLE startSemaphoreHandle) {
	this->sourceManager->createRuntimeSource();

	VkExtent2D resolution = FzbRenderer::globalData.getResolution();
	uint32_t texelCount = resolution.width * resolution.height;
	uint32_t rayCount = texelCount * setting.spp;
	uint32_t gridSize = (rayCount + blockSize - 1) / blockSize;

	uint32_t sharedMemorySize = (setting.spp >= sharedMemorySPP ? blockSize / setting.spp : 0) * sizeof(float3);

	if (startSemaphoreHandle) {
		if (!this->extStartSemphores.count(startSemaphoreHandle))
			this->extStartSemphores.insert({ startSemaphoreHandle, importVulkanSemaphoreObjectFromNTHandle(startSemaphoreHandle) });
		CHECK(waitExternalSemaphore(this->extStartSemphores[startSemaphoreHandle], sourceManager->stream));
	}

	//CHECK(cudaDeviceSynchronize());
	//double start = cpuSecond();

	if (setting.spp >= sharedMemorySPP) 
		svoPathGuiding_cuda<true> << <gridSize, blockSize, sharedMemorySize, sourceManager->stream >> >
		(
			sourceManager->resultBuffer, sourceManager->vertices, sourceManager->materialTextures,
			sourceManager->bvhNodeArray, sourceManager->bvhTriangleInfoArray, rayCount
		);
	else
		svoPathGuiding_cuda<false> << <gridSize, blockSize, sharedMemorySize, sourceManager->stream >> > 
		(
			sourceManager->resultBuffer, sourceManager->vertices, sourceManager->materialTextures,
			sourceManager->bvhNodeArray, sourceManager->bvhTriangleInfoArray, rayCount
		);
	//checkKernelFunction();
	//CHECK(cudaDeviceSynchronize());
	//this->meanRunTime += cpuSecond() - start;
	//++runCount;
	//if (runCount == 20) {
	//	std::cout << meanRunTime / runCount << std::endl;
	//	runCount = 0;
	//	meanRunTime = 0.0;
	//}

	signalExternalSemaphore(sourceManager->extRayTracingFinishedSemaphore, sourceManager->stream);
}
void FzbSVOPathGuidingCuda::clean() {
	for (auto& pair : this->extStartSemphores) CHECK(cudaDestroyExternalSemaphore(pair.second));
}
