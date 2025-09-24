#include "./PathTracing_CUDA.cuh"
#include "../../../../common/FzbRenderer.h"
#include "../../../CUDA/FzbRayGenerate.cuh"
#include "../../../CUDA/FzbGetTriangleAttribute.cuh"
#include "../../../CUDA/FzbCollisionDetection.cuh"
#include "../../../CUDA/FzbGetIllumination.cuh"

//----------------------------------------------uniformBuffer--------------------------------------
__constant__ FzbPathTracingSetting pathTracingSetting;
__constant__ FzbPathTracingCameraInfo systemCameraInfo;
__constant__ FzbPathTracingMaterialUniformObject materialInfoArray[maxMaterialCount];
__constant__ bool useCudaRandom;
__constant__ curandState* systemRandomNumberStates;
__constant__ uint32_t systemRandomNumberSeed;

__constant__ uint32_t systemPointLightCount;
__constant__ FzbRayTracingPointLight systemPointLightInfoArray[maxPointLightCount];
__constant__ uint32_t systemAreaLightCount;
__constant__ FzbRayTracingAreaLight systemAreaLightInfoArray[maxAreaLightCount];

const uint32_t blockSize = 128;
const uint32_t sharedMemorySPP = 2;
//-------------------------------------------------------------------------------------------------
__global__ void pathTracing_cuda(float4* resultBuffer, const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, const uint32_t rayCount);

FzbPathTracingCuda::FzbPathTracingCuda() {};
FzbPathTracingCuda::FzbPathTracingCuda(FzbPathTracingCudaSourceSet& sourceSet) {
	if (getCudaDeviceForVulkanPhysicalDevice(FzbRenderer::globalData.physicalDevice) == cudaInvalidDeviceId) {
		throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");
	}
	//获得pathTracing的setting
	this->setting = sourceSet.setting;
	CHECK(cudaMemcpyToSymbol(pathTracingSetting, &sourceSet.setting, sizeof(FzbPathTracingSetting)));

	//结果输出的buffer
	resultBufferExtMem = importVulkanMemoryObjectFromNTHandle(sourceSet.pathTracingResultBuffer.handle, sourceSet.pathTracingResultBuffer.size, false);
	resultBuffer = (float4*)mapBufferOntoExternalMemory(resultBufferExtMem, 0, sourceSet.pathTracingResultBuffer.size);

	//获取pathTracing结束描述符
	extPathTracingFinishedSemaphore = importVulkanSemaphoreObjectFromNTHandle(sourceSet.pathTracingFinishedSemphore.handle);

	//获取场景顶点数据
	vertexExtMem = importVulkanMemoryObjectFromNTHandle(sourceSet.sceneVertices.handle, sourceSet.sceneVertices.size, false);
	vertices = (float*)mapBufferOntoExternalMemory(vertexExtMem, 0, sourceSet.sceneVertices.size);

	//获取场景Material信息
	uint32_t textureCount = sourceSet.sceneTextures.size();
	this->textureExtMems.resize(textureCount);
	this->textureMipmap.resize(textureCount);
	this->textureObjects.resize(textureCount);
	uint32_t textureIndex = 0;
	for (auto& sceneTexture : sourceSet.sceneTextures) {
		FzbImage& texture = sceneTexture;
		uint32_t textureSize = texture.width * texture.height * texture.depth * sizeof(uint32_t);
		fromVulkanImageToCudaTexture(FzbRenderer::globalData.physicalDevice, texture, texture.handle, textureSize, false, textureExtMems[textureIndex], textureMipmap[textureIndex], textureObjects[textureIndex]);
		++textureIndex;
	}
	CHECK(cudaMalloc((void**)&materialTextures, textureCount * sizeof(cudaTextureObject_t)));
	cudaMemcpy(materialTextures, textureObjects.data(), textureCount * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

	//将materialInfoArrayBuffer放入uniform中
	if (sourceSet.sceneMaterialInfoArray.size() > maxMaterialCount) throw std::runtime_error("material数量超过maxMaterialCount:" + std::to_string(maxMaterialCount) + "，需要扩大");
	CHECK(cudaMemcpyToSymbol(materialInfoArray, sourceSet.sceneMaterialInfoArray.data(), sourceSet.sceneMaterialInfoArray.size() * sizeof(FzbPathTracingMaterialUniformObject)));
	
	//创建stream
	CHECK(cudaStreamCreate(&stream));

	//组件信息
	//this->extBvhFinishedSemaphore = importVulkanSemaphoreObjectFromNTHandle(sourceSet.bvhSemaphoreHandle);
	this->bvhNodeArray = sourceSet.bvhNodeArray;
	this->bvhTriangleInfoArray = sourceSet.bvhTriangleInfoArray;

	//创建光源信息
	CHECK(cudaMemcpyToSymbol(systemPointLightCount, &sourceSet.pointLightCount, sizeof(uint32_t)));
	if (sourceSet.pointLightInfoArray.size() > maxPointLightCount) throw std::runtime_error("点光源数量超过maxPointLightCount" + std::to_string(maxPointLightCount) + "，需要扩大");
	CHECK(cudaMemcpyToSymbol(systemPointLightInfoArray, sourceSet.pointLightInfoArray.data(), sourceSet.pointLightInfoArray.size() * sizeof(FzbRayTracingPointLight)));
	CHECK(cudaMemcpyToSymbol(systemAreaLightCount, &sourceSet.areaLightCount, sizeof(uint32_t)));
	if (sourceSet.areaLightInfoArray.size() > maxAreaLightCount) throw std::runtime_error("面光源数量超过maxAreaLightCount" + std::to_string(maxAreaLightCount) + "，需要扩大");
	CHECK(cudaMemcpyToSymbol(systemAreaLightInfoArray, sourceSet.areaLightInfoArray.data(), sourceSet.areaLightInfoArray.size() * sizeof(FzbRayTracingAreaLight)));

	CHECK(cudaMemcpyToSymbol(useCudaRandom, &setting.useCudaRandom, sizeof(bool)));
	if (setting.useCudaRandom) {
		VkExtent2D resolution = FzbRenderer::globalData.getResolution();
		uint32_t texelCount = resolution.width * resolution.height;
		uint32_t rayCount = texelCount * setting.spp;
		uint32_t gridSize = (rayCount + blockSize - 1) / blockSize;
		cudaMalloc(&systemRandomNumberStates_device, rayCount * sizeof(curandState));
		init_curand_states << <gridSize, blockSize >> > (systemRandomNumberStates_device, time(0), rayCount);
		CHECK(cudaMemcpyToSymbol(systemRandomNumberStates, &systemRandomNumberStates_device, sizeof(curandState*)));
	}

	//设置cuda配置，更多的使用L1 cache
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	//cudaFuncSetAttribute(pathTracing_cuda, cudaFuncAttributeMaxDynamicSharedMemorySize, 25);	//3070 128KB，则L1 96KB，sharedMemory 32KB
}

__global__ void pathTracing_cuda(float4* resultBuffer, const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, const uint32_t rayCount) {
	extern __shared__ float3 groupResultRadiance[];	//如果spp大于sharedMemorySPP，则使用共享内存， 数量为blockDim / spp, 最大6KB
	__shared__ uint32_t spp_group;
	__shared__ FzbPathTracingCameraInfo groupCameraInfo;				//216B
	__shared__ uint32_t groupRandomNumberSeed;			
	__shared__ FzbRayTracingPointLight groupPointLightInfoArray[maxPointLightCount];	//512B
	__shared__ FzbRayTracingAreaLight grouprAreaLightInfoArray[maxAreaLightCount];		//692B
	__shared__ FzbRayTracingLightSet lightSet;

	volatile const uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= rayCount) return;

	if (threadIdx.x < systemPointLightCount) groupPointLightInfoArray[threadIdx.x] = systemPointLightInfoArray[threadIdx.x];
	if (threadIdx.x < systemAreaLightCount) grouprAreaLightInfoArray[threadIdx.x] = systemAreaLightInfoArray[threadIdx.x];
	if (threadIdx.x == 0) {
		spp_group = pathTracingSetting.spp;
		groupCameraInfo = systemCameraInfo;
		groupRandomNumberSeed = systemRandomNumberSeed;
		lightSet.pointLightCount = systemPointLightCount;
		lightSet.areaLightCount = systemAreaLightCount;
		lightSet.pointLightInfoArray = groupPointLightInfoArray;
		lightSet.areaLightInfoArray = grouprAreaLightInfoArray;
	}
	__syncthreads();

	volatile const uint32_t& spp = spp_group;		//寄存器不够用，挤到了localMmeory，那还不如直接用sharedMemory呢
	uint32_t resultIndex = threadIndex / spp;	//属于第几个spp，即bufferIndex
	uint32_t groupSppIndex = threadIdx.x / spp;		//组内第几个spp
	uint32_t sppLane = threadIndex % spp;	//不能用&，因为spp可能不是2的幂次
	if (spp >= sharedMemorySPP) {
		if (threadIdx.x < blockDim.x / spp) groupResultRadiance[threadIdx.x] = make_float3(0.0f);
	}
	//if (threadIndex < systemCameraInfo.screenWidth * systemCameraInfo.screenHeight * spp) resultBuffer[threadIndex] = make_float4(0.0f);
	if (sppLane == 0) resultBuffer[resultIndex] = make_float4(0.0f);
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

	uint32_t texelX = resultIndex % (groupCameraInfo.screenWidth);
	uint32_t texelY = resultIndex / (groupCameraInfo.screenWidth);
	FzbRay ray = generateFirstRay(&groupCameraInfo, glm::vec2(texelX, texelY), spp, sppLane);
	FzbRay lastRay;
	
	hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, ray, hitTriangleAttribute);
	if (!hit) return;
	ray.hitPos = ray.direction * ray.depth + ray.startPos;
	radiance += getRadiance(hitTriangleAttribute, ray, &lightSet, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, randomNumberSeed);
	
	uint32_t maxBonceDepth = 3;
	//#pragma nounroll
	while (maxBonceDepth > 0) {
		float randomNumber = getRandomNumber(randomNumberSeed);
		if (randomNumber > RR) break;
		pdf *= randomNumber;
		lastHitTriangleAttribute = hitTriangleAttribute;
		
		lastRay = ray;
		generateRay(0, pdf, ray, randomNumberSeed);
		hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, ray, hitTriangleAttribute);
		if (!hit) break;
		bsdf *= getBSDF(lastHitTriangleAttribute, ray.direction, -lastRay.direction) * glm::clamp(glm::dot(ray.direction, lastHitTriangleAttribute.normal), 0.0f, 1.0f);
		ray.hitPos = ray.direction * ray.depth + ray.startPos;
		radiance += getRadiance(hitTriangleAttribute, ray, &lightSet, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, randomNumberSeed) * bsdf / pdf;
		--maxBonceDepth;
	}

	radiance /= spp;
	if (spp >= sharedMemorySPP && threadIdx.x < groupSppIndex * spp) {
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
	if (spp >= sharedMemorySPP) {
		if (sppLane == 0) {
			atomicAdd(&resultBuffer[resultIndex].x, groupResultRadiance[groupSppIndex].x);
			atomicAdd(&resultBuffer[resultIndex].y, groupResultRadiance[groupSppIndex].y);
			atomicAdd(&resultBuffer[resultIndex].z, groupResultRadiance[groupSppIndex].z);
		}
	}
}
void FzbPathTracingCuda::pathTracing(HANDLE startSemaphoreHandle){
	//为cameraInfo赋值
	FzbPathTracingCameraInfo cameraInfo_host;
	FzbCamera* camera = FzbRenderer::globalData.camera;
	cameraInfo_host.cameraWorldPos = camera->position;
	cameraInfo_host.inversePVMatrix = glm::inverse(camera->GetProjMatrix() * camera->GetViewMatrix());
	VkExtent2D resolution = FzbRenderer::globalData.getResolution();
	cameraInfo_host.screenWidth = resolution.width;
	cameraInfo_host.screenHeight = resolution.height;
	CHECK(cudaMemcpyToSymbol(systemCameraInfo, &cameraInfo_host, sizeof(FzbPathTracingCameraInfo)));

	if(!setting.useCudaRandom) CHECK(cudaMemcpyToSymbol(systemRandomNumberSeed, &FzbRenderer::globalData.randomNumber, sizeof(uint32_t)));

	uint32_t texelCount = resolution.width * resolution.height;
	uint32_t rayCount = texelCount * setting.spp;
	uint32_t gridSize = (rayCount + blockSize - 1) / blockSize;

	uint32_t sharedMemorySize = (setting.spp >= sharedMemorySPP ? blockSize / setting.spp : 0) * sizeof(float3);

	if (!this->extStartSemphores.count(startSemaphoreHandle)) this->extStartSemphores.insert({ startSemaphoreHandle, importVulkanSemaphoreObjectFromNTHandle(startSemaphoreHandle) });
	waitExternalSemaphore(this->extStartSemphores[startSemaphoreHandle], stream);

	//CHECK(cudaDeviceSynchronize());
	//double start = cpuSecond();
	pathTracing_cuda<<<gridSize, blockSize, sharedMemorySize, stream>>>(resultBuffer, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, rayCount);
	//CHECK(cudaDeviceSynchronize());
	//this->meanRunTime += cpuSecond() - start;
	//++runCount;
	//if (runCount == 20) {
	//	std::cout << meanRunTime / runCount << std::endl;
	//	runCount = 0;
	//	meanRunTime = 0.0;
	//}
	
	signalExternalSemaphore(extPathTracingFinishedSemaphore, stream);
}

void FzbPathTracingCuda::clean() {
	CHECK(cudaFree(resultBuffer));
	CHECK(cudaDestroyExternalMemory(resultBufferExtMem));

	for (auto& pair : this->extStartSemphores) CHECK(cudaDestroyExternalSemaphore(pair.second));
	CHECK(cudaDestroyExternalSemaphore(extPathTracingFinishedSemaphore));

	CHECK(cudaFree(vertices));
	CHECK(cudaDestroyExternalMemory(vertexExtMem));

	for (int i = 0; i < textureExtMems.size(); ++i) {
		CHECK(cudaDestroyTextureObject(textureObjects[i]));
		CHECK(cudaFreeMipmappedArray(textureMipmap[i]));
		CHECK(cudaDestroyExternalMemory(textureExtMems[i]));
	}
	CHECK(cudaFree(materialTextures));
	CHECK(cudaStreamDestroy(stream));

	if(systemRandomNumberStates_device) CHECK(cudaFree(systemRandomNumberStates_device));
}