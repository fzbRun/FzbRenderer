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
template<bool useExternSharedMemory>
__global__ void pathTracing_cuda(float4* resultBuffer, const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, const uint32_t rayCount) {
	extern __shared__ float3 groupResultRadiance[];	//���spp����sharedMemorySPP����ʹ�ù����ڴ棬 ����ΪblockDim / spp, ���6KB
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

	volatile const uint32_t& spp = spp_group;		//�Ĵ��������ã�������localMmeory���ǻ�����ֱ����sharedMemory��
	uint32_t resultIndex = threadIndex / spp;	//���ڵڼ���spp����bufferIndex
	uint32_t groupSppIndex = threadIdx.x / spp;		//���ڵڼ���spp
	uint32_t sppLane = threadIndex % spp;	//������&����Ϊspp���ܲ���2���ݴ�
	if (useExternSharedMemory) {
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

	glm::vec2 texelXY = glm::vec2(resultIndex % groupCameraInfo.screenWidth, resultIndex / groupCameraInfo.screenWidth);
	glm::vec4 screenPos = glm::vec4(((texelXY + Hammersley(sppLane, spp)) / glm::vec2(groupCameraInfo.screenWidth, groupCameraInfo.screenHeight)) * 2.0f - 1.0f, 0.0f, 1.0f);	//vulkan�н�ƽ��ndcDepth��[0,1]
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
	radiance += getRadiance(hitTriangleAttribute, ray, &lightSet, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, randomNumberSeed);
	
	uint32_t maxBonceDepth = 1;
	#pragma nounroll
	while (maxBonceDepth > 0) {
		float randomNumber = rand(randomNumberSeed);
		if (randomNumber > RR) break;
		pdf *= randomNumber;
		
		lastDirection = -ray.direction;
		generateRay(hitTriangleAttribute, pdf, ray, randomNumberSeed);

		lastHitTriangleAttribute = hitTriangleAttribute;
		hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, ray, hitTriangleAttribute);
		if (!hit) break;
		ray.hitPos = ray.direction * ray.depth + ray.startPos;

		bsdf *= getBSDF(lastHitTriangleAttribute, ray.direction, lastDirection, ray) * glm::abs(glm::dot(ray.direction, lastHitTriangleAttribute.normal));
		//radiance = hitTriangleAttribute.albedo;
		//break;
		radiance += getRadiance(hitTriangleAttribute, ray, &lightSet, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, randomNumberSeed) * bsdf / pdf;
		--maxBonceDepth;
	}

	radiance /= spp;
	if (useExternSharedMemory && threadIdx.x < groupSppIndex * spp) {
		//�����������sppΪ32�������������������warp�д���
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

FzbPathTracingCuda::FzbPathTracingCuda() {};
FzbPathTracingCuda::FzbPathTracingCuda(FzbPathTracingCudaSourceSet& sourceSet) {
	if (getCudaDeviceForVulkanPhysicalDevice(FzbRenderer::globalData.physicalDevice) == cudaInvalidDeviceId) {
		throw std::runtime_error("CUDA��Vulkan�õĲ���ͬһ��GPU������");
	}
	//���pathTracing��setting
	this->setting = sourceSet.setting;
	CHECK(cudaMemcpyToSymbol(pathTracingSetting, &sourceSet.setting, sizeof(FzbPathTracingSetting)));

	//��������buffer
	resultBufferExtMem = importVulkanMemoryObjectFromNTHandle(sourceSet.pathTracingResultBuffer.handle, sourceSet.pathTracingResultBuffer.size, false);
	resultBuffer = (float4*)mapBufferOntoExternalMemory(resultBufferExtMem, 0, sourceSet.pathTracingResultBuffer.size);

	//��ȡpathTracing����������
	extPathTracingFinishedSemaphore = importVulkanSemaphoreObjectFromNTHandle(sourceSet.pathTracingFinishedSemphore.handle);

	//��ȡ������������
	vertexExtMem = importVulkanMemoryObjectFromNTHandle(sourceSet.sceneVertices.handle, sourceSet.sceneVertices.size, false);
	vertices = (float*)mapBufferOntoExternalMemory(vertexExtMem, 0, sourceSet.sceneVertices.size);

	//��ȡ����Material��Ϣ
	uint32_t textureCount = sourceSet.sceneTextures.size();
	this->textureExtMems.resize(textureCount);
	this->textureMipmap.resize(textureCount);
	this->textureObjects.resize(textureCount);
	uint32_t textureIndex = 0;
	for (auto& sceneTexture : sourceSet.sceneTextures) {
		FzbImage& texture = sceneTexture;
		uint32_t textureSize = texture.width * texture.height * texture.depth * sizeof(uint32_t);
		fromVulkanImageToCudaTexture(FzbRenderer::globalData.physicalDevice, texture, texture.handle, textureSize, false, textureExtMems[textureIndex], textureMipmap[textureIndex], textureObjects[textureIndex], true);
		++textureIndex;
	}
	CHECK(cudaMalloc((void**)&materialTextures, textureCount * sizeof(cudaTextureObject_t)));
	cudaMemcpy(materialTextures, textureObjects.data(), textureCount * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

	//��materialInfoArrayBuffer����uniform��
	if (sourceSet.sceneMaterialInfoArray.size() > maxMaterialCount) throw std::runtime_error("material��������maxMaterialCount:" + std::to_string(maxMaterialCount) + "����Ҫ����");
	CHECK(cudaMemcpyToSymbol(materialInfoArray, sourceSet.sceneMaterialInfoArray.data(), sourceSet.sceneMaterialInfoArray.size() * sizeof(FzbPathTracingMaterialUniformObject)));
	
	//����stream
	CHECK(cudaStreamCreate(&stream));

	//�����Ϣ
	//this->extBvhFinishedSemaphore = importVulkanSemaphoreObjectFromNTHandle(sourceSet.bvhSemaphoreHandle);
	this->bvhNodeArray = sourceSet.bvhNodeArray;
	this->bvhTriangleInfoArray = sourceSet.bvhTriangleInfoArray;

	//������Դ��Ϣ
	CHECK(cudaMemcpyToSymbol(systemPointLightCount, &sourceSet.pointLightCount, sizeof(uint32_t)));
	if (sourceSet.pointLightInfoArray.size() > maxPointLightCount) throw std::runtime_error("���Դ��������maxPointLightCount" + std::to_string(maxPointLightCount) + "����Ҫ����");
	CHECK(cudaMemcpyToSymbol(systemPointLightInfoArray, sourceSet.pointLightInfoArray.data(), sourceSet.pointLightInfoArray.size() * sizeof(FzbRayTracingPointLight)));
	CHECK(cudaMemcpyToSymbol(systemAreaLightCount, &sourceSet.areaLightCount, sizeof(uint32_t)));
	if (sourceSet.areaLightInfoArray.size() > maxAreaLightCount) throw std::runtime_error("���Դ��������maxAreaLightCount" + std::to_string(maxAreaLightCount) + "����Ҫ����");
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

	//����cuda���ã������ʹ��L1 cache
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaFuncSetAttribute(pathTracing_cuda<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (setting.spp >= sharedMemorySPP ? blockSize / setting.spp : 0) * sizeof(float3));	//3070 128KB����L1 96KB��sharedMemory 32KB
	cudaFuncSetAttribute(pathTracing_cuda<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
	//cudaFuncSetAttribute(pathTracing_cuda, cudaFuncAttributePreferredSharedMemoryCarveout, 10);	//3070 128KB����L1 96KB��sharedMemory 32KB
	//cudaFuncAttributes attr;
	//cudaFuncGetAttributes(&attr, pathTracing_cuda);
	//printf("�����ڴ��С: %d bytes\n", attr.sharedSizeBytes);
	//printf("����߳���: %d bytes\n", attr.maxThreadsPerBlock);
	//printf("�Ƿ�ʹ��L1: %d\n", attr.cacheModeCA);
}
void FzbPathTracingCuda::pathTracing(HANDLE startSemaphoreHandle) {
	//ΪcameraInfo��ֵ
	FzbPathTracingCameraInfo cameraInfo_host;
	FzbCamera* camera = FzbRenderer::globalData.camera;
	cameraInfo_host.cameraWorldPos = camera->position;
	cameraInfo_host.inversePVMatrix = glm::inverse(camera->GetProjMatrix() * camera->GetViewMatrix());
	VkExtent2D resolution = FzbRenderer::globalData.getResolution();
	cameraInfo_host.screenWidth = resolution.width;
	cameraInfo_host.screenHeight = resolution.height;
	CHECK(cudaMemcpyToSymbol(systemCameraInfo, &cameraInfo_host, sizeof(FzbPathTracingCameraInfo)));

	if (!setting.useCudaRandom) CHECK(cudaMemcpyToSymbol(systemRandomNumberSeed, &FzbRenderer::globalData.randomNumber, sizeof(uint32_t)));

	uint32_t texelCount = resolution.width * resolution.height;
	uint32_t rayCount = texelCount * setting.spp;
	uint32_t gridSize = (rayCount + blockSize - 1) / blockSize;

	uint32_t sharedMemorySize = (setting.spp >= sharedMemorySPP ? blockSize / setting.spp : 0) * sizeof(float3);

	if (!this->extStartSemphores.count(startSemaphoreHandle)) this->extStartSemphores.insert({ startSemaphoreHandle, importVulkanSemaphoreObjectFromNTHandle(startSemaphoreHandle) });
	waitExternalSemaphore(this->extStartSemphores[startSemaphoreHandle], stream);

	//CHECK(cudaDeviceSynchronize());
	//double start = cpuSecond();
	if (setting.spp >= sharedMemorySPP) pathTracing_cuda<true> << <gridSize, blockSize, sharedMemorySize, stream >> > (resultBuffer, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, rayCount);
	else pathTracing_cuda<false> << <gridSize, blockSize, sharedMemorySize, stream >> > (resultBuffer, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, rayCount);
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

	if (systemRandomNumberStates_device) CHECK(cudaFree(systemRandomNumberStates_device));
}

