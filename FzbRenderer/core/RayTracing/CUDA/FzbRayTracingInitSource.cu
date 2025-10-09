#include "./FzbRayTracingInitSource.cuh"
#include "../../common/FzbRenderer.h"

__constant__ FzbPathTracingCameraInfo systemCameraInfo;
__constant__ FzbRayTracingMaterialUniformObject materialInfoArray[maxMaterialCount];
__constant__ bool useCudaRandom;
__constant__ curandState* systemRandomNumberStates;
__constant__ uint32_t systemRandomNumberSeed;

__constant__ uint32_t systemPointLightCount;
__constant__ FzbRayTracingPointLight systemPointLightInfoArray[maxPointLightCount];
__constant__ uint32_t systemAreaLightCount;
__constant__ FzbRayTracingAreaLight systemAreaLightInfoArray[maxAreaLightCount];

FzbRayTracingSourceManager_Cuda::FzbRayTracingSourceManager_Cuda() {};
void FzbRayTracingSourceManager_Cuda::initRayTracingSource(FzbRayTracingCudaSourceSet& sourceSet) {
	if (getCudaDeviceForVulkanPhysicalDevice(FzbRenderer::globalData.physicalDevice) == cudaInvalidDeviceId) {
		throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");
	}

	//结果输出的buffer
	resultBufferExtMem = importVulkanMemoryObjectFromNTHandle(sourceSet.rayTracingResultBuffer.handle, sourceSet.rayTracingResultBuffer.size, false);
	resultBuffer = (float4*)mapBufferOntoExternalMemory(resultBufferExtMem, 0, sourceSet.rayTracingResultBuffer.size);
	extRayTracingFinishedSemaphore = importVulkanSemaphoreObjectFromNTHandle(sourceSet.rayTracingFinishedSemphore.handle);

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
		fromVulkanImageToCudaTexture(FzbRenderer::globalData.physicalDevice, texture, texture.handle, textureSize, false, textureExtMems[textureIndex], textureMipmap[textureIndex], textureObjects[textureIndex], true);
		++textureIndex;
	}
	CHECK(cudaMalloc((void**)&materialTextures, textureCount * sizeof(cudaTextureObject_t)));
	cudaMemcpy(materialTextures, textureObjects.data(), textureCount * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

	//将materialInfoArrayBuffer放入uniform中
	if (sourceSet.sceneMaterialInfoArray.size() > maxMaterialCount) throw std::runtime_error("material数量超过maxMaterialCount:" + std::to_string(maxMaterialCount) + "，需要扩大");
	CHECK(cudaMemcpyToSymbol(materialInfoArray, sourceSet.sceneMaterialInfoArray.data(), sourceSet.sceneMaterialInfoArray.size() * sizeof(FzbRayTracingMaterialUniformObject)));

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

	//CHECK(cudaMemcpyToSymbol(useCudaRandom, &setting.useCudaRandom, sizeof(bool)));
	//if (setting.useCudaRandom) {
	//	VkExtent2D resolution = FzbRenderer::globalData.getResolution();
	//	uint32_t texelCount = resolution.width * resolution.height;
	//	uint32_t rayCount = texelCount * setting.spp;
	//	uint32_t gridSize = (rayCount + blockSize - 1) / blockSize;
	//	cudaMalloc(&systemRandomNumberStates_device, rayCount * sizeof(curandState));
	//	init_curand_states << <gridSize, blockSize >> > (systemRandomNumberStates_device, time(0), rayCount);
	//	CHECK(cudaMemcpyToSymbol(systemRandomNumberStates, &systemRandomNumberStates_device, sizeof(curandState*)));
	//}

	//创建stream
	CHECK(cudaStreamCreate(&stream));
}
void FzbRayTracingSourceManager_Cuda::createRuntimeSource() {
	//为cameraInfo赋值
	FzbPathTracingCameraInfo cameraInfo_host;
	FzbCamera* camera = FzbRenderer::globalData.camera;
	cameraInfo_host.cameraWorldPos = camera->position;
	cameraInfo_host.inversePVMatrix = glm::inverse(camera->GetProjMatrix() * camera->GetViewMatrix());
	VkExtent2D resolution = FzbRenderer::globalData.getResolution();
	cameraInfo_host.screenWidth = resolution.width;
	cameraInfo_host.screenHeight = resolution.height;
	CHECK(cudaMemcpyToSymbol(systemCameraInfo, &cameraInfo_host, sizeof(FzbPathTracingCameraInfo)));
}
void FzbRayTracingSourceManager_Cuda::clean() {
	CHECK(cudaFree(resultBuffer));
	CHECK(cudaDestroyExternalMemory(resultBufferExtMem));

	CHECK(cudaDestroyExternalSemaphore(extRayTracingFinishedSemaphore));

	CHECK(cudaFree(vertices));
	CHECK(cudaDestroyExternalMemory(vertexExtMem));

	for (int i = 0; i < textureExtMems.size(); ++i) {
		CHECK(cudaDestroyTextureObject(textureObjects[i]));
		CHECK(cudaFreeMipmappedArray(textureMipmap[i]));
		CHECK(cudaDestroyExternalMemory(textureExtMems[i]));
	}
	CHECK(cudaFree(materialTextures));
	CHECK(cudaStreamDestroy(stream));
}