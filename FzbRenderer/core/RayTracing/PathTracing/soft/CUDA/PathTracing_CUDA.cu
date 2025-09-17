#include "./PathTracing_CUDA.cuh"
#include "../../../CUDA/FzbRayGenerate.cuh"
#include "../../../CUDA/FzbGetTriangleAttribute.cuh"
#include "../../../CUDA/FzbCollisionDetection.cuh"
#include "../../../CUDA/FzbGetIllumination.cuh"

//----------------------------------------------uniformBuffer--------------------------------------
__constant__ FzbPathTracingSetting pathTracingSetting;
extern __constant__ FzbPathTracingCameraInfo cameraInfo;
extern __constant__ FzbPathTracingMaterialUniformObject materialInfoArray[128];

extern __constant__ uint32_t frameIndex;

extern __constant__ uint32_t pointLightCount;
extern __constant__ FzbRayTracingPointLight pointLightInfoArray[16];
extern __constant__ uint32_t areaLightCount;
extern __constant__ FzbRayTracingAreaLight areaLightInfoArray[8];
//-------------------------------------------------------------------------------------------------

FzbPathTracingCuda::FzbPathTracingCuda() {};
FzbPathTracingCuda::FzbPathTracingCuda(VkPhysicalDevice vkPhysicalDevice, FzbMainScene* scene,
	FzbPathTracingSetting setting, FzbBuffer pathTracingResultBuffer, FzbSemaphore pathTracingFinishedSemphore,
	std::vector<FzbImage>& sceneTextures, std::vector<FzbPathTracingMaterialUniformObject> sceneMaterialInfoArray,
	BVHCuda* bvh) {
	if (getCudaDeviceForVulkanPhysicalDevice(vkPhysicalDevice) == cudaInvalidDeviceId) {
		throw std::runtime_error("CUDA与Vulkan用的不是同一个GPU！！！");
	}

	//获得pathTracing的setting
	this->setting = setting;
	CHECK(cudaMemcpyToSymbol(pathTracingSetting, &setting, sizeof(FzbPathTracingSetting)));

	//结果输出的buffer
	resultBufferExtMem = importVulkanMemoryObjectFromNTHandle(pathTracingResultBuffer.handle, pathTracingResultBuffer.size, false);
	resultBuffer = (float4*)mapBufferOntoExternalMemory(resultBufferExtMem, 0, pathTracingResultBuffer.size);

	//获取pathTracing结束描述符
	extPathTracingFinishedSemaphore = importVulkanSemaphoreObjectFromNTHandle(pathTracingFinishedSemphore.handle);
	//获取pathTracing开始描述符
	//extStartSemaphores.resize(startSemaphores.size());
	//for (int i = 0; i < extStartSemaphores.size(); ++i) extStartSemaphores[i] = importVulkanSemaphoreObjectFromNTHandle(startSemaphores[i].handle);
	//获取场景顶点数据
	vertexExtMem = importVulkanMemoryObjectFromNTHandle(scene->vertexBuffer.handle, scene->vertexBuffer.size, false);
	vertices = (float*)mapBufferOntoExternalMemory(vertexExtMem, 0, scene->vertexBuffer.size);
	//获取场景Material信息
	uint32_t textureCount = scene->sceneImages.size();
	this->textureExtMems.resize(textureCount);
	this->textureMipmap.resize(textureCount);
	this->textureObjects.resize(textureCount);
	uint32_t textureIndex = 0;
	for (auto& sceneImage : scene->sceneImages) {
		FzbImage& texture = sceneImage.second;
		uint32_t textureSize = texture.width * texture.height * texture.depth * sizeof(uint32_t);
		fromVulkanImageToCudaTexture(vkPhysicalDevice, texture, texture.handle, textureSize, false, textureExtMems[textureIndex], textureMipmap[textureIndex], textureObjects[textureIndex]);
		++textureIndex;
	}
	CHECK(cudaMalloc((void**)&materialTextures, textureCount * sizeof(cudaTextureObject_t)));
	cudaMemcpy(materialTextures, textureObjects.data(), textureCount * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

	//将materialInfoArrayBuffer放入uniform中
	if (sceneMaterialInfoArray.size() > 128) throw std::runtime_error("material数量超过16，需要扩大");
	CHECK(cudaMemcpyToSymbol(materialInfoArray, sceneMaterialInfoArray.data(), sceneMaterialInfoArray.size() * sizeof(FzbPathTracingMaterialUniformObject)));
	
	//创建stream
	CHECK(cudaStreamCreate(&stream));

	this->bvh = bvh;
}

__global__ void pathTracing_cuda(float4* resultBuffer, float* __restrict__ vertices, cudaTextureObject_t* __restrict__ materialTextures,
	FzbBvhNode* __restrict__ bvhNodeArray, FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray) {
	extern __shared__ float3 resultRadiance[];	//数量为blockDim / spp，如果spp大于32，则使用共享内存

	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t resultIndex = threadIdx.x / pathTracingSetting.spp;

	glm::vec3 radiance = glm::vec3(0.0f);
	float RR = 0.8f;
	float pdf = 1.0f;
	float bsdf = 1.0f;

	uint32_t texelX = threadIndex & (cameraInfo.screenWidth * pathTracingSetting.spp);
	uint32_t texelY = threadIndex / (cameraInfo.screenHeight * pathTracingSetting.spp);
	uint32_t sppIndex = threadIdx.x / pathTracingSetting.spp;
	uint32_t sppLane = threadIndex & (pathTracingSetting.spp - 1);
	FzbRay ray = generateFirstRay(glm::vec2(texelX, texelY), pathTracingSetting.spp, sppLane);
	FzbTriangleAttribute hitTriangleAttribute;

	//第一次碰撞
	bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, ray, hitTriangleAttribute);
	if (!hit) return;
	ray.hitPos = ray.direction * ray.depth + ray.startPos;
	radiance += NEE(hitTriangleAttribute, ray, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray);
	uint32_t randomNumberSeed = frameIndex + threadIndex;
	float randomNumber = rand(randomNumberSeed);
	if (randomNumber > RR) {

		
	}
	//后续bounce
	while (hit) {
		hit = false;
	}

	if (pathTracingSetting.spp > 32 && threadIdx.x < sppIndex * pathTracingSetting.spp) {
		atomicAddFloat(&resultRadiance[sppIndex].x, radiance.x);
		atomicAddFloat(&resultRadiance[sppIndex].y, radiance.y);
		atomicAddFloat(&resultRadiance[sppIndex].z, radiance.z);
	}
	else {
		atomicAddFloat(&resultBuffer[sppIndex].x, radiance.x);
		atomicAddFloat(&resultBuffer[sppIndex].y, radiance.y);
		atomicAddFloat(&resultBuffer[sppIndex].z, radiance.z);
		return;
	}
	__syncthreads();
	if (pathTracingSetting.spp > 32) {
		if (sppLane == 0) {
			atomicAddFloat(&resultBuffer[sppIndex].x, resultRadiance[sppIndex].x);
			atomicAddFloat(&resultBuffer[sppIndex].y, resultRadiance[sppIndex].y);
			atomicAddFloat(&resultBuffer[sppIndex].z, resultRadiance[sppIndex].z);
		}
	}
}
void FzbPathTracingCuda::pathTracing(VkSemaphore startSemaphore, uint32_t screenWidth, uint32_t screenHeight){
	importVulkanSemaphoreObjectFromNTHandle(startSemaphore.handle);
	waitExternalSemaphore(extVgmSemaphore, stream);

	uint32_t texelCount = screenWidth * screenHeight;
	uint32_t rayCount = texelCount * setting.spp;
	uint32_t blockSize = 256;
	uint32_t gridSize = (rayCount + blockSize - 1) / blockSize;
	pathTracing_cuda<<<gridSize, blockSize>>>(resultBuffer, vertices, materialTextures, bvh->bvhNodeArray, bvh->bvhTriangleInfoArray);
	signalExternalSemaphore(extPathTracingFinishedSemaphore, stream);
}

void FzbPathTracingCuda::clean() {
	CHECK(cudaDestroySurfaceObject(resultMapObject));
	CHECK(cudaFreeMipmappedArray(resultMapMipmap));
	CHECK(cudaDestroyExternalMemory(resultMapExtMem));
	CHECK(cudaDestroyExternalSemaphore(extPathTracingFinishedSemaphore));
	//for (int i = 0; i < extStartSemaphores.size(); ++i) CHECK(cudaDestroyExternalSemaphore(extStartSemaphores[i]));
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