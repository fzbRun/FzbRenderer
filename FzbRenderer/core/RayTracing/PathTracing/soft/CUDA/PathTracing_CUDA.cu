#include "./PathTracing_CUDA.cuh"

FzbPathTracingCuda::FzbPathTracingCuda() {};
FzbPathTracingCuda::FzbPathTracingCuda(VkPhysicalDevice vkPhysicalDevice, FzbScene* scene, FzbImage pathTracingResultMap, HANDLE pathTracingFinishedSemphoreHandle, std::vector<HANDLE> startSemaphoreHandles) {
	uint32_t resultMapSize = pathTracingResultMap.width * pathTracingResultMap.height * sizeof(uint32_t);		//SRGB和uint32_t大小相同
	fromVulkanImageToCudaSurface(vkPhysicalDevice, pathTracingResultMap, pathTracingResultMap.handle, resultMapSize, false, resultMapExtMem, resultMapMipmap, resultMapObject);

	extPathTracingFinishedSemaphore = importVulkanSemaphoreObjectFromNTHandle(pathTracingFinishedSemphoreHandle);
	startSemaphoreNum = startSemaphoreHandles.size();
	extStartSemaphores.resize(startSemaphoreNum);
	for (int i = 0; i < startSemaphoreNum; ++i) extStartSemaphores[i] = importVulkanSemaphoreObjectFromNTHandle(startSemaphoreHandles[i]);

	vertexExtMem = importVulkanMemoryObjectFromNTHandle(scene->vertexBuffer.handle, scene->vertexBuffer.size, false);
	vertices = (float*)mapBufferOntoExternalMemory(vertexExtMem, 0, scene->vertexBuffer.size);

	for (auto& sceneImage : scene->sceneImages) {
		FzbImage& texture = sceneImage.second;
		uint32_t textureSize = texture.width * texture.height * sizeof(uint32_t);

	}
	
	CHECK(cudaStreamCreate(&stream));
}
void FzbPathTracingCuda::pathTracing(){
	for (int i = 0; i < startSemaphoreNum; ++i) waitExternalSemaphore(extStartSemaphores[i], stream);
}

void FzbPathTracingCuda::clean() {
	CHECK(cudaDestroySurfaceObject(resultMapObject));
	CHECK(cudaFreeMipmappedArray(resultMapMipmap));
	CHECK(cudaDestroyExternalMemory(resultMapExtMem));
	CHECK(cudaDestroyExternalSemaphore(extPathTracingFinishedSemaphore));
	for (int i = 0; i < startSemaphoreNum; ++i) CHECK(cudaDestroyExternalSemaphore(extStartSemaphores[i]));
	CHECK(cudaDestroyExternalMemory(vertexExtMem));
	CHECK(cudaFree(vertices));
	CHECK(cudaStreamDestroy(stream));
}