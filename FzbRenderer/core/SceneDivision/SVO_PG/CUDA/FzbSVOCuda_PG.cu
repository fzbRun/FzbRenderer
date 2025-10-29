#include "FzbSVOCuda_PG.cuh"
#include "../../../common/FzbRenderer.h"

//----------------------------------------------uniformBuffer--------------------------------------
__constant__ FzbVGBUniformData systemVGBUniformData;
__constant__ FzbSVOUnformData systemSVOUniformData;
//-------------------------------------------------------------------------------------------------
FzbSVOCuda_PG::FzbSVOCuda_PG() {};

FzbSVOCuda_PG::FzbSVOCuda_PG(std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager, FzbSVOSetting_PG setting, 
	FzbVGBUniformData VGBUniformData, FzbBuffer VGB, HANDLE SVOFinishedSemaphore_PG, FzbSVOUnformData SVOUniformData) {
	this->sourceManager = sourceManager;
	this->stream = sourceManager->stream;
	this->setting = setting;

	this->VGBUniformData = VGBUniformData;
	CHECK(cudaMemcpyToSymbol(systemVGBUniformData, &VGBUniformData, sizeof(FzbVGBUniformData)));

	this->SVOUniformData = SVOUniformData;
	CHECK(cudaMemcpyToSymbol(systemSVOUniformData, &SVOUniformData, sizeof(FzbSVOUnformData)));

	this->VGBExtMem = importVulkanMemoryObjectFromNTHandle(VGB.handle, VGB.size, false);
	this->VGB = (FzbVoxelData_PG*)mapBufferOntoExternalMemory(VGBExtMem, 0, VGB.size);

	//创建各级SVO数组，从第二级开始
	uint32_t vgmSize = 1;
	while (vgmSize <= setting.voxelNum) {
		SVONodes_maxDepth++;
		vgmSize <<= 1;
	}

	initLightInjectSource();
	initCreateOctreeNodesSource();
	initCreateSVONodesSource();
	initGetSVONodesWeightSource();

	this->extSvoSemaphore_PG = importVulkanSemaphoreObjectFromNTHandle(SVOFinishedSemaphore_PG);
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
//--------------------------------------------------------------------创造SVO_PG-------------------------------------------------------------------------
void FzbSVOCuda_PG::createSVOCuda_PG(HANDLE VGBFinishedSemaphore) {
	if (setting.voxelNum > 128) {
		std::cout << "voxelNum超过128，就需要两次压缩，目前无法实现，SVO未压缩" << std::endl;
		return;
	}
	waitExternalSemaphore(importVulkanSemaphoreObjectFromNTHandle(VGBFinishedSemaphore), stream);
	lightInject();
	createOctreeNodes();
	createSVONodes();
	getSVONodesWeight();
}

//----------------------------------------------------------------------------------------------------------------------------------------------------
void FzbSVOCuda_PG::clean() {
	CHECK(cudaDestroyExternalMemory(VGBExtMem));
	CHECK(cudaFree(VGB));
	for (int i = 0; i < this->OctreeNodes_multiLayer.size(); ++i) if (this->OctreeNodes_multiLayer[i]) CHECK(cudaFree(OctreeNodes_multiLayer[i]));

	for (int i = 0; i < this->SVONodeThreadBlockInfos.size(); ++i) if(SVONodeThreadBlockInfos[i]) CHECK(cudaFree(this->SVONodeThreadBlockInfos[i]));
	for (int i = 0; i < this->SVODivisibleNodeTempInfos.size(); ++i) if(SVODivisibleNodeTempInfos[i]) CHECK(cudaFree(this->SVODivisibleNodeTempInfos[i]));
	
	if (SVOLayerInfos) CHECK(cudaFree(SVOLayerInfos));
	if (SVOIndivisibleNodeInfos) CHECK(cudaFree(SVOIndivisibleNodeInfos));
	for (int i = 0; i < this->SVONodes_multiLayer.size(); ++i) if (SVONodes_multiLayer[i]) CHECK(cudaFree(this->SVONodes_multiLayer[i]));

	if (this->SVONodes_multiLayer_Array) CHECK(cudaFree(this->SVONodes_multiLayer_Array));
	if (this->SVODivisibleNodeBlockWeight) CHECK(cudaFree(this->SVODivisibleNodeBlockWeight));
	if (this->SVONodeWeights) CHECK(cudaFree(this->SVONodeWeights));

	CHECK(cudaDestroyExternalSemaphore(extSvoSemaphore_PG));
}

void FzbSVOCuda_PG::coypyOctreeDataToBuffer(std::vector<FzbBuffer>& OctreeNodesBuffers) {
	if (OctreeNodesBuffers.size() != this->SVONodes_maxDepth - 2) throw std::runtime_error("OctreeBuffer数量不匹配");
	for (int i = 0; i < this->SVONodes_maxDepth - 2; ++i) {
		FzbBuffer& OctreeNodesBuffer = OctreeNodesBuffers[i];
		cudaExternalMemory_t OctreeNodesBufferExtMem = importVulkanMemoryObjectFromNTHandle(OctreeNodesBuffer.handle, OctreeNodesBuffer.size, false);
		FzbSVONodeData_PG* OctreeNodesBuffer_ptr = (FzbSVONodeData_PG*)mapBufferOntoExternalMemory(OctreeNodesBufferExtMem, 0, OctreeNodesBuffer.size);
		CHECK(cudaMemcpy(OctreeNodesBuffer_ptr, OctreeNodes_multiLayer[i + 1], OctreeNodesBuffer.size, cudaMemcpyDeviceToDevice));
		CHECK(cudaDestroyExternalMemory(OctreeNodesBufferExtMem));
		CHECK(cudaFree(OctreeNodesBuffer_ptr));
	}
}
void FzbSVOCuda_PG::copySVODataToBuffer(std::vector<FzbBuffer>& SVONodesBuffers, FzbBuffer SVOWeightsBuffer) {
	if (SVONodesBuffers.size() != this->SVONodes_maxDepth - 1) throw std::runtime_error("SVOBuffer数量不匹配");
	for (int i = 0; i < this->SVONodes_maxDepth - 1; ++i) {
		FzbBuffer& SVONodesBuffer = SVONodesBuffers[i];
		cudaExternalMemory_t SVONodesBufferExtMem = importVulkanMemoryObjectFromNTHandle(SVONodesBuffer.handle, SVONodesBuffer.size, false);
		FzbSVONodeData_PG* SVONodesBuffer_ptr = (FzbSVONodeData_PG*)mapBufferOntoExternalMemory(SVONodesBufferExtMem, 0, SVONodesBuffer.size);
		CHECK(cudaMemcpy(SVONodesBuffer_ptr, SVONodes_multiLayer[i + 1], SVONodesBuffer.size, cudaMemcpyDeviceToDevice));
		CHECK(cudaDestroyExternalMemory(SVONodesBufferExtMem));
		CHECK(cudaFree(SVONodesBuffer_ptr));
	}
	cudaExternalMemory_t SVOWeightBufferExtMem = importVulkanMemoryObjectFromNTHandle(SVOWeightsBuffer.handle, SVOWeightsBuffer.size, false);
	FzbSVONodeData_PG* SVOWeightBuffer_ptr = (FzbSVONodeData_PG*)mapBufferOntoExternalMemory(SVOWeightBufferExtMem, 0, SVOWeightsBuffer.size);
	CHECK(cudaMemcpy(SVOWeightBuffer_ptr, SVONodeWeights, SVOWeightsBuffer.size, cudaMemcpyDeviceToDevice));
	CHECK(cudaDestroyExternalMemory(SVOWeightBufferExtMem));
	CHECK(cudaFree(SVOWeightBuffer_ptr));
}
