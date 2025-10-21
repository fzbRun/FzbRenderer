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

	//��������SVO���飬�ӵڶ�����ʼ
	uint32_t vgmSize = 1;
	while (vgmSize <= setting.voxelNum) {
		SVONodes_maxDepth++;
		vgmSize <<= 1;
	}

	initLightInjectSource();
	initCreateOctreeNodesSource();
	initCreateSVONodesSource();
	//initGetSVONodesWeightSource();

	this->extSvoSemaphore_PG = importVulkanSemaphoreObjectFromNTHandle(SVOFinishedSemaphore_PG);
}

//--------------------------------------------------------------------��ʼ��VGB-------------------------------------------------------------------------
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
//--------------------------------------------------------------------����SVO_PG-------------------------------------------------------------------------
void FzbSVOCuda_PG::createSVOCuda_PG(HANDLE VGBFinishedSemaphore) {
	if (setting.voxelNum > 128) {
		std::cout << "voxelNum����128������Ҫ����ѹ����Ŀǰ�޷�ʵ�֣�SVOδѹ��" << std::endl;
		return;
	}
	waitExternalSemaphore(importVulkanSemaphoreObjectFromNTHandle(VGBFinishedSemaphore), stream);
	lightInject();
	createOctreeNodes();
	createSVONodes();
	//getSVONodesWeight();
}
//--------------------------------------------------------------------����weight-------------------------------------------------------------------------
__global__ void getSVONodesWeight_device() {

}

//void FzbSVOCuda_PG::getSVONodesWeight() {
//	uint32_t nodeCount = 0;
//	for (int i = 0; i <= this->SVONodes_multiLayer.size(); ++i) nodeCount += this->SVONodeCount_host[i] * 8;
//	for (int i = 1; i <= this->SVONodes_multiLayer.size(); ++i) nodeCount -= this->SVONodeCount_host[i];	//ȥ���ɷ�Node
//
//	for (int i = 0; i < this->SVONodeWeights.size(); ++i) CHECK(cudaFree(this->SVONodeWeights[i]));
//	this->SVONodeWeights.resize(nodeCount);
//	for (int i = 0; i < nodeCount; ++i) CHECK(cudaMalloc((void**)&this->SVONodeWeights[i], sizeof(float) * nodeCount));
//}
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

	for (int i = 0; i < this->SVONodeWeights.size(); ++i) if (this->SVONodeWeights[i]) CHECK(cudaFree(this->SVONodeWeights[i]));
	if (this->SVONodeWeightsArray) CHECK(cudaFree(this->SVONodeWeightsArray));
	if (this->SVONodeTotalWeightArray) CHECK(cudaFree(this->SVONodeTotalWeightArray));

	CHECK(cudaDestroyExternalSemaphore(extSvoSemaphore_PG));
}

void FzbSVOCuda_PG::copyDataToBuffer(std::vector<FzbBuffer>& buffers) {
	if (buffers.size() != this->SVONodes_maxDepth - 1) throw std::runtime_error("SVOBuffer������ƥ��");
	for (int i = 0; i < this->SVONodes_maxDepth - 1; ++i) {
		FzbBuffer& SVOBuffer = buffers[i];
		cudaExternalMemory_t SVOBufferExtMem = importVulkanMemoryObjectFromNTHandle(SVOBuffer.handle, SVOBuffer.size, false);
		FzbSVONodeData_PG* SVOBuffer_ptr = (FzbSVONodeData_PG*)mapBufferOntoExternalMemory(SVOBufferExtMem, 0, SVOBuffer.size);
		CHECK(cudaMemcpy(SVOBuffer_ptr, SVONodes_multiLayer[i + 1], SVOBuffer.size, cudaMemcpyDeviceToDevice));
		CHECK(cudaDestroyExternalMemory(SVOBufferExtMem));
		CHECK(cudaFree(SVOBuffer_ptr));
	}
}
