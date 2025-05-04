#pragma once

#include "./createSVO.cuh"

#ifndef CREATE_SVO_CU
#define CREATE_SVO_CU
//-------------------------------------------------------------------�˺���------------------------------------------------------------------

__device__ float4 unpackUnorm4x8(uint32_t valueU) {
	float4 value;
	value.x = ((float(valueU & 0xFF)) / 255);
	value.y = ((float((valueU >> 8) & 0xFF)) / 255.0f);
	value.z = ((float((valueU >> 16) & 0xFF)) / 255.0f);
	value.w = ((float((valueU >> 24) & 0xFF)) / 255.0f);
	return value;
}

__device__ uint32_t packUnorm4x8(float4 value) {
	return static_cast<unsigned int>(value.x * 255.0f) | (static_cast<unsigned int>(value.y * 255.0f) << 8)
		| (static_cast<unsigned int>(value.z * 255.0f) << 16) | (static_cast<unsigned int>(value.y * 255.0f) << 24);
}

/*
����˺����У����ǿ���64x64x64���̣߳�ÿ��8x8x8���̡߳�
ÿ���߳̿���585��С�Ĺ����ڴ棬�����ڴ��ÿ��Ԫ�ش�������ÿ���ڵ����������������
ͬʱ���������˲����Ľڵ�������
*/
//�Ȳ��ù����ڴ�
/*
����˺������Եõ�һ�����˲������飬�����ܶ�ڵ�����ֵ�ģ�����Ҫ���Ǻ������ѹ��
������ֵ�Ľڵ��������Ϊ���ڵ�İ˲�����Ҷ�ڵ�������ֵ�ĺ��ӽڵ������
*/
__global__ void getSVONum_withoutShared(cudaTextureObject_t voxelGridMap, uint32_t* voxelNum, FzbSVONode* svoNodeArray, uint32_t svoDepth, FzbVoxelValue* svoVoxelValueArray) {

	//extern __shared__ int subSVONodeNum[];	//�����ڴ�����Ϊ�ȱ�������ͣ����Ƿ����ⲿ����
	uint3 voxelIndexU3;
	voxelIndexU3.x = threadIdx.x;
	voxelIndexU3.y = threadIdx.y;
	voxelIndexU3.z = threadIdx.z;

	//���ﲻ֪������ȡ�Ƿ��죬֮�����һ��
	uint32_t valueU = tex3D<uint32_t>(voxelGridMap, voxelIndexU3.x + 0.5f, voxelIndexU3.y + 0.5f, voxelIndexU3.z + 0.5f);
	float4 value = unpackUnorm4x8(valueU);
	if (value.w <= 0) {
		return;
	}
	uint32_t svoVoxelValueIndex = atomicAdd(voxelNum, 1);
	svoVoxelValueArray[svoVoxelValueIndex].pos_num = valueU;

	atomicAdd(&svoNodeArray[0].voxelNum, 1);
	int fatherNodeIndex = 0;
	int curNodeIndex = 0;
	int detailLevel = gridDim.x * blockDim.x;
	for (int i = 1; i < svoDepth; i++) {
		uint3 index;
		index.x = voxelIndexU3.x % detailLevel;
		index.y = voxelIndexU3.y % detailLevel;
		index.z = voxelIndexU3.z % detailLevel;
		detailLevel /= 2;
		index.x /= detailLevel;
		index.y /= detailLevel;
		index.z /= detailLevel;

		int subNodeIndex = index.y * 2 + index.x + index.z * 4;
		curNodeIndex = subNodeIndex + fatherNodeIndex * 8 + 1;
		atomicAdd(&svoNodeArray[curNodeIndex].voxelNum, 1);

		uint32_t hasSubNode = 1 << subNodeIndex;
		atomicOr(&svoNodeArray[fatherNodeIndex].hasSubNode, hasSubNode);

		fatherNodeIndex = curNodeIndex;
	}
	svoNodeArray[curNodeIndex].subsequentIndex = svoVoxelValueIndex;

}

__global__ void compressSVO(FzbSVONode* nodeArray, FzbSVONode* nodePool, uint32_t nodeStartIndex, uint32_t subArrayStartIndex, uint32_t* subArrayNum) {

	int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	int nodeIndex = nodeStartIndex + threadIndex;
	FzbSVONode node = nodeArray[nodeIndex];
	node.nodeIndex = nodeIndex;

	int nodePoolIndex = subArrayStartIndex + threadIndex;
	nodePool[nodePoolIndex] = node;

	if (node.hasSubNode == 0)
		return;

	int subNodeNum = 0;
	for (int i = 0; i < 8; i++) {
		if (node.hasSubNode & (1u << i))
			subNodeNum++;
	}
	int subArrayIndex = 0;
	if (subNodeNum > 0) {
		subArrayIndex = atomicAdd(subArrayNum, 1);
		nodePool[nodePoolIndex].subsequentIndex = subArrayIndex * 8;
	}
	//printf("%d\n", subArrayIndex);
	compressSVO << <1, 8 >> > (nodeArray, nodePool, nodeIndex * 8 + 1, subArrayIndex * 8, subArrayNum);

}
/*
__global__ void compressSVO(FzbSVONode* svoNodeArray, FzbSVONode* svoNodeCompressedArray, int* subArrayNum, int* svoDepth) {

	int nodeIndex = svoNodeCompressedArray[blockDim.x * blockIdx.x + threadIdx.x].nodeIndex;
	FzbSVONode node = svoNodeArray[nodeIndex];
	if (node.voxelNum == 0)
		return;
	int subArray = atomicAdd(subArrayNum, 1);

	int subNodeNum = 0;
	int subNodeStartIndex = 8 * nodeIndex + 1;
	for (int i = 0; i < 8; i++) {
		if (!(node.hasSubNode & 1 << i))
			continue;
		svoNodeCompressedArray[subArray * 8 + subNodeNum].nodeIndex = subNodeStartIndex + i;
		subNodeNum++;
	}

}
*/
//-------------------------------------------------------------------------------------------------------------------------
void CUDART_CB cleanTempData(cudaStream_t stream, cudaError_t status, void* userData) {

	SVOCuda* svoCuda = (SVOCuda*)userData;
	CHECK(cudaDestroyExternalSemaphore(svoCuda->extVgmSemaphore));
	CHECK(cudaDestroyExternalSemaphore(svoCuda->extSvoSemaphore));

	CHECK(cudaDestroyTextureObject(svoCuda->vgm));
	CHECK(cudaFreeMipmappedArray(svoCuda->vgmMipmap));
	CHECK(cudaDestroyExternalMemory(svoCuda->vgmExtMem));

}

void SVOCuda::createSVOCuda(VkPhysicalDevice vkPhysicalDevice, MyImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle) {

	unsigned long long size = voxelGridMap.width * voxelGridMap.height * voxelGridMap.depth * sizeof(uint32_t);
	fromVulkanImageToCudaTexture(vkPhysicalDevice, voxelGridMap, voxelGridMap.handle, size, true, vgmExtMem, vgmMipmap, vgm);

	extVgmSemaphore = importVulkanSemaphoreObjectFromNTHandle(vgmSemaphoreHandle);
	extSvoSemaphore = importVulkanSemaphoreObjectFromNTHandle(svoSemaphoreHandle);

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	waitExternalSemaphore(extVgmSemaphore, stream);

	dim3 gridSize(voxelGridMap.width / 8, voxelGridMap.height / 8, voxelGridMap.depth / 8);
	dim3 blockSize(8, 8, 8);
	//���SVO�����
	int svoDepth = 1;
	int vgmSize = voxelGridMap.width;
	while (vgmSize > 1) {
		svoDepth++;
		vgmSize >>= 1;
	}
	//int svoDepth_group = svoDepth - 3;	//ÿһ��ʵ������3��������������������ʣ��Ĳ���

	uint32_t* voxelNum_p;	//��ֵ�����ص�����
	CHECK(cudaMalloc((void**)&voxelNum_p, sizeof(uint32_t)));	//��ʹ�ù̶��ڴ棬���޷�ʹ��ԭ������

	uint32_t maxNodeNum = uint32_t((pow(8, svoDepth) - 1) / 7);	//���˲����������ڵ���

	//����һ�����˲������飬�����ŷ�
	FzbSVONode* svoNodeArray;
	CHECK(cudaMalloc((void**)&svoNodeArray, sizeof(FzbSVONode) * maxNodeNum));
	//����һ����ֵ�������ݵ�����
	FzbVoxelValue* svoVoxelValueArray;
	CHECK(cudaMalloc((void**)&svoVoxelValueArray, sizeof(FzbVoxelValue) * voxelGridMap.width * voxelGridMap.height * voxelGridMap.depth));

	getSVONum_withoutShared << <gridSize, blockSize, 0, stream >> > (vgm, voxelNum_p, svoNodeArray, svoDepth, svoVoxelValueArray);
	CHECK(cudaStreamSynchronize(stream));

	//����ѹ�����������������
	this->voxelNum = *voxelNum_p;
	CHECK(cudaMalloc((void**)&svoVoxelValueCompressedArray, sizeof(FzbVoxelValue) * this->voxelNum));
	CHECK(cudaMemcpy(svoVoxelValueCompressedArray, svoVoxelValueArray, sizeof(FzbVoxelValue) * this->voxelNum, cudaMemcpyDeviceToDevice));

	uint32_t* subArrayNum_host = (uint32_t*)malloc(sizeof(uint32_t));
	*subArrayNum_host = 1;
	uint32_t* subArrayNum;
	CHECK(cudaMalloc((void**)&subArrayNum, sizeof(uint32_t)));
	CHECK(cudaMemcpy(subArrayNum, subArrayNum_host, sizeof(uint32_t), cudaMemcpyHostToDevice));
	
	CHECK(cudaMalloc((void**)&nodePool, sizeof(FzbSVONode) * maxNodeNum));

	compressSVO << <1, 1, 0, stream >> > (svoNodeArray, nodePool, 0, 0, subArrayNum);
	CHECK(cudaStreamSynchronize(stream));
	this->nodeArrayNum = *subArrayNum_host;

	//CHECK(cudaStreamSynchronize(stream));
	signalExternalSemaphore(extSvoSemaphore, stream);
	CHECK(cudaStreamAddCallback(stream, cleanTempData, this, 0));

	CHECK(cudaFree(voxelNum_p));
	CHECK(cudaFree(svoNodeArray));
	CHECK(cudaFree(svoVoxelValueArray));
	CHECK(cudaFree(subArrayNum));
	free(subArrayNum_host);
	CHECK(cudaStreamDestroy(stream));

}

void SVOCuda::getSVOCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE nodePoolHandle, HANDLE voxelValueArrayHandle) {
	//���ж��Ƿ���ͬһ�������豸
	if (getCudaDeviceForVulkanPhysicalDevice(vkPhysicalDevice) == cudaInvalidDeviceId) {
		throw std::runtime_error("CUDA��Vulkan�õĲ���ͬһ��GPU������");
	}
	cudaExternalMemory_t nodePoolExtMem = importVulkanMemoryObjectFromNTHandle(nodePoolHandle, 0, false);
	FzbSVONode* vkNodePool = (FzbSVONode*)mapBufferOntoExternalMemory(nodePoolExtMem, 0, sizeof(FzbSVONode) * 8 * this->nodeArrayNum);
	CHECK(cudaMemcpy(vkNodePool, this->nodePool, sizeof(FzbSVONode) * 8 * this->nodeArrayNum, cudaMemcpyDeviceToDevice));

	cudaExternalMemory_t voxelValueArrayExtMem = importVulkanMemoryObjectFromNTHandle(voxelValueArrayHandle, 0, false);
	FzbVoxelValue* vkVoxelValueArray = (FzbVoxelValue*)mapBufferOntoExternalMemory(voxelValueArrayExtMem, 0, sizeof(FzbVoxelValue) * this->voxelNum);
	CHECK(cudaMemcpy(vkVoxelValueArray, this->svoVoxelValueCompressedArray, sizeof(FzbVoxelValue) * this->voxelNum, cudaMemcpyDeviceToDevice));

	CHECK(cudaDestroyExternalMemory(this->nodePoolExtMem));
	CHECK(cudaDestroyExternalMemory(this->voxelValueArrayExtMem));
	CHECK(cudaFree(nodePool));
	CHECK(cudaFree(svoVoxelValueCompressedArray));

}

void SVOCuda::clean() {
	//CloseHandle(this->nodePoolHandle);
	//CHECK(cudaDestroyExternalMemory(this->nodePoolExtMem));
	//CloseHandle(this->voxelValueArrayHanlde);
	//CHECK(cudaDestroyExternalMemory(this->voxelValueArrayExtMem));

	//CHECK(cudaFree(nodePool));
	//CHECK(cudaFree(svoVoxelValueCompressedArray));
}

#endif