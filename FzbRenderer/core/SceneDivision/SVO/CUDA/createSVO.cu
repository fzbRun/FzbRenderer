#pragma once

#include "./createSVO.cuh"

#ifndef CREATE_SVO_CU
#define CREATE_SVO_CU
//-------------------------------------------------------------------�˺���------------------------------------------------------------------

/*
__device__ float4 unpackUnorm4x8(uint32_t valueU) {
	float4 value;
	value.w = ((float(valueU & 0xFF)) / 255);
	value.z = ((float((valueU >> 8) & 0xFF)) / 255.0f);
	value.y = ((float((valueU >> 16) & 0xFF)) / 255.0f);
	value.x = ((float((valueU >> 24) & 0xFF)) / 255.0f);
	return value;
}

__device__ uint32_t packUnorm4x8(float4 value) {
	return static_cast<unsigned int>(value.w * 255.0f) | (static_cast<unsigned int>(value.z * 255.0f) << 8)
		| (static_cast<unsigned int>(value.y * 255.0f) << 16) | (static_cast<unsigned int>(value.x * 255.0f) << 24);
}
*/

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
__global__ void getSVONum(cudaTextureObject_t voxelGridMap, uint32_t* voxelNum, FzbSVONode* svoNodeArray, uint32_t svoDepth, FzbVoxelValue* svoVoxelValueArray) {

	//extern __shared__ int subSVONodeNum[];	//�����ڴ�����Ϊ�ȱ�������ͣ����Ƿ����ⲿ����
	glm::uvec3 voxelIndexU3 = glm::uvec3(blockDim.x * blockIdx.x + threadIdx.x , blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);

	//���ﲻ֪������ȡ�Ƿ��죬֮�����һ��
	uint32_t valueU = tex3D<uint32_t>(voxelGridMap, voxelIndexU3.x, voxelIndexU3.y, voxelIndexU3.z);
	//glm::vec4 value = glm::unpackUnorm4x8(valueU);
	if (valueU <= 0) {
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

__global__ void compressSVO(FzbSVONode* nodeArray, FzbSVONode* nodePool, uint32_t nodeStartIndex, uint32_t subArrayStartIndex, uint32_t* subArrayNum, glm::vec4 fatherNodePos_Size, uint32_t recursionEnd) {

	int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	int nodeIndex = nodeStartIndex + threadIndex;
	FzbSVONode node = nodeArray[nodeIndex];
	if (node.voxelNum == 0)
		return;

	float nodeSize = fatherNodePos_Size.w / 2;
	glm::vec4 nodePos_Size = fatherNodePos_Size;
	nodePos_Size.x += nodeSize * (threadIndex % 2);
	nodePos_Size.y += nodeSize * ((threadIndex % 4) / 2);
	nodePos_Size.z += nodeSize * (threadIndex / 4);
	nodePos_Size.w = nodeSize;

	node.nodeIndex = nodeIndex;
	node.nodePos_Size = nodePos_Size;

	int nodePoolIndex = subArrayStartIndex + threadIndex;
	nodePool[nodePoolIndex] = node;

	//int subNodeNum = 0;
	//for (int i = 0; i < 8; i++) {
	//	if (node.hasSubNode & (1u << i))
	//		subNodeNum++;
	//}
	//if (subNodeNum == 0)
	//	return;
	if (nodeStartIndex > recursionEnd)
		return;
	int subArrayIndex = atomicAdd(subArrayNum, 1);
	nodePool[nodePoolIndex].subsequentIndex = subArrayIndex * 8;
	compressSVO << <1, 8 >> > (nodeArray, nodePool, nodeIndex * 8 + 1, subArrayIndex * 8, subArrayNum, nodePos_Size, recursionEnd);

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

void SVOCuda::createSVOCuda(VkPhysicalDevice vkPhysicalDevice, FzbImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle, glm::vec4 vgmStartPos, float voxelSize) {

	unsigned long long size = voxelGridMap.width * voxelGridMap.height * voxelGridMap.depth * sizeof(uint32_t);
	fromVulkanImageToCudaTexture(vkPhysicalDevice, voxelGridMap, voxelGridMap.handle, size, false, vgmExtMem, vgmMipmap, vgm);

	extVgmSemaphore = importVulkanSemaphoreObjectFromNTHandle(vgmSemaphoreHandle);
	extSvoSemaphore = importVulkanSemaphoreObjectFromNTHandle(svoSemaphoreHandle);

	double start = cpuSecond();

	CHECK(cudaStreamCreate(&stream));

	dim3 gridSize(voxelGridMap.width / 8, voxelGridMap.height / 8, voxelGridMap.depth / 8);
	dim3 blockSize(8, 8, 8);
	//���SVO�����
	int svoDepth = 1;
	int vgmSize = voxelGridMap.width;
	while (vgmSize > 1) {
		svoDepth++;
		vgmSize >>= 1;
	}

	uint32_t* voxelNum_p;	//��ֵ�����ص�����
	CHECK(cudaMalloc((void**)&voxelNum_p, sizeof(uint32_t)));	//��ʹ�ù̶��ڴ棬���޷�ʹ��ԭ������
	CHECK(cudaMemset(voxelNum_p, 0, sizeof(uint32_t)));

	uint32_t maxNodeNum = uint32_t((pow(8, svoDepth) - 1) / 7);	//���˲����������ڵ���

	//����һ�����˲������飬�����ŷ�
	FzbSVONode* svoNodeArray;
	CHECK(cudaMalloc((void**)&svoNodeArray, sizeof(FzbSVONode) * maxNodeNum));
	CHECK(cudaMemset(svoNodeArray, 0, sizeof(FzbSVONode) * maxNodeNum));
	//����һ����ֵ�������ݵ�����
	CHECK(cudaMalloc((void**)&svoVoxelValueArray, sizeof(FzbVoxelValue) * voxelGridMap.width * voxelGridMap.height * voxelGridMap.depth));
	CHECK(cudaMemset(svoVoxelValueArray, 0, sizeof(FzbVoxelValue) * voxelGridMap.width * voxelGridMap.height * voxelGridMap.depth));

	waitExternalSemaphore(extVgmSemaphore, stream);

	getSVONum << <gridSize, blockSize, 0, stream >> > (vgm, voxelNum_p, svoNodeArray, svoDepth, svoVoxelValueArray);
	//CHECK(cudaStreamSynchronize(stream));

	//����ѹ�����������������
	CHECK(cudaMemcpy(&this->voxelNum, voxelNum_p, sizeof(uint32_t), cudaMemcpyDeviceToHost));

	uint32_t* subArrayNum_host = (uint32_t*)malloc(sizeof(uint32_t));
	*subArrayNum_host = 1;
	uint32_t* subArrayNum;
	CHECK(cudaMalloc((void**)&subArrayNum, sizeof(uint32_t)));
	CHECK(cudaMemcpy(subArrayNum, subArrayNum_host, sizeof(uint32_t), cudaMemcpyHostToDevice));
	
	CHECK(cudaMalloc((void**)&nodePool, sizeof(FzbSVONode) * maxNodeNum));
	CHECK(cudaMemset(nodePool, 0, sizeof(FzbSVONode) * maxNodeNum));

	glm::vec4 nodePos_Size = glm::vec4(vgmStartPos.x, vgmStartPos.y, vgmStartPos.z, voxelSize * 2.0f);
	uint32_t recursionEnd = uint32_t((pow(8, svoDepth - 1) - 1) / 7) - 1;
	compressSVO << <1, 1, 0, stream >> > (svoNodeArray, nodePool, 0, 0, subArrayNum, nodePos_Size, recursionEnd);
	//CHECK(cudaStreamSynchronize(stream));
	CHECK(cudaMemcpy(&this->nodeArrayNum, subArrayNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));

	//CHECK(cudaStreamSynchronize(stream));
	//CHECK(cudaStreamAddCallback(stream, cleanTempData, this, 0));

	CHECK(cudaDestroyExternalSemaphore(this->extVgmSemaphore));
	//CHECK(cudaDestroyExternalSemaphore(this->extSvoSemaphore));

	CHECK(cudaDestroyTextureObject(this->vgm));
	CHECK(cudaFreeMipmappedArray(this->vgmMipmap));
	CHECK(cudaDestroyExternalMemory(this->vgmExtMem));

	CHECK(cudaFree(voxelNum_p));
	CHECK(cudaFree(svoNodeArray));
	CHECK(cudaFree(subArrayNum));
	free(subArrayNum_host);

	std::cout << cpuSecond() - start << std::endl;

}

void SVOCuda::getSVOCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE nodePoolHandle, HANDLE voxelValueArrayHandle) {
	//���ж��Ƿ���ͬһ�������豸
	if (getCudaDeviceForVulkanPhysicalDevice(vkPhysicalDevice) == cudaInvalidDeviceId) {
		throw std::runtime_error("CUDA��Vulkan�õĲ���ͬһ��GPU������");
	}

	//waitExternalSemaphore(extSvoSemaphore);

	cudaExternalMemory_t nodePoolExtMem = importVulkanMemoryObjectFromNTHandle(nodePoolHandle, sizeof(FzbSVONode) * 8 * this->nodeArrayNum, false);
	FzbSVONode* vkNodePool = (FzbSVONode*)mapBufferOntoExternalMemory(nodePoolExtMem, 0, sizeof(FzbSVONode) * 8 * this->nodeArrayNum);
	CHECK(cudaMemcpy(vkNodePool, this->nodePool, sizeof(FzbSVONode) * 8 * this->nodeArrayNum, cudaMemcpyDeviceToDevice));

	cudaExternalMemory_t voxelValueArrayExtMem = importVulkanMemoryObjectFromNTHandle(voxelValueArrayHandle, sizeof(FzbVoxelValue) * this->voxelNum, false);
	FzbVoxelValue* vkVoxelValueArray = (FzbVoxelValue*)mapBufferOntoExternalMemory(voxelValueArrayExtMem, 0, sizeof(FzbVoxelValue) * this->voxelNum);
	CHECK(cudaMemcpy(vkVoxelValueArray, this->svoVoxelValueArray, sizeof(FzbVoxelValue) * this->voxelNum, cudaMemcpyDeviceToDevice));

	signalExternalSemaphore(extSvoSemaphore, stream);

	CHECK(cudaDestroyExternalSemaphore(this->extSvoSemaphore));
	CHECK(cudaDestroyExternalMemory(nodePoolExtMem));
	CHECK(cudaDestroyExternalMemory(voxelValueArrayExtMem));
	CHECK(cudaFree(nodePool));
	CHECK(cudaFree(svoVoxelValueArray));
	CHECK(cudaStreamDestroy(stream));

}

void SVOCuda::clean() {
}

#endif