#pragma once

#include "../../../CUDA/vulkanCudaInterop.cuh"

#ifndef CREATE_SVO_CUH
#define CREATE_SVO_CUH

struct FzbVoxelValue {
	uint32_t pos_num;	//ǰ24Ϊ��ʾ��һ�����������꣬��8λ��ʾ�����������ΰ�������������
};

struct FzbSVONode {
	uint32_t shuffleKey;	//��ǰ�ڵ��ڰ˲����еĸ�����ά����, z1y1x1����z7y7x7��ǰ4λ�����ʾ�㼶
	uint32_t voxelNum;	//�ýڵ���������Ҷ�ӽڵ���
	//uint32_t subsequentIndex;
	uint32_t label;
	uint32_t hasSubNode;
	
	__device__ __host__ FzbSVONode() {
		shuffleKey = 0;
		voxelNum = 0;
		label = 1;
		hasSubNode = 0;
	}

};

struct FzbNodePoolBlock {
	uint32_t startIndex;
	uint32_t blockNum;
	uint32_t nodeNum;

	__device__ FzbNodePoolBlock(uint32_t startIndex, uint32_t blockNum, uint32_t nodeNum) {
		this->startIndex = startIndex;
		this->blockNum = blockNum;
		this->nodeNum = nodeNum;
	}

	__device__ FzbNodePoolBlock() {
		this->startIndex = 0;
		this->blockNum = 0;
	}
};

class SVOCuda {

public:

	cudaExternalMemory_t vgmExtMem;
	cudaMipmappedArray_t vgmMipmap;
	cudaTextureObject_t vgm = 0;
	cudaExternalSemaphore_t extVgmSemaphore;
	cudaExternalSemaphore_t extSvoSemaphore;

	cudaExternalMemory_t nodePoolExtMem;
	cudaExternalMemory_t voxelValueArrayExtMem;

	cudaStream_t stream;

	uint32_t nodeBlockNum;	//8�����ݵ�block����������Ҫ�����1��ʾ���ڵ�
	uint32_t voxelNum;
	FzbSVONode* nodePool;	//��������Ҫ�Ľڵ�����
	FzbVoxelValue* svoVoxelValueArray;	//��������Ҫ����������


	SVOCuda() {};

	void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, FzbImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle, bool isPresent = false);
	void getSVOCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE nodePoolHandle, HANDLE voxelValueArrayHandle);
	void clean();

};

#endif