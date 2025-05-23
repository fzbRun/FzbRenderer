#pragma once

#include "../../../CUDA/vulkanCudaInterop.cuh"

#ifndef CREATE_SVO_CUH
#define CREATE_SVO_CUH

struct FzbVoxelValue {
	uint32_t pos_num;	//前24为表示归一化的世界坐标，后8位表示体素内三角形包含的像素数量
};

struct FzbSVONode {
	uint32_t shuffleKey;	//当前节点在八叉树中的各级三维索引, z1y1x1……z7y7x7；前4位额外表示层级
	uint32_t voxelNum;	//该节点所包含的叶子节点数
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

	uint32_t nodeBlockNum;	//8个数据的block的数量，需要额外加1表示根节点
	uint32_t voxelNum;
	FzbSVONode* nodePool;	//后续所需要的节点数组
	FzbVoxelValue* svoVoxelValueArray;	//后续所需要的体素数据


	SVOCuda() {};

	void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, FzbImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle, bool isPresent = false);
	void getSVOCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE nodePoolHandle, HANDLE voxelValueArrayHandle);
	void clean();

};

#endif