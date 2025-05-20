#pragma once

#include "../../../CUDA/vulkanCudaInterop.cuh"

#ifndef CREATE_SVO_CUH
#define CREATE_SVO_CUH

struct FzbVoxelValue {
	uint32_t pos_num;	//前24为表示归一化的世界坐标，后8位表示体素内三角形包含的像素数量
};

struct FzbSVONode {
	uint32_t nodeIndex;	//当前节点在八叉树中的索引
	uint32_t voxelNum;	//该节点所包含的叶子节点数
	uint32_t subsequentIndex;
	uint32_t hasSubNode;
	glm::vec4 nodePos_Size;	//当前node左下角坐标，每个分量占10位

	__device__ __host__ FzbSVONode() {
		nodeIndex = 0;
		voxelNum = 0;
		subsequentIndex = 0;
		hasSubNode = 0;
		nodePos_Size = glm::vec4();
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

	uint32_t nodeArrayNum;
	uint32_t voxelNum;
	FzbSVONode* nodePool;	//后续所需要的节点数组
	FzbVoxelValue* svoVoxelValueArray;	//后续所需要的体素数据


	SVOCuda() {};

	void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, FzbImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle, glm::vec4 vgmStartPos, float voxelSize);
	void getSVOCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE nodePoolHandle, HANDLE voxelValueArrayHandle);
	void clean();

};

#endif