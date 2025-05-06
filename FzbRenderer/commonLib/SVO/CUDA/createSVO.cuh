#pragma once

#include "../../CUDA/vulkanCudaInterop.cuh"

#ifndef CREATE_SVO_CUH
#define CREATE_SVO_CUH

struct FzbVoxelValue {
	uint32_t pos_num;	//前24为表示归一化的世界坐标，后8位表示体素内三角形包含的像素数量
};

struct FzbSVONode {
	uint32_t nodeIndex;	//当前节点在八叉树中的索引
	uint32_t voxelNum;	//该节点所包含的叶子节点数
	//uint32_t subNodeInfomation;	//前25位表示子节点起始索引，中间3位表示当前节点是父节点的第几个子节点，后4位表示子节点数
	//如果是中间节点，则subsequentIndex表示子节点的起始索引；如果是叶节点，则表示数据所在的数组索引。
	//是否是叶节点的判断依据是hasSubNode是否为0x00
	uint32_t subsequentIndex;
	uint32_t hasSubNode;
	glm::vec4 nodePos_Size;	//当前node左下角坐标，每个分量占10位

	FzbSVONode() {
		nodeIndex = 0;
		voxelNum = 0;
		subsequentIndex = 0;
		hasSubNode = 0;
		nodePos_Size = glm::vec4();
		hasSubNode = 0.0f;
	}

};

class SVOCuda {

public:

	cudaExternalMemory_t vgmExtMem;
	cudaMipmappedArray_t vgmMipmap;
	cudaTextureObject_t vgm = 0;
	cudaExternalSemaphore_t extVgmSemaphore;
	cudaExternalSemaphore_t extSvoSemaphore;

	uint32_t nodeArrayNum;
	uint32_t voxelNum;
	FzbSVONode* nodePool;	//后续所需要的节点数组
	FzbVoxelValue* svoVoxelValueCompressedArray;	//后续所需要的体素数据

	SVOCuda() {};

	void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, MyImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle, glm::vec4 vgmStartPos, float voxelSize);
	void getSVOCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE nodePoolHandle, HANDLE voxelValueArrayHandle);
	void clean();

};

//struct FzbSVOCudaVariable;
//struct FzbVoxelValue;
//void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, MyImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle, FzbSVOCudaVariable*& fzbSVOCudaVar);
//void cleanSVOCuda(FzbSVOCudaVariable* fzbSVOCudaVar);

#endif