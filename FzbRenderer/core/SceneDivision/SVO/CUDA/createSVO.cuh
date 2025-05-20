#pragma once

#include "../../../CUDA/vulkanCudaInterop.cuh"

#ifndef CREATE_SVO_CUH
#define CREATE_SVO_CUH

struct FzbVoxelValue {
	uint32_t pos_num;	//ǰ24Ϊ��ʾ��һ�����������꣬��8λ��ʾ�����������ΰ�������������
};

struct FzbSVONode {
	uint32_t nodeIndex;	//��ǰ�ڵ��ڰ˲����е�����
	uint32_t voxelNum;	//�ýڵ���������Ҷ�ӽڵ���
	uint32_t subsequentIndex;
	uint32_t hasSubNode;
	glm::vec4 nodePos_Size;	//��ǰnode���½����꣬ÿ������ռ10λ

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
	FzbSVONode* nodePool;	//��������Ҫ�Ľڵ�����
	FzbVoxelValue* svoVoxelValueArray;	//��������Ҫ����������


	SVOCuda() {};

	void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, FzbImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle, glm::vec4 vgmStartPos, float voxelSize);
	void getSVOCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE nodePoolHandle, HANDLE voxelValueArrayHandle);
	void clean();

};

#endif