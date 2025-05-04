#pragma once

#include "../../CUDA/vulkanCudaInterop.cuh"

#ifndef CREATE_SVO_CUH
#define CREATE_SVO_CUH

struct FzbVoxelValue {
	uint32_t pos_num;	//ǰ24Ϊ��ʾ��һ�����������꣬��8λ��ʾ�����������ΰ�������������
};

struct FzbSVONode {
	uint32_t nodeIndex;	//��ǰ�ڵ��ڰ˲����е�����
	uint32_t voxelNum;	//�ýڵ���������Ҷ�ӽڵ���
	//uint32_t subNodeInfomation;	//ǰ25λ��ʾ�ӽڵ���ʼ�������м�3λ��ʾ��ǰ�ڵ��Ǹ��ڵ�ĵڼ����ӽڵ㣬��4λ��ʾ�ӽڵ���
	//������м�ڵ㣬��subsequentIndex��ʾ�ӽڵ����ʼ�����������Ҷ�ڵ㣬���ʾ�������ڵ�����������
	//�Ƿ���Ҷ�ڵ���ж�������hasSubNode�Ƿ�Ϊ0x00
	uint32_t subsequentIndex;
	uint32_t hasSubNode;
	glm::vec4 nodePos;	//��ǰnode���½����꣬ÿ������ռ10λ
	float nodeSize;

	FzbSVONode() {
		nodeIndex = 0;
		voxelNum = 0;
		subsequentIndex = 0;
		hasSubNode = 0;
		nodePos = glm::vec4(0.0f);
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
	FzbSVONode* nodePool;	//��������Ҫ�Ľڵ�����
	FzbVoxelValue* svoVoxelValueCompressedArray;	//��������Ҫ����������

	HANDLE voxelValueArrayHanlde;
	HANDLE nodePoolHandle;
	cudaExternalMemory_t voxelValueArrayExtMem;
	cudaExternalMemory_t nodePoolExtMem;

	SVOCuda() {};

	void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, MyImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle);
	void getSVOCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE nodePoolHandle, HANDLE voxelValueArrayHandle);
	void clean();

};

//struct FzbSVOCudaVariable;
//struct FzbVoxelValue;
//void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, MyImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle, FzbSVOCudaVariable*& fzbSVOCudaVar);
//void cleanSVOCuda(FzbSVOCudaVariable* fzbSVOCudaVar);

#endif