#pragma once

//#include "./createSVO.cuh"
//#include "../../CUDA/vulkanCudaInterop.cuh"
#include "./createSVO.cuh"

#ifndef CREATE_SVO_CU
#define CREATE_SVO_CU

//-------------------------------------------------------------------------------------------------------------------------
struct FzbSVOCudaVariable {
	cudaExternalMemory_t extMem;
	cudaMipmappedArray_t mipmap;
	cudaSurfaceObject_t surfObj = 0;
	cudaExternalSemaphore_t extVgmSemaphore;
	cudaExternalSemaphore_t extSvoSemaphore;

	FzbSVOCudaVariable() {
		extMem = nullptr;
		mipmap = nullptr;
		surfObj = 0;
		extVgmSemaphore = nullptr;
		extSvoSemaphore = nullptr;
	};
};


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

__global__ void test(cudaSurfaceObject_t voxelGridMap) {
	int voxelIndex_x = blockDim.x * blockIdx.x + threadIdx.x;
	int voxelIndex_y = blockDim.y * blockIdx.y + threadIdx.y;
	int voxelIndex_z = blockDim.z * blockIdx.z + threadIdx.z;

	uint32_t valueU = surf3Dread<uint32_t>(voxelGridMap, voxelIndex_x * sizeof(uint32_t), voxelIndex_y, voxelIndex_z, cudaBoundaryModeTrap);
	float4 value = unpackUnorm4x8(valueU);
	if (value.w > 0) {
		surf3Dwrite(uint32_t((1 << 32) - 1), voxelGridMap, voxelIndex_x * sizeof(uint32_t), voxelIndex_y, voxelIndex_z);
	}

}

/*
����˺����У����ǿ���64x64x64���̣߳�ÿ��8x8x8���̡߳�
ÿ���߳̿���585��С�Ĺ����ڴ棬�����ڴ��ÿ��Ԫ�ش�������ÿ���ڵ����������������
ͬʱ���������˲����Ľڵ�������
*/
__global__ void getSVONum(cudaSurfaceObject_t voxelGridMap, int* svoNodeNum, int* svoNode, int svoDepth_group) {

	extern __shared__ int subSVONodeNum[];	//�����ڴ�����Ϊ�ȱ�������ͣ����Ƿ����ⲿ����
	uint3 voxelIndex;
	voxelIndex.x = blockDim.x * blockIdx.x + threadIdx.x;
	voxelIndex.y = blockDim.y * blockIdx.y + threadIdx.y;
	voxelIndex.z = blockDim.z * blockIdx.z + threadIdx.z;

	//���ﲻ֪������ȡ�Ƿ��죬֮�����һ��
	uint32_t valueU = surf3Dread<uint32_t>(voxelGridMap, voxelIndex.x * sizeof(uint32_t), voxelIndex.y, voxelIndex.z, cudaBoundaryModeTrap);
	float4 value = unpackUnorm4x8(valueU);
	if (value.w <= 0) {
		return;
	}
	
	atomicAdd(svoNodeNum, 1);
	atomicAdd(svoNode[0], 1);
	for (int i = 1; i < svoDepth_group; i++) {
		int nodeIndex = gridDim.x
	}

}

//-------------------------------------------------------------------------------------------------------------------------
void createSVOCuda(VkPhysicalDevice vkPhysicalDevice, MyImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle, FzbSVOCudaVariable*& fzbSVOCudaVar) {

	unsigned long long size = voxelGridMap.width * voxelGridMap.height * voxelGridMap.depth * sizeof(uint32_t);
	fzbSVOCudaVar = new FzbSVOCudaVariable();
	fromVulkanImageToCudaSurface(vkPhysicalDevice, voxelGridMap, voxelGridMap.handle, size, true, fzbSVOCudaVar->extMem, fzbSVOCudaVar->mipmap, fzbSVOCudaVar->surfObj);

	fzbSVOCudaVar->extVgmSemaphore = importVulkanSemaphoreObjectFromNTHandle(vgmSemaphoreHandle);
	fzbSVOCudaVar->extSvoSemaphore = importVulkanSemaphoreObjectFromNTHandle(svoSemaphoreHandle);

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	waitExternalSemaphore(fzbSVOCudaVar->extVgmSemaphore, stream);

	dim3 gridSize(voxelGridMap.width / 8, voxelGridMap.height / 8, voxelGridMap.depth / 8);
	dim3 blockSize(8, 8, 8);
	int svoDepth = 0;
	int vgmSize = voxelGridMap.width;
	while (vgmSize > 1) {
		svoDepth++;
		vgmSize >> 1;
	}
	int svoDepth_group = svoDepth - 3;	//ÿһ��ʵ������3��������������������ʣ��Ĳ���

	CHECK(cudaStreamSynchronize(stream));
	signalExternalSemaphore(fzbSVOCudaVar->extSvoSemaphore, stream);

}

void cleanSVOCuda(FzbSVOCudaVariable* fzbSVOCudaVar) {
	CHECK(cudaDestroyExternalSemaphore(fzbSVOCudaVar->extVgmSemaphore));
	CHECK(cudaDestroyExternalSemaphore(fzbSVOCudaVar->extSvoSemaphore));

	CHECK(cudaDestroyTextureObject(fzbSVOCudaVar->surfObj));
	CHECK(cudaFreeMipmappedArray(fzbSVOCudaVar->mipmap));
	CHECK(cudaDestroyExternalMemory(fzbSVOCudaVar->extMem));
}

#endif