#pragma once

#include "./createSVO.cuh"

#ifndef CREATE_SVO_CU
#define CREATE_SVO_CU

//-------------------------------------------------------------------�˺���------------------------------------------------------------------
/*
__global__ void getSVONum(cudaTextureObject_t voxelGridMap, uint32_t* voxelNum, FzbSVONode* svoNodeArray, uint32_t svoDepth, FzbVoxelValue* svoVoxelValueArray) {

	//extern __shared__ int subSVONodeNum[];	//�����ڴ�����Ϊ�ȱ�������ͣ����Ƿ����ⲿ����
	glm::uvec3 voxelIndexU3 = glm::uvec3(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);

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
__global__ void getSVONum_UseShared(cudaTextureObject_t voxelGridMap, uint32_t* voxelNum, FzbSVONode* svoNodeArray, uint32_t svoDepth, FzbVoxelValue* svoVoxelValueArray) {

	__shared__ uint32_t svoVoxelValueIndex;
	__shared__ uint32_t voxelValueGroupNum;
	extern __shared__ FzbSVONode subSVONodeSharedArray[];	//�����ڴ�����Ϊ�ȱ�������ͣ����Ƿ����ⲿ���㡣��༸�������м�ڵ�

	uint3 voxelIndexU3 = make_uint3(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);
	uint32_t threadGroupIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	uint32_t laneIndex = threadGroupIndex % warpSize;	//һ��warp 32���߳�

	FzbSVONode initNode;
	if (threadGroupIndex == 0) {
		voxelValueGroupNum = 0;
	}
	if (threadGroupIndex < svoDepth + 69) {	//68 = 8 * 8 + 8 - 3
		subSVONodeSharedArray[threadGroupIndex] = initNode;
	}

	uint32_t valueU = tex3D<uint32_t>(voxelGridMap, voxelIndexU3.x, voxelIndexU3.y, voxelIndexU3.z);
	uint32_t hasValue = valueU > 0 ? 1 : 0;
	uint32_t warpVoxelNum = warpReduce(hasValue);
	if (laneIndex == 0)
		atomicAdd(&subSVONodeSharedArray[0].voxelNum, warpVoxelNum);
	__syncthreads();
	uint32_t blockVoxelNum = subSVONodeSharedArray[0].voxelNum;
	if (blockVoxelNum == 0)
		return;
	if (threadGroupIndex == 0)
		svoVoxelValueIndex = atomicAdd(voxelNum, blockVoxelNum);

	int curNodeIndex = 0;
	uint32_t sharedDataIndex = 0;
	int detailLevel = gridDim.x * blockDim.x;
	int subNodeIndex = 0;
	uint32_t sharedDataOffset = svoDepth - 4;
	for (int i = 0; i < svoDepth - 4; i++) {
		if (threadGroupIndex == 0)
			subSVONodeSharedArray[i].nodeIndex = curNodeIndex;

		uint3 index;
		index.x = voxelIndexU3.x & (detailLevel - 1);	//�൱��voxelIndexU3.x % detailLevel
		index.y = voxelIndexU3.y & (detailLevel - 1);
		index.z = voxelIndexU3.z & (detailLevel - 1);
		detailLevel /= 2;
		index.x /= detailLevel;
		index.y /= detailLevel;
		index.z /= detailLevel;

		subNodeIndex = index.y * 2 + index.x + index.z * 4;
		curNodeIndex = subNodeIndex + curNodeIndex * 8 + 1;

		if (threadGroupIndex == 0) {
			uint32_t hasSubNode = 1 << subNodeIndex;
			subSVONodeSharedArray[i].hasSubNode = hasSubNode;
		}
	}
	if (threadGroupIndex == 0) {
		subSVONodeSharedArray[sharedDataOffset].nodeIndex = curNodeIndex;
	}
	if (valueU > 0) {
		for (int i = 0; i < 2; i++) {

			uint3 index;
			index.x = voxelIndexU3.x & (detailLevel - 1);
			index.y = voxelIndexU3.y & (detailLevel - 1);
			index.z = voxelIndexU3.z & (detailLevel - 1);
			detailLevel /= 2;
			index.x /= detailLevel;
			index.y /= detailLevel;
			index.z /= detailLevel;

			subNodeIndex = index.y * 2 + index.x + index.z * 4;
			curNodeIndex = subNodeIndex + curNodeIndex * 8 + 1;

			uint32_t hasSubNode = 1 << subNodeIndex;
			atomicOr(&subSVONodeSharedArray[sharedDataIndex + sharedDataOffset].hasSubNode, hasSubNode);

			sharedDataIndex = sharedDataIndex * 8 + 1 + subNodeIndex;
			atomicAdd(&subSVONodeSharedArray[sharedDataIndex + sharedDataOffset].voxelNum, 1);
			subSVONodeSharedArray[sharedDataIndex + sharedDataOffset].nodeIndex = curNodeIndex;
		}

		uint3 index;
		index.x = voxelIndexU3.x & (detailLevel - 1);
		index.y = voxelIndexU3.y & (detailLevel - 1);
		index.z = voxelIndexU3.z & (detailLevel - 1);
		detailLevel /= 2;
		index.x /= detailLevel;
		index.y /= detailLevel;
		index.z /= detailLevel;

		subNodeIndex = index.y * 2 + index.x + index.z * 4;
		curNodeIndex = subNodeIndex + curNodeIndex * 8 + 1;

		uint32_t hasSubNode = 1 << subNodeIndex;
		atomicOr(&subSVONodeSharedArray[sharedDataIndex + sharedDataOffset].hasSubNode, hasSubNode);
		svoNodeArray[curNodeIndex].voxelNum = 1;
	}
	__syncthreads();

	if (valueU > 0) {
		uint32_t voxelValueGroupIndex = atomicAdd(&voxelValueGroupNum, 1);
		svoVoxelValueArray[svoVoxelValueIndex + voxelValueGroupIndex].pos_num = valueU;
		svoNodeArray[curNodeIndex].subsequentIndex = svoVoxelValueIndex + voxelValueGroupIndex;
	}

	if (threadGroupIndex < svoDepth - 3) {
		atomicAdd(&svoNodeArray[subSVONodeSharedArray[threadGroupIndex].nodeIndex].voxelNum, blockVoxelNum);
		atomicOr(&svoNodeArray[subSVONodeSharedArray[threadGroupIndex].nodeIndex].hasSubNode, subSVONodeSharedArray[threadGroupIndex].hasSubNode);
	}
	if (threadGroupIndex < 8) {
		sharedDataIndex = svoDepth - 3 + threadGroupIndex;
		if (subSVONodeSharedArray[sharedDataIndex].voxelNum > 0) {
			atomicAdd(&svoNodeArray[subSVONodeSharedArray[sharedDataIndex].nodeIndex].voxelNum, subSVONodeSharedArray[sharedDataIndex].voxelNum);
			atomicOr(&svoNodeArray[subSVONodeSharedArray[sharedDataIndex].nodeIndex].hasSubNode, subSVONodeSharedArray[sharedDataIndex].hasSubNode);
		}
	}
	if (threadGroupIndex < 64) {
		sharedDataIndex = svoDepth + 5 + threadGroupIndex;	//5 = 8 - 3
		if (subSVONodeSharedArray[sharedDataIndex].voxelNum > 0) {
			atomicAdd(&svoNodeArray[subSVONodeSharedArray[sharedDataIndex].nodeIndex].voxelNum, subSVONodeSharedArray[sharedDataIndex].voxelNum);
			atomicOr(&svoNodeArray[subSVONodeSharedArray[sharedDataIndex].nodeIndex].hasSubNode, subSVONodeSharedArray[sharedDataIndex].hasSubNode);
		}
	}
}

__global__ void getSVONum_step1(cudaTextureObject_t voxelGridMap, uint32_t svoDepth, uint32_t nonLeafNodeNum, uint32_t* voxelNum, FzbSVONode* svoNodeArray, FzbVoxelValue* svoVoxelValueArray) {

	__shared__ uint32_t groupVoxelNum;
	__shared__ uint32_t groupVoxelOffset;

	uint3 voxelIndexU3 = make_uint3(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);
	uint32_t threadGroupIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	uint32_t warpIndex = threadGroupIndex / warpSize;	//һ��warp 32���߳�
	uint32_t laneIndex = threadGroupIndex % warpSize;

	if (threadGroupIndex == 0) {
		groupVoxelNum = 0;
		groupVoxelOffset = 0;
	}
	__syncthreads();

	uint32_t valueU = tex3D<uint32_t>(voxelGridMap, voxelIndexU3.x, voxelIndexU3.y, voxelIndexU3.z);
	uint32_t voxelLocalIndex;
	if(valueU > 0)
		voxelLocalIndex = atomicAdd(&groupVoxelNum, 1);
	__syncthreads();
	if (threadGroupIndex == 0)
		groupVoxelOffset = atomicAdd(voxelNum, groupVoxelNum);
	__syncthreads();

	if (valueU > 0) {
		uint32_t voxelIndexU = packUint3(voxelIndexU3);
		svoNodeArray[voxelLocalIndex + groupVoxelOffset + nonLeafNodeNum].shuffleKey = voxelIndexU;	//����Ҷ�ӽڵ��λ�ã���ռ֮ǰ�Ĺ����ڵ��λ��
		svoVoxelValueArray[voxelLocalIndex + groupVoxelOffset].pos_num = valueU;
	}

	if (threadGroupIndex == 0) {
		atomicAdd(&svoNodeArray[0].voxelNum, groupVoxelNum);

		int curNodeIndex = 0;
		int detailLevel = gridDim.x * blockDim.x;
		int subNodeIndex = 0;
		uint32_t shuffleKey = 0;
		for (int i = 1; i < svoDepth - 3; i++) {

			uint3 index = make_uint3(voxelIndexU3.x & (detailLevel - 1), voxelIndexU3.y & (detailLevel - 1), voxelIndexU3.z & (detailLevel - 1));
			detailLevel /= 2;
			index.x /= detailLevel;
			index.y /= detailLevel;
			index.z /= detailLevel;
			subNodeIndex = index.y * 2 + index.x + index.z * 4;
			shuffleKey = (shuffleKey << 3) | subNodeIndex;

			uint32_t hasSubNode = 1 << subNodeIndex;
			if (i < svoDepth - 4)
				atomicOr(&svoNodeArray[curNodeIndex].hasSubNode, hasSubNode);

			curNodeIndex = subNodeIndex + curNodeIndex * 8 + 1;
			atomicAdd(&svoNodeArray[curNodeIndex].voxelNum, groupVoxelNum);
			svoNodeArray[curNodeIndex].shuffleKey = shuffleKey;
		}
	}

}
__global__ void getSVONum_step2(uint32_t voxelNum, uint32_t svoDepth, uint32_t nonLeafNodeNum, uint32_t svoSize, FzbSVONode* svoNodeArray) {

	uint32_t localThreadIndex = threadIdx.x;
	uint32_t globalThreadIndex = blockIdx.x * blockDim.x + localThreadIndex;
	if (globalThreadIndex >= voxelNum)
		return;

	uint32_t voxelIndex = globalThreadIndex + nonLeafNodeNum;
	FzbSVONode voxelIndexU3Info = svoNodeArray[voxelIndex];
	uint3 voxelIndexU3 = unpackUint(voxelIndexU3Info.shuffleKey);

	int curNodeIndex = 0;
	int detailLevel = svoSize;
	int subNodeIndex = 0;
	uint32_t shuffleKey = 0;
	for (int i = 0; i < svoDepth - 4; i++) {
		uint3 index = make_uint3(voxelIndexU3.x & (detailLevel - 1), voxelIndexU3.y & (detailLevel - 1), voxelIndexU3.z & (detailLevel - 1));
		detailLevel /= 2;
		index.x /= detailLevel;
		index.y /= detailLevel;
		index.z /= detailLevel;
		subNodeIndex = index.y * 2 + index.x + index.z * 4;
		shuffleKey = (shuffleKey << 3) | subNodeIndex;
		curNodeIndex = subNodeIndex + curNodeIndex * 8 + 1;
	}

	for (int i = 0; i < 2; i++) {
		uint3 index = make_uint3(voxelIndexU3.x & (detailLevel - 1), voxelIndexU3.y & (detailLevel - 1), voxelIndexU3.z & (detailLevel - 1));
		detailLevel /= 2;
		index.x /= detailLevel;
		index.y /= detailLevel;
		index.z /= detailLevel;

		subNodeIndex = index.y * 2 + index.x + index.z * 4;
		shuffleKey = (shuffleKey << 3) | subNodeIndex;
		uint32_t hasSubNode = 1 << subNodeIndex;
		atomicOr(&svoNodeArray[curNodeIndex].hasSubNode, hasSubNode);

		curNodeIndex = subNodeIndex + curNodeIndex * 8 + 1;

		atomicAdd(&svoNodeArray[curNodeIndex].voxelNum, 1);
		svoNodeArray[curNodeIndex].shuffleKey = shuffleKey;
	}
	uint3 index = make_uint3(voxelIndexU3.x & (detailLevel - 1), voxelIndexU3.y & (detailLevel - 1), voxelIndexU3.z & (detailLevel - 1));
	detailLevel /= 2;
	index.x /= detailLevel;
	index.y /= detailLevel;
	index.z /= detailLevel;
	subNodeIndex = index.y * 2 + index.x + index.z * 4;
	shuffleKey = (shuffleKey << 3) | subNodeIndex;

	uint32_t hasSubNode = 1 << subNodeIndex;
	atomicOr(&svoNodeArray[curNodeIndex].hasSubNode, hasSubNode);

	curNodeIndex = subNodeIndex + curNodeIndex * 8 + 1;
	svoNodeArray[curNodeIndex].voxelNum = 1;
	svoNodeArray[curNodeIndex].shuffleKey = shuffleKey;
	svoNodeArray[curNodeIndex].subsequentIndex = globalThreadIndex;
}

__global__ void compressSVO(FzbSVONode* nodeArray, FzbSVONode* nodePool, uint32_t nodeStartIndex, uint32_t subArrayStartIndex, uint32_t* subArrayNum, glm::vec4 fatherNodePos_Size, uint32_t nonLeafNodeNum) {

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

	node.nodePos_Size = nodePos_Size;

	int nodePoolIndex = subArrayStartIndex + threadIndex;
	nodePool[nodePoolIndex] = node;

	if (nodeStartIndex >= nonLeafNodeNum)
		return;
	int subArrayIndex = atomicAdd(subArrayNum, 1);
	nodePool[nodePoolIndex].subsequentIndex = subArrayIndex * 8;
	compressSVO << <1, 8 >> > (nodeArray, nodePool, nodeIndex * 8 + 1, subArrayIndex * 8, subArrayNum, nodePos_Size, nonLeafNodeNum);

}
*/
/*
getSVONum_step1�ҵ�������ֵ�����أ�������������svoNodeArray��Ҷ�ڵ��shuffleKey�У�Ȼ��Ը��߳���Ĺ����ڵ㸳ֵ
getSVONum_step2���̶߳�Ӧ����ֵ���أ��ҵ������м�ڵ㲢��ֵ�����յõ����˲�������
compressSVO_Step1����ִ�С�ʹ��ÿһ����ÿ���߳���������
compressSVO_Step2ͬ������ִ�У����߳����������յõ������ѹ�����顣
*/
__global__ void getSVONum_step1(cudaTextureObject_t voxelGridMap, uint32_t svoDepth, uint32_t nonLeafNodeNum, uint32_t* voxelNum, FzbSVONode* svoNodeArray, FzbVoxelValue* svoVoxelValueArray) {

	__shared__ uint32_t groupVoxelNum;
	__shared__ uint32_t groupVoxelOffset;

	uint3 voxelIndexU3 = make_uint3(blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y, blockDim.z * blockIdx.z + threadIdx.z);
	uint32_t threadGroupIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
	uint32_t warpIndex = threadGroupIndex / warpSize;	//һ��warp 32���߳�
	uint32_t laneIndex = threadGroupIndex % warpSize;

	if (threadGroupIndex == 0) {
		groupVoxelNum = 0;
		groupVoxelOffset = 0;
	}
	__syncthreads();

	uint32_t valueU = tex3D<uint32_t>(voxelGridMap, voxelIndexU3.x, voxelIndexU3.y, voxelIndexU3.z);
	uint32_t voxelLocalIndex;
	if (valueU > 0)
		voxelLocalIndex = atomicAdd(&groupVoxelNum, 1);
	__syncthreads();
	if (threadGroupIndex == 0)
		groupVoxelOffset = atomicAdd(voxelNum, groupVoxelNum);
	__syncthreads();

	if (valueU > 0) {
		uint32_t voxelIndexU = packUint3(voxelIndexU3);
		svoNodeArray[voxelLocalIndex + groupVoxelOffset + nonLeafNodeNum].shuffleKey = voxelIndexU;	//����Ҷ�ӽڵ��λ�ã���ռ֮ǰ�Ĺ����ڵ��λ��
		svoVoxelValueArray[voxelLocalIndex + groupVoxelOffset].pos_num = valueU;
	}

	if (threadGroupIndex == 0 && groupVoxelNum > 0) {
		atomicAdd(&svoNodeArray[0].voxelNum, groupVoxelNum);

		int curNodeIndex = 0;
		int detailLevel = gridDim.x * blockDim.x;
		int subNodeIndex = 0;
		for (int i = 1; i < svoDepth - 3; i++) {

			uint3 index = make_uint3(voxelIndexU3.x & (detailLevel - 1), voxelIndexU3.y & (detailLevel - 1), voxelIndexU3.z & (detailLevel - 1));
			detailLevel /= 2;
			index.x /= detailLevel;
			index.y /= detailLevel;
			index.z /= detailLevel;
			subNodeIndex = index.y * 2 + index.x + index.z * 4;

			uint32_t hasSubNode = 1 << subNodeIndex;
			if (i < svoDepth - 4)
				atomicOr(&svoNodeArray[curNodeIndex].hasSubNode, hasSubNode);

			curNodeIndex = subNodeIndex + curNodeIndex * 8 + 1;
			atomicAdd(&svoNodeArray[curNodeIndex].voxelNum, groupVoxelNum);
		}
	}

}
__global__ void getSVONum_step2(uint32_t voxelNum, uint32_t svoDepth, uint32_t nonLeafNodeNum, uint32_t svoSize, FzbSVONode* svoNodeArray) {

	uint32_t localThreadIndex = threadIdx.x;
	uint32_t globalThreadIndex = blockIdx.x * blockDim.x + localThreadIndex;
	if (globalThreadIndex >= voxelNum)
		return;

	uint32_t voxelIndex = globalThreadIndex + nonLeafNodeNum;
	FzbSVONode voxelIndexU3Info = svoNodeArray[voxelIndex];
	uint3 voxelIndexU3 = unpackUint(voxelIndexU3Info.shuffleKey);

	int curNodeIndex = 0;
	int detailLevel = svoSize;
	int subNodeIndex = 0;
	for (int i = 0; i < svoDepth - 4; i++) {
		uint3 index = make_uint3(voxelIndexU3.x & (detailLevel - 1), voxelIndexU3.y & (detailLevel - 1), voxelIndexU3.z & (detailLevel - 1));
		detailLevel /= 2;
		index.x /= detailLevel;
		index.y /= detailLevel;
		index.z /= detailLevel;
		subNodeIndex = index.y * 2 + index.x + index.z * 4;
		curNodeIndex = subNodeIndex + curNodeIndex * 8 + 1;
	}

	for (int i = 0; i < 2; i++) {
		uint3 index = make_uint3(voxelIndexU3.x & (detailLevel - 1), voxelIndexU3.y & (detailLevel - 1), voxelIndexU3.z & (detailLevel - 1));
		detailLevel /= 2;
		index.x /= detailLevel;
		index.y /= detailLevel;
		index.z /= detailLevel;

		subNodeIndex = index.y * 2 + index.x + index.z * 4;
		uint32_t hasSubNode = 1 << subNodeIndex;
		atomicOr(&svoNodeArray[curNodeIndex].hasSubNode, hasSubNode);

		curNodeIndex = subNodeIndex + curNodeIndex * 8 + 1;

		atomicAdd(&svoNodeArray[curNodeIndex].voxelNum, 1);
	}
	uint3 index = make_uint3(voxelIndexU3.x & (detailLevel - 1), voxelIndexU3.y & (detailLevel - 1), voxelIndexU3.z & (detailLevel - 1));
	detailLevel /= 2;
	index.x /= detailLevel;
	index.y /= detailLevel;
	index.z /= detailLevel;
	subNodeIndex = index.y * 2 + index.x + index.z * 4;

	uint32_t hasSubNode = 1 << subNodeIndex;
	atomicOr(&svoNodeArray[curNodeIndex].hasSubNode, hasSubNode);

	curNodeIndex = subNodeIndex + curNodeIndex * 8 + 1;
	svoNodeArray[curNodeIndex].voxelNum = 1;
	svoNodeArray[curNodeIndex].label = globalThreadIndex;
}

template <int type>
__global__ void compressSVO_Step1(FzbSVONode* nodeArray, FzbSVONode* tempNodePool, FzbSVONode* nodePool, uint32_t* subArrayNum, FzbNodePoolBlock* threadBlockInfos, uint32_t nodeStartIndex,uint32_t svoDepth) {
	__shared__ uint64_t blockHasValue;	//ÿ���˸��߳�һ��Ԫ��, 8x64 = 512��һ���߳������512���߳�
	__shared__ uint32_t blockIndexOfGroup;
	__shared__ uint32_t nodeNum;
	__shared__ uint32_t blockNodeNum[64];

	uint32_t threadLocalIndex = threadIdx.x;
	uint32_t threadGlobalIndex = blockDim.x * blockIdx.x + threadLocalIndex;
	uint32_t blockIndexInGroup = threadLocalIndex / 8;
	uint32_t threadOffsetInBlock = threadLocalIndex & 7;
	if (threadLocalIndex == 0) {
		blockHasValue = 0;
		blockIndexOfGroup = 0;
		nodeNum = 0;
	}
	if (threadGlobalIndex < 64)
		blockNodeNum[threadGlobalIndex] = 0;
	__syncthreads();

	uint32_t nodeIndex = threadGlobalIndex + nodeStartIndex;
	FzbSVONode node = nodeArray[nodeIndex];
	FzbSVONode fatherNode = nodeArray[nodeIndex / 8];
	if (node.voxelNum > 0) {	//ǰ���64��512����Ҫ
		if (type == 2) {
			atomicAdd(&nodeNum, 1);
		}
		atomicAdd(&blockNodeNum[blockIndexInGroup], 1);
	}
	if (fatherNode.voxelNum > 0) {
		if (threadOffsetInBlock == 0)
			atomicOr(&blockHasValue, uint64_t(1) << blockIndexInGroup);
	}
	__syncthreads();

	if (threadLocalIndex == 0) {
		uint32_t blockNum = __popcll(blockHasValue);
		blockIndexOfGroup = atomicAdd(subArrayNum, blockNum);	//�߳�����ȫ�ֿ��������ʼ����
		if (type == 2)	//ǰ���64��512����Ҫ
			threadBlockInfos[blockIdx.x] = FzbNodePoolBlock(blockIndexOfGroup, blockNum, nodeNum);
	}
	__syncthreads();

	uint32_t blockIndex = __popcll(blockHasValue & ((uint64_t(1) << blockIndexInGroup) - 1)) + blockIndexOfGroup;	//�߳�����block��ȫ���е�����
	if (node.voxelNum > 0) {
		uint32_t label = 0;
		for (int i = 0; i < blockIndexInGroup; i++) {
			label += blockNodeNum[i];
		}
		node.label = __popcll(fatherNode.hasSubNode & ((1u << threadOffsetInBlock) - 1)) + label + 1;
		//uint32_t blockIndexInWarp = threadLocalIndex & 3;
		//uint32_t offset = 8 * blockIndexInWarp;
		//node.label = __popc(((__ballot_sync(__activemask(), 1) & (0xFF << offset)) >> offset) & ((1u << threadOffsetInBlock) - 1)) + label + 1;
		if (type == 2) {
			tempNodePool[1 + blockIndex * 8 + threadOffsetInBlock] = node;
		}
		else {
			nodePool[1 + blockIndex * 8 + threadOffsetInBlock] = node;
		}
	}
	if (blockNodeNum[blockIndexInGroup] > 0) {
		if (type == 2) {
			tempNodePool[1 + blockIndex * 8 + threadOffsetInBlock].shuffleKey = threadGlobalIndex | (svoDepth << 28);
		}
		else {
			nodePool[1 + blockIndex * 8 + threadOffsetInBlock].shuffleKey = threadGlobalIndex | (svoDepth << 28);
		}
	}

	//�����ڵ�͵�һ��ڵ����
	if (type == 0) {
		FzbSVONode rootNode = nodeArray[0];
		rootNode.label = 1;
		if (threadGlobalIndex == 0)
			nodePool[0] = rootNode;
		if (threadGlobalIndex < 8) {
			FzbSVONode node = nodeArray[threadGlobalIndex + 1];
			node.shuffleKey = threadGlobalIndex | (uint32_t(1) << 28);
			node.label = __popc(rootNode.hasSubNode & ((1u << threadGlobalIndex) - 1)) + 1;
			nodePool[threadGlobalIndex + 1] = node;
		}

	}
}

__global__ void compressSVO_Step2(FzbNodePoolBlock* threadBlockInfos, FzbSVONode* tempNodePool, FzbSVONode* nodePool) {
	__shared__ FzbNodePoolBlock threadBlockInfo;
	__shared__ uint32_t firstBlockIndex;

	uint32_t threadLocalIndex = threadIdx.x;
	uint32_t threadGlobalIndex = blockDim.x * blockIdx.x + threadLocalIndex;

	if (threadLocalIndex == 0) {
		threadBlockInfo = threadBlockInfos[blockIdx.x];
		firstBlockIndex = threadBlockInfos[0].startIndex;
	}
	__syncthreads();

	if (threadLocalIndex >= threadBlockInfo.blockNum * 8)
		return;

	uint32_t nodeIndex = threadBlockInfo.startIndex * 8 + threadLocalIndex + 1;
	FzbSVONode node = tempNodePool[nodeIndex];
	uint32_t newNodeIndex = firstBlockIndex * 8 + threadLocalIndex + 1;
	uint32_t label = 0;
	for (int i = 0; i < blockIdx.x; i++) {
		FzbNodePoolBlock blockInfo = threadBlockInfos[i];
		label += blockInfo.nodeNum;
		newNodeIndex += blockInfo.blockNum * 8;
	}
	node.label += label;
	nodePool[newNodeIndex] = node;
}

//-------------------------------------------------------------------------------------------------------------------------
/*
void CUDART_CB cleanTempData(cudaStream_t stream, cudaError_t status, void* userData) {

	SVOCuda* svoCuda = (SVOCuda*)userData;

	CHECK(cudaDestroyExternalSemaphore(svoCuda->extVgmSemaphore));
	CHECK(cudaDestroyExternalSemaphore(svoCuda->extSvoSemaphore));
	CHECK(cudaDestroyTextureObject(svoCuda->vgm));
	CHECK(cudaFreeMipmappedArray(svoCuda->vgmMipmap));
	CHECK(cudaDestroyExternalMemory(svoCuda->vgmExtMem));
	CHECK(cudaDestroyExternalMemory(svoCuda->nodePoolExtMem));
	CHECK(cudaDestroyExternalMemory(svoCuda->voxelValueArrayExtMem));

	CHECK(cudaFreeHost(svoCuda->voxelNum));
	CHECK(cudaFreeHost(svoCuda->nodeArrayNum));
	CHECK(cudaFreeHost(svoCuda->subArrayNum_host));

	CHECK(cudaFreeAsync(svoCuda->voxelNum_p, stream));
	CHECK(cudaFreeAsync(svoCuda->svoNodeArray, stream));
	CHECK(cudaFreeAsync(svoCuda->subArrayNum, stream));

	CHECK(cudaFreeAsync(svoCuda->nodePool, stream));
	CHECK(cudaFreeAsync(svoCuda->svoVoxelValueArray, stream));

	CHECK(cudaStreamDestroy(svoCuda->stream));

}
*/
void SVOCuda::createSVOCuda(VkPhysicalDevice vkPhysicalDevice, FzbImage& voxelGridMap, HANDLE vgmSemaphoreHandle, HANDLE svoSemaphoreHandle, bool isPresent) {

	double start = cpuSecond();

	unsigned long long size = voxelGridMap.width * voxelGridMap.height * voxelGridMap.depth * sizeof(uint32_t);
	fromVulkanImageToCudaTexture(vkPhysicalDevice, voxelGridMap, voxelGridMap.handle, size, false, vgmExtMem, vgmMipmap, vgm);

	extVgmSemaphore = importVulkanSemaphoreObjectFromNTHandle(vgmSemaphoreHandle);
	extSvoSemaphore = importVulkanSemaphoreObjectFromNTHandle(svoSemaphoreHandle);

	dim3 gridSize(voxelGridMap.width / 8, voxelGridMap.height / 8, voxelGridMap.depth / 8);
	dim3 blockSize(8, 8, 8);
	//���SVO�����
	uint32_t svoDepth = 1;
	uint32_t vgmSize = voxelGridMap.width;
	while (vgmSize > 1) {
		svoDepth++;
		vgmSize >>= 1;
	}
	uint32_t maxNodeNum = uint32_t((pow(8, svoDepth) - 1) / 7);	//���˲����������ڵ���
	uint32_t nonLeafNodeNum = uint32_t((pow(8, svoDepth - 1) - 1) / 7);

	CHECK(cudaStreamCreate(&stream));
	uint32_t* voxelNum_p;
	CHECK(cudaMalloc((void**)&voxelNum_p, sizeof(uint32_t)));	//��ʹ�ù̶��ڴ棬���޷�ʹ��ԭ������
	CHECK(cudaMemset(voxelNum_p, 0, sizeof(uint32_t)));

	//����һ�����˲������飬�����ŷ�
	FzbSVONode* svoNodeArray;
	CHECK(cudaMalloc((void**)&svoNodeArray, sizeof(FzbSVONode) * maxNodeNum));
	CHECK(cudaMemset(svoNodeArray, 0, sizeof(FzbSVONode) * maxNodeNum));
	//����һ����ֵ�������ݵ�����
	CHECK(cudaMalloc((void**)&svoVoxelValueArray, sizeof(FzbVoxelValue) * voxelGridMap.width * voxelGridMap.height * voxelGridMap.depth));
	CHECK(cudaMemset(svoVoxelValueArray, 0, sizeof(FzbVoxelValue) * voxelGridMap.width * voxelGridMap.height * voxelGridMap.depth));

	uint32_t subArrayNum_host = 1;
	uint32_t* subArrayNum;
	CHECK(cudaMalloc((void**)&subArrayNum, sizeof(uint32_t)));
	CHECK(cudaMemcpy(subArrayNum, &subArrayNum_host, sizeof(uint32_t), cudaMemcpyHostToDevice));

	FzbSVONode* tempNodePool;
	CHECK(cudaMalloc((void**)&tempNodePool, sizeof(FzbSVONode) * maxNodeNum));
	CHECK(cudaMemset(tempNodePool, 0, sizeof(FzbSVONode) * maxNodeNum));

	CHECK(cudaMalloc((void**)&nodePool, sizeof(FzbSVONode) * maxNodeNum));
	CHECK(cudaMemset(nodePool, 0, sizeof(FzbSVONode) * maxNodeNum));

	FzbNodePoolBlock** threadBlockInfos = (FzbNodePoolBlock**)malloc(sizeof(FzbNodePoolBlock*) * (svoDepth - 4));
	for (int i = 0; i < svoDepth - 4; i++) {
		CHECK(cudaMalloc((void**)&threadBlockInfos[i], sizeof(FzbNodePoolBlock) * pow(8, i + 4) / 512));
	}
	waitExternalSemaphore(extVgmSemaphore, stream);
	
	getSVONum_step1 << < gridSize, blockSize, 0, stream >> > (vgm, svoDepth, nonLeafNodeNum, voxelNum_p, svoNodeArray, svoVoxelValueArray);
	//����ѹ�����������������
	CHECK(cudaMemcpy(&voxelNum, voxelNum_p, sizeof(uint32_t), cudaMemcpyDeviceToHost));
	uint32_t blockNum = std::ceil((float)voxelNum / 512);
	getSVONum_step2 << < blockNum, 512, 0, stream >> > (voxelNum, svoDepth, nonLeafNodeNum, voxelGridMap.width, svoNodeArray);

	for (int i = 0; i < svoDepth - 2; i++) {
		uint32_t nodeStartIndex = uint32_t((pow(8, i + 2) - 1) / 7);
		if (i == 0) {
			compressSVO_Step1<0> << <1, 64 >> > (svoNodeArray, tempNodePool, nodePool, subArrayNum, nullptr, nodeStartIndex, 2);
		}
		else if (i == 1) {
			compressSVO_Step1<1> << <1, 512 >> > (svoNodeArray, tempNodePool, nodePool, subArrayNum, nullptr, nodeStartIndex, 3);
		}
		else {
			uint32_t gridSize = pow(8, i + 2) / 512;
			compressSVO_Step1<2> << <gridSize, 512 >> > (svoNodeArray, tempNodePool, nodePool, subArrayNum, threadBlockInfos[i - 2], nodeStartIndex, i+2);
			compressSVO_Step2 << <gridSize, 1024 >> > (threadBlockInfos[i - 2], tempNodePool, nodePool);
		}
	}
	CHECK(cudaMemcpy(&nodeBlockNum, subArrayNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));

	CHECK(cudaDestroyExternalSemaphore(extVgmSemaphore));
	CHECK(cudaDestroyExternalSemaphore(extSvoSemaphore));
	CHECK(cudaDestroyTextureObject(vgm));
	CHECK(cudaFreeMipmappedArray(vgmMipmap));
	CHECK(cudaDestroyExternalMemory(vgmExtMem));

	CHECK(cudaFree(voxelNum_p));
	CHECK(cudaFree(svoNodeArray));
	CHECK(cudaFree(subArrayNum));
	CHECK(cudaFree(tempNodePool));

	std::cout << cpuSecond() - start << std::endl;

}

void SVOCuda::getSVOCuda(VkPhysicalDevice vkPhysicalDevice, HANDLE nodePoolHandle, HANDLE voxelValueArrayHandle) {
	//���ж��Ƿ���ͬһ�������豸
	if (getCudaDeviceForVulkanPhysicalDevice(vkPhysicalDevice) == cudaInvalidDeviceId) {
		throw std::runtime_error("CUDA��Vulkan�õĲ���ͬһ��GPU������");
	}

	nodePoolExtMem = importVulkanMemoryObjectFromNTHandle(nodePoolHandle, sizeof(FzbSVONode) * (8 * nodeBlockNum + 1), false);
	FzbSVONode* vkNodePool = (FzbSVONode*)mapBufferOntoExternalMemory(nodePoolExtMem, 0, sizeof(FzbSVONode) * (8 * nodeBlockNum + 1));
	CHECK(cudaMemcpy(vkNodePool, this->nodePool, sizeof(FzbSVONode) * (8 * nodeBlockNum + 1), cudaMemcpyDeviceToDevice));

	voxelValueArrayExtMem = importVulkanMemoryObjectFromNTHandle(voxelValueArrayHandle, sizeof(FzbVoxelValue) * voxelNum, false);
	FzbVoxelValue* vkVoxelValueArray = (FzbVoxelValue*)mapBufferOntoExternalMemory(voxelValueArrayExtMem, 0, sizeof(FzbVoxelValue) * voxelNum);
	CHECK(cudaMemcpy(vkVoxelValueArray, this->svoVoxelValueArray, sizeof(FzbVoxelValue) * voxelNum, cudaMemcpyDeviceToDevice));

	signalExternalSemaphore(extSvoSemaphore, stream);

	//CHECK(cudaStreamAddCallback(stream, cleanTempData, this, 0));
	CHECK(cudaDestroyExternalMemory(nodePoolExtMem));
	CHECK(cudaDestroyExternalMemory(voxelValueArrayExtMem));

	CHECK(cudaFree(nodePool));
	CHECK(cudaFree(svoVoxelValueArray));

	CHECK(cudaStreamDestroy(stream));
}

void SVOCuda::clean() {

}

#endif