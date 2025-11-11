#include "FzbSVOCuda_PG.cuh"

/*
template<bool notOnlyOneBlock>
__global__ void compressSVO_PG_firstStep(FzbSVONodeData_PG* SVONodes, FzbSVONodeData_PG* SVOFatherNodes, FzbSVONodeBlock* blockInfo,
	FzbSVONodeTempInfo* SVONodeTempInfos, uint32_t* SVONodeCount, uint32_t nodeCount) {
	__shared__ uint64_t groupHasDataNodeBlockMask;	//一个线程组512个node，每个8个node为一组，共有64组，groupHasDataNodeBlock每一位表示一组有无值
	__shared__ uint32_t groupHasDataAndDivisibleNodeCountInWarp[16];	//每个warp中有值且可分node的数量
	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadIndex >= nodeCount) return;
	uint32_t blockIndexInGroup = threadIdx.x / 8;
	uint32_t laneInBlock = threadIdx.x & 7;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;
	uint32_t firstBlockLaneInWarp = (blockIndexInGroup & 3) * 8;
	if (threadIdx.x == 0) groupHasDataNodeBlockMask = 0;
	if (threadIdx.x < 16) groupHasDataAndDivisibleNodeCountInWarp[threadIdx.x] = 0;
	__syncthreads();

	FzbSVONodeData_PG nodeData = SVONodes[threadIndex];
	//nodeData.shuffleKey = threadIndex;
	bool hasData = !SVOFatherNodes[threadIndex / 8].indivisible && glm::length(nodeData.irradiance) != 0.0f;	//如果父节点不可分，则不用存子节点
	uint32_t blockHasData = hasData;
	for (int offset = 4; offset > 0; offset /= 2) 	//只要一个node有值，这个nodeBlock就有值
		blockHasData |= __shfl_down_sync(0xFFFFFFFF, blockHasData, offset);
	uint32_t warpHasDataBlockMask = blockHasData;	//这个warp中有值nodeBlock的mask
	warpHasDataBlockMask |= __shfl_sync(0xFFFFFFFF, blockHasData, 8) << 1;
	warpHasDataBlockMask |= __shfl_sync(0xFFFFFFFF, blockHasData, 16) << 2;
	warpHasDataBlockMask |= __shfl_sync(0xFFFFFFFF, blockHasData, 24) << 3;

	bool hasDataAndDivisible = hasData && !nodeData.indivisible;
	uint32_t warpHasDataAndDivisibleNodeMask = (uint32_t)hasDataAndDivisible << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpHasDataAndDivisibleNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpHasDataAndDivisibleNodeMask, offset);
	warpHasDataAndDivisibleNodeMask = __shfl_sync(0xFFFFFFFF, warpHasDataAndDivisibleNodeMask, 0);

	if (warpLane == 0) {
		atomicOr(&groupHasDataNodeBlockMask, (uint64_t)warpHasDataBlockMask << (warpIndex * 4));
		groupHasDataAndDivisibleNodeCountInWarp[warpIndex] = __popc(warpHasDataAndDivisibleNodeMask);
	}
	__syncthreads();

	if (gridDim.x == 8 && threadIndex == 28) printf("%u\n", SVOFatherNodes[threadIndex / 8].indivisible);

	if (threadIdx.x == 0) atomicAdd(SVONodeCount, __popcll(groupHasDataNodeBlockMask) * 8);
	if constexpr (notOnlyOneBlock) {
		if (warpIndex == 0) {
			uint32_t blockHasDataAndDivisibleNodeTotalCount = threadIdx.x < 16 ? groupHasDataAndDivisibleNodeCountInWarp[threadIdx.x] : 0;
			for (int offset = 16; offset > 0; offset /= 2)
				blockHasDataAndDivisibleNodeTotalCount += __shfl_down_sync(0xFFFFFFFF, blockHasDataAndDivisibleNodeTotalCount, offset);
			if (threadIdx.x == 0) {
				blockInfo[blockIdx.x].nodeCount = blockHasDataAndDivisibleNodeTotalCount;	//每个线程组有多少有值且可分node
				blockInfo[blockIdx.x].blockCount = __popcll(groupHasDataNodeBlockMask);	//每个线程组有多少有值nodeBlock
			}
		}
	}

	if ((groupHasDataNodeBlockMask & (((uint64_t)15) << (warpIndex * 4))) == 0) return; //若warp中没有有值node，则无需任何操作

	uint32_t label = warpLane < warpIndex ? groupHasDataAndDivisibleNodeCountInWarp[warpLane] : 0;
	for (int offset = 16; offset > 0; offset /= 2)
		label += __shfl_down_sync(0xFFFFFFFF, label, offset);
	label = __shfl_sync(0xFFFFFFFF, label, 0);
	label += __popc(warpHasDataAndDivisibleNodeMask << (32 - warpLane));	//加上warp中前面有几个有值node，得到最后自身在线程组中是第几个有值且可分node

	uint32_t nodeIndex = __popcll(groupHasDataNodeBlockMask << (64 - blockIndexInGroup)) * 8;
	nodeIndex += laneInBlock;
	//if (gridDim.x == 64 && blockDim.x == 512 && blockIdx.x == 0 && threadIdx.x == 393) printf("%u\n", nodeIndex);

	if (hasDataAndDivisible) nodeData.label = label + 1;	//label从1开始

	if constexpr (notOnlyOneBlock) {
		FzbSVONodeTempInfo tempInfo;
		tempInfo.hasData = hasData;
		if (hasData) {
			tempInfo.nodeData = nodeData;
			tempInfo.nodeIndexInThreadBlock = nodeIndex;
		}
		SVONodeTempInfos[threadIndex] = tempInfo;
	}
	if constexpr (!notOnlyOneBlock) {
		if (hasData) SVONodes[nodeIndex] = nodeData;
	}
}
__global__ void compressSVO_PG_secondStep(FzbSVONodeData_PG* SVONodes, FzbSVONodeBlock* blockInfo,
	FzbSVONodeTempInfo* SVONodeTempInfos, uint32_t nodeCount) {
	extern __shared__ uint32_t groupData[];	//前一半表示warp中有值且可分的node数量；后一半表示warp中有值nodeBlock的数量。最多32个
	__shared__ uint32_t groupHasDataAndDivisibleNodeStartIndex;		//前面线程组中有值且可分的node的数量
	__shared__ uint32_t groupHasDataNodeBlockStartIndex;			//前面线程组中有值的nodeBlock的数量

	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadIndex >= nodeCount) return;
	//uint32_t maxNeedWarpCount = (gridDim.x + 31) / 32;		//计算前面的线程组的数据最多需要多少个warp
	uint32_t actualNeedWarpCount = (blockIdx.x + 31) / 32;	//计算前面的线程组的数据实际需要多少个warp
	if (threadIdx.x < actualNeedWarpCount * 2) groupData[threadIdx.x] = 0;
	if (threadIdx.x == 0) {
		groupHasDataAndDivisibleNodeStartIndex = 0;
		groupHasDataNodeBlockStartIndex = 0;
	}
	__syncthreads();
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;

	if (warpIndex < actualNeedWarpCount) {
		uint32_t hasDataAndDivisbleNodeCount = threadIdx.x < blockIdx.x ? blockInfo[threadIdx.x].nodeCount : 0;
		uint32_t blockStartIndex = threadIdx.x < blockIdx.x ? blockInfo[threadIdx.x].blockCount : 0;
		for (int offset = 16; offset > 0; offset /= 2) {
			hasDataAndDivisbleNodeCount += __shfl_down_sync(0xFFFFFFFF, hasDataAndDivisbleNodeCount, offset);
			blockStartIndex += __shfl_down_sync(0xFFFFFFFF, blockStartIndex, offset);
		}
		if (warpLane == 0) {
			groupData[warpIndex] = hasDataAndDivisbleNodeCount;
			groupData[warpIndex + actualNeedWarpCount] = blockStartIndex;
		}
	}
	__syncthreads();

	if (warpIndex == 0) {
		uint32_t hasDataAndDivisibleNodeStartIndex = threadIdx.x < actualNeedWarpCount ? groupData[threadIdx.x] : 0;
		uint32_t hasDataNodeBlockCount = threadIdx.x < actualNeedWarpCount ? groupData[threadIdx.x + actualNeedWarpCount] : 0;
		for (int offset = 16; offset > 0; offset /= 2) {
			hasDataAndDivisibleNodeStartIndex += __shfl_down_sync(0xFFFFFFFF, hasDataAndDivisibleNodeStartIndex, offset);
			hasDataNodeBlockCount += __shfl_down_sync(0xFFFFFFFF, hasDataNodeBlockCount, offset);
		}
		if (warpLane == 0) {
			groupHasDataAndDivisibleNodeStartIndex = hasDataAndDivisibleNodeStartIndex;
			groupHasDataNodeBlockStartIndex = hasDataNodeBlockCount;
		}
	}
	__syncthreads();

	if (SVONodeTempInfos[threadIndex].hasData) {
		uint32_t nodeIndex = groupHasDataNodeBlockStartIndex * 8 + SVONodeTempInfos[threadIndex].nodeIndexInThreadBlock;
		FzbSVONodeData_PG nodeData = SVONodeTempInfos[threadIndex].nodeData;
		if(!nodeData.indivisible) nodeData.label += groupHasDataAndDivisibleNodeStartIndex;
		SVONodes[nodeIndex] = nodeData;
	}
}
*/
/*
template<bool notOnlyOneBlock>
__global__ void compressSVO_PG_TopToBottom_firstStep(FzbSVONodeData_PG* SVONodes,
	const uint32_t* __restrict__ hasDataAndDivisibleFatherNodeIndices, const uint32_t* __restrict__ hasDataAndDivisibleFatherNodeCount,
	FzbSVONodeBlock* blockInfo, FzbSVONodeTempInfo* SVONodeTempInfos,
	uint32_t* hasDataAndDivisibleNodeIndices, uint32_t* hasDataAndDivisibleNodeCount) {
	extern __shared__ uint32_t groupHasDataAndDivisibleNodeCountInWarp[];	//每个warp中有值且可分node的数量
	__shared__ uint32_t groupNodeCount;
	__shared__ uint64_t groupHasDataNodeBlockMask;	//一个线程组512个node，每个8个node为一组，共有64组，groupHasDataNodeBlock每一位表示一组有无值
	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t warpCount = (blockDim.x + 31) / 32;
	if (threadIdx.x == 0) {
		groupNodeCount = *hasDataAndDivisibleFatherNodeCount * 8;
		groupHasDataNodeBlockMask = 0;
	}
	if (threadIdx.x < warpCount) groupHasDataAndDivisibleNodeCountInWarp[threadIdx.x] = 0;
	__syncthreads();
	if (threadIndex >= groupNodeCount) return;
	uint32_t blockIndexInGroup = threadIdx.x / 8;
	uint32_t laneInBlock = threadIdx.x & 7;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;

	uint32_t fatherNodeIndex = hasDataAndDivisibleFatherNodeIndices[threadIndex / 8];
	uint32_t nodeIndex = fatherNodeIndex * 8 + laneInBlock;
	FzbSVONodeData_PG nodeData = SVONodes[nodeIndex];
	bool hasData = glm::length(nodeData.irradiance) != 0.0f;	//父节点必然是可分的
	bool hasDataAndDivisible = hasData && !nodeData.indivisible;

	uint32_t blockHasData = hasData;
	for (int offset = 4; offset > 0; offset /= 2) 	//只要一个node有值，这个nodeBlock就有值
		blockHasData |= __shfl_down_sync(0xFFFFFFFF, blockHasData, offset);
	uint32_t warpHasDataBlockMask = blockHasData;	//这个warp中有值nodeBlock的mask
	warpHasDataBlockMask |= __shfl_sync(0xFFFFFFFF, blockHasData, 8) << 1;
	warpHasDataBlockMask |= __shfl_sync(0xFFFFFFFF, blockHasData, 16) << 2;
	warpHasDataBlockMask |= __shfl_sync(0xFFFFFFFF, blockHasData, 24) << 3;

	uint32_t warpHasDataAndDivisibleNodeMask = (uint32_t)hasDataAndDivisible << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpHasDataAndDivisibleNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpHasDataAndDivisibleNodeMask, offset);
	warpHasDataAndDivisibleNodeMask = __shfl_sync(0xFFFFFFFF, warpHasDataAndDivisibleNodeMask, 0);

	if (warpLane == 0) {
		atomicOr(&groupHasDataNodeBlockMask, (uint64_t)warpHasDataBlockMask << (warpIndex * 4));
		groupHasDataAndDivisibleNodeCountInWarp[warpIndex] = __popc(warpHasDataAndDivisibleNodeMask);
	}
	__syncthreads();

	if (warpIndex == 0) {
		uint32_t blockHasDataAndDivisibleNodeTotalCount;
		blockHasDataAndDivisibleNodeTotalCount = threadIdx.x < warpCount ? groupHasDataAndDivisibleNodeCountInWarp[threadIdx.x] : 0;
		for (int offset = 16; offset > 0; offset /= 2)
			blockHasDataAndDivisibleNodeTotalCount += __shfl_down_sync(0xFFFFFFFF, blockHasDataAndDivisibleNodeTotalCount, offset);
		if (threadIdx.x == 0) atomicAdd(hasDataAndDivisibleNodeCount, blockHasDataAndDivisibleNodeTotalCount);
		if constexpr (notOnlyOneBlock) {
			if (threadIdx.x == 0) {
				blockInfo[blockIdx.x].nodeCount = blockHasDataAndDivisibleNodeTotalCount;	//每个线程组有多少有值且可分node
				blockInfo[blockIdx.x].blockCount = __popcll(groupHasDataNodeBlockMask);	//每个线程组有多少有值nodeBlock
			}
		}
	}

	bool noHasDataNodeInWarp = false;
	if (warpLane == 0 && warpHasDataBlockMask == 0) noHasDataNodeInWarp = true;
	noHasDataNodeInWarp = __shfl_sync(0xFFFFFFFF, noHasDataNodeInWarp, 0);
	if (noHasDataNodeInWarp) return;
	//if ((groupHasDataNodeBlockMask & (((uint64_t)15) << (warpIndex * 4))) == 0) return; //若warp中没有有值node，则无需任何操作

	uint32_t label = warpLane < warpIndex ? groupHasDataAndDivisibleNodeCountInWarp[warpLane] : 0;
	for (int offset = 16; offset > 0; offset /= 2)
		label += __shfl_down_sync(0xFFFFFFFF, label, offset);
	label = __shfl_sync(0xFFFFFFFF, label, 0);
	label += __popc(warpHasDataAndDivisibleNodeMask << (32 - warpLane));	//加上warp中前面有几个有值node，得到最后自身在线程组中是第几个有值且可分node

	uint32_t newNodeIndex = __popcll(groupHasDataNodeBlockMask << (64 - blockIndexInGroup)) * 8;
	newNodeIndex += laneInBlock;
	//if (blockDim.x == 8 && hasDataAndDivisible) printf("%u %u\n", threadIndex, nodeData.label);
	//return;

	if (hasDataAndDivisible) nodeData.label = label + 1;	//label从1开始

	if constexpr (notOnlyOneBlock) {
		FzbSVONodeTempInfo tempInfo;
		tempInfo.hasData = hasData;
		if (hasData) {
			tempInfo.nodeData = nodeData;
			tempInfo.nodeIndexInThreadBlock = newNodeIndex;
		}
		if (hasDataAndDivisible) tempInfo.nodeIndex = nodeIndex;
		SVONodeTempInfos[threadIndex] = tempInfo;
	}
	if constexpr (!notOnlyOneBlock) {
		if (hasData) SVONodes[newNodeIndex] = nodeData;
		if (hasDataAndDivisible) hasDataAndDivisibleNodeIndices[nodeData.label - 1] = nodeIndex;
	}
}
__global__ void compressSVO_PG_TopToBottom_secondStep(FzbSVONodeData_PG* SVONodes,
	const uint32_t* __restrict__ hasDataAndDivisibleFatherNodeCount,
	const FzbSVONodeBlock* __restrict__ blockInfo, const FzbSVONodeTempInfo* __restrict__ SVONodeTempInfos,
	uint32_t* hasDataAndDivisibleNodeIndices, uint32_t* hasDataAndDivisibleNodeCount) {
	extern __shared__ uint32_t groupData[];	//前一半表示warp中有值且可分的node数量；后一半表示warp中有值nodeBlock的数量。最多32个
	__shared__ uint32_t groupNodeCount;
	__shared__ uint32_t groupHasDataAndDivisibleNodeStartIndex;		//前面线程组中有值且可分的node的数量
	__shared__ uint32_t groupHasDataNodeBlockStartIndex;			//前面线程组中有值的nodeBlock的数量

	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadIdx.x == 0) groupNodeCount = *hasDataAndDivisibleFatherNodeCount * 8;
	__syncthreads();
	if (threadIndex >= groupNodeCount) return;
	//uint32_t maxNeedWarpCount = (gridDim.x + 31) / 32;		//计算前面的线程组的数据最多需要多少个warp
	uint32_t actualNeedWarpCount = (blockIdx.x + 31) / 32;	//计算前面的线程组的数据实际需要多少个warp
	if (threadIdx.x < actualNeedWarpCount * 2) groupData[threadIdx.x] = 0;
	if (threadIdx.x == 0) {
		groupHasDataAndDivisibleNodeStartIndex = 0;
		groupHasDataNodeBlockStartIndex = 0;
	}
	__syncthreads();
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;

	if (warpIndex < actualNeedWarpCount) {
		uint32_t hasDataAndDivisbleNodeCount = threadIdx.x < blockIdx.x ? blockInfo[threadIdx.x].nodeCount : 0;
		uint32_t blockStartIndex = threadIdx.x < blockIdx.x ? blockInfo[threadIdx.x].blockCount : 0;
		for (int offset = 16; offset > 0; offset /= 2) {
			hasDataAndDivisbleNodeCount += __shfl_down_sync(0xFFFFFFFF, hasDataAndDivisbleNodeCount, offset);
			blockStartIndex += __shfl_down_sync(0xFFFFFFFF, blockStartIndex, offset);
		}
		if (warpLane == 0) {
			groupData[warpIndex] = hasDataAndDivisbleNodeCount;
			groupData[warpIndex + actualNeedWarpCount] = blockStartIndex;
		}
	}
	__syncthreads();

	if (warpIndex == 0) {
		uint32_t hasDataAndDivisibleNodeStartIndex = threadIdx.x < actualNeedWarpCount ? groupData[threadIdx.x] : 0;
		uint32_t hasDataNodeBlockCount = threadIdx.x < actualNeedWarpCount ? groupData[threadIdx.x + actualNeedWarpCount] : 0;
		for (int offset = 16; offset > 0; offset /= 2) {
			hasDataAndDivisibleNodeStartIndex += __shfl_down_sync(0xFFFFFFFF, hasDataAndDivisibleNodeStartIndex, offset);
			hasDataNodeBlockCount += __shfl_down_sync(0xFFFFFFFF, hasDataNodeBlockCount, offset);
		}
		if (warpLane == 0) {
			groupHasDataAndDivisibleNodeStartIndex = hasDataAndDivisibleNodeStartIndex;
			groupHasDataNodeBlockStartIndex = hasDataNodeBlockCount;
		}
	}
	__syncthreads();

	if (SVONodeTempInfos[threadIndex].hasData) {
		FzbSVONodeTempInfo tempData = SVONodeTempInfos[threadIndex];
		uint32_t nodeIndex = groupHasDataNodeBlockStartIndex * 8 + tempData.nodeIndexInThreadBlock;
		FzbSVONodeData_PG nodeData = tempData.nodeData;
		if (!nodeData.indivisible) {
			nodeData.label += groupHasDataAndDivisibleNodeStartIndex;
			hasDataAndDivisibleNodeIndices[nodeData.label - 1] = tempData.nodeIndex;
		}
		SVONodes[nodeIndex] = nodeData;
	}
}
*/
//template<bool notOnlyOneBlock>
//__global__ void compresSVO_PG_TopToBottom_lastSetp1(FzbSVONodeData_PG* SVONodes,
//	const FzbVoxelData_PG* __restrict__ VGB,
//	const uint32_t* __restrict__ hasDataAndDivisibleFatherNodeIndices, const uint32_t* __restrict__ hasDataAndDivisibleFatherNodeCount,
//	FzbSVONodeBlock* blockInfo, FzbSVONodeTempInfo* SVONodeTempInfos,
//	uint32_t* hasDataAndDivisibleNodeIndices, uint32_t* hasDataAndDivisibleNodeCount) {
//	__shared__ uint32_t groupNodeCount;
//	__shared__ uint64_t groupHasDataNodeBlockMask;	//一个线程组512个node，每个8个node为一组，共有64组，groupHasDataNodeBlock每一位表示一组有无值
//	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
//	uint32_t warpCount = (blockDim.x + 31) / 32;
//	if (threadIdx.x == 0) {
//		groupNodeCount = *hasDataAndDivisibleFatherNodeCount * 8;
//		groupHasDataNodeBlockMask = 0;
//	}
//	__syncthreads();
//	if (threadIndex >= groupNodeCount) return;
//	uint32_t blockIndexInGroup = threadIdx.x / 8;
//	uint32_t laneInBlock = threadIdx.x & 7;
//	uint32_t warpIndex = threadIdx.x / 32;
//	uint32_t warpLane = threadIdx.x & 31;
//
//	uint32_t fatherNodeIndex = hasDataAndDivisibleFatherNodeIndices[threadIndex / 8];
//	uint32_t voxelIndex = fatherNodeIndex * 8 + laneInBlock;
//	FzbVoxelData_PG voxelData = VGB[voxelIndex];
//	SVONodes[]
//
//	uint32_t blockHasData = hasData;
//	for (int offset = 4; offset > 0; offset /= 2) 	//只要一个node有值，这个nodeBlock就有值
//		blockHasData |= __shfl_down_sync(0xFFFFFFFF, blockHasData, offset);
//	uint32_t warpHasDataBlockMask = blockHasData;	//这个warp中有值nodeBlock的mask
//	warpHasDataBlockMask |= __shfl_sync(0xFFFFFFFF, blockHasData, 8) << 1;
//	warpHasDataBlockMask |= __shfl_sync(0xFFFFFFFF, blockHasData, 16) << 2;
//	warpHasDataBlockMask |= __shfl_sync(0xFFFFFFFF, blockHasData, 24) << 3;
//
//	uint32_t warpHasDataAndDivisibleNodeMask = (uint32_t)hasDataAndDivisible << warpLane;
//	for (int offset = 16; offset > 0; offset /= 2)
//		warpHasDataAndDivisibleNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpHasDataAndDivisibleNodeMask, offset);
//	warpHasDataAndDivisibleNodeMask = __shfl_sync(0xFFFFFFFF, warpHasDataAndDivisibleNodeMask, 0);
//
//	if (warpLane == 0) {
//		atomicOr(&groupHasDataNodeBlockMask, (uint64_t)warpHasDataBlockMask << (warpIndex * 4));
//		groupHasDataAndDivisibleNodeCountInWarp[warpIndex] = __popc(warpHasDataAndDivisibleNodeMask);
//	}
//	__syncthreads();
//
//	if (warpIndex == 0) {
//		uint32_t blockHasDataAndDivisibleNodeTotalCount;
//		blockHasDataAndDivisibleNodeTotalCount = threadIdx.x < warpCount ? groupHasDataAndDivisibleNodeCountInWarp[threadIdx.x] : 0;
//		for (int offset = 16; offset > 0; offset /= 2)
//			blockHasDataAndDivisibleNodeTotalCount += __shfl_down_sync(0xFFFFFFFF, blockHasDataAndDivisibleNodeTotalCount, offset);
//		if (threadIdx.x == 0) atomicAdd(hasDataAndDivisibleNodeCount, blockHasDataAndDivisibleNodeTotalCount);
//		if constexpr (notOnlyOneBlock) {
//			if (threadIdx.x == 0) {
//				blockInfo[blockIdx.x].nodeCount = blockHasDataAndDivisibleNodeTotalCount;	//每个线程组有多少有值且可分node
//				blockInfo[blockIdx.x].blockCount = __popcll(groupHasDataNodeBlockMask);	//每个线程组有多少有值nodeBlock
//			}
//		}
//	}
//
//	bool noHasDataNodeInWarp = false;
//	if (warpLane == 0 && warpHasDataBlockMask == 0) noHasDataNodeInWarp = true;
//	noHasDataNodeInWarp = __shfl_sync(0xFFFFFFFF, noHasDataNodeInWarp, 0);
//	if (noHasDataNodeInWarp) return;
//	//if ((groupHasDataNodeBlockMask & (((uint64_t)15) << (warpIndex * 4))) == 0) return; //若warp中没有有值node，则无需任何操作
//
//	uint32_t label = warpLane < warpIndex ? groupHasDataAndDivisibleNodeCountInWarp[warpLane] : 0;
//	for (int offset = 16; offset > 0; offset /= 2)
//		label += __shfl_down_sync(0xFFFFFFFF, label, offset);
//	label = __shfl_sync(0xFFFFFFFF, label, 0);
//	label += __popc(warpHasDataAndDivisibleNodeMask << (32 - warpLane));	//加上warp中前面有几个有值node，得到最后自身在线程组中是第几个有值且可分node
//
//	uint32_t newNodeIndex = __popcll(groupHasDataNodeBlockMask << (64 - blockIndexInGroup)) * 8;
//	newNodeIndex += laneInBlock;
//	//if (blockDim.x == 8 && hasDataAndDivisible) printf("%u %u\n", threadIndex, nodeData.label);
//	//return;
//
//	if (hasDataAndDivisible) nodeData.label = label + 1;	//label从1开始
//
//	if constexpr (notOnlyOneBlock) {
//		FzbSVONodeTempInfo tempInfo;
//		tempInfo.hasData = hasData;
//		if (hasData) {
//			tempInfo.nodeData = nodeData;
//			tempInfo.nodeIndexInThreadBlock = newNodeIndex;
//		}
//		if (hasDataAndDivisible) tempInfo.nodeIndex = nodeIndex;
//		SVONodeTempInfos[threadIndex] = tempInfo;
//	}
//	if constexpr (!notOnlyOneBlock) {
//		if (hasData) SVONodes[newNodeIndex] = nodeData;
//		if (hasDataAndDivisible) hasDataAndDivisibleNodeIndices[nodeData.label - 1] = nodeIndex;
//	}
//}

//__global__ void mergeSVONodes(FzbSVONodeData_PG* SVONodes_oneArray,
//	const FzbVoxelData_PG* __restrict__ VGB, FzbSVONodeData_PG** SVONodes_multiLayer, uint32_t** SVONodeCount_multiLayer,
//	uint32_t SVONodeTotalCount, uint32_t SVONodes_maxDepth) {
//	__shared__ uint32_t groupSVONodeCounts[7];	//不算叶节点，128^3 最多7层
//	__shared__ const FzbSVONodeData_PG* groupSVONodes[7];
//
//	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
//	if (threadIndex >= SVONodeTotalCount) return;
//	if (threadIdx.x < SVONodes_maxDepth + 1) {
//		groupSVONodeCounts[threadIdx.x] = *SVONodeCount_multiLayer[threadIdx.x] * 8;
//		groupSVONodes[threadIdx.x] = SVONodes_multiLayer[threadIdx.x];
//	}
//	__syncthreads();
//
//	int depth = 0;
//	int nodeIndex = threadIndex;
//	uint32_t curDepthSVONodeCount = groupSVONodeCounts[depth] * 8;
//	while (nodeIndex >= curDepthSVONodeCount) {
//		nodeIndex -= curDepthSVONodeCount;
//		++depth;
//		if (depth == SVONodes_maxDepth) break;
//		curDepthSVONodeCount = groupSVONodeCounts[depth] * 8;
//	}
//	FzbSVONodeData_PG nodeData = groupSVONodes[depth][nodeIndex];
//	if (depth == SVONodes_maxDepth) {
//		FzbVoxelData_PG voxelData = VGB[nodeData.]
//	}
//	SVONodes_oneArray[threadIndex] = 
//}

/*
void FzbSVOCuda_PG::compressSVONodes() {
	//uint32_t blockInfoIndex = SVONodeBlockInfos.size() - 1;
//for (int i = SVOs_PG.size() - 1; i > 0; --i) {
//	uint32_t nodeTotalCount = pow(8, i + 1);
//	blockSize = nodeTotalCount > createSVOKernelBlockSize ? createSVOKernelBlockSize : nodeTotalCount;
//	gridSize = (nodeTotalCount + blockSize - 1) / blockSize;
//
//	FzbSVONodeData_PG* SVONodes = SVOs_PG[i];
//	FzbSVONodeData_PG* SVOFatherNodes = SVOs_PG[i - 1];
//	uint32_t* svoNodeCount = SVONodeCount[i];
//	if (nodeTotalCount <= 512) compressSVO_PG_firstStep<false><<<gridSize, blockSize, 0, sourceManager->stream>>>
//		(SVONodes, SVOFatherNodes, nullptr, nullptr, svoNodeCount, nodeTotalCount);
//	else {
//		FzbSVONodeBlock* blockInfo = SVONodeBlockInfos[blockInfoIndex];
//		FzbSVONodeTempInfo* tempNodeInfo = SVONodeTempInfos[blockInfoIndex--];
//		uint32_t sharedDataSize = 2 * sizeof(uint32_t) * ((gridSize + 31) / 32);
//		compressSVO_PG_firstStep<true> <<<gridSize, blockSize, 0, sourceManager->stream >>> 
//			(SVONodes, SVOFatherNodes, blockInfo, tempNodeInfo, svoNodeCount, nodeTotalCount);
//		CHECK(cudaMemset(SVONodes, 0, nodeTotalCount * sizeof(FzbSVONodeData_PG)));
//		compressSVO_PG_secondStep <<<gridSize, blockSize, sharedDataSize, sourceManager->stream >>>
//			(SVONodes, blockInfo, tempNodeInfo, nodeTotalCount);
//	}
//}

	uint32_t blockInfoIndex = 0;
	SVONodeCount_host[0] = 1;
	for (int i = 0; i < SVONodes_maxDepth && SVONodeCount_host[i] > 0; ++i) {
		uint32_t nodeTotalCount = SVONodeCount_host[i] * 8;
		uint32_t blockSize = nodeTotalCount > createSVOKernelBlockSize ? createSVOKernelBlockSize : nodeTotalCount;
		uint32_t gridSize = (nodeTotalCount + blockSize - 1) / blockSize;

		FzbSVONodeData_PG* SVONodes = SVONodes_multiLayer[i];
		uint32_t* hasDataAndDivisibleFatherNodeCount = SVONodeCount[i];
		uint32_t* hasDataAndDivisibleFatherNodeIndices = SVONodeIndices[i];
		uint32_t* hasDataAndDivisibleNodeCount = SVONodeCount[i + 1];
		uint32_t* hasDataAndDivisibleNodeIndices = SVONodeIndices[i + 1];
		uint32_t firstStepSharedDataSize = ((blockSize + 31) / 32) * sizeof(uint32_t);
		if (nodeTotalCount <= 512)
			compressSVO_PG_TopToBottom_firstStep<false> << <gridSize, blockSize, firstStepSharedDataSize, stream >> >
			(
				SVONodes,
				hasDataAndDivisibleFatherNodeIndices, hasDataAndDivisibleFatherNodeCount,
				nullptr, nullptr,
				hasDataAndDivisibleNodeIndices, hasDataAndDivisibleNodeCount
				);
		else {
			FzbSVONodeBlock* blockInfo = SVONodeBlockInfos[blockInfoIndex];
			FzbSVONodeTempInfo* tempNodeInfo = SVONodeTempInfos[blockInfoIndex++];
			uint32_t secondStepSharedDataSize = 2 * sizeof(uint32_t) * ((gridSize + 31) / 32);
			compressSVO_PG_TopToBottom_firstStep<true> << <gridSize, blockSize, firstStepSharedDataSize, stream >> >
				(
					SVONodes,
					hasDataAndDivisibleFatherNodeIndices, hasDataAndDivisibleFatherNodeCount,
					blockInfo, tempNodeInfo,
					hasDataAndDivisibleNodeIndices, hasDataAndDivisibleNodeCount
					);
			compressSVO_PG_TopToBottom_secondStep << <gridSize, blockSize, secondStepSharedDataSize, stream >> >
				(
					SVONodes,
					hasDataAndDivisibleFatherNodeCount,
					blockInfo, tempNodeInfo,
					hasDataAndDivisibleNodeIndices, hasDataAndDivisibleNodeCount
					);
		}
		CHECK(cudaMemcpy(&SVONodeCount_host[i + 1], hasDataAndDivisibleNodeCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
	}

	for (int i = 0; i < SVONodes_maxDepth + 1; ++i)
		this->SVONodeTotalCount_host += this->SVONodeCount_host[i] * 8;
	if (setting.useOneArray) {
		CHECK(cudaMemcpy(SVONodeCount_multiLayer, this->SVONodeCount.data(), (SVONodes_maxDepth + 1) * sizeof(uint32_t*), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(this->SVONodeTotalCount, &SVONodeTotalCount_host, sizeof(uint32_t), cudaMemcpyHostToDevice));
		CHECK(cudaMalloc((void**)&this->SVONodes_oneArray, SVONodeTotalCount_host * sizeof(FzbSVONodeData_PG)));
		CHECK(cudaMemcpy(SVONodes_multiLayer_device, this->SVONodes_multiLayer.data(), SVONodes_maxDepth * sizeof(FzbSVONodeData_PG*), cudaMemcpyHostToDevice));

		uint32_t blockSize = SVONodeTotalCount_host > 1024 ? 1024 : SVONodeTotalCount_host;
		uint32_t gridSize = (SVONodeTotalCount_host + blockSize - 1) / blockSize;
		//mergeSVONodes<<<gridSize, blockSize, 0, stream>>>
		//	(SVONodes_oneArray, VGB, SVONodes_multiLayer_device, SVONodeCount_multiLayer, SVONodeTotalCount_host, SVONodes_maxDepth);
	}
}
*/

/*
/*
type = 0：说明当前线程小于一个warp，无需共享内存
type = 1：说明当前线程大于一个warp，但小于一个线程组，需要共享内存，但无需处理线程组之间的同步问题
type = 2：说当前线程大于一个线程组，需要处理多个线程组之间的同步问题

fatherNodeType = 0：说明父线程小于一个线程组，无需处理同步问题
fatherNodeType = 1：说明父线程大于一个线程组，需要处理同步问题

template<uint32_t type>
__global__ void createSVO_PG_first(
	FzbSVONodeData_PG* SVONodes,
	const FzbSVONodeData_PG* __restrict__ OctreeNodes,
	const FzbSVONodeTempInfo* __restrict__ divisibleFatherNodeInfos, uint32_t divisibleFatherNodeCount,
	FzbSVONodeThreadBlockInfo* threadBlockInfos,
	FzbSVONodeTempInfo* divisibleNodeTempInfos, uint32_t nodeInfosHalfSize,
	FzbSVONodeTempInfo* indivisibleNodeTempInfos,
	FzbSVOLayerInfo* layerInfo,
	FzbSVOIndivisibleNodeInfo* indivisibleNodeInfos, uint32_t indivisibleNodeTotalCount, uint32_t layerIndex
) {
	extern __shared__ uint32_t sharedData[];
	__shared__ uint32_t groupDivisibleNodeTempInfoOffset;	//在divisibleTempInfo中的偏移
	__shared__ uint32_t groupIndivisibleNodeTempInfoOffset;	//在indivisibleTempInfo中的偏移
	__shared__ uint32_t groupIndivisibleNodeInfoOffset;

	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t laneInBlock = threadIdx.x & 7;
	uint32_t warpCount = blockDim.x / 32;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;

	FzbSVONodeTempInfo fatherNodeInfo;
	FzbSVONodeData_PG nodeData;
	uint32_t nodeIndex;
	bool hasData = false;
	bool divisible = false;
	bool indivisible = false;
	//开的线程数就是有值可分父节点数 * 8，再向32取整，所以最后的一个warp中线程可能没有父节点
	//之所以不直接开相应数量的线程，是因为需要一个完整的warp（至少大于16个线程）取进行洗牌操作
	if (threadIndex < divisibleFatherNodeCount * 8) {
		fatherNodeInfo = divisibleFatherNodeInfos[threadIndex / 8];
		nodeIndex = fatherNodeInfo.nodeIndex * 8 + laneInBlock;
		nodeData = OctreeNodes[nodeIndex];
		hasData = glm::length(nodeData.irradiance) != 0.0f;
		divisible = hasData && !nodeData.indivisible;
		indivisible = hasData && nodeData.indivisible;
	}
	uint32_t storageIndex = (fatherNodeInfo.label - 1) * 8 + laneInBlock;

	//得到warp中可分node的掩码
	uint32_t warpDivisibleNodeMask = (uint32_t)divisible << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpDivisibleNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpDivisibleNodeMask, offset);
	warpDivisibleNodeMask = __shfl_sync(0xFFFFFFFF, warpDivisibleNodeMask, 0);

	//得到warp中不可分node的掩码
	uint32_t warpIndivisibleNodeMask = (uint32_t)indivisible << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpIndivisibleNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpIndivisibleNodeMask, offset);
	warpIndivisibleNodeMask = __shfl_sync(0xFFFFFFFF, warpIndivisibleNodeMask, 0);

	if constexpr (type == 0) {
		if (divisible) {
			uint32_t divisibleLabel = __popc(warpDivisibleNodeMask << (32 - warpLane));		//得到warp中比当前线程靠前的可分node数量

			FzbSVONodeTempInfo nodeInfo;
			nodeInfo.label = divisibleLabel + 1;	//label从1开始
			nodeInfo.nodeIndex = nodeIndex;
			divisibleNodeInfos[divisibleLabel] = nodeInfo;

			nodeData.label = divisibleLabel + 1;
		}
		else if (indivisible) {
			uint32_t indivisibleLabel = __popc(warpIndivisibleNodeMask << (32 - warpLane));	//得到warp中比当前线程靠前的不可分node数量

			nodeData.label = indivisibleLabel + 1;

			FzbSVOIndivisibleNodeInfo nodeInfo;
			nodeInfo.layerIndex = layerIndex;
			nodeInfo.nodeIndex = threadIndex;
			indivisibleNodeInfos[indivisibleNodeTotalCount + indivisibleLabel] = nodeInfo;	//将不可分node的信息存入大数组中，从之前层的不可分node数开始
		}

		if (threadIdx.x == 0) {
			layerInfo->divisibleNodeCount = __popc(warpDivisibleNodeMask);
			layerInfo->indivisibleNodeCount = __popc(warpIndivisibleNodeMask);
		}
	}
	else {
		uint32_t* groupDivisibleNodeCountInWarp = sharedData;	//sharedData前半部分为每个warp的可分node数
		if (warpLane == 0) groupDivisibleNodeCountInWarp[warpIndex] = __popc(warpDivisibleNodeMask);

		uint32_t* groupIndivisibleNodeCountInWarp = &sharedData[warpCount];		//sharedData后半部分为每个warp的不可分node数
		if (warpLane == 0) groupIndivisibleNodeCountInWarp[warpIndex] = __popc(warpIndivisibleNodeMask);
		__syncthreads();

		uint32_t divisibleLabel = warpLane < warpIndex ? groupDivisibleNodeCountInWarp[warpLane] : 0;	//计算比当前warp靠前的warp的可分node数
		for (int offset = 16; offset > 0; offset /= 2)
			divisibleLabel += __shfl_down_sync(0xFFFFFFFF, divisibleLabel, offset);
		divisibleLabel = __shfl_sync(0xFFFFFFFF, divisibleLabel, 0);

		uint32_t indivisibleLabel = warpLane < warpIndex ? groupIndivisibleNodeCountInWarp[warpLane] : 0;	//同理
		for (int offset = 16; offset > 0; offset /= 2)
			indivisibleLabel += __shfl_down_sync(0xFFFFFFFF, indivisibleLabel, offset);
		indivisibleLabel = __shfl_sync(0xFFFFFFFF, indivisibleLabel, 0);

		if constexpr (type == 1) {
			if (divisible) {
				divisibleLabel += __popc(warpDivisibleNodeMask << (32 - warpLane));	//将warp的label和线程在warp中的label加在一起，得到最终现在在线程组中的label

				FzbSVONodeTempInfo nodeInfo;
				nodeInfo.label = divisibleLabel + 1;
				nodeInfo.nodeIndex = nodeIndex;
				divisibleNodeTempInfos[divisibleLabel] = nodeInfo;

				nodeData.label = divisibleLabel + 1;
			}
			else if (indivisible) {
				indivisibleLabel += __popc(warpIndivisibleNodeMask << (32 - warpLane));	//同理

				nodeData.label = indivisibleLabel + 1;

				FzbSVOIndivisibleNodeInfo nodeInfo;
				nodeInfo.layerIndex = layerIndex;
				nodeInfo.nodeIndex = threadIndex;
				indivisibleNodeInfos[indivisibleNodeTotalCount + indivisibleLabel] = nodeInfo;	//同理
			}

			if (warpIndex == warpCount - 1 && warpLane == 0) {	//最后一个warp的label代表前面warp的node数之和，加上最后一个warp自身的node数，就是总数
				layerInfo->divisibleNodeCount = divisibleLabel + groupDivisibleNodeCountInWarp[warpIndex];
				layerInfo->indivisibleNodeCount = indivisibleLabel + groupIndivisibleNodeCountInWarp[warpIndex];
			}
		}

		if constexpr (type == 2) {
			if (warpIndex == warpCount - 1 && warpLane == 0) {
				uint32_t groupDivisibleNodeCount = divisibleLabel + groupDivisibleNodeCountInWarp[warpIndex];
				groupDivisibleNodeTempInfoOffset = atomicAdd(&layerInfo->divisibleNodeCount, groupDivisibleNodeCount);	//多个线程组需要同步

				uint32_t groupIndivisibleNodeCount = indivisibleLabel + groupIndivisibleNodeCountInWarp[warpIndex];
				groupIndivisibleNodeTempInfoOffset = atomicAdd(&layerInfo->indivisibleNodeCount, groupIndivisibleNodeCount);

				FzbSVONodeThreadBlockInfo blockInfo;
				blockInfo.divisibleNodeCount = groupDivisibleNodeCount;
				blockInfo.indivisibleNodeCount = groupIndivisibleNodeCount;
				threadBlockInfos[blockIdx.x] = blockInfo;
			}
			__syncthreads();

			if (divisible) {
				FzbSVONodeTempInfo nodeInfo;
				nodeInfo.label = divisibleLabel + 1;
				nodeInfo.nodeIndex = nodeIndex;
				nodeInfo.storageIndex = storageIndex;
				nodeInfo.threadBlockIndex = blockIdx.x;

				uint32_t nodeInfoStorageIndex = nodeInfosHalfSize + groupDivisibleNodeTempInfoOffset + divisibleLabel;	//数组大小为原来的2倍，先放到后半部分，等之后同步后放到前半部分
				divisibleNodeTempInfos[nodeInfoStorageIndex] = nodeInfo;

				nodeData.label = divisibleLabel + 1;
			}
			else if (indivisible) {
				FzbSVONodeTempInfo nodeInfo;
				nodeInfo.label = indivisibleLabel + 1;
				nodeInfo.nodeIndex = nodeIndex;
				nodeInfo.storageIndex = storageIndex;
				nodeInfo.threadBlockIndex = blockIdx.x;

				uint32_t nodeInfoStorageIndex = groupIndivisibleNodeTempInfoOffset + divisibleLabel;
				indivisibleNodeTempInfos[nodeInfoStorageIndex] = nodeInfo;

				nodeData.label = divisibleLabel + 1;

				FzbSVOIndivisibleNodeInfo nodeInfo2;
				nodeInfo2.layerIndex = layerIndex;
				nodeInfo2.nodeIndex = threadIndex;
				indivisibleNodeInfos[indivisibleNodeTotalCount + groupIndivisibleNodeTempInfoOffset + indivisibleLabel] = nodeInfo2;	//可以无序，所以直接存放即可
			}
		}
	}

	if (hasData) SVONodes[storageIndex] = nodeData;
}

__global__ void createSVO_PG_second_indivisible(
	FzbSVONodeData_PG* SVONodes,
	FzbSVONodeThreadBlockInfo* threadBlockInfos,
	FzbSVONodeTempInfo* indivisibleNodeTempInfos
) {
	extern __shared__ uint32_t threadBlockIndivisibleLabelOffset[];
	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadIdx.x < gridDim.x) threadBlockIndivisibleLabelOffset[threadIdx.x] = threadBlockInfos[threadIdx.x].indivisibleNodeCount;
	__syncthreads();
	if (threadIdx.x == 0) for (int i = 1; i < gridDim.x; ++i) threadBlockIndivisibleLabelOffset[i] += threadBlockIndivisibleLabelOffset[i - 1];
	__syncthreads();

	FzbSVONodeTempInfo nodeInfo = indivisibleNodeTempInfos[threadIndex];
	uint32_t labelOffset = threadBlockIndivisibleLabelOffset[nodeInfo.threadBlockIndex];

	uint32_t label = nodeInfo.label + labelOffset;
	SVONodes[nodeInfo.storageIndex].label = label;
}
*/

__global__ void createSVO_PG_first_type1(
	FzbSVONodeData_PG* SVONodes,
	const FzbSVONodeData_PG* __restrict__ OctreeNodes,
	const FzbSVONodeTempInfo* __restrict__ divisibleFatherNodeInfos,
	FzbSVONodeTempInfo* divisibleNodeTempInfos,
	FzbSVOLayerInfo* layerInfo, uint32_t layerIndex,
	FzbSVOIndivisibleNodeInfo* indivisibleNodeInfos, uint32_t indivisibleNodeTotalCount
) {
	uint32_t laneInBlock = threadIdx.x & 7;
	uint32_t warpLane = threadIdx.x & 31;
	uint32_t warpIndex = threadIdx.x / 32;

	FzbSVONodeTempInfo fatherNodeTempInfo = divisibleFatherNodeInfos[threadIdx.x / 8];
	uint32_t octreeNodeIndex = fatherNodeTempInfo.nodeIndex * 8 + laneInBlock;
	//uint32_t storageIndex = (fatherNodeTempInfo.label - 1) * 8 + laneInBlock;
	FzbSVONodeData_PG nodeData = OctreeNodes[octreeNodeIndex];
	//bool hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z != 0.0f;
	bool hasData = nodeData.AABB_G.leftX != FLT_MAX;
	bool divisible = hasData && !nodeData.indivisible;
	bool indivisible = hasData && nodeData.indivisible;

	uint32_t warpDivisibleNodeMask = divisible << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpDivisibleNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpDivisibleNodeMask, offset);
	warpDivisibleNodeMask = __shfl_sync(0xFFFFFFFF, warpDivisibleNodeMask, 0);

	uint32_t warpIndivisibleNodeMask = indivisible << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpIndivisibleNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpIndivisibleNodeMask, offset);
	warpIndivisibleNodeMask = __shfl_sync(0xFFFFFFFF, warpIndivisibleNodeMask, 0);

	if (divisible) {
		uint32_t nodeOffsetInLayer = __popc(warpDivisibleNodeMask << (32 - warpLane));

		FzbSVONodeTempInfo tempInfo;
		tempInfo.label = nodeOffsetInLayer + 1;
		tempInfo.nodeIndex = octreeNodeIndex;
		divisibleNodeTempInfos[nodeOffsetInLayer] = tempInfo;

		nodeData.label = nodeOffsetInLayer + 1;
	}
	else if(indivisible) {
		uint32_t nodeOffsetInLayer = __popc(warpIndivisibleNodeMask << (32 - warpLane));

		FzbSVOIndivisibleNodeInfo nodeInfo;
		nodeInfo.nodeLayer = layerIndex;
		nodeInfo.nodeIndex = threadIdx.x;
		indivisibleNodeInfos[indivisibleNodeTotalCount + nodeOffsetInLayer] = nodeInfo;

		nodeData.label = indivisibleNodeTotalCount + nodeOffsetInLayer + 1;
	}

	if (threadIdx.x == 0) {
		layerInfo[layerIndex].divisibleNodeCount = __popc(warpDivisibleNodeMask);
		layerInfo[layerIndex].indivisibleNodeCount = __popc(warpIndivisibleNodeMask);
	}

	if (hasData) SVONodes[threadIdx.x] = nodeData;
}
__global__ void createSVO_PG_first_type2(
	FzbSVONodeData_PG* SVONodes,
	const FzbSVONodeData_PG* __restrict__ OctreeNodes,
	const FzbSVONodeTempInfo* __restrict__ divisibleFatherNodeInfos, uint32_t divisibleFatherNodeCount,
	FzbSVONodeTempInfo* divisibleNodeTempInfos,
	FzbSVOLayerInfo* layerInfo, uint32_t layerIndex,
	FzbSVOIndivisibleNodeInfo* indivisibleNodeInfos, uint32_t indivisibleNodeTotalCount
) {
	extern __shared__ uint32_t sharedData[];

	uint32_t laneInBlock = threadIdx.x & 7;
	uint32_t warpCount = blockDim.x / 32;	//向32取整，所以必然整除
	uint32_t warpLane = threadIdx.x & 31;
	uint32_t warpIndex = threadIdx.x / 32;

	FzbSVONodeData_PG nodeData;
	uint32_t octreeNodeIndex;
	bool hasData = false;
	bool divisible = false;
	bool indivisible = false;
	if (threadIdx.x < divisibleFatherNodeCount * 8) {	//向32取整(为了洗牌操作)，所以最后一个warp中可能有的线程没有父节点
		FzbSVONodeTempInfo fatherNodeTempInfo = divisibleFatherNodeInfos[threadIdx.x / 8];
		octreeNodeIndex = fatherNodeTempInfo.nodeIndex * 8 + laneInBlock;
		nodeData = OctreeNodes[octreeNodeIndex];
		//hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z != 0.0f;
		hasData = nodeData.AABB_G.leftX != FLT_MAX;
		divisible = hasData && !nodeData.indivisible;
		indivisible = hasData && nodeData.indivisible;
	}

	uint32_t warpDivisibleNodeMask = uint32_t(divisible) << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpDivisibleNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpDivisibleNodeMask, offset);
	warpDivisibleNodeMask = __shfl_sync(0xFFFFFFFF, warpDivisibleNodeMask, 0);

	uint32_t warpIndivisibleNodeMask = uint32_t(indivisible) << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpIndivisibleNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpIndivisibleNodeMask, offset);
	warpIndivisibleNodeMask = __shfl_sync(0xFFFFFFFF, warpIndivisibleNodeMask, 0);

	uint32_t warpDivisibleNodeCount = __popc(warpDivisibleNodeMask);
	uint32_t warpIndivisibleNodeCount = __popc(warpIndivisibleNodeMask);

	uint32_t* groupDivisibleNodeOffsetInWarp = sharedData;
	uint32_t* groupIndivisibleNodeOffsetInWarp = &sharedData[warpCount];
	if (warpLane == 0) {
		groupDivisibleNodeOffsetInWarp[warpIndex] = warpDivisibleNodeCount;
		groupIndivisibleNodeOffsetInWarp[warpIndex] = warpIndivisibleNodeCount;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		for (int i = 1; i < warpCount; ++i) {
			groupDivisibleNodeOffsetInWarp[i] += groupDivisibleNodeOffsetInWarp[i - 1];
			groupIndivisibleNodeOffsetInWarp[i] += groupIndivisibleNodeOffsetInWarp[i - 1];
		}
	}
	__syncthreads();

	uint32_t warpDivisibleNodeOffset = 0;
	uint32_t warpIndivisibleNodeOffset = 0;
	if (warpLane == 0 && warpIndex > 0) {
		warpDivisibleNodeOffset = groupDivisibleNodeOffsetInWarp[warpIndex - 1];
		warpIndivisibleNodeOffset = groupIndivisibleNodeOffsetInWarp[warpIndex - 1];
	}
	warpDivisibleNodeOffset = __shfl_sync(0xFFFFFFFF, warpDivisibleNodeOffset, 0);
	warpIndivisibleNodeOffset = __shfl_sync(0xFFFFFFFF, warpIndivisibleNodeOffset, 0);

	if (divisible) {
		uint32_t nodeOffsetInLayer = warpDivisibleNodeOffset + __popc(warpDivisibleNodeMask << (32 - warpLane));

		FzbSVONodeTempInfo tempInfo;
		tempInfo.label = nodeOffsetInLayer + 1;
		tempInfo.nodeIndex = octreeNodeIndex;
		divisibleNodeTempInfos[nodeOffsetInLayer] = tempInfo;

		nodeData.label = nodeOffsetInLayer + 1;
	}
	else if (indivisible){
		uint32_t nodeOffsetInLayer = warpIndivisibleNodeOffset + __popc(warpIndivisibleNodeMask << (32 - warpLane));

		FzbSVOIndivisibleNodeInfo nodeInfo;
		nodeInfo.nodeLayer = layerIndex;
		nodeInfo.nodeIndex = threadIdx.x;
		indivisibleNodeInfos[indivisibleNodeTotalCount + nodeOffsetInLayer] = nodeInfo;

		nodeData.label = indivisibleNodeTotalCount + nodeOffsetInLayer + 1;
	}

	if (warpIndex == warpCount - 1 && warpLane == 0) {
		layerInfo[layerIndex].divisibleNodeCount = groupDivisibleNodeOffsetInWarp[warpIndex];
		layerInfo[layerIndex].indivisibleNodeCount = groupIndivisibleNodeOffsetInWarp[warpIndex];
	}

	if (hasData) SVONodes[threadIdx.x] = nodeData;
}
__global__ void createSVO_PG_first_type3(
	FzbSVONodeData_PG* SVONodes,
	const FzbSVONodeData_PG* __restrict__ OctreeNodes,
	const FzbSVONodeTempInfo* __restrict__ divisibleFatherNodeInfos, uint32_t divisibleFatherNodeCount,
	FzbSVONodeThreadBlockInfo* threadBlockInfos,
	FzbSVONodeTempInfo* divisibleNodeTempInfos, uint32_t nodeInfosHalfSize,
	FzbSVOLayerInfo* layerInfo, uint32_t layerIndex,
	FzbSVOIndivisibleNodeInfo* indivisibleNodeInfos, uint32_t indivisibleNodeTotalCount
) {
	extern __shared__ uint32_t sharedData[];
	__shared__ uint32_t groupDivisibleNodeOffset;
	__shared__ uint32_t groupIndivisibleNodeOffset;

	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t laneInBlock = threadIdx.x & 7;
	uint32_t warpCount = blockDim.x / 32;	//向32取整，所以必然整除
	uint32_t warpLane = threadIdx.x & 31;
	uint32_t warpIndex = threadIdx.x / 32;

	if (threadIdx.x == 0) {
		groupDivisibleNodeOffset = 0;
		groupIndivisibleNodeOffset = 0;
	}

	FzbSVONodeData_PG nodeData;
	uint32_t octreeNodeIndex;
	bool hasData = false;
	bool divisible = false;
	bool indivisible = false;
	if (threadIndex < divisibleFatherNodeCount * 8) {	//向32取整(为了洗牌操作)，所以最后一个warp中可能有的线程没有父节点
		FzbSVONodeTempInfo fatherNodeTempInfo = divisibleFatherNodeInfos[threadIndex / 8];
		octreeNodeIndex = fatherNodeTempInfo.nodeIndex * 8 + laneInBlock;
		nodeData = OctreeNodes[octreeNodeIndex];
		//hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z != 0.0f;
		hasData = nodeData.AABB_G.leftX != FLT_MAX;
		divisible = hasData && !nodeData.indivisible;
		indivisible = hasData && nodeData.indivisible;
	}

	uint32_t warpDivisibleNodeMask = divisible << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpDivisibleNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpDivisibleNodeMask, offset);
	warpDivisibleNodeMask = __shfl_sync(0xFFFFFFFF, warpDivisibleNodeMask, 0);

	uint32_t warpIndivisibleNodeMask = indivisible << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpIndivisibleNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpIndivisibleNodeMask, offset);
	warpIndivisibleNodeMask = __shfl_sync(0xFFFFFFFF, warpIndivisibleNodeMask, 0);

	uint32_t warpDivisibleNodeCount = __popc(warpDivisibleNodeMask);
	uint32_t warpIndivisibleNodeCount = __popc(warpIndivisibleNodeMask);

	uint32_t* groupDivisibleNodeOffsetInWarp = sharedData;
	uint32_t* groupIndivisibleNodeOffsetInWarp = &sharedData[warpCount];
	if (warpLane == 0) {
		groupDivisibleNodeOffsetInWarp[warpIndex] = warpDivisibleNodeCount;
		groupIndivisibleNodeOffsetInWarp[warpIndex] = warpIndivisibleNodeCount;
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		for (int i = 1; i < warpCount; ++i) {
			groupDivisibleNodeOffsetInWarp[i] += groupDivisibleNodeOffsetInWarp[i - 1];
			groupIndivisibleNodeOffsetInWarp[i] += groupIndivisibleNodeOffsetInWarp[i - 1];
		}
	}
	__syncthreads();

	uint32_t warpDivisibleNodeOffset = 0;
	uint32_t warpIndivisibleNodeOffset = 0;
	if (warpLane == 0 && warpIndex > 0) {
		warpDivisibleNodeOffset = groupDivisibleNodeOffsetInWarp[warpIndex - 1];
		warpIndivisibleNodeOffset = groupIndivisibleNodeOffsetInWarp[warpIndex - 1];
	}

	if (threadIdx.x == 0) {
		groupDivisibleNodeOffset = atomicAdd(&layerInfo[layerIndex].divisibleNodeCount, groupDivisibleNodeOffsetInWarp[warpCount - 1]);
		groupIndivisibleNodeOffset = atomicAdd(&layerInfo[layerIndex].indivisibleNodeCount, groupIndivisibleNodeOffsetInWarp[warpCount - 1]);
	}
	__syncthreads();

	//if (warpLane == 0) {
	//	//warpDivisibleNodeOffset += groupDivisibleNodeOffset;
	//	//warpIndivisibleNodeOffset += groupIndivisibleNodeOffset;
	//}
	warpDivisibleNodeOffset = __shfl_sync(0xFFFFFFFF, warpDivisibleNodeOffset, 0);
	warpIndivisibleNodeOffset = __shfl_sync(0xFFFFFFFF, warpIndivisibleNodeOffset, 0);

	if (divisible) {
		uint32_t nodeOffsetInLayer = warpDivisibleNodeOffset + __popc(warpDivisibleNodeMask << (32 - warpLane));

		FzbSVONodeTempInfo tempInfo;
		tempInfo.label = nodeOffsetInLayer + 1;
		tempInfo.nodeIndex = octreeNodeIndex;
		tempInfo.storageIndex = threadIndex;
		tempInfo.threadBlockIndex = blockIdx.x;
		divisibleNodeTempInfos[groupDivisibleNodeOffset + nodeOffsetInLayer + nodeInfosHalfSize] = tempInfo;	//先放到后半部分，同步后再搬到前半部分

		nodeData.label = nodeOffsetInLayer + 1;
	}
	else if(indivisible) {
		uint32_t nodeOffsetInLayer = warpIndivisibleNodeOffset + __popc(warpIndivisibleNodeMask << (32 - warpLane));

		FzbSVOIndivisibleNodeInfo nodeInfo;
		nodeInfo.nodeLayer = layerIndex;
		nodeInfo.nodeIndex = threadIndex;
		indivisibleNodeInfos[indivisibleNodeTotalCount + groupIndivisibleNodeOffset + nodeOffsetInLayer] = nodeInfo;

		nodeData.label = indivisibleNodeTotalCount + nodeOffsetInLayer + 1;		//线程组内的label
	}

	if (hasData) SVONodes[threadIndex] = nodeData;

	if (threadIdx.x == 0) {
		FzbSVONodeThreadBlockInfo blockInfo;
		blockInfo.divisibleNodeCount = groupDivisibleNodeOffsetInWarp[warpCount - 1];
		blockInfo.indivisibleNodeCount = groupIndivisibleNodeOffsetInWarp[warpCount - 1];
		threadBlockInfos[blockIdx.x] = blockInfo;
	}
}

__global__ void createSVO_PG_second_divisible(
	FzbSVONodeData_PG* SVONodes,
	FzbSVONodeThreadBlockInfo* threadBlockInfos, uint32_t threadBlockCount,
	FzbSVONodeTempInfo* divisibleNodeTempInfos, uint32_t nodeInfosHalfSize
) {
	extern __shared__ uint32_t threadBlockDivisibleLabelOffset[];
	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadIdx.x == 0) {	//每个元素表示该线程组前面的线程组的可分node数之和
		threadBlockDivisibleLabelOffset[0] = 0;
		for(int i = 1; i < threadBlockCount; ++i)
			threadBlockDivisibleLabelOffset[i] = threadBlockDivisibleLabelOffset[i - 1] + threadBlockInfos[i - 1].divisibleNodeCount;
	}	
	__syncthreads();

	FzbSVONodeTempInfo nodeInfo = divisibleNodeTempInfos[threadIndex + nodeInfosHalfSize];	//同步前放到了后半部分
	uint32_t labelOffset = threadBlockDivisibleLabelOffset[nodeInfo.threadBlockIndex];	//得到前面线程组的可分node数之和

	uint32_t label = nodeInfo.label + labelOffset;	//当前可分node在原线程组中的label加上前面线程组的可分node数之和就是在该layer的label
	divisibleNodeTempInfos[label - 1] = nodeInfo;
	SVONodes[nodeInfo.storageIndex].label = label;
}
__global__ void createSVO_PG_second_inDivisible(
	FzbSVONodeData_PG* SVONodes,
	FzbSVONodeThreadBlockInfo* threadBlockInfos, uint32_t threadBlockCount, uint32_t threadBlockSize,
	FzbSVOIndivisibleNodeInfo* indivisibleNodeInfos, uint32_t indivisibleNodeTotalCount, uint32_t inDivisibleNodeCount
) {
	extern __shared__ uint32_t threadBlockDivisibleLabelOffset[];
	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadIdx.x == 0) {	//每个元素表示该线程组前面的线程组的可分node数之和
		threadBlockDivisibleLabelOffset[0] = 0;
		for (int i = 1; i < threadBlockCount; ++i)
			threadBlockDivisibleLabelOffset[i] = threadBlockDivisibleLabelOffset[i - 1] + threadBlockInfos[i - 1].indivisibleNodeCount;
	}
	__syncthreads();

	if (threadIndex >= inDivisibleNodeCount) return;

	FzbSVOIndivisibleNodeInfo nodeInfo = indivisibleNodeInfos[indivisibleNodeTotalCount + threadIndex];
	uint32_t threadBlockIndex = nodeInfo.nodeIndex / threadBlockSize;

	if (threadBlockCount == 3) {
		printf("%d %d %d %d\n", 
			indivisibleNodeTotalCount, threadIndex, nodeInfo.nodeIndex, threadBlockSize);
		return;
	}

	uint32_t labelOffset = threadBlockDivisibleLabelOffset[threadBlockIndex];
	SVONodes[nodeInfo.nodeIndex].label += labelOffset;
}

__global__ void createSVO_PG_last_type1(
	FzbSVONodeData_PG* SVONodes,
	const FzbVoxelData_PG* __restrict__ VGB,
	const FzbSVONodeTempInfo* __restrict__ divisibleFatherNodeInfos,
	FzbSVOLayerInfo* layerInfo, uint32_t layerIndex,
	FzbSVOIndivisibleNodeInfo* indivisibleNodeInfos, uint32_t indivisibleNodeTotalCount
) {
	uint32_t laneInBlock = threadIdx.x & 7;
	uint32_t warpLane = threadIdx.x & 31;

	FzbSVONodeTempInfo fatherNodeInfo = divisibleFatherNodeInfos[threadIdx.x / 8];
	uint32_t VGBIndex = fatherNodeInfo.nodeIndex * 8 + laneInBlock;
	FzbVoxelData_PG voxelData = VGB[VGBIndex];
	bool hasData = __int_as_float(voxelData.AABB.leftX) != FLT_MAX;

	uint32_t warpHasDataMask = (uint32_t)hasData << warpLane;
	for(int offset = 16; offset > 0; offset /= 2)
		warpHasDataMask |= __shfl_down_sync(0xFFFFFFFF, warpHasDataMask, offset);
	warpHasDataMask = __shfl_sync(0xFFFFFFFF, warpHasDataMask, 0);

	if (warpLane == 0) {
		layerInfo[layerIndex].divisibleNodeCount = 0;
		layerInfo[layerIndex].indivisibleNodeCount = __popc(warpHasDataMask);
	}

	if (hasData) {
		uint32_t label = indivisibleNodeTotalCount + __popc(warpHasDataMask << (32 - warpLane));

		FzbSVONodeData_PG nodeData;
		nodeData.AABB_E = {
			__int_as_float(voxelData.AABB.leftX),
			__int_as_float(voxelData.AABB.rightX),
			__int_as_float(voxelData.AABB.leftY),
			__int_as_float(voxelData.AABB.rightY),
			__int_as_float(voxelData.AABB.leftZ),
			__int_as_float(voxelData.AABB.rightZ)
		};
		nodeData.AABB_G = nodeData.AABB_E;
		nodeData.indivisible = 1;
		nodeData.irradiance = voxelData.irradiance;
		nodeData.label = label + 1;
		nodeData.meanNormal_G = voxelData.meanNormal_G;
		nodeData.meanNormal_E = voxelData.meanNormal_E;
		//nodeData.pdf = 1.0f;
		SVONodes[threadIdx.x] = nodeData;

		FzbSVOIndivisibleNodeInfo nodeInfo;
		nodeInfo.nodeLayer = layerIndex;
		nodeInfo.nodeIndex = threadIdx.x;
		indivisibleNodeInfos[label] = nodeInfo;
	}
}
__global__ void createSVO_PG_last_type2(
	FzbSVONodeData_PG* SVONodes,
	const FzbVoxelData_PG* __restrict__ VGB,
	const FzbSVONodeTempInfo* __restrict__ divisibleFatherNodeInfos, uint32_t divisibleFatherNodeCount,
	FzbSVOLayerInfo* layerInfo, uint32_t layerIndex,
	FzbSVOIndivisibleNodeInfo* indivisibleNodeInfos, uint32_t indivisibleNodeTotalCount
) {
	extern __shared__ uint32_t groupHasDataNodeOffsetInWarp[];
	uint32_t laneInBlock = threadIdx.x & 7;
	uint32_t warpCount = blockDim.x / 32;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;

	FzbVoxelData_PG voxelData;
	bool hasData = false;
	if (threadIdx.x < divisibleFatherNodeCount * 8) {
		FzbSVONodeTempInfo fatherNodeInfo = divisibleFatherNodeInfos[threadIdx.x / 8];
		uint32_t VGBIndex = fatherNodeInfo.nodeIndex * 8 + laneInBlock;
		voxelData = VGB[VGBIndex];
		hasData = __int_as_float(voxelData.AABB.leftX) != FLT_MAX;
	}

	uint32_t warpHasDataMask = (uint32_t)hasData << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpHasDataMask |= __shfl_down_sync(0xFFFFFFFF, warpHasDataMask, offset);
	warpHasDataMask = __shfl_sync(0xFFFFFFFF, warpHasDataMask, 0);
	uint32_t warpHasDataNodeCount = __popc(warpHasDataMask);

	if (warpLane == 0) groupHasDataNodeOffsetInWarp[warpIndex] = warpHasDataNodeCount;
	__syncthreads();
	if (threadIdx.x == 0) {
		for (int i = 1; i < warpCount; ++i) groupHasDataNodeOffsetInWarp[i] += groupHasDataNodeOffsetInWarp[i - 1];
	}
	__syncthreads();

	uint32_t warpHasDataNodeOffset = 0;
	if (warpLane == 0 && warpIndex > 0) warpHasDataNodeOffset = groupHasDataNodeOffsetInWarp[warpIndex - 1];

	if (warpIndex == warpCount - 1 && warpLane == 0)
		layerInfo[layerIndex].indivisibleNodeCount = groupHasDataNodeOffsetInWarp[warpIndex];
	__syncthreads();

	warpHasDataNodeOffset = __shfl_sync(0xFFFFFFFF, warpHasDataNodeOffset, 0);

	if (hasData) {
		uint32_t label = indivisibleNodeTotalCount + warpHasDataNodeOffset + __popc(warpHasDataMask << (32 - warpLane));

		FzbSVONodeData_PG nodeData;
		nodeData.AABB_E = {
			__int_as_float(voxelData.AABB.leftX),
			__int_as_float(voxelData.AABB.rightX),
			__int_as_float(voxelData.AABB.leftY),
			__int_as_float(voxelData.AABB.rightY),
			__int_as_float(voxelData.AABB.leftZ),
			__int_as_float(voxelData.AABB.rightZ)
		};
		nodeData.AABB_G = nodeData.AABB_E;
		nodeData.indivisible = 1;
		nodeData.irradiance = voxelData.irradiance;
		nodeData.label = label;
		nodeData.meanNormal_G = voxelData.meanNormal_G;
		nodeData.meanNormal_E = voxelData.meanNormal_E;
		SVONodes[threadIdx.x] = nodeData;

		FzbSVOIndivisibleNodeInfo nodeInfo;
		nodeInfo.nodeLayer = layerIndex;
		nodeInfo.nodeIndex = threadIdx.x;
		indivisibleNodeInfos[label] = nodeInfo;
	}
}
__global__ void createSVO_PG_last_type3(
	FzbSVONodeData_PG* SVONodes,
	const FzbVoxelData_PG* __restrict__ VGB,
	const FzbSVONodeTempInfo* __restrict__ divisibleFatherNodeInfos, uint32_t divisibleFatherNodeCount,
	FzbSVONodeThreadBlockInfo* threadBlockInfos,
	FzbSVOLayerInfo* layerInfo, uint32_t layerIndex,
	FzbSVOIndivisibleNodeInfo* indivisibleNodeInfos, uint32_t indivisibleNodeTotalCount
) {
	extern __shared__ uint32_t groupHasDataNodeOffsetInWarp[];
	__shared__ uint32_t groupHasNodeOffset;

	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t laneInBlock = threadIdx.x & 7;
	uint32_t warpCount = blockDim.x / 32;	//向32取整，所以必然整除
	uint32_t warpLane = threadIdx.x & 31;
	uint32_t warpIndex = threadIdx.x / 32;

	FzbVoxelData_PG voxelData;
	bool hasData = false;
	if (threadIndex < divisibleFatherNodeCount * 8) {	//向32取整(为了洗牌操作)，所以最后一个warp中可能有的线程没有父节点
		FzbSVONodeTempInfo fatherNodeTempInfo = divisibleFatherNodeInfos[threadIndex / 8];
		uint32_t VGBNodeIndex = fatherNodeTempInfo.nodeIndex * 8 + laneInBlock;
		voxelData = VGB[VGBNodeIndex];
		hasData = __int_as_float(voxelData.AABB.leftX) != FLT_MAX;
	}

	uint32_t warpHasDataNodeMask = (uint32_t)hasData << warpLane;
	for (int offset = 16; offset > 0; offset /= 2)
		warpHasDataNodeMask |= __shfl_down_sync(0xFFFFFFFF, warpHasDataNodeMask, offset);
	warpHasDataNodeMask = __shfl_sync(0xFFFFFFFF, warpHasDataNodeMask, 0);

	uint32_t warpHasDataNodeCount = __popc(warpHasDataNodeMask);

	if (warpLane == 0) groupHasDataNodeOffsetInWarp[warpIndex] = warpHasDataNodeCount;
	__syncthreads();
	if (threadIdx.x == 0) {
		for (int i = 1; i < warpCount; ++i) groupHasDataNodeOffsetInWarp[i] += groupHasDataNodeOffsetInWarp[i - 1];
	}
	__syncthreads();

	uint32_t warpHasDataNodeOffset = 0;
	if (warpLane == 0 && warpIndex > 0) warpHasDataNodeOffset = groupHasDataNodeOffsetInWarp[warpIndex - 1];

	if (threadIdx.x == 0)
		groupHasNodeOffset = atomicAdd(&layerInfo[layerIndex].indivisibleNodeCount, groupHasDataNodeOffsetInWarp[warpCount - 1]);
	__syncthreads();

	//if (warpLane == 0) warpHasDataNodeOffset += groupHasNodeOffset;
	warpHasDataNodeOffset = __shfl_sync(0xFFFFFFFF, warpHasDataNodeOffset, 0);

	if(hasData) {
		uint32_t labelOffset = indivisibleNodeTotalCount + warpHasDataNodeOffset + __popc(warpHasDataNodeMask << (32 - warpLane));

		FzbSVONodeData_PG nodeData;
		nodeData.AABB_E = {
			__int_as_float(voxelData.AABB.leftX),
			__int_as_float(voxelData.AABB.rightX),
			__int_as_float(voxelData.AABB.leftY),
			__int_as_float(voxelData.AABB.rightY),
			__int_as_float(voxelData.AABB.leftZ),
			__int_as_float(voxelData.AABB.rightZ)
		};
		nodeData.AABB_G = nodeData.AABB_E;
		nodeData.indivisible = 1;
		nodeData.irradiance = voxelData.irradiance;
		nodeData.label = labelOffset + 1;	//不算上线程组的labelOffset
		nodeData.meanNormal_G = voxelData.meanNormal_G;
		nodeData.meanNormal_E = voxelData.meanNormal_E;
		SVONodes[threadIndex] = nodeData;

		FzbSVOIndivisibleNodeInfo nodeInfo;
		nodeInfo.nodeLayer = layerIndex;
		nodeInfo.nodeIndex = threadIndex;
		indivisibleNodeInfos[labelOffset + groupHasNodeOffset] = nodeInfo;
	}

	if (threadIdx.x == 0) {
		FzbSVONodeThreadBlockInfo threadBlockInfo;
		threadBlockInfo.indivisibleNodeCount = groupHasDataNodeOffsetInWarp[warpCount - 1];
		threadBlockInfos[blockIdx.x] = threadBlockInfo;
		//printf("%d %d %d\n", blockIdx.x, threadBlockInfos[blockIdx.x].divisibleNodeCount, threadBlockInfos[blockIdx.x].indivisibleNodeCount);
	}
}

void FzbSVOCuda_PG::createSVONodes() {
	uint32_t blockInfoIndex = 0;
	FzbSVOLayerInfo layerInfo_host = SVOLayerInfos_host[0];
	uint32_t divisibleFatherNodeCount = layerInfo_host.divisibleNodeCount;
	this->SVOInDivisibleNodeTotalCount_host = 0;
	for (int i = 1; i < SVONodes_maxDepth - 1 && divisibleFatherNodeCount > 0; ++i) {
		const FzbSVONodeData_PG* OctreeNodes = OctreeNodes_multiLayer[i];
		FzbSVONodeData_PG* SVONodes = SVONodes_multiLayer[i];
		const FzbSVONodeTempInfo* divisibleFatherNodeInfos = SVODivisibleNodeTempInfos[i - 1];
		FzbSVONodeTempInfo* divisibleNodeTempInfos = SVODivisibleNodeTempInfos[i];

		uint32_t nodeTotalCount = divisibleFatherNodeCount * 8;
		if(nodeTotalCount > SVONodesMaxCount[i]) throw std::runtime_error("NodeCount还真超出了，想想办法");

		uint32_t blockSize = createSVOKernelBlockSize;
		uint32_t gridSize = 1;
		if (nodeTotalCount <= 32) {
			blockSize = nodeTotalCount;
			gridSize = 1;

			createSVO_PG_first_type1<<<gridSize, blockSize, 0, stream>>>
			(	SVONodes,
				OctreeNodes,
				divisibleFatherNodeInfos,
				divisibleNodeTempInfos,
				SVOLayerInfos, i,
				SVOIndivisibleNodeInfos, SVOInDivisibleNodeTotalCount_host
			);
		}
		else {
			nodeTotalCount = ((nodeTotalCount + 31) / 32) * 32;	//向32取整，因为warp内要用洗牌操作，线程不够，可能拿不到数据
			blockSize = nodeTotalCount > createSVOKernelBlockSize ? createSVOKernelBlockSize : nodeTotalCount;
			gridSize = (nodeTotalCount + blockSize - 1) / blockSize;

			uint32_t sharedDataSize = 2 * (blockSize / 32) * sizeof(uint32_t);
			if (gridSize == 1) {
				createSVO_PG_first_type2<<<gridSize, blockSize, sharedDataSize, stream>>>
				(	SVONodes, 
					OctreeNodes,
					divisibleFatherNodeInfos, divisibleFatherNodeCount,
					divisibleNodeTempInfos,
					SVOLayerInfos, i,
					SVOIndivisibleNodeInfos, SVOInDivisibleNodeTotalCount_host
				);
			}
			else {
				FzbSVONodeThreadBlockInfo* threadBlockInfos = SVONodeThreadBlockInfos[i];
				uint32_t nodeInfosHalfSize = SVONodesMaxCount[i];
				createSVO_PG_first_type3 <<<gridSize, blockSize, sharedDataSize, stream >>>
				(
					SVONodes,
					OctreeNodes,
					divisibleFatherNodeInfos, divisibleFatherNodeCount,
					threadBlockInfos,
					divisibleNodeTempInfos, nodeInfosHalfSize,
					SVOLayerInfos, i,
					SVOIndivisibleNodeInfos, SVOInDivisibleNodeTotalCount_host
				);
			}
		}

		CHECK(cudaMemcpy(&SVOLayerInfos_host[i], SVOLayerInfos + i, sizeof(FzbSVOLayerInfo), cudaMemcpyDeviceToHost));
		uint32_t divisibleNodeCount = SVOLayerInfos_host[i].divisibleNodeCount;
		uint32_t inDivisibleNodeCount = SVOLayerInfos_host[i].indivisibleNodeCount;
		
		if (gridSize > 1) {
			FzbSVONodeThreadBlockInfo* threadBlockInfos = SVONodeThreadBlockInfos[i];
			uint32_t nodeInfosHalfSize = SVONodesMaxCount[i];

			uint32_t lastBlockSize = blockSize;
			uint32_t lastGridSize = gridSize;
			blockSize = divisibleNodeCount > createSVOKernelBlockSize ? createSVOKernelBlockSize : divisibleNodeCount;
			gridSize = (divisibleNodeCount + blockSize - 1) / blockSize;
			uint32_t sharedDataSize = lastGridSize * sizeof(uint32_t);
			createSVO_PG_second_divisible<<<gridSize, blockSize, sharedDataSize, stream>>>
				(SVONodes, threadBlockInfos, lastGridSize, divisibleNodeTempInfos, nodeInfosHalfSize);

			blockSize = inDivisibleNodeCount > createSVOKernelBlockSize ? createSVOKernelBlockSize : inDivisibleNodeCount;
			gridSize = (inDivisibleNodeCount + blockSize - 1) / blockSize;
			createSVO_PG_second_inDivisible<<<gridSize, blockSize, sharedDataSize, stream>>>
			(SVONodes, threadBlockInfos, lastGridSize, lastBlockSize, SVOIndivisibleNodeInfos, SVOInDivisibleNodeTotalCount_host, inDivisibleNodeCount);
		}
		divisibleFatherNodeCount = divisibleNodeCount;
		SVOInDivisibleNodeTotalCount_host += inDivisibleNodeCount;
	}

	if (divisibleFatherNodeCount > 0) {
		uint32_t nodeTotalCount = divisibleFatherNodeCount * 8;
		if (nodeTotalCount > SVONodesMaxCount[SVONodes_maxDepth - 1]) throw std::runtime_error("NodeCount还真超出了，想想办法");

		FzbSVONodeData_PG* SVONodes = SVONodes_multiLayer[SVONodes_maxDepth - 1];
		const FzbSVONodeTempInfo* divisibleFatherNodeInfos = SVODivisibleNodeTempInfos[SVONodes_maxDepth - 2];
		uint32_t blockSize;
		uint32_t gridSize;
		if (nodeTotalCount <= 32) {
			blockSize = nodeTotalCount;
			gridSize = 1;

			createSVO_PG_last_type1<<<gridSize, blockSize, 0, stream>>>
				(	SVONodes,
					VGB,
					divisibleFatherNodeInfos,
					SVOLayerInfos, SVONodes_maxDepth - 1,
					SVOIndivisibleNodeInfos, SVOInDivisibleNodeTotalCount_host
				);
		}
		else {
			nodeTotalCount = ((nodeTotalCount + 31) / 32) * 32;	//向32取整，因为warp内要用洗牌操作，线程不够，可能拿不到数据
			blockSize = nodeTotalCount > createSVOKernelBlockSize ? createSVOKernelBlockSize : nodeTotalCount;
			gridSize = (nodeTotalCount + blockSize - 1) / blockSize;

			uint32_t sharedDataSize = (blockSize / 32) * sizeof(uint32_t);
			if (gridSize == 1) {
				createSVO_PG_last_type2<<<gridSize, blockSize, sharedDataSize, stream>>>
					(	SVONodes,
						VGB,
						divisibleFatherNodeInfos, divisibleFatherNodeCount,
						SVOLayerInfos, SVONodes_maxDepth - 1,
						SVOIndivisibleNodeInfos, SVOInDivisibleNodeTotalCount_host
					);
			}
			else {
				FzbSVONodeThreadBlockInfo* threadBlockInfos = SVONodeThreadBlockInfos[SVONodes_maxDepth - 1];

				createSVO_PG_last_type3<<<gridSize, blockSize, sharedDataSize, stream>>>
				(	SVONodes,
					VGB,
					divisibleFatherNodeInfos, divisibleFatherNodeCount,
					threadBlockInfos,
					SVOLayerInfos, SVONodes_maxDepth - 1,
					SVOIndivisibleNodeInfos, SVOInDivisibleNodeTotalCount_host
				);
			}
		}

		CHECK(cudaMemcpy(&SVOLayerInfos_host[SVONodes_maxDepth - 1], SVOLayerInfos + SVONodes_maxDepth - 1, sizeof(FzbSVOLayerInfo), cudaMemcpyDeviceToHost));
		uint32_t inDivisibleNodeCount = SVOLayerInfos_host[SVONodes_maxDepth - 1].indivisibleNodeCount;
		if (gridSize > 1) {
			FzbSVONodeThreadBlockInfo* threadBlockInfos = SVONodeThreadBlockInfos[SVONodes_maxDepth - 1];
			std::vector<FzbSVONodeThreadBlockInfo> threadBlockInfos_host(gridSize);
			CHECK(cudaMemcpy(threadBlockInfos_host.data(), threadBlockInfos, gridSize * sizeof(FzbSVONodeThreadBlockInfo), cudaMemcpyDeviceToHost));

			uint32_t lastBlockSize = blockSize;
			uint32_t lastGridSize = gridSize;
			blockSize = inDivisibleNodeCount > createSVOKernelBlockSize ? createSVOKernelBlockSize : inDivisibleNodeCount;
			gridSize = (inDivisibleNodeCount + blockSize - 1) / blockSize;
			uint32_t sharedDataSize = lastGridSize * sizeof(uint32_t);
			createSVO_PG_second_inDivisible << <gridSize, blockSize, sharedDataSize, stream >> >
				(SVONodes, threadBlockInfos, lastGridSize, lastBlockSize, SVOIndivisibleNodeInfos, SVOInDivisibleNodeTotalCount_host, inDivisibleNodeCount);
		}
		SVOInDivisibleNodeTotalCount_host += inDivisibleNodeCount;
	}

	checkKernelFunction();

	for (int i = 0; i < SVONodes_maxDepth - 1; ++i) SVONodeTotalCount_host += SVOLayerInfos_host[i].divisibleNodeCount * 8;
	this->SVOHasDataNodeTotalCount_host = this->SVOInDivisibleNodeTotalCount_host;
	for (int i = 1; i < SVONodes_maxDepth - 1; ++i) this->SVOHasDataNodeTotalCount_host += SVOLayerInfos_host[i].divisibleNodeCount;
}

__device__ void initAABB_SVO(FzbAABB& AABB) {
	AABB.leftX = FLT_MAX;
	AABB.leftY = FLT_MAX;
	AABB.leftZ = FLT_MAX;
	AABB.rightX = -FLT_MAX;
	AABB.rightY = -FLT_MAX;
	AABB.rightZ = -FLT_MAX;
}
__global__ void initSVO(FzbSVONodeData_PG* SVO, uint32_t svoCount) {
	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= svoCount) return;

	FzbSVONodeData_PG data;
	data.indivisible = 1;
	data.label = 0;
	initAABB_SVO(data.AABB_G);
	initAABB_SVO(data.AABB_E);
	data.irradiance = glm::vec3(0.0f);
	data.meanNormal_G = glm::vec3(0.0f);
	data.meanNormal_E = glm::vec3(0.0f);
	SVO[threadIndex] = data;
}
void FzbSVOCuda_PG::initCreateSVONodesSource() {
	for (int i = 1; i < this->SVONodes_maxDepth; ++i) {
		SVONodeMaxCount += SVONodesMaxCount[i];
		SVOIndivisibleNodeMaxCount += SVONodesMaxCount[i];
	}

	this->SVONodes_multiLayer.resize(SVONodes_maxDepth);
	SVONodes_multiLayer[0] = nullptr;
	for (int i = 1; i < SVONodes_maxDepth; ++i) {
		uint32_t nodeCount = SVONodesMaxCount[i];	// std::pow(8, i);
		CHECK(cudaMalloc((void**)&SVONodes_multiLayer[i], nodeCount * sizeof(FzbSVONodeData_PG)));
		uint32_t blockSize = nodeCount > 1024 ? 1024 : nodeCount;
		uint32_t gridSize = (nodeCount + blockSize - 1) / blockSize;
		initSVO << <gridSize, blockSize, 0, stream >> > (this->SVONodes_multiLayer[i], nodeCount);
	}

	//避免循环时分配和销毁显存，所以开辟最大范围的空间
	this->SVONodeThreadBlockInfos.resize(SVONodes_maxDepth);
	for (int i = 0; i < SVONodes_maxDepth; ++i) {
		uint32_t nodeCount = SVONodesMaxCount[i];	// std::pow(8, i);
		uint32_t blockCount = nodeCount / createSVOKernelBlockSize;
		if (blockCount <= 1) {
			this->SVONodeThreadBlockInfos[i] == nullptr;
			continue;
		}
		CHECK(cudaMalloc((void**)&SVONodeThreadBlockInfos[i], blockCount * sizeof(FzbSVONodeThreadBlockInfo)));
	}

	this->SVODivisibleNodeTempInfos.resize(SVONodes_maxDepth);
	for (int i = 0; i < this->SVONodes_maxDepth - 1; ++i) {		//要算叶节点
		uint32_t maxNodeCount = SVONodesMaxCount[i] * 2;	// std::pow(8, i);
		CHECK(cudaMalloc((void**)&SVODivisibleNodeTempInfos[i], maxNodeCount * sizeof(FzbSVONodeTempInfo)));
	}
	FzbSVONodeTempInfo rootNodeTempInfo;
	rootNodeTempInfo.label = 1;
	rootNodeTempInfo.nodeIndex = 0;
	CHECK(cudaMemcpy(SVODivisibleNodeTempInfos[0], &rootNodeTempInfo, sizeof(FzbSVONodeTempInfo), cudaMemcpyHostToDevice));

	CHECK(cudaMalloc((void**)&this->SVOLayerInfos, SVONodes_maxDepth * sizeof(FzbSVOLayerInfo)));
	CHECK(cudaMemset(this->SVOLayerInfos, 0, SVONodes_maxDepth * sizeof(FzbSVOLayerInfo)))
	FzbSVOLayerInfo rootLayer = { 1, 0 };
	CHECK(cudaMemcpy(this->SVOLayerInfos, &rootLayer, sizeof(FzbSVOLayerInfo), cudaMemcpyHostToDevice));

	SVOLayerInfos_host.resize(SVONodes_maxDepth);
	SVOLayerInfos_host[0] = { 1, 0 };

	CHECK(cudaMalloc((void**)&SVOIndivisibleNodeInfos, SVOIndivisibleNodeMaxCount * sizeof(FzbSVOIndivisibleNodeInfo)));
}

