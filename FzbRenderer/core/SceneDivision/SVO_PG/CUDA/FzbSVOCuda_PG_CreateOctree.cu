#include "FzbSVOCuda_PG.cuh"

__global__ void createSVO_PG_device_first(const FzbVoxelData_PG* __restrict__ VGB, FzbSVONodeData_PG* OctreeNodes, uint32_t voxelCount) {
	__shared__ FzbVGBUniformData groupVGBUniformData;
	__shared__ FzbSVOUnformData groupSVOUniformData;

	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;
	if (threadIndex >= voxelCount) return;
	if (threadIdx.x == 0) {
		groupVGBUniformData = systemVGBUniformData;
		groupSVOUniformData = systemSVOUniformData;
	}
	__syncthreads();

	//这里的block指的是父级node
	uint32_t indexInBlock = threadIndex & 7;	//在8个兄弟node中的索引
	uint32_t blockIndex = threadIndex / 8;		//block在全局的索引
	uint32_t blockIndexInWarpBit = (blockIndex & 3) * 8;	//当前block在warp中的位索引
	uint32_t blockIndexInGroup = threadIdx.x / 8;

	FzbVoxelData_PG voxelData = VGB[threadIndex];
	bool hasData = voxelData.hasData && voxelData.irradiance != glm::vec3(0.0f);
	uint32_t activeMask = __ballot_sync(0xFFFFFFFF, hasData);
	int firstActiveLaneInBlock = __ffs(activeMask & (0xff << blockIndexInWarpBit)) - 1;
	if (firstActiveLaneInBlock == -1) return;	//当前block中node全部没有数据

	FzbAABB AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	if (hasData) {
		AABB = {
			__int_as_float(voxelData.AABB.leftX),
			__int_as_float(voxelData.AABB.rightX),
			__int_as_float(voxelData.AABB.leftY),
			__int_as_float(voxelData.AABB.rightY),
			__int_as_float(voxelData.AABB.leftZ),
			__int_as_float(voxelData.AABB.rightZ)
		};
	}
	if (__popc(activeMask) == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasData) {
			OctreeNodes[blockIndex].indivisible = 1;
			OctreeNodes[blockIndex].AABB = AABB;
			OctreeNodes[blockIndex].irradiance = voxelData.irradiance;
		}
		return;
	}
	//------------------------------------------------irradiance判断-------------------------------------------------
	uint indivisible = 1;
	float irrdianceValue = glm::length(voxelData.irradiance);
	uint32_t ignore = 0;
	for (int i = 0; i < 8; ++i) {
		float other_val = __shfl_sync(0xFFFFFFFF, irrdianceValue, blockIndexInWarpBit + i);
		float minIrradiance = min(irrdianceValue, other_val);
		float maxIrradiance = max(irrdianceValue, other_val);
		if (minIrradiance == 0.0f) continue;
		if (maxIrradiance / minIrradiance > groupSVOUniformData.irradianceThreshold) {
			if (irrdianceValue == minIrradiance) {
				if (irrdianceValue < groupSVOUniformData.ignoreIrradianceValueThreshold) ignore = 1;
				else indivisible = 0;
			}
		}
	}
	for (int offset = 4; offset > 0; offset /= 2) {
		uint32_t other_val = __shfl_down_sync(0xFFFFFFFF, indivisible, offset);
		indivisible = indivisible & other_val;
	}
	//------------------------------------------------计算irradiance-------------------------------------------------
	glm::vec3 mergeIrradianceTotal = voxelData.irradiance;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradianceTotal.x += __shfl_down_sync(0xFFFFFFFF, mergeIrradianceTotal.x, offset);
		mergeIrradianceTotal.y += __shfl_down_sync(0xFFFFFFFF, mergeIrradianceTotal.y, offset);
		mergeIrradianceTotal.z += __shfl_down_sync(0xFFFFFFFF, mergeIrradianceTotal.z, offset);
	}
	glm::vec3 mergeIrradiance = ignore ? glm::vec3(0.0f) : voxelData.irradiance;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradiance.x += __shfl_down_sync(0xFFFFFFFF, mergeIrradiance.x, offset);
		mergeIrradiance.y += __shfl_down_sync(0xFFFFFFFF, mergeIrradiance.y, offset);
		mergeIrradiance.z += __shfl_down_sync(0xFFFFFFFF, mergeIrradiance.z, offset);
	}
	//------------------------------------------得到整合后的AABB---------------------------------------------------------
	if (ignore == 1) AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	FzbAABB mergeAABB = AABB;
	//得到整合后的AABB的left
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftX, offset);
		mergeAABB.leftX = fminf(mergeAABB.leftX, other_val);
	}
	mergeAABB.leftX = __shfl_sync(0xFFFFFFFF, mergeAABB.leftX, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftY, offset);
		mergeAABB.leftY = fminf(mergeAABB.leftY, other_val);
	}
	mergeAABB.leftY = __shfl_sync(0xFFFFFFFF, mergeAABB.leftY, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftZ, offset);
		mergeAABB.leftZ = fminf(mergeAABB.leftZ, other_val);
	}
	mergeAABB.leftZ = __shfl_sync(0xFFFFFFFF, mergeAABB.leftZ, blockIndexInWarpBit);

	//得到整合后的AABB的right
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightX, offset);
		mergeAABB.rightX = fmaxf(mergeAABB.rightX, other_val);
	}
	mergeAABB.rightX = __shfl_sync(0xFFFFFFFF, mergeAABB.rightX, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightY, offset);
		mergeAABB.rightY = fmaxf(mergeAABB.rightY, other_val);
	}
	mergeAABB.rightY = __shfl_sync(0xFFFFFFFF, mergeAABB.rightY, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightZ, offset);
		mergeAABB.rightZ = fmaxf(mergeAABB.rightZ, other_val);
	}
	mergeAABB.rightZ = __shfl_sync(0xFFFFFFFF, mergeAABB.rightZ, blockIndexInWarpBit);
	//------------------------------------------------计算表面积-------------------------------------------------
	float surfaceArea = 0.0f;
	if (hasData && ignore == 0) {
		float lengthX = AABB.rightX - AABB.leftX;
		float lengthY = AABB.rightY - AABB.leftY;
		float lengthZ = AABB.rightZ - AABB.leftZ;
		surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	for (int offset = 4; offset > 0; offset /= 2) {
		surfaceArea += __shfl_down_sync(0xFFFFFFFF, surfaceArea, offset);
	}
	//--------------------------------------------------对父节点赋值-------------------------------------------------
	if (warpLane == blockIndexInWarpBit) {
		float lengthX = mergeAABB.rightX - mergeAABB.leftX;
		float lengthY = mergeAABB.rightY - mergeAABB.leftY;
		float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
		float mergeSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		if (surfaceArea != 0.0f && mergeSurfaceArea / surfaceArea > groupSVOUniformData.surfaceAreaThreshold) indivisible = 0;

		OctreeNodes[blockIndex].indivisible = indivisible;
		if (indivisible) OctreeNodes[blockIndex].pdf = glm::length(mergeIrradiance) / glm::length(mergeIrradianceTotal);
		OctreeNodes[blockIndex].irradiance = mergeIrradianceTotal;
		OctreeNodes[blockIndex].AABB = mergeAABB;
	}
}
__global__ void createSVO_PG_device(FzbSVONodeData_PG* OctreeNodes_children, FzbSVONodeData_PG* OctreeNodes, uint32_t nodeCount) {
	__shared__ FzbSVOUnformData groupSVOUniformData;

	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;
	if (threadIndex >= nodeCount * nodeCount * nodeCount) return;
	if (threadIdx.x == 0) {
		groupSVOUniformData = systemSVOUniformData;
	}
	__syncthreads();

	//这里的block指的是父级node
	uint32_t indexInBlock = threadIndex & 7;	//在8个兄弟node中的索引
	uint32_t blockIndex = threadIndex / 8;		//block在全局的索引
	uint32_t blockIndexInWarpBit = (blockIndex & 3) * 8;	//当前block在warp中的位索引
	uint32_t blockIndexInGroup = threadIdx.x / 8;
	//uint32_t blockCount = nodeCount / 2;	//每个轴有几个block
	//uint32_t nodeIndexZ = (blockIndex / (blockCount * blockCount));
	//uint32_t nodeIndexY = (blockIndex - nodeIndexZ * (blockCount * blockCount)) / blockCount;
	//uint32_t nodeIndexX = blockIndex % blockCount;
	//nodeIndexX = nodeIndexX * 2 + (indexInBlock & 1);
	//nodeIndexY = nodeIndexY * 2 + ((indexInBlock >> 1) & 1);
	//nodeIndexZ = nodeIndexZ * 2 + ((indexInBlock >> 2) & 1);
	//uint32_t voxelIndexU = nodeIndexZ * (nodeCount * nodeCount) +
	//	nodeIndexY * nodeCount + nodeIndexX;
	FzbSVONodeData_PG nodeData = OctreeNodes_children[threadIndex];
	bool hasData = glm::length(nodeData.irradiance) > 0.01f;
	uint32_t activeMask = __ballot_sync(0xFFFFFFFF, hasData);
	int firstActiveLaneInBlock = __ffs(activeMask & (0xff << blockIndexInWarpBit)) - 1;
	if (firstActiveLaneInBlock == -1) return;	//当前block中node全部没有数据

	if (__popc(activeMask) == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasData) {
			OctreeNodes[blockIndex].indivisible = 1;
			OctreeNodes[blockIndex].AABB = nodeData.AABB;
			OctreeNodes[blockIndex].irradiance = nodeData.irradiance;
		}
		return;
	}
	//------------------------------------------------irradiance判断-------------------------------------------------
	uint indivisible = 1;
	float irrdianceValue = glm::length(nodeData.irradiance);
	uint32_t ignore = 0;
	for (int i = 0; i < 8; ++i) {
		float other_val = __shfl_sync(0xFFFFFFFF, irrdianceValue, blockIndexInWarpBit + i);
		float minIrradiance = min(irrdianceValue, other_val);
		float maxIrradiance = max(irrdianceValue, other_val);
		if (minIrradiance == 0.0f) continue;
		if (maxIrradiance / minIrradiance > groupSVOUniformData.irradianceThreshold) {
			if (irrdianceValue == minIrradiance) {
				if (irrdianceValue < groupSVOUniformData.ignoreIrradianceValueThreshold) ignore = 1;
				else indivisible = 0;
			}
		}
	}
	for (int offset = 4; offset > 0; offset /= 2) {
		uint32_t other_val = __shfl_down_sync(0xFFFFFFFF, indivisible, offset);
		indivisible = indivisible & other_val;
	}
	//------------------------------------------------计算irradiance-------------------------------------------------
	glm::vec3 mergeIrradianceTotal = nodeData.irradiance;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradianceTotal.x += __shfl_down_sync(0xFFFFFFFF, mergeIrradianceTotal.x, offset);
		mergeIrradianceTotal.y += __shfl_down_sync(0xFFFFFFFF, mergeIrradianceTotal.y, offset);
		mergeIrradianceTotal.z += __shfl_down_sync(0xFFFFFFFF, mergeIrradianceTotal.z, offset);
	}
	glm::vec3 mergeIrradiance = ignore ? glm::vec3(0.0f) : nodeData.irradiance;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradiance.x += __shfl_down_sync(0xFFFFFFFF, mergeIrradiance.x, offset) * __shfl_down_sync(0xFFFFFFFF, nodeData.pdf, offset);
		mergeIrradiance.y += __shfl_down_sync(0xFFFFFFFF, mergeIrradiance.y, offset) * __shfl_down_sync(0xFFFFFFFF, nodeData.pdf, offset);
		mergeIrradiance.z += __shfl_down_sync(0xFFFFFFFF, mergeIrradiance.z, offset) * __shfl_down_sync(0xFFFFFFFF, nodeData.pdf, offset);
	}
	//------------------------------------------得到整合后的AABB---------------------------------------------------------
	if (ignore == 1) nodeData.AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	FzbAABB mergeAABB = nodeData.AABB;
	//得到整合后的AABB的left
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftX, offset);
		mergeAABB.leftX = fminf(mergeAABB.leftX, other_val);
	}
	mergeAABB.leftX = __shfl_sync(0xFFFFFFFF, mergeAABB.leftX, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftY, offset);
		mergeAABB.leftY = fminf(mergeAABB.leftY, other_val);
	}
	mergeAABB.leftY = __shfl_sync(0xFFFFFFFF, mergeAABB.leftY, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.leftZ, offset);
		mergeAABB.leftZ = fminf(mergeAABB.leftZ, other_val);
	}
	mergeAABB.leftZ = __shfl_sync(0xFFFFFFFF, mergeAABB.leftZ, blockIndexInWarpBit);

	//得到整合后的AABB的right
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightX, offset);
		mergeAABB.rightX = fmaxf(mergeAABB.rightX, other_val);
	}
	mergeAABB.rightX = __shfl_sync(0xFFFFFFFF, mergeAABB.rightX, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightY, offset);
		mergeAABB.rightY = fmaxf(mergeAABB.rightY, other_val);
	}
	mergeAABB.rightY = __shfl_sync(0xFFFFFFFF, mergeAABB.rightY, blockIndexInWarpBit);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(0xFFFFFFFF, mergeAABB.rightZ, offset);
		mergeAABB.rightZ = fmaxf(mergeAABB.rightZ, other_val);
	}
	mergeAABB.rightZ = __shfl_sync(0xFFFFFFFF, mergeAABB.rightZ, blockIndexInWarpBit);
	//------------------------------------------------计算表面积-------------------------------------------------
	float surfaceArea = 0.0f;
	if (hasData) {
		float lengthX = nodeData.AABB.rightX - nodeData.AABB.leftX;
		float lengthY = nodeData.AABB.rightY - nodeData.AABB.leftY;
		float lengthZ = nodeData.AABB.rightZ - nodeData.AABB.leftZ;
		surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	for (int offset = 4; offset > 0; offset /= 2) {
		surfaceArea += __shfl_down_sync(0xFFFFFFFF, surfaceArea, offset);
	}
	//--------------------------------------------------对父节点赋值-------------------------------------------------
	if (warpLane == blockIndexInWarpBit) {
		float lengthX = mergeAABB.rightX - mergeAABB.leftX;
		float lengthY = mergeAABB.rightY - mergeAABB.leftY;
		float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
		float mergeSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		if (surfaceArea != 0.0f && mergeSurfaceArea / surfaceArea > groupSVOUniformData.surfaceAreaThreshold) indivisible = 0;

		OctreeNodes[blockIndex].indivisible = indivisible;
		if (indivisible) OctreeNodes[blockIndex].pdf = glm::length(mergeIrradiance) / glm::length(mergeIrradianceTotal);
		OctreeNodes[blockIndex].irradiance = mergeIrradianceTotal;
		OctreeNodes[blockIndex].AABB = mergeAABB;
	}
}

void FzbSVOCuda_PG::createOctreeNodes() {
	uint32_t voxelCount = std::pow(setting.voxelNum, 3);
	uint32_t blockSize = createSVOKernelBlockSize;
	uint32_t gridSize = (voxelCount + blockSize - 1) / blockSize;
	createSVO_PG_device_first << <gridSize, blockSize, 0, stream >> > (VGB, OctreeNodes_multiLayer[SVONodes_maxDepth - 2], voxelCount);
	for (int i = SVONodes_maxDepth - 2; i > 1; --i) {
		FzbSVONodeData_PG* SVONodes_children = OctreeNodes_multiLayer[i];
		FzbSVONodeData_PG* SVONodes = OctreeNodes_multiLayer[i - 1];
		uint32_t nodeCount = pow(2, i);
		uint32_t nodeTotalCount = nodeCount * nodeCount * nodeCount;
		blockSize = nodeTotalCount > createSVOKernelBlockSize ? createSVOKernelBlockSize : nodeTotalCount;
		gridSize = (nodeTotalCount + blockSize - 1) / blockSize;
		createSVO_PG_device << <gridSize, blockSize, 0, stream >> > (SVONodes_children, SVONodes, nodeCount);
	}
}

__global__ void initOctree(FzbSVONodeData_PG* Octree, uint32_t svoCount) {
	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= svoCount) return;

	FzbSVONodeData_PG data;
	data.indivisible = 1;
	data.pdf = 1.0f;
	//data.shuffleKey = 0;
	data.label = 0;
	data.AABB.leftX = FLT_MAX;
	data.AABB.leftY = FLT_MAX;
	data.AABB.leftZ = FLT_MAX;
	data.AABB.rightX = -FLT_MAX;
	data.AABB.rightY = -FLT_MAX;
	data.AABB.rightZ = -FLT_MAX;
	data.irradiance = glm::vec3(0.0f);
	Octree[threadIndex] = data;
}
void FzbSVOCuda_PG::initCreateOctreeNodesSource() {
	this->OctreeNodes_multiLayer.resize(SVONodes_maxDepth);
	for (int i = 0; i < SVONodes_maxDepth; ++i) {
		uint32_t nodeCount = std::pow(8, i);
		if (nodeCount == 1 || nodeCount == pow(setting.voxelNum, 3)) {	 //不存储根节点和叶节点
			this->OctreeNodes_multiLayer[i] = nullptr;
			continue;
		}
		CHECK(cudaMalloc((void**)&this->OctreeNodes_multiLayer[i], nodeCount * sizeof(FzbSVONodeData_PG)));
		uint32_t blockSize = nodeCount > 1024 ? 1024 : nodeCount;
		uint32_t gridSize = (nodeCount + blockSize - 1) / blockSize;
		initOctree << <gridSize, blockSize >> > (this->OctreeNodes_multiLayer[i], nodeCount);
	}
}