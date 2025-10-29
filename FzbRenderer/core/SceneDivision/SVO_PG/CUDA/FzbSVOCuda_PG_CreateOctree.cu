#include "FzbSVOCuda_PG.cuh"

__global__ void createSVO_PG_device_first(const FzbVoxelData_PG* __restrict__ VGB, FzbSVONodeData_PG* OctreeNodes, uint32_t voxelCount) {
	__shared__ FzbVGBUniformData groupVGBUniformData;
	__shared__ FzbSVOUnformData groupSVOUniformData;

	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;

	if (threadIndex >= voxelCount) return;		//voxelCount一定是32的整数倍，return不影响洗牌操纵
	if (threadIdx.x == 0) {
		groupVGBUniformData = systemVGBUniformData;
		groupSVOUniformData = systemSVOUniformData;
	}
	__syncthreads();

	//这里的block指的是父级node
	uint32_t indexInBlock = threadIndex & 7;	//在8个兄弟node中的索引
	uint32_t blockIndex = threadIndex / 8;		//block在全局的索引
	uint32_t blockFirstWarpLane = (blockIndex & 3) * 8;	//当前block在warp中的位索引
	uint32_t blockNodeMask = 0xff << blockFirstWarpLane;
	uint32_t blockIndexInGroup = threadIdx.x / 8;

	FzbVoxelData_PG voxelData = VGB[threadIndex];
	bool hasData = voxelData.hasData && voxelData.irradiance != glm::vec3(0.0f);
	uint32_t warpHasDataMask = __ballot_sync(0xFFFFFFFF, hasData);
	uint32_t blockHasDataNodeCount = __popc(warpHasDataMask & blockNodeMask);

	warpHasDataMask = blockHasDataNodeCount > 1 ? blockNodeMask : 0;
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 0);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 8);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 16);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 24);

	if (blockHasDataNodeCount == 0) return;	//当前block中node全部没有数据

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
	if (blockHasDataNodeCount == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasData) {
			FzbSVONodeData_PG nodeData;
			nodeData.indivisible = 1;
			nodeData.AABB = AABB;
			nodeData.irradiance = voxelData.irradiance;
			nodeData.label = 0;
			OctreeNodes[blockIndex] = nodeData;
		}
		return;
	}
	//------------------------------------------------irradiance差距判断-------------------------------------------------
	uint indivisible = 1;
	float irrdianceValue = glm::length(voxelData.irradiance);
	uint32_t ignore = 0;
	for (int i = 0; i < 8; ++i) {
		float other_val = __shfl_sync(warpHasDataMask, irrdianceValue, blockFirstWarpLane + i);
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
		uint32_t other_val = __shfl_down_sync(warpHasDataMask, indivisible, offset, 8);
		indivisible = indivisible & other_val;
	}
	indivisible = __shfl_sync(warpHasDataMask, indivisible, blockFirstWarpLane);
	//---------------------------------------计算不被忽略的整合后的AABB-------------------------------------------------
	FzbAABB mergeNotIgnoreAABB = AABB;
	if (ignore == 1 && indivisible) mergeNotIgnoreAABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	//得到整合后的AABB的left
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftX, offset, 8);
		mergeNotIgnoreAABB.leftX = fminf(mergeNotIgnoreAABB.leftX, other_val);
	}
	mergeNotIgnoreAABB.leftX = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftX, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftY, offset, 8);
		mergeNotIgnoreAABB.leftY = fminf(mergeNotIgnoreAABB.leftY, other_val);
	}
	mergeNotIgnoreAABB.leftY = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftY, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftZ, offset, 8);
		mergeNotIgnoreAABB.leftZ = fminf(mergeNotIgnoreAABB.leftZ, other_val);
	}
	mergeNotIgnoreAABB.leftZ = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftZ, blockFirstWarpLane);

	//得到整合后的AABB的right
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightX, offset, 8);
		mergeNotIgnoreAABB.rightX = fmaxf(mergeNotIgnoreAABB.rightX, other_val);
	}
	mergeNotIgnoreAABB.rightX = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightX, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightY, offset, 8);
		mergeNotIgnoreAABB.rightY = fmaxf(mergeNotIgnoreAABB.rightY, other_val);
	}
	mergeNotIgnoreAABB.rightY = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightY, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightZ, offset, 8);
		mergeNotIgnoreAABB.rightZ = fmaxf(mergeNotIgnoreAABB.rightZ, other_val);
	}
	mergeNotIgnoreAABB.rightZ = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightZ, blockFirstWarpLane);

	//计算所有不被忽略的node聚类后的AABB的表面积
	float lengthX = mergeNotIgnoreAABB.rightX - mergeNotIgnoreAABB.leftX;
	float lengthY = mergeNotIgnoreAABB.rightY - mergeNotIgnoreAABB.leftY;
	float lengthZ = mergeNotIgnoreAABB.rightZ - mergeNotIgnoreAABB.leftZ;
	float mergeNotIgnoreSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;

	float surfaceArea = 0.0f;
	if (hasData) {
		float lengthX = AABB.rightX - AABB.leftX;
		float lengthY = AABB.rightY - AABB.leftY;
		float lengthZ = AABB.rightZ - AABB.leftZ;
		surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}

	//不可忽略node的表面积之和
	float surfaceAreaSum = ignore ? 0.0f : surfaceArea;
	for (int offset = 4; offset > 0; offset /= 2)
		surfaceAreaSum += __shfl_down_sync(warpHasDataMask, surfaceAreaSum, offset, 8);
	//---------------------------------------根据聚类后的AABB表面积判断可分-------------------------------------------------
	if (warpLane == blockFirstWarpLane)
		if (surfaceAreaSum != 0.0f && mergeNotIgnoreSurfaceArea / surfaceAreaSum > groupSVOUniformData.surfaceAreaThreshold) indivisible = 0;
	indivisible = __shfl_sync(warpHasDataMask, indivisible, blockFirstWarpLane);
	//------------------------------------------------计算的irradiance之和-------------------------------------------------
	glm::vec3 mergeIrradiance = voxelData.irradiance;
	if (ignore && indivisible) mergeIrradiance *= mergeNotIgnoreSurfaceArea < 1e-6 ? 0.0f : surfaceArea / mergeNotIgnoreSurfaceArea;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradiance.x += __shfl_down_sync(warpHasDataMask, mergeIrradiance.x, offset, 8);
		mergeIrradiance.y += __shfl_down_sync(warpHasDataMask, mergeIrradiance.y, offset, 8);
		mergeIrradiance.z += __shfl_down_sync(warpHasDataMask, mergeIrradiance.z, offset, 8);
	}
	if (warpLane == blockFirstWarpLane) {
		FzbSVONodeData_PG nodeData;
		nodeData.indivisible = indivisible;
		nodeData.AABB = mergeNotIgnoreAABB;
		nodeData.irradiance = mergeIrradiance;
		nodeData.label = 0;
		OctreeNodes[blockIndex] = nodeData;
	}
#ifdef _DEBUG
	//虽然后续得到和使用SVO时不需要可分node的AABB，但是debug可视化时需要
	mergeNotIgnoreAABB = AABB;
	//得到整合后的AABB的left
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftX, offset);
		mergeNotIgnoreAABB.leftX = fminf(mergeNotIgnoreAABB.leftX, other_val);
	}
	mergeNotIgnoreAABB.leftX = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftX, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftY, offset);
		mergeNotIgnoreAABB.leftY = fminf(mergeNotIgnoreAABB.leftY, other_val);
	}
	mergeNotIgnoreAABB.leftY = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftY, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftZ, offset);
		mergeNotIgnoreAABB.leftZ = fminf(mergeNotIgnoreAABB.leftZ, other_val);
	}
	mergeNotIgnoreAABB.leftZ = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftZ, blockFirstWarpLane);

	//得到整合后的AABB的right
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightX, offset);
		mergeNotIgnoreAABB.rightX = fmaxf(mergeNotIgnoreAABB.rightX, other_val);
	}
	mergeNotIgnoreAABB.rightX = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightX, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightY, offset);
		mergeNotIgnoreAABB.rightY = fmaxf(mergeNotIgnoreAABB.rightY, other_val);
	}
	mergeNotIgnoreAABB.rightY = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightY, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightZ, offset);
		mergeNotIgnoreAABB.rightZ = fmaxf(mergeNotIgnoreAABB.rightZ, other_val);
	}
	mergeNotIgnoreAABB.rightZ = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightZ, blockFirstWarpLane);

	if (warpLane == blockFirstWarpLane) OctreeNodes[blockIndex].AABB = mergeNotIgnoreAABB;
#endif
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
	uint32_t blockFirstWarpLane = (blockIndex & 3) * 8;	//当前block在warp中的位索引
	uint32_t blockNodeMask = 0xff << blockFirstWarpLane;
	uint32_t blockIndexInGroup = threadIdx.x / 8;

	FzbSVONodeData_PG nodeData = OctreeNodes_children[threadIndex];
	bool hasData = nodeData.irradiance != glm::vec3(0.0f);
	uint32_t warpHasDataMask = __ballot_sync(0xFFFFFFFF, hasData);
	uint32_t blockHasDataNodeCount = __popc(warpHasDataMask & blockNodeMask);

	warpHasDataMask = blockHasDataNodeCount > 1 ? blockNodeMask : 0;
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 0);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 8);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 16);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 24);

	if (blockHasDataNodeCount == 0) return;	//当前block中node全部没有数据
	if (blockHasDataNodeCount == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasData) {
			nodeData.indivisible = 1;
			OctreeNodes[blockIndex] = nodeData;
		}
		return;
	}
	//------------------------------------------------irradiance差距判断-------------------------------------------------
	uint indivisible = 1;
	float irrdianceValue = glm::length(nodeData.irradiance);
	uint32_t ignore = 0;
	for (int i = 0; i < 8; ++i) {
		float other_val = __shfl_sync(warpHasDataMask, irrdianceValue, blockFirstWarpLane + i);
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
		uint32_t other_val = __shfl_down_sync(warpHasDataMask, indivisible, offset, 8);
		indivisible = indivisible & other_val;
	}
	indivisible = __shfl_sync(warpHasDataMask, indivisible, blockFirstWarpLane);
	//---------------------------------------计算不被忽略的整合后的AABB-------------------------------------------------
	FzbAABB mergeNotIgnoreAABB = nodeData.AABB;
	if (ignore == 1 && indivisible) mergeNotIgnoreAABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	//得到整合后的AABB的left
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftX, offset, 8);
		mergeNotIgnoreAABB.leftX = fminf(mergeNotIgnoreAABB.leftX, other_val);
	}
	mergeNotIgnoreAABB.leftX = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftX, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftY, offset, 8);
		mergeNotIgnoreAABB.leftY = fminf(mergeNotIgnoreAABB.leftY, other_val);
	}
	mergeNotIgnoreAABB.leftY = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftY, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftZ, offset, 8);
		mergeNotIgnoreAABB.leftZ = fminf(mergeNotIgnoreAABB.leftZ, other_val);
	}
	mergeNotIgnoreAABB.leftZ = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftZ, blockFirstWarpLane);

	//得到整合后的AABB的right
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightX, offset, 8);
		mergeNotIgnoreAABB.rightX = fmaxf(mergeNotIgnoreAABB.rightX, other_val);
	}
	mergeNotIgnoreAABB.rightX = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightX, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightY, offset, 8);
		mergeNotIgnoreAABB.rightY = fmaxf(mergeNotIgnoreAABB.rightY, other_val);
	}
	mergeNotIgnoreAABB.rightY = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightY, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightZ, offset, 8);
		mergeNotIgnoreAABB.rightZ = fmaxf(mergeNotIgnoreAABB.rightZ, other_val);
	}
	mergeNotIgnoreAABB.rightZ = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightZ, blockFirstWarpLane);

	//计算所有不被忽略的node聚类后的AABB的表面积
	float lengthX = mergeNotIgnoreAABB.rightX - mergeNotIgnoreAABB.leftX;
	float lengthY = mergeNotIgnoreAABB.rightY - mergeNotIgnoreAABB.leftY;
	float lengthZ = mergeNotIgnoreAABB.rightZ - mergeNotIgnoreAABB.leftZ;
	float mergeNotIgnoreSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;

	float surfaceArea = 0.0f;
	if (hasData) {
		float lengthX = nodeData.AABB.rightX - nodeData.AABB.leftX;
		float lengthY = nodeData.AABB.rightY - nodeData.AABB.leftY;
		float lengthZ = nodeData.AABB.rightZ - nodeData.AABB.leftZ;
		surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}

	//不可忽略node的表面积之和
	float surfaceAreaSum = ignore ? 0.0f : surfaceArea;
	for (int offset = 4; offset > 0; offset /= 2)
		surfaceAreaSum += __shfl_down_sync(warpHasDataMask, surfaceAreaSum, offset, 8);
	//---------------------------------------根据聚类后的AABB表面积判断可分-------------------------------------------------
	if (warpLane == blockFirstWarpLane)
		if (surfaceAreaSum != 0.0f && mergeNotIgnoreSurfaceArea / surfaceAreaSum > groupSVOUniformData.surfaceAreaThreshold) indivisible = 0;
	indivisible = __shfl_sync(warpHasDataMask, indivisible, blockFirstWarpLane);
	//------------------------------------------------计算的irradiance之和-------------------------------------------------
	glm::vec3 mergeIrradiance = nodeData.irradiance;
	if (ignore && indivisible) mergeIrradiance *= mergeNotIgnoreSurfaceArea < 1e-6 ? 0.0f : surfaceArea / mergeNotIgnoreSurfaceArea;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradiance.x += __shfl_down_sync(warpHasDataMask, mergeIrradiance.x, offset, 8);
		mergeIrradiance.y += __shfl_down_sync(warpHasDataMask, mergeIrradiance.y, offset, 8);
		mergeIrradiance.z += __shfl_down_sync(warpHasDataMask, mergeIrradiance.z, offset, 8);
	}
	if (warpLane == blockFirstWarpLane) {
		FzbSVONodeData_PG nodeData;
		nodeData.indivisible = indivisible;
		nodeData.AABB = mergeNotIgnoreAABB;
		nodeData.irradiance = mergeIrradiance;
		nodeData.label = 0;
		OctreeNodes[blockIndex] = nodeData;
	}
#ifdef _DEBUG
	//虽然后续得到和使用SVO时不需要可分node的AABB，但是debug可视化时需要
	mergeNotIgnoreAABB = nodeData.AABB;
	//得到整合后的AABB的left
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftX, offset);
		mergeNotIgnoreAABB.leftX = fminf(mergeNotIgnoreAABB.leftX, other_val);
	}
	mergeNotIgnoreAABB.leftX = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftX, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftY, offset);
		mergeNotIgnoreAABB.leftY = fminf(mergeNotIgnoreAABB.leftY, other_val);
	}
	mergeNotIgnoreAABB.leftY = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftY, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.leftZ, offset);
		mergeNotIgnoreAABB.leftZ = fminf(mergeNotIgnoreAABB.leftZ, other_val);
	}
	mergeNotIgnoreAABB.leftZ = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.leftZ, blockFirstWarpLane);

	//得到整合后的AABB的right
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightX, offset);
		mergeNotIgnoreAABB.rightX = fmaxf(mergeNotIgnoreAABB.rightX, other_val);
	}
	mergeNotIgnoreAABB.rightX = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightX, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightY, offset);
		mergeNotIgnoreAABB.rightY = fmaxf(mergeNotIgnoreAABB.rightY, other_val);
	}
	mergeNotIgnoreAABB.rightY = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightY, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(warpHasDataMask, mergeNotIgnoreAABB.rightZ, offset);
		mergeNotIgnoreAABB.rightZ = fmaxf(mergeNotIgnoreAABB.rightZ, other_val);
	}
	mergeNotIgnoreAABB.rightZ = __shfl_sync(warpHasDataMask, mergeNotIgnoreAABB.rightZ, blockFirstWarpLane);

	if (warpLane == blockFirstWarpLane) OctreeNodes[blockIndex].AABB = mergeNotIgnoreAABB;
#endif
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
	//data.pdf = 1.0f;
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