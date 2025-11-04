#include "FzbSVOCuda_PG.cuh"

__device__ void getAABBSum(uint32_t mask, FzbAABB& AABB, uint32_t blockFirstWarpLane) {
	//得到整合后的AABB的left
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(mask, AABB.leftX, offset, 8);
		AABB.leftX = fminf(AABB.leftX, other_val);
	}
	AABB.leftX = __shfl_sync(mask, AABB.leftX, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(mask, AABB.leftY, offset, 8);
		AABB.leftY = fminf(AABB.leftY, other_val);
	}
	AABB.leftY = __shfl_sync(mask, AABB.leftY, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(mask, AABB.leftZ, offset, 8);
		AABB.leftZ = fminf(AABB.leftZ, other_val);
	}
	AABB.leftZ = __shfl_sync(mask, AABB.leftZ, blockFirstWarpLane);

	//得到整合后的AABB的right
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(mask, AABB.rightX, offset, 8);
		AABB.rightX = fmaxf(AABB.rightX, other_val);
	}
	AABB.rightX = __shfl_sync(mask, AABB.rightX, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(mask, AABB.rightY, offset, 8);
		AABB.rightY = fmaxf(AABB.rightY, other_val);
	}
	AABB.rightY = __shfl_sync(mask, AABB.rightY, blockFirstWarpLane);

	for (int offset = 4; offset > 0; offset /= 2) {
		float other_val = __shfl_down_sync(mask, AABB.rightZ, offset, 8);
		AABB.rightZ = fmaxf(AABB.rightZ, other_val);
	}
	AABB.rightZ = __shfl_sync(mask, AABB.rightZ, blockFirstWarpLane);
}
__device__ void getAABBSum(FzbAABB& AABB, uint32_t maxEWarpLane) {
	float maxEAABB_leftX = __shfl_sync(__activemask(), AABB.leftX, maxEWarpLane);
	float maxEAABB_leftY = __shfl_sync(__activemask(), AABB.leftY, maxEWarpLane);
	float maxEAABB_leftZ = __shfl_sync(__activemask(), AABB.leftZ, maxEWarpLane);
	float maxEAABB_rightX = __shfl_sync(__activemask(), AABB.rightX, maxEWarpLane);
	float maxEAABB_rightY = __shfl_sync(__activemask(), AABB.rightY, maxEWarpLane);
	float maxEAABB_rightZ = __shfl_sync(__activemask(), AABB.rightZ, maxEWarpLane);

	AABB.leftX = fminf(AABB.leftX, maxEAABB_leftX);
	AABB.leftY = fminf(AABB.leftY, maxEAABB_leftY);
	AABB.leftZ = fminf(AABB.leftZ, maxEAABB_leftZ);
	AABB.rightX = fmaxf(AABB.rightX, maxEAABB_rightX);
	AABB.rightY = fmaxf(AABB.rightY, maxEAABB_rightY);
	AABB.rightZ = fmaxf(AABB.rightZ, maxEAABB_rightZ);
}

__global__ void createOctree_device_first(const FzbVoxelData_PG* __restrict__ VGB, FzbSVONodeData_PG* OctreeNodes, uint32_t voxelCount) {
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
	bool hasData = voxelData.irradiance != glm::vec3(0.0f);
	uint32_t warpHasDataMask = __ballot_sync(0xFFFFFFFF, hasData);
	uint32_t blockHasDataNodeCount = __popc(warpHasDataMask & blockNodeMask);

	warpHasDataMask = blockHasDataNodeCount > 1 ? blockNodeMask : 0;
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 0);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 8);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 16);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 24);

	if (blockHasDataNodeCount == 0) return;	//当前block中node全部没有数据

	glm::vec3 normal = voxelData.meanNormal.w == 0.0f ? glm::vec3(0.0f) : glm::normalize(glm::vec3(voxelData.meanNormal) / voxelData.meanNormal.w);
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
	if (blockHasDataNodeCount == 1) {	//第一层只有一个，很大概率是噪声，直接去掉试试
		if (hasData) {
			FzbSVONodeData_PG nodeData;
			nodeData.indivisible = 1;
			nodeData.AABB = AABB;
			nodeData.irradiance = voxelData.irradiance;
			nodeData.label = 0;
			nodeData.normal = normal;
			OctreeNodes[blockIndex] = nodeData;
		}
		return;
	}
	//------------------------------------------------irradiance差距判断-------------------------------------------------
	float irradianceValue = glm::length(voxelData.irradiance);
	float maxEIrradianceValue = irradianceValue;
	uint32_t maxEWarpLane = warpLane;
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_maxEIrradianceValue = __shfl_down_sync(__activemask(), maxEIrradianceValue, offset, 8);
		uint32_t other_maxEWarpLane = __shfl_down_sync(__activemask(), maxEWarpLane, offset, 8);
		if (other_maxEIrradianceValue > maxEIrradianceValue) {
			maxEIrradianceValue = other_maxEIrradianceValue;
			maxEWarpLane = other_maxEWarpLane;
		}
	}
	maxEIrradianceValue = __shfl_sync(__activemask(), maxEIrradianceValue, blockFirstWarpLane);
	maxEWarpLane = __shfl_sync(__activemask(), maxEWarpLane, blockFirstWarpLane);
	//----------------------------------------------判断忽略--------------------------------------------------------
	uint32_t indivisible = 1;
	uint32_t ignore = 0;
	float relIrradianceRatio = irradianceValue / maxEIrradianceValue;
	if (relIrradianceRatio <= groupSVOUniformData.irradianceRelRatioThreshold) {
		if (irradianceValue <= groupSVOUniformData.ignoreIrradianceValueThreshold) ignore = 1;
		else indivisible = 0;
	}
	
	for (int offset = 4; offset > 0; offset /= 2) 
		indivisible &= __shfl_down_sync(warpHasDataMask, indivisible, offset, 8);
	indivisible = __shfl_sync(warpHasDataMask, indivisible, blockFirstWarpLane);
	//---------------------------------------计算整合后的AABB-------------------------------------------------
	FzbAABB mergeAABB = AABB;
	if (ignore == 1) mergeAABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	getAABBSum(warpHasDataMask, mergeAABB, blockFirstWarpLane);

	//计算所有不被忽略的node聚类后的AABB的表面积
	float mergeAABBSurfaceArea = 0.0f;
	if (warpLane == blockFirstWarpLane) {
		float lengthX = mergeAABB.rightX - mergeAABB.leftX;
		float lengthY = mergeAABB.rightY - mergeAABB.leftY;
		float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
		mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	mergeAABBSurfaceArea = __shfl_sync(warpHasDataMask, mergeAABBSurfaceArea, blockFirstWarpLane);

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
	bool lastIndivisible = indivisible;
	if (warpLane == blockFirstWarpLane)
		if (surfaceAreaSum != 0.0f && mergeAABBSurfaceArea / surfaceAreaSum > groupSVOUniformData.surfaceAreaThreshold) indivisible = 0;
	indivisible = __shfl_sync(warpHasDataMask, indivisible, blockFirstWarpLane);
	//--------------------------如果之前不可分，但是根据表面积判断可分，则需要计算所有的node的AABB之和--------------------------
	uint32_t reGetAABBMask = (lastIndivisible > 0 && indivisible == 0) ? blockNodeMask : 0;
	if(warpHasDataMask & 1) reGetAABBMask |= __shfl_sync(warpHasDataMask, reGetAABBMask, 0);
	if (warpHasDataMask & (1 << 8)) reGetAABBMask |= __shfl_sync(warpHasDataMask, reGetAABBMask, 8);
	if (warpHasDataMask & (1 << 16)) reGetAABBMask |= __shfl_sync(warpHasDataMask, reGetAABBMask, 16);
	if (warpHasDataMask & (1 << 24)) reGetAABBMask |= __shfl_sync(warpHasDataMask, reGetAABBMask, 24);
	if (lastIndivisible > 0 && indivisible == 0) {
		mergeAABB = AABB;
		getAABBSum(reGetAABBMask, mergeAABB, blockFirstWarpLane);
		if (warpLane == blockFirstWarpLane) {
			float lengthX = mergeAABB.rightX - mergeAABB.leftX;
			float lengthY = mergeAABB.rightY - mergeAABB.leftY;
			float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
			mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		}
		mergeAABBSurfaceArea = __shfl_sync(reGetAABBMask, mergeAABBSurfaceArea, blockFirstWarpLane);
	}
	//-----------------------------------------------计算法线--------------------------------------------------------------
	glm::vec3 meanNormal = ignore ? glm::vec3(0.0f) : normal;
	for (int offset = 4; offset > 0; offset /= 2) {
		meanNormal.x += __shfl_down_sync(warpHasDataMask, meanNormal.x, offset, 8);
		meanNormal.y += __shfl_down_sync(warpHasDataMask, meanNormal.y, offset, 8);
		meanNormal.z += __shfl_down_sync(warpHasDataMask, meanNormal.z, offset, 8);
	}
	uint32_t notIgnoreNodeCount = __popc(__ballot_sync(warpHasDataMask, ignore == 0) & (0xff << blockFirstWarpLane));
	meanNormal /= notIgnoreNodeCount;
	//------------------------------------------------计算的irradiance之和-------------------------------------------------
	//如果可分，则计算的是所有的node的irradiance之和；如果不可分，则计算的是乘以因子的irradiance之和
	glm::vec3 mergeIrradiance = voxelData.irradiance;
	mergeIrradiance *= mergeAABBSurfaceArea < 1e-6 ? 0.0f : surfaceArea / mergeAABBSurfaceArea;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradiance.x += __shfl_down_sync(warpHasDataMask, mergeIrradiance.x, offset, 8);
		mergeIrradiance.y += __shfl_down_sync(warpHasDataMask, mergeIrradiance.y, offset, 8);
		mergeIrradiance.z += __shfl_down_sync(warpHasDataMask, mergeIrradiance.z, offset, 8);
	}
	//-----------------------------------------------赋值------------------------------------------------------------------
	if (warpLane == blockFirstWarpLane) {
		FzbSVONodeData_PG nodeData;
		nodeData.indivisible = indivisible;
		nodeData.AABB = mergeAABB;
		nodeData.irradiance = mergeIrradiance;
		nodeData.label = 0;
		nodeData.normal = meanNormal;
		OctreeNodes[blockIndex] = nodeData;
	}
}
__global__ void createOctree_device(FzbSVONodeData_PG* OctreeNodes_children, FzbSVONodeData_PG* OctreeNodes, uint32_t nodeCount) {
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
	uint indivisible = 1;
	//------------------------------------------------AABB表面积判断-------------------------------------------------
	float surfaceArea = 0.0f;
	if (hasData) {
		float lengthX = nodeData.AABB.rightX - nodeData.AABB.leftX;
		float lengthY = nodeData.AABB.rightY - nodeData.AABB.leftY;
		float lengthZ = nodeData.AABB.rightZ - nodeData.AABB.leftZ;
		surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	if (surfaceArea > groupSVOUniformData.voxelMultiple) indivisible = 0;
	//------------------------------------------------irradiance差距判断-------------------------------------------------
	float irrdianceValue = glm::length(nodeData.irradiance);
	uint32_t ignore = 0;
	for (int i = 0; i < 8; ++i) {
		float other_val = __shfl_sync(warpHasDataMask, irrdianceValue, blockFirstWarpLane + i);
		if (i == indexInBlock) continue;
		float minIrradiance = min(irrdianceValue, other_val);
		float maxIrradiance = max(irrdianceValue, other_val);
		if (maxIrradiance == 0.0f) continue;
		if (minIrradiance / maxIrradiance < groupSVOUniformData.irradianceRelRatioThreshold) {
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
	FzbAABB mergeAABB = nodeData.AABB;
	if (ignore == 1) mergeAABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	getAABBSum(warpHasDataMask, mergeAABB, blockFirstWarpLane);

	//计算所有不被忽略的node聚类后的AABB的表面积
	float mergeAABBSurfaceArea = 0.0f;
	if (warpLane == blockFirstWarpLane) {
		float lengthX = mergeAABB.rightX - mergeAABB.leftX;
		float lengthY = mergeAABB.rightY - mergeAABB.leftY;
		float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
		mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	mergeAABBSurfaceArea = __shfl_sync(warpHasDataMask, mergeAABBSurfaceArea, blockFirstWarpLane);

	//不可忽略node的表面积之和
	float surfaceAreaSum = ignore ? 0.0f : surfaceArea;
	for (int offset = 4; offset > 0; offset /= 2)
		surfaceAreaSum += __shfl_down_sync(warpHasDataMask, surfaceAreaSum, offset, 8);
	//---------------------------------------根据聚类后的AABB表面积判断可分-------------------------------------------------
	bool lastIndivisible = indivisible;
	if (warpLane == blockFirstWarpLane)
		if (surfaceAreaSum != 0.0f && mergeAABBSurfaceArea / surfaceAreaSum > groupSVOUniformData.surfaceAreaThreshold) indivisible = 0;
	indivisible = __shfl_sync(warpHasDataMask, indivisible, blockFirstWarpLane);
	//--------------------------如果之前不可分，但是根据表面积判断可分，则需要计算所有的node的AABB之和--------------------------
	uint32_t reGetAABBMask = (lastIndivisible > 0 && indivisible == 0) ? blockNodeMask : 0;
	if (warpHasDataMask & 1) reGetAABBMask |= __shfl_sync(warpHasDataMask, reGetAABBMask, 0);
	if (warpHasDataMask & (1 << 8)) reGetAABBMask |= __shfl_sync(warpHasDataMask, reGetAABBMask, 8);
	if (warpHasDataMask & (1 << 16)) reGetAABBMask |= __shfl_sync(warpHasDataMask, reGetAABBMask, 16);
	if (warpHasDataMask & (1 << 24)) reGetAABBMask |= __shfl_sync(warpHasDataMask, reGetAABBMask, 24);
	if (lastIndivisible > 0 && indivisible == 0) {
		mergeAABB = nodeData.AABB;
		getAABBSum(reGetAABBMask, mergeAABB, blockFirstWarpLane);
		if (warpLane == blockFirstWarpLane) {
			float lengthX = mergeAABB.rightX - mergeAABB.leftX;
			float lengthY = mergeAABB.rightY - mergeAABB.leftY;
			float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
			mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		}
		mergeAABBSurfaceArea = __shfl_sync(reGetAABBMask, mergeAABBSurfaceArea, blockFirstWarpLane);
	}
	//------------------------------------------------计算的irradiance之和-------------------------------------------------
	glm::vec3 mergeIrradiance = nodeData.irradiance;
	mergeIrradiance *= mergeAABBSurfaceArea < 1e-6 ? 0.0f : surfaceArea / mergeAABBSurfaceArea;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradiance.x += __shfl_down_sync(warpHasDataMask, mergeIrradiance.x, offset, 8);
		mergeIrradiance.y += __shfl_down_sync(warpHasDataMask, mergeIrradiance.y, offset, 8);
		mergeIrradiance.z += __shfl_down_sync(warpHasDataMask, mergeIrradiance.z, offset, 8);
	}
	//-----------------------------------------------计算法线--------------------------------------------------------------
	glm::vec3 meanNormal = ignore ? glm::vec3(0.0f) : nodeData.normal;
	for (int offset = 4; offset > 0; offset /= 2) {
		meanNormal.x += __shfl_down_sync(warpHasDataMask, meanNormal.x, offset, 8);
		meanNormal.y += __shfl_down_sync(warpHasDataMask, meanNormal.y, offset, 8);
		meanNormal.z += __shfl_down_sync(warpHasDataMask, meanNormal.z, offset, 8);
	}
	uint32_t notIgnoreNodeCount = __popc(__ballot_sync(warpHasDataMask, ignore == 0) & (0xff << blockFirstWarpLane));
	meanNormal /= notIgnoreNodeCount;
	//-----------------------------------------------赋值------------------------------------------------------------------
	if (warpLane == blockFirstWarpLane) {
		nodeData.indivisible = indivisible;
		nodeData.AABB = mergeAABB;
		nodeData.irradiance = mergeIrradiance;
		nodeData.label = 0;
		nodeData.normal = meanNormal;
		OctreeNodes[blockIndex] = nodeData;
	}
}
/*
__global__ void createOctree_device_first(const FzbVoxelData_PG* __restrict__ VGB, FzbSVONodeData_PG* OctreeNodes, uint32_t voxelCount) {
	__shared__ FzbVGBUniformData groupVGBUniformData;
	__shared__ FzbSVOUnformData groupSVOUniformData;

	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= voxelCount) return;		//voxelCount一定是32的整数倍，return不影响洗牌操纵
	if (threadIdx.x == 0) {
		groupVGBUniformData = systemVGBUniformData;
		groupSVOUniformData = systemSVOUniformData;
	}
	__syncthreads();

	uint32_t laneInBlock = threadIndex & 7;
	uint32_t blockIndex = threadIndex / 8;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;
	uint32_t blockFirstWarpLane = (warpLane / 8) << 3;

	FzbVoxelData_PG voxelData = VGB[threadIndex];
	glm::vec3 normal = voxelData.meanNormal.w == 0.0f ? glm::vec3(0.0f) : glm::normalize(glm::vec3(voxelData.meanNormal) / voxelData.meanNormal.w);
	FzbAABB AABB = {
			__int_as_float(voxelData.AABB.leftX),
			__int_as_float(voxelData.AABB.rightX),
			__int_as_float(voxelData.AABB.leftY),
			__int_as_float(voxelData.AABB.rightY),
			__int_as_float(voxelData.AABB.leftZ),
			__int_as_float(voxelData.AABB.rightZ)
	};

	//没光、有光没AABB都认为没有data
	bool hasData = (voxelData.irradiance.x + voxelData.irradiance.y + voxelData.irradiance.z > 0.0f) &&
		AABB.leftX != FLT_MAX && AABB.leftY != FLT_MAX && AABB.leftZ != FLT_MAX &&
		AABB.rightX != -FLT_MAX && AABB.rightY != -FLT_MAX && AABB.rightZ != -FLT_MAX;
	uint32_t warpHasDataMask = __ballot_sync(0xFFFFFFFF, hasData);
	uint32_t blockHasDataNodeCount = __popc(warpHasDataMask & (0xff << blockFirstWarpLane));

	if (blockHasDataNodeCount == 0) return;	//当前block中node全部没有数据
	if (blockHasDataNodeCount == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasData) {
			FzbSVONodeData_PG nodeData;
			nodeData.indivisible = 1;
			nodeData.AABB = AABB;
			nodeData.irradiance = voxelData.irradiance;
			nodeData.label = 0;
			nodeData.normal = normal;
			nodeData.influence = 0.0f;
			OctreeNodes[blockIndex] = nodeData;
		}
		return;
	}
	__syncwarp();
	//---------------------------------------找到maxE-------------------------------------------------
	float irradianceValue = glm::length(voxelData.irradiance);
	float maxEIrradianceValue = irradianceValue;
	uint32_t maxEWarpLane = warpLane;
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_maxEIrradianceValue = __shfl_down_sync(__activemask(), maxEIrradianceValue, offset, 8);
		uint32_t other_maxEWarpLane = __shfl_down_sync(__activemask(), maxEWarpLane, offset, 8);
		if (other_maxEIrradianceValue > maxEIrradianceValue) {
			maxEIrradianceValue = other_maxEIrradianceValue;
			maxEWarpLane = other_maxEWarpLane;
		}
	}
	maxEIrradianceValue = __shfl_sync(__activemask(), maxEIrradianceValue, blockFirstWarpLane);
	maxEWarpLane = __shfl_sync(__activemask(), maxEWarpLane, blockFirstWarpLane);
	//--------------------------------------计算表面积-------------------------------------------------
	float surfaceArea = 0.0f;
	if (hasData) {
		float lengthX = AABB.rightX - AABB.leftX;
		float lengthY = AABB.rightY - AABB.leftY;
		float lengthZ = AABB.rightZ - AABB.leftZ;
		surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	float maxESurfaceArea = __shfl_sync(__activemask(), surfaceArea, maxEWarpLane);
	if (maxESurfaceArea == 0) maxESurfaceArea = 1.0f;	//说明几何注入的时候之注入了一个片元
	//--------------------------------------计算与maxE合并后的表面积-----------------------------------
	FzbAABB mergeAABB = AABB;
	getAABBSum(mergeAABB, maxEWarpLane);
	float lengthX = mergeAABB.rightX - mergeAABB.leftX;
	float lengthY = mergeAABB.rightY - mergeAABB.leftY;
	float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
	float mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	if (mergeAABBSurfaceArea == 0) mergeAABBSurfaceArea = 1.0f;
	//---------------------------------------计算余弦-------------------------------------------------
	glm::vec3 maxENormal;
	maxENormal.x = __shfl_sync(__activemask(), normal.x, maxEWarpLane);
	maxENormal.y = __shfl_sync(__activemask(), normal.y, maxEWarpLane);
	maxENormal.z = __shfl_sync(__activemask(), normal.z, maxEWarpLane);
	float cosTheta = max(glm::dot(maxENormal, normal), 0.0f);
	//---------------------------------------得到ignore和indivisible信息--------------------------------------------
	uint32_t ignore = !hasData;
	uint32_t indivisible = 1;
	float geometricDistance = 1.0f - maxESurfaceArea / mergeAABBSurfaceArea;	//表面积越接近，几何距离越小
	float irradianceRelRatio = irradianceValue / maxEIrradianceValue;	//相对光照比例
	float irradianceAbsRatio = min(irradianceValue / groupSVOUniformData.irradianceThreshold, 1.0f);	//绝对光照比例

	float weight = geometricDistance * irradianceAbsRatio * (2.0f - cosTheta);
	if (warpLane != maxEWarpLane) {
		if (geometricDistance > groupSVOUniformData.geometricDistanceThreshold) {
			if (weight < groupSVOUniformData.ignoreThreshold) ignore = 1;
			else indivisible = 0;
		}
		else if(irradianceRelRatio < groupSVOUniformData.irradianceRelRatioThreshold){
			if (weight < groupSVOUniformData.ignoreThreshold) ignore = 1;
			else indivisible = 0;
		}
	}

	for (int offset = 4; offset > 0; offset /= 2)
		indivisible &= __shfl_down_sync(__activemask(), indivisible, offset, 8);
	indivisible = __shfl_sync(__activemask(), indivisible, blockFirstWarpLane);
	//----------------------------------------得到合并后的AABB----------------------------------------
	mergeAABB = AABB;
	if (ignore && indivisible) mergeAABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	getAABBSum(__activemask(), mergeAABB, blockFirstWarpLane);
	if (hasData) {
		float lengthX = mergeAABB.rightX - mergeAABB.leftX;
		float lengthY = mergeAABB.rightY - mergeAABB.leftY;
		float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
		mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	mergeAABBSurfaceArea = __shfl_sync(__activemask(), mergeAABBSurfaceArea, blockFirstWarpLane);
	//-----------------------------------得到合并后的irradiance----------------------------------------
	glm::vec3 mergeIrradiance = voxelData.irradiance;
	mergeIrradiance *= mergeAABBSurfaceArea < 1e-6 ? 0.0f : surfaceArea / mergeAABBSurfaceArea;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradiance.x += __shfl_down_sync(__activemask(), mergeIrradiance.x, offset, 8);
		mergeIrradiance.y += __shfl_down_sync(__activemask(), mergeIrradiance.y, offset, 8);
		mergeIrradiance.z += __shfl_down_sync(__activemask(), mergeIrradiance.z, offset, 8);
	}
	//--------------------------------------计算influence-------------------------------------------
	float influence = weight;
	for (int offset = 4; offset > 0; offset /= 2)
		influence += __shfl_down_sync(__activemask(), influence, offset, 8);
	//---------------------------------------------赋值-----------------------------------------------
	if (laneInBlock == 0) {
		FzbSVONodeData_PG nodeData;
		nodeData.indivisible = indivisible;
		nodeData.AABB = mergeAABB;
		nodeData.irradiance = mergeIrradiance;
		nodeData.label = 0;
		nodeData.influence = indivisible ? 0.0f : influence;
		nodeData.normal = maxENormal;
		OctreeNodes[blockIndex] = nodeData;
	}
}
__global__ void createOctree_device(FzbSVONodeData_PG* OctreeNodes_children, FzbSVONodeData_PG* OctreeNodes, uint32_t nodeCount) {
	__shared__ FzbSVOUnformData groupSVOUniformData;

	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= nodeCount * nodeCount * nodeCount) return;
	if (threadIdx.x == 0) {
		groupSVOUniformData = systemSVOUniformData;
	}
	__syncthreads();

	uint32_t laneInBlock = threadIndex & 7;
	uint32_t blockIndex = threadIndex / 8;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;
	uint32_t blockFirstWarpLane = (warpLane / 8) << 3;

	FzbSVONodeData_PG nodeData = OctreeNodes_children[threadIndex];
	if (nodeData.AABB.leftY >= 1.95) printf("%f\n", nodeData.AABB.leftY);
	bool hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
	uint32_t warpHasDataMask = __ballot_sync(0xFFFFFFFF, hasData);
	uint32_t blockHasDataNodeCount = __popc(warpHasDataMask & (0xff << blockFirstWarpLane));
	if (blockHasDataNodeCount == 0) return;	//当前block中node全部没有数据
	if (blockHasDataNodeCount == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasData) {
			nodeData.indivisible = 1;
			OctreeNodes[blockIndex] = nodeData;
		}
		return;
	}
	//---------------------------------------找到maxE-------------------------------------------------
	float irradianceValue = glm::length(nodeData.irradiance);
	float maxEIrradianceValue = irradianceValue;
	uint32_t maxEWarpLane = warpLane;
	for (int offset = 4; offset > 0; offset /= 2) {
		float other_maxEIrradianceValue = __shfl_down_sync(__activemask(), maxEIrradianceValue, offset, 8);
		uint32_t other_maxEWarpLane = __shfl_down_sync(__activemask(), maxEWarpLane, offset, 8);
		if (other_maxEIrradianceValue > maxEIrradianceValue) {
			maxEIrradianceValue = other_maxEIrradianceValue;
			maxEWarpLane = other_maxEWarpLane;
		}
	}
	maxEIrradianceValue = __shfl_sync(__activemask(), maxEIrradianceValue, blockFirstWarpLane);
	maxEWarpLane = __shfl_sync(__activemask(), maxEWarpLane, blockFirstWarpLane);
	//--------------------------------------计算表面积-------------------------------------------------
	float surfaceArea = 0.0f;
	if (hasData) {
		float lengthX = nodeData.AABB.rightX - nodeData.AABB.leftX;
		float lengthY = nodeData.AABB.rightY - nodeData.AABB.leftY;
		float lengthZ = nodeData.AABB.rightZ - nodeData.AABB.leftZ;
		surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	float maxESurfaceArea = __shfl_sync(__activemask(), surfaceArea, maxEWarpLane);
	//--------------------------------------计算与maxE合并后的表面积-----------------------------------
	FzbAABB mergeAABB = nodeData.AABB;
	getAABBSum(mergeAABB, maxEWarpLane);
	float lengthX = mergeAABB.rightX - mergeAABB.leftX;
	float lengthY = mergeAABB.rightY - mergeAABB.leftY;
	float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
	float mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	//---------------------------------------计算余弦-------------------------------------------------
	glm::vec3 maxENormal;
	maxENormal.x = __shfl_sync(__activemask(), nodeData.normal.x, maxEWarpLane);
	maxENormal.y = __shfl_sync(__activemask(), nodeData.normal.y, maxEWarpLane);
	maxENormal.z = __shfl_sync(__activemask(), nodeData.normal.z, maxEWarpLane);
	float cosTheta = max(glm::dot(maxENormal, nodeData.normal), 0.0f);
	//----------------------------------------更新influence--------------------------------------------
	float influence = nodeData.influence * 0.125f;
	//---------------------------------------得到ignore和indivisible信息--------------------------------------------
	uint32_t ignore = !hasData;
	uint32_t indivisible = nodeData.indivisible;
	float geometricDistance = 1.0f - maxESurfaceArea / mergeAABBSurfaceArea;	//表面积越接近，几何距离越小
	float irradianceRelRatio = irradianceValue / maxEIrradianceValue;	//相对光照比例
	float irradianceAbsRatio = min(irradianceValue / groupSVOUniformData.irradianceThreshold, 1.0f);	//绝对光照比例

	float weight = geometricDistance * irradianceAbsRatio * (2.0f - cosTheta) + influence;
	if (warpLane != maxEWarpLane) {
		if (geometricDistance > groupSVOUniformData.geometricDistanceThreshold) {
			if (irradianceValue < groupSVOUniformData.irradianceThreshold) ignore = 1;	//weight < groupSVOUniformData.ignoreThreshold
			else indivisible = 0;
		}
		else if (irradianceRelRatio < groupSVOUniformData.irradianceRelRatioThreshold) {
			if (irradianceValue < groupSVOUniformData.irradianceThreshold) ignore = 1;
			else indivisible = 0;
		}
	}

	for (int offset = 4; offset > 0; offset /= 2)
		indivisible &= __shfl_down_sync(__activemask(), indivisible, offset, 8);
	indivisible = __shfl_sync(__activemask(), indivisible, blockFirstWarpLane);
	//----------------------------------------得到合并后的AABB----------------------------------------
	mergeAABB = nodeData.AABB;
	if (ignore && indivisible) mergeAABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	getAABBSum(__activemask(), mergeAABB, blockFirstWarpLane);
	if (hasData) {
		float lengthX = mergeAABB.rightX - mergeAABB.leftX;
		float lengthY = mergeAABB.rightY - mergeAABB.leftY;
		float lengthZ = mergeAABB.rightZ - mergeAABB.leftZ;
		mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	mergeAABBSurfaceArea = __shfl_sync(__activemask(), mergeAABBSurfaceArea, blockFirstWarpLane);
	//-----------------------------------得到合并后的irradiance----------------------------------------
	glm::vec3 mergeIrradiance = nodeData.irradiance;
	mergeIrradiance *= mergeAABBSurfaceArea < 1e-6 ? 0.0f : surfaceArea / mergeAABBSurfaceArea;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradiance.x += __shfl_down_sync(__activemask(), mergeIrradiance.x, offset, 8);
		mergeIrradiance.y += __shfl_down_sync(__activemask(), mergeIrradiance.y, offset, 8);
		mergeIrradiance.z += __shfl_down_sync(__activemask(), mergeIrradiance.z, offset, 8);
	}
	//--------------------------------------计算influence-------------------------------------------
	influence = weight;
	for (int offset = 4; offset > 0; offset /= 2)
		influence += __shfl_down_sync(__activemask(), influence, offset, 8);
	//---------------------------------------------赋值-----------------------------------------------
	if (laneInBlock == 0) {
		FzbSVONodeData_PG nodeData;
		nodeData.indivisible = indivisible;
		nodeData.AABB = mergeAABB;
		nodeData.irradiance = mergeIrradiance;
		nodeData.label = 0;
		nodeData.influence = indivisible ? 0.0f : influence;
		nodeData.normal = maxENormal;
		OctreeNodes[blockIndex] = nodeData;
	}

	//if(nodeCount == pow(2, 4))
	//	printf("%f %f %f\n", mergeIrradiance.x, mergeIrradiance.y, mergeIrradiance.z);
}
*/
void FzbSVOCuda_PG::createOctreeNodes() {
	uint32_t voxelCount = std::pow(setting.voxelNum, 3);
	uint32_t blockSize = createSVOKernelBlockSize;
	uint32_t gridSize = (voxelCount + blockSize - 1) / blockSize;
	createOctree_device_first<<<gridSize, blockSize, 0, stream>>> (VGB, OctreeNodes_multiLayer[SVONodes_maxDepth - 2], voxelCount);
	checkKernelFunction();
	for (int i = SVONodes_maxDepth - 2; i > 1; --i) {
		FzbSVONodeData_PG* SVONodes_children = OctreeNodes_multiLayer[i];
		FzbSVONodeData_PG* SVONodes = OctreeNodes_multiLayer[i - 1];
		uint32_t nodeCount = pow(2, i);
		uint32_t nodeTotalCount = nodeCount * nodeCount * nodeCount;
		blockSize = nodeTotalCount > createSVOKernelBlockSize ? createSVOKernelBlockSize : nodeTotalCount;
		gridSize = (nodeTotalCount + blockSize - 1) / blockSize;
		createOctree_device<<<gridSize, blockSize, 0, stream>>> (SVONodes_children, SVONodes, nodeCount);
		checkKernelFunction();
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
	data.influence = 0.0;
	data.irradiance = glm::vec3(0.0f);
	data.normal = glm::vec3(0.0f);
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