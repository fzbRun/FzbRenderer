#include "FzbSVOCuda_PG.cuh"

__device__ void initAABB(FzbAABB& AABB) {
	AABB.leftX = FLT_MAX;
	AABB.leftY = FLT_MAX;
	AABB.leftZ = FLT_MAX;
	AABB.rightX = -FLT_MAX;
	AABB.rightY = -FLT_MAX;
	AABB.rightZ = -FLT_MAX;
}
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
__device__ void getAABB(FzbAABB& AABB, uint32_t warpLane) {
	AABB.leftX = __shfl_sync(__activemask(), AABB.leftX, warpLane);
	AABB.leftY = __shfl_sync(__activemask(), AABB.leftY, warpLane);
	AABB.leftZ = __shfl_sync(__activemask(), AABB.leftZ, warpLane);
	AABB.rightX = __shfl_sync(__activemask(), AABB.rightX, warpLane);
	AABB.rightY = __shfl_sync(__activemask(), AABB.rightY, warpLane);
	AABB.rightZ = __shfl_sync(__activemask(), AABB.rightZ, warpLane);
}
__device__ void getVec3(glm::vec3& data, uint32_t warpLane) {
	data.x = __shfl_sync(__activemask(), data.x, warpLane);
	data.y = __shfl_sync(__activemask(), data.y, warpLane);
	data.z = __shfl_sync(__activemask(), data.z, warpLane);
}

/*
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
	bool hasData = voxelData.irradiance.w != 0.0f;
	//hasData &= (voxelData.irradiance.x + voxelData.irradiance.y + voxelData.irradiance.z) >= groupSVOUniformData.ignoreIrradianceValueThreshold;
	glm::vec3 irradiance = hasData ? voxelData.irradiance /= voxelData.irradiance.w : glm::vec3(0.0f);
	uint32_t warpHasDataMask = __ballot_sync(0xFFFFFFFF, hasData);
	uint32_t blockHasDataNodeCount = __popc(warpHasDataMask & blockNodeMask);

	warpHasDataMask = blockHasDataNodeCount > 1 ? blockNodeMask : 0;
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 0);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 8);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 16);
	warpHasDataMask |= __shfl_sync(0xFFFFFFFF, warpHasDataMask, 24);

	if (blockHasDataNodeCount == 0) return;	//当前block中node全部没有数据

	//glm::vec3 normal = voxelData.meanNormal.w == 0.0f ? glm::vec3(0.0f) : glm::normalize(glm::vec3(voxelData.meanNormal) / voxelData.meanNormal.w);
	glm::vec3 normal = glm::normalize(glm::vec3(voxelData.meanNormal));
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
			nodeData.irradiance = irradiance;
			nodeData.label = 0;
			nodeData.normal = normal;
			OctreeNodes[blockIndex] = nodeData;
		}
		return;
	}
	//------------------------------------------------irradiance差距判断-------------------------------------------------
	float irradianceValue = glm::length(irradiance);
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
	glm::vec3 mergeIrradiance = irradiance;
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
*/
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
/*
__global__ void createOctree_device_VGB(const FzbVoxelData_PG* __restrict__ VGB, FzbSVONodeData_PG* OctreeNodes, uint32_t voxelCount) {
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
	FzbAABB AABB = {
			__int_as_float(voxelData.AABB.leftX),
			__int_as_float(voxelData.AABB.rightX),
			__int_as_float(voxelData.AABB.leftY),
			__int_as_float(voxelData.AABB.rightY),
			__int_as_float(voxelData.AABB.leftZ),
			__int_as_float(voxelData.AABB.rightZ)
	};

	bool hasIrradiance = voxelData.irradiance.w != 0.0f;
	bool hasGeometry = AABB.leftX != FLT_MAX;
	glm::vec3 irradiance = voxelData.irradiance;

	uint32_t warpHasGeometryMask = __ballot_sync(0xFFFFFFFF, hasGeometry);
	uint32_t blockHasGeometryNodeCount = __popc(warpHasGeometryMask & blockNodeMask);

	if (blockHasGeometryNodeCount == 0) return;	//当前block中node全部没有数据
	if (blockHasGeometryNodeCount == 1) {
		if (hasGeometry) {
			FzbSVONodeData_PG nodeData;
			nodeData.indivisible = 1;
			nodeData.AABB_G = AABB;
			nodeData.AABB_E = AABB;
			//initAABB(nodeData.AABB_E);
			nodeData.irradiance = voxelData.irradiance;
			nodeData.label = 0;
			nodeData.meanNormal_G = voxelData.meanNormal_G;
			nodeData.meanNormal_E = voxelData.meanNormal_E;
			OctreeNodes[blockIndex] = nodeData;
		}
		return;
	}

	//---------------------------------------计算整合后的AABB_G-------------------------------------------------
	FzbAABB mergeAABB_G = AABB;
	getAABBSum(__activemask(), mergeAABB_G, blockFirstWarpLane);
	//---------------------------------------计算整合后的meanNormal_G-------------------------------------------------
	glm::vec3 meanNormal_G = voxelData.meanNormal_G;
	if (hasGeometry) meanNormal_G /= voxelData.meanNormal_G.w;
	for (int offset = 4; offset > 0; offset /= 2) {
		meanNormal_G.x += __shfl_down_sync(__activemask(), meanNormal_G.x, offset, 8);
		meanNormal_G.y += __shfl_down_sync(__activemask(), meanNormal_G.y, offset, 8);
		meanNormal_G.z += __shfl_down_sync(__activemask(), meanNormal_G.z, offset, 8);
	}
	meanNormal_G /= blockHasGeometryNodeCount;
	//------------------------------------------------irradiance差距判断-------------------------------------------------
	float irradianceValue = glm::length(irradiance);
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

	if (maxEIrradianceValue == 0.0f) {
		//getAABB(mergeAABB_G, blockFirstWarpLane);
		//getVec3(meanNormal_G, blockFirstWarpLane);
		if (warpLane == blockFirstWarpLane) {
			FzbSVONodeData_PG nodeData;
			nodeData.indivisible = 1;
			nodeData.AABB_G = mergeAABB_G;
			//nodeData.AABB_E = AABB;
			initAABB(nodeData.AABB_E);
			nodeData.irradiance = glm::vec3(0.0f); 	// voxelData.irradiance;
			nodeData.label = 0;
			nodeData.meanNormal_G = meanNormal_G;
			nodeData.meanNormal_E = glm::vec3(0.0f); //normal;
			OctreeNodes[blockIndex] = nodeData;
		}
		return;
	}
	//----------------------------------------------判断忽略--------------------------------------------------------
	uint32_t ignore = 0;
	float relIrradianceRatio = irradianceValue / maxEIrradianceValue;
	if (relIrradianceRatio <= groupSVOUniformData.irradianceRelRatioThreshold) ignore = 1;

	uint32_t blockIgnoreMask = ignore << indexInBlock;
	for (int offset = 4; offset > 0; offset /= 2)
		blockIgnoreMask |= __shfl_down_sync(__activemask(), blockIgnoreMask, offset, 8);
	blockIgnoreMask = __shfl_sync(__activemask(), blockIgnoreMask, blockFirstWarpLane);
	uint32_t blockNotIgnoreCount = 8 - __popc(blockIgnoreMask);
	if (blockNotIgnoreCount == 1) {	//只有一个不被忽略的node，就当作噪声去掉
		if (maxEIrradianceValue >= 10.0f) {
			getAABB(mergeAABB_G, blockFirstWarpLane);
			getVec3(meanNormal_G, blockFirstWarpLane);
			if (warpLane == maxEWarpLane) {
				FzbSVONodeData_PG nodeData;
				nodeData.indivisible = 1;
				nodeData.AABB_G = mergeAABB_G;
				nodeData.AABB_E = AABB;
				nodeData.irradiance = voxelData.irradiance;
				nodeData.label = 0;
				nodeData.meanNormal_G = meanNormal_G;
				nodeData.meanNormal_E = voxelData.meanNormal_E;
				OctreeNodes[blockIndex] = nodeData;
			}
		}
		else {
			if (warpLane == blockFirstWarpLane) {
				FzbSVONodeData_PG nodeData;
				nodeData.indivisible = 1;
				nodeData.AABB_G = mergeAABB_G;
				//nodeData.AABB_E = AABB;
				initAABB(nodeData.AABB_E);
				nodeData.irradiance = glm::vec3(0.0f); 	// voxelData.irradiance;
				nodeData.label = 0;
				nodeData.meanNormal_G = meanNormal_G;
				nodeData.meanNormal_E = glm::vec3(0.0f); //normal;
				OctreeNodes[blockIndex] = nodeData;
			}
		}
		return;
	}
	//---------------------------------------余弦判断-------------------------------------------------
	//bool doubleSided = false;
	//glm::vec3 normal_G = glm::vec3(0.0f);
	//if (hasGeometry) normal_G = glm::vec3(voxelData.meanNormal_G) / voxelData.meanNormal_G.w;
	//if (glm::length(normal_G) <= 0.5f) doubleSided = true;	//双面的就设为0，防止其余不是双面的进行聚类

	//bool indivisible = true;
	//glm::vec3 normalizeNormal_G = hasGeometry ? glm::normalize(voxelData.meanNormal_G) : glm::vec3(0.0f);
	//for (int blockLane = 0; blockLane < 8; ++blockLane) {
	//	glm::vec3 other_normal = normalizeNormal_G;
	//	uint32_t otherWarpLane = blockFirstWarpLane + blockLane;
	//	getVec3(other_normal, otherWarpLane);

	//	if(!hasGeometry || ((warpHasGeometryMask & (1u << otherWarpLane)) == 0)) continue;	//有一个没有几何，可以聚类
	//	//bool other_doubleSided = __shfl_sync(__activemask(), doubleSided, otherWarpLane);
	//	//if (doubleSided ^ other_doubleSided) {	//一个双面一个不是双面，则无法聚类
	//	//	indivisible = false;
	//	//	continue;
	//	//}

	//	float cosine = glm::dot(other_normal, normalizeNormal_G);
	//	if (cosine <= groupSVOUniformData.cosineDiffThreshold) indivisible = false;
	//}
	//for (int offset = 4; offset > 0; offset /= 2)
	//	indivisible &= __shfl_down_sync(__activemask(), indivisible, offset, 8);
	//indivisible = __shfl_sync(__activemask(), indivisible, blockFirstWarpLane);
	//---------------------------------------计算整合后的AABB-------------------------------------------------
	FzbAABB mergeAABB_E = AABB;
	if (ignore == 1) mergeAABB_E = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	getAABBSum(__activemask(), mergeAABB_E, blockFirstWarpLane);

	//计算所有不被忽略的node聚类后的AABB的表面积
	float mergeAABBSurfaceArea = 0.0f;
	if (warpLane == blockFirstWarpLane) {
		float lengthX = mergeAABB_E.rightX - mergeAABB_E.leftX;
		float lengthY = mergeAABB_E.rightY - mergeAABB_E.leftY;
		float lengthZ = mergeAABB_E.rightZ - mergeAABB_E.leftZ;
		mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	mergeAABBSurfaceArea = __shfl_sync(__activemask(), mergeAABBSurfaceArea, blockFirstWarpLane);

	float surfaceArea = 0.0f;
	if (hasIrradiance && hasGeometry) {	//有光照才需要surfaceArea来计算缩放因子，虽然说有光照必有几何，但是光栅化可能没有注入进去
		float lengthX = AABB.rightX - AABB.leftX;
		float lengthY = AABB.rightY - AABB.leftY;
		float lengthZ = AABB.rightZ - AABB.leftZ;
		surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	//-----------------------------------------------计算法线--------------------------------------------------------------
	glm::vec3 meanNormal_E = ignore ? glm::vec3(0.0f) : voxelData.meanNormal_E;
	for (int offset = 4; offset > 0; offset /= 2) {
		meanNormal_E.x += __shfl_down_sync(__activemask(), meanNormal_E.x, offset, 8);
		meanNormal_E.y += __shfl_down_sync(__activemask(), meanNormal_E.y, offset, 8);
		meanNormal_E.z += __shfl_down_sync(__activemask(), meanNormal_E.z, offset, 8);
	}
	//------------------------------------------------计算的irradiance之和-------------------------------------------------
	//如果可分，则计算的是所有的node的irradiance之和；如果不可分，则计算的是乘以因子的irradiance之和
	glm::vec3 mergeIrradiance = irradiance;
	mergeIrradiance *= mergeAABBSurfaceArea < 1e-6 ? 0.0f : surfaceArea / mergeAABBSurfaceArea;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradiance.x += __shfl_down_sync(__activemask(), mergeIrradiance.x, offset, 8);
		mergeIrradiance.y += __shfl_down_sync(__activemask(), mergeIrradiance.y, offset, 8);
		mergeIrradiance.z += __shfl_down_sync(__activemask(), mergeIrradiance.z, offset, 8);
	}
	//-----------------------------------------------赋值------------------------------------------------------------------
	if (warpLane == blockFirstWarpLane) {
		FzbSVONodeData_PG nodeData;
		nodeData.indivisible = 1;
		nodeData.AABB_G = mergeAABB_G;
		nodeData.AABB_E = mergeAABB_E;
		nodeData.irradiance = mergeIrradiance;
		nodeData.label = 0;
		nodeData.meanNormal_G = meanNormal_G;
		nodeData.meanNormal_E = meanNormal_E;
		OctreeNodes[blockIndex] = nodeData;
	}
}
__global__ void createOctree_device_clusterLayer(FzbSVONodeData_PG* OctreeNodes_children, FzbSVONodeData_PG* OctreeNodes, uint32_t nodeCount, uint32_t clusterIndex) {
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

	bool hasIrradiance = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
	bool hasGeometry = nodeData.AABB_G.leftX != FLT_MAX;

	uint32_t warpHasGeometryMask = __ballot_sync(0xFFFFFFFF, hasGeometry);
	uint32_t blockHasGeometryNodeCount = __popc(warpHasGeometryMask & blockNodeMask);

	//bool indivisible = nodeData.indivisible;
	//uint32_t warpIndivisibleNodeMask = __ballot_sync(0xFFFFFFFF, indivisible);
	//uint32_t blockIndivisibleNodeCount = __popc(warpIndivisibleNodeMask & blockNodeMask);

	if (blockHasGeometryNodeCount == 0) return;	//当前block中node全部没有数据
	if (blockHasGeometryNodeCount == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasGeometry) {
			nodeData.indivisible = 1;
			OctreeNodes[blockIndex] = nodeData;
		}
		return;
	}

	//if (blockIndivisibleNodeCount != 8) {	//有一个可分，就无法聚类
	//	if (warpLane == blockFirstWarpLane) {
	//		nodeData.indivisible = 0;
	//		nodeData.irradiance.x = 1.0f;
	//		nodeData.AABB_G.leftX = 1.0f;
	//		OctreeNodes[blockIndex] = nodeData;
	//	}
	//	return;
	//}
	//---------------------------------------计算整合后的AABB_G-------------------------------------------------
	FzbAABB mergeAABB_G = nodeData.AABB_G;
	getAABBSum(__activemask(), mergeAABB_G, blockFirstWarpLane);
	//---------------------------------------计算整合后的meanNormal_G-------------------------------------------------
	glm::vec3 meanNormal_G = nodeData.meanNormal_G;
	for (int offset = 4; offset > 0; offset /= 2) {
		meanNormal_G.x += __shfl_down_sync(__activemask(), meanNormal_G.x, offset, 8);
		meanNormal_G.y += __shfl_down_sync(__activemask(), meanNormal_G.y, offset, 8);
		meanNormal_G.z += __shfl_down_sync(__activemask(), meanNormal_G.z, offset, 8);
	}
	meanNormal_G /= blockHasGeometryNodeCount;
	//------------------------------------------------irradiance差距判断-------------------------------------------------
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

	if (maxEIrradianceValue == 0.0f) {
		if (warpLane == blockFirstWarpLane) {
			FzbSVONodeData_PG nodeData;
			nodeData.indivisible = 1;
			nodeData.AABB_G = mergeAABB_G;
			initAABB(nodeData.AABB_E);
			nodeData.irradiance = glm::vec3(0.0f);
			nodeData.label = 0;
			nodeData.meanNormal_G = meanNormal_G;
			nodeData.meanNormal_E = glm::vec3(0.0f);
			OctreeNodes[blockIndex] = nodeData;
		}
		return;
	}
	//----------------------------------------------判断忽略--------------------------------------------------------
	uint32_t ignore = 0;
	float relIrradianceRatio = irradianceValue / maxEIrradianceValue;
	if (relIrradianceRatio <= groupSVOUniformData.irradianceRelRatioThreshold) ignore = 1;

	uint32_t blockIgnoreMask = ignore << indexInBlock;
	for (int offset = 4; offset > 0; offset /= 2)
		blockIgnoreMask |= __shfl_down_sync(__activemask(), blockIgnoreMask, offset, 8);
	blockIgnoreMask = __shfl_sync(__activemask(), blockIgnoreMask, blockFirstWarpLane);
	uint32_t blockNotIgnoreCount = 8 - __popc(blockIgnoreMask);
	if (blockNotIgnoreCount == 1) {
		getAABB(mergeAABB_G, blockFirstWarpLane);
		getVec3(meanNormal_G, blockFirstWarpLane);
		if (warpLane == maxEWarpLane) {
			nodeData.indivisible = 1;
			nodeData.AABB_G = mergeAABB_G;
			nodeData.meanNormal_G = meanNormal_G;
			OctreeNodes[blockIndex] = nodeData;
		}
		return;
	}
	//---------------------------------------余弦判断-------------------------------------------------
	//bool doubleSided = glm::length(nodeData.meanNormal_G) <= 0.5f;	//双面的就设为0，防止其余不是双面的进行聚类

	//bool indivisible = true;
	//glm::vec3 normalizeNormal_G = hasGeometry ? glm::normalize(nodeData.meanNormal_G) : glm::vec3(0.0f);
	//for (int blockLane = 0; blockLane < 8; ++blockLane) {
	//	glm::vec3 other_normal = normalizeNormal_G;
	//	uint32_t otherWarpLane = blockFirstWarpLane + blockLane;
	//	getVec3(other_normal, otherWarpLane);

	//	if (!hasGeometry || ((warpHasGeometryMask & (1u << otherWarpLane)) == 0)) continue;	//有一个没有几何，可以聚类
	//	//bool other_doubleSided = __shfl_sync(__activemask(), doubleSided, otherWarpLane);
	//	//if (doubleSided ^ other_doubleSided) {	//一个双面一个不是双面，则无法聚类
	//	//	indivisible = false;
	//	//	continue;
	//	//}

	//	float cosine = glm::dot(other_normal, normalizeNormal_G);
	//	if (cosine <= groupSVOUniformData.cosineDiffThreshold) indivisible = false;
	//}
	//for (int offset = 4; offset > 0; offset /= 2)
	//	indivisible &= __shfl_down_sync(__activemask(), indivisible, offset, 8);
	////indivisible = __shfl_sync(__activemask(), indivisible, blockFirstWarpLane);
	//---------------------------------------计算整合后的AABB-------------------------------------------------
	FzbAABB mergeAABB_E = nodeData.AABB_E;
	if (ignore == 1) mergeAABB_E = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	getAABBSum(__activemask(), mergeAABB_E, blockFirstWarpLane);

	//计算所有不被忽略的node聚类后的AABB的表面积
	float mergeAABBSurfaceArea = 0.0f;
	if (warpLane == blockFirstWarpLane) {
		float lengthX = mergeAABB_E.rightX - mergeAABB_E.leftX;
		float lengthY = mergeAABB_E.rightY - mergeAABB_E.leftY;
		float lengthZ = mergeAABB_E.rightZ - mergeAABB_E.leftZ;
		mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	mergeAABBSurfaceArea = __shfl_sync(__activemask(), mergeAABBSurfaceArea, blockFirstWarpLane);

	float surfaceArea = 0.0f;
	if (hasIrradiance && hasGeometry) {	//有光照才需要surfaceArea来计算缩放因子，虽然说有光照必有几何，但是光栅化可能没有注入进去
		float lengthX = nodeData.AABB_E.rightX - nodeData.AABB_E.leftX;
		float lengthY = nodeData.AABB_E.rightY - nodeData.AABB_E.leftY;
		float lengthZ = nodeData.AABB_E.rightZ - nodeData.AABB_E.leftZ;
		surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
	}
	//-----------------------------------------------计算法线--------------------------------------------------------------
	glm::vec3 meanNormal_E = ignore ? glm::vec3(0.0f) : nodeData.meanNormal_E;
	for (int offset = 4; offset > 0; offset /= 2) {
		meanNormal_E.x += __shfl_down_sync(__activemask(), meanNormal_E.x, offset, 8);
		meanNormal_E.y += __shfl_down_sync(__activemask(), meanNormal_E.y, offset, 8);
		meanNormal_E.z += __shfl_down_sync(__activemask(), meanNormal_E.z, offset, 8);
	}
	//------------------------------------------------计算的irradiance之和-------------------------------------------------
	//如果可分，则计算的是所有的node的irradiance之和；如果不可分，则计算的是乘以因子的irradiance之和
	glm::vec3 mergeIrradiance = nodeData.irradiance;
	mergeIrradiance *= mergeAABBSurfaceArea < 1e-6 ? 0.0f : surfaceArea / mergeAABBSurfaceArea;
	for (int offset = 4; offset > 0; offset /= 2) {
		mergeIrradiance.x += __shfl_down_sync(__activemask(), mergeIrradiance.x, offset, 8);
		mergeIrradiance.y += __shfl_down_sync(__activemask(), mergeIrradiance.y, offset, 8);
		mergeIrradiance.z += __shfl_down_sync(__activemask(), mergeIrradiance.z, offset, 8);
	}
	//-----------------------------------------------赋值------------------------------------------------------------------
	if (warpLane == blockFirstWarpLane) {
		nodeData.indivisible = 1;
		nodeData.AABB_G = mergeAABB_G;
		nodeData.AABB_E = mergeAABB_E;
		nodeData.irradiance = mergeIrradiance;
		nodeData.meanNormal_G = meanNormal_G;
		nodeData.meanNormal_E = meanNormal_E;
		OctreeNodes[blockIndex] = nodeData;
	}
}
__global__ void createOctree_device_noClusterLayer(FzbSVONodeData_PG* OctreeNodes_children, FzbSVONodeData_PG* OctreeNodes, uint32_t nodeCount) {
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

	bool hasIrradiance = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
	bool hasGeometry = nodeData.AABB_G.leftX != FLT_MAX;

	uint32_t warpHasGeometryMask = __ballot_sync(0xFFFFFFFF, hasGeometry);
	uint32_t blockHasGeometryNodeCount = __popc(warpHasGeometryMask & blockNodeMask);

	if (blockHasGeometryNodeCount == 0) return;	//当前block中node全部没有数据
	if (blockHasGeometryNodeCount == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasGeometry) {
			nodeData.indivisible = 0;
			OctreeNodes[blockIndex] = nodeData;
		}
	}else if (warpLane == __ffs(warpHasGeometryMask & blockNodeMask) - 1) {
		nodeData.indivisible = 0;
		nodeData.irradiance.x = 1.0f;
		OctreeNodes[blockIndex] = nodeData;
	}
}
*/

__device__ bool checkPointInAABB(const glm::vec3& point, const FzbAABB& AABB) {
	if (point.x < AABB.leftX || point.x > AABB.rightX ||
		point.y < AABB.leftY || point.y > AABB.rightY ||
		point.z < AABB.leftZ || point.z > AABB.rightZ) return false;
	return true;
}
__device__ bool checkLightInNode(FzbAABB& AABB, FzbRayTracingLightSet& groupLightSet) {
	for (int i = 0; i < groupLightSet.pointLightCount; ++i) {
		const FzbRayTracingPointLight light = groupLightSet.pointLightInfoArray[i];
		return checkPointInAABB(light.worldPos, AABB);
	}
	for (int i = 0; i < groupLightSet.areaLightCount; ++i) {
		const FzbRayTracingAreaLight light = groupLightSet.areaLightInfoArray[i];
		bool result = false;
		result |= checkPointInAABB(light.worldPos, AABB);
		result |= checkPointInAABB(light.worldPos + light.edge0, AABB);
		result |= checkPointInAABB(light.worldPos + light.edge1, AABB);
		result |= checkPointInAABB(light.worldPos + light.edge0 + light.edge1, AABB);
		return result;
	}
}

__global__ void createOctree_device_VGB(
	const FzbVoxelData_PG* __restrict__ VGB, uint32_t voxelCount,
	FzbSVONodeData_PG_G* OctreeNodes_G, FzbSVONodeData_PG_E* OctreeNodes_E) 
{
	__shared__ FzbVGBUniformData groupVGBUniformData;
	__shared__ FzbSVOUnformData groupSVOUniformData;

	//__shared__ FzbRayTracingPointLight groupPointLightInfoArray[maxPointLightCount];	//512B
	//__shared__ FzbRayTracingAreaLight grouprAreaLightInfoArray[maxAreaLightCount];		//692B
	//__shared__ FzbRayTracingLightSet groupLightSet;

	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t warpIndex = threadIdx.x / 32;
	uint32_t warpLane = threadIdx.x & 31;

	if (threadIndex >= voxelCount) return;		//voxelCount一定是32的整数倍，return不影响洗牌操纵
	//if (threadIdx.x < systemPointLightCount) groupPointLightInfoArray[threadIdx.x] = systemPointLightInfoArray[threadIdx.x];
	//if (threadIdx.x < systemAreaLightCount) grouprAreaLightInfoArray[threadIdx.x] = systemAreaLightInfoArray[threadIdx.x];
	if (threadIdx.x == 0) {
		groupVGBUniformData = systemVGBUniformData;
		groupSVOUniformData = systemSVOUniformData;

		//groupLightSet.pointLightCount = systemPointLightCount;
		//groupLightSet.areaLightCount = systemAreaLightCount;
		//groupLightSet.pointLightInfoArray = groupPointLightInfoArray;
		//groupLightSet.areaLightInfoArray = grouprAreaLightInfoArray;
	}
	__syncthreads();

	//这里的block指的是父级node
	uint32_t indexInBlock = threadIndex & 7;	//在8个兄弟node中的索引
	uint32_t blockIndex = threadIndex / 8;		//block在全局的索引
	uint32_t blockFirstWarpLane = (blockIndex & 3) * 8;	//当前block在warp中的位索引
	uint32_t blockNodeMask = 0xff << blockFirstWarpLane;
	uint32_t blockIndexInGroup = threadIdx.x / 8;

	FzbVoxelData_PG voxelData = VGB[threadIndex];
	FzbAABB AABB = {
			__int_as_float(voxelData.AABB.leftX),
			__int_as_float(voxelData.AABB.rightX),
			__int_as_float(voxelData.AABB.leftY),
			__int_as_float(voxelData.AABB.rightY),
			__int_as_float(voxelData.AABB.leftZ),
			__int_as_float(voxelData.AABB.rightZ)
	};
	//-----------------------------------------------处理OctreeNode_G---------------------------------------------------------
	bool hasGeometry = AABB.leftX != FLT_MAX;
	uint32_t warpHasGeometryMask = __ballot_sync(0xFFFFFFFF, hasGeometry);
	uint32_t blockHasGeometryNodeCount = __popc(warpHasGeometryMask & blockNodeMask);

	glm::vec3 normal = voxelData.meanNormal_G.w > 0.0f ? glm::vec3(voxelData.meanNormal_G) / voxelData.meanNormal_G.w : glm::vec3(0.0f);
	if (blockHasGeometryNodeCount == 0) {}	//几何注入不是百分百的，可能没有注入进去，但是这个地方有光照，那么不能直接return，还有处理E
	else if (blockHasGeometryNodeCount == 1) {
		if (hasGeometry) {
			FzbSVONodeData_PG_G nodeData;
			nodeData.indivisible = 1;
			nodeData.label = 0;
			nodeData.AABB = AABB;
			nodeData.entropy = 0.0f;
			nodeData.meanNormal = normal;
			OctreeNodes_G[blockIndex] = nodeData;
		}
	}
	else {
		//---------------------------------------计算整合后的AABB_G-------------------------------------------------
		FzbAABB mergeAABB_G = AABB;
		getAABBSum(__activemask(), mergeAABB_G, blockFirstWarpLane);
		//---------------------------------------计算整合后的meanNormal_G-------------------------------------------------
		glm::vec3 meanNormal_G = normal;
		for (int offset = 4; offset > 0; offset /= 2) {
			meanNormal_G.x += __shfl_down_sync(__activemask(), meanNormal_G.x, offset, 8);
			meanNormal_G.y += __shfl_down_sync(__activemask(), meanNormal_G.y, offset, 8);
			meanNormal_G.z += __shfl_down_sync(__activemask(), meanNormal_G.z, offset, 8);
		}
		meanNormal_G /= blockHasGeometryNodeCount;
		//---------------------------------------计算表面积------------------------------------------
		float mergeAABBSurfaceArea = 0.0f;
		if (warpLane == blockFirstWarpLane) {
			float lengthX = mergeAABB_G.rightX - mergeAABB_G.leftX;
			float lengthY = mergeAABB_G.rightY - mergeAABB_G.leftY;
			float lengthZ = mergeAABB_G.rightZ - mergeAABB_G.leftZ;
			mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		}
		mergeAABBSurfaceArea = __shfl_sync(__activemask(), mergeAABBSurfaceArea, blockFirstWarpLane);

		float surfaceArea = 0.0f;
		if (hasGeometry) {	//有光照才需要surfaceArea来计算缩放因子，虽然说有光照必有几何，但是光栅化可能没有注入进去
			float lengthX = voxelData.AABB.rightX - voxelData.AABB.leftX;
			float lengthY = voxelData.AABB.rightY - voxelData.AABB.leftY;
			float lengthZ = voxelData.AABB.rightZ - voxelData.AABB.leftZ;
			surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		}

		float surfaceAreaSum = surfaceArea;
		for (int offset = 4; offset > 0; offset /= 2)
			surfaceAreaSum += __shfl_down_sync(__activemask(), surfaceAreaSum, offset, 8);
		//---------------------------------------计算node中的熵---------------------------------------
		bool indivisible = true;
		float entropy = 0.0f;
		if (warpLane == blockFirstWarpLane) {
			float normalEntropy = max(1.0f - glm::length(meanNormal_G), 0.0f);
			float surfaceAreaEntropy = max(1.0f - surfaceAreaSum / mergeAABBSurfaceArea, 0.0f);
			entropy = (normalEntropy + surfaceAreaEntropy) * 0.5f;
			indivisible = entropy < groupSVOUniformData.entropyThreshold;
		}
		indivisible = __shfl_sync(__activemask(), indivisible, blockFirstWarpLane);

		if (!indivisible) {
			indivisible = true;
			glm::vec3 normalizeNormal = normal == glm::vec3(0.0f) ? glm::vec3(0.0f) : glm::normalize(normal);
			for (int blockLane = 0; blockLane < 8; ++blockLane) {
				uint32_t otherWarpLane = blockLane + blockFirstWarpLane;
				glm::vec3 other_normal = normalizeNormal;
				getVec3(other_normal, otherWarpLane);
				if (normalizeNormal == glm::vec3(0.0f) || other_normal == glm::vec3(0.0f)) continue;
				float cosine = abs(glm::dot(normalizeNormal, other_normal));
				if (cosine < 0.707f) indivisible = false;
			}
			for (int offset = 4; offset > 0; offset /= 2)
				indivisible &= __shfl_down_sync(__activemask(), indivisible, offset, 8);
			//indivisible = __shfl_sync(__activemask(), indivisible, blockFirstWarpLane);
		}
		//----------------------------------------赋值--------------------------------------------------------
		if (warpLane == blockFirstWarpLane) {
			FzbSVONodeData_PG_G nodeData;
			nodeData.indivisible = indivisible;
			nodeData.label = 0;
			nodeData.AABB = mergeAABB_G;
			nodeData.entropy = entropy;
			nodeData.meanNormal = meanNormal_G;
			OctreeNodes_G[blockIndex] = nodeData;
		}
	}
	//-----------------------------------------------处理OctreeNode_E---------------------------------------------------------
	bool hasIrradiance = voxelData.irradiance.x + voxelData.irradiance.y + voxelData.irradiance.z > 0.0f;
	glm::vec3 irradiance = glm::vec3(0.0f);
	if (hasIrradiance) irradiance = voxelData.irradiance / voxelData.irradiance.w;
	uint32_t warpHasIrradianceMask = __ballot_sync(0xFFFFFFFF, hasIrradiance);
	uint32_t blockHasIrradianceNodeCount = __popc(warpHasIrradianceMask & blockNodeMask);

	if (blockHasIrradianceNodeCount == 0) return;
	if (blockHasIrradianceNodeCount == 1) {
		if (hasIrradiance) {
			//if (!checkLightInNode(AABB, groupLightSet)) return;	//如果光源不在AABB中，则认为是噪声，直接去掉
			FzbSVONodeData_PG_E nodeData;
			nodeData.indivisible = 1;
			nodeData.label = 0;
			nodeData.AABB = AABB;
			nodeData.notIgnoreRatio = 1;
			nodeData.irradiance = irradiance;
			OctreeNodes_E[blockIndex] = nodeData;
		}
		return;
	}
	else {
		//-----------------------------------------找到最亮的node----------------------------------------------
		float irradianceValue = glm::length(irradiance);
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
		//-----------------------------------------判断忽略node----------------------------------------------
		uint32_t ignore = 0;
		float relIrradianceRatio = irradianceValue / maxEIrradianceValue;
		if (relIrradianceRatio <= groupSVOUniformData.irradianceRelRatioThreshold) ignore = 1;

		uint32_t blockIgnoreMask = ignore << indexInBlock;
		for (int offset = 4; offset > 0; offset /= 2)
			blockIgnoreMask |= __shfl_down_sync(__activemask(), blockIgnoreMask, offset, 8);
		blockIgnoreMask = __shfl_sync(__activemask(), blockIgnoreMask, blockFirstWarpLane);
		uint32_t blockNotIgnoreCount = 8 - __popc(blockIgnoreMask);
		if (blockNotIgnoreCount == 1) {
			ignore = !hasIrradiance;	//如果只有一个重要node则进行平均

			blockIgnoreMask = ignore << indexInBlock;
			for (int offset = 4; offset > 0; offset /= 2)
				blockIgnoreMask |= __shfl_down_sync(__activemask(), blockIgnoreMask, offset, 8);
			blockIgnoreMask = __shfl_sync(__activemask(), blockIgnoreMask, blockFirstWarpLane);
			blockNotIgnoreCount = 8 - __popc(blockIgnoreMask);
		}
		//---------------------------------------------计算Node的法线---------------------------------------------------------
		glm::vec3 meanNormal_E = ignore ? glm::vec3(0.0f) : voxelData.meanNormal_E / length(glm::vec3(voxelData.irradiance));
		for (int offset = 4; offset > 0; offset /= 2) {
			meanNormal_E.x += __shfl_down_sync(__activemask(), meanNormal_E.x, offset, 8);
			meanNormal_E.y += __shfl_down_sync(__activemask(), meanNormal_E.y, offset, 8);
			meanNormal_E.z += __shfl_down_sync(__activemask(), meanNormal_E.z, offset, 8);
		}
		meanNormal_E /= blockNotIgnoreCount;
		//---------------------------------------计算整合后的AABB-------------------------------------------------
		FzbAABB mergeAABB_E = AABB;
		if (ignore == 1) mergeAABB_E = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
		getAABBSum(__activemask(), mergeAABB_E, blockFirstWarpLane);

		//计算所有不被忽略的node聚类后的AABB的表面积
		float mergeAABBSurfaceArea = 0.0f;
		if (warpLane == blockFirstWarpLane) {
			float lengthX = mergeAABB_E.rightX - mergeAABB_E.leftX;
			float lengthY = mergeAABB_E.rightY - mergeAABB_E.leftY;
			float lengthZ = mergeAABB_E.rightZ - mergeAABB_E.leftZ;
			mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		}
		mergeAABBSurfaceArea = __shfl_sync(__activemask(), mergeAABBSurfaceArea, blockFirstWarpLane);

		float surfaceArea = 0.0f;
		if (hasIrradiance) {	//有光照才需要surfaceArea来计算缩放因子，虽然说有光照必有几何，但是光栅化可能没有注入进去
			float lengthX = AABB.rightX - AABB.leftX;
			float lengthY = AABB.rightY - AABB.leftY;
			float lengthZ = AABB.rightZ - AABB.leftZ;
			surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		}
		//------------------------------------------------计算的irradiance之和-------------------------------------------------
		glm::vec3 mergeIrradiance = irradiance;
		float surfaceAreaRatio = mergeAABBSurfaceArea < 1e-6 ? 0.0f : surfaceArea / mergeAABBSurfaceArea;
		//mergeIrradiance *= surfaceAreaRatio;
		for (int offset = 4; offset > 0; offset /= 2) {
			mergeIrradiance.x += __shfl_down_sync(__activemask(), mergeIrradiance.x, offset, 8);
			mergeIrradiance.y += __shfl_down_sync(__activemask(), mergeIrradiance.y, offset, 8);
			mergeIrradiance.z += __shfl_down_sync(__activemask(), mergeIrradiance.z, offset, 8);
		}
		float notIgnoreRatio = ignore ? 0.0f : irradianceValue;	// *surfaceAreaRatio;
		for (int offset = 4; offset > 0; offset /= 2) 
			notIgnoreRatio += __shfl_down_sync(__activemask(), notIgnoreRatio, offset, 8);
		notIgnoreRatio = min(notIgnoreRatio / glm::length(mergeIrradiance), 1.0f);
		//-----------------------------------------------赋值------------------------------------------------------------------
		if (warpLane == blockFirstWarpLane) {
			FzbSVONodeData_PG_E nodeData;
			nodeData.indivisible = 1;	//这里后面可以搞复杂一点，先实现功能再说
			nodeData.label = 0;
			nodeData.AABB = mergeAABB_E;
			nodeData.notIgnoreRatio = notIgnoreRatio;
			nodeData.irradiance = mergeIrradiance;
			nodeData.meanNormal = meanNormal_E;
			OctreeNodes_E[blockIndex] = nodeData;
		}
	}
}
__global__ void createOctree_device_clusterLayer(
	uint32_t nodeCount, uint32_t clusterIndex,
	FzbSVONodeData_PG_G* OctreeNodes_G_children, FzbSVONodeData_PG_G* OctreeNodes_G,
	FzbSVONodeData_PG_E* OctreeNodes_E_children, FzbSVONodeData_PG_E* OctreeNodes_E)
{
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

	//---------------------------------------------处理OctreeNode_G--------------------------------------------------------
	FzbSVONodeData_PG_G nodeData_G = OctreeNodes_G_children[threadIndex];
	bool hasGeometry = nodeData_G.AABB.leftX != FLT_MAX;

	uint32_t warpHasGeometryMask = __ballot_sync(0xFFFFFFFF, hasGeometry);
	uint32_t blockHasGeometryNodeCount = __popc(warpHasGeometryMask & blockNodeMask);

	if (blockHasGeometryNodeCount == 0) {}	//几何注入不是百分百的，可能没有注入进去，但是这个地方有光照，那么不能直接return，还有处理E
	else if (blockHasGeometryNodeCount == 1) {
		if (hasGeometry) {
			//nodeData_G.indivisible = 1;
			//nodeData_G.label = 0;
			//nodeData_G.AABB = nodeData_G.AABB;
			//nodeData_G.meanNormal = nodeData_G.meanNormal;
			OctreeNodes_G[blockIndex] = nodeData_G;
		}
	}
	else {
		//---------------------------------------计算整合后的AABB_G-------------------------------------------------
		FzbAABB mergeAABB_G = nodeData_G.AABB;
		getAABBSum(__activemask(), mergeAABB_G, blockFirstWarpLane);
		//---------------------------------------计算整合后的meanNormal_G-------------------------------------------------
		glm::vec3 meanNormal_G = nodeData_G.meanNormal;
		for (int offset = 4; offset > 0; offset /= 2) {
			meanNormal_G.x += __shfl_down_sync(__activemask(), meanNormal_G.x, offset, 8);
			meanNormal_G.y += __shfl_down_sync(__activemask(), meanNormal_G.y, offset, 8);
			meanNormal_G.z += __shfl_down_sync(__activemask(), meanNormal_G.z, offset, 8);
		}
		meanNormal_G /= blockHasGeometryNodeCount;
		//---------------------------------------根据表面积判断能不能分---------------------------------------
		float mergeAABBSurfaceArea = 0.0f;
		if (warpLane == blockFirstWarpLane) {
			float lengthX = mergeAABB_G.rightX - mergeAABB_G.leftX;
			float lengthY = mergeAABB_G.rightY - mergeAABB_G.leftY;
			float lengthZ = mergeAABB_G.rightZ - mergeAABB_G.leftZ;
			mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		}
		mergeAABBSurfaceArea = __shfl_sync(__activemask(), mergeAABBSurfaceArea, blockFirstWarpLane);

		float surfaceArea = 0.0f;
		if (hasGeometry) {	//有光照才需要surfaceArea来计算缩放因子，虽然说有光照必有几何，但是光栅化可能没有注入进去
			float lengthX = nodeData_G.AABB.rightX - nodeData_G.AABB.leftX;
			float lengthY = nodeData_G.AABB.rightY - nodeData_G.AABB.leftY;
			float lengthZ = nodeData_G.AABB.rightZ - nodeData_G.AABB.leftZ;
			surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		}

		float surfaceAreaSum = surfaceArea;
		for (int offset = 4; offset > 0; offset /= 2)
			surfaceAreaSum += __shfl_down_sync(__activemask(), surfaceAreaSum, offset, 8);
		//---------------------------------------根据法线判断能不能分---------------------------------------
		bool indivisible = true;
		float entropy = 0.0f;
		if (warpLane == blockFirstWarpLane) {
			float normalEntropy = max(1.0f - glm::length(meanNormal_G), 0.0f);
			float surfaceAreaEntropy = max(1.0f - surfaceAreaSum / mergeAABBSurfaceArea, 0.0f);
			entropy = (normalEntropy + surfaceAreaEntropy) * 0.5f;
			indivisible = entropy < groupSVOUniformData.entropyThreshold;
		}
		indivisible = __shfl_sync(__activemask(), indivisible, blockFirstWarpLane);

		if (!indivisible) {
			indivisible = true;
			glm::vec3 normalizeNormal = nodeData_G.meanNormal == glm::vec3(0.0f) ? glm::vec3(0.0f) : glm::normalize(nodeData_G.meanNormal);
			for (int blockLane = 0; blockLane < 8; ++blockLane) {
				uint32_t otherWarpLane = blockLane + blockFirstWarpLane;
				glm::vec3 other_normal = normalizeNormal;
				getVec3(other_normal, otherWarpLane);
				if (normalizeNormal == glm::vec3(0.0f) || other_normal == glm::vec3(0.0f)) continue;
				float cosine = abs(glm::dot(normalizeNormal, other_normal));
				if (cosine < 0.707f) indivisible = false;
			}
			for (int offset = 4; offset > 0; offset /= 2)
				indivisible &= __shfl_down_sync(__activemask(), indivisible, offset, 8);
			//indivisible = __shfl_sync(__activemask(), indivisible, blockFirstWarpLane);
		}
		//----------------------------------------------赋值-------------------------------------------
		if (warpLane == blockFirstWarpLane) {
			nodeData_G.indivisible = indivisible;
			//nodeData_G.label = 0;
			nodeData_G.AABB = mergeAABB_G;
			nodeData_G.entropy = entropy;
			nodeData_G.meanNormal = meanNormal_G;
			OctreeNodes_G[blockIndex] = nodeData_G;
		}
	}
	//---------------------------------------------处理OctreeNode_E--------------------------------------------------------
	FzbSVONodeData_PG_E nodeData_E = OctreeNodes_E_children[threadIndex];
	bool hasIrradiance = nodeData_E.irradiance.x + nodeData_E.irradiance.y + nodeData_E.irradiance.z > 0.0f;

	uint32_t warpHasIrradianceMask = __ballot_sync(0xFFFFFFFF, hasIrradiance);
	uint32_t blockHasIrradianceNodeCount = __popc(warpHasIrradianceMask & blockNodeMask);

	if (blockHasIrradianceNodeCount == 0) return;
	if (blockHasIrradianceNodeCount == 1) {
		if (hasIrradiance) {
			OctreeNodes_E[blockIndex] = nodeData_E;
		}
	}
	else {
		//-----------------------------------------找到最亮的node----------------------------------------------
		float irradianceValue = glm::length(nodeData_E.irradiance);
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
		//-----------------------------------------判断忽略node----------------------------------------------
		uint32_t ignore = 0;
		float relIrradianceRatio = irradianceValue / maxEIrradianceValue;
		if (relIrradianceRatio <= groupSVOUniformData.irradianceRelRatioThreshold) ignore = 1;

		uint32_t blockIgnoreMask = ignore << indexInBlock;
		for (int offset = 4; offset > 0; offset /= 2)
			blockIgnoreMask |= __shfl_down_sync(__activemask(), blockIgnoreMask, offset, 8);
		blockIgnoreMask = __shfl_sync(__activemask(), blockIgnoreMask, blockFirstWarpLane);
		uint32_t blockNotIgnoreCount = 8 - __popc(blockIgnoreMask);		
		//---------------------------------------------计算Node的法线---------------------------------------------------------
		glm::vec3 meanNormal_E = ignore ? glm::vec3(0.0f) : nodeData_E.meanNormal;
		for (int offset = 4; offset > 0; offset /= 2) {
			meanNormal_E.x += __shfl_down_sync(__activemask(), meanNormal_E.x, offset, 8);
			meanNormal_E.y += __shfl_down_sync(__activemask(), meanNormal_E.y, offset, 8);
			meanNormal_E.z += __shfl_down_sync(__activemask(), meanNormal_E.z, offset, 8);
		}
		meanNormal_E /= blockNotIgnoreCount;
		//---------------------------------------计算整合后的AABB-------------------------------------------------
		FzbAABB mergeAABB_E = nodeData_E.AABB;
		if (ignore == 1) mergeAABB_E = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
		getAABBSum(__activemask(), mergeAABB_E, blockFirstWarpLane);

		//计算所有不被忽略的node聚类后的AABB的表面积
		float mergeAABBSurfaceArea = 0.0f;
		if (warpLane == blockFirstWarpLane) {
			float lengthX = mergeAABB_E.rightX - mergeAABB_E.leftX;
			float lengthY = mergeAABB_E.rightY - mergeAABB_E.leftY;
			float lengthZ = mergeAABB_E.rightZ - mergeAABB_E.leftZ;
			mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		}
		mergeAABBSurfaceArea = __shfl_sync(__activemask(), mergeAABBSurfaceArea, blockFirstWarpLane);

		float surfaceArea = 0.0f;
		if (hasIrradiance) {	//有光照才需要surfaceArea来计算缩放因子，虽然说有光照必有几何，但是光栅化可能没有注入进去
			float lengthX = nodeData_E.AABB.rightX - nodeData_E.AABB.leftX;
			float lengthY = nodeData_E.AABB.rightY - nodeData_E.AABB.leftY;
			float lengthZ = nodeData_E.AABB.rightZ - nodeData_E.AABB.leftZ;
			surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
		}
		//-----------------------------------------------根据表面积判断是否可以聚类------------------------------------------
		bool indivisible = true;
		float notIgnoreSurfaceAreaSum = ignore ? 0.0f : surfaceArea;
		for (int offset = 4; offset > 0; offset /= 2)
			notIgnoreSurfaceAreaSum += __shfl_down_sync(__activemask(), notIgnoreSurfaceAreaSum, offset, 8);
		if (warpLane == blockFirstWarpLane)
			indivisible = (notIgnoreSurfaceAreaSum / mergeAABBSurfaceArea) > 0.5f;
		//------------------------------------------------计算的irradiance之和-------------------------------------------------
		glm::vec3 mergeIrradiance = nodeData_E.irradiance;
		float surfaceAreaRatio = mergeAABBSurfaceArea < 1e-6 ? 0.0f : surfaceArea / mergeAABBSurfaceArea;
		//mergeIrradiance *= surfaceAreaRatio;
		for (int offset = 4; offset > 0; offset /= 2) {
			mergeIrradiance.x += __shfl_down_sync(__activemask(), mergeIrradiance.x, offset, 8);
			mergeIrradiance.y += __shfl_down_sync(__activemask(), mergeIrradiance.y, offset, 8);
			mergeIrradiance.z += __shfl_down_sync(__activemask(), mergeIrradiance.z, offset, 8);
		}
		float notIgnoreRatio = ignore ? 0.0f : irradianceValue * nodeData_E.notIgnoreRatio;	// * surfaceAreaRatio
		for (int offset = 4; offset > 0; offset /= 2)
			notIgnoreRatio += __shfl_down_sync(__activemask(), notIgnoreRatio, offset, 8);
		notIgnoreRatio = min(notIgnoreRatio / glm::length(mergeIrradiance), 1.0f);
		//-----------------------------------------------赋值------------------------------------------------------------------
		if (warpLane == blockFirstWarpLane) {
			nodeData_E.indivisible = indivisible;	//这里后面可以搞复杂一点，先实现功能再说
			//nodeData.label = 0;
			nodeData_E.AABB = mergeAABB_E;
			nodeData_E.notIgnoreRatio = notIgnoreRatio;
			nodeData_E.irradiance = mergeIrradiance;
			nodeData_E.meanNormal = meanNormal_E;
			OctreeNodes_E[blockIndex] = nodeData_E;
		}
	}
}
__global__ void createOctree_device_noClusterLayer(
	uint32_t nodeCount,
	FzbSVONodeData_PG_G* OctreeNodes_G_children, FzbSVONodeData_PG_G* OctreeNodes_G,
	FzbSVONodeData_PG_E* OctreeNodes_E_children, FzbSVONodeData_PG_E* OctreeNodes_E)
{
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

	//------------------------------------------------处理Octree_G--------------------------------------------
	FzbSVONodeData_PG_G nodeData_G = OctreeNodes_G_children[threadIndex];
	bool hasGeometry = nodeData_G.AABB.leftX != FLT_MAX;

	uint32_t warpHasGeometryMask = __ballot_sync(0xFFFFFFFF, hasGeometry);
	uint32_t blockHasGeometryNodeCount = __popc(warpHasGeometryMask & blockNodeMask);

	if (blockHasGeometryNodeCount == 0) {}
	else if (blockHasGeometryNodeCount == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasGeometry) {
			nodeData_G.indivisible = 1;
			OctreeNodes_G[blockIndex] = nodeData_G;
		}
	}
	else{
		FzbAABB mergeAABB_G = nodeData_G.AABB;
		getAABBSum(__activemask(), mergeAABB_G, blockFirstWarpLane);
		if (warpLane == blockFirstWarpLane) {
			nodeData_G.indivisible = 0;
			nodeData_G.AABB = mergeAABB_G;
			OctreeNodes_G[blockIndex] = nodeData_G;
		}
	}
	//------------------------------------------------处理Octree_E--------------------------------------------
	FzbSVONodeData_PG_E nodeData_E = OctreeNodes_E_children[threadIndex];
	bool hasIrradiance = nodeData_E.irradiance.x + nodeData_E.irradiance.y + nodeData_E.irradiance.z > 0.0f;

	uint32_t warpHasIrradianceMask = __ballot_sync(0xFFFFFFFF, hasIrradiance);
	uint32_t blockHasIrradianceNodeCount = __popc(warpHasIrradianceMask & blockNodeMask);

	if (blockHasIrradianceNodeCount == 0) return;
	if (blockHasIrradianceNodeCount == 1) {	//只有一个有值的node，则直接赋值即可
		if (hasIrradiance) {
			nodeData_E.indivisible = 1;
			OctreeNodes_E[blockIndex] = nodeData_E;
		}
	}
	else if (warpLane == blockFirstWarpLane) {
		nodeData_E.indivisible = 0;
		nodeData_E.AABB.leftX = 1.0f;
		nodeData_E.irradiance.x = 1.0f;
		OctreeNodes_E[blockIndex] = nodeData_E;
	}
}

void FzbSVOCuda_PG::createOctreeNodes() {
	uint32_t voxelCount = std::pow(setting.voxelNum, 3);
	uint32_t blockSize = createSVOKernelBlockSize;
	uint32_t gridSize = (voxelCount + blockSize - 1) / blockSize;

	FzbSVONodeData_PG_G* octreeNode_G = OctreeNodes_multiLayer_G[SVONodes_maxDepth - 2];
	FzbSVONodeData_PG_E* octreeNode_E = OctreeNodes_multiLayer_E[SVONodes_maxDepth - 2];
	createOctree_device_VGB <<<gridSize, blockSize, 0, stream>>> (VGB, voxelCount, octreeNode_G, octreeNode_E);
	for (int i = SVONodes_maxDepth - 2; i > 1; --i) {
		FzbSVONodeData_PG_G* octreeNodes_G_children = OctreeNodes_multiLayer_G[i];
		octreeNode_G = OctreeNodes_multiLayer_G[i - 1];

		FzbSVONodeData_PG_E* octreeNodes_E_children = OctreeNodes_multiLayer_E[i];
		octreeNode_E = OctreeNodes_multiLayer_E[i - 1];

		uint32_t nodeCount = pow(2, i);
		uint32_t nodeTotalCount = nodeCount * nodeCount * nodeCount;
		blockSize = nodeTotalCount > createSVOKernelBlockSize ? createSVOKernelBlockSize : nodeTotalCount;
		gridSize = (nodeTotalCount + blockSize - 1) / blockSize;

		if (i > 3) createOctree_device_clusterLayer << <gridSize, blockSize, 0, stream >> >
			(
				nodeCount, SVONodes_maxDepth - i,
				octreeNodes_G_children, octreeNode_G,
				octreeNodes_E_children, octreeNode_E
			);
		else createOctree_device_noClusterLayer << <gridSize, blockSize, 0, stream >> > 
			(
				nodeCount,
				octreeNodes_G_children, octreeNode_G,
				octreeNodes_E_children, octreeNode_E
			);
	}
	checkKernelFunction();
}

__global__ void initOctree_G(FzbSVONodeData_PG_G* Octree, uint32_t svoCount) {
	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= svoCount) return;

	FzbSVONodeData_PG_G data;
	data.indivisible = 1;
	data.label = 0;
	initAABB(data.AABB);
	data.entropy = 0.0f;
	data.meanNormal = glm::vec3(0.0f);
	Octree[threadIndex] = data;
}
__global__ void initOctree_E(FzbSVONodeData_PG_E* Octree, uint32_t svoCount) {
	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= svoCount) return;

	FzbSVONodeData_PG_E data;
	data.indivisible = 1;
	data.label = 0;
	initAABB(data.AABB);
	data.notIgnoreRatio = 1.0f;
	data.irradiance = glm::vec3(0.0f);
	data.meanNormal = glm::vec3(0.0f);
	Octree[threadIndex] = data;
}
void FzbSVOCuda_PG::initCreateOctreeNodesSource(bool allocate) {
	this->OctreeNodes_multiLayer_G.resize(SVONodes_maxDepth);
	for (int i = 0; i < SVONodes_maxDepth; ++i) {
		uint32_t nodeCount = std::pow(8, i);
		if (nodeCount == 1 || nodeCount == pow(setting.voxelNum, 3)) {	 //不存储根节点和叶节点
			this->OctreeNodes_multiLayer_G[i] = nullptr;
			continue;
		}
		if(allocate) CHECK(cudaMalloc((void**)&this->OctreeNodes_multiLayer_G[i], nodeCount * sizeof(FzbSVONodeData_PG_G)));
		uint32_t blockSize = nodeCount > 1024 ? 1024 : nodeCount;
		uint32_t gridSize = (nodeCount + blockSize - 1) / blockSize;
		initOctree_G << <gridSize, blockSize >> > (this->OctreeNodes_multiLayer_G[i], nodeCount);
	}

	this->OctreeNodes_multiLayer_E.resize(SVONodes_maxDepth);
	for (int i = 0; i < SVONodes_maxDepth; ++i) {
		uint32_t nodeCount = std::pow(8, i);
		if (nodeCount == 1 || nodeCount == pow(setting.voxelNum, 3)) {	 //不存储根节点和叶节点
			this->OctreeNodes_multiLayer_E[i] = nullptr;
			continue;
		}
		if (allocate) CHECK(cudaMalloc((void**)&this->OctreeNodes_multiLayer_E[i], nodeCount * sizeof(FzbSVONodeData_PG_E)));
		uint32_t blockSize = nodeCount > 1024 ? 1024 : nodeCount;
		uint32_t gridSize = (nodeCount + blockSize - 1) / blockSize;
		initOctree_E << <gridSize, blockSize >> > (this->OctreeNodes_multiLayer_E[i], nodeCount);
	}
}