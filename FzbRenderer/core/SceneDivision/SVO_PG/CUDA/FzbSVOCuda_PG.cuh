#pragma once

#include "../../../CUDA/vulkanCudaInterop.cuh"
#include "../../../CUDA/commonCudaFunction.cuh"
#include "../../../CUDA/commonStruct.cuh"
#include "../../../RayTracing/CUDA/FzbRayTracingInitSource.cuh"

#ifndef CREATE_SVO_PATH_GUIDING_CUH
#define CREATE_SVO_PATH_GUIDING_CUH

struct FzbSVOSetting_PG {
	uint32_t voxelNum = 64;
	bool useCube = true;
	bool useOneArray = true;
	bool useOBB = true;
};
__host__ __device__ struct FzbVoxelData_PG {
	uint hasData;
	glm::vec3 irradiance;
	FzbAABBUint AABB;
};
struct FzbSVONodeData_PG {
	uint32_t indivisible;
	float pdf;	//去掉影响小的子node后需要弥补的pdf
	//uint32_t shuffleKey;
	uint32_t label;		//该node是当前层中第几个可分的node
	FzbAABB AABB;
	glm::vec3 irradiance;
};

struct FzbSVONodeThreadBlockInfo {
	uint32_t divisibleNodeCount;	//有值且可分的node数量
	uint32_t indivisibleNodeCount;	//有值且可分的node数量
};
struct FzbSVONodeTempInfo {
	uint32_t nodeIndex;		//在octree中的索引
	uint32_t storageIndex;	//在svo中的索引
	uint32_t label;
	uint32_t threadBlockIndex;
};
struct FzbSVOLayerInfo {
	uint32_t divisibleNodeCount;
	uint32_t indivisibleNodeCount;
};
struct FzbSVOIndivisibleNodeInfo {
	uint32_t nodeLayer;
	uint32_t nodeIndex;	//在svo的一层的索引
};

struct FzbVGBUniformData {
	uint32_t voxelCount;
	glm::vec3 voxelSize;
	glm::vec3 voxelStartPos;
};
struct FzbSVOUnformData {
	float surfaceAreaThreshold = 3.0f;	//如何合并后的AABB的表面超过原有子node的表面积之和的surfaceAreaThreshold倍，则认为不能合并
	float irradianceThreshold = 10.0f;	//如果某个两个兄弟node的irradiance之比超过irradianceThreshold，则认为不能合并
	float ignoreIrradianceValueThreshold = 10.0f;
};

//----------------------------------------------常量
extern __constant__ FzbVGBUniformData systemVGBUniformData;
extern __constant__ FzbSVOUnformData systemSVOUniformData;
const uint32_t createSVOKernelBlockSize = 512;
const std::vector<uint32_t> SVONodesMaxCount = {	//node越到上层越难聚类，所以上几层的node数应该不变，下几层的node数较原来小
	1, 8, 64, 512, 1024, 1024, 1024, 1024
};
//----------------------------------------------常量-------------------------------------------------

struct FzbSVOCuda_PG {
public:
	FzbSVOSetting_PG setting;
	cudaExternalMemory_t VGBExtMem;
	FzbVoxelData_PG* VGB;
	FzbVGBUniformData VGBUniformData;
	FzbSVOUnformData SVOUniformData;

	uint32_t SVONodes_maxDepth = 0;

	std::vector<FzbSVONodeData_PG*> OctreeNodes_multiLayer;	//每级的node，满八叉树
	//------------------------------多层SVO------------------------------
	uint32_t SVONodeMaxCount = 0;
	uint32_t SVOIndivisibleNodeMaxCount = 0;		//假设聚类后最多有indivisibleNodeMaxCount个不可分node，如果超出则重新分配

	std::vector<FzbSVONodeThreadBlockInfo*> SVONodeThreadBlockInfos;	//用于多个线程组之间同步
	std::vector<FzbSVONodeTempInfo*> SVODivisibleNodeTempInfos;	//每层中可分node的信息
	
	FzbSVOLayerInfo* SVOLayerInfos = nullptr;
	std::vector<FzbSVOLayerInfo> SVOLayerInfos_host;

	FzbSVOIndivisibleNodeInfo* SVOIndivisibleNodeInfos = nullptr;

	std::vector<FzbSVONodeData_PG*> SVONodes_multiLayer;

	uint32_t SVONodeTotalCount_host = 0;	//SVONodes中每层node数量之和
	uint32_t SVOHasDataNodeTotalCount_host = 0;	//SVONodes中每层有值node数量之和
	uint32_t SVOInDivisibleNodeTotalCount_host;	//SVONodes中每层不可分node数量之和
	//-----------------------------计算weight---------------------------
	FzbSVONodeData_PG** SVONodes_multiLayer_Array = nullptr;	//每层有值node的数组指针
	uint32_t* SVODivisibleNodeBlockWeight;
	float* SVONodeWeights = nullptr;	//每个元素代表一个node和node之间的weigh(weight以每层为基础）

	cudaExternalSemaphore_t extSvoSemaphore_PG;
	std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager;

	cudaStream_t stream = nullptr;

	FzbSVOCuda_PG();
	FzbSVOCuda_PG(std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager, FzbSVOSetting_PG setting, FzbVGBUniformData VGBUniformData, FzbBuffer VGB, HANDLE SVOFinishedSemaphore_PG, FzbSVOUnformData SVOUniformData);

	void initVGB();
	void createSVOCuda_PG(HANDLE VGBFinishedSemaphore);
	void clean();

	void copyDataToBuffer(std::vector<FzbBuffer>& SVONodesBuffers, FzbBuffer SVOWeightsBuffer);

private:
	void initLightInjectSource();
	void lightInject();
	void initCreateOctreeNodesSource();
	void createOctreeNodes();
	void initCreateSVONodesSource();
	void createSVONodes();
	void initGetSVONodesWeightSource();
	void getSVONodesWeight();
};
#endif