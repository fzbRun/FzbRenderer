#pragma once

#include "../../../CUDA/vulkanCudaInterop.cuh"
#include "../../../CUDA/commonCudaFunction.cuh"
#include "../../../CUDA/commonStruct.cuh"
#include "../../../RayTracing/CUDA/FzbRayTracingInitSource.cuh"

#ifndef CREATE_SVO_PATH_GUIDING_CUH
#define CREATE_SVO_PATH_GUIDING_CUH
struct FzbVGBUniformData {
	uint32_t voxelCount;
	glm::vec3 voxelSize;
	glm::vec3 voxelStartPos;
};
struct FzbSVOUnformData {
	float irradianceRelRatioThreshold = 0.1f;
	float entropyThreshold = 0.707f;
};

struct FzbSVOSetting_PG {
	uint32_t voxelNum = 64;
	bool useCube = true;
	uint32_t lightInjectSPP = 16;
	FzbSVOUnformData thresholds;
};
struct FzbVoxelData_PG {
	glm::vec4 irradiance;
	glm::vec4 meanNormal_G;
	glm::vec4 meanNormal_E;
	FzbAABBUint AABB;
};

//struct FzbSVONodeData_PG {
//	uint32_t indivisible;
//	uint32_t label;		//该node是当前层中第几个可分的node
//	FzbAABB AABB_G;		//node中几何的AABB
//	FzbAABB AABB_E;		//node中光照的AABB
//	glm::vec3 irradiance;
//	glm::vec3 meanNormal_G;	//没有归一化
//	glm::vec3 meanNormal_E;	//没有归一化
//};
struct FzbSVONodeData_PG_G {
	uint32_t indivisible;
	uint32_t label;		//该node是当前层中第几个可分的node
	FzbAABB AABB;
	float entropy;
	glm::vec3 meanNormal;
	//uint32_t hasDataChildrenMask;
};
struct FzbSVONodeData_PG_E {
	uint32_t indivisible;
	uint32_t label;		//该node是当前层中第几个可分的node
	FzbAABB AABB;
	float notIgnoreRatio;
	glm::vec3 irradiance;
	glm::vec3 meanNormal;
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

//----------------------------------------------常量
extern __constant__ FzbVGBUniformData systemVGBUniformData;
extern __constant__ FzbSVOUnformData systemSVOUniformData;
const uint32_t createSVOKernelBlockSize = 512;
const std::vector<uint32_t> SVONodesMaxCount = {	//node越到上层越难聚类，所以上几层的node数应该不变，下几层的node数较原来小
	1, 8, 64, 512, 2048, 2048, 4096, 4096
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

	std::vector<FzbSVONodeData_PG_G*> OctreeNodes_multiLayer_G;
	std::vector<FzbSVONodeData_PG_E*> OctreeNodes_multiLayer_E;
	//------------------------------多层SVO------------------------------
	std::vector<FzbSVONodeThreadBlockInfo*> SVONodeThreadBlockInfos;	//用于多个线程组之间同步
	std::vector<FzbSVONodeTempInfo*> SVODivisibleNodeTempInfos;	//每层中可分node的信息
	
	FzbSVOLayerInfo* SVOLayerInfos_G = nullptr;
	std::vector<FzbSVOLayerInfo> SVOLayerInfos_G_host;

	FzbSVOLayerInfo* SVOLayerInfos_E = nullptr;
	std::vector<FzbSVOLayerInfo> SVOLayerInfos_E_host;

	FzbSVOIndivisibleNodeInfo* SVOIndivisibleNodeInfos_G = nullptr;

	std::vector<FzbSVONodeData_PG_G*> SVONodes_multiLayer_G;
	std::vector<FzbSVONodeData_PG_E*> SVONodes_multiLayer_E;

	uint32_t SVONode_E_TotalCount_host = 0;	//SVONodes中每层node数量之和
	uint32_t SVOInDivisibleNode_G_TotalCount_host = 0;	//SVONodes中每层不可分node数量之和

	//-----------------------------计算weight---------------------------
	FzbSVONodeData_PG_G** SVONodes_G_multiLayer_Array = nullptr;	//每层有值node的数组指针
	FzbSVONodeData_PG_E** SVONodes_E_multiLayer_Array = nullptr;	//每层有值node的数组指针
	float* SVODivisibleNodeBlockWeight = nullptr;	//每个block将weightSum存入这个临时数组，用于上一层父节点获取（因为block只知道父节点的label，而不知道具体nodeIndex）
	float* SVOFatherDivisibleNodeBlockWeight = nullptr;
	float* SVONodeWeights = nullptr;	//每个元素代表一个node和node之间的weigh(weight以每层为基础）

	cudaExternalSemaphore_t extSvoSemaphore_PG;
	std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager;

	cudaStream_t stream = nullptr;

	FzbSVOCuda_PG();
	FzbSVOCuda_PG(std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager, FzbSVOSetting_PG setting, FzbVGBUniformData VGBUniformData, FzbBuffer VGB, HANDLE SVOFinishedSemaphore_PG, FzbSVOUnformData SVOUniformData);

	void initVGB();
	void createSVOCuda_PG(HANDLE VGBFinishedSemaphore);
	void clean();

	void coypyOctreeDataToBuffer(std::vector<FzbBuffer>& OctreeNodesBuffers, bool isG);
	void copySVODataToBuffer(std::vector<FzbBuffer>& SVONodesBuffers, bool isG);
	void copySVONodeWeightsToBuffer(FzbBuffer& SVONodeWeightsBuffer);

private:
	void initLightInjectSource();
	void lightInject();
	void initCreateOctreeNodesSource(bool allocate = true);
	void createOctreeNodes();
	void initCreateSVONodesSource(bool allocate = true);
	void createSVONodes();
	void initGetSVONodesWeightSource(bool allocate = true);
	void getSVONodesWeight();
};
#endif