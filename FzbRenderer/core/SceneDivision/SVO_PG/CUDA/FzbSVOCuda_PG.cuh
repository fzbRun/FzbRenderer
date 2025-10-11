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
	uint32_t shuffleKey;
	uint32_t label;
	FzbAABB AABB;
	glm::vec3 irradiance;
};
__host__ __device__ struct FzbVoxelData_PG_OBB {
	uint hasData;
	glm::vec3 irradiance;
	FzbOBBUint OBB;
};
struct FzbSVONodeData_PG_OBB {
	uint32_t shuffleKey;
	FzbOBB OBB;
	glm::vec3 irradiance;
};

struct FzbSVONodeBlock {
	uint32_t nodeCount;	//有值的node数量
	uint32_t blockCount;
};
struct FzbSVONodeTempInfo {
	FzbSVONodeData_PG nodeData;
	uint32_t nodeIndexInThreadBlock;
};

struct FzbVGBUniformData {
	uint32_t voxelCount;
	glm::vec3 voxelSize;
	glm::vec3 voxelStartPos;
};
struct FzbSVOUnformData {
	float surfaceAreaThreshold = 10.5f;	//如何合并后的AABB的表面超过原有子node的表面积之和的surfaceAreaThreshold倍，则认为不能合并
	float irradianceThreshold = 10.5f;	//如果某个两个兄弟node的irradiance之比超过irradianceThreshold，则认为不能合并
	float ignoreIrradianceValueThreshold = 1.0f;
};

struct FzbSVOCuda_PG {
public:
	FzbSVOSetting_PG setting;
	cudaExternalMemory_t VGBExtMem;
	FzbVoxelData_PG* VGB;
	FzbVGBUniformData VGBUniformData;
	FzbSVOUnformData SVOUniformData;
	std::vector<FzbSVONodeBlock*> SVONodeBlockInfos;
	std::vector<FzbSVONodeTempInfo*> SVONodeTempInfos;
	std::vector<uint32_t*> SVONodeCount;
	std::vector<FzbSVONodeData_PG*> SVOs_PG;	//每级的node
	cudaExternalSemaphore_t extSvoSemaphore_PG;
	std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager;

	FzbSVOCuda_PG();
	FzbSVOCuda_PG(std::shared_ptr<FzbRayTracingSourceManager_Cuda> sourceManager, FzbSVOSetting_PG setting, FzbVGBUniformData VGBUniformData, FzbBuffer VGB, HANDLE SVOFinishedSemaphore_PG, FzbSVOUnformData SVOUniformData);

	void initVGB();
	void lightInject();
	void createSVOCuda_PG();
	void clean();

	void copyDataToBuffer(std::vector<FzbBuffer>& buffers);
};
#endif