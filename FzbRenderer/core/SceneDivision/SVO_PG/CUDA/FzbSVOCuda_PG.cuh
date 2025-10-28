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
	float pdf;	//ȥ��Ӱ��С����node����Ҫ�ֲ���pdf
	//uint32_t shuffleKey;
	uint32_t label;		//��node�ǵ�ǰ���еڼ����ɷֵ�node
	FzbAABB AABB;
	glm::vec3 irradiance;
};

struct FzbSVONodeThreadBlockInfo {
	uint32_t divisibleNodeCount;	//��ֵ�ҿɷֵ�node����
	uint32_t indivisibleNodeCount;	//��ֵ�ҿɷֵ�node����
};
struct FzbSVONodeTempInfo {
	uint32_t nodeIndex;		//��octree�е�����
	uint32_t storageIndex;	//��svo�е�����
	uint32_t label;
	uint32_t threadBlockIndex;
};
struct FzbSVOLayerInfo {
	uint32_t divisibleNodeCount;
	uint32_t indivisibleNodeCount;
};
struct FzbSVOIndivisibleNodeInfo {
	uint32_t nodeLayer;
	uint32_t nodeIndex;	//��svo��һ�������
};

struct FzbVGBUniformData {
	uint32_t voxelCount;
	glm::vec3 voxelSize;
	glm::vec3 voxelStartPos;
};
struct FzbSVOUnformData {
	float surfaceAreaThreshold = 3.0f;	//��κϲ����AABB�ı��泬��ԭ����node�ı����֮�͵�surfaceAreaThreshold��������Ϊ���ܺϲ�
	float irradianceThreshold = 10.0f;	//���ĳ�������ֵ�node��irradiance֮�ȳ���irradianceThreshold������Ϊ���ܺϲ�
	float ignoreIrradianceValueThreshold = 10.0f;
};

//----------------------------------------------����
extern __constant__ FzbVGBUniformData systemVGBUniformData;
extern __constant__ FzbSVOUnformData systemSVOUniformData;
const uint32_t createSVOKernelBlockSize = 512;
const std::vector<uint32_t> SVONodesMaxCount = {	//nodeԽ���ϲ�Խ�Ѿ��࣬�����ϼ����node��Ӧ�ò��䣬�¼����node����ԭ��С
	1, 8, 64, 512, 1024, 1024, 1024, 1024
};
//----------------------------------------------����-------------------------------------------------

struct FzbSVOCuda_PG {
public:
	FzbSVOSetting_PG setting;
	cudaExternalMemory_t VGBExtMem;
	FzbVoxelData_PG* VGB;
	FzbVGBUniformData VGBUniformData;
	FzbSVOUnformData SVOUniformData;

	uint32_t SVONodes_maxDepth = 0;

	std::vector<FzbSVONodeData_PG*> OctreeNodes_multiLayer;	//ÿ����node�����˲���
	//------------------------------���SVO------------------------------
	uint32_t SVONodeMaxCount = 0;
	uint32_t SVOIndivisibleNodeMaxCount = 0;		//�������������indivisibleNodeMaxCount�����ɷ�node��������������·���

	std::vector<FzbSVONodeThreadBlockInfo*> SVONodeThreadBlockInfos;	//���ڶ���߳���֮��ͬ��
	std::vector<FzbSVONodeTempInfo*> SVODivisibleNodeTempInfos;	//ÿ���пɷ�node����Ϣ
	
	FzbSVOLayerInfo* SVOLayerInfos = nullptr;
	std::vector<FzbSVOLayerInfo> SVOLayerInfos_host;

	FzbSVOIndivisibleNodeInfo* SVOIndivisibleNodeInfos = nullptr;

	std::vector<FzbSVONodeData_PG*> SVONodes_multiLayer;

	uint32_t SVONodeTotalCount_host = 0;	//SVONodes��ÿ��node����֮��
	uint32_t SVOHasDataNodeTotalCount_host = 0;	//SVONodes��ÿ����ֵnode����֮��
	uint32_t SVOInDivisibleNodeTotalCount_host;	//SVONodes��ÿ�㲻�ɷ�node����֮��
	//-----------------------------����weight---------------------------
	FzbSVONodeData_PG** SVONodes_multiLayer_Array = nullptr;	//ÿ����ֵnode������ָ��
	uint32_t* SVODivisibleNodeBlockWeight;
	float* SVONodeWeights = nullptr;	//ÿ��Ԫ�ش���һ��node��node֮���weigh(weight��ÿ��Ϊ������

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