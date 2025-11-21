#pragma once

#include "../../common/FzbCommon.h"
#include "../../common/FzbComponent/FzbFeatureComponent.h"
#include "./CUDA/FzbSVOCuda_PG.cuh"
#include "../../common/FzbRasterizationRender/FzbRasterizationSourceManager.h"
#include "../../RayTracing/common/FzbRayTracingSourceManager.h"

#ifndef FZB_SVO_PATH_GUIDING_H
#define FZB_SVO_PATH_GUIDING_H

struct FzbSVOUniformBufferObject {
	glm::mat4 VP[3];
	glm::vec4 voxelSize_Num;
	glm::vec4 voxelStartPos;
};
struct FzbSVOUniformBufferObject2 {
	uint32_t SVONodeCount[6];	//最多128x128x128，那么去掉根节点和叶节点就是6层

};
struct FzbSVO_PG : public FzbFeatureComponent_PreProcess {
public:
	FzbSVOSetting_PG setting;

	FzbSVOUniformBufferObject uniformBufferObject;
	FzbBuffer uniformBuffer;
	FzbBuffer VGB;

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;

	FzbRasterizationSourceManager vgbSourceManager;
	FzbRenderPass vgbRenderPass;

	FzbSemaphore VGBFinishedSemaphore;
	FzbSemaphore SVOFinishedSemaphore;

	std::shared_ptr<FzbSVOCuda_PG> svoCuda_pg;
	std::vector<FzbBuffer> OctreeNodesBuffers;
	std::vector<FzbBuffer> SVONodesBuffers;
	FzbBuffer SVOWeightsBuffer;
	FzbRayTracingSourceManager rayTracingSourceManager;

	FzbSVO_PG();
	FzbSVO_PG(pugi::xml_node& SVO_PGNode);
	void addMainSceneInfo() override;
	void init() override;
	void clean() override;

	void createOctreeBuffers(bool isG);
	void createSVOBuffers(bool isG, bool useDeviceAddress = true);
	void createSVOWeightsBuffer();

	void createSVO_PG();
private:
	void addExtensions();

	void createUniformBuffer();
	void createStorageBuffer();
	void createDescriptor();
	void createBufferAndDescriptor();
	void createSemaphore();

	void createVGBRenderPass();
	void createVGB();
};

struct FzbSVOSetting_PG_Debug {
	FzbSVOSetting_PG SVO_PGSetting;
	bool voxelAABBDebugInfo = true;
	bool voxelIrradianceDebugInfo = false;
	bool OctreeNodeDebugInfo = false;
	bool SVONodeClusterDebugInfo = false;
	bool lookCube = 0;
	bool isG = true;
	bool SVOWeightsDebugInfo = false;
	bool useDeviceAddress = false;	//用设备地址在renderDoc中看不到数据
};
struct FzbSVONodeClusterUniformObject {
	glm::vec4 nodeSize_Num;
	glm::vec4 startPos;
	uint32_t maxDepth;
	int nodeClusterInfoLevel = -1;	//-1表示全看
	uint32_t nodeCounts[6] = { 0 };	//每层有多少个有值node
	uint64_t SVONodesAddress[6] = { 0 };	//最大128^3，去掉根节点和叶节点，最多6层
};
struct FzbSVONodeBlockData {
	uint32_t startIndex;	//当前线程组
	uint32_t blockNum;
	uint32_t nodeNum;
};

struct FzbSVOWeightsUniformObject {
	glm::vec4 nodeSize_Num;
	glm::vec4 startPos;
	uint32_t maxDepth;
	uint32_t svoNodeTotalCount = 0;
	int divisibleNodeCounts_G[7];
	int indivisibleNodeCounts_G[7];
	int divisibleNodeCounts_E[7];
	int indivisibleNodeCounts_E[7];
};

struct FzbSVO_PG_Debug : public FzbFeatureComponent_LoopRender {
public:
	FzbSVOSetting_PG_Debug setting;
	std::shared_ptr<FzbSVO_PG> SVO_PG;
	FzbRasterizationSourceManager presentSourceManager;
	FzbImage depthMap;

	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	VkDescriptorSet descriptorSet = nullptr;

	FzbBuffer debugUniformBuffer;

	FzbSVO_PG_Debug();
	FzbSVO_PG_Debug(pugi::xml_node& SVO_PG_DebugNode);
	void addMainSceneInfo() override;
	void init() override;
	FzbSemaphore render(uint32_t imageIndex, FzbSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) override;
	void clean() override;
private:
	uint32_t SVO_PG_MaxDepth = 0;	//从0开始
	void addExtensions();
	void createImages() override;
	void presentPrepare() override;
	
	void createVGBRenderPass_AABBInfo();
	void createVGBRenderPass_IrradianceInfo();

	void createOctreeDebugBufferAndDescirptor();
	void createOctreeRenderPass();

	void createSVODebugBufferAndDescirptor();
	void createVGBRenderPass_SVONodeClusterInfo();

	void createSVOWeightsDebugBufferAndDescirptor();
	void createSVORenderPass_SVOWeightsInfo();
};

#endif