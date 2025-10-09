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
	std::vector<FzbBuffer> SVOBuffers;
	FzbRayTracingSourceManager rayTracingSourceManager;

	FzbSVO_PG();
	FzbSVO_PG(pugi::xml_node& SVO_PGNode);
	void addMainSceneInfo() override;
	void init() override;
	void clean() override;

	void createSVOBuffers();
private:
	void addExtensions();

	void createUniformBuffer();
	void createStorageBuffer();
	void createDescriptor();
	void createBufferAndDescriptor();
	void createSemaphore();

	void createVGBRenderPass();
	void createVGB();
	void createSVO_PG();
};

struct FzbSVOSetting_PG_Debug {
	FzbSVOSetting_PG SVO_PGSetting;
	bool voxelAABBDebugInfo = true;
	bool voxelIrradianceDebugInfo = false;
	bool SVONodeClusterDebugInfo = false;
	uint32_t SVONodeClusterLevel = 0;
};
struct FzbSVONodeClusterUniformObject {
	glm::vec4 nodeSize_Num;
	glm::vec4 startPos;
};
struct FzbSVONodeBlockData {
	uint32_t startIndex;	//当前线程组
	uint32_t blockNum;
	uint32_t nodeNum;
};

struct FzbSVO_PG_Debug : public FzbFeatureComponent_LoopRender {
public:
	FzbSVOSetting_PG_Debug setting;
	std::shared_ptr<FzbSVO_PG> SVO_PG;
	FzbRasterizationSourceManager presentSourceManager;
	FzbImage depthMap;

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;

	FzbBuffer SVONodeClusterUniformBuffer;

	FzbSVO_PG_Debug();
	FzbSVO_PG_Debug(pugi::xml_node& SVO_PG_DebugNode);
	void addMainSceneInfo() override;
	void init() override;
	FzbSemaphore render(uint32_t imageIndex, FzbSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) override;
	void clean() override;
private:
	void addExtensions();
	void createImages() override;
	void presentPrepare() override;
	void createBufferAndDescirptor();
	void createVGBRenderPass_AABBInfo();
	void createVGBRenderPass_IrradianceInfo();
	void createVGBRenderPass_SVONodeClusterInfo();
};

#endif