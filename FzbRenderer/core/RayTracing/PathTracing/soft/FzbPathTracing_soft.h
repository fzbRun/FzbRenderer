#pragma once

#include "../../../common/FzbCommon.h"
#include "../../../common/FzbComponent/FzbFeatureComponent.h"
#include "../../../SceneDivision/BVH/FzbBVH.h"
#include "CUDA/PathTracing_CUDA.cuh"
#include "../../common/FzbRayTracingSourceManager.h"

#ifndef FZB_PATH_TRACING_H
#define FZB_PATH_TRACING_H

struct FzbPathTracingSettingUniformObject {
	uint32_t screenWidth;
	uint32_t screenHeight;
};

struct FzbPathTracing_soft : public FzbFeatureComponent_LoopRender {
public:
	FzbPathTracing_soft();
	FzbPathTracing_soft(pugi::xml_node& PathTracingNode);

	void init() override;
	FzbSemaphore render(uint32_t imageIndex, FzbSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) override;
	void clean() override;

private:
	FzbPathTracingSetting setting;
	FzbBuffer settingBuffer;
	FzbRayTracingSourceManager rayTracingSourceManager;
	FzbRasterizationSourceManager presentSourceManager;
	std::unique_ptr<FzbPathTracingCuda> pathTracingCUDA;

	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	VkDescriptorSet descriptorSet;

	void addMainSceneInfo() override;
	void addExtensions() override;

	void presentPrepare() override;
	void createBuffer();
	void createImages() override;

	void createDescriptor();
	//将cuda得到的pathTracing结果的buffer复制到帧缓冲中
	void createRenderPass();
};

#endif