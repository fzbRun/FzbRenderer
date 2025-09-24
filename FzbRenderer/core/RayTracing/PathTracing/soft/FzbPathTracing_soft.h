#pragma once

#include "../../../common/FzbCommon.h"
#include "../../../common/FzbComponent/FzbFeatureComponent.h"
#include "../../../SceneDivision/BVH/FzbBVH.h"
#include "CUDA/PathTracing_CUDA.cuh"

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
	FzbRasterizationSourceManager presentSourceManager;
	//FzbImage pathTracingResultMap;
	FzbBuffer pathTracingResultBuffer;
	FzbSemaphore pathTracingFinishedSemphore;
	std::unique_ptr<FzbPathTracingCuda> pathTracingCUDA;

	//我们不让mainScene去创建materialSource，而是我们自己创造，自己维护
	std::vector<FzbImage> sceneTextures;
	std::vector<FzbPathTracingMaterialUniformObject> sceneMaterialInfoArray;

	std::shared_ptr<FzbBVH> bvh;

	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	VkDescriptorSet descriptorSet;

	void addMainSceneInfo() override;
	void addExtensions() override;

	void presentPrepare() override;

	//与raserizationSourceManager的功能相同，这里需要创建各种后续渲染所用资源
	FzbPathTracingCudaSourceSet createSource();
	void createImages() override;

	void createDescriptor();
	//将cuda得到的pathTracing结果的buffer复制到帧缓冲中
	void createRenderPass();
};

#endif