#pragma once
#include "../../../SceneDivision/SVO_PG/FzbSVO_PG.h"
#include "./CUDA/FzbSVOPathGuidingCuda.cuh"

#ifndef FZB_SVO_PATH_GUIDING
#define FZB_SVO_PATH_GUIDING

struct FzbSVOPathGuidingSettingUniformObject {
	uint32_t screenWidth;
	uint32_t screenHeight;
};

struct FzbSVOPathGuiding_soft : public FzbFeatureComponent_LoopRender {
public:
	FzbSVOPathGuidingSetting_soft setting;
	FzbSVOPathGuidingCudaSetting cudaSetting;
	std::shared_ptr<FzbSVO_PG> SVO_PG;
	FzbRayTracingSourceManager* rayTracingSourceManager;
	FzbRasterizationSourceManager presentSourceManager;

	std::unique_ptr<FzbSVOPathGuidingCuda> svoPathGuidingCUDA;

	FzbBuffer settingBuffer;
	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	VkDescriptorSet descriptorSet;

	FzbSVOPathGuiding_soft();
	FzbSVOPathGuiding_soft(pugi::xml_node& SVOPathGuidingNode_soft);
	void addMainSceneInfo() override;
	void init() override;
	FzbSemaphore render(uint32_t imageIndex, FzbSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) override;
	void clean() override;

private:
	uint32_t SVO_PG_MaxDepth = 0;	//´Ó0¿ªÊ¼
	void addExtensions();
	void createImages() override;
	void presentPrepare() override;
	void createBufferAndDescirptor();
	void createRenderPass();

	void updateCudaSetting();
};

#endif