#pragma once

#include "../common/FzbComponent/FzbFeatureComponent.h"
#include "../common/FzbRenderer.h"
#include "../common/FzbRenderPass/FzbRenderPass.h"
#include "../common/FzbRasterizationRender/FzbRasterizationSourceManager.h"

#ifndef FZB_FORWARD_RENDER
#define FZB_FORWARD_RENDER

struct FzbForwardRenderSetting {
	
};

struct FzbForwardRender : public FzbFeatureComponent_LoopRender {

public:
	FzbForwardRender();
	FzbForwardRender(pugi::xml_node& ForwardRenderNode);

	void init() override;

	FzbSemaphore render(uint32_t imageIndex, FzbSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) override;

	void clean() override;


private:
	FzbForwardRenderSetting setting;
	FzbRasterizationSourceManager sourceManager;
	FzbImage depthMap;

	void addMainSceneInfo() override;
	void addExtensions() override;

	void presentPrepare() override;
	
	void createImages() override;
};

#endif