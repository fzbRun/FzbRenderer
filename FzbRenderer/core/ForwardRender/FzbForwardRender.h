#pragma once

#include "../common/FzbComponent/FzbFeatureComponent.h"
#include "../common/FzbRenderer.h"
#include "../common/FzbRenderPass/FzbRenderPass.h"

#ifndef FZB_FORWARD_RENDER
#define FZB_FORWARD_RENDER

struct FzbForwardRenderSetting {
	
};

struct FzbForwardRender : public FzbFeatureComponent_LoopRender {

public:
	FzbForwardRender();
	FzbForwardRender(pugi::xml_node& ForwardRenderNode);

	void init() override;

	VkSemaphore render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE);

	void clean() override;


private:
	FzbForwardRenderSetting setting;
	FzbImage depthMap;

	void addExtensions() override;

	void presentPrepare() override;
	
	void createImages() override;
};

#endif