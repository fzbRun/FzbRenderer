#pragma once

#include "../../../common/FzbCommon.h"
#include "../../../common/FzbComponent/FzbFeatureComponent.h"
#include "../../../common/FzbRenderer.h"
#include "CUDA/PathTracing_CUDA.cuh"

#ifndef FZB_PATH_TRACING_H
#define FZB_PATH_TRACING_H

struct FzbPathTracingSetting {
	bool spp = 1;
};

struct FzbPathTracing_soft : public FzbFeatureComponent_LoopRender {
public:
	FzbPathTracing_soft() {};
	FzbPathTracing_soft(pugi::xml_node& PathTracingNode) {
		if (std::string(PathTracingNode.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
		else return;

		this->componentInfo.name = FZB_RENDERER_PATH_TRACING_SOFT;
		this->componentInfo.type = FZB_RENDER_COMPONENT;
		this->componentInfo.vertexFormat = FzbVertexFormat(true);
		this->componentInfo.useMainSceneBufferHandle = { true, false, false };	//需要全部格式的顶点buffer和索引buffer，用来创建svo

		addExtensions();
	}

	void init() override {
		FzbFeatureComponent_LoopRender::init();
		presentPrepare();
	}

	VkSemaphore render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) {
		VkCommandBuffer commandBuffer = commandBuffers[0];
		vkResetCommandBuffer(commandBuffer, 0);
		fzbBeginCommandBuffer(commandBuffer);

		pathTracingCUDA->pathTracing(this->mainScene);
		VkExtent2D resolution = FzbRenderer::globalData.getResolution();
		VkExtent3D copyExtent = { resolution.width, resolution.height , 0.0f };
		fzbCopyImageToImage(commandBuffer, pathTracingResultMap.image, FzbRenderer::globalData.swapChainImages[imageIndex], copyExtent);
		
		std::vector<VkSemaphore> waitSemaphores = { pathTracingFinishedSemphore.semaphore };
		std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
		fzbSubmitCommandBuffer(commandBuffer, { startSemaphore }, waitStages, { renderFinishedSemaphore.semaphore }, fence);

		return renderFinishedSemaphore.semaphore;
	};

	void clean() override {
		FzbFeatureComponent_LoopRender::clean();
		pathTracingResultMap.clean();
	};

private:
	FzbPathTracingSetting setting;
	FzbImage pathTracingResultMap;
	FzbSemaphore pathTracingFinishedSemphore;
	std::unique_ptr<FzbPathTracingCuda> pathTracingCUDA;

	void addExtensions() override {};

	void presentPrepare() override {
		fzbCreateCommandBuffers(1);
		pathTracingCUDA = std::make_unique<FzbPathTracingCuda>();
	};

	void createImages() override {
		VkExtent2D resolution = FzbRenderer::globalData.getResolution();

		pathTracingResultMap = {};
		pathTracingResultMap.width = resolution.width;
		pathTracingResultMap.height = resolution.height;
		pathTracingResultMap.type = VK_IMAGE_TYPE_2D;
		pathTracingResultMap.viewType = VK_IMAGE_VIEW_TYPE_2D;
		pathTracingResultMap.format = FzbRenderer::globalData.swapChainImageFormat;		//SRGB
		pathTracingResultMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		pathTracingResultMap.UseExternal = true;
		pathTracingResultMap.initImage();
		pathTracingResultMap.transitionImageLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1);

		frameBufferImages.push_back(&pathTracingResultMap);
	}
};

#endif