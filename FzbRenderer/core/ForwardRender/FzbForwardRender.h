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
	FzbForwardRender() {};
	FzbForwardRender(pugi::xml_node& ForwardRenderNode) {
		if (std::string(ForwardRenderNode.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
		else return;

		this->componentInfo.name = FZB_RENDERER_FORWARD;
		this->componentInfo.type = FZB_RENDER_COMPONENT;
		this->componentInfo.vertexFormat = FzbVertexFormat(true);
		this->componentInfo.useMainSceneBufferHandle = { true, false, false };	//需要全部格式的顶点buffer和索引buffer，用来创建svo

		addExtensions();
	};

	void init() override {
		FzbFeatureComponent_LoopRender::init();
		presentPrepare();
	}

	VkSemaphore render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) {
		VkCommandBuffer commandBuffer = commandBuffers[0];
		vkResetCommandBuffer(commandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		renderRenderPass.render(commandBuffer, imageIndex);

		std::vector<VkSemaphore> waitSemaphores = { startSemaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;	// waitSemaphores.size();
		submitInfo.pWaitSemaphores = waitSemaphores.data();
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &renderFinishedSemaphore.semaphore;

		if (vkQueueSubmit(FzbRenderer::globalData.graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		return renderFinishedSemaphore.semaphore;
	}

	void clean() override {
		FzbFeatureComponent_LoopRender::clean();
		depthMap.clean();
	}


private:
	FzbForwardRenderSetting setting;
	FzbImage depthMap;

	void addExtensions() override {}

	void presentPrepare() override {
		fzbCreateCommandBuffers(1);
		
		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(FzbRenderer::globalData.swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment();
		VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

		VkSubpassDescription presentSubpass = fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef);
		std::vector<VkSubpassDescription> subpasses = { presentSubpass };

		std::vector<VkSubpassDependency> dependencies = { fzbCreateSubpassDependency() };

		FzbRenderPassSetting renderPassSetting = { true, 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
		renderRenderPass.setting = renderPassSetting;
		renderRenderPass.createRenderPass(&attachments, subpasses, dependencies);
		renderRenderPass.createFramebuffers(true);

		FzbSubPass presentSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
			{ mainScene->cameraAndLightsDescriptorSetLayout }, { mainScene->cameraAndLightsDescriptorSet },
			mainScene->vertexBuffer.buffer, mainScene->indexBuffer.buffer, mainScene->sceneShaders_vector);
		renderRenderPass.subPasses.push_back(presentSubPass);
	}
	
	void createImages() override  {
		depthMap = {};
		depthMap.width = FzbRenderer::globalData.swapChainExtent.width;
		depthMap.height = FzbRenderer::globalData.swapChainExtent.height;
		depthMap.type = VK_IMAGE_TYPE_2D;
		depthMap.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthMap.format = fzbFindDepthFormat();
		depthMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		depthMap.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
		depthMap.initImage();

		frameBufferImages.push_back(&depthMap);
	}
};

#endif