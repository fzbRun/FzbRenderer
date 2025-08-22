#pragma once

#include "../common/FzbComponent.h"

#ifndef FZB_FORWARD_RENDER
#define FZB_FORWARD_RENDER

struct FzbForwardRenderSetting {
	bool useForwardRender = true;
};

struct FzbForwardRender : public FzbComponent {

public:
	FzbForwardRenderSetting setting;
	FzbScene* scene;

	FzbRenderPass renderPass;
	FzbImage depthMap;
	
	VkDescriptorSetLayout descriptorSetLayout = nullptr;	//将scene的相机和光照描述符作为组件描述符
	VkDescriptorSet descriptorSet;

	VkCommandBuffer commandBuffer;
	FzbSemaphore renderFinishedSemaphores;

	void addExtensions(FzbForwardRenderSetting frSetting, std::vector<const char*>& instanceExtensions, std::vector<const char*>& deviceExtensions, VkPhysicalDeviceFeatures& deviceFeatures) {

	}

	FzbForwardRender() {};

	void clean() {
		renderPass.clean();
		depthMap.clean();
		renderFinishedSemaphores.clean(logicalDevice);
		//descriptorSetLayout是相机光源信息，由scene维护并清除
	}

	FzbVertexFormat getComponentVertexFormat() {
		return FzbVertexFormat(true);
	}

	void init(FzbMainComponent* renderer, FzbForwardRenderSetting setting, FzbScene* scene, std::vector<FzbRenderPass*>& renderPasses) {
		initComponent(renderer);
		this->setting = setting;
		this->scene = scene;
		this->descriptorSetLayout = scene->cameraAndLightsDescriptorSetLayout;
		this->descriptorSet = scene->cameraAndLightsDescriptorSet;
		fzbCreateCommandBuffers(1);
		renderFinishedSemaphores = FzbSemaphore(logicalDevice, false);
		createImages();
		createRenderPass(renderPasses);
	}

	void createImages() {
		depthMap = {};
		depthMap.width = swapChainExtent.width;
		depthMap.height = swapChainExtent.height;
		depthMap.type = VK_IMAGE_TYPE_2D;
		depthMap.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthMap.format = fzbFindDepthFormat(physicalDevice);
		depthMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		depthMap.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
		depthMap.fzbCreateImage(physicalDevice, logicalDevice, commandPool, graphicsQueue);
	}
	void createRenderPass(std::vector<FzbRenderPass*>& renderPasses) {
		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment(physicalDevice);
		VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

		VkSubpassDescription presentSubpass = fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef);
		std::vector<VkSubpassDescription> subpasses = { presentSubpass };

		std::vector<VkSubpassDependency> dependencies = { fzbCreateSubpassDependency() };

		FzbRenderPassSetting setting = { true, 1, swapChainExtent, swapChainImageViews.size(), true };
		renderPass = FzbRenderPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, setting);
		renderPass.images.push_back(&depthMap);
		renderPass.createRenderPass(&attachments, subpasses, dependencies);
		renderPass.createFramebuffers(swapChainImageViews);

		FzbSubPass presentSubPass = FzbSubPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, renderPass.renderPass, { descriptorSetLayout }, 0, scene, {}, swapChainExtent);
		renderPass.subPasses.push_back(presentSubPass);

		renderPasses.push_back(&renderPass);
	}

	VkSemaphore render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = nullptr) {
		scene->updateCameraBuffer();

		VkCommandBuffer commandBuffer = commandBuffers[0];
		vkResetCommandBuffer(commandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		renderPass.render(commandBuffer, imageIndex, { scene }, { {descriptorSet} });

		VkSemaphore waitSemaphores[] = { startSemaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &renderFinishedSemaphores.semaphore;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		return renderFinishedSemaphores.semaphore;
	}
};

#endif