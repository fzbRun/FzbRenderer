#include "FzbRenderPass.h"

#include <stdexcept>

FzbSubPass::FzbSubPass() {};
FzbSubPass::FzbSubPass(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, VkRenderPass renderPass, std::vector<VkDescriptorSetLayout> componentDescriptorSetLayouts, uint32_t subPassIndex, FzbScene* scene, std::vector<FzbShader*> shaders, VkExtent2D extent) {
	this->physicalDevice = physicalDevice;
	this->logicalDevice = logicalDevice;
	this->commandPool = commandPool;
	this->graphicsQueue = graphicsQueue;
	this->renderPass = renderPass;
	this->subPassIndex = subPassIndex;
	this->extent = extent;

	this->shaders = shaders;
	if (shaders.size() == 0) {
		for (auto& shader : scene->sceneShaders) {
			this->shaders.push_back(&shader.second);
		}
	}
	for (int i = 0; i < this->shaders.size(); i++) {
		FzbShader* shader = this->shaders[i];
		//shader->createMeshBatch(scene->sceneMeshSet, scene->sceneIndices, physicalDevice, commandPool, graphicsQueue);
		shader->createPipeline(renderPass, subPassIndex, scene->meshDescriptorSetLayout, componentDescriptorSetLayouts);
	}
}
FzbSubPass::FzbSubPass(FzbSubPassCreateInfo* createInfo) {
	this->physicalDevice = createInfo->physicalDevice;
	this->logicalDevice = createInfo->logicalDevice;
	this->commandPool = createInfo->commandPool;
	this->graphicsQueue = createInfo->graphicsQueue;
	this->renderPass = createInfo->renderPass;
	this->subPassIndex = createInfo->subPassIndex;
	this->extent = createInfo->extent;

	this->shaders = shaders;
	if (shaders.size() == 0) {
		for (auto& shader : createInfo->scene->sceneShaders) {
			this->shaders.push_back(&shader.second);
		}
	}
	for (int i = 0; i < this->shaders.size(); i++) {
		FzbShader* shader = this->shaders[i];
		//shader->createMeshBatch(scene->sceneMeshSet, scene->sceneIndices, physicalDevice, commandPool, graphicsQueue);
		shader->createPipeline(renderPass, subPassIndex, createInfo->scene->meshDescriptorSetLayout, createInfo->componentDescriptorSetLayouts);
	}
}
void FzbSubPass::clean() {}
void FzbSubPass::render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets) {
	for (int i = 0; i < this->shaders.size(); i++) {
		this->shaders[i]->render(commandBuffer, componentDescriptorSets, extent);
	}
}

FzbRenderPass::FzbRenderPass() {};
FzbRenderPass::FzbRenderPass(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, FzbRenderPassSetting setting) {
	this->physicalDevice = physicalDevice;
	this->logicalDevice = logicalDevice;
	this->commandPool = commandPool;
	this->graphicsQueue = graphicsQueue;
	this->setting = setting;
}
void FzbRenderPass::createRenderPass(std::vector<VkAttachmentDescription>* attachments, std::vector<VkSubpassDescription> subpasses, std::vector<VkSubpassDependency> dependencies) {
	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = attachments ? attachments->size() : 0;
	renderPassInfo.pAttachments = attachments ? attachments->data() : nullptr;
	renderPassInfo.subpassCount = subpasses.size();
	renderPassInfo.pSubpasses = subpasses.data();
	renderPassInfo.dependencyCount = dependencies.size();
	renderPassInfo.pDependencies = dependencies.data();

	if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
		throw std::runtime_error("failed to create render pass!");
	}
}
void FzbRenderPass::clean() {
	if (renderPass) vkDestroyRenderPass(logicalDevice, renderPass, nullptr);
	for (int i = 0; i < subPasses.size(); i++) {
		subPasses[i].clean();
	}
	for (int i = 0; i < framebuffers.size(); i++) {
		vkDestroyFramebuffer(logicalDevice, framebuffers[i], nullptr);
	}
}
void FzbRenderPass::createFramebuffers(std::vector<VkImageView> swapChainImageViews) {
	this->framebuffers.resize(setting.framebufferNum);
	std::vector<std::vector<VkImageView>> attachmentViews;
	attachmentViews.resize(setting.framebufferNum);
	for (int i = 0; i < setting.framebufferNum; i++) {
		if (setting.present) {
			attachmentViews[i].push_back(swapChainImageViews[i]);
		}
		for (int j = 0; j < images.size(); j++) {
			attachmentViews[i].push_back(images[j]->imageView);
		}
	}
	for (int i = 0; i < setting.framebufferNum; i++) {
		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = attachmentViews[i].size();
		framebufferInfo.pAttachments = attachmentViews[i].data();
		framebufferInfo.width = setting.extent.width;
		framebufferInfo.height = setting.extent.height;
		framebufferInfo.layers = 1;

		if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create framebuffer!");
		}
	}
}
void FzbRenderPass::render(VkCommandBuffer commandBuffer, uint32_t imageIndex, FzbScene* scene, std::vector<std::vector<VkDescriptorSet>> componentDescriptorSets) {

	VkRenderPassBeginInfo renderPassBeginInfo{};
	renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.renderPass = renderPass;
	renderPassBeginInfo.framebuffer = framebuffers[imageIndex];
	renderPassBeginInfo.renderArea.offset = { 0, 0 };
	renderPassBeginInfo.renderArea.extent = setting.extent;

	uint32_t clearAttachemtnNum = setting.useDepth + setting.colorAttachmentNum;
	std::vector<VkClearValue> clearValues(clearAttachemtnNum);
	for (int i = 0; i < clearAttachemtnNum - 1; i++) {
		clearValues[i].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
	}
	if (setting.useDepth) clearValues[clearAttachemtnNum - 1].depthStencil = { 1.0f, 0 };
	renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
	renderPassBeginInfo.pClearValues = clearValues.data();

	vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
	VkBuffer vertexBuffers[] = { scene->vertexBuffer.buffer };
	VkDeviceSize offsets[] = { 0 };
	vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
	vkCmdBindIndexBuffer(commandBuffer, scene->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	uint32_t subPassNum = subPasses.size();
	//std::map<std::string, FzbMaterial>& sceneMaterials = scene->sceneMaterials;
	for (int i = 0; i < subPasses.size(); i++) {
		subPasses[i].render(commandBuffer, componentDescriptorSets[i]);	//������ʵ���Խ�����������ŵ�subPass�ṹ����
		if (--subPassNum > 0) vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdEndRenderPass(commandBuffer);

	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("failed to record command buffer!");
	}
}


VkAttachmentDescription fzbCreateDepthAttachment(VkPhysicalDevice physicalDevice) {
	VkAttachmentDescription depthMapAttachment{};
	depthMapAttachment.format = fzbFindDepthFormat(physicalDevice);
	depthMapAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthMapAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthMapAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depthMapAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthMapAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthMapAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthMapAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
	return depthMapAttachment;
}
VkAttachmentDescription fzbCreateColorAttachment(VkFormat format, VkImageLayout layout) {
	VkAttachmentDescription colorAttachmentResolve{};
	colorAttachmentResolve.format = format;
	colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachmentResolve.finalLayout = layout;
	return colorAttachmentResolve;
}
VkAttachmentReference fzbCreateAttachmentReference(uint32_t attachmentIndex, VkImageLayout layout) {
	VkAttachmentReference depthMapAttachmentResolveRef{};
	depthMapAttachmentResolveRef.attachment = attachmentIndex;
	depthMapAttachmentResolveRef.layout = layout;
	return depthMapAttachmentResolveRef;
}
VkSubpassDescription fzbCreateSubPass(uint32_t colorAttachmentCount, VkAttachmentReference* colorAttachmentRefs, VkAttachmentReference* depthStencilAttachmentRefs, uint32_t inputAttachmentCount, VkAttachmentReference* inputAttachmentRefs) {
	VkSubpassDescription subPass{};
	subPass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subPass.colorAttachmentCount = colorAttachmentCount;
	subPass.pColorAttachments = colorAttachmentRefs;
	subPass.pDepthStencilAttachment = depthStencilAttachmentRefs;
	subPass.inputAttachmentCount = inputAttachmentCount;
	subPass.pInputAttachments = inputAttachmentRefs;
	return subPass;
}
VkSubpassDependency fzbCreateSubpassDependency(uint32_t scrSubpass, uint32_t dstSubpass, VkPipelineStageFlags srcStageMask, VkAccessFlags srcAccessMask, VkPipelineStageFlags dstStageMask, VkAccessFlags dstAccessMask) {
	VkSubpassDependency dependency{};
	dependency.srcSubpass = scrSubpass;
	dependency.dstSubpass = dstSubpass;
	dependency.srcStageMask = srcStageMask;
	dependency.srcAccessMask = srcAccessMask;
	dependency.dstStageMask = dstStageMask;
	dependency.dstAccessMask = dstAccessMask;
	return dependency;
}
