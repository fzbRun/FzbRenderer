#pragma once

#include "StructSet.h"
#include <stdexcept>

#ifndef FZB_RENDERPASS_H
#define FZB_RENDERPASS_H

struct FzbRenderPass {

	std::vector<VkAttachmentDescription> attachments;
	std::vector<VkSubpassDescription> subpasses;
	std::vector<VkSubpassDependency> dependencies;

	VkRenderPass renderPass;
	
};

FzbRenderPass fzbCreateFzbRenderPass(VkDevice logicalDevice, std::vector<VkAttachmentDescription> attachments, std::vector<VkSubpassDescription> subpasses, std::vector<VkSubpassDependency> dependencies) {
	
	FzbRenderPass fzbRenderPass;
	fzbRenderPass.attachments = attachments;
	fzbRenderPass.subpasses = subpasses;
	fzbRenderPass.dependencies = dependencies;

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = attachments.size();
	renderPassInfo.pAttachments = attachments.data();
	renderPassInfo.subpassCount = subpasses.size();
	renderPassInfo.pSubpasses = subpasses.data();
	renderPassInfo.dependencyCount = dependencies.size();
	renderPassInfo.pDependencies = dependencies.data();

	if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &fzbRenderPass.renderPass) != VK_SUCCESS) {
		throw std::runtime_error("failed to create render pass!");
	}

	return fzbRenderPass;
}

VkAttachmentDescription createDepthAttachment(VkFormat format) {
	VkAttachmentDescription depthMapAttachment{};
	depthMapAttachment.format = format;
	depthMapAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthMapAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthMapAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depthMapAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthMapAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthMapAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthMapAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
}

#endif