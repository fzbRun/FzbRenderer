#include "FzbPipeline.h"

#include <stdexcept>
#include <string>
#include <vector>
#include<fstream>
#include<filesystem>
#include <map>

VkPipelineVertexInputStateCreateInfo fzbCreateVertexInputCreateInfo(VkBool32 vertexInput, VkVertexInputBindingDescription* inputBindingDescriptor, std::vector<VkVertexInputAttributeDescription>* inputAttributeDescription) {
	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	if (vertexInput) {
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = inputBindingDescriptor;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(inputAttributeDescription->size());
		vertexInputInfo.pVertexAttributeDescriptions = inputAttributeDescription->data();

		return vertexInputInfo;

	}
	vertexInputInfo.vertexAttributeDescriptionCount = 0;
	vertexInputInfo.pVertexAttributeDescriptions = nullptr;
	vertexInputInfo.vertexBindingDescriptionCount = 0;
	vertexInputInfo.pVertexBindingDescriptions = nullptr;

	return vertexInputInfo;

}

VkPipelineInputAssemblyStateCreateInfo fzbCreateInputAssemblyStateCreateInfo(VkPrimitiveTopology primitiveTopology) {
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = primitiveTopology;
	inputAssembly.primitiveRestartEnable = VK_FALSE;
	return inputAssembly;
}

VkPipelineRasterizationConservativeStateCreateInfoEXT getRasterizationConservativeState(float OverestimationSize, void* pNext) {
	VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeState{};
	conservativeState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
	conservativeState.pNext = pNext;
	conservativeState.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
	conservativeState.extraPrimitiveOverestimationSize = OverestimationSize; // 根据需要设置
	return conservativeState;
}

VkPipelineRasterizationStateCreateInfo fzbCreateRasterizationStateCreateInfo(VkCullModeFlagBits cullMode, VkFrontFace frontFace, VkPolygonMode polyMode, float lineWidth, const void* pNext) {
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = polyMode;
	rasterizer.lineWidth = lineWidth;
	rasterizer.cullMode = cullMode;
	rasterizer.frontFace = frontFace;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasConstantFactor = 0.0f;
	rasterizer.depthBiasClamp = 0.0f;
	rasterizer.depthBiasSlopeFactor = 0.0f;
	rasterizer.pNext = pNext;
	return rasterizer;
}

VkPipelineRasterizationStateCreateInfo fzbCreateRasterizationStateCreateInfo(FzbPipelineCreateInfo pipelineCreateInfo) {
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = pipelineCreateInfo.polyMode;
	rasterizer.lineWidth = pipelineCreateInfo.lineWidth;
	rasterizer.cullMode = pipelineCreateInfo.cullMode;
	rasterizer.frontFace = pipelineCreateInfo.frontFace;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasConstantFactor = 0.0f;
	rasterizer.depthBiasClamp = 0.0f;
	rasterizer.depthBiasSlopeFactor = 0.0f;
	rasterizer.pNext = pipelineCreateInfo.rasterizerExtensions;
	return rasterizer;
}

VkPipelineMultisampleStateCreateInfo fzbCreateMultisampleStateCreateInfo(VkBool32 sampleShadingEnable, VkSampleCountFlagBits sampleCount) {
	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = sampleCount;
	multisampling.minSampleShading = 1.0f;// .2f;
	multisampling.pSampleMask = nullptr;
	multisampling.alphaToCoverageEnable = VK_FALSE;
	multisampling.alphaToOneEnable = VK_FALSE;
	return multisampling;
}

VkPipelineColorBlendAttachmentState fzbCreateColorBlendAttachmentState(
	VkColorComponentFlags colorWriteMask,
	VkBool32 blendEnable,
	VkBlendFactor srcColorBlendFactor,
	VkBlendFactor dstColorBlendFactor,
	VkBlendOp colorBlendOp,
	VkBlendFactor srcAlphaBlendFactor,
	VkBlendFactor dstAlphaBlendFactor,
	VkBlendOp alphaBlendOp
) {
	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = colorWriteMask;
	colorBlendAttachment.blendEnable = VK_FALSE;
	colorBlendAttachment.srcColorBlendFactor = srcColorBlendFactor;
	colorBlendAttachment.dstColorBlendFactor = dstColorBlendFactor;
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optiona
	return colorBlendAttachment;
}

VkPipelineColorBlendStateCreateInfo fzbCreateColorBlendStateCreateInfo(const std::vector<VkPipelineColorBlendAttachmentState>& colorBlendAttachments) {
	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
	colorBlending.attachmentCount = colorBlendAttachments.size();
	colorBlending.pAttachments = colorBlendAttachments.data();
	colorBlending.blendConstants[0] = 0.0f; // Optional
	colorBlending.blendConstants[1] = 0.0f; // Optional
	colorBlending.blendConstants[2] = 0.0f; // Optional
	colorBlending.blendConstants[3] = 0.0f; // Optional
	return colorBlending;
}

VkPipelineDepthStencilStateCreateInfo fzbCreateDepthStencilStateCreateInfo(VkBool32 depthTestEnable, VkBool32 depthWriteEnable, VkCompareOp depthCompareOp) {
	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = depthTestEnable;
	depthStencil.depthWriteEnable = depthWriteEnable;
	depthStencil.depthCompareOp = depthCompareOp;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.minDepthBounds = 0.0f; // Optional
	depthStencil.maxDepthBounds = 1.0f; // Optional
	depthStencil.stencilTestEnable = VK_FALSE;
	depthStencil.front = {}; // Optional
	depthStencil.back = {}; // Optional
	return depthStencil;
}

VkPipelineDepthStencilStateCreateInfo fzbCreateDepthStencilStateCreateInfo(FzbPipelineCreateInfo pipelineCreateInfo) {
	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = pipelineCreateInfo.depthTestEnable;
	depthStencil.depthWriteEnable = pipelineCreateInfo.depthWriteEnable;
	depthStencil.depthCompareOp = pipelineCreateInfo.depthCompareOp;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.minDepthBounds = 0.0f; // Optional
	depthStencil.maxDepthBounds = 1.0f; // Optional
	depthStencil.stencilTestEnable = VK_FALSE;
	depthStencil.front = {}; // Optional
	depthStencil.back = {}; // Optional
	return depthStencil;
}

VkPipelineDynamicStateCreateInfo createDynamicStateCreateInfo(const std::vector<VkDynamicState>& dynamicStates) {

	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();
	return dynamicState;
}

VkPipelineViewportSwizzleStateCreateInfoNV getViewportSwizzleState(std::vector<VkViewportSwizzleNV>& swizzles, void* pNext) {
	VkPipelineViewportSwizzleStateCreateInfoNV viewportSwizzleState{};
	viewportSwizzleState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV;
	viewportSwizzleState.pNext = pNext;
	viewportSwizzleState.pViewportSwizzles = swizzles.data();
	viewportSwizzleState.viewportCount = swizzles.size();
	return viewportSwizzleState;
}

VkPipelineViewportStateCreateInfo fzbCreateViewStateCreateInfo(uint32_t viewportNum) {
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = viewportNum;
	viewportState.scissorCount = viewportNum;
	return viewportState;
}
VkPipelineViewportStateCreateInfo fzbCreateViewStateCreateInfo(std::vector<VkViewport>& viewports, std::vector<VkRect2D>& scissors, const void* pNext) {
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = viewports.size();
	viewportState.scissorCount = scissors.size();
	viewportState.pViewports = viewports.data();
	viewportState.pScissors = scissors.data();
	viewportState.pNext = pNext;

	return viewportState;
}

VkPipelineLayout fzbCreatePipelineLayout(VkDevice logicalDevice, std::vector<VkDescriptorSetLayout>* descriptorSetLayout) {
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	if (descriptorSetLayout) {
		pipelineLayoutInfo.setLayoutCount = descriptorSetLayout->size();
		pipelineLayoutInfo.pSetLayouts = descriptorSetLayout->data();
	}
	else {
		pipelineLayoutInfo.setLayoutCount = 0;
		pipelineLayoutInfo.pSetLayouts = nullptr;
	}
	pipelineLayoutInfo.pushConstantRangeCount = 0;
	pipelineLayoutInfo.pPushConstantRanges = nullptr;

	VkPipelineLayout pipelineLayout;
	if (vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline layout!");
	}

	return pipelineLayout;
}

FzbPipelineCreateInfo::FzbPipelineCreateInfo() {

	this->colorBlendAttachments = { fzbCreateColorBlendAttachmentState() };

	VkViewport viewport;
	viewport.x = 0;
	viewport.y = 0;
	viewport.width = static_cast<float>(512);
	viewport.height = static_cast<float>(512);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	VkRect2D scissor;
	scissor.offset = { 0, 0 };
	scissor.extent = { 512, 512 };
	this->viewports.push_back(viewport);
	this->scissors.push_back(scissor);
};
FzbPipelineCreateInfo::FzbPipelineCreateInfo(VkExtent2D extent) {
	this->colorBlendAttachments = { fzbCreateColorBlendAttachmentState() };
	this->extent = extent;
}