#pragma once

#include "StructSet.h"
#include <stdexcept>
#include <string>
#include <vector>
#include<fstream>
#include<filesystem>
#include <map>

#ifndef FZB_PIPELINE_H
#define FZB_PIPELINE_H

static std::vector<char> readFile(const std::string& filename) {

	//std::cout << "Current working directory: "
	//	<< std::filesystem::current_path() << std::endl;

	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();
	return buffer;

}

VkShaderModule createShaderModule(VkDevice logicalDevice, const std::vector<char>& code) {

	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
		throw std::runtime_error("failed to create shader module!");
	}

	return shaderModule;

}

std::vector<VkPipelineShaderStageCreateInfo> fzbCreateShader(VkDevice logicalDevice, std::map<VkShaderStageFlagBits, std::string> shaders) {

	std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
	for (const auto& pair : shaders) {

		auto shadowVertShaderCode = readFile(pair.second);
		VkShaderModule shaderModule = createShaderModule(logicalDevice, shadowVertShaderCode);

		VkPipelineShaderStageCreateInfo shaderStageInfo{};
		shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStageInfo.stage = pair.first;
		shaderStageInfo.module = shaderModule;
		shaderStageInfo.pName = "main";
		//允许指定着色器常量的值，比起在渲染时指定变量配置更加有效，因为可以通过编译器优化（没搞懂）
		shaderStageInfo.pSpecializationInfo = nullptr;

		shaderStages.push_back(shaderStageInfo);

	}

	return shaderStages;

}

VkPipelineVertexInputStateCreateInfo fzbCreateVertexInputCreateInfo(VkBool32 vertexInput = VK_TRUE, VkVertexInputBindingDescription* inputBindingDescriptor = nullptr, std::vector<VkVertexInputAttributeDescription>* inputAttributeDescription = nullptr) {
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

VkPipelineViewportStateCreateInfo fzbCreateViewStateCreateInfo(std::vector<VkViewport>& viewports, std::vector<VkRect2D>& scissors, const void* pNext = nullptr) {
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = viewports.size();
	viewportState.scissorCount = scissors.size();
	viewportState.pViewports = viewports.data();
	viewportState.pScissors = scissors.data();
	viewportState.pNext = pNext;

	return viewportState;
}

VkPipelineRasterizationStateCreateInfo fzbCreateRasterizationStateCreateInfo(VkCullModeFlagBits cullMode = VK_CULL_MODE_BACK_BIT, VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, const void* pNext = nullptr) {
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = cullMode;
	rasterizer.frontFace = frontFace;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasConstantFactor = 0.0f;
	rasterizer.depthBiasClamp = 0.0f;
	rasterizer.depthBiasSlopeFactor = 0.0f;
	rasterizer.pNext = pNext;
	return rasterizer;
}

VkPipelineMultisampleStateCreateInfo fzbCreateMultisampleStateCreateInfo(VkBool32 sampleShadingEnable = VK_FALSE, VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT) {
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
	VkColorComponentFlags colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	VkBool32 blendEnable = VK_FALSE,
	VkBlendFactor srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
	VkBlendFactor dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
	VkBlendOp colorBlendOp = VK_BLEND_OP_ADD,
	VkBlendFactor srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
	VkBlendFactor dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
	VkBlendOp alphaBlendOp = VK_BLEND_OP_ADD
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

VkPipelineDepthStencilStateCreateInfo fzbCreateDepthStencilStateCreateInfo(VkBool32 depthTestEnable = VK_TRUE, VkBool32 depthWriteEnable = VK_TRUE, VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS) {
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

VkPipelineDynamicStateCreateInfo createDynamicStateCreateInfo(const std::vector<VkDynamicState>& dynamicStates) {

	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
	dynamicState.pDynamicStates = dynamicStates.data();
	return dynamicState;
}

VkPipelineLayout fzbCreatePipelineLayout(VkDevice logicalDevice, std::vector<VkDescriptorSetLayout>* descriptorSetLayout = nullptr) {
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

struct FzbPipelineCreateInfo {

	bool screenSpace = false;
	VkPrimitiveTopology primitiveTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkCullModeFlagBits cullMode = VK_CULL_MODE_BACK_BIT;
	VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	const void* rasterizerExtensions = nullptr;
	VkPolygonMode polyMode = VK_POLYGON_MODE_FILL;
	float lineWidth = 1.0f;;

	VkBool32 sampleShadingEnable = VK_FALSE;
	VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT;
	std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments;	//这个要根据renderPass来，我在想是否要搞个renderPass结构体去存储信息
	VkBool32 depthTestEnable = VK_TRUE;
	VkBool32 depthWriteEnable = VK_TRUE;
	VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS;
	bool dynamicView = false;
	std::vector<VkDynamicState> dynamicStates;
	VkExtent2D extent;
	std::vector<VkViewport> viewports;
	std::vector<VkRect2D> scissors;
	const void* viewportExtensions = nullptr;

	VkRenderPass renderPass;
	uint32_t subPassIndex = 0;

	FzbPipelineCreateInfo() {};
	FzbPipelineCreateInfo(VkRenderPass renderPass, VkExtent2D extent) {
		this->colorBlendAttachments = { fzbCreateColorBlendAttachmentState() };
		this->renderPass = renderPass;
		this->extent = extent;
	}

};

#endif