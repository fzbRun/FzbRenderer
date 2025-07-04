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

struct FzbPipelineCreateInfo {

	VkPipelineVertexInputStateCreateInfo vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo;
	VkPipelineRasterizationStateCreateInfo rasterizerInfo;
	VkPipelineMultisampleStateCreateInfo multisamplingInfo;
	VkPipelineColorBlendStateCreateInfo colorBlendingInfo;
	VkPipelineDepthStencilStateCreateInfo depthStencilInfo;
	VkPipelineDynamicStateCreateInfo dynamicStateInfo;
	VkPipelineViewportStateCreateInfo viewportStateInfo;
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
	VkPipelineLayout pipelineLayout = nullptr;
	VkGraphicsPipelineCreateInfo pipelineInfo;
	VkPipeline pipeline = nullptr;

	bool screenSpace = false;
	VkPrimitiveTopology primitiveTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkCullModeFlagBits cullMode = VK_CULL_MODE_BACK_BIT;
	VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	VkPolygonMode polyMode = VK_POLYGON_MODE_FILL;
	float lineWidth = 1.0f;

	VkBool32 sampleShadingEnable = VK_FALSE;
	VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT;
	std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments;	//这个要根据renderPass来，我在想是否要搞个renderPass结构体去存储信息
	VkBool32 depthTestEnable = VK_TRUE;
	VkBool32 depthWriteEnable = VK_TRUE;
	VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS;
	bool dynamicView = false;
	std::vector<VkDynamicState> dynamicStates;
	VkExtent2D extent = { 512, 512 };
	std::vector<VkViewport> viewports;
	std::vector<VkRect2D> scissors;

	//VkRenderPass renderPass;
	//uint32_t subPassIndex = 0;

//----------------------------------------------光栅拓展--------------------------------------------
	//std::vector<std::shared_ptr<void>> rasterizerExtensionSet;
	VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeState;
	void* rasterizerExtensions = nullptr;

	//----------------------------------------------视口拓展--------------------------------------------
		//	std::vector<std::shared_ptr<void>> viewportExtensionSet;
	std::vector<VkViewportSwizzleNV> swizzles;
	VkPipelineViewportSwizzleStateCreateInfoNV viewportSwizzleState;
	void* viewportExtensions = nullptr;

	//--------------------------------------------------------------------------------------------------
	FzbPipelineCreateInfo();
	FzbPipelineCreateInfo(VkExtent2D extent);

};

/*
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
*/
VkPipelineVertexInputStateCreateInfo fzbCreateVertexInputCreateInfo(VkBool32 vertexInput = VK_TRUE, VkVertexInputBindingDescription* inputBindingDescriptor = nullptr, std::vector<VkVertexInputAttributeDescription>* inputAttributeDescription = nullptr);

VkPipelineInputAssemblyStateCreateInfo fzbCreateInputAssemblyStateCreateInfo(VkPrimitiveTopology primitiveTopology);

//std::shared_ptr<VkPipelineRasterizationConservativeStateCreateInfoEXT> getRasterizationConservativeState(float OverestimationSize = 0.5f, void* pNext = NULL) {
//	std::shared_ptr<VkPipelineRasterizationConservativeStateCreateInfoEXT> conservativeState = std::make_shared<VkPipelineRasterizationConservativeStateCreateInfoEXT>();
//	conservativeState->sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
//	conservativeState->pNext = pNext;
//	conservativeState->conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
//	conservativeState->extraPrimitiveOverestimationSize = OverestimationSize; // 根据需要设置
//	return conservativeState;
//}
VkPipelineRasterizationConservativeStateCreateInfoEXT getRasterizationConservativeState(float OverestimationSize = 0.5f, void* pNext = NULL);

VkPipelineRasterizationStateCreateInfo fzbCreateRasterizationStateCreateInfo(VkCullModeFlagBits cullMode = VK_CULL_MODE_BACK_BIT, VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, VkPolygonMode polyMode = VK_POLYGON_MODE_FILL, float lineWidth = 1.0f, const void* pNext = nullptr);

VkPipelineRasterizationStateCreateInfo fzbCreateRasterizationStateCreateInfo(FzbPipelineCreateInfo pipelineCreateInfo);

VkPipelineMultisampleStateCreateInfo fzbCreateMultisampleStateCreateInfo(VkBool32 sampleShadingEnable = VK_FALSE, VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT);

VkPipelineColorBlendAttachmentState fzbCreateColorBlendAttachmentState(
	VkColorComponentFlags colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	VkBool32 blendEnable = VK_FALSE,
	VkBlendFactor srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
	VkBlendFactor dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
	VkBlendOp colorBlendOp = VK_BLEND_OP_ADD,
	VkBlendFactor srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
	VkBlendFactor dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
	VkBlendOp alphaBlendOp = VK_BLEND_OP_ADD
);

VkPipelineColorBlendStateCreateInfo fzbCreateColorBlendStateCreateInfo(const std::vector<VkPipelineColorBlendAttachmentState>& colorBlendAttachments);

VkPipelineDepthStencilStateCreateInfo fzbCreateDepthStencilStateCreateInfo(VkBool32 depthTestEnable = VK_TRUE, VkBool32 depthWriteEnable = VK_TRUE, VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS);

VkPipelineDepthStencilStateCreateInfo fzbCreateDepthStencilStateCreateInfo(FzbPipelineCreateInfo pipelineCreateInfo);


VkPipelineDynamicStateCreateInfo createDynamicStateCreateInfo(const std::vector<VkDynamicState>& dynamicStates);

//std::shared_ptr<VkPipelineViewportSwizzleStateCreateInfoNV> getViewportSwizzleState(std::vector<VkViewportSwizzleNV>& swizzles, void* pNext = NULL) {
//	std::shared_ptr<VkPipelineViewportSwizzleStateCreateInfoNV> viewportSwizzleState = std::make_shared<VkPipelineViewportSwizzleStateCreateInfoNV>();
//	viewportSwizzleState->sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV;
//	viewportSwizzleState->pNext = pNext;
//	viewportSwizzleState->pViewportSwizzles = swizzles.data();
//	viewportSwizzleState->viewportCount = swizzles.size();
//	return viewportSwizzleState;
//}
VkPipelineViewportSwizzleStateCreateInfoNV getViewportSwizzleState(std::vector<VkViewportSwizzleNV>& swizzles, void* pNext = NULL);

VkPipelineViewportStateCreateInfo fzbCreateViewStateCreateInfo(std::vector<VkViewport>& viewports, std::vector<VkRect2D>& scissors, const void* pNext = nullptr);

VkPipelineLayout fzbCreatePipelineLayout(VkDevice logicalDevice, std::vector<VkDescriptorSetLayout>* descriptorSetLayout = nullptr);



#endif