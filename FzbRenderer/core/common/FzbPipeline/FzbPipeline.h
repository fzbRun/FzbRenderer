#pragma once

#include "../StructSet.h"

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
	std::vector<VkViewport> viewports;
	std::vector<VkRect2D> scissors;
//----------------------------------------------光栅拓展--------------------------------------------
	VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeRasterizationState;
	void* rasterizerExtensions = nullptr;
//----------------------------------------------视口拓展--------------------------------------------
	std::vector<VkViewportSwizzleNV> swizzles;
	VkPipelineViewportSwizzleStateCreateInfoNV viewportSwizzleState;
	void* viewportExtensions = nullptr;
//--------------------------------------------------------------------------------------------------
	FzbPipelineCreateInfo();
};

VkPipelineVertexInputStateCreateInfo fzbCreateVertexInputCreateInfo(VkBool32 vertexInput = VK_TRUE, VkVertexInputBindingDescription* inputBindingDescriptor = nullptr, std::vector<VkVertexInputAttributeDescription>* inputAttributeDescription = nullptr);
VkPipelineInputAssemblyStateCreateInfo fzbCreateInputAssemblyStateCreateInfo(VkPrimitiveTopology primitiveTopology);

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
VkPipelineViewportSwizzleStateCreateInfoNV getViewportSwizzleState(std::vector<VkViewportSwizzleNV>& swizzles, void* pNext = NULL);
VkPipelineViewportStateCreateInfo fzbCreateViewStateCreateInfo(uint32_t viewportNum);
VkPipelineViewportStateCreateInfo fzbCreateViewStateCreateInfo(std::vector<VkViewport>& viewports, std::vector<VkRect2D>& scissors, const void* pNext = nullptr);

VkPipelineLayout fzbCreatePipelineLayout(std::vector<VkDescriptorSetLayout>* descriptorSetLayout = nullptr);
#endif