#pragma once

#include "StructSet.h"
#include "FzbPipeline.h"

#ifndef FZB_SHADER_H
#define FZB_SHADER_H

struct FzbShader {

	VkDevice logicalDevice;

	std::pair<bool, std::string> vertexShader;
	std::pair<bool, std::string> tessellationControlShader;
	std::pair<bool, std::string> tessellationEvaluateShader;
	std::pair<bool, std::string> geometryShader;
	std::pair<bool, std::string> fragmentShader;
	std::pair<bool, std::string> amplifyShader;
	std::pair<bool, std::string> meshShader;
	std::pair<bool, std::string> rayTracingShader;

	//bool useNormal;	//开启顶点属性的normal
	//bool useTexture;	//开启顶点属性的texCoord
	//bool useTBN;	//开启顶点属性的tangent
	FzbVertexFormat vertexFormat;
	bool useFaceNormal;	//如果mesh没有normal，但是shader一定要使用normal

	bool albedoTexture;
	bool normalTexture;
	bool materialTexture;
	bool heightTexture;

	VkPipelineLayout pipelineLayout = nullptr;
	VkPipeline pipeline = nullptr;

	FzbShader() {}

	FzbShader(VkDevice logicalDevice, bool useNormal, bool useTexCoord, bool useTangent) {
		this->logicalDevice = logicalDevice;
		this->vertexFormat.useNormal = useNormal;
		this->vertexFormat.useTexCoord = useTexCoord;
		this->vertexFormat.useTangent = useTangent;

		this->vertexShader = { true, "./shaders/spv/LitVertShader.spv" };
		this->fragmentShader = { true,  "./shaders/spv/LitFragShader.spv" };
	}

	FzbShader(VkDevice logicalDevice, FzbVertexFormat vertexFormat) {
		this->logicalDevice = logicalDevice;
		this->vertexFormat.useNormal = vertexFormat.useNormal;
		this->vertexFormat.useTexCoord = vertexFormat.useTexCoord;
		this->vertexFormat.useTangent = vertexFormat.useTangent;

		this->vertexShader = { true, "./shaders/spv/LitVertShader.spv" };
		this->fragmentShader = { true,  "./shaders/spv/LitFragShader.spv" };
	}

	void clean() {
		if (pipeline) {
			vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
			vkDestroyPipeline(logicalDevice, pipeline, nullptr);
		}
	}

	void clear() {
		this->vertexShader = { false, "" };
		this->tessellationControlShader = { false, "" };
		this->tessellationEvaluateShader = { false, "" };
		this->geometryShader = { false, "" };
		this->fragmentShader = { false, "" };
		this->amplifyShader = { false, "" };
		this->meshShader = { false, "" };
		this->rayTracingShader = { false, "" };

		clean();
	}

	void createPipeline(FzbPipelineCreateInfo pipelineCreateInfo, std::vector<VkDescriptorSetLayout> componentDescriptorSetLayouts, bool useSceneDescriptor, VkDescriptorSetLayout sceneDescriptorSetLayout, VkDescriptorSetLayout meshBatchDescriptorSetLayout) {
		std::map<VkShaderStageFlagBits, std::string> shaders;
		if (this->vertexShader.first) {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, this->vertexShader.second });
			if (this->tessellationControlShader.first && this->tessellationEvaluateShader.first) {
				shaders.insert({ VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, this->tessellationControlShader.second });
				shaders.insert({ VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, this->tessellationEvaluateShader.second });
			} 
			if(this->geometryShader.first) shaders.insert({ VK_SHADER_STAGE_GEOMETRY_BIT, this->geometryShader.second });
			if(this->fragmentShader.first) shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, this->fragmentShader.second });
		}
		else if (this->meshShader.first) {

		}
		else if (this->rayTracingShader.first) {

		}
		std::vector<VkPipelineShaderStageCreateInfo> shaderStages = fzbCreateShader(logicalDevice, shaders);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		VkVertexInputBindingDescription inputBindingDescriptor;
		std::vector<VkVertexInputAttributeDescription> inputAttributeDescription;
		if (!pipelineCreateInfo.screenSpace) {
			inputBindingDescriptor = this->vertexFormat.getBindingDescription();
			inputAttributeDescription = this->vertexFormat.getAttributeDescriptions();
			vertexInputInfo = fzbCreateVertexInputCreateInfo(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		}
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = fzbCreateInputAssemblyStateCreateInfo(pipelineCreateInfo.primitiveTopology);

		VkPipelineRasterizationStateCreateInfo rasterizer = fzbCreateRasterizationStateCreateInfo(pipelineCreateInfo.cullMode, pipelineCreateInfo.frontFace, pipelineCreateInfo.rasterizerExtensions);
		rasterizer.polygonMode = pipelineCreateInfo.polyMode;
		rasterizer.lineWidth = pipelineCreateInfo.lineWidth;

		VkPipelineMultisampleStateCreateInfo multisampling = fzbCreateMultisampleStateCreateInfo(pipelineCreateInfo.sampleShadingEnable, pipelineCreateInfo.sampleCount);
		std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments = { pipelineCreateInfo.colorBlendAttachments };
		VkPipelineColorBlendStateCreateInfo colorBlending = fzbCreateColorBlendStateCreateInfo(colorBlendAttachments);
		VkPipelineDepthStencilStateCreateInfo depthStencil = fzbCreateDepthStencilStateCreateInfo(pipelineCreateInfo.depthTestEnable, pipelineCreateInfo.depthWriteEnable, pipelineCreateInfo.depthCompareOp);
		
		VkPipelineDynamicStateCreateInfo dynamicState{};
		VkPipelineViewportStateCreateInfo viewportState{};
		VkViewport viewport = {};
		VkRect2D scissor = {};
		if (pipelineCreateInfo.dynamicView) {
			dynamicState = createDynamicStateCreateInfo(pipelineCreateInfo.dynamicStates);
		}
		else {
			viewportState = fzbCreateViewStateCreateInfo(pipelineCreateInfo.viewports, pipelineCreateInfo.scissors, pipelineCreateInfo.viewportExtensions);
			if (pipelineCreateInfo.viewports.size() == 0 || pipelineCreateInfo.scissors.size() == 0) {
				viewport.x = 0;
				viewport.y = 0;
				viewport.width = static_cast<float>(pipelineCreateInfo.extent.width);
				viewport.height = static_cast<float>(pipelineCreateInfo.extent.height);
				viewport.minDepth = 0.0f;
				viewport.maxDepth = 1.0f;
				scissor.offset = { 0, 0 };
				scissor.extent = pipelineCreateInfo.extent;
				viewportState.viewportCount = 1;
				viewportState.scissorCount = 1;
				viewportState.pViewports = &viewport;
				viewportState.pScissors = &scissor;
			}
		}

		std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
		for (int i = 0; i < componentDescriptorSetLayouts.size(); i++) {
			descriptorSetLayouts.push_back(componentDescriptorSetLayouts[i]);
		}
		if (useSceneDescriptor) {
			descriptorSetLayouts.push_back(sceneDescriptorSetLayout);
			descriptorSetLayouts.push_back(meshBatchDescriptorSetLayout);
		}
		pipelineLayout = fzbCreatePipelineLayout(logicalDevice, &descriptorSetLayouts);

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = shaderStages.size();
		pipelineInfo.pStages = shaderStages.data();
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		if (pipelineCreateInfo.dynamicView) {
			pipelineInfo.pDynamicState = &dynamicState;
		}
		else {
			pipelineInfo.pViewportState = &viewportState;
		}
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = pipelineCreateInfo.renderPass;	//先建立连接，获得索引
		pipelineInfo.subpass = pipelineCreateInfo.subPassIndex;	//对应renderpass的哪个子部分
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//可以直接使用现有pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		for (VkPipelineShaderStageCreateInfo shaderModule : shaderStages) {
			vkDestroyShaderModule(logicalDevice, shaderModule.module, nullptr);
		}
		
	}

	bool operator==(const FzbShader& other) const {
		return vertexShader == other.vertexShader &&
			tessellationControlShader == other.tessellationControlShader &&
			tessellationEvaluateShader == other.tessellationEvaluateShader &&
			geometryShader == other.geometryShader &&
			fragmentShader == other.fragmentShader &&
			amplifyShader == other.amplifyShader &&
			meshShader == other.meshShader &&
			rayTracingShader == other.rayTracingShader &&
			albedoTexture == other.albedoTexture &&
			normalTexture == other.normalTexture &&
			materialTexture == other.materialTexture &&
			heightTexture == other.heightTexture &&
			vertexFormat == other.vertexFormat;
	}

	//sbool operator=(const FzbShader& other) {
	//s	this->logicalDevice = other.logicalDevice;
	//s	this->vertexShader = other.vertexShader;
	//s	this->tessellationControlShader = other.tessellationControlShader;
	//s	this->tessellationEvaluateShader = other.tessellationEvaluateShader;
	//s	this->geometryShader = other.geometryShader;
	//s	this->fragmentShader = other.fragmentShader;
	//s	this->amplifyShader = other.amplifyShader;
	//s	this->meshShader = other.meshShader;
	//s	this->rayTracingShader = other.rayTracingShader;
	//s	this->vertexFormat = other.vertexFormat;
	//s	this->useFaceNormal = other.useFaceNormal;
	//s	this->albedoTexture = other.albedoTexture;
	//s	this->normalTexture = other.normalTexture;
	//s	this->materialTexture = other.materialTexture;
	//s	this->heightTexture = other.heightTexture;
	//s	this->pipelineLayout = other.pipelineLayout;
	//s	this->pipeline = other.pipeline;
	//s}
};

#endif