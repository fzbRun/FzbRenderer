#pragma once

#include "StructSet.h"
#include "FzbDevice.h"
#include<fstream>

#ifndef FZB_PIPELINE_H
#define FZB_PIPELINE_H

/*
FzbPipeline��Ҫ���ڸ���������Ⱦģ��ʹ��
App�಻��ҪFzbPipeline���䱾�����FzbPipeline�Ĺ���
*/
class FzbPipeline {

public:

	//����
	VkDevice logicalDevice;

	FzbPipeline(std::unique_ptr<FzbDevice>& fzbDevice) {
		this->logicalDevice = fzbDevice->logicalDevice;
	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {

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

	std::vector<VkPipelineShaderStageCreateInfo> createShader(std::map<VkShaderStageFlagBits, std::string> shaders) {

		std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
		for (const auto& pair : shaders) {

			auto shadowVertShaderCode = readFile(pair.second);
			VkShaderModule shaderModule = createShaderModule(shadowVertShaderCode);

			VkPipelineShaderStageCreateInfo shaderStageInfo{};
			shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStageInfo.stage = pair.first;
			shaderStageInfo.module = shaderModule;
			shaderStageInfo.pName = "main";
			//����ָ����ɫ��������ֵ����������Ⱦʱָ���������ø�����Ч����Ϊ����ͨ���������Ż���û�㶮��
			shaderStageInfo.pSpecializationInfo = nullptr;

			shaderStages.push_back(shaderStageInfo);

		}

		return shaderStages;

	}

	template<typename T>
	VkPipelineVertexInputStateCreateInfo createVertexInputCreateInfo(VkBool32 vertexInput = VK_TRUE, VkVertexInputBindingDescription* inputBindingDescriptor = nullptr, std::vector<VkVertexInputAttributeDescription>* inputAttributeDescription = nullptr) {
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

	VkPipelineInputAssemblyStateCreateInfo createInputAssemblyStateCreateInfo(VkPrimitiveTopology primitiveTopology) {
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = primitiveTopology;
		inputAssembly.primitiveRestartEnable = VK_FALSE;
		return inputAssembly;
	}

	VkPipelineViewportStateCreateInfo createViewStateCreateInfo() {
		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		return viewportState;
	}

	VkPipelineRasterizationStateCreateInfo createRasterizationStateCreateInfo(VkCullModeFlagBits cullMode = VK_CULL_MODE_BACK_BIT, VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, const void* pNext = nullptr) {
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

	VkPipelineMultisampleStateCreateInfo createMultisampleStateCreateInfo(VkBool32 sampleShadingEnable = VK_FALSE, VkSampleCountFlagBits = VK_SAMPLE_COUNT_1_BIT) {
		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;// .2f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;
		return multisampling;
	}

	VkPipelineColorBlendAttachmentState createColorBlendAttachmentState(
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

	VkPipelineColorBlendStateCreateInfo createColorBlendStateCreateInfo(const std::vector<VkPipelineColorBlendAttachmentState>& colorBlendAttachments) {
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

	VkPipelineDepthStencilStateCreateInfo createDepthStencilStateCreateInfo(VkBool32 depthTestEnable = VK_TRUE, VkBool32 depthWriteEnable = VK_TRUE, VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS) {
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

	VkPipelineLayout createPipelineLayout(std::vector<VkDescriptorSetLayout>* descriptorSetLayout = nullptr) {
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

};

#endif