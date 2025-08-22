#pragma once

#include "StructSet.h"
#include "FzbPipeline.h"
#include "FzbMesh.h"
#include "FzbMaterial.h"
#include <fstream>
#include <set>

#ifndef FZB_SHADER_H
#define FZB_SHADER_H

const std::map<std::string, std::string> shaderPaths = {
	{"diffuse", "/core/ForwardRender/shaders/diffuse"}
};

//------------------------------------------------------------------------------------------------
glm::vec2 getfloat2FromString(std::string str);

glm::vec4 getRGBAFromString(std::string str);

//-----------------------------------------------------shader编译---------------------------------------------------------
//递归处理包含文件
std::string preprocessGLSL(
	const std::string& source,
	const std::filesystem::path& parentPath,
	std::set<std::filesystem::path>& includedFiles,
	int depth = 0);
// 入口函数
std::string preprocessShaderFile(const std::string& filePath, uint32_t& version);
std::vector<uint32_t> compileGLSL(
	const std::string& filePath,
	VkShaderStageFlagBits stage,
	const std::map<std::string, bool>& macros = {});

//----------------------------------------------------------------------------------------------------------------------------
struct FzbShader;
struct FzbShaderVariant {
public:
	VkDevice logicalDevice;
	FzbShader* publicShader;

	FzbVertexFormat vertexFormat;
	FzbShaderProperty properties;
	std::map<std::string, bool> macros;

	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	std::vector<FzbMaterial*> materials;

	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
	FzbMeshBatch meshBatch;

	VkPipelineLayout pipelineLayout = nullptr;
	VkPipeline pipeline = nullptr;
	//FzbSubPass* subPass;

	void changeVertexFormatAndMacros(FzbVertexFormat vertexFormat = FzbVertexFormat());

	FzbShaderVariant();
	FzbShaderVariant(FzbShader* publicShader, FzbMaterial* material, FzbVertexFormat vertexFormat = FzbVertexFormat());

	void createDescriptor(VkDescriptorPool sceneDescriptorPool, std::map<std::string, FzbImage>& sceneImages);

	//void changeVertexFormat(FzbVertexFormat newFzbVertexFormat);

	void clean();

	void createMeshBatch(VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, std::vector<FzbMesh>& sceneMeshSet);

	std::vector<VkPipelineShaderStageCreateInfo> createShaderStates();
	void createPipeline(VkRenderPass renderPass, uint32_t subPassIndex, std::vector<VkDescriptorSetLayout> descriptorSetLayouts);

	void render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets);

	bool operator==(const FzbShaderVariant& other) const;
};

struct FzbShader {

	VkDevice logicalDevice;

	std::string path;
	FzbShaderProperty properties;
	std::map<std::string, bool> macros;
	std::map<VkShaderStageFlagBits, std::string> shaders;

	std::vector<FzbShaderVariant> shaderVariants;
	FzbPipelineCreateInfo pipelineCreateInfo;

	FzbShader();

	FzbShader(VkDevice logicalDevice, bool useNormal, bool useTexCoord, bool useTangent);

	/*
	shader会从xml中读取数据，主要有4个：1. 所需参数（缓冲和纹理） 2. 宏  3. shader运行阶段  4. 图形管线信息
	无论宏是否开启，相关参数都会加入shader的properties，只不过没开启宏的数据会在编译时被注释掉
	*/
	FzbShader(VkDevice logicalDevice, std::string path);

	void clean();

	void createShaderVariant(FzbMaterial* material, FzbVertexFormat vertexFormat = FzbVertexFormat());
	void createMeshBatch(VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, std::vector<FzbMesh>& sceneMeshSet);

	void createDescriptor(VkDescriptorPool sceneDescriptorPool, std::map<std::string, FzbImage>& sceneImages);

	void createPipeline(VkRenderPass renderPass, uint32_t subPassIndex, VkDescriptorSetLayout meshDescriptorSetLayout, std::vector<VkDescriptorSetLayout> descriptorSetLayouts);

	void render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets, VkExtent2D extent = VkExtent2D());

/*
void initVertexFormat() {
	if (macros["useVertexNormal"]) this->vertexFormat.useNormal = true;
	if (macros["useVertexTexCoords"]) this->vertexFormat.useTexCoord = true;
	if (macros["useVertexTangent"]) this->vertexFormat.useTangent = true;
}

void changeVertexFormat(FzbVertexFormat newFzbVertexFormat) {
	this->vertexFormat = newFzbVertexFormat;
	if (this->vertexFormat.useNormal) macros["useVertexNormal"] = true;
	if (this->vertexFormat.useTexCoord) macros["useVertexTexCoords"] = true;
	if (this->vertexFormat.useTangent) macros["useVertexTangent"] = true;
}

void clean() {
	if (pipeline) {
		vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);
		vkDestroyPipeline(logicalDevice, pipeline, nullptr);
	}
	meshBatch.clean();
}

void clear() {

	clean();
}

//-----------------------------------------------------创建pipeline------------------------------------------------------------
	std::vector<VkPipelineShaderStageCreateInfo> createShaderStates() {

		std::vector<VkPipelineShaderStageCreateInfo> shaderStates;
		for (auto& shader : shaders) {
			VkShaderStageFlagBits shaderStage = shader.first;
			std::string shaderPath = this->path + "/" + shader.second;
			std::vector<uint32_t> shaderSpvCode = compileGLSL(shaderPath, shaderStage, this->macros);

			VkShaderModuleCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			createInfo.codeSize = shaderSpvCode.size() * sizeof(uint32_t);
			createInfo.pCode = shaderSpvCode.data();

			VkShaderModule shaderModule;
			if (vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
				throw std::runtime_error("failed to create shader module!");
			}

			VkPipelineShaderStageCreateInfo shaderStageInfo{};
			shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			shaderStageInfo.stage = shaderStage;
			shaderStageInfo.module = shaderModule;
			shaderStageInfo.pName = "main";
			//允许指定着色器常量的值，比起在渲染时指定变量配置更加有效，因为可以通过编译器优化（没搞懂）
			shaderStageInfo.pSpecializationInfo = nullptr;

			shaderStates.push_back(shaderStageInfo);
		}

		return shaderStates;
	}
	void createPipeline(VkRenderPass renderPass, uint32_t subPassIndex, std::vector<VkDescriptorSetLayout> descriptorSetLayouts) {
		std::vector<VkPipelineShaderStageCreateInfo> shaderStages = this->createShaderStates();

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		VkVertexInputBindingDescription inputBindingDescriptor;
		std::vector<VkVertexInputAttributeDescription> inputAttributeDescription;
		if (!pipelineCreateInfo.screenSpace) {
			inputBindingDescriptor = this->vertexFormat.getBindingDescription();
			inputAttributeDescription = this->vertexFormat.getAttributeDescriptions();
			vertexInputInfo = fzbCreateVertexInputCreateInfo(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		}
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = fzbCreateInputAssemblyStateCreateInfo(pipelineCreateInfo.primitiveTopology);

		VkPipelineRasterizationStateCreateInfo rasterizer = fzbCreateRasterizationStateCreateInfo(pipelineCreateInfo);

		VkPipelineMultisampleStateCreateInfo multisampling = fzbCreateMultisampleStateCreateInfo(pipelineCreateInfo.sampleShadingEnable, pipelineCreateInfo.sampleCount);
		std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments = { pipelineCreateInfo.colorBlendAttachments };
		VkPipelineColorBlendStateCreateInfo colorBlending = fzbCreateColorBlendStateCreateInfo(colorBlendAttachments);
		VkPipelineDepthStencilStateCreateInfo depthStencil = fzbCreateDepthStencilStateCreateInfo(pipelineCreateInfo);

		VkPipelineDynamicStateCreateInfo dynamicState{};
		VkPipelineViewportStateCreateInfo viewportState{};
		if (pipelineCreateInfo.dynamicView) {
			dynamicState = createDynamicStateCreateInfo(pipelineCreateInfo.dynamicStates);
		}
		else {
			viewportState = fzbCreateViewStateCreateInfo(pipelineCreateInfo.viewports, pipelineCreateInfo.scissors, pipelineCreateInfo.viewportExtensions);
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
		pipelineInfo.renderPass = renderPass;	//先建立连接，获得索引
		pipelineInfo.subpass = subPassIndex;	//对应renderpass的哪个子部分
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//可以直接使用现有pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		for (VkPipelineShaderStageCreateInfo shaderModule : shaderStages) {
			vkDestroyShaderModule(logicalDevice, shaderModule.module, nullptr);
		}

	}

	void render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets) {
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		//0：组件的uniform,1: material、texture、transform，2：不同meshBatch的materialIndex
		for (int i = 0; i < componentDescriptorSets.size(); i++) {
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, i, 1, &componentDescriptorSets[i], 0, nullptr);
		}
		if (meshBatch.meshes.size() > 0) {
			meshBatch.render(commandBuffer, pipelineLayout, componentDescriptorSets.size());
		}
	}
	*/
	bool operator==(const FzbShader& other) const;
};
namespace std {
	template<>
	struct hash<FzbShader> {
		std::size_t operator()(const FzbShader& s) const {
			using std::size_t;
			using std::hash;

			// 计算哈希值的种子
			size_t seed = 0;

			// 辅助函数：组合多个哈希值
			auto combine_hash = [](size_t& seed, size_t hash) {
				seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			};

			// 计算每个 pair<bool, std::string> 成员的哈希值
			auto hash_pair = [&combine_hash](size_t& seed, const auto& p) {
				combine_hash(seed, hash<bool>{}(p.first));
				combine_hash(seed, hash<std::string>{}(p.second));
			};

			// 对所有着色器路径和启用标志进行哈希
			//hash_pair(seed, s.vertexShader);
			//hash_pair(seed, s.tessellationControlShader);
			//hash_pair(seed, s.tessellationEvaluateShader);
			//hash_pair(seed, s.geometryShader);
			//hash_pair(seed, s.fragmentShader);
			//hash_pair(seed, s.amplifyShader);
			//hash_pair(seed, s.meshShader);
			//hash_pair(seed, s.rayTracingShader);

			// 对顶点格式进行哈希
			//combine_hash(seed, hash<FzbVertexFormat>{}(s.vertexFormat));

			// 对其他布尔标志进行哈希
			//combine_hash(seed, hash<bool>{}(s.useFaceNormal));
			//combine_hash(seed, hash<bool>{}(s.albedoTexture));
			//combine_hash(seed, hash<bool>{}(s.normalTexture));
			//combine_hash(seed, hash<bool>{}(s.materialTexture));
			//combine_hash(seed, hash<bool>{}(s.heightTexture));

			return seed;
		}
	};
}

#endif