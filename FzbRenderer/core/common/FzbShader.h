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

//-----------------------------------------------------shader����---------------------------------------------------------
//�ݹ鴦������ļ�
std::string preprocessGLSL(
	const std::string& source,
	const std::filesystem::path& parentPath,
	std::set<std::filesystem::path>& includedFiles,
	int depth = 0);
// ��ں���
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
	shader���xml�ж�ȡ���ݣ���Ҫ��4����1. ������������������ 2. ��  3. shader���н׶�  4. ͼ�ι�����Ϣ
	���ۺ��Ƿ�������ز����������shader��properties��ֻ����û����������ݻ��ڱ���ʱ��ע�͵�
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

//-----------------------------------------------------����pipeline------------------------------------------------------------
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
			//����ָ����ɫ��������ֵ����������Ⱦʱָ���������ø�����Ч����Ϊ����ͨ���������Ż���û�㶮��
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
		pipelineInfo.renderPass = renderPass;	//�Ƚ������ӣ��������
		pipelineInfo.subpass = subPassIndex;	//��Ӧrenderpass���ĸ��Ӳ���
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//����ֱ��ʹ������pipeline
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
		//0�������uniform,1: material��texture��transform��2����ͬmeshBatch��materialIndex
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

			// �����ϣֵ������
			size_t seed = 0;

			// ������������϶����ϣֵ
			auto combine_hash = [](size_t& seed, size_t hash) {
				seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			};

			// ����ÿ�� pair<bool, std::string> ��Ա�Ĺ�ϣֵ
			auto hash_pair = [&combine_hash](size_t& seed, const auto& p) {
				combine_hash(seed, hash<bool>{}(p.first));
				combine_hash(seed, hash<std::string>{}(p.second));
			};

			// ��������ɫ��·�������ñ�־���й�ϣ
			//hash_pair(seed, s.vertexShader);
			//hash_pair(seed, s.tessellationControlShader);
			//hash_pair(seed, s.tessellationEvaluateShader);
			//hash_pair(seed, s.geometryShader);
			//hash_pair(seed, s.fragmentShader);
			//hash_pair(seed, s.amplifyShader);
			//hash_pair(seed, s.meshShader);
			//hash_pair(seed, s.rayTracingShader);

			// �Զ����ʽ���й�ϣ
			//combine_hash(seed, hash<FzbVertexFormat>{}(s.vertexFormat));

			// ������������־���й�ϣ
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