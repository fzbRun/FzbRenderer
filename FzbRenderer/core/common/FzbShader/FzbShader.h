#pragma once

#include "../FzbCommon.h"
#include "../FzbPipeline/FzbPipeline.h"
#include "../FzbMesh/FzbMesh.h"
#include "../FzbMaterial/FzbMaterial.h"
#include <fstream>
#include <set>

#ifndef FZB_SHADER_H
#define FZB_SHADER_H

//----------------------------------------------------------------------------------------------------------------------------
struct FzbShader;
struct FzbShaderVariant {
public:
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
	FzbShaderVariant(FzbShader* publicShader, FzbMaterial* material);

	void createDescriptor(VkDescriptorPool sceneDescriptorPool);

	//void changeVertexFormat(FzbVertexFormat newFzbVertexFormat);

	void clean();

	void createMeshBatch(std::map<FzbMesh*, FzbMaterial*>& meshMaterialPairs);

	std::vector<VkPipelineShaderStageCreateInfo> createShaderStates();
	void createPipeline(VkRenderPass renderPass, uint32_t subPassIndex, std::vector<VkDescriptorSetLayout> descriptorSetLayouts);

	void render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets);

	bool operator==(const FzbShaderVariant& other) const;
	bool operator==(const FzbMaterial& other) const;
};

//��Ҫ��һЩ��չ�����ж��Ƿ���Ҫ����
struct FzbShaderExtensionsSetting{
	bool swizzle = false;
	bool conservativeRasterization = false;
};
struct FzbShaderInfo {
	std::string shaderPath = "";
	FzbShaderExtensionsSetting extensions;
	bool staticCompile = false;
};
struct FzbShader {
	std::string path;
	FzbShaderProperty properties;
	std::string shaderVersion = "460";
	std::map<std::string, bool> macros;
	std::map<VkShaderStageFlagBits, std::string> shaders;
	std::vector<FzbShaderVariant> shaderVariants;

	VkExtent2D resolution = { 512, 512 };	//��ǰshaderҪ����ķֱ���
	FzbPipelineCreateInfo pipelineCreateInfo;

	bool useStaticCompile = false;
	std::map<VkShaderStageFlagBits, std::vector<uint32_t>> shaderSpvs;	//��ʹ������ʱ���룬ֱ��ʹ��ͨ��glslc����Ľ��

	FzbShader();
	/*
	shader���xml�ж�ȡ���ݣ���Ҫ��4����1. ������������������ 2. ��  3. shader���н׶�  4. ͼ�ι�����Ϣ
	���ۺ��Ƿ�������ز����������shader��properties��ֻ����û����������ݻ��ڱ���ʱ��ע�͵�
	*/
	FzbShader(std::string path, FzbShaderExtensionsSetting extensionsSetting = FzbShaderExtensionsSetting(), bool useStaticCompile = false);

	void clean();

	VkExtent2D getResolution();
	void setViewStateInfo(std::vector<VkViewport>& viewports, std::vector<VkRect2D>& scissors, void* viewportExtensios);

	void createShaderVariant(FzbMaterial* material);
	void createMeshBatch(std::map<FzbMesh*, FzbMaterial*>& meshMaterialPairs);
	void createDescriptor(VkDescriptorPool sceneDescriptorPool);
	void createPipeline(VkRenderPass renderPass, uint32_t subPassIndex, std::vector<VkDescriptorSetLayout> descriptorSetLayouts);
	void render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets);
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

			return seed;
		}
	};
}

#endif