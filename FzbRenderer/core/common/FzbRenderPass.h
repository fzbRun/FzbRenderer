#pragma once

#include "StructSet.h"
#include "FzbScene.h"
#include "FzbShader.h"
#include "FzbMesh.h"

#ifndef FZB_RENDERPASS_H
#define FZB_RENDERPASS_H
/*
namespace std {
	template<>
	struct hash<FzbVertexFormat> {
		std::size_t operator()(const FzbVertexFormat& vf) const {
			using std::size_t;
			using std::hash;

			// 计算哈希值的种子
			size_t seed = 0;

			// 辅助函数：组合多个哈希值
			auto combine_hash = [](size_t& seed, size_t hash) {
				seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			};

			// 对 FzbVertexFormat 的成员进行哈希
			combine_hash(seed, hash<bool>{}(vf.useNormal));
			combine_hash(seed, hash<bool>{}(vf.useTexCoord));
			combine_hash(seed, hash<bool>{}(vf.useTangent));

			// 如果 FzbVertexFormat 有其他成员，也需要添加到这里

			return seed;
		}
	};

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
*/

struct FzbSubPassCreateInfo {
	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;
	VkRenderPass renderPass;
	std::vector<VkDescriptorSetLayout> componentDescriptorSetLayouts;
	uint32_t subPassIndex;
	FzbScene* scene;
	std::vector<FzbShader*> shaders;
	VkExtent2D extent;
};

struct FzbSubPass {

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;
	VkRenderPass renderPass;
	uint32_t subPassIndex;
	VkExtent2D extent;	//当前交换链extent

	FzbPipelineCreateInfo pipelineCreateInfo;	//主要是pipeline的公共信息，如是否要背面剔除什么的
	//std::vector<FzbMeshBatch> meshBatchs;
	//std::vector<FzbScene*> scene;	//这个subPass要对哪些scene进行渲染
	std::vector<FzbShader*> shaders;	//这个subPass要渲染哪些shader

	FzbSubPass();
	FzbSubPass(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, VkRenderPass renderPass, std::vector<VkDescriptorSetLayout> componentDescriptorSetLayouts, uint32_t subPassIndex, FzbScene* scene, std::vector<FzbShader*> shaders = std::vector<FzbShader*>(), VkExtent2D extent = VkExtent2D());
	FzbSubPass(FzbSubPassCreateInfo* createInfo);
	void clean();

	/*
	//给定scene，就表明改subPass要处理的是这个scene中的mesh。excludShaderMap包含我不想在这个subpass中处理的shader
	void createMeshBatch(FzbScene* scene, std::vector<VkDescriptorSetLayout> componentDescriptorSetLayouts, std::unordered_map<FzbShader, uint32_t> excludShaderMap = std::unordered_map<FzbShader, uint32_t>()) {
		std::unordered_map<FzbShader, uint32_t> uniqueShaderMap{};
		for (int i = 0; i < scene->sceneMeshSet.size(); i++) {
			FzbMesh& mesh = scene->sceneMeshSet[i];
			FzbMaterial material = scene->sceneMaterials[mesh.materialID];
			if (excludShaderMap.count(material.shader) > 0)		//如果材质的shader在排除集合之中，则不处理
				continue;
			if (uniqueShaderMap.count(material.shader) == 0) {	//如果当前subPass没有这种shader，则创建相应的meshBatch
				uniqueShaderMap[material.shader] = this->meshBatchs.size();

				FzbMeshBatch meshBatch(physicalDevice, logicalDevice, commandPool, graphicsQueue);
				meshBatch.materialID = mesh.materialID;
				this->meshBatchs.push_back(meshBatch);
			}
			this->meshBatchs[uniqueShaderMap[material.shader]].meshes.push_back(&scene->sceneMeshSet[i]);	//这里defulatMaterial的shader可能与某些material的相同，就合并了。
		}

		for (int i = 0; i < this->meshBatchs.size(); i++) {
			meshBatchs[i].createMeshBatchIndexBuffer(scene->sceneIndices);

			//创建pipeline
			FzbMaterial& material = scene->sceneMaterials[meshBatchs[i].materialID];
			std::vector<VkDescriptorSetLayout> descriptorSetLayouts = componentDescriptorSetLayouts;
			if(material.descriptorSetLayout) descriptorSetLayouts.push_back(material.descriptorSetLayout);	//材质描述符，不一定会有
			descriptorSetLayouts.push_back(meshBatchs[i].meshes[0]->descriptorSetLayout);	//mesh描述符，基本上会有
			material.shader.createPipeline(renderPass, subPassIndex, descriptorSetLayouts);
		}
	}
	*/

	void render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets);
};

struct FzbRenderPassSetting {
	bool useDepth = false;
	uint32_t colorAttachmentNum = 1;
	VkExtent2D extent;
	uint32_t framebufferNum;
	bool present;
};

struct FzbRenderPass {

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;

	FzbRenderPassSetting setting;
	std::vector<FzbImage*> images;

	FzbScene* scene;
	VkRenderPass renderPass = nullptr;
	std::vector<VkFramebuffer> framebuffers;
	std::vector<FzbSubPass> subPasses;

	FzbRenderPass();
	FzbRenderPass(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, FzbRenderPassSetting setting);

	void createRenderPass(std::vector<VkAttachmentDescription>* attachments, std::vector<VkSubpassDescription> subpasses, std::vector<VkSubpassDependency> dependencies);

	void clean();

	void createFramebuffers(std::vector<VkImageView> swapChainImageViews = std::vector<VkImageView>());

	void render(VkCommandBuffer commandBuffer, uint32_t imageIndex, FzbScene* scene, std::vector<std::vector<VkDescriptorSet>> componentDescriptorSets);
};

//----------------------------------------------------------------------------------------------

VkAttachmentDescription fzbCreateDepthAttachment(VkPhysicalDevice physicalDevice);

VkAttachmentDescription fzbCreateColorAttachment(VkFormat format = VK_FORMAT_R8G8B8A8_SRGB, VkImageLayout layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

VkAttachmentReference fzbCreateAttachmentReference(uint32_t attachmentIndex, VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

VkSubpassDescription fzbCreateSubPass(uint32_t colorAttachmentCount = 0, VkAttachmentReference* colorAttachmentRefs = nullptr,
	VkAttachmentReference* depthStencilAttachmentRefs = nullptr,
	uint32_t inputAttachmentCount = 0, VkAttachmentReference* inputAttachmentRefs = nullptr);

VkSubpassDependency fzbCreateSubpassDependency(uint32_t scrSubpass = VK_SUBPASS_EXTERNAL, uint32_t dstSubpass = 0,
	VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VkAccessFlags srcAccessMask = 0,
	VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VkAccessFlags dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
);

#endif