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

			// �����ϣֵ������
			size_t seed = 0;

			// ������������϶����ϣֵ
			auto combine_hash = [](size_t& seed, size_t hash) {
				seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			};

			// �� FzbVertexFormat �ĳ�Ա���й�ϣ
			combine_hash(seed, hash<bool>{}(vf.useNormal));
			combine_hash(seed, hash<bool>{}(vf.useTexCoord));
			combine_hash(seed, hash<bool>{}(vf.useTangent));

			// ��� FzbVertexFormat ��������Ա��Ҳ��Ҫ��ӵ�����

			return seed;
		}
	};

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
	VkExtent2D extent;	//��ǰ������extent

	FzbPipelineCreateInfo pipelineCreateInfo;	//��Ҫ��pipeline�Ĺ�����Ϣ�����Ƿ�Ҫ�����޳�ʲô��
	//std::vector<FzbMeshBatch> meshBatchs;
	//std::vector<FzbScene*> scene;	//���subPassҪ����Щscene������Ⱦ
	std::vector<FzbShader*> shaders;	//���subPassҪ��Ⱦ��Щshader

	FzbSubPass();
	FzbSubPass(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, VkRenderPass renderPass, std::vector<VkDescriptorSetLayout> componentDescriptorSetLayouts, uint32_t subPassIndex, FzbScene* scene, std::vector<FzbShader*> shaders = std::vector<FzbShader*>(), VkExtent2D extent = VkExtent2D());
	FzbSubPass(FzbSubPassCreateInfo* createInfo);
	void clean();

	/*
	//����scene���ͱ�����subPassҪ����������scene�е�mesh��excludShaderMap�����Ҳ��������subpass�д����shader
	void createMeshBatch(FzbScene* scene, std::vector<VkDescriptorSetLayout> componentDescriptorSetLayouts, std::unordered_map<FzbShader, uint32_t> excludShaderMap = std::unordered_map<FzbShader, uint32_t>()) {
		std::unordered_map<FzbShader, uint32_t> uniqueShaderMap{};
		for (int i = 0; i < scene->sceneMeshSet.size(); i++) {
			FzbMesh& mesh = scene->sceneMeshSet[i];
			FzbMaterial material = scene->sceneMaterials[mesh.materialID];
			if (excludShaderMap.count(material.shader) > 0)		//������ʵ�shader���ų�����֮�У��򲻴���
				continue;
			if (uniqueShaderMap.count(material.shader) == 0) {	//�����ǰsubPassû������shader���򴴽���Ӧ��meshBatch
				uniqueShaderMap[material.shader] = this->meshBatchs.size();

				FzbMeshBatch meshBatch(physicalDevice, logicalDevice, commandPool, graphicsQueue);
				meshBatch.materialID = mesh.materialID;
				this->meshBatchs.push_back(meshBatch);
			}
			this->meshBatchs[uniqueShaderMap[material.shader]].meshes.push_back(&scene->sceneMeshSet[i]);	//����defulatMaterial��shader������ĳЩmaterial����ͬ���ͺϲ��ˡ�
		}

		for (int i = 0; i < this->meshBatchs.size(); i++) {
			meshBatchs[i].createMeshBatchIndexBuffer(scene->sceneIndices);

			//����pipeline
			FzbMaterial& material = scene->sceneMaterials[meshBatchs[i].materialID];
			std::vector<VkDescriptorSetLayout> descriptorSetLayouts = componentDescriptorSetLayouts;
			if(material.descriptorSetLayout) descriptorSetLayouts.push_back(material.descriptorSetLayout);	//��������������һ������
			descriptorSetLayouts.push_back(meshBatchs[i].meshes[0]->descriptorSetLayout);	//mesh�������������ϻ���
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