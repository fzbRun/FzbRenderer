#pragma once

#include "../FzbCommon.h"
#include "../FzbScene/FzbScene.h"
#include "../FzbShader/FzbShader.h"
#include "../FzbMesh/FzbMesh.h"
#include "../FzbPipeline/FzbPipeline.h"

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

struct FzbSubPass {
	VkRenderPass renderPass;
	std::vector<VkDescriptorSetLayout> componentDescriptorSetLayouts;
	std::vector<VkDescriptorSet> componentDescriptorSets;
	uint32_t subPassIndex;
	VkBuffer vertexBuffer;
	VkBuffer indexBuffer;

	std::vector<FzbShader*> shaders;	//���subPassҪ��Ⱦ��Щshader

	FzbSubPass();
	FzbSubPass(VkRenderPass renderPass, uint32_t subPassIndex, 
		std::vector<VkDescriptorSetLayout> componentDescriptorSetLayouts, std::vector<VkDescriptorSet> componentDescriptorSets,
		VkBuffer vertexBuffer, VkBuffer indexBuffer, std::vector<FzbShader*> shaders);
	void clean();
	void render(VkCommandBuffer commandBuffer, VkBuffer& vertexBuffer);

private:
	void createPipeline();
};

struct FzbRenderPassSetting {
	bool useDepth = false;
	uint32_t colorAttachmentNum = 1;
	VkExtent2D resolution;
	size_t framebufferNum;
	bool present;
};

struct FzbRenderPass {
	FzbRenderPassSetting setting;
	std::vector<FzbImage*> images;

	FzbScene* scene;
	VkRenderPass renderPass = nullptr;
	std::vector<VkFramebuffer> framebuffers;
	std::vector<FzbSubPass> subPasses;

	FzbRenderPass();
	FzbRenderPass(FzbRenderPassSetting setting);

	void createRenderPass(std::vector<VkAttachmentDescription>* attachments, std::vector<VkSubpassDescription> subpasses, std::vector<VkSubpassDependency> dependencies);

	void clean();
	void createFramebuffers(bool present);
	void addSubPass(FzbSubPass subPass);
	void render(VkCommandBuffer commandBuffer, uint32_t imageIndex);
};

//----------------------------------------------------------------------------------------------

VkAttachmentDescription fzbCreateDepthAttachment();

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