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

struct FzbSubPass {
	VkRenderPass renderPass;
	std::vector<VkDescriptorSetLayout> componentDescriptorSetLayouts;
	std::vector<VkDescriptorSet> componentDescriptorSets;
	uint32_t subPassIndex;
	VkBuffer vertexBuffer;
	VkBuffer indexBuffer;

	std::vector<FzbShader*> shaders;	//这个subPass要渲染哪些shader

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