#pragma once

#include "StructSet.h"
#include "FzbScene.h"
#include "FzbMesh.h"
#include <stdexcept>

#ifndef FZB_RENDERPASS_H
#define FZB_RENDERPASS_H

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
			hash_pair(seed, s.vertexShader);
			hash_pair(seed, s.tessellationControlShader);
			hash_pair(seed, s.tessellationEvaluateShader);
			hash_pair(seed, s.geometryShader);
			hash_pair(seed, s.fragmentShader);
			hash_pair(seed, s.amplifyShader);
			hash_pair(seed, s.meshShader);
			hash_pair(seed, s.rayTracingShader);

			// 对顶点格式进行哈希
			combine_hash(seed, hash<FzbVertexFormat>{}(s.vertexFormat));

			// 对其他布尔标志进行哈希
			combine_hash(seed, hash<bool>{}(s.useFaceNormal));
			combine_hash(seed, hash<bool>{}(s.albedoTexture));
			combine_hash(seed, hash<bool>{}(s.normalTexture));
			combine_hash(seed, hash<bool>{}(s.materialTexture));
			combine_hash(seed, hash<bool>{}(s.heightTexture));

			return seed;
		}
	};
}

struct FzbSubPass {

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;

	VkDescriptorPool descriptorPool = nullptr;
	VkDescriptorSetLayout meshBatchDescriptorSetLayout = nullptr;	//这个描述符集合主要是材质索引

	FzbPipelineCreateInfo pipelineCreateInfo;	//主要是pipeline的公共信息，如是否要背面剔除什么的
	std::vector<FzbMeshBatch> meshBatchs;

	FzbSubPass() {};
	FzbSubPass(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue) {
		this->physicalDevice = physicalDevice;
		this->logicalDevice = logicalDevice;
		this->commandPool = commandPool;
		this->graphicsQueue = graphicsQueue;
	}

	void clean() {
		for (int i = 0; i < this->meshBatchs.size(); i++) {
			meshBatchs[i].clean();
		}
		if (descriptorPool) {
			vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
			vkDestroyDescriptorSetLayout(logicalDevice, meshBatchDescriptorSetLayout, nullptr);
		}
	}

	//给定scene，就表明改subPass要处理的是这个scene中的mesh。excludShaderMap包含我不想在这个subpass中处理的shader
	void createMeshBatch(FzbScene* scene, FzbPipelineCreateInfo& pipelineCreateInfo, std::vector<VkDescriptorSetLayout> componentDescriptorSetLayouts, bool useSceneDescriptor = true, std::unordered_map<FzbShader, uint32_t> excludShaderMap = std::unordered_map<FzbShader, uint32_t>()) {
		std::unordered_map<FzbShader, uint32_t> uniqueShaderMap{};
		for (int i = 0; i < scene->sceneMeshSet.size(); i++) {
			if (excludShaderMap.count(scene->sceneMeshSet[i].shader) > 0)
				continue;
			if (uniqueShaderMap.count(scene->sceneMeshSet[i].shader) == 0) {
				uniqueShaderMap[scene->sceneMeshSet[i].shader] = this->meshBatchs.size();

				FzbMeshBatch meshBatch(physicalDevice, logicalDevice, commandPool, graphicsQueue);
				meshBatch.shader = scene->sceneMeshSet[i].shader;
				meshBatch.useSceneDescriptor = useSceneDescriptor;
				this->meshBatchs.push_back(meshBatch);
			}
			this->meshBatchs[uniqueShaderMap[scene->sceneMeshSet[i].shader]].meshes.push_back(&scene->sceneMeshSet[i]);
		}

		for (int i = 0; i < this->meshBatchs.size(); i++) {
			meshBatchs[i].createMeshBatchIndexBuffer(scene->sceneIndices);
			if (useSceneDescriptor) meshBatchs[i].createMeshBatchMaterialBuffer();
			meshBatchs[i].createDrawIndexedIndirectCommandBuffer();
		}
		if (useSceneDescriptor) createMeshBatchDescriptor();
		for (int i = 0; i < this->meshBatchs.size(); i++) {
			meshBatchs[i].shader.createPipeline(pipelineCreateInfo, componentDescriptorSetLayouts, useSceneDescriptor, scene->sceneDecriptorSetLayout, meshBatchDescriptorSetLayout);
		}
	}

	void createMeshBatchDescriptor() {
		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, this->meshBatchs.size()});	//材质、纹理、变换索引
		this->descriptorPool = fzbCreateDescriptorPool(logicalDevice, bufferTypeAndNum);

		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL };
		meshBatchDescriptorSetLayout = fzbCreateDescriptLayout(logicalDevice, 1, descriptorTypes, descriptorShaderFlags);
		for (int i = 0; i < this->meshBatchs.size(); i++) {
			meshBatchs[i].createDescriptorSet(descriptorPool, meshBatchDescriptorSetLayout);
		}
	}

	void render(VkCommandBuffer commandBuffer, std::vector<VkDescriptorSet> componentDescriptorSets, VkDescriptorSet sceneDescriptorSet) {
		for (int i = 0; i < this->meshBatchs.size(); i++) {
			this->meshBatchs[i].render(commandBuffer, componentDescriptorSets, sceneDescriptorSet);
		}
	}
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

	VkRenderPass renderPass = nullptr;
	std::vector<VkFramebuffer> framebuffers;
	std::vector<FzbSubPass> subPasses;

	FzbRenderPass() {};
	FzbRenderPass(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, FzbRenderPassSetting setting) {
		this->physicalDevice = physicalDevice;
		this->logicalDevice = logicalDevice;
		this->commandPool = commandPool;
		this->graphicsQueue = graphicsQueue;
		this->setting = setting;
	}

	void createRenderPass(std::vector<VkAttachmentDescription>* attachments, std::vector<VkSubpassDescription> subpasses, std::vector<VkSubpassDependency> dependencies) {
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = attachments ? attachments->size() : 0;
		renderPassInfo.pAttachments = attachments ? attachments->data() : nullptr;
		renderPassInfo.subpassCount = subpasses.size();
		renderPassInfo.pSubpasses = subpasses.data();
		renderPassInfo.dependencyCount = dependencies.size();
		renderPassInfo.pDependencies = dependencies.data();

		if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void clean() {
		if(renderPass) vkDestroyRenderPass(logicalDevice, renderPass, nullptr);
		for (int i = 0; i < subPasses.size(); i++) {
			subPasses[i].clean();
		}
		for (int i = 0; i < framebuffers.size(); i++) {
			vkDestroyFramebuffer(logicalDevice, framebuffers[i], nullptr);
		}
	}

	void createFramebuffers(std::vector<VkImageView> swapChainImageViews) {
		this->framebuffers.resize(setting.framebufferNum);
		std::vector<std::vector<VkImageView>> attachmentViews;
		attachmentViews.resize(setting.framebufferNum);
		for (int i = 0; i < setting.framebufferNum; i++) {
			if (setting.present) {
				attachmentViews[i].push_back(swapChainImageViews[i]);
			}
			for (int j = 0; j < images.size(); j++) {
				attachmentViews[i].push_back(images[j]->imageView);
			}
		}
		for (int i = 0; i < setting.framebufferNum; i++) {
			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = attachmentViews[i].size();
			framebufferInfo.pAttachments = attachmentViews[i].data();
			framebufferInfo.width = setting.extent.width;
			framebufferInfo.height = setting.extent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &framebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void render(VkCommandBuffer commandBuffer, uint32_t imageIndex, std::vector<FzbScene*> scenes, std::vector<std::vector<VkDescriptorSet>> componentDescriptorSets) {

		VkRenderPassBeginInfo renderPassBeginInfo{};
		renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.framebuffer = framebuffers[imageIndex];
		renderPassBeginInfo.renderArea.offset = { 0, 0 };
		renderPassBeginInfo.renderArea.extent = setting.extent;

		uint32_t clearAttachemtnNum = setting.useDepth + setting.colorAttachmentNum;
		std::vector<VkClearValue> clearValues(clearAttachemtnNum);
		for (int i = 0; i < clearAttachemtnNum - 1; i++) {
			clearValues[i].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		}
		if(setting.useDepth) clearValues[clearAttachemtnNum - 1].depthStencil = { 1.0f, 0 };
		renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassBeginInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
		uint32_t subPassNum = subPasses.size();
		for (int i = 0; i < subPasses.size(); i++) {
			VkBuffer vertexBuffers[] = { scenes[i]->vertexBuffer.buffer};
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
			subPasses[i].render(commandBuffer, componentDescriptorSets[i], scenes[i]->descriptorSet);
			if(--subPassNum > 0) vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);
		}

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}
};


VkAttachmentDescription fzbCreateDepthAttachment(VkPhysicalDevice physicalDevice) {
	VkAttachmentDescription depthMapAttachment{};
	depthMapAttachment.format = fzbFindDepthFormat(physicalDevice);
	depthMapAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthMapAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthMapAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depthMapAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depthMapAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthMapAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthMapAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
	return depthMapAttachment;
}

VkAttachmentDescription fzbCreateColorAttachment(VkFormat format = VK_FORMAT_R8G8B8A8_SRGB, VkImageLayout layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
	VkAttachmentDescription colorAttachmentResolve{};
	colorAttachmentResolve.format = format;
	colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachmentResolve.finalLayout = layout;
	return colorAttachmentResolve;
}

VkAttachmentReference fzbCreateAttachmentReference(uint32_t attachmentIndex, VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
	VkAttachmentReference depthMapAttachmentResolveRef{};
	depthMapAttachmentResolveRef.attachment = attachmentIndex;
	depthMapAttachmentResolveRef.layout = layout;
	return depthMapAttachmentResolveRef;
}

VkSubpassDescription fzbCreateSubPass(uint32_t colorAttachmentCount = 0, VkAttachmentReference* colorAttachmentRefs = nullptr,
	VkAttachmentReference* depthStencilAttachmentRefs = nullptr,
	uint32_t inputAttachmentCount = 0, VkAttachmentReference* inputAttachmentRefs = nullptr) {
	VkSubpassDescription subPass{};
	subPass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subPass.colorAttachmentCount = colorAttachmentCount;
	subPass.pColorAttachments = colorAttachmentRefs;
	subPass.pDepthStencilAttachment = depthStencilAttachmentRefs;
	subPass.inputAttachmentCount = inputAttachmentCount;
	subPass.pInputAttachments = inputAttachmentRefs;
	return subPass;
}

VkSubpassDependency fzbCreateSubpassDependency(uint32_t scrSubpass = VK_SUBPASS_EXTERNAL, uint32_t dstSubpass = 0,
	VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VkAccessFlags srcAccessMask = 0,
	VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VkAccessFlags dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
	) {
	VkSubpassDependency dependency{};
	dependency.srcSubpass = scrSubpass;
	dependency.dstSubpass = dstSubpass;
	dependency.srcStageMask = srcStageMask;
	dependency.srcAccessMask = srcAccessMask;
	dependency.dstStageMask = dstStageMask;
	dependency.dstAccessMask = dstAccessMask;
	return dependency;
}

#endif