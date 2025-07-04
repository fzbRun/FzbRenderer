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
			combine_hash(seed, hash<FzbVertexFormat>{}(s.vertexFormat));

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

struct FzbSubPass {

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;
	VkRenderPass renderPass;
	uint32_t subPassIndex;

	FzbPipelineCreateInfo pipelineCreateInfo;	//��Ҫ��pipeline�Ĺ�����Ϣ�����Ƿ�Ҫ�����޳�ʲô��
	std::vector<FzbMeshBatch> meshBatchs;

	FzbSubPass() {};
	FzbSubPass(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, VkRenderPass renderPass, uint32_t subPassIndex) {
		this->physicalDevice = physicalDevice;
		this->logicalDevice = logicalDevice;
		this->commandPool = commandPool;
		this->graphicsQueue = graphicsQueue;
		this->renderPass = renderPass;
		this->subPassIndex = subPassIndex;
	}

	void clean() {
		for (int i = 0; i < this->meshBatchs.size(); i++) {
			meshBatchs[i].clean();
		}
	}

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

	void render(VkCommandBuffer commandBuffer, std::map<std::string, FzbMaterial>& sceneMaterials, std::vector<VkDescriptorSet> componentDescriptorSets) {
		for (int i = 0; i < this->meshBatchs.size(); i++) {
			this->meshBatchs[i].render(commandBuffer, sceneMaterials, componentDescriptorSets);
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

	void render(VkCommandBuffer commandBuffer, uint32_t imageIndex, FzbScene* scene, std::vector<std::vector<VkDescriptorSet>> componentDescriptorSets) {

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
		VkBuffer vertexBuffers[] = { scene->vertexBuffer.buffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

		uint32_t subPassNum = subPasses.size();
		std::map<std::string, FzbMaterial>& sceneMaterials = scene->sceneMaterials;
		for (int i = 0; i < subPasses.size(); i++) {
			subPasses[i].render(commandBuffer, sceneMaterials, componentDescriptorSets[i]);	//������ʵ���Խ�����������ŵ�subPass�ṹ����
			if(--subPassNum > 0) vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);
		}

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}
};

//----------------------------------------------------------------------------------------------

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