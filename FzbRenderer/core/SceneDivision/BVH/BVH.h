#pragma once

#include "../../common/StructSet.h"
#include "../../common/FzbComponent/FzbFeatureComponent.h"
#include "../../common/FzbRenderPass/FzbRenderPass.h"
#include "./CUDA/createBVH.cuh"
#include "../../common/FzbRenderer.h"

#ifndef BVH_H
#define BVH_H

struct FzbBVHSetting {
	uint32_t maxDepth;
};

struct FzbBVHUniform {
	uint32_t bvhTreeDepth;
};

struct FzbBVHPresentUniform {
	uint32_t nodeIndex;
	uint32_t bvhTreeDepth;
};

/*
class FzbBVH : public FzbFeatureComponent {

public:


private:

	FzbScene* mainScene;
	FzbBVHSetting setting;
	FzbBVHUniform kdUniform;

	FzbBuffer uniformBuffer;
	FzbBuffer bvhNodeArray;
	FzbBuffer bvhTriangleInfoArray;

	FzbImage depthMap;
	FzbRenderPass presentRenderPass;

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;

	FzbSemaphore bvhCudaSemaphore;
	FzbSemaphore presentSemaphore;

	std::unique_ptr<BVHCuda> bvhCuda;

	FzbShader presentShader;
	FzbMaterial presentMaterial;

	FzbBVH() {};

	void addExtensions(FzbBVHSetting kdSetting, std::vector<const char*>& instanceExtensions, std::vector<const char*>& deviceExtensions, VkPhysicalDeviceFeatures& deviceFeatures) {
		instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
		instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

		deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
		deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
		deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
		deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
	};

	FzbVertexFormat getComponentVertexFormat() {	//这个组件中的shader所需要用到的顶点属性
		return FzbVertexFormat();
	}

	void init(FzbMainComponent* renderer, FzbScene* scene, FzbBVHSetting setting, std::vector<FzbRenderPass*>& renderPasses) {

		initComponent(renderer);
		mainScene = scene;
		this->setting = setting;

		bvhCuda = std::make_unique<BVHCuda>();
		createBVH();
		if (setting.Present) presentPrepare(renderPasses);
	}
	void activate() {};
	void clean() {

		depthMap.clean();

		uniformBuffer.clean();
		bvhNodeArray.clean();
		bvhTriangleInfoArray.clean();

		bvhCudaSemaphore.clean(logicalDevice);
		presentSemaphore.clean(logicalDevice);

		vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);

		renderPass.clean();
	}

	void presentPrepare(std::vector<FzbRenderPass*>& renderPasses) {
		createBuffer();
		createDescriptor();
		initDepthMap();
		presentSemaphore = FzbSemaphore();
		createRenderPass(renderPasses);
	}

	void present(VkDescriptorSet mainComponentUniformDescriptorSet, uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) {
		VkCommandBuffer commandBuffer = commandBuffers[0];
		vkResetCommandBuffer(commandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		std::vector<FzbScene*> scenes = { &mainComponentScene };
		renderPass.render(commandBuffer, imageIndex, scenes, { {mainComponentUniformDescriptorSet, descriptorSet} });

		std::vector<VkSemaphore> waitSemaphores = { startSemaphore, bvhCudaSemaphore.semaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT };
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = waitSemaphores.size();
		submitInfo.pWaitSemaphores = waitSemaphores.data();
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &presentSemaphore.semaphore;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}
	}


	void createBVH() {
		bvhCudaSemaphore = FzbSemaphore(logicalDevice, true);		//当bvh创建完成后唤醒
		
		//bvhCuda->createBvhCuda_recursion(physicalDevice, mainComponentScene, bvhCudaSemaphore.handle, setting);
		bvhCuda->createBvhCuda_noRecursion(physicalDevice, *mainScene, bvhCudaSemaphore.handle, setting);

		bvhNodeArray = fzbComponentCreateStorageBuffer(sizeof(FzbBvhNode) * (bvhCuda->triangleNum * 2 - 1), true);
		bvhTriangleInfoArray = fzbComponentCreateStorageBuffer(sizeof(FzbBvhNodeTriangleInfo) * bvhCuda->triangleNum, true);

		bvhCuda->getBvhCuda(physicalDevice, bvhNodeArray.handle, bvhTriangleInfoArray.handle);

		bvhCuda->clean();
	}

	void createBuffer() {
		fzbCreateCommandBuffers(1);

		uniformBuffer = fzbComponentCreateUniformBuffers<FzbBVHPresentUniform>();
		FzbBVHPresentUniform uniform;
		uniform.nodeIndex = 12;
		memcpy(uniformBuffer.mapped, &uniform, sizeof(FzbBVHPresentUniform));
	}

	//先来说一下present的思路
	//1. 只传入bvhNodeArray，因为triangleInfoArray存的三角形信息只在光追中有用，我用传统管线展示
	//2. 在uniformBuffer中给定节点索引，在几何着色器中，判断三角形是否在节点AABB中，如果在则渲染
	//通过这种方式，我可以知道每个节点有哪些三角形，方便我debug。
	void createDescriptor() {
		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
		fzbComponentCreateDescriptorPool(bufferTypeAndNum);

		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_ALL };
		descriptorSetLayout = fzbComponentCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		descriptorSet = fzbComponentCreateDescriptorSet(descriptorSetLayout);

		std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
		VkDescriptorBufferInfo uniformBufferInfo{};
		uniformBufferInfo.buffer = uniformBuffer.buffer;
		uniformBufferInfo.offset = 0;
		uniformBufferInfo.range = sizeof(FzbBVHPresentUniform);
		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = descriptorSet;
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

		VkDescriptorBufferInfo nodeBufferInfo{};
		nodeBufferInfo.buffer = bvhNodeArray.buffer;
		nodeBufferInfo.offset = 0;
		nodeBufferInfo.range = sizeof(FzbBvhNode) * (bvhCuda->triangleNum * 2 - 1);
		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = descriptorSet;
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &nodeBufferInfo;

		vkUpdateDescriptorSets(logicalDevice, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
	}
	void initDepthMap() {
		depthMap = {};
		depthMap.width = swapChainExtent.width;
		depthMap.height = swapChainExtent.height;
		depthMap.type = VK_IMAGE_TYPE_2D;
		depthMap.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthMap.format = fzbFindDepthFormat(physicalDevice);
		depthMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		depthMap.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
		depthMap.fzbCreateImage(physicalDevice, logicalDevice, commandPool, graphicsQueue);
	}
	void createRenderPass(std::vector<FzbRenderPass*>& renderPasses) {
		presentMaterial = FzbMaterial(logicalDevice);
		presentShader = FzbShader(logicalDevice, getRootPath() + "/core/SceneDivision/BVH/shaders/present");
		presentShader.createShaderVariant(&presentMaterial, getComponentVertexFormat());
		for (int i = 0; i < mainScene->sceneMeshSet.size(); i++) {
			presentShader.shaderVariants[0].meshBatch.meshes.push_back(&mainScene->sceneMeshSet[i]);
		}
		presentShader.shaderVariants[0].meshBatch.useSameMaterial = true;
		presentShader.shaderVariants[0].meshBatch.materials.push_back(&presentMaterial);

		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment(physicalDevice);
		VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

		VkSubpassDescription subpass = fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef);
		VkSubpassDependency dependency = fzbCreateSubpassDependency();

		FzbRenderPassSetting setting = { true, 1, swapChainExtent, swapChainImageViews.size(), true };
		presentRenderPass = FzbRenderPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, setting);
		presentRenderPass.images.push_back(&depthMap);
		presentRenderPass.createRenderPass(&attachments, { subpass }, { dependency });
		presentRenderPass.createFramebuffers(swapChainImageViews);

		FzbSubPass presentSubPass = FzbSubPass(logicalDevice, presentRenderPass.renderPass, 0,
			{ mainScene->cameraAndLightsDescriptorSetLayout, descriptorSetLayout }, { mainScene->cameraAndLightsDescriptorSet, descriptorSet },
			mainScene->vertexPosBuffer.buffer, mainScene->indexPosBuffer.buffer, { &presentShader }, this->swapChainExtent);
		presentSubPass.createPipeline(mainScene->meshDescriptorSetLayout);
		presentRenderPass.subPasses.push_back(presentSubPass);

		renderPasses.push_back(&presentRenderPass);
	}

};
*/

class FzbBVH_Debug : public FzbFeatureComponent_LoopRender {

public:
	FzbBVH_Debug() {};
	FzbBVH_Debug(pugi::xml_node& BVHNode) {
		if (std::string(BVHNode.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
		else return;

		this->componentInfo.name = FZB_FEATURE_COMPONENT_BVH_DEBUG;
		this->componentInfo.type = FZB_LOOPRENDER_FEATURE_COMPONENT;
		this->componentInfo.vertexFormat = FzbVertexFormat();
		this->componentInfo.useMainSceneBufferHandle = { true, false, false };	//需要只有pos格式的顶点buffer和索引buffer，用来创建bvh

		addExtensions();
	}

	void init() override {
		FzbFeatureComponent_LoopRender::init();
		bvhCuda = std::make_unique<BVHCuda>();
		createBVH();
		createBuffer();
		createDescriptor();
		presentPrepare();
	}

	VkSemaphore render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) {
		VkCommandBuffer commandBuffer = commandBuffers[0];
		vkResetCommandBuffer(commandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		renderRenderPass.render(commandBuffer, imageIndex);

		std::vector<VkSemaphore> waitSemaphores = { startSemaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;	// waitSemaphores.size();
		submitInfo.pWaitSemaphores = waitSemaphores.data();
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &renderFinishedSemaphore.semaphore;

		if (vkQueueSubmit(FzbRenderer::globalData.graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		return renderFinishedSemaphore.semaphore;
	}

	void clean() {
		FzbFeatureComponent_LoopRender::clean();
		VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;

		depthMap.clean();

		uniformBuffer.clean();
		bvhNodeArray.clean();
		bvhTriangleInfoArray.clean();

		bvhCudaSemaphore.clean();

		presentMaterial.clean();
		presentShader.clean();

		vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
	}

private:

	FzbBVHSetting setting;
	FzbBVHUniform kdUniform;

	FzbBuffer uniformBuffer;
	FzbBuffer bvhNodeArray;
	FzbBuffer bvhTriangleInfoArray;

	FzbImage depthMap;

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;

	FzbSemaphore bvhCudaSemaphore;

	std::unique_ptr<BVHCuda> bvhCuda;

	FzbShader presentShader;
	FzbMaterial presentMaterial;

	void addExtensions() override {
		FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
		FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

		FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
		FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
		FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
		FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
	};

	void presentPrepare() override {
		createRenderPass();
	}

	void createBVH() {
		bvhCudaSemaphore = FzbSemaphore(true);		//当bvh创建完成后唤醒

		//bvhCuda->createBvhCuda_recursion(physicalDevice, mainComponentScene, bvhCudaSemaphore.handle, setting);
		bvhCuda->createBvhCuda_noRecursion(FzbRenderer::globalData.physicalDevice, mainScene, bvhCudaSemaphore.handle, setting.maxDepth);

		bvhNodeArray = fzbCreateStorageBuffer(sizeof(FzbBvhNode) * (bvhCuda->triangleNum * 2 - 1), true);
		bvhTriangleInfoArray = fzbCreateStorageBuffer(sizeof(FzbBvhNodeTriangleInfo) * bvhCuda->triangleNum, true);

		bvhCuda->getBvhCuda(FzbRenderer::globalData.physicalDevice, bvhNodeArray.handle, bvhTriangleInfoArray.handle);

		bvhCuda->clean();
	}

	void createBuffer() {
		fzbCreateCommandBuffers(1);

		uniformBuffer = fzbCreateUniformBuffers(sizeof(FzbBVHPresentUniform));
		FzbBVHPresentUniform uniform;
		uniform.nodeIndex = 0;
		memcpy(uniformBuffer.mapped, &uniform, sizeof(FzbBVHPresentUniform));
	}

	//先来说一下present的思路
	//1. 只传入bvhNodeArray，因为triangleInfoArray存的三角形信息只在光追中有用，我用传统管线展示
	//2. 在uniformBuffer中给定节点索引，在几何着色器中，判断三角形是否在节点AABB中，如果在则渲染
	//通过这种方式，我可以知道每个节点有哪些三角形，方便我debug。
	void createDescriptor() {
		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
		this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);

		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_ALL };
		descriptorSetLayout = fzbCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		descriptorSet = fzbCreateDescriptorSet(descriptorPool, descriptorSetLayout);

		std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
		VkDescriptorBufferInfo uniformBufferInfo{};
		uniformBufferInfo.buffer = uniformBuffer.buffer;
		uniformBufferInfo.offset = 0;
		uniformBufferInfo.range = sizeof(FzbBVHPresentUniform);
		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = descriptorSet;
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

		VkDescriptorBufferInfo nodeBufferInfo{};
		nodeBufferInfo.buffer = bvhNodeArray.buffer;
		nodeBufferInfo.offset = 0;
		nodeBufferInfo.range = sizeof(FzbBvhNode) * (bvhCuda->triangleNum * 2 - 1);
		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = descriptorSet;
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &nodeBufferInfo;

		vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
	}
	void createImages() override {
		VkExtent2D swapChainExtent = FzbRenderer::globalData.swapChainExtent;

		depthMap = {};
		depthMap.width = swapChainExtent.width;
		depthMap.height = swapChainExtent.height;
		depthMap.type = VK_IMAGE_TYPE_2D;
		depthMap.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthMap.format = fzbFindDepthFormat();
		depthMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		depthMap.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
		depthMap.initImage();

		frameBufferImages.push_back(&depthMap);
	}
	void createRenderPass() {
		presentMaterial = FzbMaterial();
		presentShader = FzbShader(fzbGetRootPath() + "/core/SceneDivision/BVH/shaders/present");
		presentShader.createShaderVariant(&presentMaterial, this->componentInfo.vertexFormat);
		for (int i = 0; i < mainScene->sceneMeshSet.size(); i++) {
			presentShader.shaderVariants[0].meshBatch.meshes.push_back(&mainScene->sceneMeshSet[i]);
		}
		presentShader.shaderVariants[0].meshBatch.useSameMaterial = true;
		presentShader.shaderVariants[0].meshBatch.materials.push_back(&presentMaterial);

		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(FzbRenderer::globalData.swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment();
		VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

		VkSubpassDescription subpass = fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef);
		VkSubpassDependency dependency = fzbCreateSubpassDependency();

		FzbRenderPassSetting renderPassSetting = { true, 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
		renderRenderPass.setting = renderPassSetting;
		renderRenderPass.createRenderPass(&attachments, { subpass }, { dependency });
		renderRenderPass.createFramebuffers(true);

		FzbSubPass presentSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
			{ mainScene->cameraAndLightsDescriptorSetLayout, descriptorSetLayout }, { mainScene->cameraAndLightsDescriptorSet, descriptorSet },
			mainScene->vertexPosBuffer.buffer, mainScene->indexPosBuffer.buffer, { &presentShader });
		renderRenderPass.subPasses.push_back(presentSubPass);
	}

};
#endif