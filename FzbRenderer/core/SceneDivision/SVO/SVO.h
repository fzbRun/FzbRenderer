#pragma once

#include "./CUDA/createSVO.cuh"
#include "../../common/FzbComponent.h"


#ifndef SVO_H	//Sparse voxel octree
#define SVO_H

struct FzbSVOSetting {
	bool UseSVO = true;
	bool UseSVO_OnlyVoxelGridMap = false;
	bool UseSwizzle = false;
	bool UseBlock = false;
	bool UseConservativeRasterization = false;
	bool Present = false;
	int voxelNum = 64;
};

struct SVOUniform {
	glm::mat4 modelMatrix;
	glm::mat4 VP[3];
	glm::vec4 voxelSize_Num;
	glm::vec4 voxelStartPos;
};

class FzbSVO : public FzbComponent {

public:

	FzbSVOSetting svoSetting;
	FzbImage voxelGridMap;
	VkDescriptorSetLayout voxelGridMapDescriptorSetLayout;
	VkDescriptorSet voxelGridMapDescriptorSet;

	FzbScene* scene;
	std::vector<FzbVertex> cubeVertices;
	std::vector<uint32_t> cubeIndices;

	FzbBuffer sceneVertexBuffer;
	FzbBuffer sceneIndexBuffer;
	FzbBuffer sceneCubeVertexBuffer;
	FzbBuffer sceneCubeIndexBuffer;
	FzbBuffer sceneWireframeVertexBuffer;
	FzbBuffer sceneWireframeIndexBuffer;
	FzbBuffer svoUniformBuffer;

	VkRenderPass voxelGridMapRenderPass;
	VkPipeline voxelGridMapPipeline;
	VkPipelineLayout voxelGridMapPipelineLayout;

	VkRenderPass presentRenderPass;
	VkPipeline presentPipeline;
	VkPipelineLayout presentPipelineLayout;

	VkPipeline presentWireframePipeline;
	VkPipelineLayout presentWireframePipelineLayout;

	FzbImage depthMap;

	std::unique_ptr<SVOCuda> svoCuda;
	FzbBuffer nodePool;
	FzbBuffer voxelValueBuffer;

	VkDescriptorSetLayout svoDescriptorSetLayout;
	VkDescriptorSet svoDescriptorSet;

	FzbSemaphore vgmSemaphore;
	FzbSemaphore svoCudaSemaphore;
	FzbSemaphore presentSemaphore;

	VkFence fence;

	void addExtensions(FzbSVOSetting svoSetting, std::vector<const char*>& instanceExtensions, std::vector<const char*>& deviceExtensions, VkPhysicalDeviceFeatures& deviceFeatures) {

		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
			instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
		}

		if (svoSetting.UseConservativeRasterization) {
			deviceExtensions.push_back(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
		}
		if (svoSetting.UseSwizzle) {
			deviceExtensions.push_back(VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
			deviceExtensions.push_back(VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME);
			deviceExtensions.push_back(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME);
		}
		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
		}

		if (svoSetting.UseSwizzle)
			deviceFeatures.multiViewport = VK_TRUE;
		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			deviceFeatures.fillModeNonSolid = VK_TRUE;
			deviceFeatures.wideLines = VK_TRUE;
		}
	}

	void init(FzbMainComponent* renderer, FzbScene* scene, FzbSVOSetting setting) {
		
		this->physicalDevice = renderer->physicalDevice;
		this->logicalDevice = renderer->logicalDevice;
		this->graphicsQueue = renderer->graphicsQueue;
		this->swapChainExtent = renderer->swapChainExtent;
		this->swapChainImageFormat = renderer->swapChainImageFormat;
		this->swapChainImageViews = renderer->swapChainImageViews;
		this->commandPool = renderer->commandPool;
		this->scene = scene;
		this->svoSetting = setting;

		if (!this->svoSetting.UseSVO_OnlyVoxelGridMap) {
			this->svoCuda = std::make_unique<SVOCuda>();
		}

	}

	void activate() {

		createVoxelGridMapBuffer();
		initVoxelGridMap();
		createDescriptorPool();
		createVoxelGridMapDescriptor();
		createVoxelGridMapRenderPass();
		createVoxelGridMapFramebuffer();
		createVoxelGridMapPipeline();
		createVoxelGridMapSyncObjects();
		createVoxelGridMap();
		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			createSVOCuda();
			createSVODescriptor();
		}

	}

	void presentPrepare(VkDescriptorSetLayout uniformDescriptorSetLayout) {
		createPresentBuffer();
		initDepthMap();
		createPresentRenderPass(swapChainImageFormat);
		createPresentFrameBuffer(swapChainImageViews);
		if (svoSetting.UseSVO_OnlyVoxelGridMap) {
			createVGMPresentPipeline(uniformDescriptorSetLayout);
		}
		else {
			createSVOPresentPipeline(uniformDescriptorSetLayout);
		}
		presentSemaphore = fzbCreateSemaphore(false);
	}

	//用于测试体素化结果
	void present(VkDescriptorSet uniformDescriptorSet, uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) {

		VkCommandBuffer commandBuffer = commandBuffers[1];
		vkResetCommandBuffer(commandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassBeginInfo{};
		renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassBeginInfo.renderPass = presentRenderPass;
		renderPassBeginInfo.framebuffer = framebuffers[1][imageIndex];
		renderPassBeginInfo.renderArea.offset = { 0, 0 };
		renderPassBeginInfo.renderArea.extent = swapChainExtent;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };
		renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassBeginInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		if (svoSetting.UseBlock && svoSetting.UseSVO_OnlyVoxelGridMap) {
			VkBuffer cube_vertexBuffers[] = { sceneCubeVertexBuffer.buffer };
			VkDeviceSize cube_offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, cube_vertexBuffers, cube_offsets);
			vkCmdBindIndexBuffer(commandBuffer, sceneCubeIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
		}
		else {
			VkBuffer vertexBuffers[] = { sceneVertexBuffer.buffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(commandBuffer, sceneIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
		}
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipelineLayout, 0, 1, &uniformDescriptorSet, 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipelineLayout, 1, 1, &voxelGridMapDescriptorSet, 0, nullptr);
		if (svoSetting.UseBlock && svoSetting.UseSVO_OnlyVoxelGridMap) {
			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(sceneCubeIndexBuffer.size), std::pow(svoSetting.voxelNum, 3), 0, 0, 0);
		}
		else {
			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(scene->sceneIndices.size()), 1, 0, 0, 0);
		}

		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);

			VkBuffer cube_vertexBuffers[] = { sceneWireframeVertexBuffer.buffer };
			VkDeviceSize cube_offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, cube_vertexBuffers, cube_offsets);
			vkCmdBindIndexBuffer(commandBuffer, sceneWireframeIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentWireframePipeline);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentWireframePipelineLayout, 0, 1, &uniformDescriptorSet, 0, nullptr);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentWireframePipelineLayout, 1, 1, &voxelGridMapDescriptorSet, 0, nullptr);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentWireframePipelineLayout, 2, 1, &svoDescriptorSet, 0, nullptr);
			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(cubeIndices.size()), svoCuda->nodeBlockNum * 8 + 1, 0, 0, 0);
		}

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

		VkSemaphore waitSemaphores[] = { startSemaphore, svoSetting.UseSVO_OnlyVoxelGridMap ? vgmSemaphore.semaphore : svoCudaSemaphore.semaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &presentSemaphore.semaphore;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

	}

	void clean() {

		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			svoCuda->clean();
		}
		voxelGridMap.clean();
		depthMap.clean();

		//清理管线
		vkDestroyPipeline(logicalDevice, voxelGridMapPipeline, nullptr);
		vkDestroyPipelineLayout(logicalDevice, voxelGridMapPipelineLayout, nullptr);
		vkDestroyPipeline(logicalDevice, presentPipeline, nullptr);
		vkDestroyPipelineLayout(logicalDevice, presentPipelineLayout, nullptr);
		vkDestroyPipeline(logicalDevice, presentWireframePipeline, nullptr);
		vkDestroyPipelineLayout(logicalDevice, presentWireframePipelineLayout, nullptr);
		//清理渲染Pass
		vkDestroyRenderPass(logicalDevice, voxelGridMapRenderPass, nullptr);
		vkDestroyRenderPass(logicalDevice, presentRenderPass, nullptr);

		vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, voxelGridMapDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, svoDescriptorSetLayout, nullptr);

		fzbCleanSemaphore(vgmSemaphore);
		fzbCleanSemaphore(svoCudaSemaphore);
		fzbCleanSemaphore(presentSemaphore);
		fzbCleanFence(fence);

		sceneVertexBuffer.clean();
		sceneIndexBuffer.clean();
		sceneCubeVertexBuffer.clean();
		sceneCubeIndexBuffer.clean();
		sceneWireframeVertexBuffer.clean();
		sceneWireframeIndexBuffer.clean();
		svoUniformBuffer.clean();
		nodePool.clean();
		voxelValueBuffer.clean();
		for (int i = 0; i < framebuffers.size(); i++) {
			for (int j = 0; j < framebuffers[i].size(); j++) {
				vkDestroyFramebuffer(logicalDevice, framebuffers[i][j], nullptr);
			}
		}

	}

private:

	void createVoxelGridMapBuffer() {

		fzbCreateCommandBuffers(2);

		sceneVertexBuffer = fzbComponentCreateStorageBuffer<FzbVertex>(&scene->sceneVertices);
		sceneIndexBuffer = fzbComponentCreateStorageBuffer<uint32_t>(&scene->sceneIndices);

		svoUniformBuffer = fzbComponentCreateUniformBuffers<SVOUniform>();
		SVOUniform svoUniform;
		svoUniform.modelMatrix = glm::mat4(1.0f);

		float distanceX = scene->AABB.rightX - scene->AABB.leftX;
		float distanceY = scene->AABB.rightY - scene->AABB.leftY;
		float distanceZ = scene->AABB.rightZ - scene->AABB.leftZ;
		//想让顶点通过swizzle变换后得到正确的结果，必须保证投影矩阵是立方体的，这样xyz通过1减后才能是对应的
		//但是其实不需要VP，shader中其实没啥用
		float distance = glm::max(distanceX, glm::max(distanceY, distanceZ));
		float centerX = (scene->AABB.rightX + scene->AABB.leftX) * 0.5f;
		float centerY = (scene->AABB.rightY + scene->AABB.leftY) * 0.5f;
		float centerZ = (scene->AABB.rightZ + scene->AABB.leftZ) * 0.5f;
		//前面
		glm::vec3 viewPoint = glm::vec3(centerX, centerY, scene->AABB.rightZ + 0.2f);	//世界坐标右手螺旋，即+z朝后
		glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceX, 0.51f * distanceX, -0.51f * distanceY, 0.51f * distanceY, 0.1f, distanceZ + 0.5f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[0] = orthoMatrix * viewMatrix;

		//左边
		viewPoint = glm::vec3(scene->AABB.leftX - 0.2f, centerY, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceZ, 0.51f * distanceZ, -0.51f * distanceY, 0.51f * distanceY, 0.1f, distanceX + 0.5f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[1] = orthoMatrix * viewMatrix;

		//下面
		viewPoint = glm::vec3(centerX, scene->AABB.leftY - 0.2f, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceX, 0.51f * distanceX, -0.51f * distanceZ, 0.51f * distanceZ, 0.1f, distanceY + 0.5f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[2] = orthoMatrix * viewMatrix;
		svoUniform.voxelSize_Num = glm::vec4(distance / svoSetting.voxelNum, svoSetting.voxelNum, distance, 0.0f);
		svoUniform.voxelStartPos = glm::vec4(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f, 0.0f);

		memcpy(svoUniformBuffer.mapped, &svoUniform, sizeof(SVOUniform));

	}

	void initVoxelGridMap(){
		voxelGridMap = {};
		voxelGridMap.width = svoSetting.voxelNum;
		voxelGridMap.height = svoSetting.voxelNum;
		voxelGridMap.depth = svoSetting.voxelNum;
		voxelGridMap.type = VK_IMAGE_TYPE_3D;
		voxelGridMap.viewType = VK_IMAGE_VIEW_TYPE_3D;
		voxelGridMap.format = VK_FORMAT_R32_UINT;
		voxelGridMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		voxelGridMap.UseExternal = !svoSetting.UseSVO_OnlyVoxelGridMap;
		voxelGridMap.fzbCreateImage(physicalDevice, logicalDevice, commandPool, graphicsQueue);
	}

	void createDescriptorPool() {
		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 });
		}
		fzbComponentCreateDescriptorPool(bufferTypeAndNum);
	}

	void createVoxelGridMapDescriptor() {

		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_FRAGMENT_BIT };
		voxelGridMapDescriptorSetLayout = fzbComponentCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		voxelGridMapDescriptorSet = fzbComponentCreateDescriptorSet(voxelGridMapDescriptorSetLayout);

		std::array<VkWriteDescriptorSet, 2> voxelGridMapDescriptorWrites{};
		VkDescriptorBufferInfo voxelGridMapUniformBufferInfo{};
		voxelGridMapUniformBufferInfo.buffer = svoUniformBuffer.buffer;
		voxelGridMapUniformBufferInfo.offset = 0;
		voxelGridMapUniformBufferInfo.range = sizeof(SVOUniform);
		voxelGridMapDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voxelGridMapDescriptorWrites[0].dstSet = voxelGridMapDescriptorSet;
		voxelGridMapDescriptorWrites[0].dstBinding = 0;
		voxelGridMapDescriptorWrites[0].dstArrayElement = 0;
		voxelGridMapDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		voxelGridMapDescriptorWrites[0].descriptorCount = 1;
		voxelGridMapDescriptorWrites[0].pBufferInfo = &voxelGridMapUniformBufferInfo;

		VkDescriptorImageInfo voxelGridMapInfo{};
		voxelGridMapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		voxelGridMapInfo.imageView = voxelGridMap.imageView;
		voxelGridMapInfo.sampler = voxelGridMap.textureSampler;
		voxelGridMapDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voxelGridMapDescriptorWrites[1].dstSet = voxelGridMapDescriptorSet;
		voxelGridMapDescriptorWrites[1].dstBinding = 1;
		voxelGridMapDescriptorWrites[1].dstArrayElement = 0;
		voxelGridMapDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		voxelGridMapDescriptorWrites[1].descriptorCount = 1;
		voxelGridMapDescriptorWrites[1].pImageInfo = &voxelGridMapInfo;

		vkUpdateDescriptorSets(logicalDevice, voxelGridMapDescriptorWrites.size(), voxelGridMapDescriptorWrites.data(), 0, nullptr);
		//fzbDescriptor->descriptorSets.push_back({ voxelGridMapDescriptorSet });

	}

	void createVoxelGridMapRenderPass() {

		VkSubpassDescription voxelGridMapSubpass{};
		voxelGridMapSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

		VkSubpassDependency dependency = fzbCreateSubpassDependency();

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 0;
		renderPassInfo.pAttachments = nullptr;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &voxelGridMapSubpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &voxelGridMapRenderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

	}

	void createVoxelGridMapFramebuffer() {
		std::vector<std::vector<VkImageView>> attachmentImageViews;
		VkExtent2D extent = {
			static_cast<uint32_t>(this->swapChainExtent.width),
			static_cast<uint32_t>(swapChainExtent.height)
		};
		fzbCreateFramebuffer(1, extent, 0, attachmentImageViews, voxelGridMapRenderPass);
	}

	void createVoxelGridMapPipeline() {
		std::map<VkShaderStageFlagBits, std::string> shaders;
		if (svoSetting.UseSwizzle) {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "core/SceneDivision/SVO/shaders/useSwizzle/spv/voxelVert.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "core/SceneDivision/SVO/shaders/useSwizzle/spv/voxelFrag.spv" });
		}
		else {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "core/SceneDivision/SVO/shaders/unuseSwizzle/spv/voxelVert.spv" });
			shaders.insert({ VK_SHADER_STAGE_GEOMETRY_BIT, "core/SceneDivision/SVO/shaders/unuseSwizzle/spv/voxelGemo.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "core/SceneDivision/SVO/shaders/unuseSwizzle/spv/voxelFrag.spv" });
		}
		std::vector<VkPipelineShaderStageCreateInfo> shaderStages = fzbCreateShader(logicalDevice, shaders);

		VkVertexInputBindingDescription inputBindingDescriptor = FzbVertex::getBindingDescription();
		auto inputAttributeDescription = FzbVertex::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = fzbCreateVertexInputCreateInfo(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = fzbCreateInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeState{};
		if (svoSetting.UseConservativeRasterization) {
			conservativeState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
			conservativeState.pNext = NULL;
			conservativeState.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
			conservativeState.extraPrimitiveOverestimationSize = 0.5f; // 根据需要设置
			rasterizer = fzbCreateRasterizationStateCreateInfo(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE, &conservativeState);
		}
		else {
			rasterizer = fzbCreateRasterizationStateCreateInfo(VK_CULL_MODE_NONE);
		}

		VkPipelineMultisampleStateCreateInfo multisampling = fzbCreateMultisampleStateCreateInfo();
		VkPipelineColorBlendAttachmentState colorBlendAttachment = fzbCreateColorBlendAttachmentState();
		std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments = { colorBlendAttachment };
		VkPipelineColorBlendStateCreateInfo colorBlending = fzbCreateColorBlendStateCreateInfo(colorBlendAttachments);
		VkPipelineDepthStencilStateCreateInfo depthStencil = fzbCreateDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE);

		VkPipelineViewportStateCreateInfo viewportState = fzbCreateViewStateCreateInfo();
		VkViewport viewport = {};
		viewport.x = 0;
		viewport.y = 0;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		viewportState.pViewports = &viewport;
		viewportState.pScissors = &scissor;

		std::array< VkViewport, 3> viewports = {};
		std::array< VkRect2D, 3> scissors = {};
		std::array<VkViewportSwizzleNV, 3> swizzles = {};
		VkPipelineViewportSwizzleStateCreateInfoNV viewportSwizzleInfo{};
		if (svoSetting.UseSwizzle) {
			//std::array< VkViewport, 4> viewports = {};
			//std::array< VkRect2D, 4> scissors = {};
			//for (int y = 0; y < 2; y++) {
			//	for (int x = 0; x < 2; x++) {
			//		viewports[x + y * 2].x = x * swapChainExtent.width / 2;
			//		viewports[x + y * 2].y = y * swapChainExtent.height / 2;
			//		viewports[x + y * 2].width = static_cast<float>(swapChainExtent.width / 2);
			//		viewports[x + y * 2].height = static_cast<float>(swapChainExtent.height / 2);
			//		viewports[x + y * 2].minDepth = 0.0f;
			//		viewports[x + y * 2].maxDepth = 1.0f;
			//		scissors[x + y * 2].offset = { x * (int)swapChainExtent.width / 2, y * (int)swapChainExtent.height / 2 };
			//		scissors[x + y * 2].extent = { swapChainExtent.width / 2, swapChainExtent.height / 2 };;
			//	}
			//}
			//std::array< VkViewportSwizzleNV, 4 > swizzles = {};
			//for (int i = 0; i < 4; i++) {
			//	swizzles[i] = {
			//		i % 2 == 0 ? VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV : VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_X_NV, /* x */
			//		i / 2 == 0 ? VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV : VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV, /* y */
			//		VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV, /* z */
			//		VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV /* w */
			//	};
			//}
			for (int i = 0; i < viewports.size(); i++) {
				viewports[i].x = 0;
				viewports[i].y = 0;
				viewports[i].width = static_cast<float>(swapChainExtent.width);
				viewports[i].height = static_cast<float>(swapChainExtent.height);
				viewports[i].minDepth = 0.0f;
				viewports[i].maxDepth = 1.0f;

				scissors[i].offset = { 0, 0 };
				scissors[i].extent = swapChainExtent;
			}

			/*
			哇哇哇，这里很有意思，就是由于vulkan的NDC是y轴朝下的，所以我的投影矩阵的y乘了-1，所以才能正确显示。
			但是呢，在这里swizzle时，swizzles[0]和swizzle[1]都是同一个y，所以反转y后才能正确显示，所以没问题
			而对于swizzle[2]来说，反转后的y变成了它的z，这就导致原本是从下向上看的，编程了从上向下看，并且由于NEGATIVE_Z没有反转，因此图像变到了上方
			因此想要正确显示就必须样swizzle[2]的y变为VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV，z变为VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV
			*/
			
			swizzles[0] = {		//前面
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
			};
			swizzles[1] = {		//左面
				VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Z_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
			};
			swizzles[2] = {		//下面
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
			};

			viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewportState.viewportCount = viewports.size();
			viewportState.pViewports = viewports.data();
			viewportState.scissorCount = scissors.size();
			viewportState.pScissors = scissors.data();

			
			viewportSwizzleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV;
			viewportSwizzleInfo.pViewportSwizzles = swizzles.data();
			viewportSwizzleInfo.viewportCount = swizzles.size();

			viewportState.pNext = &viewportSwizzleInfo;

		}
		////一般渲染管道状态都是固定的，不能渲染循环中修改，但是某些状态可以，如视口，长宽和混合常数
		////同样通过宏来确定可动态修改的状态
		//std::vector<VkDynamicState> dynamicStates = {
		//	VK_DYNAMIC_STATE_VIEWPORT,
		//	VK_DYNAMIC_STATE_SCISSOR
		//};
		//VkPipelineDynamicStateCreateInfo dynamicState{};
		//dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		//dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		//dynamicState.pDynamicStates = dynamicStates.data();

		std::vector< VkDescriptorSetLayout> descriptorSetLayouts = { voxelGridMapDescriptorSetLayout };
		voxelGridMapPipelineLayout = fzbCreatePipelineLayout(logicalDevice, &descriptorSetLayouts);

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = shaderStages.size();
		pipelineInfo.pStages = shaderStages.data();
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		//pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = voxelGridMapPipelineLayout;
		pipelineInfo.renderPass = voxelGridMapRenderPass;	//先建立连接，获得索引
		pipelineInfo.subpass = 0;	//对应renderpass的哪个子部分
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//可以直接使用现有pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &voxelGridMapPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		for (VkPipelineShaderStageCreateInfo shaderModule : shaderStages) {
			vkDestroyShaderModule(logicalDevice, shaderModule.module, nullptr);
		}
	}

	void createVoxelGridMapSyncObjects() {	//这里应该返回一个信号量，然后阻塞主线程，知道渲染完成，才能唤醒
		if (svoSetting.UseSVO_OnlyVoxelGridMap) {
			vgmSemaphore = fzbCreateSemaphore(false);	//当vgm创建完成后唤醒
		}
		else {
			vgmSemaphore = fzbCreateSemaphore(true);		//当vgm创建完成后唤醒
			svoCudaSemaphore = fzbCreateSemaphore(true);		//当svo创建完成后唤醒
		}

		fence = fzbCreateFence();
	}

	void createVoxelGridMap() {

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		vkResetFences(logicalDevice, 1, &fence);
		VkCommandBuffer commandBuffer = commandBuffers[0];
		vkResetCommandBuffer(commandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkClearColorValue voxel_clearColor = {};
		voxel_clearColor.uint32[0] = 0;
		voxel_clearColor.uint32[1] = 0;
		voxel_clearColor.uint32[2] = 0;
		voxel_clearColor.uint32[3] = 0;
		voxelGridMap.fzbClearTexture(commandBuffer, voxel_clearColor, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = voxelGridMapRenderPass;
		renderPassInfo.framebuffer = framebuffers[0][0];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkBuffer vertexBuffers[] = { sceneVertexBuffer.buffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, sceneIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelGridMapPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelGridMapPipelineLayout, 0, 1, &voxelGridMapDescriptorSet, 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(scene->sceneIndices.size()), 1, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

		submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores =  &vgmSemaphore.semaphore;

		//执行完后解开fence
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

	}

	void createSVOCuda() {

		svoCuda->createSVOCuda(physicalDevice, voxelGridMap, vgmSemaphore.handle, svoCudaSemaphore.handle);

		//由于不能从cuda中直接导出数组的handle，因此我们需要先创建一个buffer，然后在cuda中将数据copy进去
		nodePool = fzbComponentCreateStorageBuffer(sizeof(FzbSVONode) * (8 * svoCuda->nodeBlockNum + 1), true);
		voxelValueBuffer = fzbComponentCreateStorageBuffer(sizeof(FzbVoxelValue) * svoCuda->voxelNum, true);

		svoCuda->getSVOCuda(physicalDevice, nodePool.handle, voxelValueBuffer.handle);

	}

	void createSVODescriptor() {
		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_ALL };
		svoDescriptorSetLayout = fzbComponentCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		svoDescriptorSet = fzbComponentCreateDescriptorSet(svoDescriptorSetLayout);

		std::array<VkWriteDescriptorSet, 2> svoDescriptorWrites{};
		VkDescriptorBufferInfo nodePoolBufferInfo{};
		nodePoolBufferInfo.buffer = nodePool.buffer;
		nodePoolBufferInfo.offset = 0;
		nodePoolBufferInfo.range = sizeof(FzbSVONode) * (svoCuda->nodeBlockNum * 8 + 1);
		svoDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		svoDescriptorWrites[0].dstSet = svoDescriptorSet;
		svoDescriptorWrites[0].dstBinding = 0;
		svoDescriptorWrites[0].dstArrayElement = 0;
		svoDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		svoDescriptorWrites[0].descriptorCount = 1;
		svoDescriptorWrites[0].pBufferInfo = &nodePoolBufferInfo;

		VkDescriptorBufferInfo voxelValueBufferInfo{};
		voxelValueBufferInfo.buffer = voxelValueBuffer.buffer;
		voxelValueBufferInfo.offset = 0;
		voxelValueBufferInfo.range = sizeof(FzbVoxelValue) * svoCuda->voxelNum;
		svoDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		svoDescriptorWrites[1].dstSet = svoDescriptorSet;
		svoDescriptorWrites[1].dstBinding = 1;
		svoDescriptorWrites[1].dstArrayElement = 0;
		svoDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		svoDescriptorWrites[1].descriptorCount = 1;
		svoDescriptorWrites[1].pBufferInfo = &voxelValueBufferInfo;

		vkUpdateDescriptorSets(logicalDevice, svoDescriptorWrites.size(), svoDescriptorWrites.data(), 0, nullptr);
	}

	void createPresentBuffer() {
		glm::vec3 cubeVertexOffset[8] = { glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f),
						  glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.0f, 1.0f, 1.0f) };
		if (svoSetting.UseSVO_OnlyVoxelGridMap) {
			if (svoSetting.UseBlock) {
				float distanceX = scene->AABB.rightX - scene->AABB.leftX;
				float distanceY = scene->AABB.rightY - scene->AABB.leftY;
				float distanceZ = scene->AABB.rightZ - scene->AABB.leftZ;
				float distance = glm::max(distanceX, glm::max(distanceY, distanceZ));
				float voxelSize = distance / svoSetting.voxelNum;

				float centerX = (scene->AABB.rightX + scene->AABB.leftX) * 0.5f;
				float centerY = (scene->AABB.rightY + scene->AABB.leftY) * 0.5f;
				float centerZ = (scene->AABB.rightZ + scene->AABB.leftZ) * 0.5f;

				cubeVertices.resize(8);
				for (int i = 0; i < 8; i++) {
					cubeVertices[i].pos = cubeVertexOffset[i] * voxelSize + glm::vec3(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f);
				}
				cubeIndices  = {
					1, 0, 3, 1, 3, 2,
					4, 5, 6, 4, 6, 7,
					5, 1, 2, 5, 2, 6,
					0, 4, 7, 0, 7, 3,
					7, 6, 2, 7, 2, 3,
					0, 1, 5, 0, 5, 4
				};
				sceneCubeVertexBuffer = fzbComponentCreateStorageBuffer<FzbVertex>(&cubeVertices);
				sceneCubeIndexBuffer = fzbComponentCreateStorageBuffer<uint32_t>(&cubeIndices);
			}
		}
		else {
			cubeVertices;
			cubeVertices.resize(8);
			for (int i = 0; i < 8; i++) {
				cubeVertices[i].pos = cubeVertexOffset[i];
			}
			cubeIndices = {
				0, 1, 1, 2, 2, 3, 3, 0,
				4, 5, 5, 6, 6, 7, 7, 4,
				0, 4, 1, 5, 2, 6, 3, 7
			};
			sceneWireframeVertexBuffer = fzbComponentCreateStorageBuffer<FzbVertex>(&cubeVertices);
			sceneWireframeIndexBuffer = fzbComponentCreateStorageBuffer<uint32_t>(&cubeIndices);
		}
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

	void createPresentRenderPass(VkFormat swapChainImageFormat) {
		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve };

		VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment(physicalDevice);
		VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		if(!svoSetting.UseSVO_OnlyVoxelGridMap)
			attachments.push_back(depthMapAttachment);

		std::vector<VkSubpassDescription> subpasses;
		VkSubpassDescription presentSubpass = fzbCreateSubPass(1, &colorAttachmentResolveRef, nullptr);
		if (!svoSetting.UseSVO_OnlyVoxelGridMap)
			presentSubpass.pDepthStencilAttachment = &depthMapAttachmentResolveRef;
		subpasses.push_back(presentSubpass);

		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			VkSubpassDescription presentWireframeSubpass = fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef);
			subpasses.push_back(presentWireframeSubpass);
		}

		VkSubpassDependency dependency = fzbCreateSubpassDependency();
		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			dependency = fzbCreateSubpassDependency(0, 1, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
		}
		std::array< VkSubpassDependency, 1> dependencies = { dependency };

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = attachments.size();
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = subpasses.size();
		renderPassInfo.pSubpasses = subpasses.data();
		renderPassInfo.dependencyCount = dependencies.size();
		renderPassInfo.pDependencies = dependencies.data();

		if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &presentRenderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void createPresentFrameBuffer(std::vector<VkImageView>& swapChainImageViews) {
		std::vector<std::vector<VkImageView>> attachmentImageViews;
		attachmentImageViews.resize(swapChainImageViews.size());
		for (int i = 0; i < swapChainImageViews.size(); i++) {
			attachmentImageViews[i].push_back(swapChainImageViews[i]);
			if (!svoSetting.UseSVO_OnlyVoxelGridMap)
				attachmentImageViews[i].push_back(depthMap.imageView);
		}
		fzbCreateFramebuffer(swapChainImageViews.size(), swapChainExtent, svoSetting.UseSVO_OnlyVoxelGridMap ? 1 : 2, attachmentImageViews, presentRenderPass);
	}

	void createVGMPresentPipeline(VkDescriptorSetLayout uniformDescriptorSetLayout) {
		std::map<VkShaderStageFlagBits, std::string> shaders;
		if (svoSetting.UseBlock) {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "core/SceneDivision/SVO/shaders/present_VGM/spv/presentVert_Block.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "core/SceneDivision/SVO/shaders/present_VGM/spv/presentFrag_Block.spv" });
		}
		else {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "core/SceneDivision/SVO/shaders/present/spv/presentVert.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "core/SceneDivision/SVO/shaders/present/spv/presentFrag.spv" });
		}
		std::vector<VkPipelineShaderStageCreateInfo> shaderStages = fzbCreateShader(logicalDevice, shaders);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		VkVertexInputBindingDescription inputBindingDescriptor = FzbVertex::getBindingDescription();
		auto inputAttributeDescription = FzbVertex::getAttributeDescriptions();
		vertexInputInfo = fzbCreateVertexInputCreateInfo(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = fzbCreateInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		VkPipelineRasterizationStateCreateInfo rasterizer = fzbCreateRasterizationStateCreateInfo(VK_CULL_MODE_NONE);

		VkPipelineMultisampleStateCreateInfo multisampling = fzbCreateMultisampleStateCreateInfo();
		VkPipelineColorBlendAttachmentState colorBlendAttachment = fzbCreateColorBlendAttachmentState();
		std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments = { colorBlendAttachment };
		VkPipelineColorBlendStateCreateInfo colorBlending = fzbCreateColorBlendStateCreateInfo(colorBlendAttachments);
		VkPipelineDepthStencilStateCreateInfo depthStencil = fzbCreateDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE);

		VkPipelineViewportStateCreateInfo viewportState = fzbCreateViewStateCreateInfo();
		VkViewport viewport = {};
		viewport.x = 0;
		viewport.y = 0;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		viewportState.pViewports = &viewport;
		viewportState.pScissors = &scissor;

		std::vector< VkDescriptorSetLayout> presentDescriptorSetLayouts = { uniformDescriptorSetLayout, voxelGridMapDescriptorSetLayout };
		presentPipelineLayout = fzbCreatePipelineLayout(logicalDevice, &presentDescriptorSetLayouts);

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = shaderStages.size();
		pipelineInfo.pStages = shaderStages.data();
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		//pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = presentPipelineLayout;
		pipelineInfo.renderPass = presentRenderPass;	//先建立连接，获得索引
		pipelineInfo.subpass = 0;	//对应renderpass的哪个子部分
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//可以直接使用现有pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &presentPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		for (VkPipelineShaderStageCreateInfo shaderModule : shaderStages) {
			vkDestroyShaderModule(logicalDevice, shaderModule.module, nullptr);
		}
	}

	void createSVOPresentPipeline(VkDescriptorSetLayout uniformDescriptorSetLayout) {
		std::map<VkShaderStageFlagBits, std::string> shaders;
		shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "core/SceneDivision/SVO/shaders/present/spv/presentVert.spv" });
		shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "core/SceneDivision/SVO/shaders/present/spv/presentFrag.spv" });
		std::vector<VkPipelineShaderStageCreateInfo> shaderStages = fzbCreateShader(logicalDevice, shaders);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		VkVertexInputBindingDescription inputBindingDescriptor = FzbVertex::getBindingDescription();
		auto inputAttributeDescription = FzbVertex::getAttributeDescriptions();
		vertexInputInfo = fzbCreateVertexInputCreateInfo(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = fzbCreateInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		VkPipelineRasterizationStateCreateInfo rasterizer = fzbCreateRasterizationStateCreateInfo(VK_CULL_MODE_NONE);

		VkPipelineMultisampleStateCreateInfo multisampling = fzbCreateMultisampleStateCreateInfo();
		VkPipelineColorBlendAttachmentState colorBlendAttachment = fzbCreateColorBlendAttachmentState();
		std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments = { colorBlendAttachment };
		VkPipelineColorBlendStateCreateInfo colorBlending = fzbCreateColorBlendStateCreateInfo(colorBlendAttachments);
		VkPipelineDepthStencilStateCreateInfo depthStencil = fzbCreateDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE);

		VkPipelineViewportStateCreateInfo viewportState = fzbCreateViewStateCreateInfo();
		VkViewport viewport = {};
		viewport.x = 0;
		viewport.y = 0;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		viewportState.pViewports = &viewport;
		viewportState.pScissors = &scissor;

		std::vector< VkDescriptorSetLayout> presentDescriptorSetLayouts = { uniformDescriptorSetLayout, voxelGridMapDescriptorSetLayout };
		presentPipelineLayout = fzbCreatePipelineLayout(logicalDevice, &presentDescriptorSetLayouts);

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = shaderStages.size();
		pipelineInfo.pStages = shaderStages.data();
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		//pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = presentPipelineLayout;
		pipelineInfo.renderPass = presentRenderPass;	//先建立连接，获得索引
		pipelineInfo.subpass = 0;	//对应renderpass的哪个子部分
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//可以直接使用现有pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &presentPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		for (VkPipelineShaderStageCreateInfo shaderModule : shaderStages) {
			vkDestroyShaderModule(logicalDevice, shaderModule.module, nullptr);
		}

		shaders = std::map<VkShaderStageFlagBits, std::string>();
		shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "core/SceneDivision/SVO/shaders/present_SVO/spv/vert.spv" });
		shaders.insert({ VK_SHADER_STAGE_GEOMETRY_BIT, "core/SceneDivision/SVO/shaders/present_SVO/spv/gemo.spv" });
		shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "core/SceneDivision/SVO/shaders/present_SVO/spv/frag.spv" });
		shaderStages = fzbCreateShader(logicalDevice, shaders);

		inputAssemblyInfo = fzbCreateInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);

		rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
		rasterizer.lineWidth = 1.0f;

		presentDescriptorSetLayouts = { uniformDescriptorSetLayout, voxelGridMapDescriptorSetLayout, svoDescriptorSetLayout };
		presentWireframePipelineLayout = fzbCreatePipelineLayout(logicalDevice, &presentDescriptorSetLayouts);

		pipelineInfo.stageCount = shaderStages.size();
		pipelineInfo.pStages = shaderStages.data();
		pipelineInfo.layout = presentWireframePipelineLayout;
		pipelineInfo.subpass = 1;
		if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &presentWireframePipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		for (VkPipelineShaderStageCreateInfo shaderModule : shaderStages) {
			vkDestroyShaderModule(logicalDevice, shaderModule.module, nullptr);
		}

	}

};

#endif
