#pragma once

#include "../../common/StructSet.h"
#include "../../common/FzbComponent.h"
#include "AccelerationStructure/accelerationStructure.h"

#ifndef PATHTRACING_H
#define PATHTRACING_H

enum ASType
{
	SVO = 0,
	BVH = 1,
	KD = 2
};

enum HardwarePathTracingType {
	rayQuery = 0
};

struct FzbPathTracingSetting {
	bool UseHardware;	//硬件光追还是软件光追，这里的硬件指的是RT core
	HardwarePathTracingType hardwarePTType;
	ASType asType;
	PathTracingUnifom ptUniform;
};

struct PathTracingUnifom {
	uint32_t spp = 1;
	float RR = 0.8f;
};

class PathTracing : public FzbComponent {

public:
	FzbScene* scene;
	FzbPathTracingSetting ptSetting;

	VkPhysicalDeviceBufferDeviceAddressFeaturesKHR bufferFeatures{};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
	VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};

	std::unique_ptr<FzbAccelerationStructure> topASs;
	std::unique_ptr<FzbAccelerationStructure> bottomASs;

	FzbBuffer vertexBuffer;
	FzbBuffer indexBuffer;
	FzbBuffer transformBuffer;	//VkTransformMatrixKHR

	FzbBuffer ptUniformBuffer;

	FzbImage depthMap;

	VkDescriptorSetLayout rayQueryDescriptorSetLayout;	//一个uniform，一个加速结构
	VkDescriptorSet rayQueryDescriptorSet;

	VkRenderPass rayQueryRenderPass;
	VkPipeline rayQueryPipeline;
	VkPipelineLayout rayQueryPipelineLayout;

	FzbSemaphore ptFinishedSemaphore;
	
	void addExtensions(FzbPathTracingSetting ptSetting, std::vector<const char*>& instanceExtensions, uint32_t& apiVersion, std::vector<const char*>& deviceExtensions, VkPhysicalDeviceFeatures& deviceFeatures, void* pNextFeatures) {
		if (ptSetting.UseHardware) {
			apiVersion = VK_API_VERSION_1_1;

			deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
			deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);

			bufferFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
			bufferFeatures.pNext = nullptr;
			bufferFeatures.bufferDeviceAddress = VK_TRUE;

			accelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
			accelerationStructureFeatures.pNext = &bufferFeatures;
			accelerationStructureFeatures.accelerationStructure = VK_TRUE;

			rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
			rayQueryFeatures.pNext = &accelerationStructureFeatures;
			rayQueryFeatures.rayQuery = VK_TRUE;

			pNextFeatures = &rayQueryFeatures;
		}
	}

	void init(FzbMainComponent* renderer, FzbScene* scene, FzbPathTracingSetting setting) {
		this->physicalDevice = renderer->physicalDevice;
		this->logicalDevice = renderer->logicalDevice;
		this->graphicsQueue = renderer->graphicsQueue;
		this->swapChainExtent = renderer->swapChainExtent;
		this->swapChainImageFormat = renderer->swapChainImageFormat;
		this->swapChainImageViews = renderer->swapChainImageViews;
		this->commandPool = renderer->commandPool;
		this->scene = scene;
		this->ptSetting = setting;
	}

	void ptPresentPrepare(VkDescriptorSetLayout cameraUniformDescriptorSetLayout) {
		createPTBuffer();
		if (ptSetting.UseHardware && ptSetting.hardwarePTType == rayQuery) {
			rayQueryPresentPrepare(cameraUniformDescriptorSetLayout);
		}
	}

	void present(VkDescriptorSet uniformDescriptorSet, uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) {
		if (ptSetting.UseHardware && ptSetting.hardwarePTType == rayQuery) {
			rayQueryPresent(uniformDescriptorSet, imageIndex, startSemaphore, fence);
		}
	}

	void clean() {
		depthMap.clean();
		vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);

		if (ptSetting.UseHardware) {
			if (ptSetting.hardwarePTType == rayQuery) {
				vkDestroyPipeline(logicalDevice, rayQueryPipeline, nullptr);
				vkDestroyPipelineLayout(logicalDevice, rayQueryPipelineLayout, nullptr);

				vkDestroyRenderPass(logicalDevice, rayQueryRenderPass, nullptr);

				vkDestroyDescriptorSetLayout(logicalDevice, rayQueryDescriptorSetLayout, nullptr);
			}

			ptUniformBuffer.clean();
			topASs->clean();
			bottomASs->clean();
		}
		for (int i = 0; i < framebuffers.size(); i++) {
			for (int j = 0; j < framebuffers[i].size(); j++) {
				vkDestroyFramebuffer(logicalDevice, framebuffers[i][j], nullptr);
			}
		}
		fzbCleanSemaphore(ptFinishedSemaphore);
	}
private:
	void createTopAS() {
		VkTransformMatrixKHR transform_matrix = {
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f
		};

		VkAccelerationStructureInstanceKHR asInstance{};
		asInstance.transform = transform_matrix;
		asInstance.instanceCustomIndex = 0;
		asInstance.mask = 0xFF;
		asInstance.instanceShaderBindingTableRecordOffset = 0;
		asInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
		asInstance.accelerationStructureReference = bottomASs->getDeviceAddress();

		FzbBuffer instanceBuffer(sizeof(VkAccelerationStructureInstanceKHR));
		fzbCreateBuffer(physicalDevice, logicalDevice,
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			instanceBuffer);
		fzbFillBuffer(logicalDevice, &asInstance, instanceBuffer);
		fzbGetBufferDeviceAddress(logicalDevice, instanceBuffer);

		topASs = std::make_unique<FzbAccelerationStructure>(physicalDevice, logicalDevice, commandPool, graphicsQueue, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
		topASs->addInstanceGeometry(instanceBuffer, 1);
		topASs->build();

	}

	template<typename T>
	void createBottomAS(std::vector<T> vertices, std::vector<uint32_t> indices, std::vector<VkTransformMatrixKHR> transforms) {

		//这里先假设只有一个模型好了
		fzbCreateASStorageBuffer(vertexBuffer, &vertices);
		fzbCreateASStorageBuffer(indexBuffer, &indices);
		fzbCreateASStorageBuffer(transformBuffer, &transforms);

		if (bottomASs == nullptr) {
			bottomASs = std::make_unique<FzbAccelerationStructure>(physicalDevice, logicalDevice, commandPool, graphicsQueue, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
			bottomASs->addTriangleGeometry(&vertexBuffer, &indexBuffer, &transforms, indexBuffer.size / 3, vertexBuffer.size - 1, sizeof(T), 0,
				VK_FORMAT_R32G32B32_SFLOAT, VK_INDEX_TYPE_UINT32, VK_GEOMETRY_OPAQUE_BIT_KHR);
		}
		//VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR适合静态场景，构建慢但生成的加速结构更好，存储效率高且是碰撞检测快。
		bottomASs->build(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR)

	}

	void createPTBuffer(std::vector<VkTransformMatrixKHR> sceneTransforms) {
		fzbCreateCommandBuffers(1);

		ptUniformBuffer = fzbComponentCreateUniformBuffers<PathTracingUnifom>();
		memcpy(ptUniformBuffer.mapped, &ptSetting.ptUniform, sizeof(PathTracingUnifom));
	}

//--------------------------------------------------------------RayQuery------------------------------------------------------------------
	void initRayQueryImage() {
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
	
	void createRayQueryDescriptorPool() {
		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1 });
		fzbComponentCreateDescriptorPool(bufferTypeAndNum);
	}

	void createRayQueryDescriptor() {
		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
		rayQueryDescriptorSetLayout = fzbComponentCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		rayQueryDescriptorSet = fzbComponentCreateDescriptorSet(rayQueryDescriptorSetLayout);

		std::array<VkWriteDescriptorSet, 2> ptDescriptorWrites{};
		VkDescriptorBufferInfo ptUniformBufferInfo{};
		ptUniformBufferInfo.buffer = ptUniformBuffer.buffer;
		ptUniformBufferInfo.offset = 0;
		ptUniformBufferInfo.range = sizeof(PathTracingUnifom);
		ptDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		ptDescriptorWrites[0].dstSet = rayQueryDescriptorSet;
		ptDescriptorWrites[0].dstBinding = 0;
		ptDescriptorWrites[0].dstArrayElement = 0;
		ptDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		ptDescriptorWrites[0].descriptorCount = 1;
		ptDescriptorWrites[0].pBufferInfo = &ptUniformBufferInfo;

		VkWriteDescriptorSetAccelerationStructureKHR  asBufferInfo{};
		asBufferInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
		asBufferInfo.accelerationStructureCount = 1;
		asBufferInfo.pAccelerationStructures = topASs->getAccelerationStructurePoint();
		ptDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		ptDescriptorWrites[1].dstSet = rayQueryDescriptorSet;
		ptDescriptorWrites[1].dstBinding = 1;
		ptDescriptorWrites[1].dstArrayElement = 0;
		ptDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		ptDescriptorWrites[1].descriptorCount = 1;
		ptDescriptorWrites[1].pNext = &asBufferInfo;

		vkUpdateDescriptorSets(logicalDevice, ptDescriptorWrites.size(), ptDescriptorWrites.data(), 0, nullptr);
	}

	void createRayQueryRenderPass() {
		VkAttachmentDescription colorAttachmentResolve{};
		colorAttachmentResolve.format = swapChainImageFormat;
		colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentResolveRef{};
		colorAttachmentResolveRef.attachment = 0;
		colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription depthMapAttachment{};
		depthMapAttachment.format = fzbFindDepthFormat(physicalDevice);
		depthMapAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthMapAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthMapAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depthMapAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthMapAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthMapAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthMapAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

		VkAttachmentReference depthMapAttachmentResolveRef{};
		depthMapAttachmentResolveRef.attachment = 1;
		depthMapAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		std::vector< VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };
			
		std::vector<VkSubpassDescription> subpasses;
		VkSubpassDescription presentSubpass{};
		presentSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		presentSubpass.colorAttachmentCount = 1;
		presentSubpass.pColorAttachments = &colorAttachmentResolveRef;
		presentSubpass.pDepthStencilAttachment = &depthMapAttachmentResolveRef;
		subpasses.push_back(presentSubpass);

		VkSubpassDependency dependency{};
		dependency.srcSubpass = 0;
		dependency.dstSubpass = 1;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		std::array< VkSubpassDependency, 1> dependencies = { dependency };

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = attachments.size();
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = subpasses.size();
		renderPassInfo.pSubpasses = subpasses.data();
		renderPassInfo.dependencyCount = dependencies.size();
		renderPassInfo.pDependencies = dependencies.data();

		if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &rayQueryRenderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void createRayQueryFrameBuffer() {
		std::vector<std::vector<VkImageView>> attachmentImageViews;
		attachmentImageViews.resize(swapChainImageViews.size());
		for (int i = 0; i < swapChainImageViews.size(); i++) {
			attachmentImageViews[i].push_back(swapChainImageViews[i]);
			attachmentImageViews[i].push_back(depthMap.imageView);
		}
		fzbCreateFramebuffer(swapChainImageViews.size(), swapChainExtent, 2, attachmentImageViews, rayQueryRenderPass);
	}

	void createRayQueryPipeline(VkDescriptorSetLayout cameraUniformDescriptorSetLayout) {
		std::map<VkShaderStageFlagBits, std::string> shaders;
		shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "core/RayTracing/PathTracing/shaders/rayQuery/spv/vert.spv" });
		shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "core/RayTracing/PathTracing/shaders/rayQuery/spv/frag.spv" });
		std::vector<VkPipelineShaderStageCreateInfo> shaderStages = fzbCreateShader(shaders);

		VkVertexInputBindingDescription inputBindingDescriptor = FzbVertex::getBindingDescription();
		auto inputAttributeDescription = FzbVertex::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = fzbCreateVertexInputCreateInfo(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = fzbCreateInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		VkPipelineRasterizationStateCreateInfo rasterizer = fzbCreateRasterizationStateCreateInfo(VK_CULL_MODE_BACK_BIT);

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

		std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { cameraUniformDescriptorSetLayout, rayQueryDescriptorSetLayout };
		rayQueryPipelineLayout = fzbCreatePipelineLayout(&descriptorSetLayouts);

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
		pipelineInfo.layout = rayQueryPipelineLayout;
		pipelineInfo.renderPass = rayQueryRenderPass;	//先建立连接，获得索引
		pipelineInfo.subpass = 0;	//对应renderpass的哪个子部分
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//可以直接使用现有pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &rayQueryPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		for (VkPipelineShaderStageCreateInfo shaderModule : shaderStages) {
			vkDestroyShaderModule(logicalDevice, shaderModule.module, nullptr);
		}
	}

	void createRayQuerySyncObject() {
		ptFinishedSemaphore = fzbCreateSemaphore();
	}

	void rayQueryPresentPrepare(VkDescriptorSetLayout cameraUniformDescriptorSetLayout) {
		initRayQueryImage();
		createRayQueryDescriptorPool();
		createRayQueryDescriptor();
		createRayQueryRenderPass();
		createRayQueryFrameBuffer();
		createRayQueryPipeline(cameraUniformDescriptorSetLayout);
		createRayQuerySyncObject();
	}

	void rayQueryPresent(VkDescriptorSet uniformDescriptorSet, uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence) {
		VkCommandBuffer commandBuffer = commandBuffers[0];
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
		renderPassBeginInfo.renderPass = rayQueryRenderPass;
		renderPassBeginInfo.framebuffer = framebuffers[0][imageIndex];
		renderPassBeginInfo.renderArea.offset = { 0, 0 };
		renderPassBeginInfo.renderArea.extent = swapChainExtent;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };
		renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassBeginInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkBuffer vertexBuffers[] = { vertexBuffer.buffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rayQueryPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rayQueryPipelineLayout, 0, 1, &uniformDescriptorSet, 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rayQueryPipelineLayout, 1, 1, &rayQueryDescriptorSet, 0, nullptr);

		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indexBuffer.data.size()), 1, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

		VkSemaphore waitSemaphores[] = { startSemaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &ptFinishedSemaphore.semaphore;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}
	}

//---------------------------------------------------------------辅助函数-------------------------------------------------------------------
	template<typename T>
	void fzbCreateASStorageBuffer(FzbBuffer& fzbBuffer, std::vector<T>* bufferData, bool UseExternal = false) {

		uint32_t bufferSize = bufferData->size() * sizeof(T);
		fzbBuffer.data = *bufferData;
		fzbBuffer.size = bufferSize;

		FzbBuffer stagingBuffer(bufferSize);
		fzbCreateBuffer(physicalDevice, logicalDevice, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer);
		fzbFillBuffer(logicalDevice, bufferData->data(), stagingBuffer);

		fzbCreateBuffer(physicalDevice, logicalDevice, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, fzbBuffer, UseExternal);

		copyBuffer(logicalDevice, commandPool, graphicsQueue, stagingBuffer.buffer, fzbBuffer.buffer, bufferSize);
		fzbGetBufferDeviceAddress(logicalDevice, fzbBuffer);

		vkDestroyBuffer(logicalDevice, stagingBuffer.buffer, nullptr);
		vkFreeMemory(logicalDevice, stagingBuffer.memory, nullptr);

	}

};

#endif