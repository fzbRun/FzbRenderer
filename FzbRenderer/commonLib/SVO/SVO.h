#pragma once

#include "../StructSet.h"
#include "../FzbSwapchain.h"
#include "../FzbImage.h"
#include "../FzbDescriptor.h"
#include "../FzbPipeline.h"
#include "../FzbSync.h"
#include "../Camera.h"

#include "../Component.h"
#include "./CUDA/createSVO.cuh"

#ifndef SVO_H	//Sparse voxel octree
#define SVO_H

struct FzbSVOSetting : public ComponentSetting {
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

class FzbSVO : public Component {

public:

	FzbSVOSetting svoSetting;
	SVOUniform svoUniform;
	MyImage voxelGridMap;
	VkDescriptorSetLayout voxelGridMapDescriptorSetLayout;
	VkDescriptorSet voxelGridMapDescriptorSet;
	std::vector<Vertex_onlyPos> vertices;
	std::vector<uint32_t> indices;
	std::vector<Vertex_onlyPos> cubeVertices;
	std::vector<uint32_t> cubeIndices;

	VkRenderPass voxelGridMapRenderPass;
	VkPipeline voxelGridMapPipeline;
	VkPipelineLayout voxelGridMapPipelineLayout;

	VkRenderPass presentRenderPass;
	VkPipeline presentPipeline;
	VkPipelineLayout presentPipelineLayout;

	VkPipeline presentWireframePipeline;
	VkPipelineLayout presentWireframePipelineLayout;

	MyImage depthMap;

	std::unique_ptr<SVOCuda> svoCuda;
	vector<FzbSVONode> nodePool;
	vector<FzbVoxelValue> voxelValueBuffer;

	VkDescriptorSetLayout svoDescriptorSetLayout;
	VkDescriptorSet svoDescriptorSet;

	static void addExtensions(FzbSVOSetting svoSetting, std::vector<const char*>& instanceExtensions, std::vector<const char*>& deviceExtensions, VkPhysicalDeviceFeatures& deviceFeatures) {

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

	FzbSVO(std::unique_ptr<FzbDevice>& fzbDevice, std::unique_ptr<FzbSwapchain>& fzbSwapchain, VkCommandPool& commandPool, MyModel* model, FzbSVOSetting* setting) {
		
		this->physicalDevice = fzbDevice->physicalDevice;
		this->logicalDevice = fzbDevice->logicalDevice;
		this->graphicsQueue = fzbDevice->graphicsQueue;
		this->swapChainExtent = fzbSwapchain->swapChainExtent;
		this->model = model;
		this->fzbImage = std::make_unique<FzbImage>(fzbDevice);
		this->fzbBuffer = std::make_unique<FzbBuffer>(fzbDevice, commandPool);
		this->fzbDescriptor = std::make_unique<FzbDescriptor>(fzbDevice);
		this->fzbPipeline = std::make_unique<FzbPipeline>(fzbDevice);
		this->fzbSync = std::make_unique<FzbSync>(fzbDevice);
		this->svoSetting = *setting;

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

	void presentPrepare(VkFormat swapChainImageFormat, vector<VkImageView>& swapChainImageViews, VkDescriptorSetLayout uniformDescriptorSetLayout) {
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
		fzbSync->createSemaphore(false);
	}

	//用于测试体素化结果
	void present(VkDescriptorSet uniformDescriptorSet, uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) {

		VkCommandBuffer commandBuffer = fzbBuffer->commandBuffers[1];
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
		renderPassBeginInfo.framebuffer = fzbBuffer->framebuffers[1][imageIndex];
		renderPassBeginInfo.renderArea.offset = { 0, 0 };
		renderPassBeginInfo.renderArea.extent = swapChainExtent;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };
		renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassBeginInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		if (svoSetting.UseBlock && svoSetting.UseSVO_OnlyVoxelGridMap) {
			VkBuffer cube_vertexBuffers[] = { fzbBuffer->storageBuffers[2] };
			VkDeviceSize cube_offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, cube_vertexBuffers, cube_offsets);
			vkCmdBindIndexBuffer(commandBuffer, fzbBuffer->storageBuffers[3], 0, VK_INDEX_TYPE_UINT32);
		}
		else {
			VkBuffer vertexBuffers[] = { fzbBuffer->storageBuffers[0] };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(commandBuffer, fzbBuffer->storageBuffers[1], 0, VK_INDEX_TYPE_UINT32);
		}
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipelineLayout, 0, 1, &uniformDescriptorSet, 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipelineLayout, 1, 1, &voxelGridMapDescriptorSet, 0, nullptr);
		if (svoSetting.UseBlock && svoSetting.UseSVO_OnlyVoxelGridMap) {
			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(this->cubeIndices.size()), std::pow(svoSetting.voxelNum, 3), 0, 0, 0);
		}
		else {
			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(this->indices.size()), 1, 0, 0, 0);
		}

		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);

			VkBuffer cube_vertexBuffers[] = { fzbBuffer->storageBuffers[4] };
			VkDeviceSize cube_offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, cube_vertexBuffers, cube_offsets);
			vkCmdBindIndexBuffer(commandBuffer, fzbBuffer->storageBuffers[5], 0, VK_INDEX_TYPE_UINT32);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentWireframePipeline);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentWireframePipelineLayout, 0, 1, &uniformDescriptorSet, 0, nullptr);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentWireframePipelineLayout, 1, 1, &svoDescriptorSet, 0, nullptr);
			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(this->cubeIndices.size()), this->svoCuda->nodeArrayNum * 8, 0, 0, 0);
		}

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

		VkSemaphore waitSemaphores[] = { startSemaphore,  fzbSync->fzbSemaphores[svoSetting.UseSVO_OnlyVoxelGridMap ? 0 : 1].semaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &fzbSync->fzbSemaphores[svoSetting.UseSVO_OnlyVoxelGridMap ? 1 : 2].semaphore;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

	}

	void clean() {

		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			svoCuda->clean();
		}
		fzbImage->cleanImage(voxelGridMap);
		fzbImage->cleanImage(depthMap);

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

		vkDestroyDescriptorPool(logicalDevice, fzbDescriptor->descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, voxelGridMapDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, svoDescriptorSetLayout, nullptr);

		fzbSync->cleanFzbSync();

		fzbBuffer->cleanupBuffers();

	}

private:

	template<typename T>
	void optimizeModel(MyModel* myModel, std::vector<T>& vertices, std::vector<uint32_t>& indices) {
		uint32_t indexOffset = 0;
		for (uint32_t meshIndex = 0; meshIndex < myModel->meshs.size(); meshIndex++) {

			Mesh mesh = myModel->meshs[meshIndex];
			//this->materials.push_back(my_model->meshs[i].material);
			vertices.insert(vertices.end(), mesh.vertices.begin(), mesh.vertices.end());

			//因为assimp是按一个mesh一个mesh的存，所以每个indices都是相对一个mesh的，当我们将每个mesh的顶点存到一起时，indices就会出错，我们需要增加索引
			for (uint32_t index = 0; index < mesh.indices.size(); index++) {
				mesh.indices[index] += indexOffset;
			}
			//meshIndexInIndices.push_back(this->indices.size());
			indexOffset += mesh.vertices.size();
			indices.insert(indices.end(), mesh.indices.begin(), mesh.indices.end());
		}

		std::unordered_map<T, uint32_t> uniqueVerticesMap{};
		std::vector<T> uniqueVertices;
		std::vector<uint32_t> uniqueIndices;
		for (uint32_t j = 0; j < indices.size(); j++) {
			T vertex = std::is_same_v<T, Vertex> ? vertices[indices[j]] : T(vertices[indices[j]]);
			if (uniqueVerticesMap.count(vertex) == 0) {
				uniqueVerticesMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
				uniqueVertices.push_back(vertex);
			}
			uniqueIndices.push_back(uniqueVerticesMap[vertex]);
		}
		vertices = uniqueVertices;
		indices = uniqueIndices;

	}

	void makeAABB(MyModel* myModel) {

		for (int i = 0; i < myModel->meshs.size(); i++) {

			//left right xyz
			AABBBox AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
			for (int j = 0; j < myModel->meshs[i].indices.size(); j++) {
				glm::vec3 worldPos = myModel->meshs[i].vertices[myModel->meshs[i].indices[j]].pos;
				AABB.leftX = worldPos.x < AABB.leftX ? worldPos.x : AABB.leftX;
				AABB.rightX = worldPos.x > AABB.rightX ? worldPos.x : AABB.rightX;
				AABB.leftY = worldPos.y < AABB.leftY ? worldPos.y : AABB.leftY;
				AABB.rightY = worldPos.y > AABB.rightY ? worldPos.y : AABB.rightY;
				AABB.leftZ = worldPos.z < AABB.leftZ ? worldPos.z : AABB.leftZ;
				AABB.rightZ = worldPos.z > AABB.rightZ ? worldPos.z : AABB.rightZ;
			}
			//对于面，我们给个0.2的宽度
			if (AABB.leftX == AABB.rightX) {
				AABB.leftX = AABB.leftX - 0.01;
				AABB.rightX = AABB.rightX + 0.01;
			}
			if (AABB.leftY == AABB.rightY) {
				AABB.leftY = AABB.leftY - 0.01;
				AABB.rightY = AABB.rightY + 0.01;
			}
			if (AABB.leftZ == AABB.rightZ) {
				AABB.leftZ = AABB.leftZ - 0.01;
				AABB.rightZ = AABB.rightZ + 0.01;
			}
			myModel->meshs[i].AABB = AABB;

		}

		AABBBox AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
		for (int i = 0; i < myModel->meshs.size(); i++) {
			Mesh mesh = myModel->meshs[i];
			AABB.leftX = mesh.AABB.leftX < AABB.leftX ? mesh.AABB.leftX : AABB.leftX;
			AABB.rightX = mesh.AABB.rightX > AABB.rightX ? mesh.AABB.rightX : AABB.rightX;
			AABB.leftY = mesh.AABB.leftY < AABB.leftY ? mesh.AABB.leftY : AABB.leftY;
			AABB.rightY = mesh.AABB.rightY > AABB.rightY ? mesh.AABB.rightY : AABB.rightY;
			AABB.leftZ = mesh.AABB.leftZ < AABB.leftZ ? mesh.AABB.leftZ : AABB.leftZ;
			AABB.rightZ = mesh.AABB.rightZ > AABB.rightZ ? mesh.AABB.rightZ : AABB.rightZ;
		}
		myModel->AABB = AABB;

	}

	void createVoxelGridMapBuffer() {

		fzbBuffer->createCommandBuffers(2);

		optimizeModel<Vertex_onlyPos>(model, vertices, indices);
		makeAABB(model);

		fzbBuffer->createStorageBuffer<Vertex_onlyPos>(vertices.size() * sizeof(Vertex_onlyPos), &vertices);
		fzbBuffer->createStorageBuffer<uint32_t>(indices.size() * sizeof(uint32_t), &indices);

		fzbBuffer->createUniformBuffers(sizeof(SVOUniform), true, 1);
		svoUniform.modelMatrix = glm::mat4(1.0f);

		float distanceX = model->AABB.rightX - model->AABB.leftX;
		float distanceY = model->AABB.rightY - model->AABB.leftY;
		float distanceZ = model->AABB.rightZ - model->AABB.leftZ;
		//想让顶点通过swizzle变换后得到正确的结果，必须保证投影矩阵是立方体的，这样xyz通过1减后才能是对应的
		//但是其实不需要VP，shader中其实没啥用
		float distance = glm::max(distanceX, glm::max(distanceY, distanceZ));
		float centerX = (model->AABB.rightX + model->AABB.leftX) * 0.5f;
		float centerY = (model->AABB.rightY + model->AABB.leftY) * 0.5f;
		float centerZ = (model->AABB.rightZ + model->AABB.leftZ) * 0.5f;
		//前面
		glm::vec3 viewPoint = glm::vec3(centerX, centerY, model->AABB.rightZ + 0.2f);	//世界坐标右手螺旋，即+z朝后
		glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceX, 0.51f * distanceX, -0.51f * distanceY, 0.51f * distanceY, 0.1f, distanceZ + 0.5f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[0] = orthoMatrix * viewMatrix;

		//左边
		viewPoint = glm::vec3(model->AABB.leftX - 0.2f, centerY, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceZ, 0.51f * distanceZ, -0.51f * distanceY, 0.51f * distanceY, 0.1f, distanceX + 0.5f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[1] = orthoMatrix * viewMatrix;

		//下面
		viewPoint = glm::vec3(centerX, model->AABB.leftY - 0.2f, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceX, 0.51f * distanceX, -0.51f * distanceZ, 0.51f * distanceZ, 0.1f, distanceY + 0.5f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[2] = orthoMatrix * viewMatrix;
		svoUniform.voxelSize_Num = glm::vec4(distance / svoSetting.voxelNum, svoSetting.voxelNum, distance, 0.0f);
		svoUniform.voxelStartPos = glm::vec4(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f, 0.0f);

		memcpy(fzbBuffer->uniformBuffersMappedsStatic[0], &svoUniform, sizeof(SVOUniform));

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
		fzbImage->createMyImage(voxelGridMap, fzbBuffer, !svoSetting.UseSVO_OnlyVoxelGridMap);
	}

	void createDescriptorPool() {
		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 });
		}
		fzbDescriptor->createDescriptorPool(bufferTypeAndNum);
	}

	void createVoxelGridMapDescriptor() {

		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE };
		std::vector<VkShaderStageFlagBits> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_FRAGMENT_BIT };
		voxelGridMapDescriptorSetLayout = fzbDescriptor->createDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		voxelGridMapDescriptorSet = fzbDescriptor->createDescriptorSet(voxelGridMapDescriptorSetLayout);

		std::array<VkWriteDescriptorSet, 2> voxelGridMapDescriptorWrites{};
		VkDescriptorBufferInfo voxelGridMapUniformBufferInfo{};
		voxelGridMapUniformBufferInfo.buffer = fzbBuffer->uniformBuffersStatic[0];
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

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

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
		fzbBuffer->createFramebuffer(1, extent, 0, attachmentImageViews, voxelGridMapRenderPass);
	}

	void createVoxelGridMapPipeline() {
		std::map<VkShaderStageFlagBits, std::string> shaders;
		if (svoSetting.UseSwizzle) {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "commonLib/SVO/shaders/useSwizzle/spv/voxelVert.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "commonLib/SVO/shaders/useSwizzle/spv/voxelFrag.spv" });
		}
		else {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "commonLib/SVO/shaders/unuseSwizzle/spv/voxelVert.spv" });
			shaders.insert({ VK_SHADER_STAGE_GEOMETRY_BIT, "commonLib/SVO/shaders/unuseSwizzle/spv/voxelGemo.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "commonLib/SVO/shaders/unuseSwizzle/spv/voxelFrag.spv" });
		}
		std::vector<VkPipelineShaderStageCreateInfo> shaderStages = fzbPipeline->createShader(shaders);

		VkVertexInputBindingDescription inputBindingDescriptor = Vertex_onlyPos::getBindingDescription();
		auto inputAttributeDescription = Vertex_onlyPos::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = fzbPipeline->createVertexInputCreateInfo<Vertex_onlyPos>(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = fzbPipeline->createInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeState{};
		if (svoSetting.UseConservativeRasterization) {
			conservativeState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
			conservativeState.pNext = NULL;
			conservativeState.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
			conservativeState.extraPrimitiveOverestimationSize = 0.5f; // 根据需要设置
			rasterizer = fzbPipeline->createRasterizationStateCreateInfo(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE, &conservativeState);
		}
		else {
			rasterizer = fzbPipeline->createRasterizationStateCreateInfo(VK_CULL_MODE_NONE);
		}

		VkPipelineMultisampleStateCreateInfo multisampling = fzbPipeline->createMultisampleStateCreateInfo();
		VkPipelineColorBlendAttachmentState colorBlendAttachment = fzbPipeline->createColorBlendAttachmentState();
		std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments = { colorBlendAttachment };
		VkPipelineColorBlendStateCreateInfo colorBlending = fzbPipeline->createColorBlendStateCreateInfo(colorBlendAttachments);
		VkPipelineDepthStencilStateCreateInfo depthStencil = fzbPipeline->createDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE);

		VkPipelineViewportStateCreateInfo viewportState = fzbPipeline->createViewStateCreateInfo();
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

			std::array< VkViewport, 3> viewports = {};
			std::array< VkRect2D, 3> scissors = {};
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
			std::array<VkViewportSwizzleNV, 3> swizzles = {};
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

			VkPipelineViewportSwizzleStateCreateInfoNV viewportSwizzleInfo{};
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
		voxelGridMapPipelineLayout = fzbPipeline->createPipelineLayout(&descriptorSetLayouts);

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
			fzbSync->createSemaphore(false);	//当vgm创建完成后唤醒
		}
		else {
			fzbSync->createSemaphore(true);		//当vgm创建完成后唤醒
			fzbSync->createSemaphore(true);		//当svo创建完成后唤醒
		}

		fzbSync->createFence();
	}

	void createVoxelGridMap() {

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		vkResetFences(logicalDevice, 1, &fzbSync->fzbFences[0]);
		VkCommandBuffer commandBuffer = fzbBuffer->commandBuffers[0];
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
		fzbImage->clearTexture(commandBuffer, voxelGridMap, voxel_clearColor, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = voxelGridMapRenderPass;
		renderPassInfo.framebuffer = fzbBuffer->framebuffers[0][0];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkBuffer vertexBuffers[] = { fzbBuffer->storageBuffers[0] };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, fzbBuffer->storageBuffers[1], 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelGridMapPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelGridMapPipelineLayout, 0, 1, &voxelGridMapDescriptorSet, 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(this->indices.size()), 1, 0, 0, 0);

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
		submitInfo.pSignalSemaphores =  &fzbSync->fzbSemaphores[0].semaphore;

		//执行完后解开fence
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

	}

	void createSVOCuda() {

		svoCuda->createSVOCuda(physicalDevice, voxelGridMap, fzbSync->fzbSemaphores[0].handle, fzbSync->fzbSemaphores[1].handle, svoUniform.voxelStartPos, svoUniform.voxelSize_Num.z);

		//由于不能从cuda中直接导出数组的handle，因此我们需要先创建一个buffer，然后在cuda中将数据copy进去
		nodePool.resize(svoCuda->nodeArrayNum * 8);
		this->fzbBuffer->createStorageBuffer<FzbSVONode>(svoCuda->nodeArrayNum * 8 * sizeof(FzbSVONode), &nodePool, true);
		voxelValueBuffer.resize(svoCuda->voxelNum);
		this->fzbBuffer->createStorageBuffer<FzbVoxelValue>(svoCuda->voxelNum * sizeof(FzbVoxelValue), &voxelValueBuffer, true);

		svoCuda->getSVOCuda(physicalDevice, fzbBuffer->storageBufferHandles[0], fzbBuffer->storageBufferHandles[1]);

	}

	void createSVODescriptor() {
		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		std::vector<VkShaderStageFlagBits> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_ALL };
		svoDescriptorSetLayout = fzbDescriptor->createDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		svoDescriptorSet = fzbDescriptor->createDescriptorSet(svoDescriptorSetLayout);

		std::array<VkWriteDescriptorSet, 2> svoDescriptorWrites{};
		VkDescriptorBufferInfo nodePoolBufferInfo{};
		nodePoolBufferInfo.buffer = fzbBuffer->storageBuffers[2];
		nodePoolBufferInfo.offset = 0;
		nodePoolBufferInfo.range = sizeof(FzbSVONode) * this->svoCuda->nodeArrayNum * 8;
		svoDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		svoDescriptorWrites[0].dstSet = svoDescriptorSet;
		svoDescriptorWrites[0].dstBinding = 0;
		svoDescriptorWrites[0].dstArrayElement = 0;
		svoDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		svoDescriptorWrites[0].descriptorCount = 1;
		svoDescriptorWrites[0].pBufferInfo = &nodePoolBufferInfo;

		VkDescriptorBufferInfo voxelValueBufferInfo{};
		voxelValueBufferInfo.buffer = fzbBuffer->storageBuffers[3];
		voxelValueBufferInfo.offset = 0;
		voxelValueBufferInfo.range = sizeof(FzbVoxelValue) * this->svoCuda->voxelNum;
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
				float distanceX = model->AABB.rightX - model->AABB.leftX;
				float distanceY = model->AABB.rightY - model->AABB.leftY;
				float distanceZ = model->AABB.rightZ - model->AABB.leftZ;
				float distance = glm::max(distanceX, glm::max(distanceY, distanceZ));
				float voxelSize = distance / svoSetting.voxelNum;

				float centerX = (model->AABB.rightX + model->AABB.leftX) * 0.5f;
				float centerY = (model->AABB.rightY + model->AABB.leftY) * 0.5f;
				float centerZ = (model->AABB.rightZ + model->AABB.leftZ) * 0.5f;

				this->cubeVertices.resize(8);
				for (int i = 0; i < 8; i++) {
					this->cubeVertices[i].pos = cubeVertexOffset[i] * voxelSize + glm::vec3(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f);
				}
				this->cubeIndices = {
					1, 0, 3, 1, 3, 2,
					4, 5, 6, 4, 6, 7,
					5, 1, 2, 5, 2, 6,
					0, 4, 7, 0, 7, 3,
					7, 6, 2, 7, 2, 3,
					0, 1, 5, 0, 5, 4
				};
				fzbBuffer->createStorageBuffer<Vertex_onlyPos>(cubeVertices.size() * sizeof(Vertex_onlyPos), &cubeVertices);
				fzbBuffer->createStorageBuffer<uint32_t>(cubeIndices.size() * sizeof(uint32_t), &cubeIndices);
			}
		}
		else {
			this->cubeVertices.resize(8);
			for (int i = 0; i < 8; i++) {
				this->cubeVertices[i].pos = cubeVertexOffset[i];
			}
			this->cubeIndices = {
				0, 1, 1, 2, 2, 3, 3, 0,
				4, 5, 5, 6, 6, 7, 7, 4,
				0, 4, 1, 5, 2, 6, 3, 7
			};
			fzbBuffer->createStorageBuffer<Vertex_onlyPos>(cubeVertices.size() * sizeof(Vertex_onlyPos), &cubeVertices);
			fzbBuffer->createStorageBuffer<uint32_t>(cubeIndices.size() * sizeof(uint32_t), &cubeIndices);
		}
	}

	void initDepthMap() {
		depthMap = {};
		depthMap.width = swapChainExtent.width;
		depthMap.height = swapChainExtent.height;
		depthMap.type = VK_IMAGE_TYPE_2D;
		depthMap.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthMap.format = fzbImage->findDepthFormat(physicalDevice);
		depthMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		depthMap.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
		fzbImage->createMyImage(depthMap, fzbBuffer, false);
	}

	void createPresentRenderPass(VkFormat swapChainImageFormat) {
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

		std::vector< VkAttachmentDescription> attachments = { colorAttachmentResolve };

		VkAttachmentDescription depthMapAttachment{};
		depthMapAttachment.format = fzbImage->findDepthFormat(physicalDevice);
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
		if(!svoSetting.UseSVO_OnlyVoxelGridMap)
			attachments.push_back(depthMapAttachment);

		std::vector<VkSubpassDescription> subpasses;
		VkSubpassDescription presentSubpass{};
		presentSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		presentSubpass.colorAttachmentCount = 1;
		presentSubpass.pColorAttachments = &colorAttachmentResolveRef;
		if (!svoSetting.UseSVO_OnlyVoxelGridMap)
			presentSubpass.pDepthStencilAttachment = &depthMapAttachmentResolveRef;
		subpasses.push_back(presentSubpass);

		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			VkSubpassDescription presentWireframeSubpass{};
			presentWireframeSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			presentWireframeSubpass.colorAttachmentCount = 1;
			presentWireframeSubpass.pColorAttachments = &colorAttachmentResolveRef;
			presentWireframeSubpass.pDepthStencilAttachment = &depthMapAttachmentResolveRef;
			subpasses.push_back(presentWireframeSubpass);
		}

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			dependency.srcSubpass = 0;
			dependency.dstSubpass = 1;
			dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
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

	void createPresentFrameBuffer(vector<VkImageView>& swapChainImageViews) {
		vector<vector<VkImageView>> attachmentImageViews;
		attachmentImageViews.resize(swapChainImageViews.size());
		for (int i = 0; i < swapChainImageViews.size(); i++) {
			attachmentImageViews[i].push_back(swapChainImageViews[i]);
			if (!svoSetting.UseSVO_OnlyVoxelGridMap)
				attachmentImageViews[i].push_back(depthMap.imageView);
		}
		fzbBuffer->createFramebuffer(swapChainImageViews.size(), swapChainExtent, svoSetting.UseSVO_OnlyVoxelGridMap ? 1 : 2, attachmentImageViews, presentRenderPass);
	}

	void createVGMPresentPipeline(VkDescriptorSetLayout uniformDescriptorSetLayout) {
		map<VkShaderStageFlagBits, string> shaders;
		if (svoSetting.UseBlock) {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "commonLib/SVO/shaders/present_VGM/spv/presentVert_Block.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "commonLib/SVO/shaders/present_VGM/spv/presentFrag_Block.spv" });
		}
		else {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "commonLib/SVO/shaders/present/spv/presentVert.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "commonLib/SVO/shaders/present/spv/presentFrag.spv" });
		}
		vector<VkPipelineShaderStageCreateInfo> shaderStages = fzbPipeline->createShader(shaders);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		VkVertexInputBindingDescription inputBindingDescriptor = Vertex_onlyPos::getBindingDescription();
		auto inputAttributeDescription = Vertex_onlyPos::getAttributeDescriptions();
		vertexInputInfo = fzbPipeline->createVertexInputCreateInfo<Vertex_onlyPos>(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = fzbPipeline->createInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		VkPipelineRasterizationStateCreateInfo rasterizer = fzbPipeline->createRasterizationStateCreateInfo(VK_CULL_MODE_NONE);

		VkPipelineMultisampleStateCreateInfo multisampling = fzbPipeline->createMultisampleStateCreateInfo();
		VkPipelineColorBlendAttachmentState colorBlendAttachment = fzbPipeline->createColorBlendAttachmentState();
		vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments = { colorBlendAttachment };
		VkPipelineColorBlendStateCreateInfo colorBlending = fzbPipeline->createColorBlendStateCreateInfo(colorBlendAttachments);
		VkPipelineDepthStencilStateCreateInfo depthStencil = fzbPipeline->createDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE);

		VkPipelineViewportStateCreateInfo viewportState = fzbPipeline->createViewStateCreateInfo();
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
		presentPipelineLayout = fzbPipeline->createPipelineLayout(&presentDescriptorSetLayouts);

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
		map<VkShaderStageFlagBits, string> shaders;
		shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "commonLib/SVO/shaders/present/spv/presentVert.spv" });
		shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "commonLib/SVO/shaders/present/spv/presentFrag.spv" });
		vector<VkPipelineShaderStageCreateInfo> shaderStages = fzbPipeline->createShader(shaders);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		VkVertexInputBindingDescription inputBindingDescriptor = Vertex_onlyPos::getBindingDescription();
		auto inputAttributeDescription = Vertex_onlyPos::getAttributeDescriptions();
		vertexInputInfo = fzbPipeline->createVertexInputCreateInfo<Vertex_onlyPos>(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = fzbPipeline->createInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		VkPipelineRasterizationStateCreateInfo rasterizer = fzbPipeline->createRasterizationStateCreateInfo(VK_CULL_MODE_NONE);

		VkPipelineMultisampleStateCreateInfo multisampling = fzbPipeline->createMultisampleStateCreateInfo();
		VkPipelineColorBlendAttachmentState colorBlendAttachment = fzbPipeline->createColorBlendAttachmentState();
		vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments = { colorBlendAttachment };
		VkPipelineColorBlendStateCreateInfo colorBlending = fzbPipeline->createColorBlendStateCreateInfo(colorBlendAttachments);
		VkPipelineDepthStencilStateCreateInfo depthStencil = fzbPipeline->createDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE);

		VkPipelineViewportStateCreateInfo viewportState = fzbPipeline->createViewStateCreateInfo();
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
		presentPipelineLayout = fzbPipeline->createPipelineLayout(&presentDescriptorSetLayouts);

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

		shaders = map<VkShaderStageFlagBits, string>();
		shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "commonLib/SVO/shaders/present_SVO/spv/vert.spv" });
		shaders.insert({ VK_SHADER_STAGE_GEOMETRY_BIT, "commonLib/SVO/shaders/present_SVO/spv/gemo.spv" });
		shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "commonLib/SVO/shaders/present_SVO/spv/frag.spv" });
		shaderStages = fzbPipeline->createShader(shaders);

		inputAssemblyInfo = fzbPipeline->createInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);

		rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
		rasterizer.lineWidth = 1.0f;

		presentDescriptorSetLayouts = { uniformDescriptorSetLayout, svoDescriptorSetLayout };
		presentWireframePipelineLayout = fzbPipeline->createPipelineLayout(&presentDescriptorSetLayouts);

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
