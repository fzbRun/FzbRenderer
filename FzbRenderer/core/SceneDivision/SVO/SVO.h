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
	glm::mat4 VP[3];
	glm::vec4 voxelSize_Num;
	glm::vec4 voxelStartPos;
};

struct FzbSVO : public FzbComponent {

public:

	FzbSVOSetting svoSetting;
	FzbImage voxelGridMap;
	VkDescriptorSetLayout voxelGridMapDescriptorSetLayout = nullptr;
	VkDescriptorSet voxelGridMapDescriptorSet;

	FzbScene* mainComponentScene;
	FzbScene componentScene;
	FzbBuffer svoUniformBuffer;

	FzbRenderPass voxelGridMapRenderPass;
	FzbRenderPass presentRenderPass;

	FzbImage depthMap;

	std::unique_ptr<SVOCuda> svoCuda;
	FzbBuffer nodePool;
	FzbBuffer voxelValueBuffer;

	VkDescriptorSetLayout svoDescriptorSetLayout = nullptr;
	VkDescriptorSet svoDescriptorSet;

	FzbSemaphore vgmSemaphore;
	FzbSemaphore svoCudaSemaphore;
	FzbSemaphore presentSemaphore;
	VkFence fence;

	FzbShader vgmShader;
	FzbMaterial vgmMaterial;
	FzbShader presentShader;
	FzbMaterial presentMaterial;

	FzbSVO() {}

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

	FzbVertexFormat getComponentVertexFormat() {	//这个组件中的shader所需要用到的顶点属性
		return FzbVertexFormat();
	}

	void init(FzbMainComponent* renderer,  FzbSVOSetting setting, FzbScene* scene, std::vector<FzbRenderPass*>& renderPasses) {
		initComponent(renderer);
		mainComponentScene = scene;
		this->svoSetting = setting;

		createVoxelGridMap();
		//if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
		//	this->svoCuda = std::make_unique<SVOCuda>();
		//	createSVOCuda();
		//	createSVODescriptor();
		//}
		presentPrepare(renderPasses);
	}

	//用于测试体素化结果
	VkSemaphore render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) {
		mainComponentScene->updateCameraBuffer();

		VkCommandBuffer commandBuffer = commandBuffers[1];
		vkResetCommandBuffer(commandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		presentRenderPass.render(commandBuffer, imageIndex, { mainComponentScene }, { {mainComponentScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet} });

		std::vector<VkSemaphore> waitSemaphores = { startSemaphore, vgmSemaphore.semaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;	// waitSemaphores.size();
		submitInfo.pWaitSemaphores = waitSemaphores.data();
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &presentSemaphore.semaphore;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		return presentSemaphore.semaphore;
	}

	void clean() {

		componentScene.clean();
		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			svoCuda->clean();
		}
		voxelGridMap.clean();
		depthMap.clean();

		voxelGridMapRenderPass.clean();
		presentRenderPass.clean();


		vgmShader.clean();
		vgmMaterial.clean();
		presentShader.clean();
		presentMaterial.clean();

		svoUniformBuffer.clean();
		nodePool.clean();
		voxelValueBuffer.clean();

		vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
		if(voxelGridMapDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, voxelGridMapDescriptorSetLayout, nullptr);
		if(svoDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, svoDescriptorSetLayout, nullptr);

		vgmSemaphore.clean(logicalDevice);
		svoCudaSemaphore.clean(logicalDevice);
		presentSemaphore.clean(logicalDevice);
		fzbCleanFence(fence);
	}

private:
	void createVoxelGridMap() {

		createVoxelGridMapBuffer();
		initVoxelGridMap();
		createDescriptorPool();
		createVoxelGridMapDescriptor();
		createVoxelGridMapSyncObjects();
		createVGMRenderPass();

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

		voxelGridMapRenderPass.render(commandBuffer, 0, { mainComponentScene }, { {voxelGridMapDescriptorSet} });

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &vgmSemaphore.semaphore;

		//执行完后解开fence
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

	}
	void createVoxelGridMapBuffer() {
		fzbCreateCommandBuffers(2);

		svoUniformBuffer = fzbComponentCreateUniformBuffers<SVOUniform>();
		SVOUniform svoUniform;
		
		if (mainComponentScene->AABB.isEmpty()) mainComponentScene->createAABB();
		FzbAABBBox mianSceneAABB = mainComponentScene->AABB;
		float distanceX = mianSceneAABB.rightX - mianSceneAABB.leftX;
		float distanceY = mianSceneAABB.rightY - mianSceneAABB.leftY;
		float distanceZ = mianSceneAABB.rightZ - mianSceneAABB.leftZ;
		//想让顶点通过swizzle变换后得到正确的结果，必须保证投影矩阵是立方体的，这样xyz通过1减后才能是对应的
		//但是其实不需要VP，shader中其实没啥用
		float distance = glm::max(distanceX, glm::max(distanceY, distanceZ));
		float centerX = (mianSceneAABB.rightX + mianSceneAABB.leftX) * 0.5f;
		float centerY = (mianSceneAABB.rightY + mianSceneAABB.leftY) * 0.5f;
		float centerZ = (mianSceneAABB.rightZ + mianSceneAABB.leftZ) * 0.5f;
		//前面
		glm::vec3 viewPoint = glm::vec3(centerX, centerY, mianSceneAABB.rightZ + 0.2f);	//世界坐标右手螺旋，即+z朝后
		glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceX, 0.51f * distanceX, -0.51f * distanceY, 0.51f * distanceY, 0.1f, distanceZ + 0.5f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[0] = orthoMatrix * viewMatrix;

		//左边
		viewPoint = glm::vec3(mianSceneAABB.leftX - 0.2f, centerY, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceZ, 0.51f * distanceZ, -0.51f * distanceY, 0.51f * distanceY, 0.1f, distanceX + 0.5f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[1] = orthoMatrix * viewMatrix;

		//下面
		viewPoint = glm::vec3(centerX, mianSceneAABB.leftY - 0.2f, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceX, 0.51f * distanceX, -0.51f * distanceZ, 0.51f * distanceZ, 0.1f, distanceY + 0.5f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[2] = orthoMatrix * viewMatrix;
		svoUniform.voxelSize_Num = glm::vec4(distance / svoSetting.voxelNum, svoSetting.voxelNum, distance, 0.0f);
		svoUniform.voxelStartPos = glm::vec4(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f, 0.0f);

		memcpy(svoUniformBuffer.mapped, &svoUniform, sizeof(SVOUniform));

	}
	void initVoxelGridMap() {
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
	void createVoxelGridMapSyncObjects() {	//这里应该返回一个信号量，然后阻塞主线程，知道渲染完成，才能唤醒
		if (svoSetting.UseSVO_OnlyVoxelGridMap) {
			vgmSemaphore = FzbSemaphore(logicalDevice, false);	//当vgm创建完成后唤醒
		}
		else {
			vgmSemaphore = FzbSemaphore(logicalDevice, true);		//当vgm创建完成后唤醒
			svoCudaSemaphore = FzbSemaphore(logicalDevice, true);		//当svo创建完成后唤醒
		}
		fence = fzbCreateFence();
	}
	void createVGMRenderPass() {
		vgmMaterial = FzbMaterial(logicalDevice);
		if (!svoSetting.UseSwizzle) vgmShader = FzbShader(logicalDevice, getRootPath() + "/core/SceneDivision/SVO/shaders/unuseSwizzle");
		else vgmShader = FzbShader(logicalDevice, getRootPath() + "/core/SceneDivision/SVO/shaders/useSwizzle");
		vgmShader.createShaderVariant(&vgmMaterial, getComponentVertexFormat());
		for (int i = 0; i < mainComponentScene->sceneMeshSet.size(); i++) {
			vgmShader.shaderVariants[0].meshBatch.meshes.push_back(&mainComponentScene->sceneMeshSet[i]);
		}
		vgmShader.shaderVariants[0].meshBatch.useSameMaterial = true;
		vgmShader.shaderVariants[0].meshBatch.materials.push_back(&vgmMaterial);

		VkSubpassDescription voxelGridMapSubpass{};
		voxelGridMapSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

		VkSubpassDependency dependency = fzbCreateSubpassDependency();

		FzbRenderPassSetting setting = { false, 1, swapChainExtent, 1, false };
		voxelGridMapRenderPass = FzbRenderPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, setting);
		voxelGridMapRenderPass.createRenderPass(nullptr, { voxelGridMapSubpass }, { dependency });
		voxelGridMapRenderPass.createFramebuffers();

		/*
		FzbPipelineCreateInfo pipelineCreateInfo(voxelGridMapRenderPass.renderPass, voxelGridMapRenderPass.setting.extent);
		pipelineCreateInfo.cullMode = VK_CULL_MODE_NONE;
		pipelineCreateInfo.depthTestEnable = VK_FALSE;
		pipelineCreateInfo.depthWriteEnable = VK_FALSE;
		VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeState{};
		if (svoSetting.UseConservativeRasterization) {
			conservativeState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
			conservativeState.pNext = NULL;
			conservativeState.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
			conservativeState.extraPrimitiveOverestimationSize = 0.5f; // 根据需要设置
			pipelineCreateInfo.rasterizerExtensions = &conservativeState;
		}
		std::vector<VkViewport> viewports(3);
		std::vector<VkRect2D> scissors(3);
		std::array<VkViewportSwizzleNV, 3> swizzles = {};
		VkPipelineViewportSwizzleStateCreateInfoNV viewportSwizzleInfo{};
		if (svoSetting.UseSwizzle) {
			for (int i = 0; i < viewports.size(); i++) {
				viewports[i].x = 0;
				viewports[i].y = 0;
				viewports[i].width = static_cast<float>(voxelGridMapRenderPass.setting.extent.width);
				viewports[i].height = static_cast<float>(voxelGridMapRenderPass.setting.extent.height);
				viewports[i].minDepth = 0.0f;
				viewports[i].maxDepth = 1.0f;

				scissors[i].offset = { 0, 0 };
				scissors[i].extent = voxelGridMapRenderPass.setting.extent;
			}

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

			viewportSwizzleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV;
			viewportSwizzleInfo.pViewportSwizzles = swizzles.data();
			viewportSwizzleInfo.viewportCount = swizzles.size();

			pipelineCreateInfo.viewports = viewports;
			pipelineCreateInfo.scissors = scissors;
			pipelineCreateInfo.viewportExtensions = &viewportSwizzleInfo;
		}
		*/

		FzbSubPass presentSubPass = FzbSubPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, voxelGridMapRenderPass.renderPass, { voxelGridMapDescriptorSetLayout }, 0, mainComponentScene, { &vgmShader });
		voxelGridMapRenderPass.subPasses.push_back(presentSubPass);
	}

	void presentPrepare(std::vector<FzbRenderPass*>& renderPasses) {
		if (!svoSetting.Present) return;
		presentSemaphore = FzbSemaphore(logicalDevice, false);
		if (!svoSetting.UseBlock && svoSetting.UseSVO_OnlyVoxelGridMap) {
			createVGMRenderPass_nonBlock(renderPasses);
		}
		/*
		if (!(svoSetting.UseSVO_OnlyVoxelGridMap && svoSetting.UseBlock)) initDepthMap();
		if (svoSetting.UseSVO_OnlyVoxelGridMap) {
			if (!svoSetting.UseBlock) {
				createVGMRenderPass_nonBlock(mainComponentDescriptorSetLayout);
			}
			else {
				createVGMRenderPass_block(mainComponentDescriptorSetLayout);
			}
		}
		else {
			createSVORenderPass(mainComponentDescriptorSetLayout);
			//createSVORenderPassTest(mainComponentDescriptorSetLayout);
		}
		*/
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
	void createVGMRenderPass_nonBlock(std::vector<FzbRenderPass*>& renderPasses) {
		initDepthMap();

		presentMaterial = FzbMaterial(logicalDevice);
		presentShader = FzbShader(logicalDevice, getRootPath() + "/core/SceneDivision/SVO/shaders/present");
		presentShader.createShaderVariant(&presentMaterial, getComponentVertexFormat());
		for (int i = 0; i < mainComponentScene->sceneMeshSet.size(); i++) {
			presentShader.shaderVariants[0].meshBatch.meshes.push_back(&mainComponentScene->sceneMeshSet[i]);
		}
		presentShader.shaderVariants[0].meshBatch.useSameMaterial = true;
		presentShader.shaderVariants[0].meshBatch.materials.push_back(&presentMaterial);

		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment(physicalDevice);
		VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

		std::vector<VkSubpassDescription> subpasses;
		subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));

		VkSubpassDependency dependency = fzbCreateSubpassDependency();

		FzbRenderPassSetting setting = { true, 1, swapChainExtent, swapChainImageViews.size(), true };
		presentRenderPass = FzbRenderPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, setting);
		presentRenderPass.images.push_back(&depthMap);
		presentRenderPass.createRenderPass(&attachments, subpasses, { dependency });
		presentRenderPass.createFramebuffers(swapChainImageViews);

		FzbSubPass presentSubPass = FzbSubPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, presentRenderPass.renderPass, { mainComponentScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout }, 0, mainComponentScene, { &presentShader }, this->swapChainExtent);
		presentRenderPass.subPasses.push_back(presentSubPass);

		renderPasses.push_back(&voxelGridMapRenderPass);
	}

	/*
	void createVGMRenderPass_block(VkDescriptorSetLayout mainComponentDescriptorSetLayout) {
		
		FzbMesh cubeMesh;
		cubeMesh.instanceNum = pow(svoSetting.voxelNum, 3);
		cubeMesh.shader = FzbShader(logicalDevice, false, false, false);
		cubeMesh.shader.vertexShader = { true, "core/SceneDivision/SVO/shaders/present_VGM/spv/presentVert_Block.spv" };
		cubeMesh.shader.fragmentShader = { true, "core/SceneDivision/SVO/shaders/present_VGM/spv/presentFrag_Block.spv" };
		fzbCreateCube(cubeMesh.vertices, cubeMesh.indices);

		if (mainComponentScene.AABB.isEmpty()) mainComponentScene.createAABB();
		float distanceX = mainComponentScene.AABB.rightX - mainComponentScene.AABB.leftX;
		float distanceY = mainComponentScene.AABB.rightY - mainComponentScene.AABB.leftY;
		float distanceZ = mainComponentScene.AABB.rightZ - mainComponentScene.AABB.leftZ;
		float distance = glm::max(distanceX, glm::max(distanceY, distanceZ));
		float voxelSize = distance / svoSetting.voxelNum;

		float centerX = (mainComponentScene.AABB.rightX + mainComponentScene.AABB.leftX) * 0.5f;
		float centerY = (mainComponentScene.AABB.rightY + mainComponentScene.AABB.leftY) * 0.5f;
		float centerZ = (mainComponentScene.AABB.rightZ + mainComponentScene.AABB.leftZ) * 0.5f;

		for (int i = 0; i < 24; i += 3) {
			cubeMesh.vertices[i] = cubeMesh.vertices[i] * voxelSize + centerX - distance * 0.5f;
			cubeMesh.vertices[i + 1] = cubeMesh.vertices[i + 1] * voxelSize + centerY - distance * 0.5f;
			cubeMesh.vertices[i + 2] = cubeMesh.vertices[i + 2] * voxelSize + centerZ - distance * 0.5f;
		}
		componentScene = FzbScene(physicalDevice, logicalDevice, commandPool, graphicsQueue);
		componentScene.addMeshToScene(cubeMesh);
		componentScene.initScene(false);

		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve };

		std::vector<VkSubpassDescription> subpasses;
		subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, nullptr));

		VkSubpassDependency dependency = fzbCreateSubpassDependency();

		FzbRenderPassSetting setting = { true, 1, swapChainExtent, swapChainImageViews.size(), true };
		presentRenderPass = FzbRenderPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, setting);
		presentRenderPass.createRenderPass(&attachments, subpasses, { dependency });
		presentRenderPass.createFramebuffers(swapChainImageViews);

		FzbPipelineCreateInfo pipelineCreateInfo(presentRenderPass.renderPass, presentRenderPass.setting.extent);
		pipelineCreateInfo.depthTestEnable = VK_FALSE;
		pipelineCreateInfo.depthWriteEnable = VK_FALSE;
		FzbSubPass presentSubPass = FzbSubPass(physicalDevice, logicalDevice, commandPool, graphicsQueue);
		presentSubPass.createMeshBatch(&componentScene, pipelineCreateInfo, { mainComponentDescriptorSetLayout, voxelGridMapDescriptorSetLayout }, false);
		presentRenderPass.subPasses.push_back(presentSubPass);
	}

//---------------------------------------------------------------
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
//---------------------------------------------------------------


	void createSVORenderPass(VkDescriptorSetLayout mainComponentDescriptorSetLayout) {

		for (int i = 0; i < mainComponentScene.sceneMeshSet.size(); i++) {
			mainComponentScene.sceneMeshSet[i].shader.clear();
			mainComponentScene.sceneMeshSet[i].shader.vertexShader = { true, "core/SceneDivision/SVO/shaders/present/spv/presentVert.spv" };
			mainComponentScene.sceneMeshSet[i].shader.fragmentShader = { true, "core/SceneDivision/SVO/shaders/present/spv/presentFrag.spv" };
		}

		FzbShader wireframePresentShader(logicalDevice, false, false, false);
		wireframePresentShader.vertexShader = { true, "core/SceneDivision/SVO/shaders/present_SVO/spv/vert.spv" };
		wireframePresentShader.geometryShader = { true, "core/SceneDivision/SVO/shaders/present_SVO/spv/gemo.spv" };
		wireframePresentShader.fragmentShader = { true, "core/SceneDivision/SVO/shaders/present_SVO/spv/frag.spv" };
		FzbMesh cubeMesh;
		cubeMesh.shader = wireframePresentShader;
		cubeMesh.instanceNum = svoCuda->nodeBlockNum * 8 + 1;
		fzbCreateCubeWireframe(cubeMesh.vertices, cubeMesh.indices);
		componentScene = FzbScene(physicalDevice, logicalDevice, commandPool, graphicsQueue);
		componentScene.addMeshToScene(cubeMesh);
		componentScene.initScene(false);

		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment(physicalDevice);
		VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

		std::vector<VkSubpassDescription> subpasses;
		subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));
		subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));

		VkSubpassDependency dependency = fzbCreateSubpassDependency(0, 1, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

		FzbRenderPassSetting setting = { true, 1, swapChainExtent, swapChainImageViews.size(), true };
		presentRenderPass = FzbRenderPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, setting);
		presentRenderPass.images.push_back(&depthMap);
		presentRenderPass.createRenderPass(&attachments, subpasses, { dependency });
		presentRenderPass.createFramebuffers(swapChainImageViews);

		FzbPipelineCreateInfo pipelineCreateInfo(presentRenderPass.renderPass, presentRenderPass.setting.extent);
		FzbSubPass scenePresentSubPass = FzbSubPass(physicalDevice, logicalDevice, commandPool, graphicsQueue);
		scenePresentSubPass.createMeshBatch(&mainComponentScene, pipelineCreateInfo, { mainComponentDescriptorSetLayout, voxelGridMapDescriptorSetLayout }, false);
		presentRenderPass.subPasses.push_back(scenePresentSubPass);

		pipelineCreateInfo.primitiveTopology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
		pipelineCreateInfo.polyMode = VK_POLYGON_MODE_LINE;
		pipelineCreateInfo.lineWidth = 1.0f;
		pipelineCreateInfo.subPassIndex = 1;
		FzbSubPass wireframePresentSubPass = FzbSubPass(physicalDevice, logicalDevice, commandPool, graphicsQueue);
		wireframePresentSubPass.createMeshBatch(&componentScene, pipelineCreateInfo, { mainComponentDescriptorSetLayout, voxelGridMapDescriptorSetLayout, svoDescriptorSetLayout }, false);
		presentRenderPass.subPasses.push_back(wireframePresentSubPass);
	}
	*/
};

#endif
