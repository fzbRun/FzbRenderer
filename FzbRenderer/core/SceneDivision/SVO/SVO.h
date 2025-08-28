#pragma once

#include "./CUDA/createSVO.cuh"
#include "../../common/FzbComponent.h"

#ifndef SVO_H	//Sparse voxel octree
#define SVO_H

struct FzbSVOSetting {
	bool UseSVO = true;
	bool UseSVO_OnlyVoxelGridMap = false;
	bool UseSwizzle = false;
	bool UseConservativeRasterization = false;

	bool Present = false;
	bool UseBlock = false;

	bool useCube;	//�Ƿ񽫵�ǰ��������һ�����������洢
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

	FzbBuffer vertexBuffer;		//�������ֻ��Ҫmeshes�Ķ������ԣ�ʹ��ͬһ��pipeline�����ʹ��scene��indexBuffer�ᵼ��pipelien��VAO��ͬ����ͬʹ��ͬһ��
	FzbBuffer indexBuffer;
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

	FzbVertexFormat getComponentVertexFormat() {	//�������е�shader����Ҫ�õ��Ķ�������
		return FzbVertexFormat(true);
	}

	void init(FzbMainComponent* renderer,  FzbSVOSetting setting, FzbScene* scene, std::vector<FzbRenderPass*>& renderPasses) {
		initComponent(renderer);
		mainComponentScene = scene;
		this->svoSetting = setting;

		createVoxelGridMap();
		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			this->svoCuda = std::make_unique<SVOCuda>();
			createSVOCuda();
			createSVODescriptor();
		}
		presentPrepare(renderPasses);
	}

	//���ڲ������ػ����
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

		presentRenderPass.render(commandBuffer, imageIndex);

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
	/*
	VkSemaphore render_testExtensions(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) {
		mainComponentScene->updateCameraBuffer();

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

		presentRenderPass.render(commandBuffer, imageIndex, { {voxelGridMapDescriptorSet}, {mainComponentScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet} });

		std::vector<VkSemaphore> waitSemaphores = { startSemaphore };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
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

		return presentSemaphore.semaphore;
	}
	*/
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

		voxelGridMapRenderPass.render(commandBuffer, 0);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &vgmSemaphore.semaphore;

		//ִ�����⿪fence
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}
	}
	void createVoxelGridMapBuffer() {
		fzbCreateCommandBuffers(2);

		svoUniformBuffer = fzbComponentCreateUniformBuffers<SVOUniform>();
		SVOUniform svoUniform;
		FzbAABBBox mianSceneAABB = mainComponentScene->getAABB();
		float distanceX = mianSceneAABB.rightX - mianSceneAABB.leftX;
		float distanceY = mianSceneAABB.rightY - mianSceneAABB.leftY;
		float distanceZ = mianSceneAABB.rightZ - mianSceneAABB.leftZ;
		//���ö���ͨ��swizzle�任��õ���ȷ�Ľ�������뱣֤ͶӰ������������ģ�����xyzͨ��1��������Ƕ�Ӧ��
		float distance = glm::max(distanceX, glm::max(distanceY, distanceZ)) * 1.2f;
		float centerX = (mianSceneAABB.rightX + mianSceneAABB.leftX) * 0.5f;
		float centerY = (mianSceneAABB.rightY + mianSceneAABB.leftY) * 0.5f;
		float centerZ = (mianSceneAABB.rightZ + mianSceneAABB.leftZ) * 0.5f;
		float newLeftX = centerX - distance * 0.5f;
		float newLeftY = centerY - distance * 0.5f;
		float newLeftZ = centerZ - distance * 0.5f;
		float newRightX = centerX + distance * 0.5f;
		float newRightY = centerY + distance * 0.5f;
		float newRightZ = centerZ + distance * 0.5f;
		//ǰ��
		glm::vec3 viewPoint = glm::vec3(centerX, centerY, newRightZ + 0.1f);	//��������������������+z����
		glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[0] = orthoMatrix * viewMatrix;
		//���
		viewPoint = glm::vec3(newLeftX - 0.1f, centerY, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[1] = orthoMatrix * viewMatrix;
		//����
		viewPoint = glm::vec3(centerX, newLeftY - 0.1f, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[2] = orthoMatrix * viewMatrix;

		if (svoSetting.useCube) {
			svoUniform.voxelSize_Num = glm::vec4(distance / svoSetting.voxelNum, distance / svoSetting.voxelNum, distance / svoSetting.voxelNum, svoSetting.voxelNum);
			svoUniform.voxelStartPos = glm::vec4(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f, 0.0f);
		} 
		else {
			svoUniform.voxelSize_Num = glm::vec4(distanceX / svoSetting.voxelNum, distanceY / svoSetting.voxelNum, distanceZ / svoSetting.voxelNum, svoSetting.voxelNum);
			svoUniform.voxelStartPos = glm::vec4(centerX - distanceX * 0.5f, centerY - distanceY * 0.5f, centerZ - distanceZ * 0.5f, 0.0f);
		}
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
	void createVoxelGridMapSyncObjects() {	//����Ӧ�÷���һ���ź�����Ȼ���������̣߳�֪����Ⱦ��ɣ����ܻ���
		if (svoSetting.UseSVO_OnlyVoxelGridMap) {
			vgmSemaphore = FzbSemaphore(logicalDevice, false);	//��vgm������ɺ���
		}
		else {
			vgmSemaphore = FzbSemaphore(logicalDevice, true);		//��vgm������ɺ���
			svoCudaSemaphore = FzbSemaphore(logicalDevice, true);		//��svo������ɺ���
		}
		fence = fzbCreateFence();
	}
	void createVGMRenderPass() {
		vgmMaterial = FzbMaterial(logicalDevice);
		FzbShaderExtensionsSetting extensionsSetting;
		if (svoSetting.UseConservativeRasterization) extensionsSetting.conservativeRasterization = true;
		if (!svoSetting.UseSwizzle) vgmShader = FzbShader(logicalDevice, getRootPath() + "/core/SceneDivision/SVO/shaders/unuseSwizzle", extensionsSetting);
		else {
			extensionsSetting.swizzle = true;
			vgmShader = FzbShader(logicalDevice, getRootPath() + "/core/SceneDivision/SVO/shaders/useSwizzle", extensionsSetting);
		} 
		vgmShader.createShaderVariant(&vgmMaterial, getComponentVertexFormat());
		for (int i = 0; i < mainComponentScene->sceneMeshSet.size(); i++) {
			vgmShader.shaderVariants[0].meshBatch.meshes.push_back(&mainComponentScene->sceneMeshSet[i]);
		}
		vgmShader.shaderVariants[0].meshBatch.useSameMaterial = true;
		vgmShader.shaderVariants[0].meshBatch.materials.push_back(&vgmMaterial);

		VkSubpassDescription voxelGridMapSubpass{};
		voxelGridMapSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

		VkSubpassDependency dependency = fzbCreateSubpassDependency();

		FzbRenderPassSetting setting = { false, 1, vgmShader.getResolution(), 1, false};
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
			conservativeState.extraPrimitiveOverestimationSize = 0.5f; // ������Ҫ����
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

			swizzles[0] = {		//ǰ��
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
			};
			swizzles[1] = {		//����
				VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Z_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
			};
			swizzles[2] = {		//����
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

		FzbSubPass vgmSubPass = FzbSubPass(logicalDevice, voxelGridMapRenderPass.renderPass, 0, 
			{ voxelGridMapDescriptorSetLayout }, { voxelGridMapDescriptorSet },
			mainComponentScene->vertexPosNormalBuffer.buffer, mainComponentScene->indexPosNormalBuffer.buffer, { &vgmShader });
		vgmSubPass.createPipeline(mainComponentScene->meshDescriptorSetLayout);
		voxelGridMapRenderPass.subPasses.push_back(vgmSubPass);
	}

	void createSVOCuda() {
		svoCuda->createSVOCuda(physicalDevice, voxelGridMap, vgmSemaphore.handle, svoCudaSemaphore.handle);
		nodePool = fzbComponentCreateStorageBuffer(sizeof(FzbSVONode) * (8 * svoCuda->nodeBlockNum + 1), true);
		voxelValueBuffer = fzbComponentCreateStorageBuffer(sizeof(FzbVoxelValue) * svoCuda->voxelNum, true);
		svoCuda->getSVOCuda(physicalDevice, nodePool.handle, voxelValueBuffer.handle);
	}

	void presentPrepare(std::vector<FzbRenderPass*>& renderPasses) {
		if (!svoSetting.Present) return;
		presentSemaphore = FzbSemaphore(logicalDevice, false);
		if (svoSetting.UseSVO_OnlyVoxelGridMap) {
			if (!svoSetting.UseBlock) {
				createVGMRenderPass_nonBlock(renderPasses);
				//testExtensionsPresent(renderPasses);
			}
			else createVGMRenderPass_Block(renderPasses);
		}
		else {
			createSVORenderPass(renderPasses);
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

		FzbSubPass presentSubPass = FzbSubPass(logicalDevice, presentRenderPass.renderPass, 0,
			{ mainComponentScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout }, { mainComponentScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet },
			mainComponentScene->vertexPosNormalBuffer.buffer, mainComponentScene->indexPosNormalBuffer.buffer, { &presentShader }, this->swapChainExtent);
		presentSubPass.createPipeline(mainComponentScene->meshDescriptorSetLayout);
		presentRenderPass.subPasses.push_back(presentSubPass);

		renderPasses.push_back(&presentRenderPass);
	}
	void createVGMRenderPass_Block(std::vector<FzbRenderPass*>& renderPasses) {
		initDepthMap();

		presentMaterial = FzbMaterial(logicalDevice);
		FzbMesh cubeMesh = FzbMesh(logicalDevice);
		cubeMesh.instanceNum = pow(svoSetting.voxelNum, 3);
		fzbCreateCube(cubeMesh);
		if (mainComponentScene->AABB.isEmpty()) mainComponentScene->createAABB();
		float distanceX = mainComponentScene->AABB.rightX - mainComponentScene->AABB.leftX;
		float distanceY = mainComponentScene->AABB.rightY - mainComponentScene->AABB.leftY;
		float distanceZ = mainComponentScene->AABB.rightZ - mainComponentScene->AABB.leftZ;
		float distance = glm::max(distanceX, glm::max(distanceY, distanceZ)) * 1.2f;
		float voxelSize = distance / svoSetting.voxelNum;
		float centerX = (mainComponentScene->AABB.rightX + mainComponentScene->AABB.leftX) * 0.5f;
		float centerY = (mainComponentScene->AABB.rightY + mainComponentScene->AABB.leftY) * 0.5f;
		float centerZ = (mainComponentScene->AABB.rightZ + mainComponentScene->AABB.leftZ) * 0.5f;
		for (int i = 0; i < 24; i += 3) {
			cubeMesh.vertices[i] = cubeMesh.vertices[i] * voxelSize + centerX - distance * 0.5f;
			cubeMesh.vertices[i + 1] = cubeMesh.vertices[i + 1] * voxelSize + centerY - distance * 0.5f;
			cubeMesh.vertices[i + 2] = cubeMesh.vertices[i + 2] * voxelSize + centerZ - distance * 0.5f;
		}
		this->componentScene = FzbScene(physicalDevice, logicalDevice, commandPool, graphicsQueue);
		this->componentScene.addMeshToScene(cubeMesh, presentMaterial, getRootPath() + "/core/SceneDivision/SVO/shaders/present_VGM");
		componentScene.initScene({}, false);

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

		FzbSubPass presentSubPass = FzbSubPass(logicalDevice, presentRenderPass.renderPass, 0, 
			{ mainComponentScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout }, { mainComponentScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet },
			componentScene.vertexBuffer.buffer, componentScene.indexBuffer.buffer, componentScene.sceneShaders_vector, this->swapChainExtent);
		presentSubPass.createPipeline(componentScene.meshDescriptorSetLayout);
		presentRenderPass.subPasses.push_back(presentSubPass);
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
	void createSVORenderPass(std::vector<FzbRenderPass*>& renderPasses) {
		initDepthMap();

		presentMaterial = FzbMaterial(logicalDevice);
		FzbMesh cubeMesh = FzbMesh(logicalDevice);
		cubeMesh.instanceNum = svoCuda->nodeBlockNum * 8 + 1;
		fzbCreateCubeWireframe(cubeMesh);
		this->componentScene = FzbScene(physicalDevice, logicalDevice, commandPool, graphicsQueue);
		this->componentScene.addMeshToScene(cubeMesh, presentMaterial, getRootPath() + "/core/SceneDivision/SVO/shaders/present_SVO");
		componentScene.initScene({}, false);

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
		subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));

		VkSubpassDependency dependency = fzbCreateSubpassDependency(0, 1, 
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

		FzbRenderPassSetting setting = { true, 1, swapChainExtent, swapChainImageViews.size(), true };
		presentRenderPass = FzbRenderPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, setting);
		presentRenderPass.images.push_back(&depthMap);
		presentRenderPass.createRenderPass(&attachments, subpasses, { dependency });
		presentRenderPass.createFramebuffers(swapChainImageViews);

		FzbSubPass sceneSubPass = FzbSubPass(logicalDevice, presentRenderPass.renderPass, 0, 
			{ mainComponentScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout }, 
			{ mainComponentScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet },
			mainComponentScene->vertexPosNormalBuffer.buffer, mainComponentScene->indexPosNormalBuffer.buffer, { &presentShader }, this->swapChainExtent);
		sceneSubPass.createPipeline(mainComponentScene->meshDescriptorSetLayout);
		presentRenderPass.subPasses.push_back(sceneSubPass);

		FzbSubPass CubeWireframeSubPass = FzbSubPass(logicalDevice, presentRenderPass.renderPass, 1,
			{ mainComponentScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout, svoDescriptorSetLayout },
			{ mainComponentScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet, svoDescriptorSet },
			componentScene.vertexBuffer.buffer, componentScene.indexBuffer.buffer, componentScene.sceneShaders_vector, this->swapChainExtent);
		CubeWireframeSubPass.createPipeline(componentScene.meshDescriptorSetLayout);
		presentRenderPass.subPasses.push_back(CubeWireframeSubPass);

		renderPasses.push_back(&presentRenderPass);
	}

	/*
	void testExtensionsPresent(std::vector<FzbRenderPass*>& renderPasses) {
		vgmMaterial = FzbMaterial(logicalDevice);
		FzbShaderExtensionsSetting extensionsSetting;
		if (svoSetting.UseConservativeRasterization) extensionsSetting.conservativeRasterization = true;
		if (!svoSetting.UseSwizzle) vgmShader = FzbShader(logicalDevice, getRootPath() + "/core/SceneDivision/SVO/shaders/unuseSwizzle", extensionsSetting);
		else {
			extensionsSetting.swizzle = true;
			vgmShader = FzbShader(logicalDevice, getRootPath() + "/core/SceneDivision/SVO/shaders/useSwizzle", extensionsSetting);
		}
		vgmShader.createShaderVariant(&vgmMaterial, getComponentVertexFormat());
		for (int i = 0; i < mainComponentScene->sceneMeshSet.size(); i++) {
			vgmShader.shaderVariants[0].meshBatch.meshes.push_back(&mainComponentScene->sceneMeshSet[i]);
		}
		vgmShader.shaderVariants[0].meshBatch.useSameMaterial = true;
		vgmShader.shaderVariants[0].meshBatch.materials.push_back(&vgmMaterial);

		//---------------------------------------------

		initDepthMap();

		presentMaterial = FzbMaterial(logicalDevice);
		presentShader = FzbShader(logicalDevice, getRootPath() + "/core/SceneDivision/SVO/shaders/present");
		presentShader.createShaderVariant(&presentMaterial, getComponentVertexFormat());
		for (int i = 0; i < mainComponentScene->sceneMeshSet.size(); i++) {
			presentShader.shaderVariants[0].meshBatch.meshes.push_back(&mainComponentScene->sceneMeshSet[i]);
		}
		presentShader.shaderVariants[0].meshBatch.useSameMaterial = true;
		presentShader.shaderVariants[0].meshBatch.materials.push_back(&presentMaterial);

		//---------------------------------------------
		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment(physicalDevice);
		VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

		std::vector<VkSubpassDescription> subpasses;
		subpasses.push_back(fzbCreateSubPass());
		subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));

		VkSubpassDependency dependency = fzbCreateSubpassDependency(0, 1, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);

		FzbRenderPassSetting setting = { true, 1, swapChainExtent, swapChainImageViews.size(), true };
		presentRenderPass = FzbRenderPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, setting);
		presentRenderPass.images.push_back(&depthMap);
		presentRenderPass.createRenderPass(&attachments, subpasses, { dependency });
		presentRenderPass.createFramebuffers(swapChainImageViews);

		FzbSubPass vgmSubPass = FzbSubPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, presentRenderPass.renderPass, 0, mainComponentScene->vertexPosNormalBuffer.buffer, mainComponentScene->indexPosNormalBuffer.buffer, { &vgmShader }, this->swapChainExtent);
		vgmSubPass.createPipeline({ voxelGridMapDescriptorSetLayout }, mainComponentScene->meshDescriptorSetLayout);
		presentRenderPass.subPasses.push_back(vgmSubPass);

		FzbSubPass presentSubPass = FzbSubPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, presentRenderPass.renderPass, 1, mainComponentScene->vertexPosNormalBuffer.buffer, mainComponentScene->indexPosNormalBuffer.buffer, { &presentShader }, this->swapChainExtent);
		presentSubPass.createPipeline({ mainComponentScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout }, mainComponentScene->meshDescriptorSetLayout);
		presentRenderPass.subPasses.push_back(presentSubPass);

		renderPasses.push_back(&presentRenderPass);
	}

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

		//���ڲ��ܴ�cuda��ֱ�ӵ��������handle�����������Ҫ�ȴ���һ��buffer��Ȼ����cuda�н�����copy��ȥ
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
