#pragma once

#include "./CUDA/createSVO.cuh"
#include "../../common/StructSet.h"
#include "../../common/FzbComponent/FzbFeatureComponent.h"
#include "../../common/FzbRenderer.h"
#include "../../common/FzbRenderPass/FzbRenderPass.h"

#ifndef SVO_H	//Sparse voxel octree
#define SVO_H

struct FzbSVOSetting {
	bool useSVO_OnlyVoxelGridMap = false;
	bool useSwizzle = false;
	bool useConservativeRasterization = false;

	bool useBlock = false;

	bool useCube;	//是否将当前场景当作一个立方体来存储
	int voxelNum = 64;
};

struct SVOUniform {
	glm::mat4 VP[3];
	glm::vec4 voxelSize_Num;
	glm::vec4 voxelStartPos;
};

/*
struct FzbSVO : public FzbFeatureComponent_PreProcess {

public:

	FzbSVOSetting setting;

	FzbImage voxelGridMap;
	VkDescriptorSetLayout voxelGridMapDescriptorSetLayout = nullptr;
	VkDescriptorSet voxelGridMapDescriptorSet;

	FzbBuffer vertexBuffer;		//组件可能只需要meshes的顶点属性，使用同一个pipeline，如果使用scene的indexBuffer会导致pipelien的VAO不同，不同使用同一个
	FzbBuffer indexBuffer;
	SVOUniform svoUniform;
	FzbBuffer svoUniformBuffer;

	FzbRenderPass voxelGridMapRenderPass;
	FzbRenderPass* presentRenderPass;

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

	FzbSVO() {};
	FzbSVO(pugi::xml_document& doc) {
		this->componentInfo.name = FZB_FEATURE_COMPONENT_SVO;
		this->componentInfo.type = FZB_PREPROCESS_FEATURE_COMPONENT;
		FzbFeatureComponentInfo componentInfo;
		componentInfo.vertexFormat = FzbVertexFormat(true);
		componentInfo.useMainSceneBufferHandle = { true, false, false };	//需要全部格式的顶点buffer和索引buffer，用来创建svo

		pugi::xml_node SVONode = doc.child("featureComponents").select_node(".//featureComponent[@name='SVO']").node();
		this->setting.available = SVONode.child("available").attribute("value").value() == "true";
		this->setting.present = false;
		this->setting.voxelNum = std::stoi(SVONode.child("voxelNum").attribute("voxelNum").value());
		this->setting.useSVO_OnlyVoxelGridMap = SVONode.child("useOnlyVoxelGridMap").attribute("value").value() == "true";
		this->setting.useSwizzle = SVONode.child("useSwizzle").attribute("value").value() == "true";
		this->setting.useConservativeRasterization = SVONode.child("useConservativeRasterization").attribute("value").value() == "true";
		//this->setting.useBlock = SVONode.child("useBlock").attribute("value").value() == "true";
		this->setting.useCube = SVONode.child("useCube").attribute("value").value() == "true";
		addExtensions();
	}
	void addExtensions() {
		if (!setting.useSVO_OnlyVoxelGridMap) {
			FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
			FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
		}

		if (setting.useConservativeRasterization) {
			FzbRenderer::globalData.deviceExtensions.push_back(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
		}
		if (setting.useSwizzle) {
			FzbRenderer::globalData.deviceExtensions.push_back(VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME);
		}
		if (!setting.useSVO_OnlyVoxelGridMap) {
			FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
		}

		if (setting.useSwizzle)
			FzbRenderer::globalData.deviceFeatures.multiViewport = VK_TRUE;
		if (!setting.useSVO_OnlyVoxelGridMap) {
			FzbRenderer::globalData.deviceFeatures.fillModeNonSolid = VK_TRUE;
			FzbRenderer::globalData.deviceFeatures.wideLines = VK_TRUE;
		}
	}

	void init() {
		initGlobalData();
		createVoxelGridMap();
		if (!setting.useSVO_OnlyVoxelGridMap) {
			this->svoCuda = std::make_unique<SVOCuda>();
			createSVOCuda();
		}
	}
	void clean() {

		if (!setting.useSVO_OnlyVoxelGridMap) {
			svoCuda->clean();
		}
		voxelGridMap.clean();
		depthMap.clean();

		voxelGridMapRenderPass.clean();

		vgmShader.clean();
		vgmMaterial.clean();

		svoUniformBuffer.clean();
		nodePool.clean();
		voxelValueBuffer.clean();

		vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
		if(voxelGridMapDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, voxelGridMapDescriptorSetLayout, nullptr);

		vgmSemaphore.clean(logicalDevice);
		svoCudaSemaphore.clean(logicalDevice);
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

		//执行完后解开fence
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}
	}
	void createVoxelGridMapBuffer() {
		fzbCreateCommandBuffers(2);

		svoUniformBuffer = fzbComponentCreateUniformBuffers(sizeof(SVOUniform));
		FzbAABBBox mianSceneAABB = mainScene->getAABB();
		mianSceneAABB.createDistanceAndCenter(setting.useCube, 1.2);
		float distanceX = mianSceneAABB.distanceX;
		float distanceY = mianSceneAABB.distanceY;
		float distanceZ = mianSceneAABB.distanceZ;
		//想让顶点通过swizzle变换后得到正确的结果，必须保证投影矩阵是立方体的，这样xyz通过1减后才能是对应的
		float distance = mianSceneAABB.distance;
		float centerX = mianSceneAABB.centerX;
		float centerY = mianSceneAABB.centerY;
		float centerZ = mianSceneAABB.centerZ;
		float newLeftX = centerX - distance * 0.5f;
		float newLeftY = centerY - distance * 0.5f;
		float newLeftZ = centerZ - distance * 0.5f;
		float newRightX = centerX + distance * 0.5f;
		float newRightY = centerY + distance * 0.5f;
		float newRightZ = centerZ + distance * 0.5f;
		//前面
		glm::vec3 viewPoint = glm::vec3(centerX, centerY, newRightZ + 0.1f);	//世界坐标右手螺旋，即+z朝后
		glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[0] = orthoMatrix * viewMatrix;
		//左边
		viewPoint = glm::vec3(newLeftX - 0.1f, centerY, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[1] = orthoMatrix * viewMatrix;
		//下面
		viewPoint = glm::vec3(centerX, newLeftY - 0.1f, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[2] = orthoMatrix * viewMatrix;

		if (setting.useCube) {
			svoUniform.voxelSize_Num = glm::vec4(distance / setting.voxelNum, distance / setting.voxelNum, distance / setting.voxelNum, setting.voxelNum);
			svoUniform.voxelStartPos = glm::vec4(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f, 0.0f);
		} 
		else {
			svoUniform.voxelSize_Num = glm::vec4(distanceX / setting.voxelNum, distanceY / setting.voxelNum, distanceZ / setting.voxelNum, setting.voxelNum);
			svoUniform.voxelStartPos = glm::vec4(centerX - distanceX * 0.5f, centerY - distanceY * 0.5f, centerZ - distanceZ * 0.5f, 0.0f);
		}
		memcpy(svoUniformBuffer.mapped, &svoUniform, sizeof(SVOUniform));
	}
	void initVoxelGridMap() {
		voxelGridMap = {};
		voxelGridMap.width = setting.voxelNum;
		voxelGridMap.height = setting.voxelNum;
		voxelGridMap.depth = setting.voxelNum;
		voxelGridMap.type = VK_IMAGE_TYPE_3D;
		voxelGridMap.viewType = VK_IMAGE_VIEW_TYPE_3D;
		voxelGridMap.format = VK_FORMAT_R32_UINT;
		voxelGridMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		voxelGridMap.UseExternal = !setting.useSVO_OnlyVoxelGridMap;
		voxelGridMap.fzbCreateImage(physicalDevice, logicalDevice, commandPool, graphicsQueue);
	}
	void createDescriptorPool() {
		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
		if (!setting.useSVO_OnlyVoxelGridMap) {
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
		if (setting.useSVO_OnlyVoxelGridMap) {
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
		FzbShaderExtensionsSetting extensionsSetting;
		if (setting.useConservativeRasterization) extensionsSetting.conservativeRasterization = true;
		if (!setting.useSwizzle) vgmShader = FzbShader(logicalDevice, fzbGetRootPath() + "/core/SceneDivision/SVO/shaders/unuseSwizzle", extensionsSetting);
		else {
			extensionsSetting.swizzle = true;
			vgmShader = FzbShader(logicalDevice, fzbGetRootPath() + "/core/SceneDivision/SVO/shaders/useSwizzle", extensionsSetting);
		} 
		vgmShader.createShaderVariant(&vgmMaterial, componentInfo.vertexFormat);
		for (int i = 0; i < mainScene->sceneMeshSet.size(); i++) {
			vgmShader.shaderVariants[0].meshBatch.meshes.push_back(&mainScene->sceneMeshSet[i]);
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

		FzbSubPass vgmSubPass = FzbSubPass(logicalDevice, voxelGridMapRenderPass.renderPass, 0, 
			{ voxelGridMapDescriptorSetLayout }, { voxelGridMapDescriptorSet },
			mainScene->vertexPosNormalBuffer.buffer, mainScene->indexPosNormalBuffer.buffer, { &vgmShader });
		vgmSubPass.createPipeline(mainScene->meshDescriptorSetLayout);
		voxelGridMapRenderPass.subPasses.push_back(vgmSubPass);
	}

	void createSVOCuda() {
		svoCuda->createSVOCuda(physicalDevice, voxelGridMap, vgmSemaphore.handle, svoCudaSemaphore.handle);
		nodePool = fzbComponentCreateStorageBuffer(sizeof(FzbSVONode) * (8 * svoCuda->nodeBlockNum + 1), true);
		voxelValueBuffer = fzbComponentCreateStorageBuffer(sizeof(FzbVoxelValue) * svoCuda->voxelNum, true);
		svoCuda->getSVOCuda(physicalDevice, nodePool.handle, voxelValueBuffer.handle);
	}
};
*/
struct FzbSVO_Debug : public FzbFeatureComponent_LoopRender {
public:

	FzbSVO_Debug() {};
	FzbSVO_Debug(pugi::xml_node& SVONode) {
		if (std::string(SVONode.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
		else return;

		this->componentInfo.name = FZB_FEATURE_COMPONENT_SVO_DEBUG;
		this->componentInfo.type = FZB_LOOPRENDER_FEATURE_COMPONENT;
		this->componentInfo.vertexFormat = FzbVertexFormat(true);
		this->componentInfo.useMainSceneBufferHandle = { false, false, false };

		if (pugi::xml_node SVOSettingNode = SVONode.child("featureComponentSetting")) {
			this->setting.voxelNum = std::stoi(SVOSettingNode.child("voxelNum").attribute("value").value());
			this->setting.useSVO_OnlyVoxelGridMap = std::string(SVOSettingNode.child("useOnlyVoxelGridMap").attribute("value").value()) == "true";
			this->setting.useSwizzle = std::string(SVOSettingNode.child("useSwizzle").attribute("value").value()) == "true";
			this->setting.useConservativeRasterization = std::string(SVOSettingNode.child("useConservativeRasterization").attribute("value").value()) == "true";
			this->setting.useBlock = std::string(SVOSettingNode.child("useBlock").attribute("value").value()) == "true";
			this->setting.useCube = std::string(SVOSettingNode.child("useCube").attribute("value").value()) == "true";
		}
		addExtensions();
	}

	void init() override {
		FzbFeatureComponent_LoopRender::init();
		createVoxelGridMap();
		if (!setting.useSVO_OnlyVoxelGridMap) {
			this->svoCuda = std::make_unique<SVOCuda>();
			createSVOCuda();
			createSVODescriptor();
		}
		presentPrepare();
	}

	VkSemaphore render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) {
		VkCommandBuffer commandBuffer = commandBuffers[1];
		vkResetCommandBuffer(commandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		renderRenderPass.render(commandBuffer, imageIndex);

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
		submitInfo.pSignalSemaphores = &renderFinishedSemaphore.semaphore;

		if (vkQueueSubmit(FzbRenderer::globalData.graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		return renderFinishedSemaphore.semaphore;
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
	void clean() override {
		FzbFeatureComponent_LoopRender::clean();
		VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;

		componentScene.clean();
		if (!setting.useSVO_OnlyVoxelGridMap) {
			svoCuda->clean();
		}
		voxelGridMap.clean();
		depthMap.clean();

		voxelGridMapRenderPass.clean();

		vgmShader.clean();
		vgmMaterial.clean();
		presentShader.clean();
		presentMaterial.clean();

		svoUniformBuffer.clean();
		nodePool.clean();
		voxelValueBuffer.clean();

		if (voxelGridMapDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, voxelGridMapDescriptorSetLayout, nullptr);
		if (svoDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, svoDescriptorSetLayout, nullptr);

		vgmSemaphore.clean();
		svoCudaSemaphore.clean();
	}

private:
	FzbSVOSetting setting;

	FzbImage voxelGridMap;
	VkDescriptorSetLayout voxelGridMapDescriptorSetLayout = nullptr;
	VkDescriptorSet voxelGridMapDescriptorSet;

	FzbScene componentScene;

	SVOUniform svoUniform;
	FzbBuffer svoUniformBuffer;

	FzbRenderPass voxelGridMapRenderPass;

	FzbImage depthMap;

	std::unique_ptr<SVOCuda> svoCuda;
	FzbBuffer nodePool;
	FzbBuffer voxelValueBuffer;

	VkDescriptorSetLayout svoDescriptorSetLayout = nullptr;
	VkDescriptorSet svoDescriptorSet;

	FzbSemaphore vgmSemaphore;
	FzbSemaphore svoCudaSemaphore;

	FzbShader vgmShader;
	FzbMaterial vgmMaterial;
	FzbShader presentShader;
	FzbMaterial presentMaterial;

	void addExtensions() override {
		if (!setting.useSVO_OnlyVoxelGridMap) {
			FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
			FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
		}

		if (setting.useConservativeRasterization) {
			FzbRenderer::globalData.deviceExtensions.push_back(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
		}
		if (setting.useSwizzle) {
			FzbRenderer::globalData.deviceExtensions.push_back(VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME);
		}
		if (!setting.useSVO_OnlyVoxelGridMap) {
			FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
			FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
		}

		if (setting.useSwizzle)
			FzbRenderer::globalData.deviceFeatures.multiViewport = VK_TRUE;
		if (!setting.useSVO_OnlyVoxelGridMap) {
			FzbRenderer::globalData.deviceFeatures.fillModeNonSolid = VK_TRUE;
			FzbRenderer::globalData.deviceFeatures.wideLines = VK_TRUE;
		}
	}

	void createImages() override {
		VkExtent2D swapChainExtent = FzbRenderer::globalData.swapChainExtent;

		depthMap = FzbImage();
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

	void createVoxelGridMap() {

		createVoxelGridMapBuffer();
		initVoxelGridMap();
		createDescriptorPool();
		createVoxelGridMapDescriptor();
		createVoxelGridMapSyncObjects();
		createVGMRenderPass();

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

		//执行完后解开fence
		if (vkQueueSubmit(FzbRenderer::globalData.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}
	}
	void createVoxelGridMapBuffer() {
		fzbCreateCommandBuffers(2);

		svoUniformBuffer = fzbCreateUniformBuffers(sizeof(SVOUniform));
		FzbAABBBox mianSceneAABB = mainScene->getAABB();
		mianSceneAABB.createDistanceAndCenter(setting.useCube, 1.2);
		float distanceX = mianSceneAABB.distanceX;
		float distanceY = mianSceneAABB.distanceY;
		float distanceZ = mianSceneAABB.distanceZ;
		//想让顶点通过swizzle变换后得到正确的结果，必须保证投影矩阵是立方体的，这样xyz通过1减后才能是对应的
		float distance = mianSceneAABB.distance;
		float centerX = mianSceneAABB.centerX;
		float centerY = mianSceneAABB.centerY;
		float centerZ = mianSceneAABB.centerZ;
		float newLeftX = centerX - distance * 0.5f;
		float newLeftY = centerY - distance * 0.5f;
		float newLeftZ = centerZ - distance * 0.5f;
		float newRightX = centerX + distance * 0.5f;
		float newRightY = centerY + distance * 0.5f;
		float newRightZ = centerZ + distance * 0.5f;
		//前面
		glm::vec3 viewPoint = glm::vec3(centerX, centerY, newRightZ + 0.1f);	//世界坐标右手螺旋，即+z朝后
		glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[0] = orthoMatrix * viewMatrix;
		//左边
		viewPoint = glm::vec3(newLeftX - 0.1f, centerY, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[1] = orthoMatrix * viewMatrix;
		//下面
		viewPoint = glm::vec3(centerX, newLeftY - 0.1f, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
		orthoMatrix[1][1] *= -1;
		svoUniform.VP[2] = orthoMatrix * viewMatrix;

		if (setting.useCube) {
			svoUniform.voxelSize_Num = glm::vec4(distance / setting.voxelNum, distance / setting.voxelNum, distance / setting.voxelNum, setting.voxelNum);
			svoUniform.voxelStartPos = glm::vec4(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f, 0.0f);
		}
		else {
			svoUniform.voxelSize_Num = glm::vec4(distanceX / setting.voxelNum, distanceY / setting.voxelNum, distanceZ / setting.voxelNum, setting.voxelNum);
			svoUniform.voxelStartPos = glm::vec4(centerX - distanceX * 0.5f, centerY - distanceY * 0.5f, centerZ - distanceZ * 0.5f, 0.0f);
		}
		memcpy(svoUniformBuffer.mapped, &svoUniform, sizeof(SVOUniform));
	}
	void initVoxelGridMap() {
		voxelGridMap = {};
		voxelGridMap.width = setting.voxelNum;
		voxelGridMap.height = setting.voxelNum;
		voxelGridMap.depth = setting.voxelNum;
		voxelGridMap.type = VK_IMAGE_TYPE_3D;
		voxelGridMap.viewType = VK_IMAGE_VIEW_TYPE_3D;
		voxelGridMap.format = VK_FORMAT_R32_UINT;
		voxelGridMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		voxelGridMap.UseExternal = !setting.useSVO_OnlyVoxelGridMap;
		voxelGridMap.initImage();
	}
	void createDescriptorPool() {
		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
		if (!setting.useSVO_OnlyVoxelGridMap) {
			bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 });
		}
		this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);
	}
	void createVoxelGridMapDescriptor() {

		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_FRAGMENT_BIT };
		voxelGridMapDescriptorSetLayout = fzbCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		voxelGridMapDescriptorSet = fzbCreateDescriptorSet(descriptorPool, voxelGridMapDescriptorSetLayout);

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

		vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, voxelGridMapDescriptorWrites.size(), voxelGridMapDescriptorWrites.data(), 0, nullptr);
		//fzbDescriptor->descriptorSets.push_back({ voxelGridMapDescriptorSet });

	}
	void createVoxelGridMapSyncObjects() {	//这里应该返回一个信号量，然后阻塞主线程，知道渲染完成，才能唤醒
		if (setting.useSVO_OnlyVoxelGridMap) {
			vgmSemaphore = FzbSemaphore(false);	//当vgm创建完成后唤醒
		}
		else {
			vgmSemaphore = FzbSemaphore(true);		//当vgm创建完成后唤醒
			svoCudaSemaphore = FzbSemaphore(true);		//当svo创建完成后唤醒
		}
	}
	void createVGMRenderPass() {
		vgmMaterial = FzbMaterial();
		FzbShaderExtensionsSetting extensionsSetting;
		if (setting.useConservativeRasterization) extensionsSetting.conservativeRasterization = true;
		if (!setting.useSwizzle) vgmShader = FzbShader(fzbGetRootPath() + "/core/SceneDivision/SVO/shaders/unuseSwizzle", extensionsSetting);
		else {
			extensionsSetting.swizzle = true;
			vgmShader = FzbShader(fzbGetRootPath() + "/core/SceneDivision/SVO/shaders/useSwizzle", extensionsSetting);
		}
		vgmShader.createShaderVariant(&vgmMaterial, componentInfo.vertexFormat);
		for (int i = 0; i < mainScene->sceneMeshSet.size(); i++) {
			vgmShader.shaderVariants[0].meshBatch.meshes.push_back(&mainScene->sceneMeshSet[i]);
		}
		vgmShader.shaderVariants[0].meshBatch.useSameMaterial = true;
		vgmShader.shaderVariants[0].meshBatch.materials.push_back(&vgmMaterial);

		VkSubpassDescription voxelGridMapSubpass{};
		voxelGridMapSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

		VkSubpassDependency dependency = fzbCreateSubpassDependency();

		FzbRenderPassSetting setting = { false, 1, vgmShader.getResolution(), 1, false };
		voxelGridMapRenderPass = FzbRenderPass(setting);
		voxelGridMapRenderPass.createRenderPass(nullptr, { voxelGridMapSubpass }, { dependency });
		voxelGridMapRenderPass.createFramebuffers(false);

		FzbSubPass vgmSubPass = FzbSubPass(voxelGridMapRenderPass.renderPass, 0,
			{ voxelGridMapDescriptorSetLayout }, { voxelGridMapDescriptorSet },
			mainScene->vertexPosNormalBuffer.buffer, mainScene->indexPosNormalBuffer.buffer, { &vgmShader });
		voxelGridMapRenderPass.subPasses.push_back(vgmSubPass);
	}

	void createSVOCuda() {
		VkPhysicalDevice physicalDevice = FzbRenderer::globalData.physicalDevice;
		svoCuda->createSVOCuda(physicalDevice, voxelGridMap, vgmSemaphore.handle, svoCudaSemaphore.handle);
		nodePool = fzbCreateStorageBuffer(sizeof(FzbSVONode) * (8 * svoCuda->nodeBlockNum + 1), true);
		voxelValueBuffer = fzbCreateStorageBuffer(sizeof(FzbVoxelValue) * svoCuda->voxelNum, true);
		svoCuda->getSVOCuda(physicalDevice, nodePool.handle, voxelValueBuffer.handle);
	}

	void presentPrepare() override {
		if (setting.useSVO_OnlyVoxelGridMap) {
			if (!setting.useBlock) createVGMRenderPass_nonBlock();
			else createVGMRenderPass_Block();
		}
		else createSVORenderPass();
	}

	void createVGMRenderPass_nonBlock() {
		presentMaterial = FzbMaterial();
		presentShader = FzbShader(fzbGetRootPath() + "/core/SceneDivision/SVO/shaders/present");
		presentShader.createShaderVariant(&presentMaterial, componentInfo.vertexFormat);
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

		std::vector<VkSubpassDescription> subpasses;
		subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));

		VkSubpassDependency dependency = fzbCreateSubpassDependency();

		FzbRenderPassSetting renderPassSetting = { true, 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
		renderRenderPass.setting = renderPassSetting;
		renderRenderPass.createRenderPass(&attachments, subpasses, { dependency });
		renderRenderPass.createFramebuffers(true);

		FzbSubPass presentSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
			{ mainScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout }, { mainScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet },
			mainScene->vertexPosNormalBuffer.buffer, mainScene->indexPosNormalBuffer.buffer, { &presentShader });
		renderRenderPass.addSubPass(presentSubPass);
	}
	void createVGMRenderPass_Block() {
		presentMaterial = FzbMaterial();
		FzbMesh cubeMesh = FzbMesh();
		cubeMesh.instanceNum = pow(setting.voxelNum, 3);
		fzbCreateCube(cubeMesh);
		glm::vec3 voxelSize = svoUniform.voxelSize_Num;
		glm::vec3 voxelStartPos = svoUniform.voxelStartPos;
		for (int i = 0; i < 24; i += 3) {
			cubeMesh.vertices[i] = cubeMesh.vertices[i] * voxelSize.x + voxelStartPos.x;
			cubeMesh.vertices[i + 1] = cubeMesh.vertices[i + 1] * voxelSize.y + voxelStartPos.y;
			cubeMesh.vertices[i + 2] = cubeMesh.vertices[i + 2] * voxelSize.z + voxelStartPos.z;
		}
		this->componentScene.addMeshToScene(cubeMesh, presentMaterial, fzbGetRootPath() + "/core/SceneDivision/SVO/shaders/present_VGM");
		componentScene.initScene(false, false);

		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(FzbRenderer::globalData.swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment();
		VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

		std::vector<VkSubpassDescription> subpasses;
		subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));

		VkSubpassDependency dependency = fzbCreateSubpassDependency();

		FzbRenderPassSetting renderPassSetting = { true, 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
		renderRenderPass.setting = renderPassSetting;
		renderRenderPass.createRenderPass(&attachments, subpasses, { dependency });
		renderRenderPass.createFramebuffers(true);

		FzbSubPass presentSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
			{ mainScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout }, { mainScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet },
			componentScene.vertexBuffer.buffer, componentScene.indexBuffer.buffer, componentScene.sceneShaders_vector);
		renderRenderPass.addSubPass(presentSubPass);
	}

	void createSVODescriptor() {
		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_ALL };
		svoDescriptorSetLayout = fzbCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		svoDescriptorSet = fzbCreateDescriptorSet(descriptorPool, svoDescriptorSetLayout);

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

		vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, svoDescriptorWrites.size(), svoDescriptorWrites.data(), 0, nullptr);
	}
	void createSVORenderPass() {
		presentMaterial = FzbMaterial();
		FzbMesh cubeMesh = FzbMesh();
		cubeMesh.instanceNum = svoCuda->nodeBlockNum * 8 + 1;
		fzbCreateCubeWireframe(cubeMesh);
		this->componentScene.addMeshToScene(cubeMesh, presentMaterial, fzbGetRootPath() + "/core/SceneDivision/SVO/shaders/present_SVO");
		componentScene.initScene(false, false);

		presentMaterial = FzbMaterial();
		presentShader = FzbShader(fzbGetRootPath() + "/core/SceneDivision/SVO/shaders/present");
		presentShader.createShaderVariant(&presentMaterial, componentInfo.vertexFormat);
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

		std::vector<VkSubpassDescription> subpasses;
		subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));
		subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));

		VkSubpassDependency dependency = fzbCreateSubpassDependency(0, 1,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

		FzbRenderPassSetting renderPassSetting = { true, 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
		renderRenderPass.setting = renderPassSetting;
		renderRenderPass.createRenderPass(&attachments, subpasses, { dependency });
		renderRenderPass.createFramebuffers(true);

		FzbSubPass sceneSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
			{ mainScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout },
			{ mainScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet },
			mainScene->vertexPosNormalBuffer.buffer, mainScene->indexPosNormalBuffer.buffer, { &presentShader });
		renderRenderPass.addSubPass(sceneSubPass);

		FzbSubPass CubeWireframeSubPass = FzbSubPass(renderRenderPass.renderPass, 1,
			{ mainScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout, svoDescriptorSetLayout },
			{ mainScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet, svoDescriptorSet },
			componentScene.vertexBuffer.buffer, componentScene.indexBuffer.buffer, componentScene.sceneShaders_vector);
		renderRenderPass.addSubPass(CubeWireframeSubPass);
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
