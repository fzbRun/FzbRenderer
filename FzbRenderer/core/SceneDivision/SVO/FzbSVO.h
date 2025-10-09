#pragma once

#include "./CUDA/createSVO.cuh"
#include "../../common/FzbCommon.h"
#include "../../common/FzbComponent/FzbFeatureComponent.h"
#include "../../common/FzbRasterizationRender/FzbRasterizationSourceManager.h"

#ifndef SVO_H	//Sparse voxel octree
#define SVO_H

struct SVOUniform {
	glm::mat4 VP[3];
	glm::vec4 voxelSize_Num;
	glm::vec4 voxelStartPos;
};

struct FzbSVOSetting_debug {
	bool useSVO_OnlyVoxelGridMap = false;
	bool useSwizzle = false;
	bool useConservativeRasterization = false;

	bool useBlock = false;

	bool useCube;	//是否将当前场景当作一个立方体来存储
	int voxelNum = 64;
};
struct FzbSVO_Debug : public FzbFeatureComponent_LoopRender {
public:

	FzbSVO_Debug();
	FzbSVO_Debug(pugi::xml_node& SVONode);

	void init() override;

	FzbSemaphore render(uint32_t imageIndex, FzbSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) override;
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
	void clean() override;

private:
	FzbSVOSetting_debug setting;
	FzbRasterizationSourceManager vgmSourceManager;
	FzbRasterizationSourceManager presentSourceManager;

	FzbImage voxelGridMap;
	VkDescriptorSetLayout voxelGridMapDescriptorSetLayout = nullptr;
	VkDescriptorSet voxelGridMapDescriptorSet;

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

	//FzbShader vgmShader;
	//FzbMaterial vgmMaterial;
	//FzbShader presentShader;
	//FzbMaterial presentMaterial;
	//FzbMaterial presentSVOMaterial;

	void addMainSceneInfo() override;
	void addExtensions() override;

	void createImages() override;

	void createVoxelGridMap();
	void createVoxelGridMapBuffer();
	void initVoxelGridMap();
	void createDescriptorPool();
	void createVoxelGridMapDescriptor();
	void createVoxelGridMapSyncObjects();
	void createVGMRenderPass();

	void prepocessClean() override;

	void createSVOCuda();

	void presentPrepare() override;

	void createVGMRenderPass_nonBlock();
	void createVGMRenderPass_Block();

	void createSVODescriptor();
	void createSVORenderPass();

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
