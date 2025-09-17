#pragma once

#include "../../../common/FzbCommon.h"
#include "../../../common/FzbComponent/FzbFeatureComponent.h"
#include "../../../common/FzbRenderer.h"
#include "CUDA/PathTracing_CUDA.cuh"
#include "FzbPathTracingMaterial.h"
#include <unordered_map>
#include "../../../SceneDivision/BVH/FzbBVH.h"
#include "../../CUDA/FzbCollisionDetection.cuh"

#ifndef FZB_PATH_TRACING_H
#define FZB_PATH_TRACING_H

struct FzbPathTracingSettingUniformObject {
	uint32_t screenWidth;
	uint32_t screenHeight;
};

struct FzbPathTracing_soft : public FzbFeatureComponent_LoopRender {
public:
	FzbPathTracing_soft() {};
	FzbPathTracing_soft(pugi::xml_node& PathTracingNode) {
		if (std::string(PathTracingNode.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
		else return;

		this->componentInfo.name = FZB_RENDERER_PATH_TRACING_SOFT;
		this->componentInfo.type = FZB_RENDER_COMPONENT;

		addMainSceneInfo();
		addExtensions();

		//得到bvh组件
		if (pugi::xml_node childComponentsNode = PathTracingNode.child("childComponents")) {
			getChildComponent(childComponentsNode);
			this->bvh = std::dynamic_pointer_cast<FzbBVH>(childComponents["BVH"]);
			bvh->setting = FzbBVHSetting{ BVH_MAX_DEPTH, false, true };
		}
	}

	void init() override {
		FzbFeatureComponent_LoopRender::init();
		presentPrepare();
	}
	VkSemaphore render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) {
		VkCommandBuffer commandBuffer = commandBuffers[0];
		vkResetCommandBuffer(commandBuffer, 0);
		fzbBeginCommandBuffer(commandBuffer);

		VkExtent2D resolution = FzbRenderer::globalData.getResolution();
		//pathTracingCUDA->pathTracing(startSemaphore, resolution.width, resolution.height);
		//VkExtent3D copyExtent = { resolution.width, resolution.height , 0.0f };
		//fzbCopyImageToImage(commandBuffer, pathTracingResultMap.image, FzbRenderer::globalData.swapChainImages[imageIndex], copyExtent);
		renderRenderPass.render(commandBuffer, imageIndex);
		
		std::vector<VkSemaphore> waitSemaphores = { startSemaphore };	//, pathTracingFinishedSemphore.semaphore
		std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
		fzbSubmitCommandBuffer(commandBuffer, waitSemaphores, waitStages, { renderFinishedSemaphore.semaphore }, fence);

		return renderFinishedSemaphore.semaphore;
	};
	void clean() override {
		FzbFeatureComponent_LoopRender::clean();
		settingBuffer.clean();
		pathTracingResultBuffer.clean();
		presentSourceManager.clean();
		pathTracingFinishedSemphore.clean();
		//pathTracingCUDA->clean();
		for (int i = 0; i < this->sceneTextures.size(); ++i) this->sceneTextures[i].clean();
		if (descriptorSetLayout) vkDestroyDescriptorSetLayout(FzbRenderer::globalData.logicalDevice, this->descriptorSetLayout, nullptr);
	};

private:
	FzbPathTracingSetting setting;
	FzbBuffer settingBuffer;
	FzbRasterizationSourceManager presentSourceManager;
	//FzbImage pathTracingResultMap;
	FzbBuffer pathTracingResultBuffer;
	FzbSemaphore pathTracingFinishedSemphore;
	std::unique_ptr<FzbPathTracingCuda> pathTracingCUDA;

	//我们不让mainScene去创建materialSource，而是我们自己创造，自己维护
	std::vector<FzbImage> sceneTextures;
	std::vector<FzbPathTracingMaterialUniformObject> sceneMaterialInfoArray;

	std::shared_ptr<FzbBVH> bvh;

	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	VkDescriptorSet descriptorSet;

	void addMainSceneInfo() override {
		FzbRenderer::globalData.mainScene.vertexFormat_allMesh.mergeUpward(FzbVertexFormat(true));
		FzbRenderer::globalData.mainScene.useMaterialSource = false;
	}
	void addExtensions() override {};

	void presentPrepare() override {
		fzbCreateCommandBuffers(1);
		createSource();
		//pathTracingCUDA = std::make_unique<FzbPathTracingCuda>(FzbRenderer::globalData.physicalDevice, mainScene,
		//	setting, pathTracingResultBuffer, pathTracingFinishedSemphore,
		//	sceneTextures, sceneMaterialInfoArray);
		createDescriptor();
		createRenderPass();
	};

	//与raserizationSourceManager的功能相同，这里需要创建各种后续渲染所用资源
	void createSource() {
		std::unordered_set<std::string> sceneImagePaths;
		for (auto& materialPair : this->mainScene->sceneMaterials) {
			FzbMaterial& material = materialPair.second;

			FzbPathTracingMaterialUniformObject materialUniformObject;
			materialUniformObject.materialType = getMaterialType(material.type);
			int textureCount = 0;
			int numberAttributeCount = 0;

			for (auto& texturePair : material.properties.textureProperties) {
				FzbTexture& texture = texturePair.second;
				if (sceneImagePaths.count(texture.path)) continue;
				FzbImage image;
				std::string texturePathFromModel = this->mainScene->scenePath  + "/" + texture.path;
				image.texturePath = texturePathFromModel.c_str();
				image.filter = texturePair.second.filter;
				image.layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR;
				image.UseExternal = true;
				image.initImage();

				sceneImagePaths.insert(texture.path);
				this->sceneTextures.push_back(image);
				texture.image = &this->sceneTextures[this->sceneTextures.size() - 1];
				materialUniformObject.textureIndex[textureCount++] = this->sceneTextures.size() - 1;
			}

			for (auto& numberPropertyPair : material.properties.numberProperties) {
				FzbNumberProperty& numberProperty = numberPropertyPair.second;
				materialUniformObject.numberAttribute[numberAttributeCount++] = numberProperty.value;
			}
			this->sceneMaterialInfoArray.push_back(materialUniformObject);
		}

		VkExtent2D resolution = FzbRenderer::globalData.getResolution();
		FzbPathTracingSettingUniformObject settingUniformObject = { resolution.width, resolution.height };
		settingBuffer = fzbCreateUniformBuffer(sizeof(FzbPathTracingSettingUniformObject));
		memcpy(settingBuffer.mapped, &settingUniformObject, sizeof(FzbPathTracingSettingUniformObject));
		
		this->pathTracingResultBuffer = fzbCreateStorageBuffer(resolution.width * resolution.height * sizeof(float4), true);
	}
	void createImages() override {
		//VkExtent2D resolution = FzbRenderer::globalData.getResolution();
		//
		//pathTracingResultMap = {};
		//pathTracingResultMap.width = resolution.width;
		//pathTracingResultMap.height = resolution.height;
		//pathTracingResultMap.type = VK_IMAGE_TYPE_2D;
		//pathTracingResultMap.viewType = VK_IMAGE_VIEW_TYPE_2D;
		//pathTracingResultMap.format = FzbRenderer::globalData.swapChainImageFormat;		//SRGB
		//pathTracingResultMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
		//pathTracingResultMap.UseExternal = true;
		//pathTracingResultMap.initImage();
		//pathTracingResultMap.transitionImageLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1);
		//
		//frameBufferImages.push_back(&pathTracingResultMap);
	}

	void createDescriptor() {
		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 });
		this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);

		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_FRAGMENT_BIT };
		descriptorSetLayout = fzbCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		descriptorSet = fzbCreateDescriptorSet(descriptorPool, descriptorSetLayout);

		std::array<VkWriteDescriptorSet, 2> svoDescriptorWrites{};
		VkDescriptorBufferInfo pathTracingSettingBufferInfo{};
		pathTracingSettingBufferInfo.buffer = settingBuffer.buffer;
		pathTracingSettingBufferInfo.offset = 0;
		pathTracingSettingBufferInfo.range = sizeof(FzbPathTracingSettingUniformObject);
		svoDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		svoDescriptorWrites[0].dstSet = descriptorSet;
		svoDescriptorWrites[0].dstBinding = 0;
		svoDescriptorWrites[0].dstArrayElement = 0;
		svoDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		svoDescriptorWrites[0].descriptorCount = 1;
		svoDescriptorWrites[0].pBufferInfo = &pathTracingSettingBufferInfo;

		VkExtent2D resolution = FzbRenderer::globalData.getResolution();
		VkDescriptorBufferInfo pathTracingResultBufferInfo{};
		pathTracingResultBufferInfo.buffer = pathTracingResultBuffer.buffer;
		pathTracingResultBufferInfo.offset = 0;
		pathTracingResultBufferInfo.range = sizeof(float4) * resolution.width * resolution.height;
		svoDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		svoDescriptorWrites[1].dstSet = descriptorSet;
		svoDescriptorWrites[1].dstBinding = 1;
		svoDescriptorWrites[1].dstArrayElement = 0;
		svoDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		svoDescriptorWrites[1].descriptorCount = 1;
		svoDescriptorWrites[1].pBufferInfo = &pathTracingResultBufferInfo;

		vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, svoDescriptorWrites.size(), svoDescriptorWrites.data(), 0, nullptr);
	}
	//将cuda得到的pathTracing结果的buffer复制到帧缓冲中
	void createRenderPass() {
		this->presentSourceManager.createCanvas(FzbMaterial("present", "present"));
		FzbShaderInfo shaderInfo = { "/core/RayTracing/PathTracing/soft/shaders" };
		this->presentSourceManager.addSource({ {"present", shaderInfo} });

		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(FzbRenderer::globalData.swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve };

		std::vector<VkSubpassDescription> subpasses;
		subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef));

		VkSubpassDependency dependency = fzbCreateSubpassDependency();

		FzbRenderPassSetting renderPassSetting = { false , 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
		renderRenderPass.setting = renderPassSetting;
		renderRenderPass.createRenderPass(&attachments, subpasses, { dependency });
		renderRenderPass.createFramebuffers(true);

		FzbSubPass presentSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
			{ descriptorSetLayout }, { descriptorSet },
			nullptr, nullptr, this->presentSourceManager.shaders_vector);
		renderRenderPass.addSubPass(presentSubPass);
	}
};

#endif