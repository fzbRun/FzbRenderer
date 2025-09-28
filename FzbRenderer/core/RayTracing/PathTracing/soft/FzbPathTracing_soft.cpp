#include "./FzbPathTracing_soft.h"
#include "../../../common/FzbRenderer.h"
#include "FzbPathTracingMaterial.h"
#include "../../CUDA/FzbCollisionDetection.cuh"

#include <unordered_map>

FzbPathTracing_soft::FzbPathTracing_soft() {};
FzbPathTracing_soft::FzbPathTracing_soft(pugi::xml_node& PathTracingNode) {
	if (std::string(PathTracingNode.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
	else return;
	if (pugi::xml_node componentSettingNode = PathTracingNode.child("rendererComponentSetting")) {
		this->setting.spp = std::stoi(componentSettingNode.child("spp").attribute("value").value());
	}
	this->setting.useCudaRandom = false;

	this->componentInfo.name = FZB_RENDERER_PATH_TRACING_SOFT;
	this->componentInfo.type = FZB_RENDER_COMPONENT;

	addMainSceneInfo();
	addExtensions();

	//得到bvh组件
	if (pugi::xml_node childComponentsNode = PathTracingNode.child("childComponents")) {
		getChildComponent(childComponentsNode);
		this->bvh = std::dynamic_pointer_cast<FzbBVH>(childComponents["BVH"]);
		bvh->setting = FzbBVHSetting{ BVH_MAX_DEPTH, true, true };
	}
}

void FzbPathTracing_soft::init()  {
	FzbFeatureComponent_LoopRender::init();
	presentPrepare();
}
FzbSemaphore FzbPathTracing_soft::render(uint32_t imageIndex, FzbSemaphore startSemaphore, VkFence fence) {
	VkCommandBuffer commandBuffer = commandBuffers[0];
	vkResetCommandBuffer(commandBuffer, 0);
	fzbBeginCommandBuffer(commandBuffer);

	pathTracingCUDA->pathTracing(startSemaphore.handle);
	renderRenderPass.render(commandBuffer, imageIndex);

	// 设置内存屏障，确保所有颜色附件写入完成
	//VkMemoryBarrier memoryBarrier = {};
	//memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	//memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
	//memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	//vkCmdPipelineBarrier(
	//	commandBuffer,
	//	VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, // 在颜色附件输出阶段之后
	//	VK_PIPELINE_STAGE_TRANSFER_BIT,                // 在传输阶段之前
	//	0,
	//	1, &memoryBarrier,
	//	0, nullptr,
	//	0, nullptr
	//);
	//vkCmdFillBuffer(commandBuffer, pathTracingResultBuffer.buffer, 0, pathTracingResultBuffer.size, 0);

	std::vector<VkSemaphore> waitSemaphores = { pathTracingFinishedSemphore.semaphore };
	std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
	fzbSubmitCommandBuffer(commandBuffer, waitSemaphores, waitStages, { renderFinishedSemaphore.semaphore }, fence);

	return renderFinishedSemaphore;
};
void FzbPathTracing_soft::clean() {
	FzbFeatureComponent_LoopRender::clean();
	settingBuffer.clean();
	pathTracingResultBuffer.clean();
	presentSourceManager.clean();
	pathTracingFinishedSemphore.clean();
	pathTracingCUDA->clean();
	for (int i = 0; i < this->sceneTextures.size(); ++i) this->sceneTextures[i].clean();
	if (descriptorSetLayout) vkDestroyDescriptorSetLayout(FzbRenderer::globalData.logicalDevice, this->descriptorSetLayout, nullptr);
};

void FzbPathTracing_soft::addMainSceneInfo() {
	FzbRenderer::useImageAvailableSemaphoreHandle = true;
	FzbRenderer::globalData.mainScene.vertexFormat_allMesh.mergeUpward(FzbVertexFormat());
	FzbRenderer::globalData.mainScene.useMaterialSource = false;
}
void FzbPathTracing_soft::addExtensions() {};

void FzbPathTracing_soft::presentPrepare() {
	fzbCreateCommandBuffers(1);
	FzbPathTracingCudaSourceSet sourceSet = createSource();
	pathTracingCUDA = std::make_unique<FzbPathTracingCuda>(sourceSet);
	createDescriptor();
	createRenderPass();
};

FzbPathTracingCudaSourceSet FzbPathTracing_soft::createSource() {
	std::unordered_map<std::string, int> sceneImagePaths;
	for (auto& materialPair : this->mainScene->sceneMaterials) {
		FzbMaterial& material = materialPair.second;
		FzbPathTracingMaterialUniformObject materialUniformObject = createInitialMaterialUniformObject();
		materialUniformObject.materialType = fzbGetPathTracingMaterialType(material.type);

		for (auto& texturePair : material.properties.textureProperties) {
			FzbTexture& texture = texturePair.second;
			if (!sceneImagePaths.count(texture.path)) {
				FzbImage image;
				std::string texturePathFromModel = this->mainScene->scenePath + "/" + texture.path;
				image.texturePath = texturePathFromModel.c_str();
				image.filter = texturePair.second.filter;
				image.layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR;
				image.UseExternal = true;
				image.initImage();
				this->sceneTextures.push_back(image);
				sceneImagePaths.insert({ texture.path,this->sceneTextures.size() - 1 });
			}
			texture.image = &this->sceneTextures[sceneImagePaths[texture.path]];
			materialUniformObject.textureIndex[material.getMaterialAttributeIndex(texturePair.first)] = sceneImagePaths[texture.path];
		}

		for (auto& numberPropertyPair : material.properties.numberProperties) {
			FzbNumberProperty& numberProperty = numberPropertyPair.second;
			if (numberPropertyPair.first == "emissive") materialUniformObject.emissive = numberProperty.value;
			else materialUniformObject.numberAttribute[material.getMaterialAttributeIndex(numberPropertyPair.first)] = numberProperty.value;
		}
		this->sceneMaterialInfoArray.push_back(materialUniformObject);
	}
	//将mainScene中的mesh与material进行绑定

	VkExtent2D resolution = FzbRenderer::globalData.getResolution();
	FzbPathTracingSettingUniformObject settingUniformObject = { resolution.width, resolution.height };
	settingBuffer = fzbCreateUniformBuffer(sizeof(FzbPathTracingSettingUniformObject));
	memcpy(settingBuffer.mapped, &settingUniformObject, sizeof(FzbPathTracingSettingUniformObject));

	this->pathTracingResultBuffer = fzbCreateStorageBuffer(resolution.width * resolution.height * sizeof(float4), true);

	pathTracingFinishedSemphore = FzbSemaphore(true);

	FzbPathTracingCudaSourceSet sourceSet;
	sourceSet.setting = this->setting;
	sourceSet.pathTracingResultBuffer = this->pathTracingResultBuffer;
	sourceSet.pathTracingFinishedSemphore = this->pathTracingFinishedSemphore;
	sourceSet.sceneVertices = FzbRenderer::globalData.mainScene.vertexBuffer;
	sourceSet.sceneTextures = this->sceneTextures;
	sourceSet.sceneMaterialInfoArray = this->sceneMaterialInfoArray;
	//sourceSet.bvhSemaphoreHandle = this->bvh->bvhCudaSemaphore.handle;
	sourceSet.bvhNodeCount = this->bvh->bvhCuda->triangleNum * 2 - 1;
	sourceSet.bvhNodeArray = this->bvh->bvhCuda->bvhNodeArray;
	sourceSet.bvhTriangleInfoArray = this->bvh->bvhCuda->bvhTriangleInfoArray;

	for (int i = 0; i < mainScene->sceneLights.size(); ++i) {
		FzbLight& light = mainScene->sceneLights[i];
		if (light.type == FZB_POINT) {
			++sourceSet.pointLightCount;
			sourceSet.pointLightInfoArray.push_back({ glm::vec4(light.position, 0.0f), glm::vec4(light.strength, 0.0f) });
		}
		else if (light.type == FZB_AREA) {
			++sourceSet.areaLightCount;
			FzbRayTracingAreaLight areaLight;
			areaLight.worldPos = glm::vec4(light.position, 0.0f);
			areaLight.normal = glm::vec4(light.normal, 0.0f);
			areaLight.radiance = glm::vec4(light.strength, 0.0f);
			areaLight.edge0 = glm::vec4(light.edge0, 0.0f);
			areaLight.edge1 = glm::vec4(light.edge1, 0.0f);
			areaLight.area = light.area;
			sourceSet.areaLightInfoArray.push_back(areaLight);
		}
	}
	return sourceSet;
}
void FzbPathTracing_soft::createImages() {
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

void FzbPathTracing_soft::createDescriptor() {
	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 });
	this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);

	std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
	std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_FRAGMENT_BIT };
	descriptorSetLayout = fzbCreateDescriptLayout(4, descriptorTypes, descriptorShaderFlags);
	descriptorSet = fzbCreateDescriptorSet(descriptorPool, descriptorSetLayout);

	std::array<VkWriteDescriptorSet, 4> pathTracingDescriptorWrites{};
	VkDescriptorBufferInfo pathTracingSettingBufferInfo{};
	pathTracingSettingBufferInfo.buffer = settingBuffer.buffer;
	pathTracingSettingBufferInfo.offset = 0;
	pathTracingSettingBufferInfo.range = settingBuffer.size;
	pathTracingDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	pathTracingDescriptorWrites[0].dstSet = descriptorSet;
	pathTracingDescriptorWrites[0].dstBinding = 0;
	pathTracingDescriptorWrites[0].dstArrayElement = 0;
	pathTracingDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	pathTracingDescriptorWrites[0].descriptorCount = 1;
	pathTracingDescriptorWrites[0].pBufferInfo = &pathTracingSettingBufferInfo;

	VkDescriptorBufferInfo pathTracingResultBufferInfo{};
	pathTracingResultBufferInfo.buffer = pathTracingResultBuffer.buffer;
	pathTracingResultBufferInfo.offset = 0;
	pathTracingResultBufferInfo.range = pathTracingResultBuffer.size;
	pathTracingDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	pathTracingDescriptorWrites[1].dstSet = descriptorSet;
	pathTracingDescriptorWrites[1].dstBinding = 1;
	pathTracingDescriptorWrites[1].dstArrayElement = 0;
	pathTracingDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	pathTracingDescriptorWrites[1].descriptorCount = 1;
	pathTracingDescriptorWrites[1].pBufferInfo = &pathTracingResultBufferInfo;

	VkDescriptorBufferInfo bvhNodeBufferInfo{};
	bvhNodeBufferInfo.buffer = bvh->bvhNodeArray.buffer;
	bvhNodeBufferInfo.offset = 0;
	bvhNodeBufferInfo.range = bvh->bvhNodeArray.size;
	pathTracingDescriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	pathTracingDescriptorWrites[2].dstSet = descriptorSet;
	pathTracingDescriptorWrites[2].dstBinding = 2;
	pathTracingDescriptorWrites[2].dstArrayElement = 0;
	pathTracingDescriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	pathTracingDescriptorWrites[2].descriptorCount = 1;
	pathTracingDescriptorWrites[2].pBufferInfo = &bvhNodeBufferInfo;

	VkDescriptorBufferInfo bvhTriangleBufferInfo{};
	bvhTriangleBufferInfo.buffer = bvh->bvhTriangleInfoArray.buffer;
	bvhTriangleBufferInfo.offset = 0;
	bvhTriangleBufferInfo.range = bvh->bvhTriangleInfoArray.size;
	pathTracingDescriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	pathTracingDescriptorWrites[3].dstSet = descriptorSet;
	pathTracingDescriptorWrites[3].dstBinding = 3;
	pathTracingDescriptorWrites[3].dstArrayElement = 0;
	pathTracingDescriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	pathTracingDescriptorWrites[3].descriptorCount = 1;
	pathTracingDescriptorWrites[3].pBufferInfo = &bvhTriangleBufferInfo;

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, pathTracingDescriptorWrites.size(), pathTracingDescriptorWrites.data(), 0, nullptr);
}
void FzbPathTracing_soft::createRenderPass() {
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