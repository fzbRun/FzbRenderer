#include "./FzbPathTracing_soft.h"
#include "../../../common/FzbRenderer.h"
#include "../../common/FzbRayTracingMaterial.h"
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
		this->rayTracingSourceManager.bvh = std::dynamic_pointer_cast<FzbBVH>(childComponents["BVH"]);
		this->rayTracingSourceManager.bvh->setting = FzbBVHSetting{ BVH_MAX_DEPTH, true, true };
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

	std::vector<VkSemaphore> waitSemaphores = { rayTracingSourceManager.rayTracingFinishedSemphore.semaphore };
	std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
	fzbSubmitCommandBuffer(commandBuffer, waitSemaphores, waitStages, { renderFinishedSemaphore.semaphore }, fence);

	return renderFinishedSemaphore;
};
void FzbPathTracing_soft::clean() {
	FzbFeatureComponent_LoopRender::clean();
	settingBuffer.clean();
	rayTracingSourceManager.clean();
	presentSourceManager.clean();
	pathTracingCUDA->clean();
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
	rayTracingSourceManager.createSource();
	pathTracingCUDA = std::make_unique<FzbPathTracingCuda>(rayTracingSourceManager.sourceManagerCuda, this->setting);
	createBuffer();
	createDescriptor();
	createRenderPass();
};

void FzbPathTracing_soft::createBuffer() {
	VkExtent2D resolution = FzbRenderer::globalData.getResolution();
	FzbPathTracingSettingUniformObject settingUniformObject = { resolution.width, resolution.height };
	settingBuffer = fzbCreateUniformBuffer(sizeof(FzbPathTracingSettingUniformObject));
	memcpy(settingBuffer.mapped, &settingUniformObject, sizeof(FzbPathTracingSettingUniformObject));
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
	pathTracingResultBufferInfo.buffer = rayTracingSourceManager.rayTracingResultBuffer.buffer;
	pathTracingResultBufferInfo.offset = 0;
	pathTracingResultBufferInfo.range = rayTracingSourceManager.rayTracingResultBuffer.size;
	pathTracingDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	pathTracingDescriptorWrites[1].dstSet = descriptorSet;
	pathTracingDescriptorWrites[1].dstBinding = 1;
	pathTracingDescriptorWrites[1].dstArrayElement = 0;
	pathTracingDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	pathTracingDescriptorWrites[1].descriptorCount = 1;
	pathTracingDescriptorWrites[1].pBufferInfo = &pathTracingResultBufferInfo;

	VkDescriptorBufferInfo bvhNodeBufferInfo{};
	bvhNodeBufferInfo.buffer = rayTracingSourceManager.bvh->bvhNodeArray.buffer;
	bvhNodeBufferInfo.offset = 0;
	bvhNodeBufferInfo.range = rayTracingSourceManager.bvh->bvhNodeArray.size;
	pathTracingDescriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	pathTracingDescriptorWrites[2].dstSet = descriptorSet;
	pathTracingDescriptorWrites[2].dstBinding = 2;
	pathTracingDescriptorWrites[2].dstArrayElement = 0;
	pathTracingDescriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	pathTracingDescriptorWrites[2].descriptorCount = 1;
	pathTracingDescriptorWrites[2].pBufferInfo = &bvhNodeBufferInfo;

	VkDescriptorBufferInfo bvhTriangleBufferInfo{};
	bvhTriangleBufferInfo.buffer = rayTracingSourceManager.bvh->bvhTriangleInfoArray.buffer;
	bvhTriangleBufferInfo.offset = 0;
	bvhTriangleBufferInfo.range = rayTracingSourceManager.bvh->bvhTriangleInfoArray.size;
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