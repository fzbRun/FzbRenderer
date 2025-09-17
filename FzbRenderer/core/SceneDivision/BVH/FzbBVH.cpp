#include "./FzbBVH.h"
#include "../../common/FzbRenderer.h"

FzbBVH::FzbBVH() {
	addMainSceneInfo();
	addExtensions();
};
FzbBVH::FzbBVH(pugi::xml_node& BVHNode) {
	addMainSceneInfo();
	addExtensions();
	if (pugi::xml_node childComponents = BVHNode.child("featureComponents")) getChildComponent(childComponents);
}
void FzbBVH::init() {
	FzbFeatureComponent_PreProcess::init();
	bvhCuda = std::make_unique<BVHCuda>();
	createBVH();
}
void FzbBVH::clean() {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;

	uniformBuffer.clean();
	bvhNodeArray.clean();
	bvhTriangleInfoArray.clean();

	bvhCudaSemaphore.clean();
	bvhCuda->clean();

	if (descriptorPool) vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
	if (descriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
}

void FzbBVH::addMainSceneInfo() {
	FzbRenderer::globalData.mainScene.vertexFormat_allMesh.mergeUpward(FzbVertexFormat(true));
	FzbRenderer::globalData.mainScene.useVertexBufferHandle = true;
}
void FzbBVH::addExtensions() {
	FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
	FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
};
void FzbBVH::createBVH() {
	VkPhysicalDevice physicalDevice = FzbRenderer::globalData.physicalDevice;
	bvhCudaSemaphore = FzbSemaphore(true);		//当bvh创建完成后唤醒

	//bvhCuda->createBvhCuda_recursion(physicalDevice, mainComponentScene, bvhCudaSemaphore.handle, setting);
	bvhCuda->createBvhCuda_noRecursion(physicalDevice, mainScene, bvhCudaSemaphore.handle, setting);

	if (setting.useRaserization) {
		bvhNodeArray = fzbCreateStorageBuffer(sizeof(FzbBvhNode) * (bvhCuda->triangleNum * 2 - 1), true);
		bvhTriangleInfoArray = fzbCreateStorageBuffer(sizeof(FzbBvhNodeTriangleInfo) * bvhCuda->triangleNum, true);
		bvhCuda->getBvhCuda(physicalDevice, bvhNodeArray.handle, bvhTriangleInfoArray.handle);
		bvhCuda->clean();
	}
}

void FzbBVH::createBuffer() {
	uniformBuffer = fzbCreateUniformBuffer(sizeof(FzbBVHPresentUniform));
	FzbBVHPresentUniform uniform;
	uniform.nodeIndex = 0;
	memcpy(uniformBuffer.mapped, &uniform, sizeof(FzbBVHPresentUniform));
}
void FzbBVH::createDescriptor() {
	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
	this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);

	std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
	std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_ALL };
	descriptorSetLayout = fzbCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
	descriptorSet = fzbCreateDescriptorSet(descriptorPool, descriptorSetLayout);

	std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
	VkDescriptorBufferInfo uniformBufferInfo{};
	uniformBufferInfo.buffer = uniformBuffer.buffer;
	uniformBufferInfo.offset = 0;
	uniformBufferInfo.range = sizeof(FzbBVHPresentUniform);
	descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrites[0].dstSet = descriptorSet;
	descriptorWrites[0].dstBinding = 0;
	descriptorWrites[0].dstArrayElement = 0;
	descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorWrites[0].descriptorCount = 1;
	descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

	VkDescriptorBufferInfo nodeBufferInfo{};
	nodeBufferInfo.buffer = bvhNodeArray.buffer;
	nodeBufferInfo.offset = 0;
	nodeBufferInfo.range = sizeof(FzbBvhNode) * (bvhCuda->triangleNum * 2 - 1);
	descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrites[1].dstSet = descriptorSet;
	descriptorWrites[1].dstBinding = 1;
	descriptorWrites[1].dstArrayElement = 0;
	descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorWrites[1].descriptorCount = 1;
	descriptorWrites[1].pBufferInfo = &nodeBufferInfo;

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}
//----------------------------------------------------BVH Debug-----------------------------------------------
FzbBVH_Debug::FzbBVH_Debug() {};
FzbBVH_Debug::FzbBVH_Debug(pugi::xml_node& BVHNode) {
	if (std::string(BVHNode.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
	else return;

	this->componentInfo.name = FZB_FEATURE_COMPONENT_BVH_DEBUG;
	this->componentInfo.type = FZB_LOOPRENDER_FEATURE_COMPONENT;
	//this->componentInfo.useMainSceneBufferHandle = { true, false, false };	//需要只有pos格式的顶点buffer和索引buffer，用来创建bvh

	addMainSceneInfo();
	addExtensions();
}

void FzbBVH_Debug::init() {
	FzbFeatureComponent_LoopRender::init();
	bvhCuda = std::make_unique<BVHCuda>();
	createBVH();
	createBuffer();
	createDescriptor();
	presentPrepare();
}

VkSemaphore FzbBVH_Debug::render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence) {
	VkCommandBuffer commandBuffer = commandBuffers[0];
	vkResetCommandBuffer(commandBuffer, 0);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("failed to begin recording command buffer!");
	}

	renderRenderPass.render(commandBuffer, imageIndex);

	std::vector<VkSemaphore> waitSemaphores = { startSemaphore };
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

void FzbBVH_Debug::clean() {
	FzbFeatureComponent_LoopRender::clean();
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;

	depthMap.clean();

	uniformBuffer.clean();
	bvhNodeArray.clean();
	bvhTriangleInfoArray.clean();

	bvhCudaSemaphore.clean();

	presentSourceManager.clean();

	vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
}

//虽然创建bvh是预处理，但是实际上用到的适合loopRender相同的顶点数据，所以可以直接使用
void FzbBVH_Debug::addMainSceneInfo() {
	FzbRenderer::globalData.mainScene.vertexFormat_allMesh.mergeUpward(FzbVertexFormat(true));
	FzbRenderer::globalData.mainScene.useVertexBufferHandle = true;
}
void FzbBVH_Debug::addExtensions() {
	FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
	FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
};

void FzbBVH_Debug::presentPrepare() {
	createRenderPass();
}

void FzbBVH_Debug::createBVH() {
	bvhCudaSemaphore = FzbSemaphore(true);		//当bvh创建完成后唤醒

	//bvhCuda->createBvhCuda_recursion(physicalDevice, mainComponentScene, bvhCudaSemaphore.handle, setting);
	bvhCuda->createBvhCuda_noRecursion(FzbRenderer::globalData.physicalDevice, mainScene, bvhCudaSemaphore.handle, setting);

	bvhNodeArray = fzbCreateStorageBuffer(sizeof(FzbBvhNode) * (bvhCuda->triangleNum * 2 - 1), true);
	bvhTriangleInfoArray = fzbCreateStorageBuffer(sizeof(FzbBvhNodeTriangleInfo) * bvhCuda->triangleNum, true);

	bvhCuda->getBvhCuda(FzbRenderer::globalData.physicalDevice, bvhNodeArray.handle, bvhTriangleInfoArray.handle);

	bvhCuda->clean();
}

void FzbBVH_Debug::createBuffer() {
	fzbCreateCommandBuffers(1);

	uniformBuffer = fzbCreateUniformBuffer(sizeof(FzbBVHPresentUniform));
	FzbBVHPresentUniform uniform;
	uniform.nodeIndex = 0;
	memcpy(uniformBuffer.mapped, &uniform, sizeof(FzbBVHPresentUniform));
}

void FzbBVH_Debug::createDescriptor() {
	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
	this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);

	std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
	std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_ALL };
	descriptorSetLayout = fzbCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
	descriptorSet = fzbCreateDescriptorSet(descriptorPool, descriptorSetLayout);

	std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
	VkDescriptorBufferInfo uniformBufferInfo{};
	uniformBufferInfo.buffer = uniformBuffer.buffer;
	uniformBufferInfo.offset = 0;
	uniformBufferInfo.range = sizeof(FzbBVHPresentUniform);
	descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrites[0].dstSet = descriptorSet;
	descriptorWrites[0].dstBinding = 0;
	descriptorWrites[0].dstArrayElement = 0;
	descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorWrites[0].descriptorCount = 1;
	descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

	VkDescriptorBufferInfo nodeBufferInfo{};
	nodeBufferInfo.buffer = bvhNodeArray.buffer;
	nodeBufferInfo.offset = 0;
	nodeBufferInfo.range = sizeof(FzbBvhNode) * (bvhCuda->triangleNum * 2 - 1);
	descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrites[1].dstSet = descriptorSet;
	descriptorWrites[1].dstBinding = 1;
	descriptorWrites[1].dstArrayElement = 0;
	descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorWrites[1].descriptorCount = 1;
	descriptorWrites[1].pBufferInfo = &nodeBufferInfo;

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}
void FzbBVH_Debug::createImages() {
	VkExtent2D swapChainExtent = FzbRenderer::globalData.swapChainExtent;

	depthMap = {};
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
void FzbBVH_Debug::createRenderPass() {
	FzbRenderer::globalData.mainScene.createCameraAndLightDescriptor();

	this->presentSourceManager.addMeshMaterial(FzbRenderer::globalData.mainScene.sceneMeshSet, FzbMaterial("present", "present"));
	FzbShaderInfo presentShaderInfo = { "/core/SceneDivision/BVH/shaders/present" };
	this->presentSourceManager.addSource({ {"present", presentShaderInfo} });

	VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(FzbRenderer::globalData.swapChainImageFormat);
	VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment();
	VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

	VkSubpassDescription subpass = fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef);
	VkSubpassDependency dependency = fzbCreateSubpassDependency();

	FzbRenderPassSetting renderPassSetting = { true, 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
	renderRenderPass.setting = renderPassSetting;
	renderRenderPass.createRenderPass(&attachments, { subpass }, { dependency });
	renderRenderPass.createFramebuffers(true);

	FzbSubPass presentSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
		{ mainScene->cameraAndLightsDescriptorSetLayout, descriptorSetLayout }, { mainScene->cameraAndLightsDescriptorSet, descriptorSet },
		mainScene->vertexBuffer.buffer, mainScene->indexBuffer.buffer, this->presentSourceManager.shaders_vector);
	renderRenderPass.subPasses.push_back(presentSubPass);
}