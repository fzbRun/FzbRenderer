#include "./FzbForwardRender.h"

FzbForwardRender::FzbForwardRender() {};
FzbForwardRender::FzbForwardRender(pugi::xml_node& ForwardRenderNode) {
	if (std::string(ForwardRenderNode.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
	else return;

	this->componentInfo.name = FZB_RENDERER_FORWARD;
	this->componentInfo.type = FZB_RENDER_COMPONENT;

	addMainSceneInfo();
	addExtensions();
};

void FzbForwardRender::init() {
	FzbFeatureComponent_LoopRender::init();
	presentPrepare();
}

FzbSemaphore FzbForwardRender::render(uint32_t imageIndex, FzbSemaphore startSemaphore, VkFence fence) {
	VkCommandBuffer commandBuffer = commandBuffers[0];
	vkResetCommandBuffer(commandBuffer, 0);
	fzbBeginCommandBuffer(commandBuffer);

	renderRenderPass.render(commandBuffer, imageIndex);

	std::vector<VkSemaphore> waitSemaphores = { startSemaphore.semaphore };
	std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
	fzbSubmitCommandBuffer(commandBuffer, waitSemaphores, waitStages, { renderFinishedSemaphore.semaphore }, fence);

	return renderFinishedSemaphore;
}

void FzbForwardRender::addMainSceneInfo() {
	FzbRenderer::globalData.mainScene.vertexFormat_allMesh.mergeUpward(FzbVertexFormat(true));
	//FzbRenderer::globalData.mainScene.useVertexBufferHandle_looprender = { false, false, false };
}
void FzbForwardRender::addExtensions() {};

void FzbForwardRender::presentPrepare() {
	fzbCreateCommandBuffers(1);
	FzbRenderer::globalData.mainScene.createCameraAndLightDescriptor();

	this->sourceManager.addMeshMaterial(FzbRenderer::globalData.mainScene.sceneMeshSet);
	FzbShaderInfo diffuseShaderInfo = { "/core/Materials/Diffuse/shaders/forwardRender" };
	FzbShaderInfo roughconductorShaderInfo = { "/core/Materials/roughconductor/shaders/forwardRender" };
	this->sourceManager.addSource({ {"diffuse", diffuseShaderInfo }, { "roughconductor", roughconductorShaderInfo } });

	VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(FzbRenderer::globalData.swapChainImageFormat);
	VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment();
	VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

	VkSubpassDescription presentSubpass = fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef);
	std::vector<VkSubpassDescription> subpasses = { presentSubpass };

	std::vector<VkSubpassDependency> dependencies = { fzbCreateSubpassDependency() };

	FzbRenderPassSetting renderPassSetting = { true, 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
	renderRenderPass.setting = renderPassSetting;
	renderRenderPass.createRenderPass(&attachments, subpasses, dependencies);
	renderRenderPass.createFramebuffers(true);

	FzbSubPass presentSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
		{ FzbRenderer::globalData.mainScene.cameraAndLightsDescriptorSetLayout }, { FzbRenderer::globalData.mainScene.cameraAndLightsDescriptorSet },
		mainScene->vertexBuffer.buffer, mainScene->indexBuffer.buffer, sourceManager.shaders_vector);
	renderRenderPass.subPasses.push_back(presentSubPass);
}

void FzbForwardRender::createImages() {
	depthMap = {};
	depthMap.width = FzbRenderer::globalData.swapChainExtent.width;
	depthMap.height = FzbRenderer::globalData.swapChainExtent.height;
	depthMap.type = VK_IMAGE_TYPE_2D;
	depthMap.viewType = VK_IMAGE_VIEW_TYPE_2D;
	depthMap.format = fzbFindDepthFormat();
	depthMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	depthMap.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
	depthMap.initImage();

	frameBufferImages.push_back(&depthMap);
}

void FzbForwardRender::clean() {
	sourceManager.clean();
	FzbFeatureComponent_LoopRender::clean();
	depthMap.clean();
}