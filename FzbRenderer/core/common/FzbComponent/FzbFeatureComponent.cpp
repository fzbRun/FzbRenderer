#include "./FzbFeatureComponent.h"
#include "../FzbRenderer.h"

FzbFeatureComponent::FzbFeatureComponent() {};
FzbFeatureComponent::FzbFeatureComponent(pugi::xml_document& doc) {};
void FzbFeatureComponent::initGlobalData() {
	this->mainScene = &FzbRenderer::globalData.mainScene;
};
void FzbFeatureComponent::clean() {
	FzbComponent::clean();
}

FzbFeatureComponent_LoopRender::FzbFeatureComponent_LoopRender() {};
void FzbFeatureComponent_LoopRender::init() {
	initGlobalData();
	createImages();
	this->renderRenderPass.images = this->frameBufferImages;
	renderFinishedSemaphore = FzbSemaphore(false);
}

void FzbFeatureComponent_LoopRender::destroyFrameBuffer() {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;
	for (int i = 0; i < renderRenderPass.framebuffers.size(); i++) {
		vkDestroyFramebuffer(logicalDevice, renderRenderPass.framebuffers[i], nullptr);
	}
	for (int i = 0; i < this->frameBufferImages.size(); ++i) this->frameBufferImages[i]->clean();
}
void FzbFeatureComponent_LoopRender::createFrameBuffer() {
	createImages();
	renderRenderPass.setting.resolution = FzbRenderer::globalData.getResolution();
	renderRenderPass.createFramebuffers(true);
}

void FzbFeatureComponent_LoopRender::clean() {
	FzbFeatureComponent::clean();
	renderRenderPass.clean();
	renderFinishedSemaphore.clean();
}

FzbFeatureComponent_PreProcess::FzbFeatureComponent_PreProcess() {};
void FzbFeatureComponent_PreProcess::clean() {
	FzbFeatureComponent::clean();
}