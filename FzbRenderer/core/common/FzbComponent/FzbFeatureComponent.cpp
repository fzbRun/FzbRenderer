#include "./FzbFeatureComponent.h"
#include "../FzbRenderer.h"

//----------------------------------------------组件----------------------------------------------
FzbFeatureComponent::FzbFeatureComponent() {};
FzbFeatureComponent::FzbFeatureComponent(pugi::xml_document& doc) {};
void FzbFeatureComponent::getChildComponent(pugi::xml_node componentsNode) {
	for (pugi::xml_node featureComponent : componentsNode.children("featureComponent")) {
		std::string childComponentName = featureComponent.attribute("name").value();
		this->childComponents.insert({ childComponentName, createFzbComponent(childComponentName, featureComponent) });
	}
};
void FzbFeatureComponent::init() {
	this->mainScene = &FzbRenderer::globalData.mainScene;
	for (auto& childComponent : this->childComponents) childComponent.second->init();
}

void FzbFeatureComponent::prepocessClean() {};
void FzbFeatureComponent::clean() {
	FzbComponent::clean();
	for (auto& childComponent : this->childComponents) childComponent.second->clean();
}

//----------------------------------------------循环渲染组件----------------------------------------------
FzbFeatureComponent_LoopRender::FzbFeatureComponent_LoopRender() {};
void FzbFeatureComponent_LoopRender::init() {
	FzbFeatureComponent::init();
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

//----------------------------------------------预处理组件----------------------------------------------
FzbFeatureComponent_PreProcess::FzbFeatureComponent_PreProcess() {};
void FzbFeatureComponent_PreProcess::init() {
	FzbFeatureComponent::init();
}
void FzbFeatureComponent_PreProcess::clean() {
	FzbFeatureComponent::clean();
}