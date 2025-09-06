#include "FzbFeatureComponentManager.h"
#include "../FzbRenderer.h"

FzbFeatureComponentManager::FzbFeatureComponentManager() {
	this->postProcessingComponent = nullptr;
	this->vertexFormat_preprocess = FzbVertexFormat();
	this->useMainSceneBuffer_preprocess = { false, false, false };
	this->useMainSceneBufferHandle_preprocess = { false, false, false };
	this->vertexFormat_looprender = FzbVertexFormat(true);
	this->useMainSceneBuffer_looprender = { true, false, false };
	this->useMainSceneBufferHandle_looprender = { false, false, false };
};
void FzbFeatureComponentManager::addFeatureComponent(std::shared_ptr<FzbFeatureComponent> featureComponent) {
	if (featureComponent == nullptr) return;
	if (featureComponent->componentInfo.available == false) return;
	if (featureComponent->componentInfo.type == FZB_RENDER_COMPONENT) this->renderComponent = std::dynamic_pointer_cast<FzbFeatureComponent_LoopRender>(featureComponent);
	else if (featureComponent->componentInfo.type == FZB_POST_PROCESS_COMPONENT) this->postProcessingComponent = std::dynamic_pointer_cast<FzbFeatureComponent_LoopRender>(featureComponent);
	else if (featureComponent->componentInfo.type == FZB_PREPROCESS_FEATURE_COMPONENT) {
		std::shared_ptr<FzbFeatureComponent_PreProcess> featureComponent_PreProcess = std::dynamic_pointer_cast<FzbFeatureComponent_PreProcess>(featureComponent);
		this->preprocessFeatureComponent.push_back(featureComponent_PreProcess);
	}
	else {
		std::shared_ptr<FzbFeatureComponent_LoopRender> featureComponent_LoopRender = std::dynamic_pointer_cast<FzbFeatureComponent_LoopRender>(featureComponent);
		this->loopRenderFeatureComponent.push_back(featureComponent_LoopRender);
	}
}

void FzbFeatureComponentManager::init() {
	for (int i = 0; i < preprocessFeatureComponent.size(); i++) {
		FzbFeatureComponentInfo componentInfo = preprocessFeatureComponent[i]->componentInfo;
		if (!componentInfo.vertexFormat.available) continue;
		this->vertexFormat_preprocess.mergeUpward(componentInfo.vertexFormat);
		if (componentInfo.vertexFormat == FzbVertexFormat()) useMainSceneBuffer_preprocess[1] = true;
		if (componentInfo.vertexFormat == FzbVertexFormat(true)) useMainSceneBuffer_preprocess[2] = true;

		for (int i = 0; i < componentInfo.useMainSceneBufferHandle.size(); i++) {
			this->useMainSceneBufferHandle_preprocess[i] = this->useMainSceneBufferHandle_preprocess[i] || componentInfo.useMainSceneBufferHandle[i];
		}
	}
	for (int i = 0; i < loopRenderFeatureComponent.size(); i++) {
		FzbFeatureComponentInfo componentInfo = loopRenderFeatureComponent[i]->componentInfo;
		if (componentInfo.vertexFormat.available) {
			this->vertexFormat_looprender.mergeUpward(componentInfo.vertexFormat);
			if (componentInfo.vertexFormat == FzbVertexFormat()) useMainSceneBuffer_looprender[1] = true;
			if (componentInfo.vertexFormat == FzbVertexFormat(true)) useMainSceneBuffer_looprender[2] = true;
		}

		for (int i = 0; i < componentInfo.useMainSceneBufferHandle.size(); i++) {
			this->useMainSceneBufferHandle_looprender[i] = this->useMainSceneBufferHandle_looprender[i] || componentInfo.useMainSceneBufferHandle[i];
		}
	}
}

void FzbFeatureComponentManager::cleanBuffer() {
	FzbScene& mainScene = FzbRenderer::globalData.mainScene;
	if (useMainSceneBufferHandle_looprender[0] == false) {
		if (mainScene.vertexBuffer.handle != INVALID_HANDLE_VALUE) mainScene.vertexBuffer.closeHandle();
		if (mainScene.indexBuffer.handle != INVALID_HANDLE_VALUE) mainScene.indexBuffer.closeHandle();
	}
	if (useMainSceneBufferHandle_looprender[1] == false) {
		if (mainScene.vertexPosBuffer.handle != INVALID_HANDLE_VALUE) mainScene.vertexPosBuffer.closeHandle();
		if (mainScene.indexPosBuffer.handle != INVALID_HANDLE_VALUE) mainScene.indexPosBuffer.closeHandle();
	}
	if (useMainSceneBufferHandle_looprender[2] == false) {
		if (mainScene.vertexPosNormalBuffer.handle != INVALID_HANDLE_VALUE) mainScene.vertexPosNormalBuffer.closeHandle();
		if (mainScene.indexPosNormalBuffer.handle != INVALID_HANDLE_VALUE) mainScene.indexPosNormalBuffer.closeHandle();
	}

	if (useMainSceneBuffer_looprender[1] == false) {
		mainScene.vertexPosBuffer.clean();
		mainScene.indexPosBuffer.clean();
	}
	if (useMainSceneBuffer_looprender[2] == false) {
		mainScene.vertexPosNormalBuffer.clean();
		mainScene.indexPosNormalBuffer.clean();
	}
}
void FzbFeatureComponentManager::componentInit() {
	if (renderComponent) this->renderComponent->init();
	if (postProcessingComponent) this->postProcessingComponent->init();
	for (int i = 0; i < preprocessFeatureComponent.size(); i++) preprocessFeatureComponent[i]->init();
	for (int i = 0; i < loopRenderFeatureComponent.size(); i++) loopRenderFeatureComponent[i]->init();
	cleanBuffer();
}

VkSemaphore FzbFeatureComponentManager::componentActivate(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence) {
	for (int i = 0; i < this->loopRenderFeatureComponent.size(); i++) {
		if(renderComponent == nullptr && postProcessingComponent == nullptr && i == this->loopRenderFeatureComponent.size() - 1)
			startSemaphore = this->loopRenderFeatureComponent[i]->render(imageIndex, startSemaphore, fence);
		else startSemaphore = this->loopRenderFeatureComponent[i]->render(imageIndex, startSemaphore, VK_NULL_HANDLE);
	}
	if (renderComponent) startSemaphore = this->renderComponent->render(imageIndex, startSemaphore, postProcessingComponent ? VK_NULL_HANDLE : fence);
	if (postProcessingComponent) startSemaphore = this->postProcessingComponent->render(imageIndex, startSemaphore, fence);
	return startSemaphore;
}

void FzbFeatureComponentManager::clean() {
	if (renderComponent) this->renderComponent->clean();
	if (postProcessingComponent) this->postProcessingComponent->clean();
	for (int i = 0; i < this->preprocessFeatureComponent.size(); i++)
		this->preprocessFeatureComponent[i]->clean();
	for (int i = 0; i < this->loopRenderFeatureComponent.size(); i++)
		this->loopRenderFeatureComponent[i]->clean();
}