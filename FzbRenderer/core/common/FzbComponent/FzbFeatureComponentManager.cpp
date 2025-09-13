#include "FzbFeatureComponentManager.h"
#include "../FzbRenderer.h"

FzbFeatureComponentManager::FzbFeatureComponentManager() {
	this->postProcessingComponent = nullptr;
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

/*
void FzbFeatureComponentManager::init() {
	if (this->renderComponent) {
		FzbFeatureComponentInfo componentInfo = this->renderComponent->componentInfo;
		if (componentInfo.vertexFormat.available) {
			this->vertexFormat_preprocess.mergeUpward(componentInfo.vertexFormat);
			if (componentInfo.usePosBuffer) useMainSceneBuffer_preprocess[1] = true;
			if (componentInfo.usePosNormalBuffer) useMainSceneBuffer_preprocess[2] = true;
		}
		for (int i = 0; i < componentInfo.useMainSceneBufferHandle.size(); i++) {
			this->useMainSceneBufferHandle_looprender[i] = this->useMainSceneBufferHandle_looprender[i] || componentInfo.useMainSceneBufferHandle[i];
		}
	}
	if (this->postProcessingComponent) {
		FzbFeatureComponentInfo componentInfo = this->postProcessingComponent->componentInfo;
		if (componentInfo.vertexFormat.available) {
			this->vertexFormat_preprocess.mergeUpward(componentInfo.vertexFormat);
			if (componentInfo.usePosBuffer) useMainSceneBuffer_preprocess[1] = true;
			if (componentInfo.usePosNormalBuffer) useMainSceneBuffer_preprocess[2] = true;
		}
		for (int i = 0; i < componentInfo.useMainSceneBufferHandle.size(); i++) {
			this->useMainSceneBufferHandle_looprender[i] = this->useMainSceneBufferHandle_looprender[i] || componentInfo.useMainSceneBufferHandle[i];
		}
	}
	for (int i = 0; i < preprocessFeatureComponent.size(); i++) {
		FzbFeatureComponentInfo componentInfo = preprocessFeatureComponent[i]->componentInfo;
		if (!componentInfo.vertexFormat.available) continue;
		this->vertexFormat_preprocess.mergeUpward(componentInfo.vertexFormat);
		if (componentInfo.usePosBuffer) useMainSceneBuffer_preprocess[1] = true;
		if (componentInfo.usePosNormalBuffer) useMainSceneBuffer_preprocess[2] = true;

		for (int i = 0; i < componentInfo.useMainSceneBufferHandle.size(); i++) {
			this->useMainSceneBufferHandle_preprocess[i] = this->useMainSceneBufferHandle_preprocess[i] || componentInfo.useMainSceneBufferHandle[i];
		}
	}
	for (int i = 0; i < loopRenderFeatureComponent.size(); i++) {
		FzbFeatureComponentInfo componentInfo = loopRenderFeatureComponent[i]->componentInfo;
		if (componentInfo.vertexFormat.available) {
			this->vertexFormat_looprender.mergeUpward(componentInfo.vertexFormat);
			if (componentInfo.usePosBuffer) useMainSceneBuffer_preprocess[1] = true;
			if (componentInfo.usePosNormalBuffer) useMainSceneBuffer_preprocess[2] = true;
		}

		for (int i = 0; i < componentInfo.useMainSceneBufferHandle.size(); i++) {
			this->useMainSceneBufferHandle_looprender[i] = this->useMainSceneBufferHandle_looprender[i] || componentInfo.useMainSceneBufferHandle[i];
		}
	}
}
*/
void FzbFeatureComponentManager::prepocessClean() {
	vkDeviceWaitIdle(FzbRenderer::globalData.logicalDevice);	//阻塞CPU，等待logicalDevice所有任务执行完成

	if (renderComponent) this->renderComponent->prepocessClean();
	if (postProcessingComponent) this->postProcessingComponent->prepocessClean();
	for (int i = 0; i < preprocessFeatureComponent.size(); i++) preprocessFeatureComponent[i]->prepocessClean();
	for (int i = 0; i < loopRenderFeatureComponent.size(); i++) loopRenderFeatureComponent[i]->prepocessClean();

	FzbScene& mainScene = FzbRenderer::globalData.mainScene;
	if (!FzbRenderer::globalData.mainScene.useVertexBufferHandle) {		//如果loopRender不使用handle，那么可能复用顶点数据的prepocess会使用，则需要清除
		if (mainScene.vertexBuffer.handle != INVALID_HANDLE_VALUE) mainScene.vertexBuffer.closeHandle();
		if (mainScene.indexBuffer.handle != INVALID_HANDLE_VALUE) mainScene.indexBuffer.closeHandle();
	}
	if (FzbRenderer::globalData.mainScene.useVertexBuffer_prepocess) {
		FzbRenderer::globalData.mainScene.vertexBuffer_prepocess.clean();
		FzbRenderer::globalData.mainScene.indexBuffer_prepocess.clean();
	}
}
void FzbFeatureComponentManager::componentInit() {
	if (renderComponent) this->renderComponent->init();
	if (postProcessingComponent) this->postProcessingComponent->init();
	for (int i = 0; i < preprocessFeatureComponent.size(); i++) preprocessFeatureComponent[i]->init();
	for (int i = 0; i < loopRenderFeatureComponent.size(); i++) loopRenderFeatureComponent[i]->init();

	prepocessClean();
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