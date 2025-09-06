#pragma once

#include "./FzbFeatureComponent.h"

#ifndef FZB_FEATURE_COMPONENT_MANAGER
#define FZB_FEATURE_COMPONENT_MANAGER

struct FzbFeatureComponentManager {
public:
	std::shared_ptr<FzbFeatureComponent_LoopRender> renderComponent;		//渲染组件
	std::shared_ptr<FzbFeatureComponent_LoopRender> postProcessingComponent = nullptr;	//后处理组件
	std::vector<std::shared_ptr<FzbFeatureComponent_PreProcess>> preprocessFeatureComponent;	//只参与预处理的功能组件，如静态场景的bvh
	std::vector<std::shared_ptr<FzbFeatureComponent_LoopRender>> loopRenderFeatureComponent;	//参与渲染循环的功能组件，如动态场景的bvh

	FzbVertexFormat vertexFormat_preprocess = FzbVertexFormat();	//满足所有预处理组件所需的顶点属性
	std::vector<bool> useMainSceneBuffer_preprocess = { false, false, false };
	std::vector<bool> useMainSceneBufferHandle_preprocess = { false, false, false };
	FzbVertexFormat vertexFormat_looprender = FzbVertexFormat(true);	//满足所有渲染循环组件所需的顶点属性
	std::vector<bool> useMainSceneBuffer_looprender = { true, false, false };
	std::vector<bool> useMainSceneBufferHandle_looprender = { false, false, false };
	
	FzbFeatureComponentManager();
	void addFeatureComponent(std::shared_ptr<FzbFeatureComponent> featureComponent);
	void init();
	void cleanBuffer();
	void componentInit();
	
	VkSemaphore componentActivate(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE);
	void clean();
};

#endif