#pragma once

#include "./FzbComponent.h"
#include "./FzbMainComponent.h"
#include "../FzbRenderPass/FzbRenderPass.h"

#ifndef FZB_FEATURE_COMPONENT_H
#define FZB_FEATURE_COMPONENT_H

enum FzbFeatureComponentName {
	FZB_RENDERER_FORWARD,
	FZB_FEATURE_COMPONENT_BVH,
	FZB_FEATURE_COMPONENT_BVH_DEBUG,
	FZB_FEATURE_COMPONENT_SVO,
	FZB_FEATURE_COMPONENT_SVO_DEBUG
};

enum FzbFeatureComponentType {
	FZB_RENDER_COMPONENT,
	FZB_POST_PROCESS_COMPONENT,
	FZB_PREPROCESS_FEATURE_COMPONENT,
	FZB_LOOPRENDER_FEATURE_COMPONENT,
};

struct FzbFeatureComponentInfo {
	bool available = false;
	FzbFeatureComponentName name;
	FzbFeatureComponentType type;
	FzbVertexFormat vertexFormat = FzbVertexFormat();	//组件所需要的顶点属性
	std::vector<bool> useMainSceneBufferHandle = { false, false, false };
};

struct FzbFeatureComponent : public FzbComponent {
public:
	FzbFeatureComponentInfo componentInfo;
	FzbScene* mainScene;
	//----------------------------------------------------------函数---------------------------------------------------------------
	FzbFeatureComponent();
	FzbFeatureComponent(pugi::xml_document& doc);
	virtual void addExtensions() = 0;
	void initGlobalData();
	virtual void init() = 0;
	void clean() override ;
};

struct FzbFeatureComponent_LoopRender : public FzbFeatureComponent {
public:
	std::vector<FzbImage*> frameBufferImages;
	FzbRenderPass renderRenderPass;
	FzbSemaphore renderFinishedSemaphore;

	FzbFeatureComponent_LoopRender();
	void init() override;
	virtual void createImages() = 0;
	virtual void presentPrepare() = 0;	//创建各种渲染的变量，如缓冲区、renderPass
	virtual VkSemaphore render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) = 0;

	void destroyFrameBuffer();
	void createFrameBuffer();

	void clean() override;
};

struct FzbFeatureComponent_PreProcess : public FzbFeatureComponent {
	FzbFeatureComponent_PreProcess();

	void clean();
};

#endif