#pragma once

#include "./FzbComponent.h"
#include "./FzbMainComponent.h"
#include "../FzbRenderPass/FzbRenderPass.h"

#ifndef FZB_FEATURE_COMPONENT_H
#define FZB_FEATURE_COMPONENT_H

enum FzbFeatureComponentName {
	FZB_RENDERER_FORWARD,
	FZB_RENDERER_PATH_TRACING_SOFT,
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
	//std::vector<bool> useMainSceneBufferHandle = { false, false, false };
};

struct FzbFeatureComponent : public FzbComponent {
public:
	FzbFeatureComponentInfo componentInfo;
	FzbMainScene* mainScene;
	std::map<std::string, std::shared_ptr<FzbFeatureComponent>> childComponents;
	//----------------------------------------------------------函数---------------------------------------------------------------
	FzbFeatureComponent();
	FzbFeatureComponent(pugi::xml_document& doc);
	void getChildComponent(pugi::xml_node componentsNode);
	/*
	组件会先找到或创建自己所需要的mesh，如mainScene中的mesh，所需要的线框等
	对所需要使用到的mesh的vertexFormat进行赋值，用于后续创建或读取数据时获得想要的顶点属性
	创建shader
	指明是否需要vertexBuffer的handle
	*/
	virtual void addMainSceneInfo() = 0;
	virtual void addExtensions() = 0;
	virtual void init();

	virtual void prepocessClean();
	void clean() override;
};

struct FzbRendererComponent : public FzbFeatureComponent {
	std::vector<FzbImage*> frameBufferImages;
	FzbRenderPass renderRenderPass;
	FzbSemaphore renderFinishedSemaphore;

	virtual void presentPrepare() = 0;	//创建各种渲染的变量，如缓冲区、renderPass
	void destroyFrameBuffer();
	void createFrameBuffer();
};

struct FzbFeatureComponent_LoopRender : public FzbFeatureComponent {
public:
	std::vector<FzbImage*> frameBufferImages;
	//FzbSubPass subPass;
	FzbRenderPass renderRenderPass;
	FzbSemaphore renderFinishedSemaphore;

	FzbFeatureComponent_LoopRender();
	void init() override;
	virtual void createImages() = 0;
	//virtual void createSubPass() = 0;
	virtual void presentPrepare() = 0;	//创建各种渲染的变量，如缓冲区、renderPass
	virtual VkSemaphore render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE) = 0;

	void destroyFrameBuffer();
	void createFrameBuffer();

	void clean() override;
};

struct FzbFeatureComponent_PreProcess : public FzbFeatureComponent {
	FzbFeatureComponent_PreProcess();
	void init() override;
	void clean();
};

#endif