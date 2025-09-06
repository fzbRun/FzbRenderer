#pragma once

#include "./FzbFeatureComponent.h"

#ifndef FZB_FEATURE_COMPONENT_MANAGER
#define FZB_FEATURE_COMPONENT_MANAGER

struct FzbFeatureComponentManager {
public:
	std::shared_ptr<FzbFeatureComponent_LoopRender> renderComponent;		//��Ⱦ���
	std::shared_ptr<FzbFeatureComponent_LoopRender> postProcessingComponent = nullptr;	//�������
	std::vector<std::shared_ptr<FzbFeatureComponent_PreProcess>> preprocessFeatureComponent;	//ֻ����Ԥ����Ĺ���������羲̬������bvh
	std::vector<std::shared_ptr<FzbFeatureComponent_LoopRender>> loopRenderFeatureComponent;	//������Ⱦѭ���Ĺ���������綯̬������bvh

	FzbVertexFormat vertexFormat_preprocess = FzbVertexFormat();	//��������Ԥ�����������Ķ�������
	std::vector<bool> useMainSceneBuffer_preprocess = { false, false, false };
	std::vector<bool> useMainSceneBufferHandle_preprocess = { false, false, false };
	FzbVertexFormat vertexFormat_looprender = FzbVertexFormat(true);	//����������Ⱦѭ���������Ķ�������
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