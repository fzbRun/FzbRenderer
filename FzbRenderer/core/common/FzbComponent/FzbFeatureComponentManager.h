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
	
	FzbFeatureComponentManager();
	void addFeatureComponent(std::shared_ptr<FzbFeatureComponent> featureComponent);
	//void init();
	void componentInit();
	
	VkSemaphore componentActivate(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE);
	void clean();

private:
	void prepocessClean();
};

#endif