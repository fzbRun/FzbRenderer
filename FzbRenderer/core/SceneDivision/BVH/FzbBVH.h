#pragma once

#include "../../common/FzbCommon.h"
#include "../../common/FzbComponent/FzbFeatureComponent.h"
#include "./CUDA/createBVH.cuh"
#include "../../common/FzbRasterizationRender/FzbRasterizationSourceManager.h"

#ifndef BVH_H
#define BVH_H

struct FzbBVHUniform {
	uint32_t bvhTreeDepth;
};

struct FzbBVHPresentUniform {
	uint32_t nodeIndex;
	uint32_t bvhTreeDepth;
};

/*BVH一般是被其他组件调用的，不放入componentManager*/
class FzbBVH : public FzbFeatureComponent_PreProcess {

public:
	FzbBVHSetting setting;

	FzbBuffer uniformBuffer;
	FzbBuffer bvhNodeArray;
	FzbBuffer bvhTriangleInfoArray;

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;

	FzbSemaphore bvhCudaSemaphore;
	
	FzbBVH();
	FzbBVH(pugi::xml_node& BVHNode);
	void addMainSceneInfo() override;
	void init() override ;
	void clean() override;

private:
	FzbBVHUniform kdUniform;
	std::unique_ptr<BVHCuda> bvhCuda;

	void addExtensions();
	void createBVH();

	//光栅化所需资源
	void createBuffer();
	void createDescriptor();
};

class FzbBVH_Debug : public FzbFeatureComponent_LoopRender {

public:
	FzbBVH_Debug();
	FzbBVH_Debug(pugi::xml_node& BVHNode);

	void init() override;

	VkSemaphore render(uint32_t imageIndex, VkSemaphore startSemaphore, VkFence fence = VK_NULL_HANDLE);

	void clean();

private:

	FzbBVHSetting setting;
	FzbBVHUniform kdUniform;

	FzbBuffer uniformBuffer;
	FzbBuffer bvhNodeArray;
	FzbBuffer bvhTriangleInfoArray;

	FzbImage depthMap;

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;

	FzbSemaphore bvhCudaSemaphore;

	std::unique_ptr<BVHCuda> bvhCuda;

	FzbRasterizationSourceManager presentSourceManager;
	//FzbShader presentShader;
	//FzbMaterial presentMaterial;

	void addMainSceneInfo() override;
	void addExtensions() override;

	void presentPrepare() override;

	void createBVH();

	void createBuffer();

	//先来说一下present的思路
	//1. 只传入bvhNodeArray，因为triangleInfoArray存的三角形信息只在光追中有用，我用传统管线展示
	//2. 在uniformBuffer中给定节点索引，在几何着色器中，判断三角形是否在节点AABB中，如果在则渲染
	//通过这种方式，我可以知道每个节点有哪些三角形，方便我debug。
	void createDescriptor();
	void createImages() override;
	void createRenderPass();

};
#endif