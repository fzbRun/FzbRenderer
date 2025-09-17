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

/*BVHһ���Ǳ�����������õģ�������componentManager*/
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

	//��դ��������Դ
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

	//����˵һ��present��˼·
	//1. ֻ����bvhNodeArray����ΪtriangleInfoArray�����������Ϣֻ�ڹ�׷�����ã����ô�ͳ����չʾ
	//2. ��uniformBuffer�и����ڵ��������ڼ�����ɫ���У��ж��������Ƿ��ڽڵ�AABB�У����������Ⱦ
	//ͨ�����ַ�ʽ���ҿ���֪��ÿ���ڵ�����Щ�����Σ�������debug��
	void createDescriptor();
	void createImages() override;
	void createRenderPass();

};
#endif