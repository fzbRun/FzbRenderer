#pragma once

#include "../../StructSet.h"
#include "../../FzbComponent.h"

#ifndef BVH_H
#define BVH_H

struct FzbBVHSetting {
	

};

struct FzbBVHUniform {
	uint32_t maxDepth;
};

class BVH : public FzbComponent {

public:

	FzbScene* scene;
	FzbBVHSetting kdSetting;
	FzbBVHUniform kdUniform;

	static void addExtensions(FzbBVHSetting kdSetting, std::vector<const char*>& instanceExtensions, std::vector<const char*>& deviceExtensions, VkPhysicalDeviceFeatures& deviceFeatures) {};

	BVH(FzbMainComponent* renderer, FzbScene* scene, FzbBVHSetting setting) {

		this->physicalDevice = renderer->physicalDevice;
		this->logicalDevice = renderer->logicalDevice;
		this->graphicsQueue = renderer->graphicsQueue;
		this->swapChainExtent = renderer->swapChainExtent;
		this->swapChainImageFormat = renderer->swapChainImageFormat;
		this->swapChainImageViews = renderer->swapChainImageViews;
		this->commandPool = renderer->commandPool;
		this->scene = scene;
		this->kdSetting = setting;

	}

private:
	

};

#endif