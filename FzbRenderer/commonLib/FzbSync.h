#pragma once

#include "StructSet.h"
#include "FzbDevice.h"

#ifndef FZB_SYNC_H
#define FZB_SYNC_H

/*
FzbSync主要用于给单独的渲染模块使用
App类不需要FzbSync，其本身包含FzbSync的功能
*/
class FzbSync {

public:

	//依赖
	VkDevice logicalDevice;

	FzbSync(std::unique_ptr<FzbDevice>& fzbDevice) {
		this->logicalDevice = fzbDevice->logicalDevice;
	}

	VkSemaphore createSemaphore() {
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		VkSemaphore semaphore;
		if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
			throw std::runtime_error("failed to create semaphores!");
		}
		return semaphore;
	}

	VkFence createFence() {
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		//第一帧可以直接获得信号，而不会阻塞
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		VkFence fence;
		if (vkCreateFence(logicalDevice, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to create semaphores!");
		}

		return fence;

	}

};

#endif