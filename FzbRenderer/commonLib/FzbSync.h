#pragma once

#include "StructSet.h"
#include "FzbDevice.h"

#ifndef FZB_SYNC_H
#define FZB_SYNC_H

/*
FzbSync��Ҫ���ڸ���������Ⱦģ��ʹ��
App�಻��ҪFzbSync���䱾�����FzbSync�Ĺ���
*/
class FzbSync {

public:

	//����
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
		//��һ֡����ֱ�ӻ���źţ�����������
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		VkFence fence;
		if (vkCreateFence(logicalDevice, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to create semaphores!");
		}

		return fence;

	}

};

#endif