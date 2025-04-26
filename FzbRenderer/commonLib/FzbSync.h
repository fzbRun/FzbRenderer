#pragma once

#include "StructSet.h"
#include "FzbDevice.h"

#ifndef FZB_SYNC_H
#define FZB_SYNC_H

void GetSemaphoreWin32HandleKHR(VkDevice device, VkSemaphoreGetWin32HandleInfoKHR* handleInfo, HANDLE* handle) {
	auto func = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
	if (func != nullptr) {
		func(device, handleInfo, handle);
	}
}

struct FzbSemaphore {
	VkSemaphore semaphore;
	HANDLE handle;
};

/*
FzbSync主要用于给单独的渲染模块使用
App类不需要FzbSync，其本身包含FzbSync的功能
*/
class FzbSync {

public:

	//依赖
	VkDevice logicalDevice;

	std::vector<FzbSemaphore> fzbSemaphores;
	std::vector<VkFence> fzbFences;

	FzbSync(std::unique_ptr<FzbDevice>& fzbDevice) {
		this->logicalDevice = fzbDevice->logicalDevice;
	}

	void createSemaphore(bool UseExternal) {
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkExportSemaphoreCreateInfoKHR exportInfo = {};
		if (UseExternal) {
			exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
			exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
			semaphoreInfo.pNext = &exportInfo;
		}

		FzbSemaphore fzbSemphore = {};
		VkSemaphore semaphore;
		if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
			throw std::runtime_error("failed to create semaphores!");
		}
		fzbSemphore.semaphore = semaphore;

		if (UseExternal) {
			HANDLE handle;
			VkSemaphoreGetWin32HandleInfoKHR handleInfo = {};
			handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
			handleInfo.semaphore = semaphore;
			handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT; 
			GetSemaphoreWin32HandleKHR(logicalDevice, &handleInfo, &handle);
			fzbSemphore.handle = handle;
		}

		this->fzbSemaphores.push_back(fzbSemphore);

	}

	void createFence() {
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		//第一帧可以直接获得信号，而不会阻塞
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		VkFence fence;
		if (vkCreateFence(logicalDevice, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to create semaphores!");
		}
		this->fzbFences.push_back(fence);

	}

	void cleanFzbSync() {
		for (int i = 0; i < fzbSemaphores.size(); i++) {
			if (fzbSemaphores[i].handle)
				CloseHandle(fzbSemaphores[i].handle);
			vkDestroySemaphore(logicalDevice, fzbSemaphores[i].semaphore, nullptr);

		}
		for (int i = 0; i < fzbFences.size(); i++) {
			vkDestroyFence(logicalDevice, fzbFences[i], nullptr);
		}
	}

};

#endif