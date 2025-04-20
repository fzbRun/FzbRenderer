#pragma once

#include "StructSet.h"
#include "FzbDevice.h"
#include <random>


#ifndef FZB_SWAPCHAIN_H
#define FZB_SWAPCHAIN_H

class FzbSwapchain {

public:

	//依赖
	GLFWwindow* window;
	VkSurfaceKHR surface;
	VkDevice logicalDevice;
	SwapChainSupportDetails swapChainSupportDetails;
	QueueFamilyIndices queueFamilyIndices;

	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	VkSurfaceFormatKHR surfaceFormat;
	VkExtent2D extent;

	FzbSwapchain(GLFWwindow* window, VkSurfaceKHR surface, std::unique_ptr<FzbDevice>& fzbDevice) {
		this->window = window;
		this->surface = surface;
		this->logicalDevice = fzbDevice->logicalDevice;
		this->swapChainSupportDetails = fzbDevice->swapChainSupportDetails;
		this->queueFamilyIndices = fzbDevice->queueFamilyIndices;
		createSwapChain();
	}

	void createSwapChain() {

		if (swapChainSupportDetails.formats.empty() || swapChainSupportDetails.presentModes.empty() || !queueFamilyIndices.isComplete()) {
			throw std::runtime_error("设备未初始化");
		}

		this->surfaceFormat = chooseSwapSurfaceFormat();	//主要是surface所展示的纹理的通道数、精度以及色彩空间
		VkPresentModeKHR presentMode = chooseSwapPresentMode();
		this->extent = chooseSwapExtent();

		//如果交换链最小和最大的图像数相等，则确定可支持的图象数就是现在支持的图象数，否则是最小图象数+1
		//如果maxImageCount=0，则表示没有限制（但可能其他地方会限制，无法做到）
		uint32_t imageCount = swapChainSupportDetails.capabilities.minImageCount + 1;
		if (swapChainSupportDetails.capabilities.maxImageCount > 0 && imageCount > swapChainSupportDetails.capabilities.maxImageCount) {
			imageCount = swapChainSupportDetails.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;	//我要传到哪
		createInfo.minImageCount = imageCount;	//规定了交换缓冲区中纹理的数量，如2就是双缓冲
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;	//纹理数组的z，1就表示2D纹理
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		uint32_t queueFamilyIndicesArray[] = { queueFamilyIndices.graphicsAndComputeFamily.value(), queueFamilyIndices.presentFamily.value() };

		//图形队列族负责渲染功能，然后交给交换链；交换链再交给展示队列族呈现到surface上
		if (queueFamilyIndices.graphicsAndComputeFamily != queueFamilyIndices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndicesArray;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			//createInfo.queueFamilyIndexCount = 0;
			//createInfo.pQueueFamilyIndices = nullptr;
		}

		createInfo.preTransform = swapChainSupportDetails.capabilities.currentTransform;	//指明是否需要提前旋转或反转
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		//std::vector<VkImage> swapChainImagesTemp;
		vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
		this->swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, this->swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;

		createSwapChainImageViews();

	}

	void createSwapChainImageViews() {

		//imageViews和交换链中的image数量相同
		this->swapChainImageViews.resize(this->swapChainImages.size());
		for (size_t i = 0; i < this->swapChainImages.size(); i++) {

			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = swapChainImages[i];
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = this->swapChainImageFormat;
			viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;

			if (vkCreateImageView(logicalDevice, &viewInfo, nullptr, &this->swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
			}

		}

	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat() {

		for (const auto& availableFormat : swapChainSupportDetails.formats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}
		return swapChainSupportDetails.formats[0];

	}

	VkPresentModeKHR chooseSwapPresentMode() {
		for (const auto& availablePresentMode : swapChainSupportDetails.presentModes) {
			//交换链如何呈现画面，比如是直接展示还是双缓冲，等等
			//VK_PRESENT_MODE_IMMEDIATE_KHR 渲染完成后立即展示，每帧呈现后都需要等待下一帧渲染完成才能替换，如果下一帧渲染的时快时慢，就会出现卡顿
			//VK_PRESENT_MODE_FIFO_KHR V-Sync,垂直同步，多缓冲，渲染完成后提交画面到后面的缓冲，固定时间（显示器刷新时间）后呈现到显示器上。若缓冲区满了，渲染就会停止（阻塞）
			//VK_PRESENT_MODE_FIFO_RELAXED_KHR 渲染完成后提交画面到后面的缓冲，但是如果这一帧渲染的较慢，导致上一帧在刷新后仍存在，则当前帧提交后立刻呈现，那么就可能导致割裂
			//VK_PRESENT_MODE_MAILBOX_KHR Fast-Sync, 三缓冲，渲染完成后提交画面到后面的缓冲，固定时间（显示器刷新时间）后呈现到显示器上。若缓冲区满了，则画面会替换最后的缓冲区，不会阻塞
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent() {

		VkSurfaceCapabilitiesKHR& capabilities = swapChainSupportDetails.capabilities;
		if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
			return capabilities.currentExtent;
		}
		else {		//某些窗口管理器允许我们在此处使用不同的值，这通过将currentExtent的宽度和高度设置为最大值来表示，我们不想要这样，可以将之重新设置为窗口大小
			int width, height;
			//查询窗口分辨率
			glfwGetFramebufferSize(window, &width, &height);
			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;

		}
	}

};

#endif