#pragma once

#include "StructSet.h"
#include "FzbDevice.h"
#include <stdexcept>
#include <memory>

#ifndef FZB_BUFFER_H
#define FZB_BUFFER_H

void GetMemoryWin32HandleKHR(VkDevice device, VkMemoryGetWin32HandleInfoKHR* handleInfo, HANDLE* handle) {
	auto func = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
	if (func != nullptr) {
		func(device, handleInfo, handle);
	}
}

/*
FzbBuffer主要用于给单独的渲染模块使用
App类不需要FzbBuffer，其本身包含FzbBuffer的功能
*/
class FzbBuffer {

public:

	//依赖
	VkCommandPool commandPool;	//貌似传不了啊，一传就失效，那就自己创建吧
	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	QueueFamilyIndices queueFamilyIndices;
	VkQueue graphicsQueue;

	//VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;

	std::vector<VkBuffer> storageBuffers;
	std::vector<VkDeviceMemory> storageBufferMemorys;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemorys;
	std::vector<void*> uniformBuffersMappeds;

	std::vector<VkBuffer> uniformBuffersStatic;	//对于一些不会变化的uniform buffer
	std::vector<VkDeviceMemory> uniformBuffersMemorysStatic;
	std::vector<void*> uniformBuffersMappedsStatic;

	std::vector<std::vector<VkFramebuffer>> framebuffers;

	std::vector<HANDLE> storageBufferHandles;

	FzbBuffer(std::unique_ptr<FzbDevice>& fzbDevice, VkCommandPool commandPool) {
		this->commandPool = commandPool;
		this->physicalDevice = fzbDevice->physicalDevice;
		this->logicalDevice = fzbDevice->logicalDevice;
		this->queueFamilyIndices = fzbDevice->queueFamilyIndices;
		this->graphicsQueue = fzbDevice->graphicsQueue;
	}

	void createCommandPool() {

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		//VK_COMMAND_POOL_CREATE_TRANSIENT_BIT：提示命令缓冲区经常会重新记录新命令（可能会改变内存分配行为）
		//VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT：允许单独重新记录命令缓冲区，如果没有此标志，则必须一起重置所有命令缓冲区
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();
		if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &this->commandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}

	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory, bool UseExternal = false) {

		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VkExternalMemoryBufferCreateInfo extBufferInfo{};
		if (UseExternal) {
			extBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
			extBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
			bufferInfo.pNext = &extBufferInfo;
		}

		if (vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create vertex buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		VkExportMemoryAllocateInfo exportAllocInfo{};
		if (UseExternal) {
			exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
			exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
			allocInfo.pNext = &exportAllocInfo;
		}

		if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate vertex buffer memory!");
		}

		vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0);

		if (UseExternal) {
			HANDLE handle;
			VkMemoryGetWin32HandleInfoKHR handleInfo{};
			handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
			handleInfo.memory = bufferMemory;
			handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
			GetMemoryWin32HandleKHR(logicalDevice, &handleInfo, &handle);
			this->storageBufferHandles.push_back(handle);
		}

	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {

		VkPhysicalDeviceMemoryProperties memProperties;
		//找到当前物理设备（显卡）所支持的显存类型（是否支持纹理存储，顶点数据存储，通用数据存储等）
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			//这里的type八成是通过one-hot编码的，如0100表示三号,1000表示四号
			if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");

	}

	void createCommandBuffers(uint32_t bufferNum = 1) {

		//我们现在想像流水线一样绘制，所以需要多个指令缓冲区
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = this->commandPool;
		//VK_COMMAND_BUFFER_LEVEL_PRIMARY：可以提交到队列执行，但不能从其他命令缓冲区调用。
		//VK_COMMAND_BUFFER_LEVEL_SECONDARY：不能直接提交，但可以从主命令缓冲区调用
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = bufferNum;	//指定分配的命令缓冲区是主命令缓冲区还是辅助命令缓冲区的大小

		//这个函数的第三个参数可以是单个命令缓冲区指针也可以是数组
		this->commandBuffers.resize(bufferNum);
		if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, this->commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate shadow command buffers!");
		}

	}

	/*
	一个交换链图像视图代表一个展示缓冲,一个renderPass代表一帧中所有的输出流程，其中最后的输出图像是一个交换链图像,一个frameBuffer是renderPass的一种实例，将renderPass中规定的输出图像进行填充
	原来的创建帧缓冲的逻辑有问题啊，按照原来的代码，如果使用fast-Vync，那么一共有三个帧缓冲，在流水线中最后的输出对象是三个帧缓冲之一，但是流水线中的每个渲染管线
	都对应于同一个color和depth附件，这就会导致上一帧还在读，而下一帧就在改了，这就会发生脏读啊。
	但是这是创建帧缓冲的问题吗，这应该是同步没有做好的问题啊，如果每个pass都依赖于上一个pass，那么确实不能使用流水线，除非有多个color或depth缓冲，但是还是同步的问题。
	*/
	void createFramebuffer(uint32_t swapChainImageViewsSize, VkExtent2D swapChainExtent, uint32_t attachmentSize, std::vector<std::vector<VkImageView>>& attachmentImageViews, VkRenderPass renderPass) {

		std::vector<VkFramebuffer> frameBuffers;
		frameBuffers.resize(swapChainImageViewsSize);
		for (size_t i = 0; i < swapChainImageViewsSize; i++) {

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = attachmentSize;
			framebufferInfo.pAttachments = attachmentSize == 0 ? nullptr : attachmentImageViews[i].data();;
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &frameBuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}

		}

		this->framebuffers.push_back(frameBuffers);

	}

	template<typename T>	//模板函数需要在头文件中定义
	void createStorageBuffer(uint32_t bufferSize, std::vector<T>* bufferData, bool UseExternal = false) {
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, bufferData->data(), (size_t)bufferSize);
		vkUnmapMemory(logicalDevice, stagingBufferMemory);

		VkBuffer storageBuffer;
		VkDeviceMemory storageBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, storageBuffer, storageBufferMemory, UseExternal);

		copyBuffer(stagingBuffer, storageBuffer, bufferSize);

		this->storageBuffers.push_back(storageBuffer);
		this->storageBufferMemorys.push_back(storageBufferMemory);

		vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

	}

	void createUniformBuffers(VkDeviceSize bufferSize, bool uniformBufferStatic, uint32_t bufferNum = 1) {

		bufferNum = uniformBufferStatic == true ? 1 : bufferNum;
		for (int i = 0; i < bufferNum; i++) {

			VkBuffer uniformBuffer;
			VkDeviceMemory uniformBuffersMemory;
			void* uniformBuffersMapped;

			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffer, uniformBuffersMemory);
			vkMapMemory(logicalDevice, uniformBuffersMemory, 0, bufferSize, 0, &uniformBuffersMapped);

			if (uniformBufferStatic) {
				this->uniformBuffersStatic.push_back(uniformBuffer);
				this->uniformBuffersMemorysStatic.push_back(uniformBuffersMemory);
				this->uniformBuffersMappedsStatic.push_back(uniformBuffersMapped);
			}
			else {
				this->uniformBuffers.push_back(uniformBuffer);
				this->uniformBuffersMemorys.push_back(uniformBuffersMemory);
				this->uniformBuffersMappeds.push_back(uniformBuffersMapped);
			}

		}
	}

	VkCommandBuffer beginSingleTimeCommands() {

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;

	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {

		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);
		vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);

	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {

		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);	//该函数可以用于任意端的缓冲区，不像memcpy

		endSingleTimeCommands(commandBuffer);

	}

	void cleanupBuffers() {

		for (int i = 0; i < storageBuffers.size(); i++) {
			vkDestroyBuffer(logicalDevice, storageBuffers[i], nullptr);
			vkFreeMemory(logicalDevice, storageBufferMemorys[i], nullptr);
		}

		for (int i = 0; i < uniformBuffers.size(); i++) {
			vkDestroyBuffer(logicalDevice, uniformBuffers[i], nullptr);
			vkFreeMemory(logicalDevice, uniformBuffersMemorys[i], nullptr);
		}

		for (int i = 0; i < uniformBuffersStatic.size(); i++) {
			vkDestroyBuffer(logicalDevice, uniformBuffersStatic[i], nullptr);
			vkFreeMemory(logicalDevice, uniformBuffersMemorysStatic[i], nullptr);
		}

		for (int i = 0; i < storageBufferHandles.size(); i++) {
			CloseHandle(storageBufferHandles[i]);
		}

		for (int i = 0; i < framebuffers.size(); i++) {
			for (int j = 0; j < framebuffers[i].size(); j++) {
				vkDestroyFramebuffer(logicalDevice, framebuffers[i][j], nullptr);
			}
		}

		//vkDestroyCommandPool(logicalDevice, commandPool, nullptr);	//交给主程序销毁

	}

};

#endif