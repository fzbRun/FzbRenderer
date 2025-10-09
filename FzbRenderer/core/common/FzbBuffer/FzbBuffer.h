#pragma once

#include "../FzbCommon.h"

#ifndef FZB_BUFFER_H
#define FZB_BUFFER_H

void GetMemoryWin32HandleKHR(VkMemoryGetWin32HandleInfoKHR* handleInfo, HANDLE* handle);
VkDeviceAddress getBufferDeviceAddressKHR(VkBufferDeviceAddressInfoKHR* handleInfo);
//------------------------------------------------------公共函数---------------------------------------
uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
VkCommandBuffer beginSingleTimeCommands();
void endSingleTimeCommands(VkCommandBuffer commandBuffer);
void fzbBeginCommandBuffer(VkCommandBuffer commandBuffer);
void fzbSubmitCommandBuffer(VkCommandBuffer commandBuffer,
	std::vector<VkSemaphore> waitSemaphores = std::vector<VkSemaphore>(), 
	std::vector<VkPipelineStageFlags> waitStages = std::vector<VkPipelineStageFlags>(),
	std::vector<VkSemaphore> signalSemphores = std::vector<VkSemaphore>(),
	VkFence fence = nullptr);

void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
//----------------------------------------------------结构体-----------------------------------------
struct FzbBuffer {

public:

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;

	VkBuffer buffer = nullptr;
	VkDeviceMemory memory = nullptr;
	void* mapped = nullptr;
	HANDLE handle = INVALID_HANDLE_VALUE;
	uint64_t deviceAddress;
	uint32_t size;

	VkBufferUsageFlags usage;
	VkMemoryPropertyFlags properties;
	bool UseExternal = false;

	FzbBuffer();

	FzbBuffer(uint32_t bufferSize, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool UseExternal = false);

	void fzbCreateBuffer();

	//这里的buffer一定要CPU可见，即VkMemoryPropertyFlags要有VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
	void fillBuffer(void* bufferData);

	void fzbGetBufferDeviceAddress();
	
	void clearBuffer(VkCommandBuffer commandBuffer, VkPipelineStageFlagBits waitStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

	void clean();
	void closeHandle();

};
/*
struct FzbUniformBuffer : public FzbBuffer {
	

	FzbUniformBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, uint32_t bufferSize, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool UseExternal = false) {
		this->physicalDevice = physicalDevice;
		this->logicalDevice = logicalDevice;
		this->size = bufferSize;
		this->usage = usage;
		this->properties = properties;
		this->UseExternal = UseExternal;
	}

	void fzbCreateUniformBuffer() {
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
		allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

		void* extendFlagsInfo = nullptr;
		VkMemoryAllocateFlagsInfo allocFlagsInfo;
		if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
			allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
			allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT; // 设置设备地址标志
			allocFlagsInfo.pNext = extendFlagsInfo;
			extendFlagsInfo = &allocFlagsInfo;
		}

		VkExportMemoryAllocateInfo exportAllocInfo{};
		if (UseExternal) {
			exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
			exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
			exportAllocInfo.pNext = extendFlagsInfo;
			extendFlagsInfo = &exportAllocInfo;
		}
		allocInfo.pNext = extendFlagsInfo;

		if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate vertex buffer memory!");
		}

		vkBindBufferMemory(logicalDevice, buffer, memory, 0);

		if (UseExternal) {
			VkMemoryGetWin32HandleInfoKHR handleInfo{};
			handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
			handleInfo.memory = memory;
			handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
			GetMemoryWin32HandleKHR(logicalDevice, &handleInfo, &handle);
		}
	}

};

struct FzbStorageBuffer : public FzbBuffer {

	FzbStorageBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, uint32_t bufferSize, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool UseExternal = false) {
		this->physicalDevice = physicalDevice;
		this->logicalDevice = logicalDevice;
		this->size = bufferSize;
		this->usage = usage;
		this->properties = properties;
		this->UseExternal = UseExternal;
	}

	void fzbCreateStorageBuffer() {
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
		allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

		void* extendFlagsInfo = nullptr;
		VkMemoryAllocateFlagsInfo allocFlagsInfo;
		if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
			allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
			allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT; // 设置设备地址标志
			allocFlagsInfo.pNext = extendFlagsInfo;
			extendFlagsInfo = &allocFlagsInfo;
		}

		VkExportMemoryAllocateInfo exportAllocInfo{};
		if (UseExternal) {
			exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
			exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
			exportAllocInfo.pNext = extendFlagsInfo;
			extendFlagsInfo = &exportAllocInfo;
		}
		allocInfo.pNext = extendFlagsInfo;

		if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate vertex buffer memory!");
		}

		vkBindBufferMemory(logicalDevice, buffer, memory, 0);

		if (UseExternal) {
			VkMemoryGetWin32HandleInfoKHR handleInfo{};
			handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
			handleInfo.memory = memory;
			handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
			GetMemoryWin32HandleKHR(logicalDevice, &handleInfo, &handle);
		}
	}

};
*/
//---------------------------------------------------------------------------------------------------------------------------------
FzbBuffer fzbCreateStorageBuffer(void* bufferData, uint32_t bufferSize, bool UseExternal = false);
//创造一个空的buffer
FzbBuffer fzbCreateStorageBuffer(uint32_t bufferSize, bool UseExternal = false);
FzbBuffer fzbCreateUniformBuffer(uint32_t bufferSize);
FzbBuffer fzbCreateIndirectCommandBuffer(void* bufferData, uint32_t bufferSize);
#endif