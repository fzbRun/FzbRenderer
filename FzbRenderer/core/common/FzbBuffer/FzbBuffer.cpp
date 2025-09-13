#include "FzbBuffer.h"
#include "../FzbRenderer.h"
#include <stdexcept>

void GetMemoryWin32HandleKHR(VkMemoryGetWin32HandleInfoKHR* handleInfo, HANDLE* handle) {
	VkDevice device = FzbRenderer::globalData.logicalDevice;
	auto func = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
	if (func != nullptr) {
		func(device, handleInfo, handle);
	}
}
VkDeviceAddress getBufferDeviceAddressKHR(VkBufferDeviceAddressInfoKHR* handleInfo) {
	VkDevice device = FzbRenderer::globalData.logicalDevice;
	auto func = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR");
	if (func != nullptr) {
		return func(device, handleInfo);
	}
	return 0;
}
uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
	VkPhysicalDevice physicalDevice = FzbRenderer::globalData.physicalDevice;
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
VkCommandBuffer beginSingleTimeCommands() {
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = FzbRenderer::globalData.commandPool;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(FzbRenderer::globalData.logicalDevice, &allocInfo, &commandBuffer);

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

	vkQueueSubmit(FzbRenderer::globalData.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(FzbRenderer::globalData.graphicsQueue);
	vkFreeCommandBuffers(FzbRenderer::globalData.logicalDevice, FzbRenderer::globalData.commandPool, 1, &commandBuffer);
}
void fzbBeginCommandBuffer(VkCommandBuffer commandBuffer) {
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("failed to begin recording command buffer!");
	}
}
void fzbSubmitCommandBuffer(VkCommandBuffer commandBuffer,
	std::vector<VkSemaphore> waitSemaphores,
	std::vector<VkPipelineStageFlags> waitStages,
	std::vector<VkSemaphore> signalSemphores,
	VkFence fence) {

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.waitSemaphoreCount = waitSemaphores.size();
	submitInfo.pWaitSemaphores = waitSemaphores.data();
	submitInfo.pWaitDstStageMask = waitStages.data();
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	submitInfo.signalSemaphoreCount = signalSemphores.size();
	submitInfo.pSignalSemaphores = signalSemphores.data();

	if (vkQueueSubmit(FzbRenderer::globalData.graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
		throw std::runtime_error("failed to submit draw command buffer!");
	}
}

void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;
	VkCommandPool commandPool = FzbRenderer::globalData.commandPool;

	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkBufferCopy copyRegion{};
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = size;
	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);	//该函数可以用于任意端的缓冲区，不像memcpy

	endSingleTimeCommands(commandBuffer);
}

FzbBuffer::FzbBuffer() {
	this->buffer = nullptr;
	this->memory = nullptr;
	this->handle = nullptr;
	this->UseExternal = false;
};
FzbBuffer::FzbBuffer(uint32_t bufferSize, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool UseExternal) {
	this->physicalDevice = FzbRenderer::globalData.physicalDevice;
	this->logicalDevice = FzbRenderer::globalData.logicalDevice;
	this->size = bufferSize;
	this->usage = usage;
	this->properties = properties;
	this->UseExternal = UseExternal;
}
void FzbBuffer::fzbCreateBuffer() {
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
		GetMemoryWin32HandleKHR(&handleInfo, &handle);
	}
}
void FzbBuffer::fzbFillBuffer(void* bufferData) {
	void* data;	//得到一个指针
	vkMapMemory(logicalDevice, memory, 0, size, 0, &data);	//该指针指向暂存（CPU和GPU均可访问）buffer
	memcpy(data, bufferData, (size_t)size);	//将数据传入该暂存buffer
	vkUnmapMemory(logicalDevice, memory);	//解除映射
}
void FzbBuffer::fzbGetBufferDeviceAddress() {
	VkBufferDeviceAddressInfoKHR addressInfo{};
	addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	addressInfo.buffer = buffer;
	deviceAddress = getBufferDeviceAddressKHR(&addressInfo);
}
void FzbBuffer::clean() {
	if (buffer != VK_NULL_HANDLE) {
		vkDestroyBuffer(logicalDevice, buffer, nullptr);
		vkFreeMemory(logicalDevice, memory, nullptr);
		buffer = nullptr;
		closeHandle();
	}
}
void FzbBuffer::closeHandle() {
	if (handle != INVALID_HANDLE_VALUE) {
		CloseHandle(handle);
		handle = INVALID_HANDLE_VALUE;
	}

}

FzbBuffer fzbCreateStorageBuffer(void* bufferData, uint32_t bufferSize, bool UseExternal) {
	FzbBuffer stagingBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	stagingBuffer.fzbCreateBuffer();
	stagingBuffer.fzbFillBuffer(bufferData);

	FzbBuffer fzbBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UseExternal);
	fzbBuffer.fzbCreateBuffer();

	copyBuffer(stagingBuffer.buffer, fzbBuffer.buffer, bufferSize);

	stagingBuffer.clean();
	return fzbBuffer;
}
FzbBuffer fzbCreateStorageBuffer(uint32_t bufferSize, bool UseExternal) {
	FzbBuffer fzbBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UseExternal);
	fzbBuffer.fzbCreateBuffer();
	return fzbBuffer;
}
FzbBuffer fzbCreateUniformBuffers(uint32_t bufferSize) {
	FzbBuffer fzbBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	fzbBuffer.fzbCreateBuffer();
	vkMapMemory(FzbRenderer::globalData.logicalDevice, fzbBuffer.memory, 0, bufferSize, 0, &fzbBuffer.mapped);
	return fzbBuffer;
}
FzbBuffer fzbCreateIndirectCommandBuffer(void* bufferData, uint32_t bufferSize) {
	FzbBuffer stagingBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	stagingBuffer.fzbCreateBuffer();
	stagingBuffer.fzbFillBuffer(bufferData);

	FzbBuffer fzbBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
	fzbBuffer.fzbCreateBuffer();

	copyBuffer(stagingBuffer.buffer, fzbBuffer.buffer, bufferSize);

	stagingBuffer.clean();
	return fzbBuffer;
}