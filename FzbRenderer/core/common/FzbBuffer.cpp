#include "FzbBuffer.h"

void GetMemoryWin32HandleKHR(VkDevice device, VkMemoryGetWin32HandleInfoKHR* handleInfo, HANDLE* handle) {
	auto func = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
	if (func != nullptr) {
		func(device, handleInfo, handle);
	}
}

VkDeviceAddress getBufferDeviceAddressKHR(VkDevice device, VkBufferDeviceAddressInfoKHR* handleInfo) {
	auto func = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR");
	if (func != nullptr) {
		return func(device, handleInfo);
	}
	return 0;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {

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

VkCommandBuffer beginSingleTimeCommands(VkDevice logicalDevice, VkCommandPool commandPool) {

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

void endSingleTimeCommands(VkDevice logicalDevice, VkCommandPool commandPool, VkCommandBuffer commandBuffer, VkQueue queue) {

	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);
	vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);

}

void copyBuffer(VkDevice logicalDevice, VkCommandPool commandPool, VkQueue queue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {

	VkCommandBuffer commandBuffer = beginSingleTimeCommands(logicalDevice, commandPool);

	VkBufferCopy copyRegion{};
	copyRegion.srcOffset = 0;
	copyRegion.dstOffset = 0;
	copyRegion.size = size;
	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);	//该函数可以用于任意端的缓冲区，不像memcpy

	endSingleTimeCommands(logicalDevice, commandPool, commandBuffer, queue);

}

void FzbBuffer::fzbGetBufferDeviceAddress() {
	VkBufferDeviceAddressInfoKHR addressInfo{};
	addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	addressInfo.buffer = buffer;
	deviceAddress = getBufferDeviceAddressKHR(logicalDevice, &addressInfo);
}

FzbBuffer fzbCreateStorageBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, void* bufferData, uint32_t bufferSize, bool UseExternal) {

	FzbBuffer stagingBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	stagingBuffer.fzbCreateBuffer();
	stagingBuffer.fzbFillBuffer(bufferData);

	FzbBuffer fzbBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UseExternal);
	fzbBuffer.fzbCreateBuffer();

	copyBuffer(logicalDevice, commandPool, graphicsQueue, stagingBuffer.buffer, fzbBuffer.buffer, bufferSize);

	stagingBuffer.clean();

	return fzbBuffer;

}

FzbBuffer fzbCreateStorageBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, uint32_t bufferSize, bool UseExternal) {
	FzbBuffer fzbBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UseExternal);
	fzbBuffer.fzbCreateBuffer();
	return fzbBuffer;
}

FzbBuffer fzbCreateUniformBuffers(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, uint32_t bufferSize) {
	FzbBuffer fzbBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	fzbBuffer.fzbCreateBuffer();
	vkMapMemory(logicalDevice, fzbBuffer.memory, 0, bufferSize, 0, &fzbBuffer.mapped);
	return fzbBuffer;
}

FzbBuffer fzbCreateIndirectCommandBuffer(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, void* bufferData, uint32_t bufferSize) {

	FzbBuffer stagingBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	stagingBuffer.fzbCreateBuffer();
	stagingBuffer.fzbFillBuffer(bufferData);

	FzbBuffer fzbBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false);
	fzbBuffer.fzbCreateBuffer();

	copyBuffer(logicalDevice, commandPool, graphicsQueue, stagingBuffer.buffer, fzbBuffer.buffer, bufferSize);

	stagingBuffer.clean();

	return fzbBuffer;

}