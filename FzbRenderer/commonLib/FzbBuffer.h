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
FzbBuffer��Ҫ���ڸ���������Ⱦģ��ʹ��
App�಻��ҪFzbBuffer���䱾�����FzbBuffer�Ĺ���
*/
class FzbBuffer {

public:

	//����
	VkCommandPool commandPool;	//ò�ƴ����˰���һ����ʧЧ���Ǿ��Լ�������
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

	std::vector<VkBuffer> uniformBuffersStatic;	//����һЩ����仯��uniform buffer
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
		//VK_COMMAND_POOL_CREATE_TRANSIENT_BIT����ʾ����������������¼�¼��������ܻ�ı��ڴ������Ϊ��
		//VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT�����������¼�¼������������û�д˱�־�������һ�����������������
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
		//�ҵ���ǰ�����豸���Կ�����֧�ֵ��Դ����ͣ��Ƿ�֧������洢���������ݴ洢��ͨ�����ݴ洢�ȣ�
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			//�����type�˳���ͨ��one-hot����ģ���0100��ʾ����,1000��ʾ�ĺ�
			if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");

	}

	void createCommandBuffers(uint32_t bufferNum = 1) {

		//��������������ˮ��һ�����ƣ�������Ҫ���ָ�����
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = this->commandPool;
		//VK_COMMAND_BUFFER_LEVEL_PRIMARY�������ύ������ִ�У������ܴ���������������á�
		//VK_COMMAND_BUFFER_LEVEL_SECONDARY������ֱ���ύ�������Դ��������������
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = bufferNum;	//ָ����������������������������Ǹ�����������Ĵ�С

		//��������ĵ��������������ǵ����������ָ��Ҳ����������
		this->commandBuffers.resize(bufferNum);
		if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, this->commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate shadow command buffers!");
		}

	}

	/*
	һ��������ͼ����ͼ����һ��չʾ����,һ��renderPass����һ֡�����е�������̣������������ͼ����һ��������ͼ��,һ��frameBuffer��renderPass��һ��ʵ������renderPass�й涨�����ͼ��������
	ԭ���Ĵ���֡������߼������Ⱑ������ԭ���Ĵ��룬���ʹ��fast-Vync����ôһ��������֡���壬����ˮ���������������������֡����֮һ��������ˮ���е�ÿ����Ⱦ����
	����Ӧ��ͬһ��color��depth��������ͻᵼ����һ֡���ڶ�������һ֡���ڸ��ˣ���ͻᷢ���������
	�������Ǵ���֡�������������Ӧ����ͬ��û�����õ����Ⱑ�����ÿ��pass����������һ��pass����ôȷʵ����ʹ����ˮ�ߣ������ж��color��depth���壬���ǻ���ͬ�������⡣
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

	template<typename T>	//ģ�庯����Ҫ��ͷ�ļ��ж���
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
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);	//�ú���������������˵Ļ�����������memcpy

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

		//vkDestroyCommandPool(logicalDevice, commandPool, nullptr);	//��������������

	}

};

#endif