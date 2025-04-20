#pragma once

#include "StructSet.h"
#include "FzbDevice.h"
#include <stdexcept>
#include <memory>

#ifndef FZB_DESCRIPTOR_H
#define FZB_DESCRIPTOR_H

/*
FzbDescriptor主要用于给单独的渲染模块使用
App类不需要FzbDescriptor，其本身包含FzbDescriptor的功能
*/
class FzbDescriptor {

public:

	//依赖
	VkDevice logicalDevice;

	VkDescriptorPool descriptorPool;
	std::vector<std::vector<VkDescriptorSet>> descriptorSets;

	FzbDescriptor(std::unique_ptr<FzbDevice>& fzbDevice) {
		this->logicalDevice = fzbDevice->logicalDevice;
	}

	void createDescriptorPool(std::map<VkDescriptorType, uint32_t> bufferTypeAndNum) {

		std::vector<VkDescriptorPoolSize> poolSizes{};
		VkDescriptorPoolSize poolSize;

		for (const auto& pair : bufferTypeAndNum) {
			poolSize.type = pair.first;
			poolSize.descriptorCount = pair.second;
			poolSizes.push_back(poolSize);
		}

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(32);

		if (vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &this->descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}

	}

	VkDescriptorSetLayout createDescriptLayout(uint32_t descriptorNum, std::vector<VkDescriptorType> descriptorTypes, std::vector<VkShaderStageFlagBits> descriptorShaderFlags, std::vector<uint32_t> descriptorCounts = std::vector<uint32_t>()) {
		VkDescriptorSetLayout descriptorSetLayout;
		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
		layoutBindings.resize(descriptorNum);
		for (int i = 0; i < descriptorNum; i++) {
			layoutBindings[i].binding = i;
			layoutBindings[i].descriptorCount = 1;
			layoutBindings[i].descriptorType = descriptorTypes[i];
			layoutBindings[i].pImmutableSamplers = nullptr;
			layoutBindings[i].stageFlags = descriptorShaderFlags[i];
		}

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = layoutBindings.size();
		layoutInfo.pBindings = layoutBindings.data();
		if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute descriptor set layout!");
		}

		return descriptorSetLayout;

	}

	VkDescriptorSet createDescriptorSet(VkDescriptorSetLayout& descriptorSetLayout) {
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &descriptorSetLayout;

		VkDescriptorSet descriptorSet;
		if (vkAllocateDescriptorSets(logicalDevice, &allocInfo, &descriptorSet) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}
		return descriptorSet;
	}

};

#endif