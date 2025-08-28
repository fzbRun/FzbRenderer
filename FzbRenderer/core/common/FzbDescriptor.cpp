#include "FzbDescriptor.h"

VkDescriptorPool fzbCreateDescriptorPool(VkDevice logicalDevice, std::map<VkDescriptorType, uint32_t> bufferTypeAndNum) {

	std::vector<VkDescriptorPoolSize> poolSizes{};
	VkDescriptorPoolSize poolSize;

	for (const auto& pair : bufferTypeAndNum) {
		if (pair.second == 0)
			continue;
		poolSize.type = pair.first;
		poolSize.descriptorCount = pair.second;
		poolSizes.push_back(poolSize);
	}

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = static_cast<uint32_t>(32);

	VkDescriptorPool descriptorPool;
	if (vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor pool!");
	}
	return descriptorPool;
}

VkDescriptorPool fzbCreateDescriptorPool(VkDevice logicalDevice, std::vector<VkDescriptorType> type, std::vector<uint32_t> num) {

	std::vector<VkDescriptorPoolSize> poolSizes{};
	VkDescriptorPoolSize poolSize;

	if (type.size() != num.size())
		throw std::runtime_error("创建描述符池时类型数和数量数不等");
	for (int i = 0; i < type.size(); i++) {
		if (num[i] == 0)
			continue;
		poolSize.type = type[i];
		poolSize.descriptorCount = num[i];
		poolSizes.push_back(poolSize);
	}

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	poolInfo.maxSets = static_cast<uint32_t>(32);

	VkDescriptorPool descriptorPool;
	if (vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor pool!");
	}
	return descriptorPool;
}

VkDescriptorSetLayout fzbCreateDescriptLayout(VkDevice logicalDevice, uint32_t descriptorNum, std::vector<VkDescriptorType> descriptorTypes, std::vector<VkShaderStageFlags> descriptorStagesFlags, std::vector<uint32_t> descriptorCounts, std::vector<bool> bindless) {
	VkDescriptorSetLayout descriptorSetLayout;
	std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
	uint32_t descriptorBinding = 0;
	for (int i = 0; i < descriptorNum; i++) {
		if (descriptorCounts.size() > 0) {
			if (descriptorCounts[i] == 0) {
				continue;
			}
		}

		VkDescriptorSetLayoutBinding binding;
		binding.binding = descriptorBinding++;
		binding.descriptorCount = descriptorCounts.size() > 0 ? descriptorCounts[i] : 1;
		binding.descriptorType = descriptorTypes[i];
		binding.pImmutableSamplers = nullptr;
		binding.stageFlags = descriptorStagesFlags[i];	//shaderStage设置为all和特地shader不会影响数据的读取速度，只是在编译时候检测范围不同而已（影响编译速度而已）。
		layoutBindings.push_back(binding);
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = layoutBindings.size();
	layoutInfo.pBindings = layoutBindings.data();

	VkDescriptorSetLayoutBindingFlagsCreateInfoEXT flagsInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT };
	std::vector<VkDescriptorBindingFlagsEXT> bindingFlags;
	descriptorBinding = 0;
	for (int i = 0; i < bindless.size(); i++) {
		if (descriptorCounts.size() > 0) {
			if (descriptorCounts[i] == 0) {
				continue;
			}
		}
		if (bindless[descriptorBinding]) {
			bindingFlags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT);
		}
		else {
			bindingFlags.push_back(0);	//无特殊标志
		}
		descriptorBinding++;
	}
	if (bindless.size() > 0) {
		flagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
		flagsInfo.pBindingFlags = bindingFlags.data();
		layoutInfo.pNext = &flagsInfo;
	}

	if (vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create compute descriptor set layout!");
	}

	return descriptorSetLayout;
}

VkDescriptorSet fzbCreateDescriptorSet(VkDevice logicalDevice, VkDescriptorPool descriptorPool, VkDescriptorSetLayout& descriptorSetLayout) {
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