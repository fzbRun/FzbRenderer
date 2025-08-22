#pragma once

#include "StructSet.h"
#include <map>
#include <stdexcept>

#ifndef FZB_DESCRIPTOR_H
#define FZB_DESCRIPTOR_H

VkDescriptorPool fzbCreateDescriptorPool(VkDevice logicalDevice, std::map<VkDescriptorType, uint32_t> bufferTypeAndNum);

VkDescriptorPool fzbCreateDescriptorPool(VkDevice logicalDevice, std::vector<VkDescriptorType> type, std::vector<uint32_t> num);

VkDescriptorSetLayout fzbCreateDescriptLayout(VkDevice logicalDevice, uint32_t descriptorNum, std::vector<VkDescriptorType> descriptorTypes, std::vector<VkShaderStageFlags> descriptorStagesFlags, std::vector<uint32_t> descriptorCounts = std::vector<uint32_t>(), std::vector<bool> bindless = std::vector<bool>());

VkDescriptorSet fzbCreateDescriptorSet(VkDevice logicalDevice, VkDescriptorPool descriptorPool, VkDescriptorSetLayout& descriptorSetLayout);
#endif