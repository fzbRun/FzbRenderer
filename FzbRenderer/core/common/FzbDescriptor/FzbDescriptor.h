#pragma once

#include "../FzbCommon.h"
#include <map>
#include <stdexcept>

#ifndef FZB_DESCRIPTOR_H
#define FZB_DESCRIPTOR_H

VkDescriptorPool fzbCreateDescriptorPool(std::map<VkDescriptorType, uint32_t> bufferTypeAndNum);
VkDescriptorPool fzbCreateDescriptorPool(std::vector<VkDescriptorType> type, std::vector<uint32_t> num);
VkDescriptorSetLayout fzbCreateDescriptLayout(uint32_t descriptorNum, std::vector<VkDescriptorType> descriptorTypes, std::vector<VkShaderStageFlags> descriptorStagesFlags, std::vector<uint32_t> descriptorCounts = std::vector<uint32_t>(), std::vector<bool> bindless = std::vector<bool>());
VkDescriptorSet fzbCreateDescriptorSet(VkDescriptorPool descriptorPool, VkDescriptorSetLayout& descriptorSetLayout);
#endif