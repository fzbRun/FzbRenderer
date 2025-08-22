#pragma once

#include "FzbBuffer.h"

#ifndef FZB_IMAGE_H
#define FZB_IMAGE_H

struct FzbImage {

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;

	const char* texturePath = nullptr;
	bool mipmapEnable = false;

	uint32_t width = 512;
	uint32_t height = 512;
	uint32_t depth = 1;
	uint32_t layerNum = 1;
	uint32_t mipLevels = 1;
	VkImageType type = VK_IMAGE_TYPE_2D;
	VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
	VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT;
	VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
	VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
	VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	VkMemoryPropertyFlagBits properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
	VkFilter filter = VK_FILTER_LINEAR;
	VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	VkBool32 anisotropyEnable = VK_TRUE;

	VkImage image;
	VkImageView imageView;
	VkDeviceMemory imageMemory;
	VkSampler textureSampler;

	bool UseExternal = false;
	HANDLE handle = INVALID_HANDLE_VALUE;

	FzbImage() {};

	void fzbCreateImage(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue);

	void createImage();

	void transitionImageLayout(VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);

	void copyBufferToImage(VkBuffer buffer);

	void generateMipmaps();

	void createImageView();

	void createImageSampler();

	void fzbClearTexture(VkCommandBuffer commandBuffer, VkClearColorValue clearColor, VkImageLayout finalLayout, VkPipelineStageFlagBits shaderStage);

	void clean();
};

VkFormat fzbFindSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);

VkFormat fzbFindDepthFormat(VkPhysicalDevice physicalDevice);

bool hasStencilComponent(VkFormat format);

#endif