#pragma once

#include "StructSet.h"
#include "FzbBuffer.h"
#include <stdexcept>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

#ifndef FZB_IMAGE_H
#define FZB_IMAGE_H

//void GetMemoryWin32HandleKHR(VkDevice device, VkMemoryGetWin32HandleInfoKHR* handleInfo, HANDLE* handle) {
//	auto func = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
//	if (func != nullptr) {
//		func(device, handleInfo, handle);
//	}
//}

/*
FzbImage��Ҫ���ڸ���������Ⱦģ��ʹ��
App�಻��ҪFzbImage���䱾�����FzbImage�Ĺ���
*/
class FzbImage {

public:

	//����
	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;

	FzbImage(std::unique_ptr<FzbDevice>& fzbDevice) {
		this->physicalDevice = fzbDevice->physicalDevice;
		this->logicalDevice = fzbDevice->logicalDevice;
	}

	void createImage(MyImage& myImage, std::unique_ptr<FzbBuffer>& fzbBuffer, bool UseExternal = false) {

		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = myImage.type;
		imageInfo.extent.width = static_cast<uint32_t>(myImage.width);
		imageInfo.extent.height = static_cast<uint32_t>(myImage.height);
		imageInfo.extent.depth = static_cast<uint32_t>(myImage.depth);
		imageInfo.mipLevels = myImage.mipLevels;
		imageInfo.arrayLayers = 1;
		//findSupportedFormat(physicalDevice, { VK_FORMAT_R8G8B8A8_UINT }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
		imageInfo.format = myImage.format;
		//tiling��һ�����ݴ�Ÿ�ʽ��linear�ǰ��д�ţ���optimal���ǣ����岻֪�����������С��з��ʶ��Ѻã������Ը�˹ģ������ǰ�߷ǳ���
		//VK_IMAGE_TILING_LINEAR: Texels are laid out in row-major order like our pixels array�����������Ҫ����image�ͱ��������ַ�ʽ
		//VK_IMAGE_TILING_OPTIMAL: Texels are laid out in an implementation defined order for optimal access ������ǲ�����ʣ������ַ�ʽ���Ч
		//��˼����������Ƿ�������������Ҫ�õ�һ�֣�������ǲ����ʣ�ֻ����������Ӳ��ȥ�㣬��ڶ����ڲ����Ż������Ч
		imageInfo.tiling = myImage.tiling;
		//������һ������ѹ����ʽ
		//VK_IMAGE_LAYOUT_UNDEFINED: Not usable by the GPU and the very first transition will discard the texels.
		//VK_IMAGE_LAYOUT_PREINITIALIZED: Not usable by the GPU, but the first transition will preserve the texels.
		//�������˼�����Ƿ����imageԭ�е����ݲ��ֽ����޸ģ�������������ԭ�в������޸�һ���֣���ʹ��ǰ�ߣ������VK_IMAGE_LAYOUT_UNDEFINED����֮����
		//�������������ǽ��ⲿ��������ݷŵ���image�У����Բ�����imageԭ������
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = myImage.usage;
		//�����ʲôϡ��洢���
		imageInfo.samples = myImage.sampleCount;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.flags = 0;

		VkExternalMemoryImageCreateInfo externalMemoryImageCreateInfo{};
		if (UseExternal) {
			externalMemoryImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
			externalMemoryImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
			imageInfo.pNext = &externalMemoryImageCreateInfo;
		}

		if (vkCreateImage(logicalDevice, &imageInfo, nullptr, &myImage.image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(logicalDevice, myImage.image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = fzbBuffer->findMemoryType(memRequirements.memoryTypeBits, myImage.properties);

		VkExportMemoryAllocateInfo exportInfo = {};
		if (UseExternal) {
			exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
			exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
			allocInfo.pNext = &exportInfo;
		}

		if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &myImage.imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(logicalDevice, myImage.image, myImage.imageMemory, 0);

		if (UseExternal) {
			VkMemoryGetWin32HandleInfoKHR handleInfo = {};
			handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
			handleInfo.memory = myImage.imageMemory;
			handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
			GetMemoryWin32HandleKHR(logicalDevice, &handleInfo, &myImage.handle);
		}

	}

	//������Ҫ��ȷ�ľ��������ύ��Queu�󲢲���˳��ִ�У���������ִ�У���ô���Ǿͱ��뱣֤���ǲ����õ�����Ĳ����Ѿ�����Ҫ���ˣ�������ѹ��Ҫ��
	//memory barrier����ʹ��a��b�����׶��ڴ�������ڴ�ɼ�������ʵ����һ������д���Ĺ��̣���ô��������������ǾͿ�����дʱ�޸����ݵĴ洢����
	void transitionImageLayout(MyImage& myImage, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, std::unique_ptr<FzbBuffer>& fzbBuffer) {

		VkCommandBuffer commandBuffer = fzbBuffer->beginSingleTimeCommands();

		//1. ����execution barrier��һ��ͬ�����ֶΣ�ʹ��command��ͬ��������������command������һ�����ϣ���ȷ���������ڵĽ׶Σ���a��b���Ϳ���ʹ��dst�߳���b�׶�������ֱ��src�߳�ִ����a�׶�
		//2. ����execution barrierֻ�ܱ�ִ֤��˳���ͬ���������ܱ�֤�ڴ��ͬ������a�׶������������Ҫ��b�׶�ʹ�ã�����b�׶�ȥ��ȡʱ������û�и��»��Ѿ�����cache���ˣ���ô����Ҫ�õ�memory barrier
		//	ͨ��srcAccessMask��֤a�׶�������������ڴ���õģ��������Ѿ����µ�L2 Cache�У�ͨ��dstAccessMask��֤b�׶λ�õ������Ѿ������ˣ������ݴ�L2 Cache���µ�L1 Cache
		//3. VkBufferMemoryBarrier���ڱ�֤buffer֮����ڴ�����.��VkBufferMemoryBarrier��˵���ṩ��srcQueueFamilyIndex��dstQueueFamilyIndex��ת����Buffer������Ȩ������VkMemoryBarrierû�еĹ��ܡ�
		//	ʵ�����������Ϊ��ת��Buffer������Ȩ�Ļ��ģ��������й�Buffer��ͬ��������ȫ����VkMemoryBarrier����ɣ�һ����˵VkBufferMemoryBarrier�õĺ��١�
		//4. VkImageMemoryBarrier���ڱ�֤Image���ڴ�����������ͨ��ָ��subresourceRange������Image��ĳ������Դ��֤���ڴ�������ͨ��ָ�� oldLayout��newLayout��������ִ��Layout Transition��
		//	���Һ�VkBufferMemoryBarrier����ͬ���ܹ����ת����Image������Ȩ�Ĺ��ܡ�
		//5. �������ǽ�����a, barrier, b, c���������Queue��c��a�Ĳ���û�������������Ի���Ϊbarrier���ȴ�a������ɲ��ܼ�����Ⱦ�����������˷ѣ����Կ���ʹ��event��ֻͬ��a��b����c��������ִ��
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		//���Խ���������Ϊ���ض���������Ч
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//image���кܶ��subsource��������ɫ���ݡ�mipmap
		barrier.image = myImage.image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mipLevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = 0;

		//https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineStageFlagBits.html
		VkPipelineStageFlags sourceStage;	//a�׶�
		VkPipelineStageFlags destinationStage;	//b�׶�
		//������һ����Ҫע��ĵ㣬�Ǿ���AccessMask��Ҫ��VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT/VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT����ʹ�ã���Щ�׶β�ִ���ڴ���ʣ������κ�srcAccessMask��dstAccessMask���������׶ε���϶���û�������
		//	����Vulkan��������������TOP_OF_PIPE��BOTTOM_OF_PIPE������Ϊ��Execution Barrier�����ڣ�������Memory Barrier��
		//����һ����srcAccessMask���ܻᱻ����ΪXXX_READ��flag��������ȫ����ġ��ö�������һ���ڴ���õĹ�����û�������(��Ϊû��������Ҫ����)��Ҳ����˵�Ƕ�ȡ���ݵ�ʱ��û��ˢ��Cache����Ҫ������ܻ��˷�һ���ֵ����ܡ�
		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			//������ʵէһ��ͦ��ֵģ���Ϊbarrier����������command֮��ģ�����copyimage����ֻ��һ��command���������ʹ��barrier�أ�
			//�������ǿ�������������ֻ��Ҫ���ڴ�ɼ�����ʱ�޸�imageLayout����ô���ǲ�����ǰһ��command��ʲô�����Բ�ϣ��ǰ���command�������ǵ�copy command�����Կ��Խ�srcAccessMask��Ϊ0��sourceStage��ΪVK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
			//ͬʱ���ǿ���copy������command���Ǿ���Ĳ�����������VK_PIPELINE_STAGE_TRANSFER_BIT�����Խ�destinationStage��Ϊ���������ʵ������˿��������
			//ͬʱ���ƵĲ���Ҫ��VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
			barrier.srcAccessMask = 0;	//ֻ��д��������Ҫ�ڴ���ã�����ֻ��Ҫ��������Ҫ�˷�����
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;//������д����ν����Ϊд֮ǰҪ��
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;	//��Ⱦ��ʼ���ͷ�Ľ׶�
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;	//��ͬ�������в�ͬ�Ľ׶Σ�����copy�������VK_PIPELINE_STAGE_TRANSFER_BIT�������ݸ��ƽ׶�
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			//ͬ��������Ҫ�õ�image��command����ȴ�copy command��VK_PIPELINE_STAGE_TRANSFER_BIT��������ܿ�ʼ����
			//���������ڲ���ǰ���Խ����ָĳ�VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL���ַ�������Ĳ��֣���Ȼ�Ҳ�֪��ѹ������ѹ�Բ�����ʲôӰ�죩
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		fzbBuffer->endSingleTimeCommands(commandBuffer);

	}

	void copyBufferToImage(VkBuffer buffer, MyImage& myImage, std::unique_ptr<FzbBuffer>& fzbBuffer) {

		VkCommandBuffer commandBuffer = fzbBuffer->beginSingleTimeCommands();

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = { myImage.width, myImage.height, myImage.depth };

		vkCmdCopyBufferToImage(commandBuffer, buffer, myImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		fzbBuffer->endSingleTimeCommands(commandBuffer);

	}

	void generateMipmaps(MyImage& myImage, std::unique_ptr<FzbBuffer>& fzbBuffer) {

		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, myImage.format, &formatProperties);
		if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw std::runtime_error("texture image format does not support linear blitting!");
		}

		VkCommandBuffer commandBuffer = fzbBuffer->beginSingleTimeCommands();
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.image = myImage.image;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.subresourceRange.levelCount = 1;

		int32_t mipWidth = myImage.width;
		int32_t mipHeight = myImage.height;
		int32_t mipDepth = myImage.depth;

		for (uint32_t i = 1; i < myImage.mipLevels; i++) {

			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			//ÿ��blit command����һ��barrier,ʹ��һ����dst��תΪsrc
			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0, 0, 0 };	//���
			blit.srcOffsets[1] = { mipWidth, mipHeight, mipDepth };	//�ֱ���
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.dstOffsets[0] = { 0, 0, 0 };
			blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, mipDepth > 1 ? mipDepth / 2 : 1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;

			vkCmdBlitImage(commandBuffer,
				myImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				myImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				VK_FILTER_LINEAR);

			//����Ҫ�����ֶ�ȥת���֣����Ƕ����뱻������mipmap���ڱ�blitǰ������������blit�����ǰ��ͨ��barrier�Զ���srcתΪshader read
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			if (mipWidth > 1) mipWidth /= 2;
			if (mipHeight > 1) mipHeight /= 2;
			if (mipDepth > 1) mipDepth /= 2;

		}

		//Ϊ���һ��mipmap����barrier
		barrier.subresourceRange.baseMipLevel = myImage.mipLevels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		fzbBuffer->endSingleTimeCommands(commandBuffer);

	}

	void createImageView(MyImage& myImage) {

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = myImage.image;
		viewInfo.viewType = myImage.viewType;// == VK_IMAGE_TYPE_2D ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_3D;
		viewInfo.format = myImage.format;
		viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.subresourceRange.aspectMask = myImage.aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = myImage.mipLevels;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(logicalDevice, &viewInfo, nullptr, &myImage.imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image views!");
		}

	}

	void createImageSampler(MyImage& myImage) {

		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = myImage.filter;
		samplerInfo.minFilter = myImage.filter;
		samplerInfo.addressModeU = myImage.addressMode;	//�����Ϊrepeat����ӰMap�߽���������
		samplerInfo.addressModeV = myImage.addressMode;
		samplerInfo.addressModeW = myImage.addressMode;
		samplerInfo.anisotropyEnable = myImage.anisotropyEnable;
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = static_cast<float>(myImage.mipLevels);

		if (vkCreateSampler(logicalDevice, &samplerInfo, nullptr, &myImage.textureSampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}

	}

	void createMyImage(MyImage& myImage, std::unique_ptr<FzbBuffer>& fzbBuffer, bool UseExternal = false) {

		if (myImage.texturePath) {
			int texWidth, texHeight, texChannels;
			stbi_uc* pixels = stbi_load(myImage.texturePath, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
			VkDeviceSize imageSize = texWidth * texHeight * 4;
			if (!pixels) {
				throw std::runtime_error("failed to load texture image!");
			}
			myImage.width = texWidth;
			myImage.height = texHeight;

			//��ɫ���в���������ͬ��ֻ��ҪGPU�ɼ������ԺͶ��㻺����һ���������Ƚ����ݴ浽�ݴ滺�����Ŵ浽GPU����������
			VkBuffer stagingBuffer;
			VkDeviceMemory stagingBufferMemory;
			fzbBuffer->createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

			void* data;
			vkMapMemory(logicalDevice, stagingBufferMemory, 0, imageSize, 0, &data);
			memcpy(data, pixels, static_cast<size_t>(imageSize));
			vkUnmapMemory(logicalDevice, stagingBufferMemory);

			stbi_image_free(pixels);

			myImage.mipLevels = myImage.mipmapEnable ? static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1 : 1;
			createImage(myImage, fzbBuffer);

			//���Ǵ�����image��֪��ԭ����ʲô���֣�����Ҳ�����ģ�����ֻ��Ҫ��copyǰ�޸����Ĳ���,��������Ҳ������ǰ���command����ϣ��������(��Դmash��stage��Ϊ�����ģ�
			transitionImageLayout(myImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, myImage.mipLevels, fzbBuffer);
			copyBufferToImage(stagingBuffer, myImage, fzbBuffer);

			vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
			vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

			if (myImage.mipmapEnable) {
				generateMipmaps(myImage, fzbBuffer);
			}
			else {
				//�������˼�������ƬԪ��ɫ����Ҫ������������ȴ���������ɲ��ܿ�ʼ�������ڲ���ǰ�޸Ĳ��֣�����Ҫ��copy command����
				transitionImageLayout(myImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, fzbBuffer);
			}

			createImageView(myImage);
			createImageSampler(myImage);

			return;

		}

		createImage(myImage, fzbBuffer, UseExternal);
		createImageView(myImage);
		createImageSampler(myImage);

	}

	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {

		for (VkFormat format : candidates) {

			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}

		}

		throw std::runtime_error("failed to find supported format!");

	}

	VkFormat findDepthFormat(VkPhysicalDevice physicalDevice) {
		return findSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}

	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void clearTexture(VkCommandBuffer commandBuffer, MyImage& myImage, VkClearColorValue clearColor, VkImageLayout finalLayout, VkPipelineStageFlagBits shaderStage) {

		//#ifndef Voxelization_Block
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//image���кܶ��subsource��������ɫ���ݡ�mipmap
		barrier.image = myImage.image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		VkImageSubresourceRange subresourceRange = {};
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.levelCount = 1;
		subresourceRange.baseArrayLayer = 0;
		subresourceRange.layerCount = 1;
		vkCmdClearColorImage(commandBuffer, myImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &subresourceRange);

		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = finalLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//image���кܶ��subsource��������ɫ���ݡ�mipmap
		barrier.image = myImage.image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			shaderStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);
		//#endif

	}

	void cleanImage(MyImage& myImage) {
		if (myImage.handle)
			CloseHandle(myImage.handle);
		if (myImage.textureSampler) {
			vkDestroySampler(logicalDevice, myImage.textureSampler, nullptr);
		}
		vkDestroyImageView(logicalDevice, myImage.imageView, nullptr);
		vkDestroyImage(logicalDevice, myImage.image, nullptr);
		vkFreeMemory(logicalDevice, myImage.imageMemory, nullptr);
	}

};

#endif