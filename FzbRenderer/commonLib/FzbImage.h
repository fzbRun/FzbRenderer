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
FzbImage主要用于给单独的渲染模块使用
App类不需要FzbImage，其本身包含FzbImage的功能
*/
class FzbImage {

public:

	//依赖
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
		//tiling是一种数据存放格式，linear是按行存放，而optimal不是（具体不知道，但对于行、列访问都友好），所以高斯模糊是用前者非常费
		//VK_IMAGE_TILING_LINEAR: Texels are laid out in row-major order like our pixels array，如果我们想要访问image就必须用这种方式
		//VK_IMAGE_TILING_OPTIMAL: Texels are laid out in an implementation defined order for optimal access 如果我们不想访问，则这种方式最高效
		//意思就是如果我们访问纹理，我们需要用第一种，如果我们不访问，只交由驱动或硬件去搞，则第二种内部有优化，最高效
		imageInfo.tiling = myImage.tiling;
		//布局是一种纹理压缩方式
		//VK_IMAGE_LAYOUT_UNDEFINED: Not usable by the GPU and the very first transition will discard the texels.
		//VK_IMAGE_LAYOUT_PREINITIALIZED: Not usable by the GPU, but the first transition will preserve the texels.
		//这里的意思就是是否相对image原有的数据布局进行修改，比如我们想在原有布局上修改一部分，则使用前者，并配合VK_IMAGE_LAYOUT_UNDEFINED；反之后者
		//我们现在这里是将外部纹理的数据放到该image中，所以不关心image原来布局
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = myImage.usage;
		//好像和什么稀疏存储相关
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

	//首先需要明确的就是命令提交到Queu后并不是顺序执行，而是乱序执行，那么我们就必须保证我们采样用的纹理的布局已经符合要求了，即符合压缩要求
	//memory barrier可以使得a，b两个阶段内存可用与内存可见，这其实就是一个数据写读的过程，那么在这个过程中我们就可以在写时修改数据的存储布局
	void transitionImageLayout(MyImage& myImage, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, std::unique_ptr<FzbBuffer>& fzbBuffer) {

		VkCommandBuffer commandBuffer = fzbBuffer->beginSingleTimeCommands();

		//1. 屏障execution barrier是一种同步的手段，使得command间同步。我们在两个command间设置一个屏障，并确定屏障所在的阶段，如a，b，就可以使得dst线程在b阶段阻塞，直到src线程执行完a阶段
		//2. 但是execution barrier只能保证执行顺序的同步，而不能保证内存的同步，即a阶段输出的数据需要被b阶段使用，但是b阶段去获取时，数据没有更新或已经不在cache中了，那么就需要用到memory barrier
		//	通过srcAccessMask保证a阶段输出的数据是内存可用的，即数据已经更新到L2 Cache中；通过dstAccessMask保证b阶段获得的数据已经更新了，即数据从L2 Cache更新到L1 Cache
		//3. VkBufferMemoryBarrier用于保证buffer之间的内存依赖.于VkBufferMemoryBarrier来说，提供了srcQueueFamilyIndex和dstQueueFamilyIndex来转换该Buffer的所有权。这是VkMemoryBarrier没有的功能。
		//	实际上如果不是为了转换Buffer的所有权的话的，其他的有关Buffer的同步需求完全可以VkMemoryBarrier来完成，一般来说VkBufferMemoryBarrier用的很少。
		//4. VkImageMemoryBarrier用于保证Image的内存依赖，可以通过指定subresourceRange参数对Image上某个子资源保证其内存依赖，通过指定 oldLayout、newLayout参数用于执行Layout Transition。
		//	并且和VkBufferMemoryBarrier类似同样能够完成转换该Image的所有权的功能。
		//5. 加入我们将命令a, barrier, b, c，放入队列Queue，c与a的操作没有依赖，但是仍会因为barrier而等待a操作完成才能继续渲染，导致性能浪费，所以可以使用event来只同步a和b，而c可以自由执行
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		//可以将屏障设置为对特定队列族生效
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		//image含有很多的subsource，比如颜色数据、mipmap
		barrier.image = myImage.image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = mipLevels;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = 0;

		//https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineStageFlagBits.html
		VkPipelineStageFlags sourceStage;	//a阶段
		VkPipelineStageFlags destinationStage;	//b阶段
		//这里有一个需要注意的点，那就是AccessMask不要和VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT/VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT搭配使用，这些阶段不执行内存访问，所以任何srcAccessMask和dstAccessMask与这两个阶段的组合都是没有意义的
		//	并且Vulkan不允许这样做。TOP_OF_PIPE和BOTTOM_OF_PIPE纯粹是为了Execution Barrier而存在，而不是Memory Barrier。
		//还有一个是srcAccessMask可能会被设置为XXX_READ的flag，这是完全多余的。让读操作做一个内存可用的过程是没有意义的(因为没有数据需要更新)，也就是说是读取数据的时候并没有刷新Cache的需要，这可能会浪费一部分的性能。
		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			//这里其实乍一想挺奇怪的，因为barrier是用来两个command之间的，但是copyimage明明只是一个command啊，这如何使用barrier呢？
			//但是我们可以这样想我们只需要在内存可见可用时修改imageLayout，那么我们不关心前一个command是什么，所以不希望前面的command阻塞我们的copy command，所以可以将srcAccessMask设为0，sourceStage设为VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
			//同时我们可以copy（不是command，是具体的操作）发生在VK_PIPELINE_STAGE_TRANSFER_BIT，所以将destinationStage设为这个，但其实不会有丝毫被阻塞
			//同时复制的布局要是VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
			barrier.srcAccessMask = 0;	//只有写操作才需要内存可用，这里只需要读，不需要浪费性能
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;//读还是写无所谓，因为写之前要读
			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;	//渲染开始的最开头的阶段
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;	//不同的命令有不同的阶段，对于copy命令，其有VK_PIPELINE_STAGE_TRANSFER_BIT，即数据复制阶段
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			//同理，后面需要用到image的command必须等待copy command的VK_PIPELINE_STAGE_TRANSFER_BIT结束后才能开始采样
			//并且我们在采样前可以将布局改成VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL这种方便采样的布局（虽然我不知道压缩、解压对采样有什么影响）
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

			//每次blit command都放一个barrier,使上一次是dst的转为src
			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0, 0, 0 };	//起点
			blit.srcOffsets[1] = { mipWidth, mipHeight, mipDepth };	//分辨率
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

			//不需要现在手动去转布局，而是对于想被采样的mipmap，在被blit前阻塞采样，在blit后采样前，通过barrier自动将src转为shader read
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

		//为最后一级mipmap设置barrier
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
		samplerInfo.addressModeU = myImage.addressMode;	//如果设为repeat，阴影Map边界会出现问题
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

			//着色器中采样的纹理同样只需要GPU可见，所以和顶点缓冲区一样，我们先将数据存到暂存缓冲区才存到GPU的纹理缓冲中
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

			//我们创建的image不知道原来是什么布局，我们也不关心，我们只想要在copy前修改他的布局,并且我们也不关心前面的command，不希望被阻塞(将源mash和stage设为不关心）
			transitionImageLayout(myImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, myImage.mipLevels, fzbBuffer);
			copyBufferToImage(stagingBuffer, myImage, fzbBuffer);

			vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
			vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);

			if (myImage.mipmapEnable) {
				generateMipmaps(myImage, fzbBuffer);
			}
			else {
				//这里的意思就是如果片元着色器想要采样纹理，必须等待纹理传输完成才能开始，并且在采样前修改布局，所以要被copy command阻塞
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

		//image含有很多的subsource，比如颜色数据、mipmap
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

		//image含有很多的subsource，比如颜色数据、mipmap
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