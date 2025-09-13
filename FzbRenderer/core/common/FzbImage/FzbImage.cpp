#include "FzbImage.h"
#include <stdexcept>
#include "../FzbRenderer.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

void FzbImage::initImage() {
	if (this->texturePath) {
		int texWidth, texHeight, texChannels;
		stbi_uc* pixels = stbi_load(this->texturePath, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;
		if (!pixels) {
			throw std::runtime_error("failed to load texture image!");
		}
		this->width = texWidth;
		this->height = texHeight;

		//��ɫ���в���������ͬ��ֻ��ҪGPU�ɼ������ԺͶ��㻺����һ���������Ƚ����ݴ浽�ݴ滺�����Ŵ浽GPU����������
		FzbBuffer stagingBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		stagingBuffer.fzbCreateBuffer();
		stagingBuffer.fzbFillBuffer(pixels);

		stbi_image_free(pixels);

		this->mipLevels = this->mipmapEnable ? static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1 : 1;
		createImage();

		//���Ǵ�����image��֪��ԭ����ʲô���֣�����Ҳ�����ģ�����ֻ��Ҫ��copyǰ�޸����Ĳ���,��������Ҳ������ǰ���command����ϣ��������(��Դmash��stage��Ϊ�����ģ�
		transitionImageLayout(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, this->mipLevels);
		copyBufferToImage(stagingBuffer.buffer);

		stagingBuffer.clean();

		if (this->mipmapEnable) {
			generateMipmaps();
		}
		else {
			//�������˼�������ƬԪ��ɫ����Ҫ������������ȴ���������ɲ��ܿ�ʼ�������ڲ���ǰ�޸Ĳ��֣�����Ҫ��copy command����
			transitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
		}

		createImageView();
		createImageSampler();

		return;
	}
	createImage();
	createImageView();
	createImageSampler();
}
void FzbImage::createImage() {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = this->type;
	imageInfo.extent.width = static_cast<uint32_t>(this->width);
	imageInfo.extent.height = static_cast<uint32_t>(this->height);
	imageInfo.extent.depth = static_cast<uint32_t>(this->depth);
	imageInfo.mipLevels = this->mipLevels;
	imageInfo.arrayLayers = 1;
	//findSupportedFormat(physicalDevice, { VK_FORMAT_R8G8B8A8_UINT }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
	imageInfo.format = this->format;
	//tiling��һ�����ݴ�Ÿ�ʽ��linear�ǰ��д�ţ���optimal���ǣ����岻֪�����������С��з��ʶ��Ѻã������Ը�˹ģ������ǰ�߷ǳ���
	//VK_IMAGE_TILING_LINEAR: Texels are laid out in row-major order like our pixels array�����������Ҫ����image�ͱ��������ַ�ʽ
	//VK_IMAGE_TILING_OPTIMAL: Texels are laid out in an implementation defined order for optimal access ������ǲ�����ʣ������ַ�ʽ���Ч
	//��˼����������Ƿ�������������Ҫ�õ�һ�֣�������ǲ����ʣ�ֻ����������Ӳ��ȥ�㣬��ڶ����ڲ����Ż������Ч
	imageInfo.tiling = this->tiling;
	//������һ������ѹ����ʽ
	//VK_IMAGE_LAYOUT_UNDEFINED: Not usable by the GPU and the very first transition will discard the texels.
	//VK_IMAGE_LAYOUT_PREINITIALIZED: Not usable by the GPU, but the first transition will preserve the texels.
	//�������˼�����Ƿ����imageԭ�е����ݲ��ֽ����޸ģ�������������ԭ�в������޸�һ���֣���ʹ��ǰ�ߣ������VK_IMAGE_LAYOUT_UNDEFINED����֮����
	//�������������ǽ��ⲿ��������ݷŵ���image�У����Բ�����imageԭ������
	/*
	ע�⣬vulkan��image��layout����Ӱ�쵽cuda��ȡ�����ݣ���Ϊ�Դ��е�ʵ�����ݲ�����Ϊlayout�ı仯�������仯
	ʵ���ϣ�layout�ı仯ֻ�Ǹ���vulkan����ʹ�ò�ͬ�ķô�ָ���VK_IMAGE_USAGE_TRANSFER_SRC_BIT��˵��image��ֻ���ģ����ܷô����Ż�ʲô��
	*/
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = this->usage;
	//�����ʲôϡ��洢���
	imageInfo.samples = this->sampleCount;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.flags = 0;

	VkExternalMemoryImageCreateInfo externalMemoryImageCreateInfo{};
	if (UseExternal) {
		externalMemoryImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
		externalMemoryImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
		imageInfo.pNext = &externalMemoryImageCreateInfo;
	}

	if (vkCreateImage(logicalDevice, &imageInfo, nullptr, &this->image) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image!");
	}

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(logicalDevice, this->image, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, this->properties);

	VkExportMemoryAllocateInfo exportInfo = {};
	if (UseExternal) {
		exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
		exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
		allocInfo.pNext = &exportInfo;
	}

	if (vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &this->imageMemory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate image memory!");
	}

	vkBindImageMemory(logicalDevice, this->image, this->imageMemory, 0);

	if (UseExternal) {
		VkMemoryGetWin32HandleInfoKHR handleInfo = {};
		handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
		handleInfo.memory = this->imageMemory;
		handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
		GetMemoryWin32HandleKHR(&handleInfo, &this->handle);
	}

}
void FzbImage::transitionImageLayout(VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

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
	barrier.image = this->image;
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
	else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
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

	endSingleTimeCommands(commandBuffer);
}
void FzbImage::copyBufferToImage(VkBuffer buffer) {
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = { 0, 0, 0 };
	region.imageExtent = { this->width, this->height, this->depth };

	vkCmdCopyBufferToImage(commandBuffer, buffer, this->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	endSingleTimeCommands(commandBuffer);
}
void FzbImage::generateMipmaps() {
	VkFormatProperties formatProperties;
	vkGetPhysicalDeviceFormatProperties(FzbRenderer::globalData.physicalDevice, this->format, &formatProperties);
	if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
		throw std::runtime_error("texture image format does not support linear blitting!");
	}

	VkCommandBuffer commandBuffer = beginSingleTimeCommands();
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.image = this->image;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.subresourceRange.levelCount = 1;

	int32_t mipWidth = this->width;
	int32_t mipHeight = this->height;
	int32_t mipDepth = this->depth;

	for (uint32_t i = 1; i < this->mipLevels; i++) {

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
			this->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			this->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
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
	barrier.subresourceRange.baseMipLevel = this->mipLevels - 1;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

	vkCmdPipelineBarrier(commandBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
		0, nullptr,
		0, nullptr,
		1, &barrier);

	endSingleTimeCommands(commandBuffer);
}
void FzbImage::createImageView() {
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = this->image;
	viewInfo.viewType = this->viewType;// == VK_IMAGE_TYPE_2D ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_3D;
	viewInfo.format = this->format;
	viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	viewInfo.subresourceRange.aspectMask = this->aspectFlags;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = this->mipLevels;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	if (vkCreateImageView(FzbRenderer::globalData.logicalDevice, &viewInfo, nullptr, &this->imageView) != VK_SUCCESS) {
		throw std::runtime_error("failed to create image views!");
	}
}
void  FzbImage::createImageSampler() {
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(FzbRenderer::globalData.physicalDevice, &properties);

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = this->filter;
	samplerInfo.minFilter = this->filter;
	samplerInfo.addressModeU = this->addressMode;	//�����Ϊrepeat����ӰMap�߽���������
	samplerInfo.addressModeV = this->addressMode;
	samplerInfo.addressModeW = this->addressMode;
	samplerInfo.anisotropyEnable = this->anisotropyEnable;
	samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.mipLodBias = 0.0f;
	samplerInfo.minLod = 0.0f;
	samplerInfo.maxLod = static_cast<float>(this->mipLevels);

	if (vkCreateSampler(FzbRenderer::globalData.logicalDevice, &samplerInfo, nullptr, &this->textureSampler) != VK_SUCCESS) {
		throw std::runtime_error("failed to create texture sampler!");
	}
}
void FzbImage::fzbClearTexture(VkCommandBuffer commandBuffer, VkClearColorValue clearColor, VkImageLayout finalLayout, VkPipelineStageFlagBits shaderStage) {
	//#ifndef Voxelization_Block
	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

	//image���кܶ��subsource��������ɫ���ݡ�mipmap
	barrier.image = this->image;
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
	vkCmdClearColorImage(commandBuffer, this->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &subresourceRange);

	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = finalLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

	//image���кܶ��subsource��������ɫ���ݡ�mipmap
	barrier.image = this->image;
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
void FzbImage::clean() {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;
	if (this->handle)
		CloseHandle(this->handle);
	if (this->textureSampler) {
		vkDestroySampler(logicalDevice, this->textureSampler, nullptr);
	}
	if (this->image) {
		vkDestroyImageView(logicalDevice, this->imageView, nullptr);
		vkDestroyImage(logicalDevice, this->image, nullptr);
		vkFreeMemory(logicalDevice, this->imageMemory, nullptr);
	}
}

VkFormat fzbFindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
	for (VkFormat format : candidates) {

		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(FzbRenderer::globalData.physicalDevice, format, &props);
		if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
			return format;
		}
		else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
			return format;
		}

	}

	throw std::runtime_error("failed to find supported format!");
}
VkFormat fzbFindDepthFormat() {
	return fzbFindSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}
bool hasStencilComponent(VkFormat format) {
	return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}
void fzbCopyImageToImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImage dstImage, VkExtent3D copyExtent) {
	VkImageCopy copyRegion{};
	copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copyRegion.srcSubresource.mipLevel = 0;
	copyRegion.srcSubresource.baseArrayLayer = 0;
	copyRegion.srcSubresource.layerCount = 1;
	copyRegion.srcOffset = { 0,0,0 };

	copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copyRegion.dstSubresource.mipLevel = 0;
	copyRegion.dstSubresource.baseArrayLayer = 0;
	copyRegion.dstSubresource.layerCount = 1;
	copyRegion.dstOffset = { 0,0,0 };

	copyRegion.extent.width = copyExtent.width;
	copyRegion.extent.height = copyExtent.height;
	copyRegion.extent.depth = copyExtent.depth;

	vkCmdCopyImage(
		commandBuffer,
		srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1, &copyRegion
	);
}
