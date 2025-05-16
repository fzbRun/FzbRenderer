#pragma once

#include <string>
#include <vector>
#include <chrono>
#include<stdexcept>
#include<functional>
#include<cstdlib>
#include<cstdint>
#include<limits>
#include<fstream>
#include <random>
#include <iostream>
#include<map>
#include <unordered_map>
#include<set>
#include<filesystem>
#include <algorithm>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include "FzbImage.h"
#include "FzbPipeline.h"
#include "FzbCamera.h"


#ifndef FZB_COMPONENT
#define FZB_COMPONENT

//-----------------------------------------------扩展函数---------------------------------------------------
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebygMessenger) {
	//由于是扩展函数，所以需要通过vkGetInstanceProcAddr获得该函数指针
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebygMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

void GetSemaphoreWin32HandleKHR(VkDevice device, VkSemaphoreGetWin32HandleInfoKHR* handleInfo, HANDLE* handle) {
	auto func = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
	if (func != nullptr) {
		func(device, handleInfo, handle);
	}
}

namespace std {
	template<> struct hash<FzbVertex> {
		size_t operator()(FzbVertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
	}
};
	template<> struct hash<FzbVertex_OnlyPos> {
		size_t operator()(FzbVertex_OnlyPos const& vertex) const {
			// 仅计算 pos 的哈希值
			return hash<glm::vec3>()(vertex.pos);
		}
	};
}

//------------------------------------------------常量----------------------------------------------------
//如果不调试，则关闭校验层
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char*> instanceExtensions_default = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
const std::vector<const char*> validationLayers_default = { "VK_LAYER_KHRONOS_validation" };
const std::vector<const char*> deviceExtensions_default = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
const uint32_t apiVersion_default = VK_API_VERSION_1_0;

//------------------------------------------------------------------类-----------------------------------------------------
class FzbComponent {

public:
//-------------------------------------------------------------------设备-----------------------------------------------------------------------
	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;

	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkQueue computeQueue;

	FzbSwapChainSupportDetails swapChainSupportDetails;
	FzbQueueFamilyIndices queueFamilyIndices;

	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

//----------------------------------------------------------------交换链------------------------------------------------------------
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	VkSurfaceFormatKHR surfaceFormat;
	VkExtent2D extent;

//--------------------------------------------------------------缓冲区---------------------------------------------------------------
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;

	std::vector<std::vector<VkFramebuffer>> framebuffers;

	void fzbCreateCommandBuffers(uint32_t bufferNum = 1) {

		//我们现在想像流水线一样绘制，所以需要多个指令缓冲区
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = this->commandPool;
		//VK_COMMAND_BUFFER_LEVEL_PRIMARY：可以提交到队列执行，但不能从其他命令缓冲区调用。
		//VK_COMMAND_BUFFER_LEVEL_SECONDARY：不能直接提交，但可以从主命令缓冲区调用
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = bufferNum;	//指定分配的命令缓冲区是主命令缓冲区还是辅助命令缓冲区的大小

		//这个函数的第三个参数可以是单个命令缓冲区指针也可以是数组
		this->commandBuffers.resize(bufferNum);
		if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, this->commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate shadow command buffers!");
		}

	}

	/*
	一个交换链图像视图代表一个展示缓冲,一个renderPass代表一帧中所有的输出流程，其中最后的输出图像是一个交换链图像,一个frameBuffer是renderPass的一种实例，将renderPass中规定的输出图像进行填充
	原来的创建帧缓冲的逻辑有问题啊，按照原来的代码，如果使用fast-Vync，那么一共有三个帧缓冲，在流水线中最后的输出对象是三个帧缓冲之一，但是流水线中的每个渲染管线
	都对应于同一个color和depth附件，这就会导致上一帧还在读，而下一帧就在改了，这就会发生脏读啊。
	但是这是创建帧缓冲的问题吗，这应该是同步没有做好的问题啊，如果每个pass都依赖于上一个pass，那么确实不能使用流水线，除非有多个color或depth缓冲，但是还是同步的问题。
	*/
	void fzbCreateFramebuffer(uint32_t swapChainImageViewsSize, VkExtent2D swapChainExtent, uint32_t attachmentSize, std::vector<std::vector<VkImageView>>& attachmentImageViews, VkRenderPass renderPass) {

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

	template<typename T>
	FzbStorageBuffer<T> fzbCreateStorageBuffer(std::vector<T>* bufferData, bool UseExternal = false) {

		uint32_t bufferSize = bufferData->size() * sizeof(T);

		FzbStorageBuffer<uint32_t> stagingBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		stagingBuffer.fzbFillBuffer(bufferData->data());

		FzbStorageBuffer<T> fzbBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UseExternal);
		fzbBuffer.data = *bufferData;

		copyBuffer(logicalDevice, commandPool, graphicsQueue, stagingBuffer.buffer, fzbBuffer.buffer, bufferSize);

		stagingBuffer.clean();

		return fzbBuffer;

	}

	//创造一个空的buffer
	template<typename T>
	FzbStorageBuffer<T> fzbCreateStorageBuffer(uint32_t bufferSize, bool UseExternal = false) {
		return FzbStorageBuffer<T>(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UseExternal);
	}

	template<typename T>
	FzbUniformBuffer<T> fzbCreateUniformBuffers() {
		FzbUniformBuffer<T> fzbBuffer(physicalDevice, logicalDevice, sizeof(T), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		vkMapMemory(logicalDevice, fzbBuffer.memory, 0, sizeof(T), 0, &fzbBuffer.mapped);
		return fzbBuffer;
	}

//------------------------------------------------------------------模型-------------------------------------------------------------------------


	void modelChange(FzbModel& myModel) {

		/*
		glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(5.0f, 5.0f, 5.0f));
		for (int i = 0; i < this->meshs.size(); i++) {
			for (int j = 0; j < this->meshs[i].vertices.size(); j++) {
				glm::vec3 pos = this->meshs[i].vertices[j].pos;
				glm::vec4 changePos = model * glm::vec4(pos, 1.0f);
				this->meshs[i].vertices[j].pos = glm::vec3(changePos.x, changePos.y, changePos.z);
			}
		}
		*/

		for (int i = 0; i < myModel.meshs.size(); i++) {
			if (myModel.meshs[i].vertices.size() > 100) {

				glm::mat4 model = glm::translate(glm::mat4(1.0f), glm::vec3(0.6f, -0.4f, 0.6f));
				for (int j = 0; j < myModel.meshs[i].vertices.size(); j++) {

					myModel.meshs[i].vertices[j].pos = glm::vec3(model * glm::vec4(myModel.meshs[i].vertices[j].pos, 1.0f));

				}

			}
		}

	}

	void fzbMakeAABB(FzbModel& myModel) {

		for (int i = 0; i < myModel.meshs.size(); i++) {

			//left right xyz
			FzbAABBBox AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
			for (int j = 0; j < myModel.meshs[i].indices.size(); j++) {
				glm::vec3 worldPos = myModel.meshs[i].vertices[myModel.meshs[i].indices[j]].pos;
				AABB.leftX = worldPos.x < AABB.leftX ? worldPos.x : AABB.leftX;
				AABB.rightX = worldPos.x > AABB.rightX ? worldPos.x : AABB.rightX;
				AABB.leftY = worldPos.y < AABB.leftY ? worldPos.y : AABB.leftY;
				AABB.rightY = worldPos.y > AABB.rightY ? worldPos.y : AABB.rightY;
				AABB.leftZ = worldPos.z < AABB.leftZ ? worldPos.z : AABB.leftZ;
				AABB.rightZ = worldPos.z > AABB.rightZ ? worldPos.z : AABB.rightZ;
			}
			//对于面，我们给个0.2的宽度
			if (AABB.leftX == AABB.rightX) {
				AABB.leftX = AABB.leftX - 0.01;
				AABB.rightX = AABB.rightX + 0.01;
			}
			if (AABB.leftY == AABB.rightY) {
				AABB.leftY = AABB.leftY - 0.01;
				AABB.rightY = AABB.rightY + 0.01;
			}
			if (AABB.leftZ == AABB.rightZ) {
				AABB.leftZ = AABB.leftZ - 0.01;
				AABB.rightZ = AABB.rightZ + 0.01;
			}
			myModel.meshs[i].AABB = AABB;

		}

		FzbAABBBox AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
		for (int i = 0; i < myModel.meshs.size(); i++) {
			FzbMesh mesh = myModel.meshs[i];
			AABB.leftX = mesh.AABB.leftX < AABB.leftX ? mesh.AABB.leftX : AABB.leftX;
			AABB.rightX = mesh.AABB.rightX > AABB.rightX ? mesh.AABB.rightX : AABB.rightX;
			AABB.leftY = mesh.AABB.leftY < AABB.leftY ? mesh.AABB.leftY : AABB.leftY;
			AABB.rightY = mesh.AABB.rightY > AABB.rightY ? mesh.AABB.rightY : AABB.rightY;
			AABB.leftZ = mesh.AABB.leftZ < AABB.leftZ ? mesh.AABB.leftZ : AABB.leftZ;
			AABB.rightZ = mesh.AABB.rightZ > AABB.rightZ ? mesh.AABB.rightZ : AABB.rightZ;
		}
		myModel.AABB = AABB;

	}

	template<typename T>
	FzbAABBBox fzbMakeAABB(std::vector<T>& vertices) {

		FzbAABBBox AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
		for (int i = 0; i < vertices.size(); i++) {
			glm::vec3 worldPos = vertices[i].pos;
			AABB.leftX = worldPos.x < AABB.leftX ? worldPos.x : AABB.leftX;
			AABB.rightX = worldPos.x > AABB.rightX ? worldPos.x : AABB.rightX;
			AABB.leftY = worldPos.y < AABB.leftY ? worldPos.y : AABB.leftY;
			AABB.rightY = worldPos.y > AABB.rightY ? worldPos.y : AABB.rightY;
			AABB.leftZ = worldPos.z < AABB.leftZ ? worldPos.z : AABB.leftZ;
			AABB.rightZ = worldPos.z > AABB.rightZ ? worldPos.z : AABB.rightZ;
		}
		//对于面，我们给个0.2的宽度
		if (AABB.leftX == AABB.rightX) {
			AABB.leftX = AABB.leftX - 0.01;
			AABB.rightX = AABB.rightX + 0.01;
		}
		if (AABB.leftY == AABB.rightY) {
			AABB.leftY = AABB.leftY - 0.01;
			AABB.rightY = AABB.rightY + 0.01;
		}
		if (AABB.leftZ == AABB.rightZ) {
			AABB.leftZ = AABB.leftZ - 0.01;
			AABB.rightZ = AABB.rightZ + 0.01;
		}
		return AABB;

	}

//------------------------------------------------------------------图像-------------------------------------------------------------------------
	/*
	void fzbCreateImage(FzbImage& myImage, bool UseExternal = false) {

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
			FzbStorageBuffer<uint32_t> stagingBuffer(physicalDevice, logicalDevice, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
			stagingBuffer.fzbFillBuffer(pixels);

			stbi_image_free(pixels);

			myImage.mipLevels = myImage.mipmapEnable ? static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1 : 1;
			createImage(myImage);

			//我们创建的image不知道原来是什么布局，我们也不关心，我们只想要在copy前修改他的布局,并且我们也不关心前面的command，不希望被阻塞(将源mash和stage设为不关心）
			transitionImageLayout(myImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, myImage.mipLevels);
			copyBufferToImage(stagingBuffer.buffer, myImage);

			stagingBuffer.clean();

			if (myImage.mipmapEnable) {
				generateMipmaps(myImage);
			}
			else {
				//这里的意思就是如果片元着色器想要采样纹理，必须等待纹理传输完成才能开始，并且在采样前修改布局，所以要被copy command阻塞
				transitionImageLayout(myImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
			}

			createImageView(myImage);
			createImageSampler(myImage);

			return;

		}

		createImage(myImage, UseExternal);
		createImageView(myImage);
		createImageSampler(myImage);

	}

	void createImage(FzbImage& myImage, bool UseExternal = false) {

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
		allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, myImage.properties);

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

	void transitionImageLayout(FzbImage& myImage, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {

		VkCommandBuffer commandBuffer = beginSingleTimeCommands(logicalDevice, commandPool);

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

		endSingleTimeCommands(logicalDevice, commandPool, commandBuffer, graphicsQueue);

	}

	void copyBufferToImage(VkBuffer buffer,FzbImage& myImage) {

		VkCommandBuffer commandBuffer = beginSingleTimeCommands(logicalDevice, commandPool);

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

		endSingleTimeCommands(logicalDevice, commandPool, commandBuffer, graphicsQueue);

	}

	void generateMipmaps(FzbImage& myImage) {

		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, myImage.format, &formatProperties);
		if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw std::runtime_error("texture image format does not support linear blitting!");
		}

		VkCommandBuffer commandBuffer = beginSingleTimeCommands(logicalDevice, commandPool);
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

		endSingleTimeCommands(logicalDevice, commandPool, commandBuffer, graphicsQueue);

	}

	void createImageView(FzbImage& myImage) {

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

	void createImageSampler(FzbImage& myImage) {

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

	VkFormat fzbFindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {

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

	VkFormat fzbFindDepthFormat(VkPhysicalDevice physicalDevice) {
		return fzbFindSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}

	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void fzbClearTexture(VkCommandBuffer commandBuffer, FzbImage& myImage, VkClearColorValue clearColor, VkImageLayout finalLayout, VkPipelineStageFlagBits shaderStage) {

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

	void fzbCleanImage(FzbImage& myImage) {
		if (myImage.handle)
			CloseHandle(myImage.handle);
		if (myImage.textureSampler) {
			vkDestroySampler(logicalDevice, myImage.textureSampler, nullptr);
		}
		vkDestroyImageView(logicalDevice, myImage.imageView, nullptr);
		vkDestroyImage(logicalDevice, myImage.image, nullptr);
		vkFreeMemory(logicalDevice, myImage.imageMemory, nullptr);
	}
	*/
//-----------------------------------------------------------------描述符-------------------------------------------------------------------------
	VkDescriptorPool descriptorPool;

	void fzbCreateDescriptorPool(std::map<VkDescriptorType, uint32_t> bufferTypeAndNum) {

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

	VkDescriptorSetLayout fzbCreateDescriptLayout(uint32_t descriptorNum, std::vector<VkDescriptorType> descriptorTypes, std::vector<VkShaderStageFlags> descriptorShaderFlags, std::vector<uint32_t> descriptorCounts = std::vector<uint32_t>()) {
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

	VkDescriptorSet fzbCreateDescriptorSet(VkDescriptorSetLayout& descriptorSetLayout) {
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

//-------------------------------------------------------------------管线---------------------------------------------------------------------

//--------------------------------------------------------------------------栏栅和信号量-----------------------------------------------------------------
	FzbSemaphore fzbCreateSemaphore(bool UseExternal = false) {
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkExportSemaphoreCreateInfoKHR exportInfo = {};
		if (UseExternal) {
			exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
			exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
			semaphoreInfo.pNext = &exportInfo;
		}

		FzbSemaphore fzbSemphore = {};
		VkSemaphore semaphore;
		if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
			throw std::runtime_error("failed to create semaphores!");
		}
		fzbSemphore.semaphore = semaphore;

		if (UseExternal) {
			HANDLE handle;
			VkSemaphoreGetWin32HandleInfoKHR handleInfo = {};
			handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
			handleInfo.semaphore = semaphore;
			handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
			GetSemaphoreWin32HandleKHR(logicalDevice, &handleInfo, &handle);
			fzbSemphore.handle = handle;
		}

		return fzbSemphore;

	}

	VkFence fzbCreateFence() {
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		//第一帧可以直接获得信号，而不会阻塞
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		VkFence fence;
		if (vkCreateFence(logicalDevice, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to create semaphores!");
		}

		return fence;

	}

	void fzbCleanSemaphore(FzbSemaphore semaphore) {
		if (semaphore.handle)
			CloseHandle(semaphore.handle);
		vkDestroySemaphore(logicalDevice, semaphore.semaphore, nullptr);
	}

	void fzbCleanFence(VkFence fence) {
		vkDestroyFence(logicalDevice, fence, nullptr);
	}

};

class FzbMainComponent : public FzbComponent {

public:

	void run() {
		camera = FzbCamera(glm::vec3(0.0f, 5.0f, 18.0f));
		fzbInitWindow();
		initVulkan();
		mainLoop();
		clean();
	}

	void initVulkan() {

	}

	GLFWwindow* window;
	bool framebufferResized = false;
	VkInstance instance;	//vulkan实例
	VkDebugUtilsMessengerEXT debugMessenger;	//消息传递者
	VkSurfaceKHR surface;

	std::vector<const char*> instanceExtensions = instanceExtensions_default;
	std::vector<const char*> validationLayers = validationLayers_default;
	uint32_t apiVersion = apiVersion_default;

	void fzbInitWindow(uint32_t width = WIDTH, uint32_t height = HEIGHT, const char* windowName = "未命名", VkBool32 windowResizable = VK_FALSE) {

		glfwInit();

		//阻止GLFW自动创建OpenGL上下文
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		//是否禁止改变窗口大小
		glfwWindowHint(GLFW_RESIZABLE, windowResizable);

		window = glfwCreateWindow(width, height, windowName, nullptr, nullptr);
		//glfwSetFramebufferSizeCallback函数在回调时，需要为我们设置framebufferResized，但他不知道我是谁
		//所以通过对window设置我是谁，从而让回调函数知道我是谁
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwSetCursorPosCallback(window, mouse_callback);
		glfwSetScrollCallback(window, scroll_callback);

	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {

		auto app = reinterpret_cast<FzbMainComponent*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;

	}

	void fzbCreateInstance(const char* appName = "未命名", std::vector<const char*> instanceExtences = instanceExtensions_default, std::vector<const char*> validationLayers = validationLayers_default, uint32_t apiVersion = apiVersion_default) {

		//检测layer
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = appName;
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = apiVersion;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		//扩展就是Vulkan本身没有实现，但被程序员封装后的功能函数，如跨平台的各种函数，把它当成普通函数即可，别被名字唬到了
		auto extensions = getRequiredExtensions(instanceExtences);
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();	//将扩展的具体信息的指针存储在该结构体中

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();	//将校验层的具体信息的指针存储在该结构体中

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;

		}
		else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}


		//VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}

		uint32_t version;

		// 获取 Vulkan 实例的版本
		VkResult result = vkEnumerateInstanceVersion(&version);

		if (result == VK_SUCCESS) {
			uint32_t major = VK_API_VERSION_MAJOR(version);
			uint32_t minor = VK_API_VERSION_MINOR(version);
			uint32_t patch = VK_API_VERSION_PATCH(version);

			std::cout << "Vulkan Version: " << major << "." << minor << "." << patch << std::endl;
		}
		else {
			std::cout << "Failed to enumerate Vulkan version." << std::endl;
		}

	}

	bool checkValidationLayerSupport() {

		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);	//返回可用的层数
		std::vector<VkLayerProperties> availableLayers(layerCount);	//VkLayerProperties是一个结构体，记录层的名字、描述等
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers_default) {

			bool layerFound = false;
			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}

		}

		return true;
	}

	std::vector<const char*> getRequiredExtensions(std::vector<const char*> instanceExtences) {

		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);	//得到glfw所需的扩展数
		//参数1是指针起始位置，参数2是指针终止位置
		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);	//这个扩展是为了打印校验层反映的错误，所以需要知道是否需要校验层
		}
		if (instanceExtences.size() > 0)
			extensions.insert(extensions.end(), instanceExtences.begin(), instanceExtences.end());

		return extensions;
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = nullptr;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
		return VK_FALSE;
	}

	void setupDebugMessenger() {

		if (!enableValidationLayers)
			return;
		VkDebugUtilsMessengerCreateInfoEXT  createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		//通过func的构造函数给debugMessenger赋值
		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}

	}

	void fzbCreateSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface");
		}
	}

//-------------------------------------------------------------------设备-----------------------------------------------------------------------
	std::vector<const char*> deviceExtensions = deviceExtensions_default;
	VkPhysicalDeviceFeatures deviceFeatures;
	void* pNextFeatures = nullptr;

	void pickPhysicalDevice(std::vector<const char*> deviceExtensions = deviceExtensions_default) {

		if (!instance || !surface) {
			throw std::runtime_error("实例或表面未初始化");
		}

		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUS with Vulkan support");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());	//不是按优先级排的

		//按显卡能力进行排序，妈的，不排序默认用的是intel的集成显卡，我的3070只能吃灰
		std::multimap<int, VkPhysicalDevice> candidates;
		for (const auto& device : devices) {
			int score = rateDeviceSuitability(deviceExtensions, device);
			candidates.insert(std::make_pair(score, device));
		}

		if (candidates.rbegin()->first > 0) {
			this->physicalDevice = candidates.rbegin()->second;
			this->msaaSamples = getMaxUsableSampleCount();
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
			std::cout << deviceProperties.deviceName << std::endl;

		}
		else {
			throw std::runtime_error("failed to find a suitable GPU!");
		}

	}

	int rateDeviceSuitability(std::vector<const char*> deviceExtensions, VkPhysicalDevice device) {

		//VkPhysicalDeviceProperties deviceProperties;
		//VkPhysicalDeviceFeatures deviceFeatures;
		//vkGetPhysicalDeviceProperties(device, &deviceProperties);	//设备信息
		//vkGetPhysicalDeviceFeatures(device, &deviceFeatures);		//设备功能

		this->queueFamilyIndices = findQueueFamilies(device);
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(device, &deviceProperties);
		//std::cout << deviceProperties.limits.maxPerStageDescriptorStorageImages << std::endl;

		//检查设备是否支持交换链扩展
		bool extensionsSupport = checkDeviceExtensionSupport(deviceExtensions, device);
		bool swapChainAdequate = false;
		if (extensionsSupport) {
			//判断物理设备的图像和展示功能是否支持
			this->swapChainSupportDetails = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupportDetails.formats.empty() && !swapChainSupportDetails.presentModes.empty();
		}

		if (queueFamilyIndices.isComplete() && extensionsSupport && swapChainAdequate) {
			int score = 0;
			if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
				score += 1000;
			}
			score += deviceProperties.limits.maxImageDimension2D;
			VkPhysicalDeviceFeatures deviceFeatures;
			vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
			if (!deviceFeatures.geometryShader) {	//我可以只要可以支持几何着色器的显卡
				return -1;
			}
			return score;
		}

		return -1;

	}

	FzbSwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {

		FzbSwapChainSupportDetails details;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;

	}

	FzbQueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {

		FzbQueueFamilyIndices indices;
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());	//获得队列系列的详细信息

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			//这里的图像队列是不是说显卡有专门对渲染的优化
			//因为VK_QUEUE_COMPUTE_BIT是说显卡可以通用计算(计算着色器)，而渲染实际上也是一种计算，那么分开两者的原因应该就是是否有专门优化
			//注意支持VK_QUEUE_GRAPHICS_BIT与VK_QUEUE_COMPUTE_BIT的设备默认支持VK_QUEUE_TRANSFER_BIT（用来传递缓冲区数据）
			if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
				indices.graphicsAndComputeFamily = i;
			}

			VkBool32 presentSupport = false;
			//判断i族群是否也支持展示，这里展示的意思是能否将GPU渲染出来的画面传到显示器上，有些显卡可能并未连接到显示器
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (presentSupport) {
				indices.presentFamily = i;
			}

			if (indices.isComplete()) {
				break;
			}
			i++;
		}

		return indices;

	}

	bool checkDeviceExtensionSupport(std::vector<const char*> deviceExtensions, VkPhysicalDevice device) {

		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		//若requiredExtensions空了，说明需要的拓展全有
		//VkPhysicalDeviceProperties deviceProperties;
		//vkGetPhysicalDeviceProperties(device, &deviceProperties);
		//std::cout << deviceProperties.deviceName << std::endl;
		//for (const auto& element : requiredExtensions) {
		//	std::cout << element << std::endl;
		//}
		//std::cout << "    " << std::endl;
		//for (const auto& element : deviceExtensions) {
		//	std::cout << element << std::endl;
		//}
		//std::cout << "    " << std::endl;
		return requiredExtensions.empty();

	}

	VkSampleCountFlagBits getMaxUsableSampleCount() {
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

		VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
		if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
		if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
		if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
		if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
		if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
		if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

		return VK_SAMPLE_COUNT_1_BIT;
	}

	void createLogicalDevice(VkPhysicalDeviceFeatures* deviceFeatures = nullptr, std::vector<const char*> deviceExtensions = deviceExtensions_default, const void* pNextFeatures = nullptr, std::vector<const char*> validationLayers = validationLayers_default) {

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { queueFamilyIndices.graphicsAndComputeFamily.value(), queueFamilyIndices.presentFamily.value() };

		//我们选取的物理设备拥有一定的队列族（功能），但没有创建，现在需要将之创建出来
		//这里的物理设备对应一个逻辑设备，而一个逻辑设备对应两个队列
		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures_default{};
		if (!deviceFeatures) {
			deviceFeatures_default.samplerAnisotropy = VK_TRUE;
		}
		//deviceFeatures.sampleRateShading = VK_TRUE;

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pEnabledFeatures = deviceFeatures ? deviceFeatures : &deviceFeatures_default;
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();
		createInfo.pNext = pNextFeatures;

		// 为设备指定和实例相同的校验层
		// 实际上，新版本的Vulkan已经不再区分二者的校验层，
		// 会自动忽略设备中关于校验层的字段。但是设置一下的话，可以旧版本兼容
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(this->physicalDevice, &createInfo, nullptr, &this->logicalDevice) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(this->logicalDevice, queueFamilyIndices.graphicsAndComputeFamily.value(), 0, &this->graphicsQueue);
		vkGetDeviceQueue(this->logicalDevice, queueFamilyIndices.graphicsAndComputeFamily.value(), 0, &this->computeQueue);
		vkGetDeviceQueue(this->logicalDevice, queueFamilyIndices.presentFamily.value(), 0, &this->presentQueue);

	}

	void fzbCreateDevice(VkPhysicalDeviceFeatures* deviceFeatures = nullptr, std::vector<const char*> deviceExtensions = deviceExtensions_default, const void* pNextFeatures = nullptr, std::vector<const char*> validationLayers = validationLayers_default) {
		pickPhysicalDevice(deviceExtensions);
		createLogicalDevice(deviceFeatures, deviceExtensions, pNextFeatures, validationLayers);
	}

//----------------------------------------------------------------交换链------------------------------------------------------------
	void fzbCreateSwapChain() {

		if (swapChainSupportDetails.formats.empty() || swapChainSupportDetails.presentModes.empty() || !queueFamilyIndices.isComplete()) {
			throw std::runtime_error("设备未初始化");
		}

		this->surfaceFormat = chooseSwapSurfaceFormat();	//主要是surface所展示的纹理的通道数、精度以及色彩空间
		VkPresentModeKHR presentMode = chooseSwapPresentMode();
		this->extent = chooseSwapExtent();

		//如果交换链最小和最大的图像数相等，则确定可支持的图象数就是现在支持的图象数，否则是最小图象数+1
		//如果maxImageCount=0，则表示没有限制（但可能其他地方会限制，无法做到）
		uint32_t imageCount = swapChainSupportDetails.capabilities.minImageCount + 1;
		if (swapChainSupportDetails.capabilities.maxImageCount > 0 && imageCount > swapChainSupportDetails.capabilities.maxImageCount) {
			imageCount = swapChainSupportDetails.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;	//我要传到哪
		createInfo.minImageCount = imageCount;	//规定了交换缓冲区中纹理的数量，如2就是双缓冲
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;	//纹理数组的z，1就表示2D纹理
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		uint32_t queueFamilyIndicesArray[] = { queueFamilyIndices.graphicsAndComputeFamily.value(), queueFamilyIndices.presentFamily.value() };

		//图形队列族负责渲染功能，然后交给交换链；交换链再交给展示队列族呈现到surface上
		if (queueFamilyIndices.graphicsAndComputeFamily != queueFamilyIndices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndicesArray;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			//createInfo.queueFamilyIndexCount = 0;
			//createInfo.pQueueFamilyIndices = nullptr;
		}

		createInfo.preTransform = swapChainSupportDetails.capabilities.currentTransform;	//指明是否需要提前旋转或反转
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		//std::vector<VkImage> swapChainImagesTemp;
		vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
		this->swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, this->swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;

		createSwapChainImageViews();

	}

	void createSwapChainImageViews() {

		//imageViews和交换链中的image数量相同
		this->swapChainImageViews.resize(this->swapChainImages.size());
		for (size_t i = 0; i < this->swapChainImages.size(); i++) {

			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = swapChainImages[i];
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = this->swapChainImageFormat;
			viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;

			if (vkCreateImageView(logicalDevice, &viewInfo, nullptr, &this->swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
			}

		}

	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat() {

		for (const auto& availableFormat : swapChainSupportDetails.formats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}
		return swapChainSupportDetails.formats[0];

	}

	VkPresentModeKHR chooseSwapPresentMode() {
		for (const auto& availablePresentMode : swapChainSupportDetails.presentModes) {
			//交换链如何呈现画面，比如是直接展示还是双缓冲，等等
			//VK_PRESENT_MODE_IMMEDIATE_KHR 渲染完成后立即展示，每帧呈现后都需要等待下一帧渲染完成才能替换，如果下一帧渲染的时快时慢，就会出现卡顿
			//VK_PRESENT_MODE_FIFO_KHR V-Sync,垂直同步，多缓冲，渲染完成后提交画面到后面的缓冲，固定时间（显示器刷新时间）后呈现到显示器上。若缓冲区满了，渲染就会停止（阻塞）
			//VK_PRESENT_MODE_FIFO_RELAXED_KHR 渲染完成后提交画面到后面的缓冲，但是如果这一帧渲染的较慢，导致上一帧在刷新后仍存在，则当前帧提交后立刻呈现，那么就可能导致割裂
			//VK_PRESENT_MODE_MAILBOX_KHR Fast-Sync, 三缓冲，渲染完成后提交画面到后面的缓冲，固定时间（显示器刷新时间）后呈现到显示器上。若缓冲区满了，则画面会替换最后的缓冲区，不会阻塞
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent() {

		VkSurfaceCapabilitiesKHR& capabilities = swapChainSupportDetails.capabilities;
		if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
			return capabilities.currentExtent;
		}
		else {		//某些窗口管理器允许我们在此处使用不同的值，这通过将currentExtent的宽度和高度设置为最大值来表示，我们不想要这样，可以将之重新设置为窗口大小
			int width, height;
			//查询窗口分辨率
			glfwGetFramebufferSize(window, &width, &height);
			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;

		}
	}

//--------------------------------------------------------------缓冲区---------------------------------------------------------------
	void fzbCreateCommandPool() {
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		//VK_COMMAND_POOL_CREATE_TRANSIENT_BIT：提示命令缓冲区经常会重新记录新命令（可能会改变内存分配行为）
		//VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT：允许单独重新记录命令缓冲区，如果没有此标志，则必须一起重置所有命令缓冲区
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();
		if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &this->commandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
	}

	void fzbCreateFramebuffers() {};

//------------------------------------------------------场景与模型-----------------------------------------------------------------
	FzbModel fzbCreateModel(std::string path) {

		FzbModel myModel;

		Assimp::Importer import;
		const aiScene* scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
			std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
			throw std::runtime_error("ERROR::ASSIMP::" + (std::string)import.GetErrorString());
		}

		myModel.directory = path.substr(0, path.find_last_of('/'));
		processNode(scene->mRootNode, scene, myModel);

		return myModel;

	}

	//一个node含有mesh和子node，所以需要递归，将所有的mesh都拿出来
	//所有的实际数据都在scene中，而node中存储的是scene的索引
	void processNode(aiNode* node, const aiScene* scene, FzbModel& myModel) {

		for (uint32_t i = 0; i < node->mNumMeshes; i++) {
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			myModel.meshs.push_back(processMesh(mesh, scene, myModel));
		}

		for (uint32_t i = 0; i < node->mNumChildren; i++) {
			processNode(node->mChildren[i], scene, myModel);
		}

	}

	FzbMesh processMesh(aiMesh* mesh, const aiScene* scene, FzbModel& myModel) {

		std::vector<FzbVertex> vertices;
		std::vector<uint32_t> indices;
		std::vector<FzbTexture> textures;

		for (uint32_t i = 0; i < mesh->mNumVertices; i++) {

			FzbVertex vertex;
			glm::vec3 vector;

			vector.x = mesh->mVertices[i].x;
			vector.y = mesh->mVertices[i].y;
			vector.z = mesh->mVertices[i].z;
			vertex.pos = vector;

			if (mesh->HasNormals()) {

				vector.x = mesh->mNormals[i].x;
				vector.y = mesh->mNormals[i].y;
				vector.z = mesh->mNormals[i].z;
				vertex.normal = vector;

			}

			if (mesh->HasTangentsAndBitangents()) {

				vector.x = mesh->mTangents[i].x;
				vector.y = mesh->mTangents[i].y;
				vector.z = mesh->mTangents[i].z;
				vertex.tangent = vector;

			}

			if (mesh->mTextureCoords[0]) // 网格是否有纹理坐标？
			{
				glm::vec2 vec;
				vec.x = mesh->mTextureCoords[0][i].x;
				vec.y = mesh->mTextureCoords[0][i].y;
				vertex.texCoord = vec;
			}
			else {
				vertex.texCoord = glm::vec2(0.0f, 0.0f);
			}

			vertices.push_back(vertex);

		}

		for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
			aiFace face = mesh->mFaces[i];
			for (uint32_t j = 0; j < face.mNumIndices; j++) {
				indices.push_back(face.mIndices[j]);
			}
		}

		FzbMaterial mat;
		if (mesh->mMaterialIndex >= 0) {

			aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
			aiColor3D color;
			material->Get(AI_MATKEY_COLOR_AMBIENT, color);
			mat.ka = glm::vec4(color.r, color.g, color.b, 1.0);
			material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
			mat.kd = glm::vec4(color.r, color.g, color.b, 1.0);
			material->Get(AI_MATKEY_COLOR_SPECULAR, color);
			mat.ks = glm::vec4(color.r, color.g, color.b, 1.0);
			material->Get(AI_MATKEY_COLOR_EMISSIVE, color);
			mat.ke = glm::vec4(color.r, color.g, color.b, 1.0);

			std::vector<FzbTexture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_albedo", myModel);
			textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());

			//std::vector<Texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular", myModel);
			//textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

			std::vector<FzbTexture> normalMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal", myModel);
			textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());

		}

		return FzbMesh(vertices, indices, textures, mat);

	}

	std::vector<FzbTexture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName, FzbModel& myModel) {

		std::vector<FzbTexture> textures;
		for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
		{
			aiString str;
			mat->GetTexture(type, i, &str);
			bool skip = false;
			for (unsigned int j = 0; j < myModel.textures_loaded.size(); j++)
			{
				if (std::strcmp(myModel.textures_loaded[j].path.data(), str.C_Str()) == 0)
				{
					textures.push_back(myModel.textures_loaded[j]);
					skip = true;
					break;
				}
			}
			if (!skip)
			{   // 如果纹理还没有被加载，则加载它
				FzbTexture texture;
				//texture.id = TextureFromFile(str.C_Str(), directory);
				texture.type = typeName;
				texture.path = myModel.directory + '/' + str.C_Str();
				textures.push_back(texture);
				myModel.textures_loaded.push_back(texture); // 添加到已加载的纹理中
			}
		}

		return textures;

	}

	void simplify(FzbModel& myModel) {

		std::vector<FzbMesh> simpleMeshs;
		for (int i = 0; i < myModel.meshs.size(); i++) {
			if (myModel.meshs[i].indices.size() < 100) {	//2950
				simpleMeshs.push_back(myModel.meshs[i]);
			}
		}
		myModel.meshs = simpleMeshs;
	}

	//将一个mesh的冗余顶点删除
	void fzbOptimizeMesh(FzbMesh* mesh) {
		std::unordered_map<FzbVertex, uint32_t> uniqueVerticesMap{};
		std::vector<FzbVertex> uniqueVertices;
		std::vector<uint32_t> uniqueIndices;
		for (uint32_t j = 0; j < mesh->indices.size(); j++) {
			FzbVertex vertex = mesh->vertices[mesh->indices[j]];
			if (uniqueVerticesMap.count(vertex) == 0) {
				uniqueVerticesMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
				uniqueVertices.push_back(vertex);
			}
			uniqueIndices.push_back(uniqueVerticesMap[vertex]);
		}
		mesh->vertices = uniqueVertices;
		mesh->indices = uniqueIndices;
	}

	//将一个模型的所有mesh的顶点和索引存入一个数组，并删除冗余顶点
	template<typename T>
	void fzbOptimizeModel(FzbModel* myModel, std::vector<T>& vertices, std::vector<uint32_t>& indices) {
		uint32_t indexOffset = 0;
		for (uint32_t meshIndex = 0; meshIndex < myModel->meshs.size(); meshIndex++) {

			FzbMesh* mesh = &myModel->meshs[meshIndex];
			fzbOptimizeMesh(mesh);
			vertices.insert(vertices.end(), mesh->vertices.begin(), mesh->vertices.end());

			//因为assimp是按一个mesh一个mesh的存，所以每个indices都是相对一个mesh的，当我们将每个mesh的顶点存到一起时，indices就会出错，我们需要增加索引
			for (uint32_t index = 0; index < mesh->indices.size(); index++) {
				mesh->indices[index] += indexOffset;
			}
			//meshIndexInIndices.push_back(this->indices.size());
			indexOffset += mesh->vertices.size();
			indices.insert(indices.end(), mesh->indices.begin(), mesh->indices.end());
		}

		std::unordered_map<T, uint32_t> uniqueVerticesMap{};
		std::vector<T> uniqueVertices;
		std::vector<uint32_t> uniqueIndices;
		for (uint32_t j = 0; j < indices.size(); j++) {
			T vertex = std::is_same_v<T, FzbVertex> ? vertices[indices[j]] : T(vertices[indices[j]]);
			//if constexpr (std::is_same_v<T, Vertex>) {
			//	vertex = vertices[indices[j]];
			//}
			//else {
			//	vertex = T(vertices[indices[j]]);
			//}
			if (uniqueVerticesMap.count(vertex) == 0) {
				uniqueVerticesMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
				uniqueVertices.push_back(vertex);
			}
			uniqueIndices.push_back(uniqueVerticesMap[vertex]);
		}
		vertices = uniqueVertices;
		indices = uniqueIndices;

	}

	template<typename T>
	void fzbOptimizeScene(FzbScene* myScene, std::vector<T>& vertices, std::vector<uint32_t>& indices) {
		uint32_t indexOffset = 0;
		for (uint32_t meshIndex = 0; meshIndex < myScene->sceneModels.size(); meshIndex++) {

			FzbModel* model = myScene->sceneModels[meshIndex];
			std::vector<T> modelVertices;
			std::vector<uint32_t> modelIndices;
			fzbOptimizeModel<T>(model, modelVertices, modelIndices);
			vertices.insert(vertices.end(), modelVertices.begin(), modelVertices.end());

			//因为assimp是按一个mesh一个mesh的存，所以每个indices都是相对一个mesh的，当我们将每个mesh的顶点存到一起时，indices就会出错，我们需要增加索引
			for (uint32_t index = 0; index < modelIndices.size(); index++) {
				modelIndices[index] += indexOffset;
			}
			//meshIndexInIndices.push_back(this->indices.size());
			indexOffset += modelVertices.size();
			indices.insert(indices.end(), modelIndices.begin(), modelIndices.end());
		}

		std::unordered_map<T, uint32_t> uniqueVerticesMap{};
		std::vector<T> uniqueVertices;
		std::vector<uint32_t> uniqueIndices;
		for (uint32_t j = 0; j < indices.size(); j++) {
			T vertex = std::is_same_v<T, FzbVertex> ? vertices[indices[j]] : T(vertices[indices[j]]);
			//if constexpr (std::is_same_v<T, Vertex>) {
			//	vertex = vertices[indices[j]];
			//}
			//else {
			//	vertex = T(vertices[indices[j]]);
			//}
			if (uniqueVerticesMap.count(vertex) == 0) {
				uniqueVerticesMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
				uniqueVertices.push_back(vertex);
			}
			uniqueIndices.push_back(uniqueVerticesMap[vertex]);
		}
		vertices = uniqueVertices;
		indices = uniqueIndices;
	}
//--------------------------------------------------------------图像-----------------------------------------------------------------
	virtual void createImages() {};

//---------------------------------------------------------------------------渲染循环---------------------------------------------------------------------
	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			processInput(window);
			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(logicalDevice);

	}

	void processInput(GLFWwindow* window)
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			camera.ProcessKeyboard(FORWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			camera.ProcessKeyboard(BACKWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			camera.ProcessKeyboard(LEFT, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			camera.ProcessKeyboard(RIGHT, deltaTime);
	}

	virtual void drawFrame() = 0;

	void recreateSwapChain() {

		int width = 0, height = 0;
		//获得当前window的大小
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(logicalDevice);
		cleanupSwapChain();
		fzbCreateSwapChain();
		createImages();
		fzbCreateFramebuffers();
	}

	virtual void cleanupImages() {

	}

	void cleanupSwapChain() {

		cleanupImages();
		for (size_t i = 0; i < framebuffers.size(); i++) {
			for (int j = 0; j < framebuffers[i].size(); j++) {
				vkDestroyFramebuffer(logicalDevice, framebuffers[i][j], nullptr);
			}
		}
		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			vkDestroyImageView(logicalDevice, swapChainImageViews[i], nullptr);
		}
		vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);
	}

	virtual void clean() = 0;


};

#endif