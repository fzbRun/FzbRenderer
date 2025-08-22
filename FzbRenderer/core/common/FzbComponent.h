#pragma once

#include "FzbImage.h"
#include "FzbRenderPass.h"
#include "FzbPipeline.h"
#include "FzbDescriptor.h"
#include "FzbScene.h"
#include "FzbCamera.h"

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

#ifndef FZB_COMPONENT
#define FZB_COMPONENT

//-----------------------------------------------扩展函数---------------------------------------------------
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebygMessenger);

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

//void GetSemaphoreWin32HandleKHR(VkDevice device, VkSemaphoreGetWin32HandleInfoKHR* handleInfo, HANDLE* handle);

//namespace std {
//	template<> struct hash<FzbVertex> {
//		size_t operator()(FzbVertex const& vertex) const {
//			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
//	}
//};
//	template<> struct hash<FzbVertex_OnlyPos> {
//		size_t operator()(FzbVertex_OnlyPos const& vertex) const {
//			// 仅计算 pos 的哈希值
//			return hash<glm::vec3>()(vertex.pos);
//		}
//	};
//}

//------------------------------------------------常量----------------------------------------------------
//如果不调试，则关闭校验层
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char*> instanceExtensions_default = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
const std::vector<const char*> validationLayers_default = { "VK_LAYER_KHRONOS_validation" };
const std::vector<const char*> deviceExtensions_default = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME };
const uint32_t apiVersion_default = VK_API_VERSION_1_2;

struct FzbMainComponent;
//------------------------------------------------------------------类-----------------------------------------------------
struct FzbComponent {

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

	VkDescriptorPool descriptorPool;
	//----------------------------------------------------------函数---------------------------------------------------------------

	void initComponent(FzbMainComponent* renderer);

	//std::vector<std::vector<VkFramebuffer>> framebuffers;

	void fzbCreateCommandBuffers(uint32_t bufferNum = 1);

	/*
	一个交换链图像视图代表一个展示缓冲,一个renderPass代表一帧中所有的输出流程，其中最后的输出图像是一个交换链图像,一个frameBuffer是renderPass的一种实例，将renderPass中规定的输出图像进行填充
	原来的创建帧缓冲的逻辑有问题啊，按照原来的代码，如果使用fast-Vync，那么一共有三个帧缓冲，在流水线中最后的输出对象是三个帧缓冲之一，但是流水线中的每个渲染管线
	都对应于同一个color和depth附件，这就会导致上一帧还在读，而下一帧就在改了，这就会发生脏读啊。
	但是这是创建帧缓冲的问题吗，这应该是同步没有做好的问题啊，如果每个pass都依赖于上一个pass，那么确实不能使用流水线，除非有多个color或depth缓冲，但是还是同步的问题。
	*/
	/*
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
	*/
	template<typename T>
	FzbBuffer fzbComponentCreateStorageBuffer(std::vector<T>* bufferData, bool UseExternal = false) {
		return fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, bufferData->data(), bufferData->size() * sizeof(T), UseExternal);
	}

	//创造一个空的buffer
	FzbBuffer fzbComponentCreateStorageBuffer(uint32_t bufferSize, bool UseExternal = false);

	template<typename T>
	FzbBuffer fzbComponentCreateUniformBuffers() {
		return fzbCreateUniformBuffers(physicalDevice, logicalDevice, sizeof(T));
	}

//------------------------------------------------------------------模型和shader-------------------------------------------------------------------------
	virtual FzbVertexFormat getComponentVertexFormat();
//------------------------------------------------------------------图像-------------------------------------------------------------------------
//-----------------------------------------------------------------描述符-------------------------------------------------------------------------

	void fzbComponentCreateDescriptorPool(std::map<VkDescriptorType, uint32_t> bufferTypeAndNum);

	VkDescriptorSetLayout fzbComponentCreateDescriptLayout(uint32_t descriptorNum, std::vector<VkDescriptorType> descriptorTypes, std::vector<VkShaderStageFlags> descriptorShaderFlags, std::vector<uint32_t> descriptorCounts = std::vector<uint32_t>());

	VkDescriptorSet fzbComponentCreateDescriptorSet(VkDescriptorSetLayout& descriptorSetLayout);
//-------------------------------------------------------------------管线---------------------------------------------------------------------

//--------------------------------------------------------------------------栏栅和信号量-----------------------------------------------------------------
	//FzbSemaphore fzbCreateSemaphore(bool UseExternal = false);

	VkFence fzbCreateFence();

	void fzbCleanSemaphore(FzbSemaphore semaphore);

	void fzbCleanFence(VkFence fence);

};

struct FzbMainComponent : public FzbComponent {

public:

	//uint32_t WIDTH = 512;
	//uint32_t HEIGHT = 512;

	FzbCamera* camera;
	float lastTime = 0.0f;
	float deltaTime = 0.0f;
	bool firstMouse = true;
	float lastX;
	float lastY;
	void mouse_callback(double xposIn, double yposIn);
	void scroll_callback(double xoffset, double yoffset);

	void run();
	void initVulkan();

	GLFWwindow* window;
	bool framebufferResized = false;
	VkInstance instance;	//vulkan实例
	VkDebugUtilsMessengerEXT debugMessenger;	//消息传递者
	VkSurfaceKHR surface;

	std::vector<const char*> instanceExtensions = instanceExtensions_default;
	std::vector<const char*> validationLayers = validationLayers_default;
	uint32_t apiVersion = apiVersion_default;

	void fzbInitWindow(uint32_t width = 512, uint32_t height = 512, const char* windowName = "未命名", VkBool32 windowResizable = VK_FALSE);

	//static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

	void fzbCreateInstance(const char* appName = "未命名", std::vector<const char*> instanceExtences = instanceExtensions_default, std::vector<const char*> validationLayers = validationLayers_default, uint32_t apiVersion = apiVersion_default);

	bool checkValidationLayerSupport();

	std::vector<const char*> getRequiredExtensions(std::vector<const char*> instanceExtences);

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

	//static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

	void fzbCetupDebugMessenger();

	void fzbCreateSurface();

//-------------------------------------------------------------------设备-----------------------------------------------------------------------
	std::vector<const char*> deviceExtensions = deviceExtensions_default;
	VkPhysicalDeviceFeatures deviceFeatures;
	void* pNextFeatures = nullptr;

	void pickPhysicalDevice(std::vector<const char*> deviceExtensions = deviceExtensions_default);

	int rateDeviceSuitability(std::vector<const char*> deviceExtensions, VkPhysicalDevice device);

	FzbSwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

	FzbQueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

	bool checkDeviceExtensionSupport(std::vector<const char*> deviceExtensions, VkPhysicalDevice device);

	VkSampleCountFlagBits getMaxUsableSampleCount();

	VkPhysicalDeviceFeatures2 createPhysicalDeviceFeatures(VkPhysicalDeviceFeatures deviceFeatures, VkPhysicalDeviceVulkan11Features* vk11Features = nullptr, VkPhysicalDeviceVulkan12Features* vk12Features = nullptr);

	void createLogicalDevice(VkPhysicalDeviceFeatures* deviceFeatures = nullptr, std::vector<const char*> deviceExtensions = deviceExtensions_default, const void* pNextFeatures = nullptr, std::vector<const char*> validationLayers = validationLayers_default);

	void fzbCreateDevice(VkPhysicalDeviceFeatures* deviceFeatures = nullptr, std::vector<const char*> deviceExtensions = deviceExtensions_default, const void* pNextFeatures = nullptr, std::vector<const char*> validationLayers = validationLayers_default);

//----------------------------------------------------------------交换链------------------------------------------------------------
	void fzbCreateSwapChain();

	void createSwapChainImageViews();

	VkSurfaceFormatKHR chooseSwapSurfaceFormat();

	VkPresentModeKHR chooseSwapPresentMode();

	VkExtent2D chooseSwapExtent();

//--------------------------------------------------------------缓冲区---------------------------------------------------------------
	void fzbCreateCommandPool();

	//void fzbCreateFramebuffers() {};

//------------------------------------------------------场景与模型-----------------------------------------------------------------
//--------------------------------------------------------------图像-----------------------------------------------------------------
	virtual void createImages();

//---------------------------------------------------------------------------渲染循环---------------------------------------------------------------------
	void mainLoop();

	void processInput(GLFWwindow* window);

	virtual void drawFrame() = 0;

	void recreateSwapChain(std::vector<FzbRenderPass*> renderPasses);

	virtual void cleanupImages();

	void fzbCleanupSwapChain();

	virtual void clean() = 0;


};

#endif