#pragma once

#include "./FzbComponent.h"
#include "../FzbScene/FzbScene.h"
#include "../FzbCamera/FzbCamera.h"

#ifndef FZB_MAIN_COMPONENT_H
#define FZB_MAIN_COMPONENT_H

//-----------------------------------------------扩展函数---------------------------------------------------
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebygMessenger);
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

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

struct FzbMainComponent : public FzbComponent {
public:
	float lastTime = 0.0f;
	float deltaTime = 0.0f;
	bool firstMouse = true;
	float lastX;
	float lastY;
	bool framebufferResized = false;

	FzbMainScene mainScene = FzbMainScene();
	inline static FzbCamera* camera = nullptr;

	void mouse_callback(double xposIn, double yposIn);
	void scroll_callback(double xoffset, double yoffset);

	GLFWwindow* window;
	VkInstance instance;	//vulkan实例
	VkDebugUtilsMessengerEXT debugMessenger;	//消息传递者
	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkQueue computeQueue;
	std::vector<const char*> instanceExtensions = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
	uint32_t apiVersion = VK_API_VERSION_1_2;
	std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };
	std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME };
	VkPhysicalDeviceFeatures deviceFeatures;
	void* pNextFeatures = nullptr;

	FzbSwapChainSupportDetails swapChainSupportDetails;
	FzbQueueFamilyIndices queueFamilyIndices;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	VkSurfaceFormatKHR surfaceFormat;
	VkExtent2D extent;
	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

	VkCommandPool commandPool;

	FzbMainComponent();
	void setMainScenePath(std::string scenePath);

	void init(const char* name);
	void initWindow(uint32_t width = 512, uint32_t height = 512, const char* windowName = "未命名", VkBool32 windowResizable = VK_FALSE);
	void createInstance(const char* appName = "未命名", std::vector<const char*> instanceExtences = instanceExtensions_default, std::vector<const char*> validationLayers = validationLayers_default, uint32_t apiVersion = apiVersion_default);
	bool checkValidationLayerSupport();
	std::vector<const char*> getRequiredExtensions(std::vector<const char*> instanceExtences);
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
	void setupDebugMessenger();
	void createSurface();
	//-------------------------------------------------------------------设备-----------------------------------------------------------------------
	void pickPhysicalDevice(std::vector<const char*> deviceExtensions = deviceExtensions_default);
	int rateDeviceSuitability(std::vector<const char*> deviceExtensions, VkPhysicalDevice device);
	FzbSwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
	FzbQueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
	bool checkDeviceExtensionSupport(std::vector<const char*> deviceExtensions, VkPhysicalDevice device);
	VkSampleCountFlagBits getMaxUsableSampleCount();
	VkPhysicalDeviceFeatures2 createPhysicalDeviceFeatures(VkPhysicalDeviceFeatures deviceFeatures, VkPhysicalDeviceVulkan11Features* vk11Features = nullptr, VkPhysicalDeviceVulkan12Features* vk12Features = nullptr);
	void createLogicalDevice(VkPhysicalDeviceFeatures* deviceFeatures = nullptr, std::vector<const char*> deviceExtensions = deviceExtensions_default, const void* pNextFeatures = nullptr, std::vector<const char*> validationLayers = validationLayers_default);
	//void createDevice(VkPhysicalDeviceFeatures* deviceFeatures = nullptr, std::vector<const char*> deviceExtensions = deviceExtensions_default, const void* pNextFeatures = nullptr, std::vector<const char*> validationLayers = validationLayers_default);
	void createDevice();
	//----------------------------------------------------------------交换链------------------------------------------------------------
	void createSwapChain();
	void createSwapChainImageViews();
	VkSurfaceFormatKHR chooseSwapSurfaceFormat();
	VkPresentModeKHR chooseSwapPresentMode();
	VkExtent2D chooseSwapExtent();
	//--------------------------------------------------------------缓冲区---------------------------------------------------------------
	void createCommandPool();
	//-------------------------------------------------------------场景---------------------------------------------------------------
	void createScene();
	//---------------------------------------------------------------------------渲染循环---------------------------------------------------------------------
	void processInput();
	void clean();

	static void setResolution(VkExtent2D resolution);
	static VkExtent2D getResolution();

private:
	inline  static VkExtent2D resolution = { 512, 512 };

};

#endif