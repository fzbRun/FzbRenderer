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

//-----------------------------------------------��չ����---------------------------------------------------
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
//			// ������ pos �Ĺ�ϣֵ
//			return hash<glm::vec3>()(vertex.pos);
//		}
//	};
//}

//------------------------------------------------����----------------------------------------------------
//��������ԣ���ر�У���
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
//------------------------------------------------------------------��-----------------------------------------------------
struct FzbComponent {

public:
//-------------------------------------------------------------------�豸-----------------------------------------------------------------------
	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;

	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkQueue computeQueue;

	FzbSwapChainSupportDetails swapChainSupportDetails;
	FzbQueueFamilyIndices queueFamilyIndices;

	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

//----------------------------------------------------------------������------------------------------------------------------------
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	VkSurfaceFormatKHR surfaceFormat;
	VkExtent2D extent;

//--------------------------------------------------------------������---------------------------------------------------------------
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;

	VkDescriptorPool descriptorPool;
	//----------------------------------------------------------����---------------------------------------------------------------

	void initComponent(FzbMainComponent* renderer);

	//std::vector<std::vector<VkFramebuffer>> framebuffers;

	void fzbCreateCommandBuffers(uint32_t bufferNum = 1);

	/*
	һ��������ͼ����ͼ����һ��չʾ����,һ��renderPass����һ֡�����е�������̣������������ͼ����һ��������ͼ��,һ��frameBuffer��renderPass��һ��ʵ������renderPass�й涨�����ͼ��������
	ԭ���Ĵ���֡������߼������Ⱑ������ԭ���Ĵ��룬���ʹ��fast-Vync����ôһ��������֡���壬����ˮ���������������������֡����֮һ��������ˮ���е�ÿ����Ⱦ����
	����Ӧ��ͬһ��color��depth��������ͻᵼ����һ֡���ڶ�������һ֡���ڸ��ˣ���ͻᷢ���������
	�������Ǵ���֡�������������Ӧ����ͬ��û�����õ����Ⱑ�����ÿ��pass����������һ��pass����ôȷʵ����ʹ����ˮ�ߣ������ж��color��depth���壬���ǻ���ͬ�������⡣
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

	//����һ���յ�buffer
	FzbBuffer fzbComponentCreateStorageBuffer(uint32_t bufferSize, bool UseExternal = false);

	template<typename T>
	FzbBuffer fzbComponentCreateUniformBuffers() {
		return fzbCreateUniformBuffers(physicalDevice, logicalDevice, sizeof(T));
	}

//------------------------------------------------------------------ģ�ͺ�shader-------------------------------------------------------------------------
	virtual FzbVertexFormat getComponentVertexFormat();
//------------------------------------------------------------------ͼ��-------------------------------------------------------------------------
//-----------------------------------------------------------------������-------------------------------------------------------------------------

	void fzbComponentCreateDescriptorPool(std::map<VkDescriptorType, uint32_t> bufferTypeAndNum);

	VkDescriptorSetLayout fzbComponentCreateDescriptLayout(uint32_t descriptorNum, std::vector<VkDescriptorType> descriptorTypes, std::vector<VkShaderStageFlags> descriptorShaderFlags, std::vector<uint32_t> descriptorCounts = std::vector<uint32_t>());

	VkDescriptorSet fzbComponentCreateDescriptorSet(VkDescriptorSetLayout& descriptorSetLayout);
//-------------------------------------------------------------------����---------------------------------------------------------------------

//--------------------------------------------------------------------------��դ���ź���-----------------------------------------------------------------
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
	VkInstance instance;	//vulkanʵ��
	VkDebugUtilsMessengerEXT debugMessenger;	//��Ϣ������
	VkSurfaceKHR surface;

	std::vector<const char*> instanceExtensions = instanceExtensions_default;
	std::vector<const char*> validationLayers = validationLayers_default;
	uint32_t apiVersion = apiVersion_default;

	void fzbInitWindow(uint32_t width = 512, uint32_t height = 512, const char* windowName = "δ����", VkBool32 windowResizable = VK_FALSE);

	//static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

	void fzbCreateInstance(const char* appName = "δ����", std::vector<const char*> instanceExtences = instanceExtensions_default, std::vector<const char*> validationLayers = validationLayers_default, uint32_t apiVersion = apiVersion_default);

	bool checkValidationLayerSupport();

	std::vector<const char*> getRequiredExtensions(std::vector<const char*> instanceExtences);

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

	//static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);

	void fzbCetupDebugMessenger();

	void fzbCreateSurface();

//-------------------------------------------------------------------�豸-----------------------------------------------------------------------
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

//----------------------------------------------------------------������------------------------------------------------------------
	void fzbCreateSwapChain();

	void createSwapChainImageViews();

	VkSurfaceFormatKHR chooseSwapSurfaceFormat();

	VkPresentModeKHR chooseSwapPresentMode();

	VkExtent2D chooseSwapExtent();

//--------------------------------------------------------------������---------------------------------------------------------------
	void fzbCreateCommandPool();

	//void fzbCreateFramebuffers() {};

//------------------------------------------------------������ģ��-----------------------------------------------------------------
//--------------------------------------------------------------ͼ��-----------------------------------------------------------------
	virtual void createImages();

//---------------------------------------------------------------------------��Ⱦѭ��---------------------------------------------------------------------
	void mainLoop();

	void processInput(GLFWwindow* window);

	virtual void drawFrame() = 0;

	void recreateSwapChain(std::vector<FzbRenderPass*> renderPasses);

	virtual void cleanupImages();

	void fzbCleanupSwapChain();

	virtual void clean() = 0;


};

#endif