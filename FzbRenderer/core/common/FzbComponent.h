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
#include "FzbRenderPass.h"
#include "FzbPipeline.h"
#include "FzbDescriptor.h"
#include "FzbScene.h"
#include "FzbCamera.h"


#ifndef FZB_COMPONENT
#define FZB_COMPONENT

//-----------------------------------------------��չ����---------------------------------------------------
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebygMessenger) {
	//��������չ������������Ҫͨ��vkGetInstanceProcAddr��øú���ָ��
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

//------------------------------------------------------------------��-----------------------------------------------------
class FzbComponent {

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

	//std::vector<std::vector<VkFramebuffer>> framebuffers;

	void fzbCreateCommandBuffers(uint32_t bufferNum = 1) {

		//��������������ˮ��һ�����ƣ�������Ҫ���ָ�����
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = this->commandPool;
		//VK_COMMAND_BUFFER_LEVEL_PRIMARY�������ύ������ִ�У������ܴ���������������á�
		//VK_COMMAND_BUFFER_LEVEL_SECONDARY������ֱ���ύ�������Դ��������������
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = bufferNum;	//ָ����������������������������Ǹ�����������Ĵ�С

		//��������ĵ��������������ǵ����������ָ��Ҳ����������
		this->commandBuffers.resize(bufferNum);
		if (vkAllocateCommandBuffers(logicalDevice, &allocInfo, this->commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate shadow command buffers!");
		}

	}

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
	FzbBuffer fzbComponentCreateStorageBuffer(uint32_t bufferSize, bool UseExternal = false) {
		return fzbCreateStorageBuffer(physicalDevice, logicalDevice, bufferSize, UseExternal);
	}

	template<typename T>
	FzbBuffer fzbComponentCreateUniformBuffers() {
		return fzbCreateUniformBuffers(physicalDevice, logicalDevice, sizeof(T));
	}

//------------------------------------------------------------------ģ�ͺ�shader-------------------------------------------------------------------------
	virtual FzbVertexFormat getComponentVertexFormat() {
		return FzbVertexFormat(true, true, true);
	}
//------------------------------------------------------------------ͼ��-------------------------------------------------------------------------
//-----------------------------------------------------------------������-------------------------------------------------------------------------
	VkDescriptorPool descriptorPool;

	void fzbComponentCreateDescriptorPool(std::map<VkDescriptorType, uint32_t> bufferTypeAndNum) {
		this->descriptorPool = fzbCreateDescriptorPool(logicalDevice, bufferTypeAndNum);
	}

	VkDescriptorSetLayout fzbComponentCreateDescriptLayout(uint32_t descriptorNum, std::vector<VkDescriptorType> descriptorTypes, std::vector<VkShaderStageFlags> descriptorShaderFlags, std::vector<uint32_t> descriptorCounts = std::vector<uint32_t>()) {
		return fzbCreateDescriptLayout(logicalDevice, descriptorNum, descriptorTypes, descriptorShaderFlags, descriptorCounts);

	}

	VkDescriptorSet fzbComponentCreateDescriptorSet(VkDescriptorSetLayout& descriptorSetLayout) {
		return fzbCreateDescriptorSet(logicalDevice, descriptorPool, descriptorSetLayout);
	}

//-------------------------------------------------------------------����---------------------------------------------------------------------

//--------------------------------------------------------------------------��դ���ź���-----------------------------------------------------------------
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
		//��һ֡����ֱ�ӻ���źţ�����������
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
	VkInstance instance;	//vulkanʵ��
	VkDebugUtilsMessengerEXT debugMessenger;	//��Ϣ������
	VkSurfaceKHR surface;

	std::vector<const char*> instanceExtensions = instanceExtensions_default;
	std::vector<const char*> validationLayers = validationLayers_default;
	uint32_t apiVersion = apiVersion_default;

	void fzbInitWindow(uint32_t width = WIDTH, uint32_t height = HEIGHT, const char* windowName = "δ����", VkBool32 windowResizable = VK_FALSE) {

		glfwInit();

		//��ֹGLFW�Զ�����OpenGL������
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		//�Ƿ��ֹ�ı䴰�ڴ�С
		glfwWindowHint(GLFW_RESIZABLE, windowResizable);

		window = glfwCreateWindow(width, height, windowName, nullptr, nullptr);
		//glfwSetFramebufferSizeCallback�����ڻص�ʱ����ҪΪ��������framebufferResized��������֪������˭
		//����ͨ����window��������˭���Ӷ��ûص�����֪������˭
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

	void fzbCreateInstance(const char* appName = "δ����", std::vector<const char*> instanceExtences = instanceExtensions_default, std::vector<const char*> validationLayers = validationLayers_default, uint32_t apiVersion = apiVersion_default) {

		//���layer
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

		//��չ����Vulkan����û��ʵ�֣���������Ա��װ��Ĺ��ܺ��������ƽ̨�ĸ��ֺ���������������ͨ�������ɣ������ֻ�����
		auto extensions = getRequiredExtensions(instanceExtences);
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();	//����չ�ľ�����Ϣ��ָ��洢�ڸýṹ����

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();	//��У���ľ�����Ϣ��ָ��洢�ڸýṹ����

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

		// ��ȡ Vulkan ʵ���İ汾
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
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);	//���ؿ��õĲ���
		std::vector<VkLayerProperties> availableLayers(layerCount);	//VkLayerProperties��һ���ṹ�壬��¼������֡�������
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
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);	//�õ�glfw�������չ��
		//����1��ָ����ʼλ�ã�����2��ָ����ֹλ��
		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);	//�����չ��Ϊ�˴�ӡУ��㷴ӳ�Ĵ���������Ҫ֪���Ƿ���ҪУ���
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

	void fzbCetupDebugMessenger() {

		if (!enableValidationLayers)
			return;
		VkDebugUtilsMessengerCreateInfoEXT  createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		//ͨ��func�Ĺ��캯����debugMessenger��ֵ
		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}

	}

	void fzbCreateSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface");
		}
	}

//-------------------------------------------------------------------�豸-----------------------------------------------------------------------
	std::vector<const char*> deviceExtensions = deviceExtensions_default;
	VkPhysicalDeviceFeatures deviceFeatures;
	void* pNextFeatures = nullptr;

	void pickPhysicalDevice(std::vector<const char*> deviceExtensions = deviceExtensions_default) {

		if (!instance || !surface) {
			throw std::runtime_error("ʵ�������δ��ʼ��");
		}

		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUS with Vulkan support");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());	//���ǰ����ȼ��ŵ�

		//���Կ���������������ģ�������Ĭ���õ���intel�ļ����Կ����ҵ�3070ֻ�ܳԻ�
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
		//vkGetPhysicalDeviceProperties(device, &deviceProperties);	//�豸��Ϣ
		//vkGetPhysicalDeviceFeatures(device, &deviceFeatures);		//�豸����

		this->queueFamilyIndices = findQueueFamilies(device);
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(device, &deviceProperties);
		//std::cout << deviceProperties.limits.maxPerStageDescriptorStorageImages << std::endl;

		//����豸�Ƿ�֧�ֽ�������չ
		bool extensionsSupport = checkDeviceExtensionSupport(deviceExtensions, device);
		bool swapChainAdequate = false;
		if (extensionsSupport) {
			//�ж������豸��ͼ���չʾ�����Ƿ�֧��
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
			if (!deviceFeatures.geometryShader) {	//�ҿ���ֻҪ����֧�ּ�����ɫ�����Կ�
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
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());	//��ö���ϵ�е���ϸ��Ϣ

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			//�����ͼ������ǲ���˵�Կ���ר�Ŷ���Ⱦ���Ż�
			//��ΪVK_QUEUE_COMPUTE_BIT��˵�Կ�����ͨ�ü���(������ɫ��)������Ⱦʵ����Ҳ��һ�ּ��㣬��ô�ֿ����ߵ�ԭ��Ӧ�þ����Ƿ���ר���Ż�
			//ע��֧��VK_QUEUE_GRAPHICS_BIT��VK_QUEUE_COMPUTE_BIT���豸Ĭ��֧��VK_QUEUE_TRANSFER_BIT���������ݻ��������ݣ�
			if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
				indices.graphicsAndComputeFamily = i;
			}

			VkBool32 presentSupport = false;
			//�ж�i��Ⱥ�Ƿ�Ҳ֧��չʾ������չʾ����˼���ܷ�GPU��Ⱦ�����Ļ��洫����ʾ���ϣ���Щ�Կ����ܲ�δ���ӵ���ʾ��
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

		//��requiredExtensions���ˣ�˵����Ҫ����չȫ��
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

	VkPhysicalDeviceFeatures2 createPhysicalDeviceFeatures(VkPhysicalDeviceFeatures deviceFeatures, VkPhysicalDeviceVulkan11Features* vk11Features = nullptr, VkPhysicalDeviceVulkan12Features* vk12Features = nullptr) {
		if(vk12Features)
			vk12Features->pNext = vk11Features;
		
		VkPhysicalDeviceFeatures2 features2{};
		features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		features2.features = deviceFeatures;       // ����ź��Ĺ���
		features2.pNext = vk12Features ? (void*)vk12Features : (void*)vk11Features;
		return features2;
	}

	void createLogicalDevice(VkPhysicalDeviceFeatures* deviceFeatures = nullptr, std::vector<const char*> deviceExtensions = deviceExtensions_default, const void* pNextFeatures = nullptr, std::vector<const char*> validationLayers = validationLayers_default) {

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { queueFamilyIndices.graphicsAndComputeFamily.value(), queueFamilyIndices.presentFamily.value() };

		//����ѡȡ�������豸ӵ��һ���Ķ����壨���ܣ�����û�д�����������Ҫ��֮��������
		//����������豸��Ӧһ���߼��豸����һ���߼��豸��Ӧ��������
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

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pEnabledFeatures = deviceFeatures ? deviceFeatures : pNextFeatures ? nullptr : &deviceFeatures_default;
		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();
		createInfo.pNext = pNextFeatures;

		// Ϊ�豸ָ����ʵ����ͬ��У���
		// ʵ���ϣ��°汾��Vulkan�Ѿ��������ֶ��ߵ�У��㣬
		// ���Զ������豸�й���У�����ֶΡ���������һ�µĻ������Ծɰ汾����
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

//----------------------------------------------------------------������------------------------------------------------------------
	void fzbCreateSwapChain() {

		if (swapChainSupportDetails.formats.empty() || swapChainSupportDetails.presentModes.empty() || !queueFamilyIndices.isComplete()) {
			throw std::runtime_error("�豸δ��ʼ��");
		}

		this->surfaceFormat = chooseSwapSurfaceFormat();	//��Ҫ��surface��չʾ�������ͨ�����������Լ�ɫ�ʿռ�
		VkPresentModeKHR presentMode = chooseSwapPresentMode();
		this->extent = chooseSwapExtent();

		//�����������С������ͼ������ȣ���ȷ����֧�ֵ�ͼ������������֧�ֵ�ͼ��������������Сͼ����+1
		//���maxImageCount=0�����ʾû�����ƣ������������ط������ƣ��޷�������
		uint32_t imageCount = swapChainSupportDetails.capabilities.minImageCount + 1;
		if (swapChainSupportDetails.capabilities.maxImageCount > 0 && imageCount > swapChainSupportDetails.capabilities.maxImageCount) {
			imageCount = swapChainSupportDetails.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;	//��Ҫ������
		createInfo.minImageCount = imageCount;	//�涨�˽������������������������2����˫����
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;	//���������z��1�ͱ�ʾ2D����
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		uint32_t queueFamilyIndicesArray[] = { queueFamilyIndices.graphicsAndComputeFamily.value(), queueFamilyIndices.presentFamily.value() };

		//ͼ�ζ����帺����Ⱦ���ܣ�Ȼ�󽻸����������������ٽ���չʾ��������ֵ�surface��
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

		createInfo.preTransform = swapChainSupportDetails.capabilities.currentTransform;	//ָ���Ƿ���Ҫ��ǰ��ת��ת
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

		//imageViews�ͽ������е�image������ͬ
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
			//��������γ��ֻ��棬������ֱ��չʾ����˫���壬�ȵ�
			//VK_PRESENT_MODE_IMMEDIATE_KHR ��Ⱦ��ɺ�����չʾ��ÿ֡���ֺ���Ҫ�ȴ���һ֡��Ⱦ��ɲ����滻�������һ֡��Ⱦ��ʱ��ʱ�����ͻ���ֿ���
			//VK_PRESENT_MODE_FIFO_KHR V-Sync,��ֱͬ�����໺�壬��Ⱦ��ɺ��ύ���浽����Ļ��壬�̶�ʱ�䣨��ʾ��ˢ��ʱ�䣩����ֵ���ʾ���ϡ������������ˣ���Ⱦ�ͻ�ֹͣ��������
			//VK_PRESENT_MODE_FIFO_RELAXED_KHR ��Ⱦ��ɺ��ύ���浽����Ļ��壬���������һ֡��Ⱦ�Ľ�����������һ֡��ˢ�º��Դ��ڣ���ǰ֡�ύ�����̳��֣���ô�Ϳ��ܵ��¸���
			//VK_PRESENT_MODE_MAILBOX_KHR Fast-Sync, �����壬��Ⱦ��ɺ��ύ���浽����Ļ��壬�̶�ʱ�䣨��ʾ��ˢ��ʱ�䣩����ֵ���ʾ���ϡ������������ˣ�������滻���Ļ���������������
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
		else {		//ĳЩ���ڹ��������������ڴ˴�ʹ�ò�ͬ��ֵ����ͨ����currentExtent�Ŀ�Ⱥ͸߶�����Ϊ���ֵ����ʾ�����ǲ���Ҫ���������Խ�֮��������Ϊ���ڴ�С
			int width, height;
			//��ѯ���ڷֱ���
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

//--------------------------------------------------------------������---------------------------------------------------------------
	void fzbCreateCommandPool() {
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		//VK_COMMAND_POOL_CREATE_TRANSIENT_BIT����ʾ����������������¼�¼��������ܻ�ı��ڴ������Ϊ��
		//VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT�����������¼�¼������������û�д˱�־�������һ�����������������
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();
		if (vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &this->commandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
	}

	//void fzbCreateFramebuffers() {};

//------------------------------------------------------������ģ��-----------------------------------------------------------------
//--------------------------------------------------------------ͼ��-----------------------------------------------------------------
	virtual void createImages() {};

//---------------------------------------------------------------------------��Ⱦѭ��---------------------------------------------------------------------
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

	void recreateSwapChain(std::vector<FzbRenderPass> renderPasses) {

		int width = 0, height = 0;
		//��õ�ǰwindow�Ĵ�С
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(logicalDevice);
		for (int i = 0; i < renderPasses.size(); i++) {
			if (renderPasses[i].setting.extent.width == swapChainExtent.width && renderPasses[i].setting.extent.height == swapChainExtent.height) {
				for (int j = 0; j < renderPasses[i].framebuffers.size(); j++) {
					vkDestroyFramebuffer(logicalDevice, renderPasses[i].framebuffers[j], nullptr);
				}
			}
		}
		fzbCleanupSwapChain();
		fzbCreateSwapChain();
		createImages();
		for (int i = 0; i < renderPasses.size(); i++) {
			if (renderPasses[i].setting.extent.width == swapChainExtent.width && renderPasses[i].setting.extent.height == swapChainExtent.height) {
				renderPasses[i].createFramebuffers(swapChainImageViews);
			}
		}
	}

	virtual void cleanupImages() {

	}

	void fzbCleanupSwapChain() {

		cleanupImages();
		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			vkDestroyImageView(logicalDevice, swapChainImageViews[i], nullptr);
		}
		vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);
	}

	virtual void clean() = 0;


};

#endif