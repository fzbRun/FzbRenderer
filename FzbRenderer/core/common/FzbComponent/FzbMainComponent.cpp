#include "./FzbMainComponent.h"
#include "../FzbRenderer.h"
#include <iostream>

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

FzbMainComponent::FzbMainComponent() {}
void FzbMainComponent::setMainScenePath(std::string scenePath) {
	this->mainScene.scenePath = scenePath;
}

void FzbMainComponent::init(const char* name) {
	initWindow(resolution.width, resolution.height, name, VK_FALSE);
	createInstance(name, instanceExtensions, validationLayers);
	setupDebugMessenger();
	createSurface();
	createDevice();
	createSwapChain();
	createCommandPool();
	createScene();
}
static void fzbCleanupSwapChain() {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;
	std::vector<VkImageView> swapChainImageViews = FzbRenderer::globalData.swapChainImageViews;
	for (size_t i = 0; i < swapChainImageViews.size(); i++) {
		vkDestroyImageView(logicalDevice, swapChainImageViews[i], nullptr);
	}
	vkDestroySwapchainKHR(logicalDevice, FzbRenderer::globalData.swapChain, nullptr);
}
static void recreateSwapChain() {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;
	VkExtent2D swapChainExtent = FzbRenderer::globalData.swapChainExtent;

	int width = 0, height = 0;
	//获得当前window的大小
	glfwGetFramebufferSize(FzbRenderer::globalData.window, &width, &height);
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(FzbRenderer::globalData.window, &width, &height);
		glfwWaitEvents();
	}
	vkDeviceWaitIdle(FzbRenderer::globalData.logicalDevice);	//阻塞CPU，等待logicalDevice所有任务执行完成
	FzbRenderer::globalData.setResolution({ (uint32_t)width, (uint32_t)height });
	for (int i = 0; i < FzbRenderer::componentManager.loopRenderFeatureComponent.size(); i++) {
		FzbRenderer::componentManager.loopRenderFeatureComponent[i]->destroyFrameBuffer();
	}
	fzbCleanupSwapChain();
	FzbRenderer::globalData.createSwapChain();
	for (int i = 0; i < FzbRenderer::componentManager.loopRenderFeatureComponent.size(); i++) {
		FzbRenderer::componentManager.loopRenderFeatureComponent[i]->createFrameBuffer();
	}
}
static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
	auto app = reinterpret_cast<FzbMainComponent*>(glfwGetWindowUserPointer(window));
	app->framebufferResized = true;
	recreateSwapChain();
}

static void mouse_callback_static(GLFWwindow* window, double xposIn, double yposIn)
{
	// 获取类实例
	FzbMainComponent* instance = static_cast<FzbMainComponent*>(glfwGetWindowUserPointer(window));

	// 调用成员函数
	instance->mouse_callback(xposIn, yposIn);
}
static void scroll_callback_static(GLFWwindow* window, double xoffset, double yoffset)
{
	// 获取类实例
	FzbMainComponent* instance = static_cast<FzbMainComponent*>(glfwGetWindowUserPointer(window));

	// 调用成员函数
	instance->scroll_callback(xoffset, yoffset);
}
void FzbMainComponent::mouse_callback(double xposIn, double yposIn)
{
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera->ProcessMouseMovement(xoffset, yoffset);
}
void FzbMainComponent::scroll_callback(double xoffset, double yoffset)
{
	camera->ProcessMouseScroll(static_cast<float>(yoffset));
}
void FzbMainComponent::initWindow(uint32_t width, uint32_t height, const char* windowName, VkBool32 windowResizable) {

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
	glfwSetCursorPosCallback(window, mouse_callback_static);
	glfwSetScrollCallback(window, scroll_callback_static);

	this->lastX = width / 2.0f;
	this->lastY = height / 2.0f;
}

void FzbMainComponent::createInstance(const char* appName, std::vector<const char*> instanceExtences, std::vector<const char*> validationLayers, uint32_t apiVersion) {

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

bool FzbMainComponent::checkValidationLayerSupport() {

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
std::vector<const char*> FzbMainComponent::getRequiredExtensions(std::vector<const char*> instanceExtences) {

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
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
	std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
	return VK_FALSE;
}
void FzbMainComponent::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
	createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	createInfo.pfnUserCallback = debugCallback;
	createInfo.pUserData = nullptr;
}
void FzbMainComponent::setupDebugMessenger() {

	if (!enableValidationLayers)
		return;
	VkDebugUtilsMessengerCreateInfoEXT  createInfo;
	populateDebugMessengerCreateInfo(createInfo);

	//通过func的构造函数给debugMessenger赋值
	if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
		throw std::runtime_error("failed to set up debug messenger!");
	}

}

void FzbMainComponent::createSurface() {
	if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
		throw std::runtime_error("failed to create window surface");
	}
}

void FzbMainComponent::pickPhysicalDevice(std::vector<const char*> deviceExtensions) {

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

		this->queueFamilyIndices = findQueueFamilies(physicalDevice);
		this->swapChainSupportDetails = querySwapChainSupport(physicalDevice);
	}
	else {
		throw std::runtime_error("failed to find a suitable GPU!");
	}

}
int FzbMainComponent::rateDeviceSuitability(std::vector<const char*> deviceExtensions, VkPhysicalDevice device) {

	//VkPhysicalDeviceProperties deviceProperties;
	//VkPhysicalDeviceFeatures deviceFeatures;
	//vkGetPhysicalDeviceProperties(device, &deviceProperties);	//设备信息
	//vkGetPhysicalDeviceFeatures(device, &deviceFeatures);		//设备功能

	FzbQueueFamilyIndices queueFamilyIndicesTemp = findQueueFamilies(device);
	VkPhysicalDeviceProperties deviceProperties;
	vkGetPhysicalDeviceProperties(device, &deviceProperties);
	//std::cout << deviceProperties.limits.maxPerStageDescriptorStorageImages << std::endl;

	//检查设备是否支持交换链扩展
	bool extensionsSupport = checkDeviceExtensionSupport(deviceExtensions, device);
	bool swapChainAdequate = false;
	if (extensionsSupport) {
		//判断物理设备的图像和展示功能是否支持
		FzbSwapChainSupportDetails swapChainSupportDetailsTmep = querySwapChainSupport(device);
		swapChainAdequate = !swapChainSupportDetailsTmep.formats.empty() && !swapChainSupportDetailsTmep.presentModes.empty();
	}

	if (queueFamilyIndicesTemp.isComplete() && extensionsSupport && swapChainAdequate) {
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
FzbSwapChainSupportDetails FzbMainComponent::querySwapChainSupport(VkPhysicalDevice device) {

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
FzbQueueFamilyIndices FzbMainComponent::findQueueFamilies(VkPhysicalDevice device) {

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
bool FzbMainComponent::checkDeviceExtensionSupport(std::vector<const char*> deviceExtensions, VkPhysicalDevice device) {

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
VkSampleCountFlagBits FzbMainComponent::getMaxUsableSampleCount() {
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
void FzbMainComponent::createLogicalDevice(VkPhysicalDeviceFeatures* deviceFeatures, std::vector<const char*> deviceExtensions, const void* pNextFeatures, std::vector<const char*> validationLayers) {
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

	VkDeviceCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	createInfo.pEnabledFeatures = deviceFeatures ? deviceFeatures : pNextFeatures ? nullptr : &deviceFeatures_default;
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
/*
VkPhysicalDeviceFeatures2 FzbMainComponent::createPhysicalDeviceFeatures(VkPhysicalDeviceFeatures deviceFeatures, VkPhysicalDeviceVulkan11Features* vk11Features, VkPhysicalDeviceVulkan12Features* vk12Features) {
	if (vk12Features)
		vk12Features->pNext = vk11Features;

	VkPhysicalDeviceFeatures2 features2{};
	features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	features2.features = deviceFeatures;       // 这里放核心功能
	features2.pNext = vk12Features ? (void*)vk12Features : (void*)vk11Features;
	return features2;
}
*/
void FzbMainComponent::createDevice() {
	deviceFeatures.samplerAnisotropy = VK_TRUE;
	deviceFeatures.geometryShader = VK_TRUE;
	deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
	deviceFeatures.multiDrawIndirect = VK_TRUE;

	vk11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	vk11Features.shaderDrawParameters = VK_TRUE;
	vk11Features.storageBuffer16BitAccess = VK_TRUE;

	vk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	vk12Features.drawIndirectCount = VK_TRUE;
	vk12Features.descriptorIndexing = VK_TRUE;
	vk12Features.storageBuffer8BitAccess = VK_TRUE;
	vk12Features.shaderInt8 = VK_TRUE;
	vk12Features.shaderFloat16 = VK_TRUE;
	vk12Features.scalarBlockLayout = VK_TRUE;

	vk11Features.pNext = this->extensionFeatureList.featureList;
	vk12Features.pNext = &vk11Features;
	VkPhysicalDeviceFeatures2 features2{};
	features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	features2.features = deviceFeatures;       // 这里放核心功能
	features2.pNext = &vk12Features;

	pickPhysicalDevice(deviceExtensions);
	createLogicalDevice(nullptr, deviceExtensions, &features2, validationLayers_default);
}

void FzbMainComponent::createSwapChain() {
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
void FzbMainComponent::createSwapChainImageViews() {
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
VkSurfaceFormatKHR FzbMainComponent::chooseSwapSurfaceFormat() {

	for (const auto& availableFormat : swapChainSupportDetails.formats) {
		if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return availableFormat;
		}
	}
	return swapChainSupportDetails.formats[0];

}
VkPresentModeKHR FzbMainComponent::chooseSwapPresentMode() {
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
VkExtent2D FzbMainComponent::chooseSwapExtent() {

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

void FzbMainComponent::createCommandPool() {
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

void FzbMainComponent::createScene() {
	mainScene.initScene();
	this->camera = &mainScene.sceneCameras[0];
}

void FzbMainComponent::processInput()
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera->ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera->ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera->ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera->ProcessKeyboard(RIGHT, deltaTime);
}

void FzbMainComponent::clean() {
	mainScene.clean();

	fzbCleanupSwapChain();
	vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
	vkDestroyCommandPool(logicalDevice, commandPool, nullptr);
	vkDestroyDevice(logicalDevice, nullptr);
	if (enableValidationLayers) DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyInstance(instance, nullptr);
	glfwDestroyWindow(window);
	glfwTerminate();
}

void FzbMainComponent::setResolution(VkExtent2D resolution) {
	FzbMainComponent::resolution = resolution;
	if (camera) {
		camera->aspect = (float)resolution.width / resolution.height;
		camera->createProjMatrix();
	}
}
VkExtent2D FzbMainComponent::getResolution() {
	return FzbMainComponent::resolution;
}
