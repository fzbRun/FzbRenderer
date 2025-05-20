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

namespace std {
	template<> struct hash<FzbVertex> {
		size_t operator()(FzbVertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
	}
};
	template<> struct hash<FzbVertex_OnlyPos> {
		size_t operator()(FzbVertex_OnlyPos const& vertex) const {
			// ������ pos �Ĺ�ϣֵ
			return hash<glm::vec3>()(vertex.pos);
		}
	};
}

//------------------------------------------------����----------------------------------------------------
//��������ԣ���ر�У���
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char*> instanceExtensions_default = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
const std::vector<const char*> validationLayers_default = { "VK_LAYER_KHRONOS_validation" };
const std::vector<const char*> deviceExtensions_default = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
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

	std::vector<std::vector<VkFramebuffer>> framebuffers;

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
		stagingBuffer.fzbCreateStorageBuffer();
		stagingBuffer.fzbFillBuffer(bufferData->data());

		FzbStorageBuffer<T> fzbBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UseExternal);
		fzbBuffer.data = *bufferData;
		fzbBuffer.fzbCreateStorageBuffer();

		copyBuffer(logicalDevice, commandPool, graphicsQueue, stagingBuffer.buffer, fzbBuffer.buffer, bufferSize);

		stagingBuffer.clean();

		return fzbBuffer;

	}

	//����һ���յ�buffer
	template<typename T>
	FzbStorageBuffer<T> fzbCreateStorageBuffer(uint32_t bufferSize, bool UseExternal = false) {
		 FzbStorageBuffer<T> fzbBuffer(physicalDevice, logicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UseExternal);
		 fzbBuffer.fzbCreateStorageBuffer();
		 return fzbBuffer;
	}

	template<typename T>
	FzbUniformBuffer<T> fzbCreateUniformBuffers() {
		FzbUniformBuffer<T> fzbBuffer(physicalDevice, logicalDevice, sizeof(T), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		fzbBuffer.fzbCreateUniformBuffer();
		vkMapMemory(logicalDevice, fzbBuffer.memory, 0, sizeof(T), 0, &fzbBuffer.mapped);
		return fzbBuffer;
	}

//------------------------------------------------------------------ģ��-------------------------------------------------------------------------


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
			//�����棬���Ǹ���0.2�Ŀ��
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
		//�����棬���Ǹ���0.2�Ŀ��
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

//------------------------------------------------------------------ͼ��-------------------------------------------------------------------------
//-----------------------------------------------------------------������-------------------------------------------------------------------------
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

	void setupDebugMessenger() {

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
		//deviceFeatures.sampleRateShading = VK_TRUE;

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

	void fzbCreateFramebuffers() {};

//------------------------------------------------------������ģ��-----------------------------------------------------------------
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

	//һ��node����mesh����node��������Ҫ�ݹ飬�����е�mesh���ó���
	//���е�ʵ�����ݶ���scene�У���node�д洢����scene������
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

			if (mesh->mTextureCoords[0]) // �����Ƿ����������ꣿ
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
			{   // �������û�б����أ��������
				FzbTexture texture;
				//texture.id = TextureFromFile(str.C_Str(), directory);
				texture.type = typeName;
				texture.path = myModel.directory + '/' + str.C_Str();
				textures.push_back(texture);
				myModel.textures_loaded.push_back(texture); // ��ӵ��Ѽ��ص�������
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

	//��һ��mesh�����ඥ��ɾ��
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

	//��һ��ģ�͵�����mesh�Ķ������������һ�����飬��ɾ�����ඥ��
	template<typename T>
	void fzbOptimizeModel(FzbModel* myModel, std::vector<T>& vertices, std::vector<uint32_t>& indices) {
		uint32_t indexOffset = 0;
		for (uint32_t meshIndex = 0; meshIndex < myModel->meshs.size(); meshIndex++) {

			FzbMesh* mesh = &myModel->meshs[meshIndex];
			fzbOptimizeMesh(mesh);
			vertices.insert(vertices.end(), mesh->vertices.begin(), mesh->vertices.end());

			//��Ϊassimp�ǰ�һ��meshһ��mesh�Ĵ棬����ÿ��indices�������һ��mesh�ģ������ǽ�ÿ��mesh�Ķ���浽һ��ʱ��indices�ͻ����������Ҫ��������
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

			//��Ϊassimp�ǰ�һ��meshһ��mesh�Ĵ棬����ÿ��indices�������һ��mesh�ģ������ǽ�ÿ��mesh�Ķ���浽һ��ʱ��indices�ͻ����������Ҫ��������
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

	void recreateSwapChain() {

		int width = 0, height = 0;
		//��õ�ǰwindow�Ĵ�С
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