#include "./FzbRenderer.h"
#include "../ForwardRender/FzbForwardRender.h"
#include "../SceneDivision/SVO/FzbSVO.h"
#include "../SceneDivision/BVH/FzbBVH.h"
#include "../RayTracing/PathTracing/soft/FzbPathTracing_soft.h"
#include "../SceneDivision/SVO_PG/FzbSVO_PG.h"
#include "../RayTracing/SVOPathGuiding/soft/FzbSVOPathGuiding_soft.h"

#include <glslang/Public/ShaderLang.h>
#include <chrono>
#include <random>

std::map<std::string, FzbFeatureComponentName> featureComponentMap{
	{ "Forward", FZB_RENDERER_FORWARD },
	{ "PathTracing_soft", FZB_RENDERER_PATH_TRACING_SOFT },
	{ "SVOPathGuiding_soft", FZB_RENDERER_SVO_PATH_GUIDING },
	{ "BVH", FZB_FEATURE_COMPONENT_BVH },
	{ "BVH_Debug", FZB_FEATURE_COMPONENT_BVH_DEBUG },
	{ "SVO", FZB_FEATURE_COMPONENT_SVO },
	{ "SVO_Debug", FZB_FEATURE_COMPONENT_SVO_DEBUG },
	{ "SVO_PG", FZB_FEATURE_COMPONENT_SVO_PG },
	{ "SVO_PG_Debug", FZB_FEATURE_COMPONENT_SVO_PG_DEBUG },
};
std::shared_ptr<FzbFeatureComponent> createFzbComponent(std::string componentName, pugi::xml_node& node) {
	FzbFeatureComponentName name;
	if (featureComponentMap.count(componentName)) {
		name = featureComponentMap[componentName];
		switch (name) {
			case FZB_RENDERER_FORWARD: return std::make_unique<FzbForwardRender>(node);
			case FZB_RENDERER_PATH_TRACING_SOFT: return std::make_unique<FzbPathTracing_soft>(node);
			case FZB_RENDERER_SVO_PATH_GUIDING: return std::make_unique<FzbSVOPathGuiding_soft>(node);
			case FZB_FEATURE_COMPONENT_BVH: return std::make_unique<FzbBVH>(node);
			case FZB_FEATURE_COMPONENT_BVH_DEBUG: return std::make_unique<FzbBVH_Debug>(node);
			//case FZB_FEATURE_COMPONENT_SVO: return std::make_unique<FzbSVO>(node);
			case FZB_FEATURE_COMPONENT_SVO_DEBUG: return std::make_unique<FzbSVO_Debug>(node);
			case FZB_FEATURE_COMPONENT_SVO_PG: return std::make_unique<FzbSVO_PG>(node);
			case FZB_FEATURE_COMPONENT_SVO_PG_DEBUG: return std::make_unique<FzbSVO_PG_Debug>(node);
		}
		return nullptr;
	}
	else throw std::runtime_error("没有相应的功能组件" + componentName);
}

FzbRenderer::FzbRenderer(std::string rendererXML) {
	glslang::InitializeProcess();
	initRendererFromXMLInfo(rendererXML);
}
void FzbRenderer::initRendererFromXMLInfo(std::string rendererXML) {
	pugi::xml_document doc;
	auto result = doc.load_file(rendererXML.c_str());
	if (!result) {
		throw std::runtime_error("pugixml打开文件失败");
	}
	pugi::xml_node rendererInfo = doc.document_element();
	this->rendererName = rendererInfo.child("name").attribute("value").value();
	VkExtent2D resolution;
	resolution.width = std::stoi(rendererInfo.child("resolution").attribute("width").value());
	resolution.height = std::stoi(rendererInfo.child("resolution").attribute("height").value());
	this->globalData.setResolution(resolution);

	std::string scenePath = rendererInfo.child("sceneXML").attribute("path").value();
	this->globalData.mainScene = FzbMainScene(scenePath);	//看构造函数注释
	
	if (pugi::xml_node rendererTypeNode = rendererInfo.child("rendererType")) {
		std::string rendererType = rendererTypeNode.attribute("value").value();
		this->componentManager.addFeatureComponent(createFzbComponent(rendererType, rendererTypeNode));
	}
	//else throw std::runtime_error("必须指定一个渲染器");
	if (pugi::xml_node featureComponents = rendererInfo.child("featureComponents")) {
		for (pugi::xml_node featureComponet : featureComponents.children("featureComponent")) {
			std::string featureComponentName = featureComponet.attribute("name").value();
			this->componentManager.addFeatureComponent(createFzbComponent(featureComponentName, featureComponet));
		}
	}
	doc.reset();
	
	//this->componentManager.init();
	this->globalData.init(this->rendererName.c_str());
	this->componentManager.componentInit();

	imageAvailableSemaphore = FzbSemaphore(this->useImageAvailableSemaphoreHandle);
	fence = fzbCreateFence();
}

void FzbRenderer::run() {
	mainLoop();
	clean();
}

void FzbRenderer::mainLoop() {
	while (!glfwWindowShouldClose(globalData.window)) {
		globalData.processInput();
		glfwPollEvents();
		drawFrame();
	}
	//globalData.processInput();
	//glfwPollEvents();
	//drawFrame();
	vkDeviceWaitIdle(globalData.logicalDevice);
}
void FzbRenderer::drawFrame() {
	vkWaitForFences(globalData.logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);

	uint32_t imageIndex;
	VkResult result = vkAcquireNextImageKHR(globalData.logicalDevice, globalData.swapChain, UINT64_MAX, imageAvailableSemaphore.semaphore, VK_NULL_HANDLE, &imageIndex);
	/*
	//VK_ERROR_OUT_OF_DATE_KHR：交换链与表面不兼容，无法再用于渲染。通常在调整窗口大小后发生。
	//VK_SUBOPTIMAL_KHR：交换链仍可用于成功呈现到表面，但表面属性不再完全匹配。
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || globalData.framebufferResized) {
		globalData.framebufferResized = false;
		globalData.recreateSwapChain();
		return;
	}
	else if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to acquire swap chain image!");
	}
	*/
	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to acquire swap chain image!");
	}

	updateGlobalData();
	vkResetFences(globalData.logicalDevice, 1, &fence);

	FzbSemaphore renderFinishedSemaphore = this->componentManager.componentActivate(imageIndex, imageAvailableSemaphore, fence);
	
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &renderFinishedSemaphore.semaphore; 	// svoSetting.UseSVO ? &fzbSVO->presentSemaphore.semaphore : &renderFinishedSemaphores.semaphore;

	VkSwapchainKHR swapChains[] = { globalData.swapChain };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;
	presentInfo.pImageIndices = &imageIndex;
	result = vkQueuePresentKHR(globalData.presentQueue, &presentInfo);
	/*
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
		recreateSwapChain(renderPasses);
	}
	else if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to present swap chain image!");
	}
	*/
	if (result != VK_SUCCESS) {
		throw std::runtime_error("failed to present swap chain image!");
	}
}
void FzbRenderer::updateGlobalData() {
	float currentTime = static_cast<float>(glfwGetTime());
	globalData.deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - globalData.lastTime).count();
	globalData.lastTime = currentTime;
	globalData.mainScene.updateCameraBuffer();
	globalData.frameIndex = globalData.frameIndex % MAXFRAMECOUNT;
	++globalData.frameIndex;

	std::random_device rd;
	std::mt19937 gen(rd()); // Mersenne Twister 引擎
	VkExtent2D resolution = globalData.getResolution();
	uint64_t offset = uint64_t(resolution.width) * resolution.height * 1000ull;
	unsigned int maxVal = (offset < UINT_MAX) ? (UINT_MAX - offset) : 0;
	std::uniform_int_distribution<uint32_t> distInt(0, maxVal);	// 生成均匀分布整数
	globalData.randomNumber = distInt(gen);
}

void FzbRenderer::clean() {
	glslang::FinalizeProcess();
	componentManager.clean();
	imageAvailableSemaphore.clean();
	fzbCleanFence(fence);
	globalData.clean();
}