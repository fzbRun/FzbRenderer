#include "./FzbRenderer.h"
#include "../ForwardRender/FzbForwardRender.h"
#include "../SceneDivision/SVO/FzbSVO.h"
#include "../SceneDivision/BVH/FzbBVH.h"
#include <glslang/Public/ShaderLang.h>
#include <chrono>

std::map<std::string, FzbFeatureComponentName> featureComponentMap{
	{ "Forward", FZB_RENDERER_FORWARD },
	{ "BVH", FZB_FEATURE_COMPONENT_BVH },
	{ "BVH_Debug", FZB_FEATURE_COMPONENT_BVH_DEBUG },
	{ "SVO", FZB_FEATURE_COMPONENT_SVO },
	{ "SVO_Debug", FZB_FEATURE_COMPONENT_SVO_DEBUG },
};
std::shared_ptr<FzbFeatureComponent> createFzbComponent(std::string componentName, pugi::xml_node& node) {
	FzbFeatureComponentName name;
	if (featureComponentMap.count(componentName)) {
		name = featureComponentMap[componentName];
		switch (name) {
			case FZB_RENDERER_FORWARD: return std::make_shared<FzbForwardRender>(node);
			case FZB_FEATURE_COMPONENT_BVH_DEBUG: return std::make_shared<FzbBVH_Debug>(node);
			case FZB_FEATURE_COMPONENT_SVO_DEBUG: return std::make_shared<FzbSVO_Debug>(node);
		}
		return nullptr;
	}
	else throw std::runtime_error("û����Ӧ�Ĺ������" + componentName);
}

FzbRenderer::FzbRenderer(std::string rendererXML) {
	glslang::InitializeProcess();
	initRendererFromXMLInfo(rendererXML);
}
void FzbRenderer::initRendererFromXMLInfo(std::string rendererXML) {
	pugi::xml_document doc;
	auto result = doc.load_file(rendererXML.c_str());
	if (!result) {
		throw std::runtime_error("pugixml���ļ�ʧ��");
	}
	pugi::xml_node rendererInfo = doc.document_element();
	this->rendererName = rendererInfo.child("name").attribute("value").value();
	VkExtent2D resolution;
	resolution.width = std::stoi(rendererInfo.child("resolution").attribute("width").value());
	resolution.height = std::stoi(rendererInfo.child("resolution").attribute("height").value());
	this->globalData.setResolution(resolution);

	std::string scenePath = rendererInfo.child("sceneXML").attribute("path").value();
	this->globalData.mainScene = FzbMainScene(scenePath);	//�����캯��ע��
	
	if (pugi::xml_node rendererTypeNode = rendererInfo.child("rendererType")) {
		std::string rendererType = rendererTypeNode.attribute("value").value();
		this->componentManager.addFeatureComponent(createFzbComponent(rendererType, rendererTypeNode));
	}
	//else throw std::runtime_error("����ָ��һ����Ⱦ��");
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

	imageAvailableSemaphore = FzbSemaphore(false);
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
	vkDeviceWaitIdle(globalData.logicalDevice);
}
void FzbRenderer::drawFrame() {
	vkWaitForFences(globalData.logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);

	uint32_t imageIndex;
	VkResult result = vkAcquireNextImageKHR(globalData.logicalDevice, globalData.swapChain, UINT64_MAX, imageAvailableSemaphore.semaphore, VK_NULL_HANDLE, &imageIndex);
	/*
	//VK_ERROR_OUT_OF_DATE_KHR������������治���ݣ��޷���������Ⱦ��ͨ���ڵ������ڴ�С������
	//VK_SUBOPTIMAL_KHR���������Կ����ڳɹ����ֵ����棬���������Բ�����ȫƥ�䡣
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

	VkSemaphore renderFinishedSemaphore = this->componentManager.componentActivate(imageIndex, imageAvailableSemaphore.semaphore, fence);
	
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &renderFinishedSemaphore; 	// svoSetting.UseSVO ? &fzbSVO->presentSemaphore.semaphore : &renderFinishedSemaphores.semaphore;

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
}

void FzbRenderer::clean() {
	glslang::FinalizeProcess();
	componentManager.clean();
	imageAvailableSemaphore.clean();
	fzbCleanFence(fence);
	globalData.clean();
}