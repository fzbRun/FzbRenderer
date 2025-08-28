#include "core/common/FzbComponent.h"
#include "core/ForwardRender/FzbForwardRender.h"
#include "core/SceneDivision/SVO/SVO.h"

#include <glslang/Public/ShaderLang.h>

//#include<opencv2/opencv.hpp>
//#include <torch/torch.h>
//#include <torch/script.h> 

class FzbRenderer : public FzbMainComponent {

public:

	void run() {
		createGlobalData();
		initVulkan();
		mainLoop();
		clean();
	}

private:

	FzbScene scene;

	FzbSemaphore imageAvailableSemaphore;
	VkFence fence;

	std::vector<FzbRenderPass*> renderPasses;	//�����ڴ�С�ı������ʹ�õ�swapChain��֡���嶼��Ҫ���´�������һ�����鷽���޸�
	std::vector<FzbVertexFormat> componentVertexFormats;
	FzbForwardRenderSetting forwardRenderSetting;
	std::unique_ptr<FzbForwardRender> fzbForwardRender;
	FzbSVOSetting svoSetting = {};
	std::unique_ptr<FzbSVO> fzbSVO;
//---------------------------------------------����ȫ�ֱ���---------------------------------------------
	void createGlobalData() {
		glslang::InitializeProcess();	//��ʼ��GLSLang�⣬ȫ�ֳ�ʼ�����ڳ�������ʱ����һ��
		this->scene.getSceneGlobalInfo("./models/veach-ajar");	//��ȡsceneXML�е�ȫ����Ϣ
		createComponent();
	}
	void createComponent() {
		svoSetting.UseSVO = true;
		if (svoSetting.UseSVO) {
			fzbSVO = std::make_unique<FzbSVO>();
			svoSetting.UseSVO_OnlyVoxelGridMap = false;
			svoSetting.UseConservativeRasterization = false;
			svoSetting.UseSwizzle = false;	//�����е����⣬��̫���ã��о���vulkan�ü�����z��[0,1]������[-1,1]�йأ�
			svoSetting.UseBlock = false;
			svoSetting.Present = true;
			svoSetting.useCube = true;
			svoSetting.voxelNum = 64;
			fzbSVO->addExtensions(svoSetting, instanceExtensions, deviceExtensions, deviceFeatures);
			componentVertexFormats.push_back(fzbSVO->getComponentVertexFormat());
		}
		forwardRenderSetting.useForwardRender = false;
		if (forwardRenderSetting.useForwardRender) {
			fzbForwardRender = std::make_unique<FzbForwardRender>();
			fzbForwardRender->addExtensions(forwardRenderSetting, instanceExtensions, deviceExtensions, deviceFeatures);
			//componentVertexFormats.push_back(fzbForwardRender->getComponentVertexFormat());
		}
	}
//-----------------------------------------------��ʼ��Vulkan-------------------------------------------------
	void initVulkan() {
		fzbInitWindow(this->scene.width, this->scene.height, "FzbRenderer", VK_FALSE);
		fzbCreateInstance("FzbRenderer", instanceExtensions, validationLayers);
		fzbCetupDebugMessenger();
		fzbCreateSurface();
		createDevice();
		fzbCreateSwapChain();
		initBuffers();
		createScene();
		initComponent();
		createSyncObjects();
	}
	void createDevice() {
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.geometryShader = VK_TRUE;
		deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
		deviceFeatures.multiDrawIndirect = VK_TRUE;

		VkPhysicalDeviceVulkan11Features vk11Features{};
		vk11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
		vk11Features.shaderDrawParameters = VK_TRUE;

		VkPhysicalDeviceVulkan12Features vk12Features{};
		vk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		vk12Features.drawIndirectCount = VK_TRUE;
		vk12Features.descriptorIndexing = VK_TRUE;

		VkPhysicalDeviceFeatures2 feature2 = createPhysicalDeviceFeatures(deviceFeatures, &vk11Features, &vk12Features);

		fzbCreateDevice(nullptr, deviceExtensions, &feature2);
	}
	void initBuffers() {
		fzbCreateCommandPool();
		fzbCreateCommandBuffers(1);
	}
	void createScene() {
		scene.initScene(physicalDevice, logicalDevice, commandPool, graphicsQueue, componentVertexFormats);
		this->camera = &scene.sceneCameras[0];
	}
	void initComponent() {
		if (forwardRenderSetting.useForwardRender) fzbForwardRender->init(this, forwardRenderSetting, &scene, renderPasses);
		if (svoSetting.UseSVO) fzbSVO->init(this, svoSetting, &scene, renderPasses);
	}
	void createSyncObjects() {
		imageAvailableSemaphore = FzbSemaphore(logicalDevice, false);
		fence = fzbCreateFence();
	}

//--------------------------------------------------------
	void drawFrame() {

		vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(logicalDevice, swapChain, UINT64_MAX, imageAvailableSemaphore.semaphore, VK_NULL_HANDLE, &imageIndex);
		//VK_ERROR_OUT_OF_DATE_KHR������������治���ݣ��޷���������Ⱦ��ͨ���ڵ������ڴ�С������
		//VK_SUBOPTIMAL_KHR���������Կ����ڳɹ����ֵ����棬���������Բ�����ȫƥ�䡣
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain(renderPasses);
			return;
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		updateGlobalData();
		vkResetFences(logicalDevice, 1, &fence);

		//if (svoSetting.UseSVO) {
		//	fzbSVO->present(uniformDescriptorSet, imageIndex, imageAvailableSemaphores.semaphore, fence);
		//}
		//if (false) {}
		//else {
		//	VkCommandBufferBeginInfo beginInfo{};
		//	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		//	beginInfo.flags = 0;
		//	beginInfo.pInheritanceInfo = nullptr;
		//
		//	if (vkBeginCommandBuffer(commandBuffers[0], &beginInfo) != VK_SUCCESS) {
		//		throw std::runtime_error("failed to begin recording command buffer!");
		//	}
		//
		//	renderPass.render(commandBuffers[0], imageIndex, { &scene }, { {uniformDescriptorSet} });
		//
		//	VkSemaphore waitSemaphores[] = { imageAvailableSemaphores.semaphore };
		//	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
		//	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		//	submitInfo.waitSemaphoreCount = 1;
		//	submitInfo.pWaitSemaphores = waitSemaphores;
		//	submitInfo.pWaitDstStageMask = waitStages;
		//	submitInfo.commandBufferCount = 1;
		//	submitInfo.pCommandBuffers = &commandBuffers[0];
		//	submitInfo.signalSemaphoreCount = 1;
		//	submitInfo.pSignalSemaphores = &renderFinishedSemaphores.semaphore;
		//
		//	if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
		//		throw std::runtime_error("failed to submit draw command buffer!");
		//	}
		//}

		VkSemaphore renderFinishedSemaphore;
		if(forwardRenderSetting.useForwardRender) renderFinishedSemaphore = fzbForwardRender->render(imageIndex, imageAvailableSemaphore.semaphore, fence);
		if (svoSetting.UseSVO && svoSetting.Present) renderFinishedSemaphore = fzbSVO->render(imageIndex, imageAvailableSemaphore.semaphore, fence);

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinishedSemaphore; 	// svoSetting.UseSVO ? &fzbSVO->presentSemaphore.semaphore : &renderFinishedSemaphores.semaphore;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		result = vkQueuePresentKHR(presentQueue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			recreateSwapChain(renderPasses);
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}

	}
	/*
	void presentDrawCall(VkCommandBuffer commandBuffer, uint32_t imageIndex) {

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassBeginInfo{};
		renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.framebuffer = framebuffers[0][imageIndex];
		renderPassBeginInfo.renderArea.offset = { 0, 0 };
		renderPassBeginInfo.renderArea.extent = swapChainExtent;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };
		renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassBeginInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkBuffer vertexBuffers[] = { scene.vertexBuffer.buffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		presentSubPass.render(commandBuffer, uniformDescriptorSet, scene.descriptorSet);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

	}
	*/
	void updateGlobalData() {
		float currentTime = static_cast<float>(glfwGetTime());
		deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
		lastTime = currentTime;
	}

	void cleanupImages() {}

	void clean() {
		glslang::FinalizeProcess();

		if(svoSetting.UseSVO) fzbSVO->clean();
		if(forwardRenderSetting.useForwardRender) fzbForwardRender->clean();
		scene.clean();

		fzbCleanupSwapChain();

		vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);

		imageAvailableSemaphore.clean(logicalDevice);
		vkDestroyFence(logicalDevice, fence, nullptr);

		vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

		vkDestroyDevice(logicalDevice, nullptr);

		if (enableValidationLayers) DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}
};

int main() {

	FzbRenderer app;
	
	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		system("pause");
		return EXIT_FAILURE;
	}
	
	//try {
	//	//����ʹ��cuda
	//	auto device = torch::Device(torch::kCUDA, 0);
	//	//��ȡͼƬ
	//	auto image = cv::imread("C:/Users/fangzanbo/Desktop/AI/DeepLearningTest/pythonDeepLearningTest/test/images/flower.jpg");
	//	//������ָ����С
	//	cv::resize(image, image, cv::Size(224, 224));
	//
	//	//torch::set_num_threads(4);
	//
	//	//ת������
	//	auto input_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat32) / 225.0;
	//	//����ģ��
	//	auto model = torch::jit::load("C:/Users/fangzanbo/Desktop/AI/DeepLearningTest/pythonDeepLearningTest/test/resnet34.pt");
	//	model.to(device);
	//	model.eval();
	//	//ǰ�򴫲�
	//	auto output = model.forward({ input_tensor.to(device) }).toTensor();
	//	output = torch::softmax(output, 1);
	//	std::cout << "ģ��Ԥ����Ϊ��" << torch::argmax(output) << "�࣬���Ŷ�Ϊ" << output.max() << std::endl;
	//}
	//catch (const std::exception& e) {
	//	std::cerr << "LibTorch ��ع����׳��쳣��" << e.what() << std::endl;
	//	system("pause");
	//	return EXIT_FAILURE;
	//}
	
	return EXIT_SUCCESS;

}