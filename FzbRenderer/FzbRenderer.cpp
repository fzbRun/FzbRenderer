#include "core/common/FzbComponent.h"
//#include "core/SceneDivision/SVO/SVO.h"

//#include<opencv2/opencv.hpp>
//#include <torch/torch.h>
//#include <torch/script.h> 

class FzbRenderer : public FzbMainComponent {

public:

	void run() {
		camera = FzbCamera(glm::vec3(0.0f, 5.0f, 18.0f));
		fzbInitWindow(512, 512, "FzbRenderer", VK_FALSE);
		initVulkan();
		mainLoop();
		clean();
	}

private:

	FzbScene scene;

	FzbBuffer cameraUniformBuffer;
	FzbImage depthMap;

	VkDescriptorSetLayout uniformDescriptorSetLayout;
	VkDescriptorSet uniformDescriptorSet;

	FzbRenderPass renderPass;

	FzbSemaphore imageAvailableSemaphores;
	FzbSemaphore renderFinishedSemaphores;
	VkFence fence;

	//FzbSVOSetting svoSetting = {};
	//std::unique_ptr<FzbSVO> fzbSVO;

	void initVulkan() {
		initSetting();
		createScene();
		createPreparePresentSetting();
		//if (svoSetting.UseSVO) {
		//	activateSVOComponent();
		//}
		//else {
		//	createAppRenderPass();
		//}
		createAppRenderPass();
		
	}
	//---------------------------------------------------------------------------------------------------
	/*void createComponent() {

		svoSetting.UseSVO = false;
		if (svoSetting.UseSVO) {
			fzbSVO = std::make_unique<FzbSVO>();
			svoSetting.UseSVO_OnlyVoxelGridMap = false;
			svoSetting.UseBlock = true;
			svoSetting.UseConservativeRasterization = false;
			svoSetting.UseSwizzle = false;
			svoSetting.Present = true;
			svoSetting.voxelNum = 64;
			fzbSVO->addExtensions(svoSetting, instanceExtensions, deviceExtensions, deviceFeatures);
		}
	}*/

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

	void initSetting() {
		//createComponent();
		fzbCreateInstance("FzbRenderer", instanceExtensions, validationLayers);
		fzbCetupDebugMessenger();
		fzbCreateSurface();
		createDevice();
		fzbCreateSwapChain();
		initBuffers();
	}
//---------------------------------------------------------------------------------------------------
	void createScene() {

		FzbVertexFormat vertexFormat(true, true);
		//if (svoSetting.UseSVO) vertexFormat.mergeUpward(fzbSVO->getComponentVertexFormat());

		scene = FzbScene(physicalDevice, logicalDevice, commandPool, graphicsQueue);
		scene.addSceneFromMitsubaXML("./models/veach-ajar");

		scene.initScene();
	}
//---------------------------------------------------------------------------------------------------
	void createBuffers() {
		cameraUniformBuffer = fzbComponentCreateUniformBuffers<FzbCameraUniformBufferObject>();
	}

	void createDescriptor() {

		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		fzbComponentCreateDescriptorPool(bufferTypeAndNum);

		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL };
		uniformDescriptorSetLayout = fzbComponentCreateDescriptLayout(1, descriptorTypes, descriptorShaderFlags);
		uniformDescriptorSet = fzbComponentCreateDescriptorSet(uniformDescriptorSetLayout);

		std::array<VkWriteDescriptorSet, 1> uniformDescriptorWrites{};
		VkDescriptorBufferInfo cameraUniformBufferInfo{};
		cameraUniformBufferInfo.buffer = cameraUniformBuffer.buffer;
		cameraUniformBufferInfo.offset = 0;
		cameraUniformBufferInfo.range = sizeof(FzbCameraUniformBufferObject);
		uniformDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		uniformDescriptorWrites[0].dstSet = uniformDescriptorSet;
		uniformDescriptorWrites[0].dstBinding = 0;
		uniformDescriptorWrites[0].dstArrayElement = 0;
		uniformDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uniformDescriptorWrites[0].descriptorCount = 1;
		uniformDescriptorWrites[0].pBufferInfo = &cameraUniformBufferInfo;

		vkUpdateDescriptorSets(logicalDevice, uniformDescriptorWrites.size(), uniformDescriptorWrites.data(), 0, nullptr);

	}

	void createSyncObjects() {
		imageAvailableSemaphores = fzbCreateSemaphore(false);
		renderFinishedSemaphores = fzbCreateSemaphore(false);
		fence = fzbCreateFence();
	}

	void createPreparePresentSetting() {
		createBuffers();
		createDescriptor();
		createSyncObjects();
	}
//--------------------------默认展示------------------------------
	void createImages() {
		depthMap = {};
		depthMap.width = swapChainExtent.width;
		depthMap.height = swapChainExtent.height;
		depthMap.type = VK_IMAGE_TYPE_2D;
		depthMap.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthMap.format = fzbFindDepthFormat(physicalDevice);
		depthMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		depthMap.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
		depthMap.fzbCreateImage(physicalDevice, logicalDevice, commandPool, graphicsQueue);
	}
	void createRenderPass() {
		VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(swapChainImageFormat);
		VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment(physicalDevice);
		VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

		VkSubpassDescription presentSubpass = fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef);
		std::vector<VkSubpassDescription> subpasses = { presentSubpass };

		std::vector<VkSubpassDependency> dependencies = { fzbCreateSubpassDependency() };

		FzbRenderPassSetting setting = { true, 1, swapChainExtent, swapChainImageViews.size(), true };
		renderPass = FzbRenderPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, setting);
		renderPass.images.push_back(&depthMap);
		renderPass.createRenderPass(&attachments, subpasses, dependencies);
		renderPass.createFramebuffers(swapChainImageViews);

		FzbSubPass presentSubPass = FzbSubPass(physicalDevice, logicalDevice, commandPool, graphicsQueue, renderPass.renderPass, 0);
		presentSubPass.createMeshBatch(&scene, { uniformDescriptorSetLayout });	//可以得到meshBatch，并为meshBatch创建各种缓冲，以及shader的pipeline
		renderPass.subPasses.push_back(presentSubPass);
	}
	void createAppRenderPass() {
		createImages();
		createRenderPass();
	}
//----------------------组件展示----------------------------------

	//void initComponent() {
	//	if (svoSetting.UseSVO)
	//		fzbSVO->init(this, &scene, svoSetting);
	//}

	//void activateComponent() {
	//	if (svoSetting.UseSVO)
	//		fzbSVO->activate();
	//}

	//void prepareComponentPresent() {
	//	if (svoSetting.UseSVO && svoSetting.Present)
	//		fzbSVO->presentPrepare(uniformDescriptorSetLayout);
	//}

	//void activateSVOComponent() {
	//	initComponent();
	//	activateComponent();
	//	prepareComponentPresent();
	//}

//--------------------------------------------------------
	void drawFrame() {

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(logicalDevice, swapChain, UINT64_MAX, imageAvailableSemaphores.semaphore, VK_NULL_HANDLE, &imageIndex);
		//VK_ERROR_OUT_OF_DATE_KHR：交换链与表面不兼容，无法再用于渲染。通常在调整窗口大小后发生。
		//VK_SUBOPTIMAL_KHR：交换链仍可用于成功呈现到表面，但表面属性不再完全匹配。
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain({renderPass});
			return;
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		updateUniformBuffer();
		vkResetFences(logicalDevice, 1, &fence);

		//if (svoSetting.UseSVO) {
		//	fzbSVO->present(uniformDescriptorSet, imageIndex, imageAvailableSemaphores.semaphore, fence);
		//}
		if (false) {}
		else {
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = 0;
			beginInfo.pInheritanceInfo = nullptr;

			if (vkBeginCommandBuffer(commandBuffers[0], &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording command buffer!");
			}

			renderPass.render(commandBuffers[0], imageIndex, { &scene }, { {uniformDescriptorSet} });

			VkSemaphore waitSemaphores[] = { imageAvailableSemaphores.semaphore };
			VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.waitSemaphoreCount = 1;
			submitInfo.pWaitSemaphores = waitSemaphores;
			submitInfo.pWaitDstStageMask = waitStages;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffers[0];
			submitInfo.signalSemaphoreCount = 1;
			submitInfo.pSignalSemaphores = &renderFinishedSemaphores.semaphore;

			if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
				throw std::runtime_error("failed to submit draw command buffer!");
			}
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinishedSemaphores.semaphore; ;	// svoSetting.UseSVO ? &fzbSVO->presentSemaphore.semaphore : &renderFinishedSemaphores.semaphore;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		result = vkQueuePresentKHR(presentQueue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			recreateSwapChain({ renderPass });
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
	void updateUniformBuffer() {

		float currentTime = static_cast<float>(glfwGetTime());
		deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
		lastTime = currentTime;

		FzbCameraUniformBufferObject ubo{};
		ubo.model = glm::mat4(1.0f);	// glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		ubo.view = camera.GetViewMatrix();
		ubo.proj = glm::perspectiveRH_ZO(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 100.0f);
		//ubo.view = glm::lookAt(glm::vec3(0, 5, 10), glm::vec3(0, 5, 10) + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		//ubo.proj = glm::orthoRH_ZO(-10.0f, 10.0f, -10.0f, 10.0f, 0.1f, 20.1f);
		//怪不得，我从obj文件中看到场景的顶点是顺时针的，但是在shader中得是逆时针才对，原来是这里proj[1][1]1 *= -1搞的鬼
		//那我们在计算着色器中处理顶点数据似乎不需要这个啊
		ubo.proj[1][1] *= -1;
		ubo.cameraPos = glm::vec4(camera.Position, 0.0f);

		memcpy(cameraUniformBuffer.mapped, &ubo, sizeof(ubo));

	}

	void cleanupImages() {
		depthMap.clean();
	}

	void clean() {

		//if(svoSetting.UseSVO) fzbSVO->clean();
		scene.clean();
		renderPass.clean();
		cameraUniformBuffer.clean();

		fzbCleanupSwapChain();

		vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, uniformDescriptorSetLayout, nullptr);

		vkDestroySemaphore(logicalDevice, imageAvailableSemaphores.semaphore, nullptr);
		vkDestroySemaphore(logicalDevice, renderFinishedSemaphores.semaphore, nullptr);
		vkDestroyFence(logicalDevice, fence, nullptr);

		vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

		vkDestroyDevice(logicalDevice, nullptr);

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

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
	//	//定义使用cuda
	//	auto device = torch::Device(torch::kCUDA, 0);
	//	//读取图片
	//	auto image = cv::imread("C:/Users/fangzanbo/Desktop/AI/DeepLearningTest/pythonDeepLearningTest/test/images/flower.jpg");
	//	//缩放至指定大小
	//	cv::resize(image, image, cv::Size(224, 224));
	//
	//	//torch::set_num_threads(4);
	//
	//	//转成张量
	//	auto input_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat32) / 225.0;
	//	//加载模型
	//	auto model = torch::jit::load("C:/Users/fangzanbo/Desktop/AI/DeepLearningTest/pythonDeepLearningTest/test/resnet34.pt");
	//	model.to(device);
	//	model.eval();
	//	//前向传播
	//	auto output = model.forward({ input_tensor.to(device) }).toTensor();
	//	output = torch::softmax(output, 1);
	//	std::cout << "模型预测结果为第" << torch::argmax(output) << "类，置信度为" << output.max() << std::endl;
	//}
	//catch (const std::exception& e) {
	//	std::cerr << "LibTorch 相关过程抛出异常：" << e.what() << std::endl;
	//	system("pause");
	//	return EXIT_FAILURE;
	//}
	
	return EXIT_SUCCESS;

}