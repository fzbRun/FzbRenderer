#include "commonLib/commonAPP.h"
#include "commonLib/SVO/SVO.h"

/*
写这个光栅体素化花了很多时间，主要原因在于对于投影的认识不熟悉
一开始学习光栅体素化时看别人的教程，大家都是光栅体素化后再渲染一遍，采样体素得到结果，那么我就想能不能在光栅体素化时同时得到深度图
那么就不需要重新渲染一遍了，只需要通过深度重构得到世界坐标再去采样体素即可，从而变为屏幕空间的性能消耗
但是忽略了光栅体素化的投影是固定三个视点的正交投影，这会导致对于一个三角形，无论相机的远近，在片元着色器得到的像素数量都固定，因此深度图中有值的纹素数量固定
只是随着相机近时分散，远时集中罢了。那么若相机拉近，采样深度时大多数时候会采样到其他三角形的深度或默认值，导致被剔除，因此渲染结果会是散点而不是fill的
因此，我们需要使用相机的透视投影而不是固定视点的正交投影，但是问题在于使用随着相机的移动，另外的两个投影面如何移动；并且使用透视投影无法使用swizzle
并且大多数时候光栅体素化的目标是静态顶点，不需要每帧重构，因此也就不会进入渲染循环，因此深度图还是要通过其他方式获得
因此，我做了个答辩，不过在这个过程中还是有一些收获的
1. 熟悉了投影
2. 熟悉了swizzle和多视口
*/

class Voxelization : CommonApp {

public:
	void run() {
		initWindow(512, 512, "Voxelization", VK_FALSE);
		initVulkan();
		mainLoop();
		cleanupAll();
	}

private:
	VkRenderPass renderPass;
	VkPipeline presentPipeline;
	VkPipelineLayout presentPipelineLayout;

	MyModel model;
	vector<Vertex_onlyPos> vertices;
	vector<uint32_t> indices;

	VkDescriptorSetLayout uniformDescriptorSetLayout;
	VkDescriptorSet uniformDescriptorSet;

	VkSemaphore imageAvailableSemaphores;
	VkSemaphore renderFinishedSemaphores;
	VkFence fence;

	uint32_t currentFrame = 0;

	const uint32_t voxelNum = 64;

	FzbSVOSetting svoSetting = {};
	std::unique_ptr<FzbSVO> fzbSVO;

	void initVulkan() {
		setComponent();
		createInstance();
		setupDebugMessenger();
		createSurface();
		createDevice();
		createSwapChain();
		initBuffers();
		createModels();
		addComponent();
		activateComponent();
		createBuffers();
		createImages();
		createDescriptor();
		prepareComponentPresent();
		createRenderPass();
		createFramebuffers();
		createPipeline();
		createSyncObjects();
	}

	void setComponent() {

		svoSetting.UseSVO = true;
		svoSetting.UseSVO_OnlyVoxelGridMap = false;
		svoSetting.UseBlock = false;
		svoSetting.UseConservativeRasterization = false;
		svoSetting.UseSwizzle = false;
		svoSetting.Present = true;
		svoSetting.voxelNum = voxelNum;
		FzbSVO::addExtensions(svoSetting, instanceExtensions, deviceExtensions, deviceFeatures);

	}

	void createInstance() {
		fzbCreateInstance("FzbRenderer", instanceExtensions);
	}

	void createDevice() {
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.geometryShader = VK_TRUE;
		deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
		fzbCreateDevice(deviceExtensions, &deviceFeatures);
	}

	void initBuffers() {
		createCommandPool();
		createCommandBuffers(1);
	}

	void createModels() {
		model = loadModel("models/dragon.obj");
	}

	void addComponent() {
		fzbSVO = std::make_unique<FzbSVO>(fzbDevice, fzbSwapchain, commandPool, &model, &svoSetting);
	}

	void activateComponent() {
		fzbSVO->activate();
	}

	void createBuffers() {
		createUniformBuffers(sizeof(UniformBufferObject), false, 1);
	}

	void createImages() {

	}

	void createDescriptor() {

		map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		createDescriptorPool(bufferTypeAndNum);

		vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
		vector<VkShaderStageFlagBits> descriptorShaderFlags = { VK_SHADER_STAGE_ALL };
		uniformDescriptorSetLayout = createDescriptLayout(1, descriptorTypes, descriptorShaderFlags);
		uniformDescriptorSet = createDescriptorSet(uniformDescriptorSetLayout);

		std::array<VkWriteDescriptorSet, 1> uniformDescriptorWrites{};
		VkDescriptorBufferInfo cameraUniformBufferInfo{};
		cameraUniformBufferInfo.buffer = uniformBuffers[0];
		cameraUniformBufferInfo.offset = 0;
		cameraUniformBufferInfo.range = sizeof(UniformBufferObject);
		uniformDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		uniformDescriptorWrites[0].dstSet = uniformDescriptorSet;
		uniformDescriptorWrites[0].dstBinding = 0;
		uniformDescriptorWrites[0].dstArrayElement = 0;
		uniformDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uniformDescriptorWrites[0].descriptorCount = 1;
		uniformDescriptorWrites[0].pBufferInfo = &cameraUniformBufferInfo;

		vkUpdateDescriptorSets(logicalDevice, uniformDescriptorWrites.size(), uniformDescriptorWrites.data(), 0, nullptr);
		descriptorSets.push_back({ uniformDescriptorSet });

	}

	void prepareComponentPresent() {
		fzbSVO->presentPrepare(swapChainImageFormat, swapChainImageViews, uniformDescriptorSetLayout);
	}

	void createRenderPass() {
	}

	void createFramebuffers() {
	}

	void createPipeline() {

	}

	void createSyncObjects() {
		imageAvailableSemaphores = createSemaphore();
		renderFinishedSemaphores = createSemaphore();
		fence = createFence();
	}

	void waitComponentFinished() {
		//应该使用信号量而不是栏栅
		//if (currentFrame == 0)
		//	vkWaitForFences(logicalDevice, 1, &fzbSVO->fence, VK_TRUE, UINT64_MAX);
	}

	void drawFrame() {

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		waitComponentFinished();
		vkWaitForFences(logicalDevice, 1, &fence, VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(logicalDevice, swapChain, UINT64_MAX, imageAvailableSemaphores, VK_NULL_HANDLE, &imageIndex);
		//VK_ERROR_OUT_OF_DATE_KHR：交换链与表面不兼容，无法再用于渲染。通常在调整窗口大小后发生。
		//VK_SUBOPTIMAL_KHR：交换链仍可用于成功呈现到表面，但表面属性不再完全匹配。
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
			framebufferResized = false;
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		updateUniformBuffer();

		vkResetFences(logicalDevice, 1, &fence);
		fzbSVO->present(uniformDescriptorSet, imageIndex, imageAvailableSemaphores, fence);

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &fzbSVO->fzbSync->fzbSemaphores[svoSetting.UseSVO_OnlyVoxelGridMap ? 1 : 2].semaphore;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;
		result = vkQueuePresentKHR(presentQueue, &presentInfo);
		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		currentFrame = (currentFrame + 1) % UINT32_MAX;

	}

	void updateUniformBuffer() {

		float currentTime = static_cast<float>(glfwGetTime());
		deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
		lastTime = currentTime;

		UniformBufferObject ubo{};
		ubo.model = glm::mat4(1.0f);	// glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		ubo.view = camera.GetViewMatrix();
		ubo.proj = glm::perspectiveRH_ZO(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 100.0f);
		//ubo.view = glm::lookAt(glm::vec3(0, 5, 10), glm::vec3(0, 5, 10) + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		//ubo.proj = glm::orthoRH_ZO(-10.0f, 10.0f, -10.0f, 10.0f, 0.1f, 20.1f);
		//怪不得，我从obj文件中看到场景的顶点是顺时针的，但是在shader中得是逆时针才对，原来是这里proj[1][1]1 *= -1搞的鬼
		//那我们在计算着色器中处理顶点数据似乎不需要这个啊
		ubo.proj[1][1] *= -1;
		ubo.cameraPos = glm::vec4(camera.Position, 0.0f);
		ubo.swapChainExtent = glm::vec4(swapChainExtent.width, swapChainExtent.height, 0.0f, 0.0f);

		memcpy(uniformBuffersMappeds[0], &ubo, sizeof(ubo));

	}

	void cleanupImages() {
		/*
		if (voxelImage.textureSampler) {
			vkDestroySampler(logicalDevice, voxelImage.textureSampler, nullptr);
		}
		vkDestroyImageView(logicalDevice, voxelImage.imageView, nullptr);
		vkDestroyImage(logicalDevice, voxelImage.image, nullptr);
		vkFreeMemory(logicalDevice, voxelImage.imageMemory, nullptr);

		if (depthBuffer.textureSampler) {
			vkDestroySampler(logicalDevice, depthBuffer.textureSampler, nullptr);
		}
		vkDestroyImageView(logicalDevice, depthBuffer.imageView, nullptr);
		vkDestroyImage(logicalDevice, depthBuffer.image, nullptr);
		vkFreeMemory(logicalDevice, depthBuffer.imageMemory, nullptr);


		if (testTexture.textureSampler) {
			vkDestroySampler(logicalDevice, testTexture.textureSampler, nullptr);
		}
		vkDestroyImageView(logicalDevice, testTexture.imageView, nullptr);
		vkDestroyImage(logicalDevice, testTexture.image, nullptr);
		vkFreeMemory(logicalDevice, testTexture.imageMemory, nullptr);
		*/

	}

	void cleanupAll() {

		fzbSVO->clean();

		cleanupSwapChain();

		//清理管线
		//vkDestroyPipeline(logicalDevice, voxelPipeline, nullptr);
		//vkDestroyPipelineLayout(logicalDevice, voxelPipelineLayout, nullptr);
		vkDestroyPipeline(logicalDevice, presentPipeline, nullptr);
		vkDestroyPipelineLayout(logicalDevice, presentPipelineLayout, nullptr);
		//清理渲染Pass
		vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

		vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, uniformDescriptorSetLayout, nullptr);

		//清理描述符集合布局
		//清理信号量和栏栅
		vkDestroySemaphore(logicalDevice, imageAvailableSemaphores, nullptr);
		vkDestroySemaphore(logicalDevice, renderFinishedSemaphores, nullptr);
		vkDestroyFence(logicalDevice, fence, nullptr);

		cleanupBuffers();

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

	Voxelization app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		system("pause");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

}