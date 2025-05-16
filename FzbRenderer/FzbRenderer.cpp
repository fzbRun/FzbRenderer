#include "core/common/FzbComponent.h"
#include "core/SceneDivision/SVO/SVO.h"

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
	VkRenderPass renderPass;
	VkPipeline presentPipeline;
	VkPipelineLayout presentPipelineLayout;

	FzbScene scene;
	FzbModel model;
	//std::vector<FzbVertex> vertices;
	//std::vector<uint32_t> indices;

	FzbStorageBuffer<FzbVertex> sceneVertexBuffer;
	FzbStorageBuffer<uint32_t> sceneIndexBuffer;
	FzbUniformBuffer<FzbCameraUniformBufferObject> cameraUniformBuffer;

	VkDescriptorSetLayout uniformDescriptorSetLayout;
	VkDescriptorSet uniformDescriptorSet;

	FzbSemaphore imageAvailableSemaphores;
	FzbSemaphore renderFinishedSemaphores;
	VkFence fence;

	FzbSVOSetting svoSetting = {};
	std::unique_ptr<FzbSVO> fzbSVO;

	void initVulkan() {
		createComponent();
		fzbCreateInstance("FzbRenderer", instanceExtensions, validationLayers, apiVersion);
		setupDebugMessenger();
		fzbCreateSurface();
		createDevice();
		fzbCreateSwapChain();
		initBuffers();
		createModels();
		initComponent();
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

	void createComponent() {

		svoSetting.UseSVO = true;
		if (svoSetting.UseSVO) {
			fzbSVO = std::make_unique<FzbSVO>();
			svoSetting.UseSVO_OnlyVoxelGridMap = false;
			svoSetting.UseBlock = true;
			svoSetting.UseConservativeRasterization = false;
			svoSetting.UseSwizzle = true;
			svoSetting.Present = true;
			svoSetting.voxelNum = 64;
			fzbSVO->addExtensions(svoSetting, instanceExtensions, deviceExtensions, deviceFeatures);
		}
	}

	void createDevice() {
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.geometryShader = VK_TRUE;
		deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
		fzbCreateDevice(&deviceFeatures, deviceExtensions, pNextFeatures, validationLayers);
	}

	void initBuffers() {
		fzbCreateCommandPool();
		fzbCreateCommandBuffers(1);
	}

	void createModels() {
		model = fzbCreateModel("models/dragon.obj");
		scene.sceneModels.push_back(&model);

		fzbOptimizeScene(&scene, scene.sceneVertices, scene.sceneIndices);
		this->sceneVertexBuffer.data = scene.sceneVertices;
		this->sceneIndexBuffer.data = scene.sceneIndices;
		scene.AABB = fzbMakeAABB(scene.sceneVertices);


	}

	void initComponent() {
		if (svoSetting.UseSVO)
			fzbSVO->init(this, &scene, svoSetting);
	}

	void activateComponent() {
		if (svoSetting.UseSVO)
			fzbSVO->activate();
	}

	void createBuffers() {
		cameraUniformBuffer = fzbCreateUniformBuffers<FzbCameraUniformBufferObject>();
	}

	void createImages() {
	}

	void createDescriptor() {

		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		fzbCreateDescriptorPool(bufferTypeAndNum);

		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL };
		uniformDescriptorSetLayout = fzbCreateDescriptLayout(1, descriptorTypes, descriptorShaderFlags);
		uniformDescriptorSet = fzbCreateDescriptorSet(uniformDescriptorSetLayout);

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

	void prepareComponentPresent() {
		if (svoSetting.UseSVO)
			fzbSVO->presentPrepare(uniformDescriptorSetLayout);
	}

	void createRenderPass() {

	}

	void createFramebuffers() {
	}

	void createPipeline() {

	}

	void createSyncObjects() {
		imageAvailableSemaphores = fzbCreateSemaphore(false);
		renderFinishedSemaphores = fzbCreateSemaphore(false);
		fence = fzbCreateFence();
	}

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
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}


		updateUniformBuffer();

		vkResetFences(logicalDevice, 1, &fence);
		if (svoSetting.UseSVO)
			fzbSVO->present(uniformDescriptorSet, imageIndex, imageAvailableSemaphores.semaphore, fence);

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &fzbSVO->presentSemaphore.semaphore;

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

	}

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

		VkBuffer vertexBuffers[] = { sceneVertexBuffer.buffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, sceneIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipelineLayout, 0, 1, &uniformDescriptorSet, 0, nullptr);

		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(this->sceneIndexBuffer.data.size()), 1, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

	}

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

	}

	void clean() {

		fzbSVO->clean();
		sceneVertexBuffer.clean();
		sceneIndexBuffer.clean();
		cameraUniformBuffer.clean();

		cleanupSwapChain();

		vkDestroyPipeline(logicalDevice, presentPipeline, nullptr);
		vkDestroyPipelineLayout(logicalDevice, presentPipelineLayout, nullptr);

		vkDestroyRenderPass(logicalDevice, renderPass, nullptr);

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

	return EXIT_SUCCESS;

}