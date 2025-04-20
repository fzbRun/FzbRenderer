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
	//VkPipeline voxelPipeline;
	//VkPipelineLayout voxelPipelineLayout;
	VkPipeline presentPipeline;
	VkPipelineLayout presentPipelineLayout;

	MyModel model;
	vector<Vertex_onlyPos> vertices;
	vector<uint32_t> indices;

	vector<Vertex_onlyPos> cubeVertices;
	vector<uint32_t> cubeIndices;

	VkDescriptorSetLayout uniformDescriptorSetLayout;
	VkDescriptorSet uniformDescriptorSet;
	VkDescriptorSetLayout voxelDescriptorSetLayout;
	VkDescriptorSet voxelDescriptorSet;
	VkDescriptorSetLayout presentDescriptorSetLayout;
	VkDescriptorSet presentDescriptorSet;
	VkDescriptorSetLayout depthDescriptorSetLayout;
	VkDescriptorSet depthDescriptorSet;

	VkSemaphore imageAvailableSemaphores;
	VkSemaphore renderFinishedSemaphores;
	VkFence fence;

	uint32_t currentFrame = 0;

	const uint32_t voxelNum = 64;
	//MyImage voxelImage;
	//MyImage depthImage;
	//MyImage depthBuffer;
	//MyImage testTexture;
	//const uint32_t depthImageAccuracy = 33554432;

	FzbSVOSetting svoSetting = {};
	std::unique_ptr<FzbSVO> fzbSVO;

	void initVulkan() {
		initComponent();
		createInstance("Voxelization", { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME });
		setupDebugMessenger();
		createSurface();
		createDevice();
		createSwapChain();
		initBuffers();
		createModels();
		addComponent();
		createBuffers();
		createImages();
		createDescriptor();
		createRenderPass();
		createFramebuffers();
		createPipeline();
		createSyncObjects();
	}

	void initComponent() {
		svoSetting.UseSVO = true;
		svoSetting.UseBlock = true;
		svoSetting.UseConservativeRasterization = false;
		svoSetting.UseSwizzle = false;
		svoSetting.voxelNum = voxelNum;
	}

	void createDevice() {

		vector<const char*> deviceExtensions = {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
		};
		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.geometryShader = VK_TRUE;
		deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;

		if (svoSetting.UseSVO) {
			FzbSVO::getDeviceExtensions(svoSetting, deviceExtensions);
			FzbSVO::getDeviceFeatures(svoSetting, deviceFeatures);
		}

		fzbCreateDevice(deviceExtensions, &deviceFeatures);


	}

	void initBuffers() {
		createCommandPool();
		createCommandBuffers(1);
	}

	void createModels() {
		model = loadModel("models/dragon.obj");

		optimizeModel<Vertex_onlyPos>(model, vertices, indices);
		makeAABB(model);

		if (svoSetting.UseBlock) {
			glm::vec3 cubeVertexOffset[8] = { glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f),
							  glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 0.0f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(0.0f, 1.0f, 1.0f) };
			float distanceX = model.AABB.rightX - model.AABB.leftX;
			float distanceY = model.AABB.rightY - model.AABB.leftY;
			float distanceZ = model.AABB.rightZ - model.AABB.leftZ;
			float distance = glm::max(distanceX, glm::max(distanceY, distanceZ));
			float voxelSize = distance / voxelNum;

			float centerX = (model.AABB.rightX + model.AABB.leftX) * 0.5f;
			float centerY = (model.AABB.rightY + model.AABB.leftY) * 0.5f;
			float centerZ = (model.AABB.rightZ + model.AABB.leftZ) * 0.5f;

			this->cubeVertices.resize(8);
			for (int i = 0; i < 8; i++) {
				this->cubeVertices[i].pos = cubeVertexOffset[i] * voxelSize + glm::vec3(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f);
			}
			this->cubeIndices = {
				1, 0, 3, 1, 3, 2,
				4, 5, 6, 4, 6, 7,
				5, 1, 2, 5, 2, 6,
				0, 4, 7, 0, 7, 3,
				7, 6, 2, 7, 2, 3,
				0, 1, 5, 0, 5, 4
			};
		}

	}

	void addComponent() {
		fzbSVO = std::make_unique<FzbSVO>(fzbDevice, fzbSwapchain, commandPool, model, svoSetting);
		fzbSVO->initSVO(model);
	}

	void createBuffers() {

		//createStorageBuffer<Vertex_onlyPos>(vertices.size() * sizeof(Vertex_onlyPos), &vertices);
		//createStorageBuffer<uint32_t>(indices.size() * sizeof(uint32_t), &indices);

		if (svoSetting.UseBlock) {
			createStorageBuffer<Vertex_onlyPos>(cubeVertices.size() * sizeof(Vertex_onlyPos), &cubeVertices);
			createStorageBuffer<uint32_t>(cubeIndices.size() * sizeof(uint32_t), &cubeIndices);
		}

		createUniformBuffers(sizeof(UniformBufferObject), false, 1);
		/*
		createUniformBuffers(sizeof(UniformBufferObjectVoxel), true, 1);
		UniformBufferObjectVoxel voxelUniformBufferObject{};
		voxelUniformBufferObject.model = glm::mat4(1.0f);

		float distanceX = model.AABB.rightX - model.AABB.leftX;
		float distanceY = model.AABB.rightY - model.AABB.leftY;
		float distanceZ = model.AABB.rightZ - model.AABB.leftZ;
		//想让顶点通过swizzle变换后得到正确的结果，必须保证投影矩阵是立方体的，这样xyz通过1减后才能是对应的
		//但是其实不需要VP，shader中其实没啥用
		float distance = glm::max(distanceX, glm::max(distanceY, distanceZ));
		float centerX = (model.AABB.rightX + model.AABB.leftX) * 0.5f;
		float centerY = (model.AABB.rightY + model.AABB.leftY) * 0.5f;
		float centerZ = (model.AABB.rightZ + model.AABB.leftZ) * 0.5f;
		//前面
		glm::vec3 viewPoint = glm::vec3(centerX, centerY, model.AABB.rightZ + 0.2f);	//世界坐标右手螺旋，即+z朝后
		glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceX, 0.51f * distanceX, -0.51f * distanceY, 0.51f * distanceY, 0.1f, distanceZ + 0.5f);
		orthoMatrix[1][1] *= -1;
		voxelUniformBufferObject.VP[0] = orthoMatrix * viewMatrix;

		//左边
		viewPoint = glm::vec3(model.AABB.leftX - 0.2f, centerY, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceZ, 0.51f * distanceZ, -0.51f * distanceY, 0.51f * distanceY, 0.1f, distanceX + 0.5f);
		orthoMatrix[1][1] *= -1;
		voxelUniformBufferObject.VP[1] = orthoMatrix * viewMatrix;

		//下面
		viewPoint = glm::vec3(centerX, model.AABB.leftY - 0.2f, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceX, 0.51f * distanceX, -0.51f * distanceZ, 0.51f * distanceZ, 0.1f, distanceY + 0.5f);
		orthoMatrix[1][1] *= -1;
		voxelUniformBufferObject.VP[2] = orthoMatrix * viewMatrix;
		voxelUniformBufferObject.voxelSize_Num = glm::vec4(distance / voxelNum, voxelNum, 0.0f, 0.0f);
		voxelUniformBufferObject.voxelStartPos = glm::vec4(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f, 0.0f);

		memcpy(uniformBuffersMappedsStatic[0], &voxelUniformBufferObject, sizeof(UniformBufferObjectVoxel));
		*/
	}

	void createImages() {

		//voxelImage = {};
		//voxelImage.width = voxelNum;
		//voxelImage.height = voxelNum;
		//voxelImage.depth = voxelNum;
		//voxelImage.type = VK_IMAGE_TYPE_3D;
		//voxelImage.format = VK_FORMAT_R32_UINT;
		//voxelImage.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		//createMyImage(voxelImage);

		//if (UseDepth) {
		//	depthImage = {};
		//	depthImage.width = swapChainExtent.width;
		//	depthImage.height = swapChainExtent.height;
		//	depthImage.format = VK_FORMAT_R32_UINT;
		//	depthImage.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		//	createMyImage(depthImage);
		//}

		//depthBuffer = {};
		//depthBuffer.width = swapChainExtent.width;
		//depthBuffer.height = swapChainExtent.height;
		//depthBuffer.format = findDepthFormat(physicalDevice);
		//depthBuffer.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		//depthBuffer.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
		//createMyImage(depthBuffer);

		//testTexture = {};
		//testTexture.width = swapChainExtent.width;
		//testTexture.height = swapChainExtent.height;
		//testTexture.format = VK_FORMAT_R8G8B8A8_SRGB;
		//testTexture.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		//createMyImage(testTexture);

	}

	void createDescriptor() {

		map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 });
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
		createDescriptorPool(bufferTypeAndNum);

		vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
		vector<VkShaderStageFlagBits> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_ALL };
		uniformDescriptorSetLayout = createDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		uniformDescriptorSet = createDescriptorSet(uniformDescriptorSetLayout);

		std::array<VkWriteDescriptorSet, 2> uniformDescriptorWrites{};
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

		VkDescriptorBufferInfo voxelUniformBufferInfo{};
		voxelUniformBufferInfo.buffer = fzbSVO->fzbBuffer->uniformBuffersStatic[0];
		voxelUniformBufferInfo.offset = 0;
		voxelUniformBufferInfo.range = sizeof(SVOUniform);
		uniformDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		uniformDescriptorWrites[1].dstSet = uniformDescriptorSet;
		uniformDescriptorWrites[1].dstBinding = 1;
		uniformDescriptorWrites[1].dstArrayElement = 0;
		uniformDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uniformDescriptorWrites[1].descriptorCount = 1;
		uniformDescriptorWrites[1].pBufferInfo = &voxelUniformBufferInfo;

		vkUpdateDescriptorSets(logicalDevice, uniformDescriptorWrites.size(), uniformDescriptorWrites.data(), 0, nullptr);
		descriptorSets.push_back({ uniformDescriptorSet });

		descriptorTypes = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE };
		descriptorShaderFlags = { VK_SHADER_STAGE_FRAGMENT_BIT };
		voxelDescriptorSetLayout = createDescriptLayout(1, descriptorTypes, descriptorShaderFlags);
		voxelDescriptorSet = createDescriptorSet(voxelDescriptorSetLayout);

		std::array<VkWriteDescriptorSet, 1> voxelDescriptorWrites{};
		VkDescriptorImageInfo voxelMapInfo{};
		voxelMapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		voxelMapInfo.imageView = fzbSVO->voxelGridMap.imageView;
		voxelMapInfo.sampler = fzbSVO->voxelGridMap.textureSampler;
		voxelDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voxelDescriptorWrites[0].dstSet = voxelDescriptorSet;
		voxelDescriptorWrites[0].dstBinding = 0;
		voxelDescriptorWrites[0].dstArrayElement = 0;
		voxelDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		voxelDescriptorWrites[0].descriptorCount = 1;
		voxelDescriptorWrites[0].pImageInfo = &voxelMapInfo;

		vkUpdateDescriptorSets(logicalDevice, voxelDescriptorWrites.size(), voxelDescriptorWrites.data(), 0, nullptr);
		descriptorSets.push_back({ voxelDescriptorSet });

		/*
		descriptorTypes = { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT };
		descriptorShaderFlags = { VK_SHADER_STAGE_FRAGMENT_BIT };
		presentDescriptorSetLayout = createDescriptLayout(1, descriptorTypes, descriptorShaderFlags);
		presentDescriptorSet = createDescriptorSet(presentDescriptorSetLayout);

		std::array<VkWriteDescriptorSet, 1> presentDescriptorWrites{};
		VkDescriptorImageInfo testTextureInfo{};
		testTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		testTextureInfo.imageView = testTexture.imageView;
		testTextureInfo.sampler = testTexture.textureSampler;
		presentDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		presentDescriptorWrites[0].dstSet = presentDescriptorSet;
		presentDescriptorWrites[0].dstBinding = 0;
		presentDescriptorWrites[0].dstArrayElement = 0;
		presentDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		presentDescriptorWrites[0].descriptorCount = 1;
		presentDescriptorWrites[0].pImageInfo = &testTextureInfo;

		vkUpdateDescriptorSets(logicalDevice, presentDescriptorWrites.size(), presentDescriptorWrites.data(), 0, nullptr);
		descriptorSets.push_back({ presentDescriptorSet });
		*/

	}

	void createRenderPass() {

		VkAttachmentDescription colorAttachmentResolve{};
		colorAttachmentResolve.format = swapChainImageFormat;
		colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentResolveRef{};
		colorAttachmentResolveRef.attachment = 0;
		colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		/*

		VkAttachmentDescription testAttachment{};
		testAttachment.format = VK_FORMAT_R8G8B8A8_SRGB;
		testAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		testAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		testAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		testAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		testAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		testAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		testAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference testAttachmentRef{};
		testAttachmentRef.attachment = 1;
		testAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference testInputAttachmentRef{};
		testInputAttachmentRef.attachment = 1;
		testInputAttachmentRef.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		*/

		std::vector< VkAttachmentDescription> attachments = { colorAttachmentResolve };

		//VkSubpassDescription voxelSubpass{};
		//voxelSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		//voxelSubpass.colorAttachmentCount = 1;
		//voxelSubpass.pColorAttachments = &testAttachmentRef;

		VkSubpassDescription presentSubpass{};
		presentSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		presentSubpass.colorAttachmentCount = 1;
		presentSubpass.pColorAttachments = &colorAttachmentResolveRef;
		//presentSubpass.inputAttachmentCount = 1;
		//presentSubpass.pInputAttachments = &testInputAttachmentRef;

		/*VkAttachmentDescription depthAttachment{};
		VkAttachmentReference depthAttachmentResolveRef{};
		if (UseBlock) {
			depthAttachment.format = findDepthFormat(physicalDevice);
			depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
			depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			depthAttachmentResolveRef.attachment = 2;
			depthAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			attachments.push_back(depthAttachment);

			presentSubpass.pDepthStencilAttachment = &depthAttachmentResolveRef;

		}*/

		std::array< VkSubpassDescription, 1 > subpasses = { presentSubpass };

		//VkSubpassDependency subpassDependency{};
		//subpassDependency.srcSubpass = 0;
		//subpassDependency.dstSubpass = 1;
		//subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		//subpassDependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;	// VK_ACCESS_SHADER_WRITE_BIT;
		//subpassDependency.dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		//subpassDependency.dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
		//subpassDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		std::array< VkSubpassDependency, 1> dependencies = { dependency };

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = attachments.size();
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = subpasses.size();
		renderPassInfo.pSubpasses = subpasses.data();
		renderPassInfo.dependencyCount = dependencies.size();
		renderPassInfo.pDependencies = dependencies.data();

		if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

	}

	void createFramebuffers() {
		vector<vector<VkImageView>> attachmentImageViews;
		attachmentImageViews.resize(swapChainImageViews.size());
		for (int i = 0; i < swapChainImageViews.size(); i++) {
			attachmentImageViews[i].push_back(swapChainImageViews[i]);
			//attachmentImageViews[i].push_back(testTexture.imageView);
			/*if(UseBlock)
				imageViews[i].push_back(depthBuffer.imageView);*/
		}
		createFramebuffer(swapChainImageViews.size(), extent, 1, attachmentImageViews, renderPass);
	}

	void createPipeline() {

		map<VkShaderStageFlagBits, string> shaders;
		if (svoSetting.UseBlock) {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "shaders/present_Block/spv/presentVert_Block.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/present_Block/spv/presentFrag_Block.spv" });
		}
		else {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "shaders/present/spv/presentVert.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/present/spv/presentFrag.spv" });
		}
		vector<VkPipelineShaderStageCreateInfo> shaderStages = createShader(shaders);

		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		VkVertexInputBindingDescription inputBindingDescriptor = Vertex_onlyPos::getBindingDescription();
		auto inputAttributeDescription = Vertex_onlyPos::getAttributeDescriptions();
		if (svoSetting.UseBlock) {
			vertexInputInfo = createVertexInputCreateInfo<Vertex_onlyPos>(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		}
		else {
			vertexInputInfo = createVertexInputCreateInfo<Vertex_onlyPos>(VK_FALSE);
		}
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = createInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		VkPipelineRasterizationStateCreateInfo rasterizer = createRasterizationStateCreateInfo(VK_CULL_MODE_NONE);

		VkPipelineMultisampleStateCreateInfo multisampling = createMultisampleStateCreateInfo();
		VkPipelineColorBlendAttachmentState colorBlendAttachment = createColorBlendAttachmentState();
		vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments = { colorBlendAttachment };
		VkPipelineColorBlendStateCreateInfo colorBlending = createColorBlendStateCreateInfo(colorBlendAttachments);
		VkPipelineDepthStencilStateCreateInfo depthStencil = createDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE);

		VkPipelineViewportStateCreateInfo viewportState = createViewStateCreateInfo();
		VkViewport viewport = {};
		viewport.x = 0;
		viewport.y = 0;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		viewportState.pViewports = &viewport;
		viewportState.pScissors = &scissor;

		std::vector< VkDescriptorSetLayout> presentDescriptorSetLayouts = { uniformDescriptorSetLayout, voxelDescriptorSetLayout };
		presentPipelineLayout = createPipelineLayout(&presentDescriptorSetLayouts);

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = shaderStages.size();
		pipelineInfo.pStages = shaderStages.data();
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		//pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = presentPipelineLayout;
		pipelineInfo.renderPass = renderPass;	//先建立连接，获得索引
		pipelineInfo.subpass = 0;	//对应renderpass的哪个子部分
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//可以直接使用现有pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &presentPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		for (VkPipelineShaderStageCreateInfo shaderModule : shaderStages) {
			vkDestroyShaderModule(logicalDevice, shaderModule.module, nullptr);
		}

	}

	void createSyncObjects() {
		imageAvailableSemaphores = createSemaphore();
		renderFinishedSemaphores = createSemaphore();
		fence = createFence();
	}

	void waitComponentFence() {
		if (currentFrame == 0)
			vkWaitForFences(logicalDevice, 1, &fzbSVO->fence, VK_TRUE, UINT64_MAX);
	}

	void drawFrame() {

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		waitComponentFence();
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
		vkResetCommandBuffer(commandBuffers[0], 0);
		recordCommandBuffer(commandBuffers[0], imageIndex);

		//其实这里可以将两个subpass拆开，因为第一个subpass不需要等待帧缓冲
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
		submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[0];
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &renderFinishedSemaphores;

		//执行完后解开fence
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinishedSemaphores;

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

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		//if (UseDepth) {
		//	VkClearColorValue depth_clearColor = {};
		//	depth_clearColor.uint32[0] = depthImageAccuracy;	//4294967295
		//	depth_clearColor.uint32[1] = 0;
		//	depth_clearColor.uint32[2] = 0;
		//	depth_clearColor.uint32[3] = 0;
		//	clearTexture(commandBuffer, depthImage, depth_clearColor, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
		//}

		//VkClearColorValue voxel_clearColor = {};
		//voxel_clearColor.uint32[0] = 0;
		//voxel_clearColor.uint32[1] = 0;
		//voxel_clearColor.uint32[2] = 0;
		//voxel_clearColor.uint32[3] = 0;
		//clearTexture(commandBuffer, voxelImage, voxel_clearColor, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		//VkViewport viewport{};
		//viewport.x = 0.0f;
		//viewport.y = 0.0f;
		//viewport.width = static_cast<float>(swapChainExtent.width);
		//viewport.height = static_cast<float>(swapChainExtent.height);
		//viewport.minDepth = 0.0f;
		//viewport.maxDepth = 1.0f;
		//vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		//VkRect2D scissor{};
		//scissor.offset = { 0, 0 };
		//scissor.extent = swapChainExtent;
		//vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = framebuffers[imageIndex][0];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		std::array<VkClearValue, 1> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		//clearValues[2].depthStencil = { 0.0f, 0 };
		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		if (svoSetting.UseBlock) {
			VkBuffer cube_vertexBuffers[] = { storageBuffers[0] };
			VkDeviceSize cube_offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, cube_vertexBuffers, cube_offsets);
			vkCmdBindIndexBuffer(commandBuffer, storageBuffers[1], 0, VK_INDEX_TYPE_UINT32);
		}

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipelineLayout, 0, 1, &descriptorSets[0][0], 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, presentPipelineLayout, 1, 1, &descriptorSets[1][0], 0, nullptr);
		if (svoSetting.UseBlock) {
			vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(this->cubeIndices.size()), voxelNum * voxelNum * voxelNum, 0, 0, 0);
		}
		else {
			vkCmdDraw(commandBuffer, 3, 1, 0, 0);
		}

		/*
		VkBuffer vertexBuffers[] = { storageBuffers[0] };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, storageBuffers[1], 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelPipelineLayout, 0, 1, &descriptorSets[0][0], 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelPipelineLayout, 1, 1, &descriptorSets[1][0], 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(this->indices.size()), 1, 0, 0, 0);

		vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);
		*/

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
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

		fzbSVO->cleanSVO();

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
		vkDestroyDescriptorSetLayout(logicalDevice, voxelDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, presentDescriptorSetLayout, nullptr);

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