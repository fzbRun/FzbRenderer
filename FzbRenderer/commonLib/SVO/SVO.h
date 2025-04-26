#pragma once

#include "../StructSet.h"
#include "../FzbSwapchain.h"
#include "../FzbImage.h"
#include "../FzbDescriptor.h"
#include "../FzbPipeline.h"
#include "../FzbSync.h"
#include "../Camera.h"

#include "./CUDA/createSVO.cuh"

#ifndef SVO_H	//Sparse voxel octree
#define SVO_H

struct FzbSVOSetting {
	bool UseSVO = true;
	bool UseSVO_OnlyVoxelGridMap = false;
	bool UseSwizzle = false;
	bool UseBlock = false;
	bool UseConservativeRasterization = false;
	int voxelNum = 64;
};

struct SVOUniform {
	glm::mat4 modelMatrix;
	glm::mat4 VP[3];
	glm::vec4 voxelSize_Num;
	glm::vec4 voxelStartPos;
};

class FzbSVO {

public:

	//依赖
	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkQueue graphicsQueue;
	VkExtent2D swapChainExtent;
	std::unique_ptr<FzbImage> fzbImage;
	std::unique_ptr<FzbBuffer> fzbBuffer;
	std::unique_ptr<FzbDescriptor> fzbDescriptor;
	std::unique_ptr<FzbPipeline> fzbPipeline;
	std::unique_ptr<FzbSync> fzbSync;

	FzbSVOSetting svoSetting;
	MyImage voxelGridMap;
	VkDescriptorSetLayout voxelGridMapDescriptorSetLayout;
	VkDescriptorSet voxelGridMapDescriptorSet;
	std::vector<Vertex_onlyPos> vertices;
	std::vector<uint32_t> indices;
	VkRenderPass voxelGridMapRenderPass;
	VkPipeline voxelGridMapPipeline;
	VkPipelineLayout voxelGridMapPipelineLayout;

	FzbSVOCudaVariable* fzbSVOCudaVar;

	FzbSVO(std::unique_ptr<FzbDevice>& fzbDevice, std::unique_ptr<FzbSwapchain>& fzbSwapchain, VkCommandPool commandPool, MyModel& model, FzbSVOSetting svoSetting) {
		
		this->physicalDevice = fzbDevice->physicalDevice;
		this->logicalDevice = fzbDevice->logicalDevice;
		this->graphicsQueue = fzbDevice->graphicsQueue;
		this->swapChainExtent = fzbSwapchain->swapChainExtent;
		this->fzbImage = std::make_unique<FzbImage>(fzbDevice);
		this->fzbBuffer = std::make_unique<FzbBuffer>(fzbDevice, commandPool);
		this->fzbDescriptor = std::make_unique<FzbDescriptor>(fzbDevice);
		this->fzbPipeline = std::make_unique<FzbPipeline>(fzbDevice);
		this->fzbSync = std::make_unique<FzbSync>(fzbDevice);
		this->svoSetting = svoSetting;

		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			this->fzbImage->UseExternal = true;
		}

	}

	static void getInstanceExtensions(FzbSVOSetting svoSetting, std::vector<const char*>& instanceExtensions) {
		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
			instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
		}
	}

	static void getDeviceExtensions(FzbSVOSetting svoSetting, std::vector<const char*>& deviceExtensions) {

		if (svoSetting.UseConservativeRasterization) {
			deviceExtensions.push_back(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
		}
		if (svoSetting.UseSwizzle) {
			deviceExtensions.push_back(VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
			deviceExtensions.push_back(VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME);
			deviceExtensions.push_back(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME);
		}
		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
			deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
		}

	}

	static void getDeviceFeatures(FzbSVOSetting svoSetting, VkPhysicalDeviceFeatures& deviceFeatures) {
		if (svoSetting.UseSwizzle)
			deviceFeatures.multiViewport = VK_TRUE;
	}

	void initSVO(MyModel& model) {

		createBuffer(model);
		initVoxelGridMap();
		createVoxelGridMapDescriptor();
		createVoxelGridMapRenderPass();
		createVoxelGridMapFramebuffer();
		createVoxelGridMapPipeline();
		createVoxelGridMapSyncObjects();
		createVoxelGridMap();
		if (!svoSetting.UseSVO_OnlyVoxelGridMap)
			createSVOCuda(physicalDevice, voxelGridMap, fzbSync->fzbSemaphores[0].handle, fzbSync->fzbSemaphores[1].handle, fzbSVOCudaVar);
	}

	void cleanSVO() {

		if (!svoSetting.UseSVO_OnlyVoxelGridMap) {
			cleanSVOCuda(fzbSVOCudaVar);
			delete fzbSVOCudaVar;
		}

		fzbImage->cleanImage(voxelGridMap);

		for (size_t i = 0; i < fzbBuffer->framebuffers.size(); i++) {
			for (int j = 0; j < fzbBuffer->framebuffers[i].size(); j++) {
				vkDestroyFramebuffer(logicalDevice, fzbBuffer->framebuffers[i][j], nullptr);
			}
		}

		//清理管线
		vkDestroyPipeline(logicalDevice, voxelGridMapPipeline, nullptr);
		vkDestroyPipelineLayout(logicalDevice, voxelGridMapPipelineLayout, nullptr);
		//清理渲染Pass
		vkDestroyRenderPass(logicalDevice, voxelGridMapRenderPass, nullptr);

		vkDestroyDescriptorPool(logicalDevice, fzbDescriptor->descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, voxelGridMapDescriptorSetLayout, nullptr);

		fzbSync->cleanFzbSync();

		fzbBuffer->cleanupBuffers();

	}

private:

	template<typename T>
	void optimizeModel(MyModel& myModel, std::vector<T>& vertices, std::vector<uint32_t>& indices) {
		uint32_t indexOffset = 0;
		for (uint32_t meshIndex = 0; meshIndex < myModel.meshs.size(); meshIndex++) {

			Mesh mesh = myModel.meshs[meshIndex];
			//this->materials.push_back(my_model->meshs[i].material);
			vertices.insert(vertices.end(), mesh.vertices.begin(), mesh.vertices.end());

			//因为assimp是按一个mesh一个mesh的存，所以每个indices都是相对一个mesh的，当我们将每个mesh的顶点存到一起时，indices就会出错，我们需要增加索引
			for (uint32_t index = 0; index < mesh.indices.size(); index++) {
				mesh.indices[index] += indexOffset;
			}
			//meshIndexInIndices.push_back(this->indices.size());
			indexOffset += mesh.vertices.size();
			indices.insert(indices.end(), mesh.indices.begin(), mesh.indices.end());
		}

		std::unordered_map<T, uint32_t> uniqueVerticesMap{};
		std::vector<T> uniqueVertices;
		std::vector<uint32_t> uniqueIndices;
		for (uint32_t j = 0; j < indices.size(); j++) {
			T vertex = std::is_same_v<T, Vertex> ? vertices[indices[j]] : T(vertices[indices[j]]);
			if (uniqueVerticesMap.count(vertex) == 0) {
				uniqueVerticesMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
				uniqueVertices.push_back(vertex);
			}
			uniqueIndices.push_back(uniqueVerticesMap[vertex]);
		}
		vertices = uniqueVertices;
		indices = uniqueIndices;

	}

	void createBuffer(MyModel& model) {

		fzbBuffer->createCommandBuffers(1);

		optimizeModel<Vertex_onlyPos>(model, vertices, indices);

		fzbBuffer->createStorageBuffer<Vertex_onlyPos>(vertices.size() * sizeof(Vertex_onlyPos), &vertices);
		fzbBuffer->createStorageBuffer<uint32_t>(indices.size() * sizeof(uint32_t), &indices);

		fzbBuffer->createUniformBuffers(sizeof(SVOUniform), true, 1);
		SVOUniform SVOUniformBufferObject{};
		SVOUniformBufferObject.modelMatrix = glm::mat4(1.0f);

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
		SVOUniformBufferObject.VP[0] = orthoMatrix * viewMatrix;

		//左边
		viewPoint = glm::vec3(model.AABB.leftX - 0.2f, centerY, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceZ, 0.51f * distanceZ, -0.51f * distanceY, 0.51f * distanceY, 0.1f, distanceX + 0.5f);
		orthoMatrix[1][1] *= -1;
		SVOUniformBufferObject.VP[1] = orthoMatrix * viewMatrix;

		//下面
		viewPoint = glm::vec3(centerX, model.AABB.leftY - 0.2f, centerZ);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.51f * distanceX, 0.51f * distanceX, -0.51f * distanceZ, 0.51f * distanceZ, 0.1f, distanceY + 0.5f);
		orthoMatrix[1][1] *= -1;
		SVOUniformBufferObject.VP[2] = orthoMatrix * viewMatrix;
		SVOUniformBufferObject.voxelSize_Num = glm::vec4(distance / svoSetting.voxelNum, svoSetting.voxelNum, 0.0f, 0.0f);
		SVOUniformBufferObject.voxelStartPos = glm::vec4(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f, 0.0f);

		memcpy(fzbBuffer->uniformBuffersMappedsStatic[0], &SVOUniformBufferObject, sizeof(SVOUniform));

	}

	void initVoxelGridMap(){
		voxelGridMap = {};
		voxelGridMap.width = svoSetting.voxelNum;
		voxelGridMap.height = svoSetting.voxelNum;
		voxelGridMap.depth = svoSetting.voxelNum;
		voxelGridMap.type = VK_IMAGE_TYPE_3D;
		voxelGridMap.viewType = VK_IMAGE_VIEW_TYPE_3D;
		voxelGridMap.format = VK_FORMAT_R32_UINT;
		voxelGridMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
		fzbImage->createMyImage(voxelGridMap, fzbBuffer);

	}

	void createVoxelGridMapDescriptor() {

		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
		fzbDescriptor->createDescriptorPool(bufferTypeAndNum);

		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE };
		std::vector<VkShaderStageFlagBits> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_FRAGMENT_BIT };
		voxelGridMapDescriptorSetLayout = fzbDescriptor->createDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
		voxelGridMapDescriptorSet = fzbDescriptor->createDescriptorSet(voxelGridMapDescriptorSetLayout);

		std::array<VkWriteDescriptorSet, 2> voxelGridMapDescriptorWrites{};
		VkDescriptorBufferInfo voxelGridMapUniformBufferInfo{};
		voxelGridMapUniformBufferInfo.buffer = fzbBuffer->uniformBuffersStatic[0];
		voxelGridMapUniformBufferInfo.offset = 0;
		voxelGridMapUniformBufferInfo.range = sizeof(SVOUniform);
		voxelGridMapDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voxelGridMapDescriptorWrites[0].dstSet = voxelGridMapDescriptorSet;
		voxelGridMapDescriptorWrites[0].dstBinding = 0;
		voxelGridMapDescriptorWrites[0].dstArrayElement = 0;
		voxelGridMapDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		voxelGridMapDescriptorWrites[0].descriptorCount = 1;
		voxelGridMapDescriptorWrites[0].pBufferInfo = &voxelGridMapUniformBufferInfo;

		VkDescriptorImageInfo voxelGridMapInfo{};
		voxelGridMapInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		voxelGridMapInfo.imageView = voxelGridMap.imageView;
		voxelGridMapInfo.sampler = voxelGridMap.textureSampler;
		voxelGridMapDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voxelGridMapDescriptorWrites[1].dstSet = voxelGridMapDescriptorSet;
		voxelGridMapDescriptorWrites[1].dstBinding = 1;
		voxelGridMapDescriptorWrites[1].dstArrayElement = 0;
		voxelGridMapDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		voxelGridMapDescriptorWrites[1].descriptorCount = 1;
		voxelGridMapDescriptorWrites[1].pImageInfo = &voxelGridMapInfo;

		vkUpdateDescriptorSets(logicalDevice, voxelGridMapDescriptorWrites.size(), voxelGridMapDescriptorWrites.data(), 0, nullptr);
		//fzbDescriptor->descriptorSets.push_back({ voxelGridMapDescriptorSet });

	}

	void createVoxelGridMapRenderPass() {

		VkSubpassDescription voxelGridMapSubpass{};
		voxelGridMapSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 0;
		renderPassInfo.pAttachments = nullptr;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &voxelGridMapSubpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &voxelGridMapRenderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}

	}

	void createVoxelGridMapFramebuffer() {
		std::vector<std::vector<VkImageView>> attachmentImageViews;
		VkExtent2D extent = {
			static_cast<uint32_t>(this->swapChainExtent.width),
			static_cast<uint32_t>(swapChainExtent.height)
		};
		fzbBuffer->createFramebuffer(1, extent, 0, attachmentImageViews, voxelGridMapRenderPass);
	}

	void createVoxelGridMapPipeline() {
		std::map<VkShaderStageFlagBits, std::string> shaders;
		if (svoSetting.UseSwizzle) {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "commonLib/SVO/shaders/useSwizzle/spv/voxelVert.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "commonLib/SVO/shaders/useSwizzle/spv/voxelFrag.spv" });
		}
		else {
			shaders.insert({ VK_SHADER_STAGE_VERTEX_BIT, "commonLib/SVO/shaders/unuseSwizzle/spv/voxelVert.spv" });
			shaders.insert({ VK_SHADER_STAGE_GEOMETRY_BIT, "commonLib/SVO/shaders/unuseSwizzle/spv/voxelGemo.spv" });
			shaders.insert({ VK_SHADER_STAGE_FRAGMENT_BIT, "commonLib/SVO/shaders/unuseSwizzle/spv/voxelFrag.spv" });
		}
		std::vector<VkPipelineShaderStageCreateInfo> shaderStages = fzbPipeline->createShader(shaders);

		VkVertexInputBindingDescription inputBindingDescriptor = Vertex_onlyPos::getBindingDescription();
		auto inputAttributeDescription = Vertex_onlyPos::getAttributeDescriptions();
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = fzbPipeline->createVertexInputCreateInfo<Vertex_onlyPos>(VK_TRUE, &inputBindingDescriptor, &inputAttributeDescription);
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = fzbPipeline->createInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeState{};
		if (svoSetting.UseConservativeRasterization) {
			conservativeState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
			conservativeState.pNext = NULL;
			conservativeState.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
			conservativeState.extraPrimitiveOverestimationSize = 0.5f; // 根据需要设置
			rasterizer = fzbPipeline->createRasterizationStateCreateInfo(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE, &conservativeState);
		}
		else {
			rasterizer = fzbPipeline->createRasterizationStateCreateInfo(VK_CULL_MODE_NONE);
		}

		VkPipelineMultisampleStateCreateInfo multisampling = fzbPipeline->createMultisampleStateCreateInfo();
		VkPipelineColorBlendAttachmentState colorBlendAttachment = fzbPipeline->createColorBlendAttachmentState();
		std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments = { colorBlendAttachment };
		VkPipelineColorBlendStateCreateInfo colorBlending = fzbPipeline->createColorBlendStateCreateInfo(colorBlendAttachments);
		VkPipelineDepthStencilStateCreateInfo depthStencil = fzbPipeline->createDepthStencilStateCreateInfo(VK_FALSE, VK_FALSE);

		VkPipelineViewportStateCreateInfo viewportState = fzbPipeline->createViewStateCreateInfo();
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

		if (svoSetting.UseSwizzle) {

			//std::array< VkViewport, 4> viewports = {};
			//std::array< VkRect2D, 4> scissors = {};
			//for (int y = 0; y < 2; y++) {
			//	for (int x = 0; x < 2; x++) {
			//		viewports[x + y * 2].x = x * swapChainExtent.width / 2;
			//		viewports[x + y * 2].y = y * swapChainExtent.height / 2;
			//		viewports[x + y * 2].width = static_cast<float>(swapChainExtent.width / 2);
			//		viewports[x + y * 2].height = static_cast<float>(swapChainExtent.height / 2);
			//		viewports[x + y * 2].minDepth = 0.0f;
			//		viewports[x + y * 2].maxDepth = 1.0f;
			//		scissors[x + y * 2].offset = { x * (int)swapChainExtent.width / 2, y * (int)swapChainExtent.height / 2 };
			//		scissors[x + y * 2].extent = { swapChainExtent.width / 2, swapChainExtent.height / 2 };;
			//	}
			//}
			//std::array< VkViewportSwizzleNV, 4 > swizzles = {};
			//for (int i = 0; i < 4; i++) {
			//	swizzles[i] = {
			//		i % 2 == 0 ? VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV : VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_X_NV, /* x */
			//		i / 2 == 0 ? VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV : VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV, /* y */
			//		VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV, /* z */
			//		VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV /* w */
			//	};
			//}

			std::array< VkViewport, 3> viewports = {};
			std::array< VkRect2D, 3> scissors = {};
			for (int i = 0; i < viewports.size(); i++) {
				viewports[i].x = 0;
				viewports[i].y = 0;
				viewports[i].width = static_cast<float>(swapChainExtent.width);
				viewports[i].height = static_cast<float>(swapChainExtent.height);
				viewports[i].minDepth = 0.0f;
				viewports[i].maxDepth = 1.0f;

				scissors[i].offset = { 0, 0 };
				scissors[i].extent = swapChainExtent;
			}

			/*
			哇哇哇，这里很有意思，就是由于vulkan的NDC是y轴朝下的，所以我的投影矩阵的y乘了-1，所以才能正确显示。
			但是呢，在这里swizzle时，swizzles[0]和swizzle[1]都是同一个y，所以反转y后才能正确显示，所以没问题
			而对于swizzle[2]来说，反转后的y变成了它的z，这就导致原本是从下向上看的，编程了从上向下看，并且由于NEGATIVE_Z没有反转，因此图像变到了上方
			因此想要正确显示就必须样swizzle[2]的y变为VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV，z变为VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV
			*/
			std::array<VkViewportSwizzleNV, 3> swizzles = {};
			swizzles[0] = {		//前面
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
			};
			swizzles[1] = {		//左面
				VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Z_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
			};
			swizzles[2] = {		//下面
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_NEGATIVE_Y_NV,
				VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
			};

			viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewportState.viewportCount = viewports.size();
			viewportState.pViewports = viewports.data();
			viewportState.scissorCount = scissors.size();
			viewportState.pScissors = scissors.data();

			VkPipelineViewportSwizzleStateCreateInfoNV viewportSwizzleInfo{};
			viewportSwizzleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_SWIZZLE_STATE_CREATE_INFO_NV;
			viewportSwizzleInfo.pViewportSwizzles = swizzles.data();
			viewportSwizzleInfo.viewportCount = swizzles.size();

			viewportState.pNext = &viewportSwizzleInfo;

		}
		////一般渲染管道状态都是固定的，不能渲染循环中修改，但是某些状态可以，如视口，长宽和混合常数
		////同样通过宏来确定可动态修改的状态
		//std::vector<VkDynamicState> dynamicStates = {
		//	VK_DYNAMIC_STATE_VIEWPORT,
		//	VK_DYNAMIC_STATE_SCISSOR
		//};
		//VkPipelineDynamicStateCreateInfo dynamicState{};
		//dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		//dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		//dynamicState.pDynamicStates = dynamicStates.data();

		std::vector< VkDescriptorSetLayout> descriptorSetLayouts = { voxelGridMapDescriptorSetLayout };
		voxelGridMapPipelineLayout = fzbPipeline->createPipelineLayout(&descriptorSetLayouts);

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
		pipelineInfo.layout = voxelGridMapPipelineLayout;
		pipelineInfo.renderPass = voxelGridMapRenderPass;	//先建立连接，获得索引
		pipelineInfo.subpass = 0;	//对应renderpass的哪个子部分
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	//可以直接使用现有pipeline
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &voxelGridMapPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		for (VkPipelineShaderStageCreateInfo shaderModule : shaderStages) {
			vkDestroyShaderModule(logicalDevice, shaderModule.module, nullptr);
		}
	}

	void createVoxelGridMapSyncObjects() {	//这里应该返回一个信号量，然后阻塞主线程，知道渲染完成，才能唤醒
		if (svoSetting.UseSVO_OnlyVoxelGridMap) {
			fzbSync->createSemaphore(false);
		}
		else {
			fzbSync->createSemaphore(true);
			fzbSync->createSemaphore(true);
		}

		fzbSync->createFence();
	}

	void createVoxelGridMap() {

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		vkResetFences(logicalDevice, 1, &fzbSync->fzbFences[0]);
		VkCommandBuffer commandBuffer = fzbBuffer->commandBuffers[0];
		vkResetCommandBuffer(commandBuffer, 0);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkClearColorValue voxel_clearColor = {};
		voxel_clearColor.uint32[0] = 0;
		voxel_clearColor.uint32[1] = 0;
		voxel_clearColor.uint32[2] = 0;
		voxel_clearColor.uint32[3] = 0;
		fzbImage->clearTexture(commandBuffer, voxelGridMap, voxel_clearColor, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = voxelGridMapRenderPass;
		renderPassInfo.framebuffer = fzbBuffer->framebuffers[0][0];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		VkBuffer vertexBuffers[] = { fzbBuffer->storageBuffers[0] };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffer, fzbBuffer->storageBuffers[1], 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelGridMapPipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, voxelGridMapPipelineLayout, 0, 1, &voxelGridMapDescriptorSet, 0, nullptr);
		vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(this->indices.size()), 1, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}

		submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount = 0;
		submitInfo.pWaitSemaphores = nullptr;
		submitInfo.pWaitDstStageMask = nullptr;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = svoSetting.UseSVO_OnlyVoxelGridMap ? &fzbSync->fzbSemaphores[0].semaphore : &fzbSync->fzbSemaphores[1].semaphore;

		//执行完后解开fence
		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

	}

};

#endif
