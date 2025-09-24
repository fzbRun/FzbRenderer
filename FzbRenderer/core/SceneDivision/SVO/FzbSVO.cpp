#include "./FzbSVO.h"
#include "../../common/FzbRenderer.h"
#include <unordered_map>
#include "../../common/FzbMesh/FzbCreateSimpleMesh.h"

std::map<std::string, std::string> shaderPaths = {
	{"unuseSwizzle", "/core/SceneDivision/SVO/shaders/unuseSwizzle"},
	{"present", "/core/SceneDivision/SVO/shaders/present"},
	{"present_SVO", "/core/SceneDivision/SVO/shaders/present_SVO"},
	{"present_Block", "/core/SceneDivision/SVO/shaders/present_VGM"},
};

FzbSVO_Debug::FzbSVO_Debug() {};
FzbSVO_Debug::FzbSVO_Debug(pugi::xml_node& SVONode) {
	if (std::string(SVONode.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
	else return;

	this->componentInfo.name = FZB_FEATURE_COMPONENT_SVO_DEBUG;
	this->componentInfo.type = FZB_LOOPRENDER_FEATURE_COMPONENT;
	//this->componentInfo.useMainSceneBufferHandle = { false, false, false };

	if (pugi::xml_node SVOSettingNode = SVONode.child("featureComponentSetting")) {
		this->setting.voxelNum = std::stoi(SVOSettingNode.child("voxelNum").attribute("value").value());
		this->setting.useSVO_OnlyVoxelGridMap = std::string(SVOSettingNode.child("useOnlyVoxelGridMap").attribute("value").value()) == "true";
		this->setting.useSwizzle = std::string(SVOSettingNode.child("useSwizzle").attribute("value").value()) == "true";
		this->setting.useConservativeRasterization = std::string(SVOSettingNode.child("useConservativeRasterization").attribute("value").value()) == "true";
		this->setting.useBlock = std::string(SVOSettingNode.child("useBlock").attribute("value").value()) == "true";
		this->setting.useCube = std::string(SVOSettingNode.child("useCube").attribute("value").value()) == "true";
	}
	addMainSceneInfo();
	addExtensions();
}

void FzbSVO_Debug::init() {
	FzbFeatureComponent_LoopRender::init();
	createVoxelGridMap();
	if (!setting.useSVO_OnlyVoxelGridMap) {
		this->svoCuda = std::make_unique<SVOCuda>();
		createSVOCuda();
		createSVODescriptor();
	}
	presentPrepare();
}

FzbSemaphore FzbSVO_Debug::render(uint32_t imageIndex, FzbSemaphore startSemaphore, VkFence fence) {
	VkCommandBuffer commandBuffer = commandBuffers[1];
	vkResetCommandBuffer(commandBuffer, 0);
	fzbBeginCommandBuffer(commandBuffer);

	renderRenderPass.render(commandBuffer, imageIndex);

	std::vector<VkSemaphore> waitSemaphores = { startSemaphore.semaphore };
	std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
	fzbSubmitCommandBuffer(commandBuffer, { startSemaphore.semaphore }, waitStages, { renderFinishedSemaphore.semaphore }, fence);

	return renderFinishedSemaphore;
}

void FzbSVO_Debug::clean() {
	FzbFeatureComponent_LoopRender::clean();
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;

	if (!setting.useSVO_OnlyVoxelGridMap) {
		svoCuda->clean();
	}
	voxelGridMap.clean();
	depthMap.clean();

	voxelGridMapRenderPass.clean();

	//vgmShader.clean();
	//vgmMaterial.clean();
	//presentShader.clean();
	//presentMaterial.clean();
	//presentSVOMaterial.clean();
	presentSourceManager.clean();

	svoUniformBuffer.clean();
	nodePool.clean();
	voxelValueBuffer.clean();

	if (voxelGridMapDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, voxelGridMapDescriptorSetLayout, nullptr);
	if (svoDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, svoDescriptorSetLayout, nullptr);

	vgmSemaphore.clean();
	svoCudaSemaphore.clean();
}

void FzbSVO_Debug::addMainSceneInfo() {
	FzbRenderer::globalData.mainScene.vertexFormat_allMesh.mergeUpward(FzbVertexFormat(true));
	FzbRenderer::globalData.mainScene.useVertexBuffer_prepocess = true;
	FzbRenderer::globalData.mainScene.vertexFormat_allMesh_prepocess.mergeUpward(FzbVertexFormat(true));
}
void FzbSVO_Debug::addExtensions() {
	if (!setting.useSVO_OnlyVoxelGridMap) {
		FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
		FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
	}

	if (setting.useConservativeRasterization) {
		FzbRenderer::globalData.deviceExtensions.push_back(VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME);
	}
	if (setting.useSwizzle) {
		FzbRenderer::globalData.deviceExtensions.push_back(VK_NV_VIEWPORT_ARRAY2_EXTENSION_NAME);
		FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_MULTIVIEW_EXTENSION_NAME);
		FzbRenderer::globalData.deviceExtensions.push_back(VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME);
		FzbRenderer::globalData.deviceExtensions.push_back(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME);
	}
	if (!setting.useSVO_OnlyVoxelGridMap) {
		FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
		FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
		FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
		FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
	}

	if (setting.useSwizzle)
		FzbRenderer::globalData.deviceFeatures.multiViewport = VK_TRUE;
	if (!setting.useSVO_OnlyVoxelGridMap) {
		FzbRenderer::globalData.deviceFeatures.fillModeNonSolid = VK_TRUE;
		FzbRenderer::globalData.deviceFeatures.wideLines = VK_TRUE;
	}
}

void FzbSVO_Debug::createImages() {
	VkExtent2D swapChainExtent = FzbRenderer::globalData.swapChainExtent;

	depthMap = FzbImage();
	depthMap.width = swapChainExtent.width;
	depthMap.height = swapChainExtent.height;
	depthMap.type = VK_IMAGE_TYPE_2D;
	depthMap.viewType = VK_IMAGE_VIEW_TYPE_2D;
	depthMap.format = fzbFindDepthFormat();
	depthMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	depthMap.aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
	depthMap.initImage();

	frameBufferImages.push_back(&depthMap);
}

void FzbSVO_Debug::createVoxelGridMap() {
	createVoxelGridMapBuffer();
	initVoxelGridMap();
	createDescriptorPool();
	createVoxelGridMapDescriptor();
	createVoxelGridMapSyncObjects();
	createVGMRenderPass();

	VkCommandBuffer commandBuffer = commandBuffers[0];
	vkResetCommandBuffer(commandBuffer, 0);
	fzbBeginCommandBuffer(commandBuffer);
	
	VkClearColorValue voxel_clearColor = {};
	voxel_clearColor.uint32[0] = 0;
	voxel_clearColor.uint32[1] = 0;
	voxel_clearColor.uint32[2] = 0;
	voxel_clearColor.uint32[3] = 0;
	voxelGridMap.fzbClearTexture(commandBuffer, voxel_clearColor, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	
	voxelGridMapRenderPass.render(commandBuffer, 0);
	
	std::vector<VkSemaphore> waitSemaphores = {};
	std::vector<VkPipelineStageFlags> waitStages = {};
	fzbSubmitCommandBuffer(commandBuffer, {}, waitStages, { vgmSemaphore.semaphore });
}
void FzbSVO_Debug::createVoxelGridMapBuffer() {
	fzbCreateCommandBuffers(2);

	svoUniformBuffer = fzbCreateUniformBuffer(sizeof(SVOUniform));
	FzbAABBBox mianSceneAABB = mainScene->getAABB();
	mianSceneAABB.createDistanceAndCenter(setting.useCube, 1.2);
	float distanceX = mianSceneAABB.distanceX;
	float distanceY = mianSceneAABB.distanceY;
	float distanceZ = mianSceneAABB.distanceZ;
	//想让顶点通过swizzle变换后得到正确的结果，必须保证投影矩阵是立方体的，这样xyz通过1减后才能是对应的
	float distance = mianSceneAABB.distance;
	float centerX = mianSceneAABB.centerX;
	float centerY = mianSceneAABB.centerY;
	float centerZ = mianSceneAABB.centerZ;
	float newLeftX = centerX - distance * 0.5f;
	float newLeftY = centerY - distance * 0.5f;
	float newLeftZ = centerZ - distance * 0.5f;
	float newRightX = centerX + distance * 0.5f;
	float newRightY = centerY + distance * 0.5f;
	float newRightZ = centerZ + distance * 0.5f;
	//前面
	glm::vec3 viewPoint = glm::vec3(centerX, centerY, newRightZ + 0.1f);	//世界坐标右手螺旋，即+z朝后
	glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
	orthoMatrix[1][1] *= -1;
	svoUniform.VP[0] = orthoMatrix * viewMatrix;
	//左边
	viewPoint = glm::vec3(newLeftX - 0.1f, centerY, centerZ);
	viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
	orthoMatrix[1][1] *= -1;
	svoUniform.VP[1] = orthoMatrix * viewMatrix;
	//下面
	viewPoint = glm::vec3(centerX, newLeftY - 0.1f, centerZ);
	viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
	orthoMatrix[1][1] *= -1;
	svoUniform.VP[2] = orthoMatrix * viewMatrix;

	if (setting.useCube) {
		svoUniform.voxelSize_Num = glm::vec4(distance / setting.voxelNum, distance / setting.voxelNum, distance / setting.voxelNum, setting.voxelNum);
		svoUniform.voxelStartPos = glm::vec4(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f, 0.0f);
	}
	else {
		svoUniform.voxelSize_Num = glm::vec4(distanceX / setting.voxelNum, distanceY / setting.voxelNum, distanceZ / setting.voxelNum, setting.voxelNum);
		svoUniform.voxelStartPos = glm::vec4(centerX - distanceX * 0.5f, centerY - distanceY * 0.5f, centerZ - distanceZ * 0.5f, 0.0f);
	}
	memcpy(svoUniformBuffer.mapped, &svoUniform, sizeof(SVOUniform));
}
void FzbSVO_Debug::initVoxelGridMap() {
	voxelGridMap = {};
	voxelGridMap.width = setting.voxelNum;
	voxelGridMap.height = setting.voxelNum;
	voxelGridMap.depth = setting.voxelNum;
	voxelGridMap.type = VK_IMAGE_TYPE_3D;
	voxelGridMap.viewType = VK_IMAGE_VIEW_TYPE_3D;
	voxelGridMap.format = VK_FORMAT_R32_UINT;
	voxelGridMap.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
	voxelGridMap.UseExternal = !setting.useSVO_OnlyVoxelGridMap;
	voxelGridMap.initImage();
}
void FzbSVO_Debug::createDescriptorPool() {
	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
	if (!setting.useSVO_OnlyVoxelGridMap) {
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 });
	}
	this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);
}
void FzbSVO_Debug::createVoxelGridMapDescriptor() {

	std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE };
	std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_FRAGMENT_BIT };
	voxelGridMapDescriptorSetLayout = fzbCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
	voxelGridMapDescriptorSet = fzbCreateDescriptorSet(descriptorPool, voxelGridMapDescriptorSetLayout);

	std::array<VkWriteDescriptorSet, 2> voxelGridMapDescriptorWrites{};
	VkDescriptorBufferInfo voxelGridMapUniformBufferInfo{};
	voxelGridMapUniformBufferInfo.buffer = svoUniformBuffer.buffer;
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

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, voxelGridMapDescriptorWrites.size(), voxelGridMapDescriptorWrites.data(), 0, nullptr);
	//fzbDescriptor->descriptorSets.push_back({ voxelGridMapDescriptorSet });

}
void FzbSVO_Debug::createVoxelGridMapSyncObjects() {	//这里应该返回一个信号量，然后阻塞主线程，知道渲染完成，才能唤醒
	if (setting.useSVO_OnlyVoxelGridMap) {
		vgmSemaphore = FzbSemaphore(false);	//当vgm创建完成后唤醒
	}
	else {
		vgmSemaphore = FzbSemaphore(true);		//当vgm创建完成后唤醒
		svoCudaSemaphore = FzbSemaphore(true);		//当svo创建完成后唤醒
	}
}
void FzbSVO_Debug::createVGMRenderPass() {
	std::string shaderPath;
	FzbShaderExtensionsSetting extensionsSetting;
	if (setting.useConservativeRasterization) extensionsSetting.conservativeRasterization = true;
	std::string materialType = setting.useSwizzle ? "useSwizzle" : "unuseSwizzle";
	if (setting.useSwizzle) {
		materialType = "useSwizzle";
		extensionsSetting.swizzle = true;
		shaderPath = "/core/SceneDivision/SVO/shaders/useSwizzle";
	}
	else {
		materialType = "unuseSwizzle";
		shaderPath = "/core/SceneDivision/SVO/shaders/unuseSwizzle";
	}
	FzbShaderInfo shaderInfo = { shaderPath, extensionsSetting };
	this->vgmSourceManager.addMeshMaterial(FzbRenderer::globalData.mainScene.sceneMeshSet, FzbMaterial("vgm", materialType), false);
	this->vgmSourceManager.addSource({ {materialType, shaderInfo } });

	VkSubpassDescription voxelGridMapSubpass{};
	voxelGridMapSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

	VkSubpassDependency dependency = fzbCreateSubpassDependency();

	FzbRenderPassSetting setting = { false, 1, this->vgmSourceManager.shaderSet[shaderPath].getResolution(), 1, false};
	voxelGridMapRenderPass = FzbRenderPass(setting);
	voxelGridMapRenderPass.createRenderPass(nullptr, { voxelGridMapSubpass }, { dependency });
	voxelGridMapRenderPass.createFramebuffers(false);

	FzbSubPass vgmSubPass = FzbSubPass(voxelGridMapRenderPass.renderPass, 0,
		{ voxelGridMapDescriptorSetLayout }, { voxelGridMapDescriptorSet },
		mainScene->vertexBuffer_prepocess.buffer, mainScene->indexBuffer_prepocess.buffer, this->vgmSourceManager.shaders_vector);
	voxelGridMapRenderPass.subPasses.push_back(vgmSubPass);
}

void FzbSVO_Debug::prepocessClean() {
	this->vgmSourceManager.clean();
};
void FzbSVO_Debug::createSVOCuda() {
	VkPhysicalDevice physicalDevice = FzbRenderer::globalData.physicalDevice;
	svoCuda->createSVOCuda(physicalDevice, voxelGridMap, vgmSemaphore.handle, svoCudaSemaphore.handle);
	nodePool = fzbCreateStorageBuffer(sizeof(FzbSVONode) * (8 * svoCuda->nodeBlockNum + 1), true);
	voxelValueBuffer = fzbCreateStorageBuffer(sizeof(FzbVoxelValue) * svoCuda->voxelNum, true);
	svoCuda->getSVOCuda(physicalDevice, nodePool.handle, voxelValueBuffer.handle);
}

void FzbSVO_Debug::presentPrepare() {
	FzbRenderer::globalData.mainScene.createCameraAndLightDescriptor();
	if (setting.useSVO_OnlyVoxelGridMap) {
		if (!setting.useBlock) createVGMRenderPass_nonBlock();
		else createVGMRenderPass_Block();
	}
	else createSVORenderPass();
}

void FzbSVO_Debug::createVGMRenderPass_nonBlock() {
	this->presentSourceManager.addMeshMaterial(FzbRenderer::globalData.mainScene.sceneMeshSet, FzbMaterial("present", "present"));
	FzbShaderInfo shaderInfo = { "/core/SceneDivision/SVO/shaders/present" };
	this->presentSourceManager.addSource({ {"present", shaderInfo} });

	VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(FzbRenderer::globalData.swapChainImageFormat);
	VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment();
	VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

	std::vector<VkSubpassDescription> subpasses;
	subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));

	VkSubpassDependency dependency = fzbCreateSubpassDependency();

	FzbRenderPassSetting renderPassSetting = { true, 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
	renderRenderPass.setting = renderPassSetting;
	renderRenderPass.createRenderPass(&attachments, subpasses, { dependency });
	renderRenderPass.createFramebuffers(true);

	FzbSubPass presentSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
		{ mainScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout }, { mainScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet },
		mainScene->vertexBuffer.buffer, mainScene->indexBuffer.buffer, this->presentSourceManager.shaders_vector);
	renderRenderPass.addSubPass(presentSubPass);
}
void FzbSVO_Debug::createVGMRenderPass_Block() {
	FzbMesh cubeMesh = FzbMesh();
	cubeMesh.instanceNum = pow(setting.voxelNum, 3);
	glm::mat4 transformMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(svoUniform.voxelStartPos));
	transformMatrix = glm::scale(transformMatrix, glm::vec3(svoUniform.voxelSize_Num));
	fzbCreateCube(cubeMesh, FzbVertexFormat(), transformMatrix);
	//glm::vec3 voxelSize = svoUniform.voxelSize_Num;
	//glm::vec3 voxelStartPos = svoUniform.voxelStartPos;
	//for (int i = 0; i < 24; i += 3) {
	//	cubeMesh.vertices[i] = cubeMesh.vertices[i] * voxelSize.x + voxelStartPos.x;
	//	cubeMesh.vertices[i + 1] = cubeMesh.vertices[i + 1] * voxelSize.y + voxelStartPos.y;
	//	cubeMesh.vertices[i + 2] = cubeMesh.vertices[i + 2] * voxelSize.z + voxelStartPos.z;
	//}
	this->presentSourceManager.componentScene.addMeshToScene(cubeMesh);
	this->presentSourceManager.componentScene.createVertexBuffer(false, false);

	this->presentSourceManager.addMeshMaterial(this->presentSourceManager.componentScene.sceneMeshSet, FzbMaterial("present_Block", "present_Block"));
	FzbShaderInfo shaderInfo = { "/core/SceneDivision/SVO/shaders/present_VGM" };
	this->presentSourceManager.addSource({ {"present_Block", shaderInfo} });

	VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(FzbRenderer::globalData.swapChainImageFormat);
	VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment();
	VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

	std::vector<VkSubpassDescription> subpasses;
	subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));

	VkSubpassDependency dependency = fzbCreateSubpassDependency();

	FzbRenderPassSetting renderPassSetting = { true, 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
	renderRenderPass.setting = renderPassSetting;
	renderRenderPass.createRenderPass(&attachments, subpasses, { dependency });
	renderRenderPass.createFramebuffers(true);

	FzbSubPass presentSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
		{ mainScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout }, { mainScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet },
		this->presentSourceManager.componentScene.vertexBuffer.buffer, this->presentSourceManager.componentScene.indexBuffer.buffer, this->presentSourceManager.shaders_vector);
	renderRenderPass.addSubPass(presentSubPass);
}

void FzbSVO_Debug::createSVODescriptor() {
	std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
	std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_ALL };
	svoDescriptorSetLayout = fzbCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
	svoDescriptorSet = fzbCreateDescriptorSet(descriptorPool, svoDescriptorSetLayout);

	std::array<VkWriteDescriptorSet, 2> svoDescriptorWrites{};
	VkDescriptorBufferInfo nodePoolBufferInfo{};
	nodePoolBufferInfo.buffer = nodePool.buffer;
	nodePoolBufferInfo.offset = 0;
	nodePoolBufferInfo.range = sizeof(FzbSVONode) * (svoCuda->nodeBlockNum * 8 + 1);
	svoDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	svoDescriptorWrites[0].dstSet = svoDescriptorSet;
	svoDescriptorWrites[0].dstBinding = 0;
	svoDescriptorWrites[0].dstArrayElement = 0;
	svoDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	svoDescriptorWrites[0].descriptorCount = 1;
	svoDescriptorWrites[0].pBufferInfo = &nodePoolBufferInfo;

	VkDescriptorBufferInfo voxelValueBufferInfo{};
	voxelValueBufferInfo.buffer = voxelValueBuffer.buffer;
	voxelValueBufferInfo.offset = 0;
	voxelValueBufferInfo.range = sizeof(FzbVoxelValue) * svoCuda->voxelNum;
	svoDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	svoDescriptorWrites[1].dstSet = svoDescriptorSet;
	svoDescriptorWrites[1].dstBinding = 1;
	svoDescriptorWrites[1].dstArrayElement = 0;
	svoDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	svoDescriptorWrites[1].descriptorCount = 1;
	svoDescriptorWrites[1].pBufferInfo = &voxelValueBufferInfo;

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, svoDescriptorWrites.size(), svoDescriptorWrites.data(), 0, nullptr);
}
void FzbSVO_Debug::createSVORenderPass() {
	this->presentSourceManager.addMeshMaterial(FzbRenderer::globalData.mainScene.sceneMeshSet, FzbMaterial("present", "present"));
	FzbShaderInfo presentShaderInfo = { "/core/SceneDivision/SVO/shaders/present" };

	FzbMesh cubeMesh = FzbMesh();
	cubeMesh.instanceNum = svoCuda->nodeBlockNum * 8 + 1;
	fzbCreateCubeWireframe(cubeMesh);
	this->presentSourceManager.componentScene.addMeshToScene(cubeMesh);
	this->presentSourceManager.componentScene.createVertexBuffer(true, false);
	this->presentSourceManager.addMeshMaterial(this->presentSourceManager.componentScene.sceneMeshSet, FzbMaterial("present_SVO", "present_SVO"));
	FzbShaderInfo svoShaderInfo = { "/core/SceneDivision/SVO/shaders/present_SVO" };

	this->presentSourceManager.addSource({ {"present", presentShaderInfo}, { "present_SVO", svoShaderInfo } });

	VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(FzbRenderer::globalData.swapChainImageFormat);
	VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment();
	VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

	std::vector<VkSubpassDescription> subpasses;
	subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));
	subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));

	VkSubpassDependency dependency = fzbCreateSubpassDependency(0, 1,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

	FzbRenderPassSetting renderPassSetting = { true, 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
	renderRenderPass.setting = renderPassSetting;
	renderRenderPass.createRenderPass(&attachments, subpasses, { dependency });
	renderRenderPass.createFramebuffers(true);

	FzbSubPass sceneSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
		{ mainScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout },
		{ mainScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet },
		mainScene->vertexBuffer.buffer, mainScene->indexBuffer.buffer, { &this->presentSourceManager.shaderSet[presentShaderInfo.shaderPath]});
	renderRenderPass.addSubPass(sceneSubPass);

	FzbSubPass CubeWireframeSubPass = FzbSubPass(renderRenderPass.renderPass, 1,
		{ mainScene->cameraAndLightsDescriptorSetLayout, voxelGridMapDescriptorSetLayout, svoDescriptorSetLayout },
		{ mainScene->cameraAndLightsDescriptorSet, voxelGridMapDescriptorSet, svoDescriptorSet },
		this->presentSourceManager.componentScene.vertexBuffer.buffer, this->presentSourceManager.componentScene.indexBuffer.buffer, { &this->presentSourceManager.shaderSet[svoShaderInfo.shaderPath] });
	renderRenderPass.addSubPass(CubeWireframeSubPass);
}