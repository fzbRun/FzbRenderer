#include "./FzbSVO_PG.h"
#include "../../common/FzbRenderer.h"
#include "../../common/FzbMesh/FzbCreateSimpleMesh.h"
#include "../../RayTracing/CUDA/FzbCollisionDetection.cuh"

FzbSVO_PG::FzbSVO_PG() {};
FzbSVO_PG::FzbSVO_PG(pugi::xml_node& SVO_PGNode) {
	if (std::string(SVO_PGNode.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
	else return;

	this->componentInfo.name = FZB_FEATURE_COMPONENT_SVO_PG;
	this->componentInfo.type = FZB_PREPROCESS_FEATURE_COMPONENT;

	if (pugi::xml_node SVOSettingNode = SVO_PGNode.child("featureComponentSetting")) {
		this->setting.voxelNum = std::stoi(SVOSettingNode.child("voxelNum").attribute("value").value());
		this->setting.useCube = std::string(SVOSettingNode.child("useCube").attribute("value").value()) == "true";
		this->setting.lightInjectSPP = std::stoi(SVOSettingNode.child("lightInjectSPP").attribute("value").value());
		this->setting.thresholds.irradianceRelRatioThreshold = std::stof(SVOSettingNode.child("irradianceRelRatioThreshold").attribute("value").value());
		this->setting.thresholds.entropyThreshold = std::stof(SVOSettingNode.child("entropyThreshold").attribute("value").value());
	}

	addMainSceneInfo();
	addExtensions();
	if (pugi::xml_node childComponents = SVO_PGNode.child("featureComponents")) getChildComponent(childComponents);
	//得到bvh组件
	if (pugi::xml_node childComponentsNode = SVO_PGNode.child("childComponents")) {
		getChildComponent(childComponentsNode);
		this->rayTracingSourceManager.bvh = std::dynamic_pointer_cast<FzbBVH>(childComponents["BVH"]);
		this->rayTracingSourceManager.bvh->setting = FzbBVHSetting{ BVH_MAX_DEPTH, true, true };
	}
}
void FzbSVO_PG::init() {
	FzbFeatureComponent_PreProcess::init();
	createBufferAndDescriptor();
	createSemaphore();

	rayTracingSourceManager.createSource();
	FzbVGBUniformData VGBUniformData = { setting.voxelNum, glm::vec3(uniformBufferObject.voxelSize_Num), uniformBufferObject.voxelStartPos };
	svoCuda_pg = std::make_shared<FzbSVOCuda_PG>(rayTracingSourceManager.sourceManagerCuda, setting, VGBUniformData, VGB, SVOFinishedSemaphore.handle, setting.thresholds);
	createVGB();
	//createSVO_PG();
}
void FzbSVO_PG::clean() {
	FzbFeatureComponent_PreProcess::clean();
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;

	rayTracingSourceManager.clean();
	svoCuda_pg->clean();
	uniformBuffer.clean();
	VGB.clean();

	VGBFinishedSemaphore.clean();
	SVOFinishedSemaphore.clean();

	vgbSourceManager.clean();
	vgbRenderPass.clean();

	if (descriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);

	for (int i = 0; i < this->OctreeNodesBuffers.size(); ++i) this->OctreeNodesBuffers[i].clean();
	for (int i = 0; i < this->SVONodesBuffers.size(); ++i) this->SVONodesBuffers[i].clean();
	SVOWeightsBuffer.clean();
}
void FzbSVO_PG::addMainSceneInfo() {
	FzbRenderer::globalData.mainScene.useVertexBuffer_prepocess = true;
	FzbRenderer::globalData.mainScene.vertexFormat_allMesh_prepocess.mergeUpward(FzbVertexFormat());
}
void FzbSVO_PG::addExtensions() {
	FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
	FzbRenderer::globalData.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
	FzbRenderer::globalData.deviceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);

	//这个扩展要vulkan 1.4版本，但是也同时需要50系列显卡，寄！！！！！！
	//FzbRenderer::globalData.deviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
	//VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT atomicFloat2Features{};
	//atomicFloat2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
	//atomicFloat2Features.shaderBufferFloat32AtomicMinMax = true;
	//atomicFloat2Features.pNext = FzbRenderer::globalData.extensionFeatureList.featureList;
	//FzbRenderer::globalData.extensionFeatureList.featureList = &atomicFloat2Features;
};
void FzbSVO_PG::createUniformBuffer() {
	this->uniformBuffer = fzbCreateUniformBuffer(sizeof(FzbSVOUniformBufferObject));
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
	uniformBufferObject.VP[0] = orthoMatrix * viewMatrix;
	//左边
	viewPoint = glm::vec3(newLeftX - 0.1f, centerY, centerZ);
	viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
	orthoMatrix[1][1] *= -1;
	uniformBufferObject.VP[1] = orthoMatrix * viewMatrix;
	//下面
	viewPoint = glm::vec3(centerX, newLeftY - 0.1f, centerZ);
	viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	orthoMatrix = glm::orthoRH_ZO(-0.5f * distance, 0.5f * distance, -0.5f * distance, 0.5f * distance, 0.1f, distance + 0.1f);
	orthoMatrix[1][1] *= -1;
	uniformBufferObject.VP[2] = orthoMatrix * viewMatrix;

	if (setting.useCube) {
		uniformBufferObject.voxelSize_Num = glm::vec4(distance / setting.voxelNum, distance / setting.voxelNum, distance / setting.voxelNum, setting.voxelNum);
		uniformBufferObject.voxelStartPos = glm::vec4(centerX - distance * 0.5f, centerY - distance * 0.5f, centerZ - distance * 0.5f, 0.0f);
	}
	else {
		uniformBufferObject.voxelSize_Num = glm::vec4(distanceX / setting.voxelNum, distanceY / setting.voxelNum, distanceZ / setting.voxelNum, setting.voxelNum);
		uniformBufferObject.voxelStartPos = glm::vec4(centerX - distanceX * 0.5f, centerY - distanceY * 0.5f, centerZ - distanceZ * 0.5f, 0.0f);
	}
	memcpy(this->uniformBuffer.mapped, &uniformBufferObject, sizeof(FzbSVOUniformBufferObject));
}
void FzbSVO_PG::createStorageBuffer() {
	uint32_t VGBSize = setting.voxelNum * setting.voxelNum * setting.voxelNum;
	uint32_t VGBSize_Byte = VGBSize * sizeof(FzbVoxelData_PG);
	this->VGB = fzbCreateStorageBuffer(VGBSize_Byte, true);
}
void FzbSVO_PG::createDescriptor() {
	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 });
	this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);

	std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
	std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_ALL };
	descriptorSetLayout = fzbCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
	descriptorSet = fzbCreateDescriptorSet(descriptorPool, descriptorSetLayout);

	std::array<VkWriteDescriptorSet, 2> voxelGridBufferDescriptorWrites{};
	VkDescriptorBufferInfo vgbUniformBufferInfo{};
	vgbUniformBufferInfo.buffer = this->uniformBuffer.buffer;
	vgbUniformBufferInfo.offset = 0;
	vgbUniformBufferInfo.range = sizeof(FzbSVOUniformBufferObject);
	voxelGridBufferDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	voxelGridBufferDescriptorWrites[0].dstSet = descriptorSet;
	voxelGridBufferDescriptorWrites[0].dstBinding = 0;
	voxelGridBufferDescriptorWrites[0].dstArrayElement = 0;
	voxelGridBufferDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	voxelGridBufferDescriptorWrites[0].descriptorCount = 1;
	voxelGridBufferDescriptorWrites[0].pBufferInfo = &vgbUniformBufferInfo;

	VkDescriptorBufferInfo vgbInfo{};
	vgbInfo.buffer = this->VGB.buffer;
	vgbInfo.offset = 0;
	vgbInfo.range = this->VGB.size;
	voxelGridBufferDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	voxelGridBufferDescriptorWrites[1].dstSet = descriptorSet;
	voxelGridBufferDescriptorWrites[1].dstBinding = 1;
	voxelGridBufferDescriptorWrites[1].dstArrayElement = 0;
	voxelGridBufferDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	voxelGridBufferDescriptorWrites[1].descriptorCount = 1;
	voxelGridBufferDescriptorWrites[1].pBufferInfo = &vgbInfo;

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, voxelGridBufferDescriptorWrites.size(), voxelGridBufferDescriptorWrites.data(), 0, nullptr);
}
void FzbSVO_PG::createBufferAndDescriptor() {
	fzbCreateCommandBuffers(1);
	createUniformBuffer();
	createStorageBuffer();
	createDescriptor();
}
void FzbSVO_PG::createSemaphore() {
	VGBFinishedSemaphore = FzbSemaphore(true);
	SVOFinishedSemaphore = FzbSemaphore(true);
}

void FzbSVO_PG::createVGBRenderPass() {
	std::string materialType = "createVGB";
	std::string shaderPath = "/core/SceneDivision/SVO_PG/shaders/createVGB";
	this->vgbSourceManager.addMeshMaterial(FzbRenderer::globalData.mainScene.sceneMeshSet, FzbMaterial("createVGB", materialType), false);
	this->vgbSourceManager.addSource({ { materialType, { shaderPath } } });

	VkSubpassDescription vvgbSubpass{};
	vvgbSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

	VkSubpassDependency dependency = fzbCreateSubpassDependency();

	FzbRenderPassSetting setting = { false, 1, this->vgbSourceManager.shaderSet[shaderPath].getResolution(), 1, false };
	vgbRenderPass = FzbRenderPass(setting);
	vgbRenderPass.createRenderPass(nullptr, { vvgbSubpass }, { dependency });
	vgbRenderPass.createFramebuffers(false);

	FzbSubPass vgmSubPass = FzbSubPass(vgbRenderPass.renderPass, 0,
		{ descriptorSetLayout }, { descriptorSet },
		mainScene->vertexBuffer_prepocess.buffer, mainScene->indexBuffer_prepocess.buffer, this->vgbSourceManager.shaders_vector);
	vgbRenderPass.subPasses.push_back(vgmSubPass);
}
void FzbSVO_PG::createVGB() {
	svoCuda_pg->initVGB();
	createVGBRenderPass();

	VkCommandBuffer commandBuffer = commandBuffers[0];
	vkResetCommandBuffer(commandBuffer, 0);
	fzbBeginCommandBuffer(commandBuffer);

	vgbRenderPass.render(commandBuffer, 0);

	std::vector<VkSemaphore> waitSemaphores = {};
	std::vector<VkPipelineStageFlags> waitStages = {};
	fzbSubmitCommandBuffer(commandBuffer, {}, waitStages, { VGBFinishedSemaphore.semaphore });
}
void FzbSVO_PG::createSVO_PG() {
	svoCuda_pg->createSVOCuda_PG(VGBFinishedSemaphore.handle);
}

void FzbSVO_PG::createOctreeBuffers(bool isG) {
	this->OctreeNodesBuffers.resize(this->svoCuda_pg->SVONodes_maxDepth - 2);	//不算根节点和叶节点
	for (int i = 0; i < OctreeNodesBuffers.size(); ++i) {
		uint32_t nodeCount = std::pow(8, i + 1);
		uint32_t bufferSize = nodeCount * (isG ? sizeof(FzbSVONodeData_PG_G) : sizeof(FzbSVONodeData_PG_E));
		this->OctreeNodesBuffers[i] = fzbCreateStorageBuffer(bufferSize, true, false);
	}
	this->svoCuda_pg->coypyOctreeDataToBuffer(this->OctreeNodesBuffers, isG);
}
void FzbSVO_PG::createSVOBuffers(bool isG, bool useDeviceAddress) {
	this->SVONodesBuffers.resize(0);	//不算根节点
	for (int i = 0; i < this->svoCuda_pg->SVONodes_maxDepth - 1; ++i) {
		uint32_t nodeCount = (isG ? this->svoCuda_pg->SVOLayerInfos_G_host[i].divisibleNodeCount : this->svoCuda_pg->SVOLayerInfos_E_host[i].divisibleNodeCount) * 8;
		if (nodeCount == 0) break;
		uint32_t bufferSize = nodeCount * (isG ? sizeof(FzbSVONodeData_PG_G) : sizeof(FzbSVONodeData_PG_E));
		this->SVONodesBuffers.push_back(fzbCreateStorageBuffer(bufferSize, true, useDeviceAddress));
	}
	this->svoCuda_pg->copySVODataToBuffer(this->SVONodesBuffers, isG);
}
void FzbSVO_PG::createSVOWeightsBuffer() {
	std::vector<FzbBuffer> SVONodesBuffers_G(0);
	for (int i = 0; i < this->svoCuda_pg->SVONodes_maxDepth - 1; ++i) {
		uint32_t nodeCount = this->svoCuda_pg->SVOLayerInfos_G_host[i].divisibleNodeCount * 8;
		if (nodeCount == 0) break;
		uint32_t bufferSize = nodeCount * sizeof(FzbSVONodeData_PG_G);
		SVONodesBuffers_G.push_back(fzbCreateStorageBuffer(bufferSize, true, false));
	}
	this->svoCuda_pg->copySVODataToBuffer(SVONodesBuffers_G, true);

	std::vector<FzbBuffer> SVONodesBuffers_E(0);
	for (int i = 0; i < this->svoCuda_pg->SVONodes_maxDepth - 1; ++i) {
		uint32_t nodeCount = this->svoCuda_pg->SVOLayerInfos_G_host[i].divisibleNodeCount * 8;
		if (nodeCount == 0) break;
		uint32_t bufferSize = nodeCount * sizeof(FzbSVONodeData_PG_E);
		SVONodesBuffers_E.push_back(fzbCreateStorageBuffer(bufferSize, true, false));
	}
	this->svoCuda_pg->copySVODataToBuffer(SVONodesBuffers_E, false);

	this->SVONodesBuffers.resize(0);
	this->SVONodesBuffers.insert(this->SVONodesBuffers.end(), SVONodesBuffers_G.begin(), SVONodesBuffers_G.end());
	this->SVONodesBuffers.insert(this->SVONodesBuffers.end(), SVONodesBuffers_E.begin(), SVONodesBuffers_E.end());

	uint32_t bufferSize = this->svoCuda_pg->SVOInDivisibleNode_G_TotalCount_host * this->svoCuda_pg->SVONode_E_TotalCount_host * sizeof(float);
	this->SVOWeightsBuffer = fzbCreateStorageBuffer(bufferSize, true, false);
	this->svoCuda_pg->copySVONodeWeightsToBuffer(SVOWeightsBuffer);
}
//--------------------------------------------------Debug--------------------------------------------------
FzbSVO_PG_Debug::FzbSVO_PG_Debug() {};
FzbSVO_PG_Debug::FzbSVO_PG_Debug(pugi::xml_node& SVO_PG_DebugNode) {
	if (std::string(SVO_PG_DebugNode.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
	else return;

	this->componentInfo.name = FZB_FEATURE_COMPONENT_SVO_PG_DEBUG;
	this->componentInfo.type = FZB_LOOPRENDER_FEATURE_COMPONENT;

	if (pugi::xml_node SVOSettingNode = SVO_PG_DebugNode.child("rendererComponentSetting")) {
		this->setting.voxelAABBDebugInfo = std::string(SVOSettingNode.child("voxelAABBDebugInfo").attribute("value").value()) == "true";
		this->setting.voxelIrradianceDebugInfo = std::string(SVOSettingNode.child("voxelIrradianceDebugInfo").attribute("value").value()) == "true";
		this->setting.OctreeNodeDebugInfo = std::string(SVOSettingNode.child("OctreeNodeDebugInfo").attribute("value").value()) == "true";
		if(this->setting.OctreeNodeDebugInfo)
			this->setting.isG = std::string(SVOSettingNode.child("OctreeNodeDebugInfo").attribute("isG").value()) == "true";
		this->setting.SVONodeClusterDebugInfo = std::string(SVOSettingNode.child("SVONodeClusterDebugInfo").attribute("value").value()) == "true";
		if (this->setting.SVONodeClusterDebugInfo)
			this->setting.isG = std::string(SVOSettingNode.child("SVONodeClusterDebugInfo").attribute("isG").value()) == "true";
		//this->setting.SVONodeClusterLevel = std::stoi(SVOSettingNode.child("SVONodeClusterDebugInfo").attribute("level").value());
		//this->setting.lookCube = std::string(SVOSettingNode.child("SVONodeClusterDebugInfo").attribute("lookCube").value()) == "true";
		this->setting.SVOWeightsDebugInfo = std::string(SVOSettingNode.child("SVOWeightsDebugInfo").attribute("value").value()) == "true";
	}

	addMainSceneInfo();
	addExtensions();
	if (pugi::xml_node childComponentsNode = SVO_PG_DebugNode.child("childComponents")) getChildComponent(childComponentsNode); 
	if(childComponents.count("SVO_PG")) {
		this->SVO_PG = std::dynamic_pointer_cast<FzbSVO_PG>(childComponents["SVO_PG"]);
		this->setting.SVO_PGSetting = this->SVO_PG->setting;
	}
	else throw std::runtime_error("SVO_PG_Debug需要嵌套SVO_PG子组件");

	while (pow(2, this->SVO_PG_MaxDepth) <= this->SVO_PG->setting.voxelNum) ++this->SVO_PG_MaxDepth;
	//if (setting.SVONodeClusterLevel == 0) throw std::runtime_error("暂时不支持查看无聚类效果");
	//if (setting.SVONodeClusterLevel >= this->SVO_PG_MaxDepth - 1) this->setting.SVONodeClusterLevel = this->SVO_PG_MaxDepth - 2;
}
void FzbSVO_PG_Debug::addMainSceneInfo() {
	
};
void FzbSVO_PG_Debug::addExtensions() {
	FzbRenderer::globalData.deviceFeatures.fillModeNonSolid = VK_TRUE;
	FzbRenderer::globalData.deviceFeatures.wideLines = VK_TRUE;
};

void FzbSVO_PG_Debug::createImages() {
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
void FzbSVO_PG_Debug::init() {
	FzbFeatureComponent_LoopRender::init();
	presentPrepare();
}
FzbSemaphore FzbSVO_PG_Debug::render(uint32_t imageIndex, FzbSemaphore startSemaphore, VkFence fence) {
	VkCommandBuffer commandBuffer = commandBuffers[0];
	vkResetCommandBuffer(commandBuffer, 0);
	fzbBeginCommandBuffer(commandBuffer);

	renderRenderPass.render(commandBuffer, imageIndex);

	std::vector<VkSemaphore> waitSemaphores = { startSemaphore.semaphore };
	std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
	fzbSubmitCommandBuffer(commandBuffer, { startSemaphore.semaphore }, waitStages, { renderFinishedSemaphore.semaphore }, fence);

	return renderFinishedSemaphore;
}
void FzbSVO_PG_Debug::clean() {
	FzbFeatureComponent_LoopRender::clean();
	presentSourceManager.clean();
	depthMap.clean();
	if (descriptorSetLayout) vkDestroyDescriptorSetLayout(FzbRenderer::globalData.logicalDevice, descriptorSetLayout, nullptr);
	debugUniformBuffer.clean();
}

void FzbSVO_PG_Debug::presentPrepare() {
	fzbCreateCommandBuffers(1);
	FzbRenderer::globalData.mainScene.createCameraAndLightDescriptor();
	if (setting.voxelAABBDebugInfo) createVGBRenderPass_AABBInfo();
	else if (setting.voxelIrradianceDebugInfo) createVGBRenderPass_IrradianceInfo();
	else if (setting.OctreeNodeDebugInfo) {
		createOctreeDebugBufferAndDescirptor();
		createOctreeRenderPass();
	}
	else if (setting.SVONodeClusterDebugInfo) {
		createSVODebugBufferAndDescirptor();
		createVGBRenderPass_SVONodeClusterInfo();
	}
	else if (setting.SVOWeightsDebugInfo) {
		createSVOWeightsDebugBufferAndDescirptor();
		createSVORenderPass_SVOWeightsInfo();
	}
}

void FzbSVO_PG_Debug::createVGBRenderPass_AABBInfo() {
	this->presentSourceManager.addMeshMaterial(FzbRenderer::globalData.mainScene.sceneMeshSet);
	FzbShaderInfo diffuseShaderInfo = { "/core/Materials/Diffuse/shaders/forwardRender" };
	FzbShaderInfo roughconductorShaderInfo = { "/core/Materials/roughconductor/shaders/forwardRender" };

	FzbMesh cubeMesh = FzbMesh();
	cubeMesh.instanceNum = std::pow(setting.SVO_PGSetting.voxelNum, 3);
	fzbCreateCubeWireframe(cubeMesh);
	this->presentSourceManager.componentScene.addMeshToScene(cubeMesh);
	this->presentSourceManager.componentScene.createVertexBuffer(true, false);
	this->presentSourceManager.addMeshMaterial(this->presentSourceManager.componentScene.sceneMeshSet, FzbMaterial("voxelAABBDebugMaterial", "voxelAABBDebugMaterial"));
	FzbShaderInfo voxelAABBShaderInfo = { "/core/SceneDivision/SVO_PG/shaders/Debug/voxelAABBPresent" };

	this->presentSourceManager.addSource({ {"voxelAABBDebugMaterial", voxelAABBShaderInfo}, {"diffuse", diffuseShaderInfo }, { "roughconductor", roughconductorShaderInfo } });

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

	std::vector<FzbShader*> shaderPaths = { &this->presentSourceManager.shaderSet[diffuseShaderInfo.shaderPath],  &this->presentSourceManager.shaderSet[roughconductorShaderInfo.shaderPath] };
	FzbSubPass forwardSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
		{ mainScene->cameraAndLightsDescriptorSetLayout }, { mainScene->cameraAndLightsDescriptorSet },
		mainScene->vertexBuffer.buffer, mainScene->indexBuffer.buffer, shaderPaths);
	renderRenderPass.addSubPass(forwardSubPass);
	
	FzbSubPass CubeWireframeSubPass = FzbSubPass(renderRenderPass.renderPass, 1,
		{ mainScene->cameraAndLightsDescriptorSetLayout, this->SVO_PG->descriptorSetLayout },
		{ mainScene->cameraAndLightsDescriptorSet, this->SVO_PG->descriptorSet },
		this->presentSourceManager.componentScene.vertexBuffer.buffer, this->presentSourceManager.componentScene.indexBuffer.buffer,
		{ &this->presentSourceManager.shaderSet[voxelAABBShaderInfo.shaderPath] });
	renderRenderPass.addSubPass(CubeWireframeSubPass);
}
void FzbSVO_PG_Debug::createVGBRenderPass_IrradianceInfo() {
	FzbMesh cubeMesh = FzbMesh();
	cubeMesh.instanceNum = std::pow(setting.SVO_PGSetting.voxelNum, 3);
	fzbCreateCube(cubeMesh, FzbVertexFormat());
	this->presentSourceManager.componentScene.addMeshToScene(cubeMesh);
	this->presentSourceManager.componentScene.createVertexBuffer(true, false);
	this->presentSourceManager.addMeshMaterial(this->presentSourceManager.componentScene.sceneMeshSet, FzbMaterial("voxelIrradianceDebugMaterial", "voxelIrradianceDebugMaterial"));
	FzbShaderInfo voxelAABBShaderInfo = { "/core/SceneDivision/SVO_PG/shaders/Debug/voxelIrradiancePresent" };

	this->presentSourceManager.addSource({ {"voxelIrradianceDebugMaterial", voxelAABBShaderInfo} });

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

	FzbSubPass CubeWireframeSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
		{ mainScene->cameraAndLightsDescriptorSetLayout, this->SVO_PG->descriptorSetLayout },
		{ mainScene->cameraAndLightsDescriptorSet, this->SVO_PG->descriptorSet },
		this->presentSourceManager.componentScene.vertexBuffer.buffer, this->presentSourceManager.componentScene.indexBuffer.buffer,
		{ &this->presentSourceManager.shaderSet[voxelAABBShaderInfo.shaderPath] });
	renderRenderPass.addSubPass(CubeWireframeSubPass);
}

void FzbSVO_PG_Debug::createOctreeDebugBufferAndDescirptor() {
	this->SVO_PG->createOctreeBuffers(setting.isG);

	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, this->SVO_PG_MaxDepth - 2 });
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
	this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);

	uint32_t descriptorSetArraySize = this->SVO_PG_MaxDepth - 2 + 1;

	std::vector<VkDescriptorType> descriptorTypes(0);
	descriptorTypes.push_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	for (int i = 0; i < this->SVO_PG_MaxDepth - 2; ++i) descriptorTypes.push_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	std::vector<VkShaderStageFlags> descriptorShaderFlags(0);
	descriptorShaderFlags.push_back(VK_SHADER_STAGE_ALL);
	for (int i = 0; i < this->SVO_PG_MaxDepth - 2; ++i) descriptorShaderFlags.push_back(VK_SHADER_STAGE_ALL);
	descriptorSetLayout = fzbCreateDescriptLayout(descriptorSetArraySize, descriptorTypes, descriptorShaderFlags);
	descriptorSet = fzbCreateDescriptorSet(descriptorPool, descriptorSetLayout);

	this->debugUniformBuffer = fzbCreateUniformBuffer(sizeof(FzbSVONodeClusterUniformObject));
	FzbSVONodeClusterUniformObject nodeClusterUniformObject;
	nodeClusterUniformObject.nodeSize_Num = this->SVO_PG->uniformBufferObject.voxelSize_Num;
	nodeClusterUniformObject.startPos = this->SVO_PG->uniformBufferObject.voxelStartPos;
	nodeClusterUniformObject.maxDepth = this->SVO_PG_MaxDepth;
	memcpy(debugUniformBuffer.mapped, &nodeClusterUniformObject, sizeof(FzbSVONodeClusterUniformObject));

	std::vector<VkWriteDescriptorSet> OctreeBufferDescriptorWrites(descriptorSetArraySize);

	VkDescriptorBufferInfo uniformBuffer;
	uniformBuffer.buffer = debugUniformBuffer.buffer;
	uniformBuffer.offset = 0;
	uniformBuffer.range = debugUniformBuffer.size;

	OctreeBufferDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	OctreeBufferDescriptorWrites[0].dstSet = descriptorSet;
	OctreeBufferDescriptorWrites[0].dstBinding = 0;
	OctreeBufferDescriptorWrites[0].dstArrayElement = 0;
	OctreeBufferDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	OctreeBufferDescriptorWrites[0].descriptorCount = 1;
	OctreeBufferDescriptorWrites[0].pBufferInfo = &uniformBuffer;

	std::vector<VkDescriptorBufferInfo> octreeBufferInfos(this->SVO_PG_MaxDepth - 2);
	for (int i = 0; i < this->SVO_PG_MaxDepth - 2; ++i) {
		octreeBufferInfos[i].buffer = this->SVO_PG->OctreeNodesBuffers[i].buffer;
		octreeBufferInfos[i].offset = 0;
		octreeBufferInfos[i].range = this->SVO_PG->OctreeNodesBuffers[i].size;

		OctreeBufferDescriptorWrites[i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		OctreeBufferDescriptorWrites[i + 1].dstSet = descriptorSet;
		OctreeBufferDescriptorWrites[i + 1].dstBinding = i + 1;
		OctreeBufferDescriptorWrites[i + 1].dstArrayElement = 0;
		OctreeBufferDescriptorWrites[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		OctreeBufferDescriptorWrites[i + 1].descriptorCount = 1;
		OctreeBufferDescriptorWrites[i + 1].pBufferInfo = &octreeBufferInfos[i];
	}

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, OctreeBufferDescriptorWrites.size(), OctreeBufferDescriptorWrites.data(), 0, nullptr);
}
void FzbSVO_PG_Debug::createOctreeRenderPass() {
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

	std::map<std::string, FzbShaderInfo> shaderInfos;
	FzbMesh cubeMesh = FzbMesh();
	FzbShaderInfo diffuseShaderInfo = { "/core/Materials/Diffuse/shaders/forwardRender" };
	FzbShaderInfo roughconductorShaderInfo = { "/core/Materials/roughconductor/shaders/forwardRender" };
	this->presentSourceManager.addMeshMaterial(FzbRenderer::globalData.mainScene.sceneMeshSet);
	shaderInfos.insert({ "diffuse", diffuseShaderInfo });
	shaderInfos.insert({ "roughconductor", roughconductorShaderInfo });

	//搞一个线框，显示聚类后的node的范围
	uint32_t nodeCount = 0;
	for (int i = 1; i < this->SVO_PG_MaxDepth - 1; ++i) nodeCount += std::pow(8, i);
	nodeCount *= 2;
	cubeMesh.clean();
	cubeMesh = FzbMesh();
	cubeMesh.instanceNum = nodeCount;
	//cubeMesh.instanceNum = std::pow(setting.SVO_PGSetting.voxelNum, 3) / pow(8, setting.SVONodeClusterLevel + 1);
	fzbCreateCubeWireframe(cubeMesh);
	this->presentSourceManager.componentScene.addMeshToScene(cubeMesh);

	this->presentSourceManager.addMeshMaterial(&this->presentSourceManager.componentScene.sceneMeshSet[0], 
		FzbMaterial("octreeNodeClusterWireframeDebugMaterial", "octreeNodeClusterWireframeDebugMaterial"));
	FzbShaderInfo octreeNodeClusterWireframeShaderInfo = { "/core/SceneDivision/SVO_PG/shaders/Debug/octreeNodeClusterPresent" };
	shaderInfos.insert({ "octreeNodeClusterWireframeDebugMaterial", octreeNodeClusterWireframeShaderInfo });

	this->presentSourceManager.componentScene.createVertexBuffer(true, false);
	this->presentSourceManager.addSource(shaderInfos);

	this->presentSourceManager.shaderSet[octreeNodeClusterWireframeShaderInfo.shaderPath].shaderVariants[0].macros.insert({ "isG", setting.isG });

	FzbSubPass CubeWireframeSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
		{ mainScene->cameraAndLightsDescriptorSetLayout, this->descriptorSetLayout },
		{ mainScene->cameraAndLightsDescriptorSet, this->descriptorSet },
		this->presentSourceManager.componentScene.vertexBuffer.buffer, this->presentSourceManager.componentScene.indexBuffer.buffer,
		{ &this->presentSourceManager.shaderSet[octreeNodeClusterWireframeShaderInfo.shaderPath] });
	renderRenderPass.addSubPass(CubeWireframeSubPass);

	std::vector<FzbShader*> shaderPaths = { &this->presentSourceManager.shaderSet[diffuseShaderInfo.shaderPath],  &this->presentSourceManager.shaderSet[roughconductorShaderInfo.shaderPath] };
	FzbSubPass forwardSubPass = FzbSubPass(renderRenderPass.renderPass, 1,
		{ mainScene->cameraAndLightsDescriptorSetLayout }, { mainScene->cameraAndLightsDescriptorSet },
		mainScene->vertexBuffer.buffer, mainScene->indexBuffer.buffer, shaderPaths);
	renderRenderPass.addSubPass(forwardSubPass);
}

void FzbSVO_PG_Debug::createSVODebugBufferAndDescirptor() {
	this->SVO_PG->createSVOBuffers(setting.isG, this->setting.useDeviceAddress);

	this->debugUniformBuffer = fzbCreateUniformBuffer(sizeof(FzbSVONodeClusterUniformObject));
	//float scale = powf(2.0f, setting.SVONodeClusterLevel + 1);
	//glm::vec3 nodeSize = glm::vec3(this->SVO_PG->uniformBufferObject.voxelSize_Num) * scale;
	//float nodeNum = this->SVO_PG->uniformBufferObject.voxelSize_Num.w / scale;
	FzbSVONodeClusterUniformObject nodeClusterUniformObject;
	nodeClusterUniformObject.nodeSize_Num = this->SVO_PG->uniformBufferObject.voxelSize_Num;
	nodeClusterUniformObject.startPos = this->SVO_PG->uniformBufferObject.voxelStartPos;
	nodeClusterUniformObject.maxDepth = this->SVO_PG_MaxDepth;
	//nodeClusterUniformObject.nodeClusterInfoLevel = this->setting.SVONodeClusterLevel;
	for (int i = 0; i < this->SVO_PG->SVONodesBuffers.size(); ++i) {	//不算根节点
		nodeClusterUniformObject.nodeCounts[i] = 
			(setting.isG ? this->SVO_PG->svoCuda_pg->SVOLayerInfos_G_host[i].divisibleNodeCount : 
				this->SVO_PG->svoCuda_pg->SVOLayerInfos_E_host[i].divisibleNodeCount) * 8;
		if (this->setting.useDeviceAddress) {
			this->SVO_PG->SVONodesBuffers[i].getBufferDeviceAddress();
			nodeClusterUniformObject.SVONodesAddress[i] = this->SVO_PG->SVONodesBuffers[i].deviceAddress;
		}
	}
	memcpy(debugUniformBuffer.mapped, &nodeClusterUniformObject, sizeof(FzbSVONodeClusterUniformObject));

	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
	if (!setting.useDeviceAddress) bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, this->SVO_PG->SVONodesBuffers.size() });
	this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);

	uint32_t descriptorSetArraySize = setting.useDeviceAddress ? 1 : this->SVO_PG->SVONodesBuffers.size() + 1;

	std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
	for (int i = 0; i < this->SVO_PG->SVONodesBuffers.size(); ++i) descriptorTypes.push_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL };
	for (int i = 0; i < this->SVO_PG->SVONodesBuffers.size(); ++i) descriptorShaderFlags.push_back(VK_SHADER_STAGE_ALL);
	descriptorSetLayout = fzbCreateDescriptLayout(descriptorSetArraySize, descriptorTypes, descriptorShaderFlags);
	descriptorSet = fzbCreateDescriptorSet(descriptorPool, descriptorSetLayout);


	std::vector<VkWriteDescriptorSet> SVOBufferDescriptorWrites(descriptorSetArraySize);
	//VkDescriptorBufferInfo svoBufferInfo{};
	//svoBufferInfo.buffer = this->SVO_PG->SVOBuffers[this->SVO_PG->SVOBuffers.size() - setting.SVONodeClusterLevel - 1].buffer;
	//svoBufferInfo.offset = 0;
	//svoBufferInfo.range = this->SVO_PG->SVOBuffers[this->SVO_PG->SVOBuffers.size() - setting.SVONodeClusterLevel - 1].size;
	//SVOBufferDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	//SVOBufferDescriptorWrites[0].dstSet = descriptorSet;
	//SVOBufferDescriptorWrites[0].dstBinding = 0;
	//SVOBufferDescriptorWrites[0].dstArrayElement = 0;
	//SVOBufferDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	//SVOBufferDescriptorWrites[0].descriptorCount = 1;
	//SVOBufferDescriptorWrites[0].pBufferInfo = &svoBufferInfo;

	VkDescriptorBufferInfo svoUniformBufferInfo{};
	svoUniformBufferInfo.buffer = this->debugUniformBuffer.buffer;
	svoUniformBufferInfo.offset = 0;
	svoUniformBufferInfo.range = this->debugUniformBuffer.size;
	SVOBufferDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	SVOBufferDescriptorWrites[0].dstSet = descriptorSet;
	SVOBufferDescriptorWrites[0].dstBinding = 0;
	SVOBufferDescriptorWrites[0].dstArrayElement = 0;
	SVOBufferDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	SVOBufferDescriptorWrites[0].descriptorCount = 1;
	SVOBufferDescriptorWrites[0].pBufferInfo = &svoUniformBufferInfo;

	std::vector<VkDescriptorBufferInfo> svoBufferInfos(this->SVO_PG->SVONodesBuffers.size());
	if (!setting.useDeviceAddress) {
		for (int i = 0; i < this->SVO_PG->SVONodesBuffers.size(); ++i) {
			svoBufferInfos[i].buffer = this->SVO_PG->SVONodesBuffers[i].buffer;
			svoBufferInfos[i].offset = 0;
			svoBufferInfos[i].range = this->SVO_PG->SVONodesBuffers[i].size;

			SVOBufferDescriptorWrites[i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			SVOBufferDescriptorWrites[i + 1].dstSet = descriptorSet;
			SVOBufferDescriptorWrites[i + 1].dstBinding = i + 1;
			SVOBufferDescriptorWrites[i + 1].dstArrayElement = 0;
			SVOBufferDescriptorWrites[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			SVOBufferDescriptorWrites[i + 1].descriptorCount = 1;
			SVOBufferDescriptorWrites[i + 1].pBufferInfo = &svoBufferInfos[i];
		}
	}

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, SVOBufferDescriptorWrites.size(), SVOBufferDescriptorWrites.data(), 0, nullptr);
}
void FzbSVO_PG_Debug::createVGBRenderPass_SVONodeClusterInfo() {
	VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(FzbRenderer::globalData.swapChainImageFormat);
	VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	VkAttachmentDescription depthMapAttachment = fzbCreateDepthAttachment();
	VkAttachmentReference depthMapAttachmentResolveRef = fzbCreateAttachmentReference(1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve, depthMapAttachment };

	std::vector<VkSubpassDescription> subpasses;
	subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));
	subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef, &depthMapAttachmentResolveRef));

	//VkSubpassDependency dependency = {};
	//dependency.srcSubpass = 0;
	//dependency.dstSubpass = 1;
	//dependency.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	//dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	//dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	//dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	//dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	// 
	//VkSubpassDependency dependency = fzbCreateSubpassDependency();
	//
	//VkSubpassDependency dependency = fzbCreateSubpassDependency(0, 1,
	//	VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
	//	VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT);
	//
	 VkSubpassDependency dependency = fzbCreateSubpassDependency(0, 1,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

	FzbRenderPassSetting renderPassSetting = { true, 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
	renderRenderPass.setting = renderPassSetting;
	renderRenderPass.createRenderPass(&attachments, subpasses, { dependency });
	renderRenderPass.createFramebuffers(true);

	uint32_t nodeCount = 0;
	for (int i = 0; i < this->SVO_PG_MaxDepth - 1; ++i) nodeCount +=
		(setting.isG ? this->SVO_PG->svoCuda_pg->SVOLayerInfos_G_host[i].divisibleNodeCount : this->SVO_PG->svoCuda_pg->SVOLayerInfos_E_host[i].divisibleNodeCount) * 8;
	std::map<std::string, FzbShaderInfo> shaderInfos;

	FzbShaderInfo svoNodeClusterCubeShaderInfo = { "/core/SceneDivision/SVO_PG/shaders/Debug/svoNodeClusterPresent/cube" };

	FzbShaderInfo svoNodeClusterWireframeShaderInfo;
	if (setting.useDeviceAddress) svoNodeClusterWireframeShaderInfo = { "/core/SceneDivision/SVO_PG/shaders/Debug/svoNodeClusterPresent/wireFrame" };
	else svoNodeClusterWireframeShaderInfo = { "/core/SceneDivision/SVO_PG/shaders/Debug/svoNodeClusterPresent/wireFrame2" };

	FzbShaderInfo diffuseShaderInfo = { "/core/Materials/Diffuse/shaders/forwardRender" };
	FzbShaderInfo roughconductorShaderInfo = { "/core/Materials/roughconductor/shaders/forwardRender" };

	this->presentSourceManager.addMeshMaterial(FzbRenderer::globalData.mainScene.sceneMeshSet);
	shaderInfos.insert({ "diffuse", diffuseShaderInfo });
	shaderInfos.insert({ "roughconductor", roughconductorShaderInfo });

	FzbMesh cubeMesh = FzbMesh();
	if (setting.lookCube) {
		cubeMesh.instanceNum = nodeCount;
		//搞一个cube，展示聚类后内部AABB及其irradiance
		fzbCreateCube(cubeMesh, FzbVertexFormat());
		this->presentSourceManager.componentScene.addMeshToScene(cubeMesh);
		this->presentSourceManager.addMeshMaterial(&this->presentSourceManager.componentScene.sceneMeshSet[0], FzbMaterial("svoNodeClusterCubeDebugMaterial", "svoNodeClusterCubeDebugMaterial"));
		shaderInfos.insert({ "svoNodeClusterCubeDebugMaterial", svoNodeClusterCubeShaderInfo });
	}
	else {
		nodeCount *= 2;
		cubeMesh.instanceNum = nodeCount;

		//搞一个线框，显示聚类后的node的范围
		fzbCreateCubeWireframe(cubeMesh);
		this->presentSourceManager.componentScene.addMeshToScene(cubeMesh);
		this->presentSourceManager.addMeshMaterial(&this->presentSourceManager.componentScene.sceneMeshSet[0], FzbMaterial("svoNodeClusterWireframeDebugMaterial", "svoNodeClusterWireframeDebugMaterial"));
		shaderInfos.insert({ "svoNodeClusterWireframeDebugMaterial", svoNodeClusterWireframeShaderInfo });
	}

	this->presentSourceManager.componentScene.createVertexBuffer(true, false);
	this->presentSourceManager.addSource(shaderInfos);

	if (setting.lookCube) {
		for (int i = 1; i < this->SVO_PG_MaxDepth; ++i) {
			if (this->SVO_PG->SVONodesBuffers[i - 1].size > 0)
				this->presentSourceManager.shaderSet[svoNodeClusterCubeShaderInfo.shaderPath].shaderVariants[0].macros.insert({ "LAYER" + std::to_string(i), true });
		}
	}
	else {
		for (int i = 1; i < this->SVO_PG_MaxDepth; ++i) {
			if (this->SVO_PG->SVONodesBuffers[i - 1].size > 0)
				this->presentSourceManager.shaderSet[svoNodeClusterWireframeShaderInfo.shaderPath].shaderVariants[0].macros.insert({ "LAYER" + std::to_string(i), true });
		}
		this->presentSourceManager.shaderSet[svoNodeClusterWireframeShaderInfo.shaderPath].shaderVariants[0].macros.insert({ "isG", setting.isG });
	}

	if (setting.lookCube) {
		FzbSubPass CubeSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
			{ mainScene->cameraAndLightsDescriptorSetLayout, this->descriptorSetLayout },
			{ mainScene->cameraAndLightsDescriptorSet, this->descriptorSet },
			this->presentSourceManager.componentScene.vertexBuffer.buffer, this->presentSourceManager.componentScene.indexBuffer.buffer,
			{ &this->presentSourceManager.shaderSet[svoNodeClusterCubeShaderInfo.shaderPath] });
		renderRenderPass.addSubPass(CubeSubPass);
	}
	else {
		FzbSubPass CubeWireframeSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
			{ mainScene->cameraAndLightsDescriptorSetLayout, this->descriptorSetLayout },
			{ mainScene->cameraAndLightsDescriptorSet, this->descriptorSet },
			this->presentSourceManager.componentScene.vertexBuffer.buffer, this->presentSourceManager.componentScene.indexBuffer.buffer,
			{ &this->presentSourceManager.shaderSet[svoNodeClusterWireframeShaderInfo.shaderPath] });
		renderRenderPass.addSubPass(CubeWireframeSubPass);
	}

	std::vector<FzbShader*> shaderPaths = { &this->presentSourceManager.shaderSet[diffuseShaderInfo.shaderPath],  &this->presentSourceManager.shaderSet[roughconductorShaderInfo.shaderPath] };
	FzbSubPass forwardSubPass = FzbSubPass(renderRenderPass.renderPass, 1,
		{ mainScene->cameraAndLightsDescriptorSetLayout }, { mainScene->cameraAndLightsDescriptorSet },
		mainScene->vertexBuffer.buffer, mainScene->indexBuffer.buffer, shaderPaths);
	renderRenderPass.addSubPass(forwardSubPass);
}

void FzbSVO_PG_Debug::createSVOWeightsDebugBufferAndDescirptor() {
	this->SVO_PG->createSVOWeightsBuffer();

	FzbSVOWeightsUniformObject uniformObject;
	uniformObject.nodeSize_Num = this->SVO_PG->uniformBufferObject.voxelSize_Num;
	uniformObject.startPos = this->SVO_PG->uniformBufferObject.voxelStartPos;
	uniformObject.maxDepth = this->SVO_PG_MaxDepth;
	for (int i = 0; i < this->SVO_PG_MaxDepth; ++i) {
		uniformObject.divisibleNodeCounts_G[i] = this->SVO_PG->svoCuda_pg->SVOLayerInfos_G_host[i].divisibleNodeCount;
		uniformObject.indivisibleNodeCounts_G[i] = this->SVO_PG->svoCuda_pg->SVOLayerInfos_G_host[i].indivisibleNodeCount;

		uniformObject.divisibleNodeCounts_E[i] = this->SVO_PG->svoCuda_pg->SVOLayerInfos_E_host[i].divisibleNodeCount;
		uniformObject.indivisibleNodeCounts_E[i] = this->SVO_PG->svoCuda_pg->SVOLayerInfos_E_host[i].indivisibleNodeCount;
	}
	uniformObject.svoNodeTotalCount = this->SVO_PG->svoCuda_pg->SVONode_E_TotalCount_host;

	this->debugUniformBuffer = fzbCreateUniformBuffer(sizeof(FzbSVOWeightsUniformObject));
	memcpy(debugUniformBuffer.mapped, &uniformObject, sizeof(FzbSVOWeightsUniformObject));

	uint32_t descriptorSetArraySize = 2;

	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, this->SVO_PG->SVONodesBuffers.size() + 1 });
	this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);

	std::vector<VkDescriptorType> descriptorTypes(0);
	descriptorTypes.push_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
	descriptorTypes.push_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	for (int i = 0; i < this->SVO_PG->SVONodesBuffers.size(); ++i) {
		descriptorTypes.push_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
		++descriptorSetArraySize;
	}
	std::vector<VkShaderStageFlags> descriptorShaderFlags(0);
	descriptorShaderFlags.push_back(VK_SHADER_STAGE_ALL);
	descriptorShaderFlags.push_back(VK_SHADER_STAGE_ALL);
	for (int i = 0; i < this->SVO_PG->SVONodesBuffers.size(); ++i) {
		descriptorShaderFlags.push_back(VK_SHADER_STAGE_ALL);
	}
	descriptorSetLayout = fzbCreateDescriptLayout(descriptorSetArraySize, descriptorTypes, descriptorShaderFlags);
	descriptorSet = fzbCreateDescriptorSet(descriptorPool, descriptorSetLayout);

	std::vector<VkWriteDescriptorSet> svoWeightsBufferDescriptorWrites(descriptorSetArraySize);

	VkDescriptorBufferInfo uniformBuffer;
	uniformBuffer.buffer = debugUniformBuffer.buffer;
	uniformBuffer.offset = 0;
	uniformBuffer.range = debugUniformBuffer.size;
	svoWeightsBufferDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	svoWeightsBufferDescriptorWrites[0].dstSet = descriptorSet;
	svoWeightsBufferDescriptorWrites[0].dstBinding = 0;
	svoWeightsBufferDescriptorWrites[0].dstArrayElement = 0;
	svoWeightsBufferDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	svoWeightsBufferDescriptorWrites[0].descriptorCount = 1;
	svoWeightsBufferDescriptorWrites[0].pBufferInfo = &uniformBuffer;

	VkDescriptorBufferInfo weightsBuffer;
	weightsBuffer.buffer = this->SVO_PG->SVOWeightsBuffer.buffer;
	weightsBuffer.offset = 0;
	weightsBuffer.range = this->SVO_PG->SVOWeightsBuffer.size;
	svoWeightsBufferDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	svoWeightsBufferDescriptorWrites[1].dstSet = descriptorSet;
	svoWeightsBufferDescriptorWrites[1].dstBinding = 1;
	svoWeightsBufferDescriptorWrites[1].dstArrayElement = 0;
	svoWeightsBufferDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	svoWeightsBufferDescriptorWrites[1].descriptorCount = 1;
	svoWeightsBufferDescriptorWrites[1].pBufferInfo = &weightsBuffer;

	std::vector<VkDescriptorBufferInfo> svoBufferInfos(this->SVO_PG->SVONodesBuffers.size());
	for (int i = 0; i < this->SVO_PG->SVONodesBuffers.size(); ++i) {
		svoBufferInfos[i].buffer = this->SVO_PG->SVONodesBuffers[i].buffer;
		svoBufferInfos[i].offset = 0;
		svoBufferInfos[i].range = this->SVO_PG->SVONodesBuffers[i].size;

		svoWeightsBufferDescriptorWrites[i + 2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		svoWeightsBufferDescriptorWrites[i + 2].dstSet = descriptorSet;
		svoWeightsBufferDescriptorWrites[i + 2].dstBinding = i + 2;
		svoWeightsBufferDescriptorWrites[i + 2].dstArrayElement = 0;
		svoWeightsBufferDescriptorWrites[i + 2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		svoWeightsBufferDescriptorWrites[i + 2].descriptorCount = 1;
		svoWeightsBufferDescriptorWrites[i + 2].pBufferInfo = &svoBufferInfos[i];
	}

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, svoWeightsBufferDescriptorWrites.size(), svoWeightsBufferDescriptorWrites.data(), 0, nullptr);
}
//根据相机的pos，找到想要的indivisibleNode，然后显示其与各个node之间的weight
void FzbSVO_PG_Debug::createSVORenderPass_SVOWeightsInfo() {
	uint32_t nodeCount = 0;
	for (int i = 0; i < this->SVO_PG_MaxDepth - 1; ++i) nodeCount += this->SVO_PG->svoCuda_pg->SVOLayerInfos_E_host[i].divisibleNodeCount * 8;

	FzbMesh cubeMesh = FzbMesh();
	cubeMesh.instanceNum = nodeCount;
	fzbCreateCube(cubeMesh, FzbVertexFormat());
	this->presentSourceManager.componentScene.addMeshToScene(cubeMesh);
	this->presentSourceManager.componentScene.createVertexBuffer(true, false);
	this->presentSourceManager.addMeshMaterial(this->presentSourceManager.componentScene.sceneMeshSet, FzbMaterial("svoWeightsDebugMaterial", "svoWeightsDebugMaterial"));
	FzbShaderInfo svoWeightsShaderInfo = { "/core/SceneDivision/SVO_PG/shaders/Debug/svoWeightsPresent" };

	this->presentSourceManager.addSource({ {"svoWeightsDebugMaterial", svoWeightsShaderInfo} });

	for (int i = 1; i < this->SVO_PG_MaxDepth; ++i) {
		if(this->SVO_PG->svoCuda_pg->SVOLayerInfos_G_host[i - 1].divisibleNodeCount > 0)
			this->presentSourceManager.shaderSet[svoWeightsShaderInfo.shaderPath].shaderVariants[0].macros.insert({ "LAYER" + std::to_string(i) + "_G", true});
		if (this->SVO_PG->svoCuda_pg->SVOLayerInfos_E_host[i - 1].divisibleNodeCount > 0)
			this->presentSourceManager.shaderSet[svoWeightsShaderInfo.shaderPath].shaderVariants[0].macros.insert({ "LAYER" + std::to_string(i) + "_E", true });
	}

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

	FzbSubPass CubeSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
		{ mainScene->cameraAndLightsDescriptorSetLayout, this->descriptorSetLayout },
		{ mainScene->cameraAndLightsDescriptorSet, this->descriptorSet },
		this->presentSourceManager.componentScene.vertexBuffer.buffer, this->presentSourceManager.componentScene.indexBuffer.buffer,
		{ &this->presentSourceManager.shaderSet[svoWeightsShaderInfo.shaderPath] });
	renderRenderPass.addSubPass(CubeSubPass);
}