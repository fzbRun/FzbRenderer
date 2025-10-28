#include "./FzbSVOPathGuiding_soft.h"

FzbSVOPathGuiding_soft::FzbSVOPathGuiding_soft() {};
FzbSVOPathGuiding_soft::FzbSVOPathGuiding_soft(pugi::xml_node& SVOPathGuidingNode_soft) {
	if (std::string(SVOPathGuidingNode_soft.child("available").attribute("value").value()) == "true") this->componentInfo.available = true;
	else return;

	this->componentInfo.name = FZB_RENDERER_SVO_PATH_GUIDING;
	this->componentInfo.type = FZB_LOOPRENDER_FEATURE_COMPONENT;

	if (pugi::xml_node componentSettingNode = SVOPathGuidingNode_soft.child("rendererComponentSetting")) {
		this->setting.spp = std::stoi(componentSettingNode.child("spp").attribute("value").value());
		this->setting.useSphericalRectangleSample = std::string(componentSettingNode.child("useSphericalRectangleSample").attribute("value").value()) == "true";
	}

	addMainSceneInfo();
	addExtensions();
	if (pugi::xml_node childComponentsNode = SVOPathGuidingNode_soft.child("childComponents")) getChildComponent(childComponentsNode);
	if (childComponents.count("SVO_PG")) {
		this->SVO_PG = std::dynamic_pointer_cast<FzbSVO_PG>(childComponents["SVO_PG"]);
		this->setting.SVO_PGSetting = this->SVO_PG->setting;
		this->rayTracingSourceManager = &this->SVO_PG->rayTracingSourceManager;
	}
	else throw std::runtime_error("SVO_PG_Debug需要嵌套SVO_PG子组件");

	while (pow(2, this->SVO_PG_MaxDepth) <= this->SVO_PG->setting.voxelNum)++this->SVO_PG_MaxDepth;
}
void FzbSVOPathGuiding_soft::addMainSceneInfo() {
	FzbRenderer::useImageAvailableSemaphoreHandle = true;
	FzbRenderer::globalData.mainScene.vertexFormat_allMesh.mergeUpward(FzbVertexFormat());
	FzbRenderer::globalData.mainScene.useMaterialSource = false;
};
void FzbSVOPathGuiding_soft::addExtensions() {};

void FzbSVOPathGuiding_soft::createImages() {};
void FzbSVOPathGuiding_soft::init() {
	FzbFeatureComponent_LoopRender::init();
	presentPrepare();
}

void FzbSVOPathGuiding_soft::presentPrepare(){
	fzbCreateCommandBuffers(1);

	FzbSVOPathGuidingCudaSetting cudaSetting;
	cudaSetting.spp = setting.spp;
	cudaSetting.useSphericalRectangleSample = setting.useSphericalRectangleSample;
	cudaSetting.voxelCount = setting.SVO_PGSetting.voxelNum;
	cudaSetting.voxelSize = this->SVO_PG->uniformBufferObject.voxelSize_Num;
	cudaSetting.voxelGroupStartPos = this->SVO_PG->uniformBufferObject.voxelStartPos;
	cudaSetting.maxSVOLayer = this->SVO_PG->svoCuda_pg->SVONodes_maxDepth;
	cudaSetting.SVOIndivisibleNodeTotalCount = this->SVO_PG->svoCuda_pg->SVOInDivisibleNodeTotalCount_host;
	cudaSetting.SVONodeTotalCount = this->SVO_PG->svoCuda_pg->SVONodeTotalCount_host;
	cudaSetting.SVONodes = this->SVO_PG->svoCuda_pg->SVONodes_multiLayer_Array;
	cudaSetting.SVOLayerInfos = this->SVO_PG->svoCuda_pg->SVOLayerInfos;
	cudaSetting.SVONodeWeights = this->SVO_PG->svoCuda_pg->SVONodeWeights;
	svoPathGuidingCUDA = std::make_unique<FzbSVOPathGuidingCuda>(rayTracingSourceManager->sourceManagerCuda, cudaSetting, this->SVO_PG->svoCuda_pg);
	
	createBufferAndDescirptor();
	createRenderPass();
}

void FzbSVOPathGuiding_soft::createBufferAndDescirptor() {
	VkExtent2D resolution = FzbRenderer::globalData.getResolution();
	FzbSVOPathGuidingSettingUniformObject settingUniformObject = { resolution.width, resolution.height };
	settingBuffer = fzbCreateUniformBuffer(sizeof(FzbSVOPathGuidingSettingUniformObject));
	memcpy(settingBuffer.mapped, &settingUniformObject, sizeof(FzbSVOPathGuidingSettingUniformObject));

	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 });
	bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 });
	this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);

	std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
	std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_FRAGMENT_BIT };
	descriptorSetLayout = fzbCreateDescriptLayout(2, descriptorTypes, descriptorShaderFlags);
	descriptorSet = fzbCreateDescriptorSet(descriptorPool, descriptorSetLayout);

	std::array<VkWriteDescriptorSet, 2> pathTracingDescriptorWrites{};
	VkDescriptorBufferInfo pathTracingSettingBufferInfo{};
	pathTracingSettingBufferInfo.buffer = settingBuffer.buffer;
	pathTracingSettingBufferInfo.offset = 0;
	pathTracingSettingBufferInfo.range = settingBuffer.size;
	pathTracingDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	pathTracingDescriptorWrites[0].dstSet = descriptorSet;
	pathTracingDescriptorWrites[0].dstBinding = 0;
	pathTracingDescriptorWrites[0].dstArrayElement = 0;
	pathTracingDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	pathTracingDescriptorWrites[0].descriptorCount = 1;
	pathTracingDescriptorWrites[0].pBufferInfo = &pathTracingSettingBufferInfo;

	VkDescriptorBufferInfo pathTracingResultBufferInfo{};
	pathTracingResultBufferInfo.buffer = rayTracingSourceManager->rayTracingResultBuffer.buffer;
	pathTracingResultBufferInfo.offset = 0;
	pathTracingResultBufferInfo.range = rayTracingSourceManager->rayTracingResultBuffer.size;
	pathTracingDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	pathTracingDescriptorWrites[1].dstSet = descriptorSet;
	pathTracingDescriptorWrites[1].dstBinding = 1;
	pathTracingDescriptorWrites[1].dstArrayElement = 0;
	pathTracingDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	pathTracingDescriptorWrites[1].descriptorCount = 1;
	pathTracingDescriptorWrites[1].pBufferInfo = &pathTracingResultBufferInfo;

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, pathTracingDescriptorWrites.size(), pathTracingDescriptorWrites.data(), 0, nullptr);
}

void FzbSVOPathGuiding_soft::createRenderPass() {
	this->presentSourceManager.createCanvas(FzbMaterial("present", "present"));
	FzbShaderInfo shaderInfo = { "/core/RayTracing/SVOPathGuiding/soft/shaders" };
	this->presentSourceManager.addSource({ {"present", shaderInfo} });

	VkAttachmentDescription colorAttachmentResolve = fzbCreateColorAttachment(FzbRenderer::globalData.swapChainImageFormat);
	VkAttachmentReference colorAttachmentResolveRef = fzbCreateAttachmentReference(0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
	std::vector<VkAttachmentDescription> attachments = { colorAttachmentResolve };

	std::vector<VkSubpassDescription> subpasses;
	subpasses.push_back(fzbCreateSubPass(1, &colorAttachmentResolveRef));

	VkSubpassDependency dependency = fzbCreateSubpassDependency();

	FzbRenderPassSetting renderPassSetting = { false , 1, FzbRenderer::globalData.swapChainExtent, FzbRenderer::globalData.swapChainImageViews.size(), true };
	renderRenderPass.setting = renderPassSetting;
	renderRenderPass.createRenderPass(&attachments, subpasses, { dependency });
	renderRenderPass.createFramebuffers(true);

	FzbSubPass presentSubPass = FzbSubPass(renderRenderPass.renderPass, 0,
		{ descriptorSetLayout }, { descriptorSet },
		nullptr, nullptr, this->presentSourceManager.shaders_vector);
	renderRenderPass.addSubPass(presentSubPass);
}

FzbSemaphore FzbSVOPathGuiding_soft::render(uint32_t imageIndex, FzbSemaphore startSemaphore, VkFence fence) {
	VkCommandBuffer commandBuffer = commandBuffers[0];
	vkResetCommandBuffer(commandBuffer, 0);
	fzbBeginCommandBuffer(commandBuffer);

	svoPathGuidingCUDA->SVOPathGuiding(nullptr);
	renderRenderPass.render(commandBuffer, imageIndex);

	std::vector<VkSemaphore> waitSemaphores = { startSemaphore.semaphore, rayTracingSourceManager->rayTracingFinishedSemphore.semaphore };
	std::vector<VkPipelineStageFlags> waitStages = { VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
	fzbSubmitCommandBuffer(commandBuffer, waitSemaphores, waitStages, { renderFinishedSemaphore.semaphore }, fence);

	return renderFinishedSemaphore;
}

void FzbSVOPathGuiding_soft::clean() {
	FzbFeatureComponent_LoopRender::clean();
	settingBuffer.clean();
	//rayTracingSourceManager.clean();
	presentSourceManager.clean();
	svoPathGuidingCUDA->clean();
	if (descriptorSetLayout) vkDestroyDescriptorSetLayout(FzbRenderer::globalData.logicalDevice, this->descriptorSetLayout, nullptr);
}