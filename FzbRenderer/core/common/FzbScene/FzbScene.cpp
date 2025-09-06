#include "../FzbShader/FzbShader.h"
#include "../FzbMaterial/FzbMaterial.h"
#include "./FzbScene.h"
#include "../FzbRenderer.h"
#include "./compressVertices.h"

#include <glm/ext/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <vector>
#include <optional>
#include <iostream>
#include <unordered_map>
#include <sstream>
#include <regex>
#include <filesystem>
#include <thread>
#include "CUDA/FzbSceneCUDA.cuh"


std::vector<std::string> get_all_files(const std::string& dir_path) {
	std::vector<std::string> filePaths;
	for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
		if (entry.is_regular_file()) {
			filePaths.push_back(entry.path().string());
		}
	}
	return filePaths;
}
bool FzbVertexFormatLess::operator()(FzbVertexFormat const& a, FzbVertexFormat const& b) const noexcept {
	return a.getVertexSize() < b.getVertexSize();
};
//-----------------------------------------------场景----------------------------------------
FzbScene::FzbScene() {};
void FzbScene::setScenePath(std::string scenePath) {
	this->scenePath = scenePath;
}
void FzbScene::initScene(bool compress, bool isMainScene) {
	FzbVertexFormat vertexFormat = fzbVertexFormatMergeUpward(FzbRenderer::componentManager.vertexFormat_preprocess, FzbRenderer::componentManager.vertexFormat_looprender);
	addSceneFromMitsubaXML(this->scenePath, vertexFormat);

	std::vector<bool> useBuffer;
	for (int i = 0; i < FzbRenderer::componentManager.useMainSceneBuffer_preprocess.size(); ++i)
		useBuffer.push_back(FzbRenderer::componentManager.useMainSceneBuffer_preprocess[i] || FzbRenderer::componentManager.useMainSceneBuffer_looprender[i]);
	std::vector<bool> useBufferHandle;
	for (int i = 0; i < FzbRenderer::componentManager.useMainSceneBufferHandle_preprocess.size(); ++i)
		useBufferHandle.push_back(FzbRenderer::componentManager.useMainSceneBufferHandle_preprocess[i] || FzbRenderer::componentManager.useMainSceneBufferHandle_looprender[i]);
	createVertexBuffer(compress, useBufferHandle[0]);
	createVertexPairDataBuffer(useBuffer[1], useBuffer[2], compress, useBufferHandle[1], useBufferHandle[2]);

	createShaderMeshBatch();
	createBufferAndDescriptorOfMaterialAndMesh();
	createCameraAndLightBufferAndDescriptor();
}
void FzbScene::clean() {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;

	for (auto& materialPair : sceneMaterials) materialPair.second.clean();
	for (auto& shaderPair : sceneShaders) shaderPair.second.clean();
	for (auto& imagePair : sceneImages) imagePair.second.clean();
	for (int i = 0; i < this->sceneMeshSet.size(); i++) sceneMeshSet[i].clean();

	if(sceneDescriptorPool) vkDestroyDescriptorPool(logicalDevice, sceneDescriptorPool, nullptr);
	if(FzbMeshDynamic::meshDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, FzbMeshDynamic::meshDescriptorSetLayout, nullptr);

	vertexBuffer.clean();
	indexBuffer.clean();
	vertexPosBuffer.clean();
	indexPosBuffer.clean();
	vertexPosNormalBuffer.clean();
	indexPosNormalBuffer.clean();

	cameraBuffer.clean();
	lightsBuffer.clean();
	if(cameraAndLightsDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, cameraAndLightsDescriptorSetLayout, nullptr);
}

void FzbScene::addMeshToScene(FzbMesh mesh, FzbMaterial material, std::string shaderPath) {
	this->sceneMaterials.insert({ material.id, material });
	mesh.material = &this->sceneMaterials[material.id];
	if (!this->sceneShaders.count(shaderPath)) {
		this->sceneShaders.insert({ shaderPath, FzbShader(shaderPath) });
	}
	this->sceneShaders[shaderPath].createShaderVariant(&this->sceneMaterials[material.id], mesh.vertexFormat);
	this->differentVertexFormatMeshIndexs[mesh.vertexFormat].push_back(this->sceneMeshSet.size());
	this->sceneMeshSet.push_back(mesh);
}
void FzbScene::createDefaultMaterial(FzbVertexFormat vertexFormat) {
	FzbMaterial defaultMaterial = FzbMaterial("defaultMaterial", "diffuse");
	this->sceneMaterials.insert({ "defaultMaterial", defaultMaterial });
	std::string shaderPath = fzbGetRootPath() + shaderPaths.at("diffuse");
	this->sceneShaders.insert({ shaderPath, FzbShader(shaderPath) });
	this->sceneShaders[shaderPath].createShaderVariant(&this->sceneMaterials["defaultMaterial"], vertexFormat);
}
void FzbScene::addSceneFromMitsubaXML(std::string path, FzbVertexFormat vertexFormat) {
	if (path == "") return;
	this->scenePath = fzbGetRootPath() + path;
	createDefaultMaterial(vertexFormat);

	pugi::xml_document doc;
	std::string sceneXMLPath = this->scenePath + "/scene_onlyDiff.xml";
	auto result = doc.load_file(sceneXMLPath.c_str());
	if (!result) {
		throw std::runtime_error("pugixml打开文件失败");
	}
	pugi::xml_node scene = doc.document_element();	//获取根节点，即<scene>
	for (pugi::xml_node node : scene.children("sensor")) {	//这里可能有问题，我现在默认存在这些属性，且相机是透视投影的。有问题之后再说吧
		float fov = glm::radians(std::stof(node.select_node(".//float[@name='fov']").node().attribute("value").value()));
		bool isPerspective = std::string(node.attribute("type").value()) == "perspective" ? true : false;
		pugi::xml_node tranform = node.select_node(".//transform[@name='to_world']").node();
		glm::mat4 inverseViewMatrix = fzbGetMat4FromString(tranform.child("matrix").attribute("value").value());
		glm::vec4 position = inverseViewMatrix[3];
		VkExtent2D resolution = FzbRenderer::globalData.getResolution();
		float aspect = (float)resolution.width / resolution.height;
		float nearPlane = 0.1f;
		float farPlane = 100.0f;
		fov = 2.0f * atanf(tanf(fov * 0.5f) / aspect);	//三叶草中给的fov是水平方向的，而glm中要的是垂直方向的
		FzbCamera camera = FzbCamera(position, fov, aspect, nearPlane, farPlane);
		camera.setViewMatrix(inverseViewMatrix, true);
		camera.isPerspective = isPerspective;
		this->sceneCameras.push_back(camera);
	}

	pugi::xml_node bsdfsNode = scene.child("bsdfs");
	for (pugi::xml_node bsdfNode : bsdfsNode.children("bsdf")) {

		FzbMaterial material = FzbMaterial();
		material.id = bsdfNode.attribute("id").value();
		if (sceneMaterials.count(material.id)) {
			std::cout << "重复material读取" << material.id << std::endl;
			continue;
		}

		pugi::xml_node bsdf = bsdfNode.child("bsdf");
		std::string materialType = bsdf.attribute("type").value();
		material.type = materialType;	// == "diffuse" ? FZB_DIFFUSE : FZB_ROUGH_CONDUCTOR;
		//std::string shaderPath = getRootPath() + "/shaders/forward/diffuse";
		if (materialType == "diffuse") {
			if (pugi::xml_node textureNode = bsdf.select_node(".//texture[@name='reflectance']").node()) {
				std::string texturePath = textureNode.select_node(".//string[@name='filename']").node().attribute("value").value();
				std::string filterType = textureNode.select_node(".//string[@name='filter_type']").node().attribute("value").value();
				VkFilter filter = filterType == "bilinear" ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
				material.properties.textureProperties.insert({ "albedoMap", FzbTexture(texturePath, filter) });
			}
			if (pugi::xml_node rgbNode = bsdf.select_node(".//rgb[@name='reflectance']").node()) {
				glm::vec3 albedo = fzbGetRGBFromString(rgbNode.attribute("value").value());
				FzbNumberProperty numProperty(glm::vec4(albedo, 0.0f));
				material.properties.numberProperties.insert({ "albedo", numProperty });
			}
			//shaderPath = getRootPath() + "/shaders/forward/diffuse";
		}
		//else if (bsdf.attribute("type").value() == "roughconductor") {
		//	if (pugi::xml_node roughness = bsdf.select_node("//float[@name='alpha']").node()) {
		//		material.roughness = std::stof(roughness.attribute("value").value());
		//	}
		//	if (pugi::xml_node distribution = bsdf.select_node("//string[@name='distribution']").node()) {
		//		material.distribution = distribution.attribute("value").value() == "ggx" ? FZB_GGX : FZB_BECKMENN;
		//	}
		//	if (pugi::xml_node spcularReflectanceTexture = bsdf.select_node("//texture[@name='specular_reflectance']").node()) {
		//
		//	}
		//}
		this->sceneMaterials.insert({ material.id, material });
		//if (!this->sceneShaders.count(shaderPath)) {
		//	this->sceneShaders.insert({ shaderPath, FzbShader(logicalDevice, shaderPath) });
		//}
		//this->sceneShaders[shaderPath].createShaderVariant(&sceneMaterials[material.id]);
	}
	for (auto& pair : sceneMaterials) {
		FzbMaterial& material = pair.second;
		std::string shaderPath = fzbGetRootPath() + shaderPaths.at(material.type);
		if (!this->sceneShaders.count(shaderPath)) {
			this->sceneShaders.insert({ shaderPath, FzbShader(shaderPath) });
		}
		this->sceneShaders[shaderPath].createShaderVariant(&sceneMaterials[material.id], vertexFormat);
	}

	pugi::xml_node shapesNode = scene.child("shapes");
	for (pugi::xml_node shapeNode : shapesNode.children("shape")) {
		std::string meshType = shapeNode.attribute("type").value();
		if (meshType == "rectangle" || meshType == "emitter") {
			continue;
		}
		std::string meshID = shapeNode.attribute("id").value();
		pugi::xml_node tranform = shapeNode.select_node(".//transform[@name='to_world']").node();

		if (meshID == "Light") {
			glm::mat4 modelMatrix = fzbGetMat4FromString(tranform.child("matrix").attribute("value").value());
			glm::vec3 position = glm::vec3(modelMatrix * glm::vec4(1.0f));
			pugi::xml_node emitter = shapeNode.child("emitter");		//不知道会不会有多个发射器，默认1个，以后遇到了再说
			glm::vec3 strength = fzbGetRGBFromString(emitter.select_node(".//rgb[@name='radiance']").node().attribute("value").value());
			if (meshType == "point") {
				this->sceneLights.push_back(FzbLight(position, strength));
			}
			continue;
		}

		pugi::xml_node material = shapeNode.select_node(".//ref").node();
		std::string materialID = material.attribute("id").value();
		if (!sceneMaterials.count(materialID)) {
			materialID = "defaultMaterial";
		}
		glm::mat4 modelMatrix(1.0f);
		if (tranform) modelMatrix = fzbGetMat4FromString(tranform.child("matrix").attribute("value").value());

		std::vector<FzbMesh> meshes;
		if (meshType == "obj") {
			if (pugi::xml_node pathNode = shapeNode.select_node(".//string[@name='filename']").node()) {
				std::string objPath = this->scenePath + "/" + std::string(pathNode.attribute("value").value());
				meshes = fzbGetMeshFromOBJ(objPath, sceneMaterials[materialID].vertexFormat, modelMatrix);
			}
			else {
				throw std::runtime_error("obj文件没有路径");
			}
		}
		else if (meshType == "rectangle") {
			continue;
			meshes.resize(1);
			fzbCreateRectangle(meshes[0].vertices, meshes[0].indices, false);
		}

		for (int i = 0; i < meshes.size(); i++) {
			meshes[i].material = &sceneMaterials[materialID];
			meshes[i].id = meshID;
			this->differentVertexFormatMeshIndexs[meshes[i].vertexFormat].push_back(this->sceneMeshSet.size());
			this->sceneMeshSet.push_back(meshes[i]);
		}
	}
	doc.reset();
}
void FzbScene::createShaderMeshBatch() {
	for (auto& pair : sceneShaders) {
		FzbShader& shader = pair.second;
		shader.createMeshBatch(sceneMeshSet);
		this->sceneShaders_vector.push_back(&shader);
	}
}
void FzbScene::createBufferAndDescriptorOfMaterialAndMesh() {
	uint32_t textureNum = 0;
	uint32_t numberBufferNum = 0;
	for (auto& materialPair : this->sceneMaterials) {
		FzbMaterial& material = materialPair.second;
		material.createSource(scenePath, numberBufferNum, sceneImages);
		/*
		if (material.properties.numberProperties.size() > 0) {
			material.createMaterialNumberPropertiesBuffer(physicalDevice, numberBufferNum);
		}
		for (auto& texturePair : material.properties.textureProperties) {
			std::string texturePath = texturePair.second.path;
			if (this->sceneImages.count(texturePath)) {
				continue;
			}
			FzbImage image;
			std::string texturePathFromModel = scenePath + "/" + texturePath;	//./models/xxx/textures/textureName.jpg
			image.texturePath = texturePathFromModel.c_str();
			image.filter = texturePair.second.filter;
			image.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			image.fzbCreateImage(physicalDevice, logicalDevice, commandPool, graphicsQueue);
			this->sceneImages.insert({ texturePath, image });
		}
		*/
	}
	textureNum = this->sceneImages.size();
	numberBufferNum += sceneCameras.size() > 0 ? 1 : 0 + sceneLights.size() > 0 ? 1 : 0;
	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum = { {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textureNum},		//纹理信息
																{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, numberBufferNum} };	//材质信息
	this->sceneDescriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);
	for (auto& shader : sceneShaders) {
		shader.second.createDescriptor(sceneDescriptorPool, sceneImages);
	}
	//fzbCreateMeshDescriptor();	//这里应该加个判断，后面再说吧
}
void FzbScene::createVertexBuffer(bool compress, bool useExternal) {	//目前只处理静态mesh
	std::vector<float> sceneVertices;
	std::vector<uint32_t> sceneIndices;
	uint32_t FzbVertexByteSize = 0;
	for (auto& pair : differentVertexFormatMeshIndexs) {
		FzbVertexFormat vertexFormat = pair.first;
		uint32_t vertexSize = vertexFormat.getVertexSize();

		std::vector<uint32_t> meshIndexs = pair.second;
		uint32_t vertexNum = 0;
		uint32_t indexNum = 0;
		for (int i = 0; i < meshIndexs.size(); i++) {
			uint32_t meshIndex = meshIndexs[i];
			vertexNum += sceneMeshSet[meshIndex].vertices.size();
			indexNum += sceneMeshSet[meshIndex].indices.size();
		}

		std::vector<float> compressVertices;
		compressVertices.reserve(vertexNum);
		std::vector<uint32_t> compressIndices;
		compressIndices.reserve(indexNum);
		for (int i = 0; i < meshIndexs.size(); i++) {
			uint32_t meshIndex = meshIndexs[i];

			sceneMeshSet[meshIndex].indexArrayOffset = sceneIndices.size() + compressIndices.size();

			//这里用临时数组，因为可能同一个mesh被添加多次，如果每次都直接对mesh的indices加上offset就会出错，所以需要使用临时数据
			//同时这不会影响meshBatch的indexBuffer创建，因此indices的大小和在sceneIndices中的偏移都不变。
			std::vector<uint32_t> meshIndices = sceneMeshSet[meshIndex].indices;
			addData_uint(meshIndices.data(), compressVertices.size() / vertexSize, meshIndices.size());
			/*
			for (int j = 0; j < meshIndices.size(); j++) {
				meshIndices[j] += compressVertices.size() / (vertexSize / sizeof(float));
			}
			*/

			std::vector<float> meshVertices = sceneMeshSet[meshIndex].getVertices(vertexFormat);
			compressVertices.insert(compressVertices.end(), meshVertices.begin(), meshVertices.end());
			compressIndices.insert(compressIndices.end(), meshIndices.begin(), meshIndices.end());
		}

		if (compress) compressSceneVertices_sharded(compressVertices, vertexFormat, compressIndices);
		
		/*
		while (FzbVertexByteSize % vertexSize > 0) {
			sceneVertices.push_back(0.0f);
			FzbVertexByteSize += sizeof(float);
		}
		if (FzbVertexByteSize > 0) {
			for (int i = 0; i < compressIndices.size(); i++) {
				compressIndices[i] += FzbVertexByteSize / vertexSize;
			}
		}
		FzbVertexByteSize += compressVertices.size() * sizeof(float);
		*/
		addPadding(sceneVertices, compressVertices, compressIndices, FzbVertexByteSize, vertexFormat);

		sceneVertices.insert(sceneVertices.end(), compressVertices.begin(), compressVertices.end());
		sceneIndices.insert(sceneIndices.end(), compressIndices.begin(), compressIndices.end());
	}

	vertexBuffer = fzbCreateStorageBuffer(sceneVertices.data(), sceneVertices.size() * sizeof(float), useExternal);
	indexBuffer = fzbCreateStorageBuffer(sceneIndices.data(), sceneIndices.size() * sizeof(uint32_t), useExternal);
}
void FzbScene::createVertexPairDataBuffer(bool usePosData, bool usePosNormalData, bool compress, bool usePosDataExternal, bool usePosNormalDataExternal) {
	if (usePosData) {
		std::vector<float> sceneVertices;
		std::vector<uint32_t> sceneIndices;
		uint32_t sceneVerticesFloatNum = 0;
		uint32_t sceneIndicesNum = 0;
		for (int i = 0; i < this->sceneMeshSet.size(); i++) {
			FzbMesh& mesh = this->sceneMeshSet[i];
			sceneVerticesFloatNum += mesh.vertices.size() / mesh.vertexFormat.getVertexSize();
			sceneIndicesNum += mesh.indices.size();
		}
		sceneVertices.reserve(sceneVerticesFloatNum);
		sceneIndices.reserve(sceneIndicesNum);
		for (auto& meshPair : this->differentVertexFormatMeshIndexs) {
			for (int i = 0; i < meshPair.second.size(); i++) {
				FzbMesh& mesh = this->sceneMeshSet[meshPair.second[i]];
				std::vector<float> meshVertices = mesh.getVertices(FzbVertexFormat());
				std::vector<uint32_t> meshIndices = mesh.indices;
				addData_uint(meshIndices.data(), sceneVertices.size() / 3, meshIndices.size());
				/*
				for (int j = 0; j < meshIndices.size(); j++) {
					meshIndices[j] += sceneVertices.size() / 3;
				}
				*/
				sceneVertices.insert(sceneVertices.end(), meshVertices.begin(), meshVertices.end());
				sceneIndices.insert(sceneIndices.end(), meshIndices.begin(), meshIndices.end());
			}
		}
		if (compress) compressSceneVertices_sharded(sceneVertices, FzbVertexFormat(), sceneIndices);
		vertexPosBuffer = fzbCreateStorageBuffer(sceneVertices.data(), sceneVertices.size() * sizeof(float), usePosDataExternal);
		indexPosBuffer = fzbCreateStorageBuffer(sceneIndices.data(), sceneIndices.size() * sizeof(uint32_t), usePosDataExternal);
	}
	if (usePosNormalData) {
		std::vector<float> sceneVertices;
		std::vector<uint32_t> sceneIndices;
		uint32_t sceneVerticesFloatNum = 0;
		uint32_t sceneIndicesNum = 0;
		for (int i = 0; i < this->sceneMeshSet.size(); i++) {
			FzbMesh& mesh = this->sceneMeshSet[i];
			sceneVerticesFloatNum += mesh.vertices.size() / mesh.vertexFormat.getVertexSize();
			sceneIndicesNum += mesh.indices.size();
		}
		sceneVertices.reserve(sceneVerticesFloatNum);
		sceneIndices.reserve(sceneIndicesNum);
		for (auto& meshPair : this->differentVertexFormatMeshIndexs) {
			for (int i = 0; i < meshPair.second.size(); i++) {
				FzbMesh& mesh = this->sceneMeshSet[meshPair.second[i]];
				std::vector<float> meshVertices = mesh.getVertices(FzbVertexFormat(true));
				std::vector<uint32_t> meshIndices = mesh.indices;
				addData_uint(meshIndices.data(), sceneVertices.size() / 6, meshIndices.size());
				/*
				for (int j = 0; j < meshIndices.size(); j++) {
					meshIndices[j] += sceneVertices.size() / 6;
				}
				*/
				sceneVertices.insert(sceneVertices.end(), meshVertices.begin(), meshVertices.end());
				sceneIndices.insert(sceneIndices.end(), meshIndices.begin(), meshIndices.end());
			}
		}
		if (compress) compressSceneVertices_sharded(sceneVertices, FzbVertexFormat(true), sceneIndices);
		vertexPosNormalBuffer = fzbCreateStorageBuffer(sceneVertices.data(), sceneVertices.size() * sizeof(float), usePosNormalDataExternal);
		indexPosNormalBuffer = fzbCreateStorageBuffer(sceneIndices.data(), sceneIndices.size() * sizeof(uint32_t), usePosNormalDataExternal);
	}
}

void FzbScene::createCameraAndLightBufferAndDescriptor() {
	if (this->sceneCameras.size() == 0 && this->sceneLights.size() == 0) return;
	bool useCameras = this->sceneCameras.size();
	bool useLights = this->sceneLights.size();
	std::vector<VkDescriptorType> descriptorTypes;
	std::vector<VkShaderStageFlags> descriptorShaderFlags;

	//camera的数据每帧都会变化，所以这里只是创建buffer，但是不填入数据，具体数据由渲染组件填入
	if (useCameras) {
		cameraBuffer = fzbCreateUniformBuffers(sizeof(FzbCameraUniformBufferObject));
		descriptorTypes.push_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
		descriptorShaderFlags.push_back(VK_SHADER_STAGE_ALL);
	}

	if (useLights) {
		lightsBuffer = fzbCreateUniformBuffers(sizeof(FzbLightsUniformBufferObject));
		FzbLightsUniformBufferObject lightsData(sceneLights.size());
		for (int i = 0; i < sceneLights.size(); i++) {
			FzbLight& light = sceneLights[i];
			lightsData.lightData[i].pos = glm::vec4(light.position, 1.0f);
			lightsData.lightData[i].strength = glm::vec4(light.strength, 1.0f);
		}
		memcpy(lightsBuffer.mapped, &lightsData, sizeof(FzbLightsUniformBufferObject));
		descriptorTypes.push_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
		descriptorShaderFlags.push_back(VK_SHADER_STAGE_ALL);
	}

	cameraAndLightsDescriptorSetLayout = fzbCreateDescriptLayout(descriptorTypes.size(), descriptorTypes, descriptorShaderFlags);
	cameraAndLightsDescriptorSet = fzbCreateDescriptorSet(sceneDescriptorPool, cameraAndLightsDescriptorSetLayout);

	std::vector<VkWriteDescriptorSet> uniformDescriptorWrites(sceneCameras.size() + sceneLights.size());
	int index = 0;
	if (useCameras) {
		VkDescriptorBufferInfo cameraUniformBufferInfo{};
		cameraUniformBufferInfo.buffer = cameraBuffer.buffer;
		cameraUniformBufferInfo.offset = 0;
		cameraUniformBufferInfo.range = sizeof(FzbCameraUniformBufferObject);
		uniformDescriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		uniformDescriptorWrites[index].dstSet = cameraAndLightsDescriptorSet;
		uniformDescriptorWrites[index].dstBinding = 0;
		uniformDescriptorWrites[index].dstArrayElement = 0;
		uniformDescriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uniformDescriptorWrites[index].descriptorCount = 1;
		uniformDescriptorWrites[index].pBufferInfo = &cameraUniformBufferInfo;
		++index;
	}

	if (useLights) {
		VkDescriptorBufferInfo lightUniformBufferInfo{};
		lightUniformBufferInfo.buffer = lightsBuffer.buffer;
		lightUniformBufferInfo.offset = 0;
		lightUniformBufferInfo.range = sizeof(FzbLightsUniformBufferObject);
		uniformDescriptorWrites[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		uniformDescriptorWrites[index].dstSet = cameraAndLightsDescriptorSet;
		uniformDescriptorWrites[index].dstBinding = 1;
		uniformDescriptorWrites[index].dstArrayElement = 0;
		uniformDescriptorWrites[index].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uniformDescriptorWrites[index].descriptorCount = 1;
		uniformDescriptorWrites[index].pBufferInfo = &lightUniformBufferInfo;
	}

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, uniformDescriptorWrites.size(), uniformDescriptorWrites.data(), 0, nullptr);
}
void FzbScene::updateCameraBuffer() {
	FzbCameraUniformBufferObject ubo{};
	FzbCamera& camera = this->sceneCameras[0];
	ubo.view = camera.GetViewMatrix();
	ubo.proj = camera.GetProjMatrix();
	ubo.cameraPos = glm::vec4(camera.position, 0.0f);
	memcpy(cameraBuffer.mapped, &ubo, sizeof(ubo));
}

FzbAABBBox FzbScene::getAABB() {
	if (this->AABB.isEmpty()) this->createAABB();
	return this->AABB;
}
void FzbScene::createAABB() {
	this->AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	for (int i = 0; i < this->sceneMeshSet.size(); i++) {
		FzbAABBBox meshAABB = sceneMeshSet[i].getAABB();
		AABB.leftX = meshAABB.leftX < AABB.leftX ? meshAABB.leftX : AABB.leftX;
		AABB.rightX = meshAABB.rightX > AABB.rightX ? meshAABB.rightX : AABB.rightX;
		AABB.leftY = meshAABB.leftY < AABB.leftY ? meshAABB.leftY : AABB.leftY;
		AABB.rightY = meshAABB.rightY > AABB.rightY ? meshAABB.rightY : AABB.rightY;
		AABB.leftZ = meshAABB.leftZ < AABB.leftZ ? meshAABB.leftZ : AABB.leftZ;
		AABB.rightZ = meshAABB.rightZ > AABB.rightZ ? meshAABB.rightZ : AABB.rightZ;
	}
	//对于面，我们给个0.02的宽度
	if (AABB.leftX == AABB.rightX) {
		AABB.leftX = AABB.leftX - 0.01;
		AABB.rightX = AABB.rightX + 0.01;
	}
	if (AABB.leftY == AABB.rightY) {
		AABB.leftY = AABB.leftY - 0.01;
		AABB.rightY = AABB.rightY + 0.01;
	}
	if (AABB.leftZ == AABB.rightZ) {
		AABB.leftZ = AABB.leftZ - 0.01;
		AABB.rightZ = AABB.rightZ + 0.01;
	}
}