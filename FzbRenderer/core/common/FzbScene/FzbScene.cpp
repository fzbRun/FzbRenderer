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
void FzbScene::clean() {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;

	for (auto& imagePair : this->sceneImages) imagePair.second.clean();
	for (auto& materialPair : sceneMaterials) materialPair.second.clean();
	for (int i = 0; i < this->sceneMeshSet.size(); i++) sceneMeshSet[i].clean();

	if(FzbMeshDynamic::meshDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, FzbMeshDynamic::meshDescriptorSetLayout, nullptr);

	vertexBuffer.clean();
	indexBuffer.clean();
}

/*
void FzbScene::addSceneFromMitsubaXML(std::string path, FzbVertexFormat vertexFormat) {
	if (path == "") return;
	this->scenePath = fzbGetRootPath() + path;
	FzbMaterial defaultMaterial = FzbMaterial("defaultMaterial", "diffuse");
	defaultMaterial.vertexFormat.mergeUpward(vertexFormat);
	this->sceneMaterials.insert({ "defaultMaterial", defaultMaterial });

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
		material.type = materialType;
		material.getMaterialXMLInfo();

		if (materialType == "diffuse") {
			material.getSceneXMLInfo(bsdf);
		}

		material.vertexFormat.mergeUpward(vertexFormat);
		this->sceneMaterials.insert({ material.id, material });
	}

	pugi::xml_node shapesNode = scene.child("shapes");
	for (pugi::xml_node shapeNode : shapesNode.children("shape")) {
		std::string meshType = shapeNode.attribute("type").value();
		if (meshType == "rectangle" || meshType == "emitter") continue;
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
		glm::mat4 transformMatrix(1.0f);
		if (tranform) transformMatrix = fzbGetMat4FromString(tranform.child("matrix").attribute("value").value());

		std::vector<FzbMesh> meshes;
		if (meshType == "obj") {
			if (pugi::xml_node pathNode = shapeNode.select_node(".//string[@name='filename']").node()) {
				std::string objPath = this->scenePath + "/" + std::string(pathNode.attribute("value").value());
				meshes = fzbGetMeshFromOBJ(objPath, sceneMaterials[materialID].vertexFormat, transformMatrix);
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
*/
void FzbScene::createVertexBuffer(bool compress, bool useExternal) {	//目前只处理静态mesh
	std::vector<float> sceneVertices;
	std::vector<uint32_t> sceneIndices;
	uint32_t FzbVertexByteSize = 0;
	for (auto& pair : this->differentVertexFormatMeshIndexs) {
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

		addPadding(sceneVertices, compressVertices, compressIndices, FzbVertexByteSize, vertexFormat);

		sceneVertices.insert(sceneVertices.end(), compressVertices.begin(), compressVertices.end());
		sceneIndices.insert(sceneIndices.end(), compressIndices.begin(), compressIndices.end());
	}

	vertexBuffer = fzbCreateStorageBuffer(sceneVertices.data(), sceneVertices.size() * sizeof(float), useExternal);
	indexBuffer = fzbCreateStorageBuffer(sceneIndices.data(), sceneIndices.size() * sizeof(uint32_t), useExternal);
}
/*
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
				
				//for (int j = 0; j < meshIndices.size(); j++) {
				//	meshIndices[j] += sceneVertices.size() / 3;
				//}
				
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
				
				//for (int j = 0; j < meshIndices.size(); j++) {
				//	meshIndices[j] += sceneVertices.size() / 6;
				//}
				
				sceneVertices.insert(sceneVertices.end(), meshVertices.begin(), meshVertices.end());
				sceneIndices.insert(sceneIndices.end(), meshIndices.begin(), meshIndices.end());
			}
		}
		if (compress) compressSceneVertices_sharded(sceneVertices, FzbVertexFormat(true), sceneIndices);
		vertexPosNormalBuffer = fzbCreateStorageBuffer(sceneVertices.data(), sceneVertices.size() * sizeof(float), usePosNormalDataExternal);
		indexPosNormalBuffer = fzbCreateStorageBuffer(sceneIndices.data(), sceneIndices.size() * sizeof(uint32_t), usePosNormalDataExternal);
	}
}
*/
void FzbScene::addMeshToScene(FzbMesh mesh) {
	if(mesh.material) mesh.vertexFormat = mesh.material->vertexFormat;
	this->differentVertexFormatMeshIndexs[mesh.vertexFormat].push_back(this->sceneMeshSet.size());
	this->sceneMeshSet.push_back(mesh);
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

FzbMainScene::FzbMainScene() {};
FzbMainScene::FzbMainScene(std::string path) {
	if (path == "") return;
	this->scenePath = fzbGetRootPath() + path;
	FzbMaterial defaultMaterial = FzbMaterial("defaultMaterial", "diffuse");
	this->sceneMaterials.insert({ "defaultMaterial", defaultMaterial });

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
		std::string materialID = bsdfNode.attribute("id").value();
		if (this->sceneMaterials.count(materialID)) {
			std::cout << "重复material读取" << materialID << std::endl;
			continue;
		}
		pugi::xml_node bsdf = bsdfNode.child("bsdf");
		std::string materialType = bsdf.attribute("type").value();
		FzbMaterial material = FzbMaterial(materialID, materialType);
		material.getMaterialXMLInfo();
		material.getSceneXMLInfo(bsdfNode);
		this->sceneMaterials.insert({ material.id, material });
	}

	pugi::xml_node shapesNode = scene.child("shapes");
	for (pugi::xml_node shapeNode : shapesNode.children("shape")) {
		std::string meshType = shapeNode.attribute("type").value();
		if (meshType == "rectangle" || meshType == "emitter") continue;
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
		if (!sceneMaterials.count(materialID)) materialID = "defaultMaterial";
		glm::mat4 transformMatrix(1.0f);
		if (tranform) transformMatrix = fzbGetMat4FromString(tranform.child("matrix").attribute("value").value());

		FzbMesh mesh;
		if (meshType == "obj") {
			if (pugi::xml_node pathNode = shapeNode.select_node(".//string[@name='filename']").node()) {
				mesh.path = this->scenePath + "/" + std::string(pathNode.attribute("value").value());
			}
			else throw std::runtime_error("obj文件没有路径");
		}
		else if (meshType == "rectangle") continue;
		mesh.transformMatrix = transformMatrix;
		mesh.material = &sceneMaterials[materialID];
		//mesh.vertexFormat = mesh.material->vertexFormat;
		//mesh.vertexFormat_propocess.available = false;
		mesh.id = meshID;
		this->sceneMeshSet.push_back(mesh);
	}
	doc.reset();
}
void FzbMainScene::initScene(bool compress, bool isMainScene) {
	this->sceneMaterials["defaultMaterial"].vertexFormat.mergeUpward(vertexFormat_allMesh);
	for (int i = 0; i < this->sceneMeshSet.size(); ++i) {
		FzbMesh& mesh_nodata = this->sceneMeshSet[i];
		mesh_nodata.vertexFormat.mergeUpward(mesh_nodata.material->vertexFormat);
		mesh_nodata.vertexFormat.mergeUpward(vertexFormat_allMesh);
	}

	bool useVertexBufferForPrepocess = this->useVertexBuffer_prepocess;
	if (useVertexBufferForPrepocess) {		//如果预处理组件所需顶点数据格式与循环渲染组件所需顶点数据格式不同，则需要创建预处理顶点缓冲
		useVertexBufferForPrepocess = false;
		for (int i = 0; i < this->sceneMeshSet.size(); ++i) {
			FzbMesh& mesh_nodata = this->sceneMeshSet[i];
			if (mesh_nodata.vertexFormat != this->vertexFormat_allMesh_prepocess) {
				useVertexBufferForPrepocess = true;
				break;
			}
		}
	}

	std::vector<FzbMesh> sceneMeshes_hasData; sceneMeshes_hasData.reserve(this->sceneMeshSet.size());
	for (int i = 0; i < this->sceneMeshSet.size(); ++i) {
		FzbMesh& mesh_nodata = this->sceneMeshSet[i];
		std::vector<FzbMesh> meshes_hasData = fzbGetMeshFromOBJ(mesh_nodata.path, fzbVertexFormatMergeUpward(mesh_nodata.vertexFormat, vertexFormat_allMesh_prepocess), mesh_nodata.transformMatrix);
		for (int j = 0; j < meshes_hasData.size(); ++j) {
			FzbMesh& mesh_hasData = meshes_hasData[j];
			mesh_hasData.id = mesh_nodata.id;
			mesh_hasData.path = mesh_nodata.path;
			mesh_hasData.vertexFormat = mesh_nodata.vertexFormat;	//只反应渲染循环组件所需的顶点数据的格式
			mesh_hasData.material = mesh_nodata.material;
			mesh_hasData.material->vertexFormat = mesh_hasData.vertexFormat;

			uint32_t index = (uint32_t)sceneMeshes_hasData.size() + j;
			if (differentVertexFormatMeshIndexs.count(mesh_hasData.vertexFormat)) differentVertexFormatMeshIndexs[mesh_hasData.vertexFormat].push_back(index);
			else differentVertexFormatMeshIndexs[mesh_hasData.vertexFormat] = { index };
		}
		sceneMeshes_hasData.insert(sceneMeshes_hasData.end(), meshes_hasData.begin(), meshes_hasData.end());
	}
	this->sceneMeshSet = sceneMeshes_hasData;

	//不使用分两种情况：1. 预处理确实需要不同顶点数据；2. 预处理所需顶点数据与渲染循环所需顶点数据相同，则共用一个
	if (useVertexBufferForPrepocess) {
		createVertexBuffer(compress, useVertexBufferHandle);
		createVertexBuffer_prepocess(compress, useVertexBufferHandle_preprocess);
	}
	else if (this->useVertexBuffer_prepocess) {
		createVertexBuffer(compress, useVertexBufferHandle_preprocess || useVertexBufferHandle);
		this->vertexBuffer_prepocess = vertexBuffer;
		this->indexBuffer_prepocess = indexBuffer;
	}else createVertexBuffer(compress, useVertexBufferHandle);
	useVertexBuffer_prepocess = useVertexBufferForPrepocess;

	if(this->useMaterialSource) createMaterialSource();
	createCameraAndLightBuffer();
}
void FzbMainScene::createVertexBuffer_prepocess(bool compress, bool useExternal) {
	std::vector<uint32_t> meshIndexs; meshIndexs.reserve(this->sceneMeshSet.size());
	for (auto& pair : differentVertexFormatMeshIndexs) {
		for (int i = 0; i < pair.second.size(); ++i) meshIndexs.push_back(pair.second[i]);
	}
	FzbVertexFormat vertexFormat = vertexFormat_allMesh_prepocess;
	uint32_t vertexSize = vertexFormat.getVertexSize();

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
		//sceneMeshSet[meshIndex].indexArrayOffset = compressIndices.size();

		//这里用临时数组，因为可能同一个mesh被添加多次，如果每次都直接对mesh的indices加上offset就会出错，所以需要使用临时数据
		//同时这不会影响meshBatch的indexBuffer创建，因此indices的大小和在sceneIndices中的偏移都不变。
		std::vector<uint32_t> meshIndices = sceneMeshSet[meshIndex].indices;
		addData_uint(meshIndices.data(), compressVertices.size() / vertexSize, meshIndices.size());

		std::vector<float> meshVertices = sceneMeshSet[meshIndex].getVertices(vertexFormat);
		compressVertices.insert(compressVertices.end(), meshVertices.begin(), meshVertices.end());
		compressIndices.insert(compressIndices.end(), meshIndices.begin(), meshIndices.end());
	}
	if (compress) compressSceneVertices_sharded(compressVertices, vertexFormat, compressIndices);

	vertexBuffer_prepocess = fzbCreateStorageBuffer(compressVertices.data(), compressVertices.size() * sizeof(float), useExternal);
	indexBuffer_prepocess = fzbCreateStorageBuffer(compressIndices.data(), compressIndices.size() * sizeof(uint32_t), useExternal);
}
void FzbMainScene::createMaterialSource() {
	for (auto& materialPair : this->sceneMaterials) {
		FzbMaterial& material = materialPair.second;
		material.createSource(this->scenePath, this->sceneImages);
	}
}
void FzbMainScene::createCameraAndLightBuffer() {
	if (this->sceneCameras.size() == 0 && this->sceneLights.size() == 0) return;
	bool useCameras = this->sceneCameras.size();
	bool useLights = this->sceneLights.size();

	//camera的数据每帧都会变化，所以这里只是创建buffer，但是不填入数据，具体数据由渲染组件填入
	if (useCameras) {
		cameraBuffer = fzbCreateUniformBuffer(sizeof(FzbCameraUniformBufferObject));
	}

	if (useLights) {
		lightsBuffer = fzbCreateUniformBuffer(sizeof(FzbLightsUniformBufferObject));
		FzbLightsUniformBufferObject lightsData(sceneLights.size());
		for (int i = 0; i < sceneLights.size(); i++) {
			FzbLight& light = sceneLights[i];
			lightsData.lightData[i].pos = glm::vec4(light.position, 1.0f);
			lightsData.lightData[i].strength = glm::vec4(light.strength, 1.0f);
		}
		memcpy(lightsBuffer.mapped, &lightsData, sizeof(FzbLightsUniformBufferObject));
	}
}
void FzbMainScene::createCameraAndLightDescriptor() {
	if (cameraAndLightsDescriptorSet) return;

	uint32_t uniformBufferNum = this->sceneCameras.size() > 0 ? 1 : 0 + this->sceneLights.size() > 0 ? 1 : 0;
	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum = { {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, uniformBufferNum} };
	this->cameraAndLigthsDescriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);


	std::vector<FzbCamera>& sceneCameras = this->sceneCameras;
	std::vector<FzbLight>& sceneLights = this->sceneLights;

	if (sceneCameras.size() == 0 && sceneLights.size() == 0) return;
	bool useCameras = sceneCameras.size();
	bool useLights = sceneLights.size();
	std::vector<VkDescriptorType> descriptorTypes;
	std::vector<VkShaderStageFlags> descriptorShaderFlags;

	//camera的数据每帧都会变化，所以这里只是创建buffer，但是不填入数据，具体数据由渲染组件填入
	if (useCameras) {
		descriptorTypes.push_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
		descriptorShaderFlags.push_back(VK_SHADER_STAGE_ALL);
	}

	if (useLights) {
		descriptorTypes.push_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
		descriptorShaderFlags.push_back(VK_SHADER_STAGE_ALL);
	}

	cameraAndLightsDescriptorSetLayout = fzbCreateDescriptLayout(descriptorTypes.size(), descriptorTypes, descriptorShaderFlags);
	cameraAndLightsDescriptorSet = fzbCreateDescriptorSet(cameraAndLigthsDescriptorPool, cameraAndLightsDescriptorSetLayout);

	std::vector<VkWriteDescriptorSet> uniformDescriptorWrites(sceneCameras.size() + sceneLights.size());
	int index = 0;
	if (useCameras) {
		VkDescriptorBufferInfo cameraUniformBufferInfo{};
		cameraUniformBufferInfo.buffer = this->cameraBuffer.buffer;
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
		lightUniformBufferInfo.buffer = this->lightsBuffer.buffer;
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
void FzbMainScene::updateCameraBuffer() {
	FzbCameraUniformBufferObject ubo{};
	FzbCamera& camera = this->sceneCameras[0];
	ubo.view = camera.GetViewMatrix();
	ubo.proj = camera.GetProjMatrix();
	ubo.cameraPos = glm::vec4(camera.position, 0.0f);
	memcpy(cameraBuffer.mapped, &ubo, sizeof(ubo));
}
void FzbMainScene::addMaterialToScene(FzbMaterial material) {
	this->sceneMaterials.insert({ material.id, material });
}

void FzbMainScene::clean() {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;

	for (auto& imagePair : this->sceneImages) imagePair.second.clean();
	for (auto& materialPair : sceneMaterials) materialPair.second.clean();
	for (int i = 0; i < this->sceneMeshSet.size(); i++) sceneMeshSet[i].clean();

	if (FzbMeshDynamic::meshDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, FzbMeshDynamic::meshDescriptorSetLayout, nullptr);

	vertexBuffer.clean();
	indexBuffer.clean();
	//vertexBuffer_prepocess会被componentManager在mainLoop前释放

	cameraBuffer.clean();
	lightsBuffer.clean();
	if (cameraAndLightsDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, cameraAndLightsDescriptorSetLayout, nullptr);
	if (cameraAndLigthsDescriptorPool) vkDestroyDescriptorPool(logicalDevice, cameraAndLigthsDescriptorPool, nullptr);
}