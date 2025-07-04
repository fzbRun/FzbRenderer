#pragma once

#include <glm/ext/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <pugixml/src/pugixml.hpp>

#include <optional>
#include <iostream>
#include <unordered_map>
#include <sstream>

#include "FzbMesh.h"
#include "FzbImage.h"
#include "FzbDescriptor.h"
#include <regex>

#ifndef FZB_SCENE_H
#define FZB_SCENE_H

//------------------------------------------------------辅助函数------------------------------------------------
std::vector<std::string> get_all_files(const std::string& dir_path) {
	std::vector<std::string> filePaths;
	for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
		if (entry.is_regular_file()) {
			filePaths.push_back(entry.path().string());
		}
	}
	return filePaths;
}

glm::vec3 getRGBFromString(std::string str) {
	std::vector<float> float3_array;
	std::stringstream ss(str);
	std::string token;
	while (std::getline(ss, token, ',')) {
		float3_array.push_back(std::stof(token));
	}
	return glm::vec3(float3_array[0], float3_array[1], float3_array[2]);
}

glm::mat4 getMat4FromString(std::string str) {
	std::vector<float> mat4_array;
	std::stringstream ss(str);
	std::string token;
	while (std::getline(ss, token, ' ')) {
		mat4_array.push_back(std::stof(token));
	}
	return glm::mat4(mat4_array[0], mat4_array[4], mat4_array[8], mat4_array[12],
		mat4_array[1], mat4_array[5], mat4_array[9], mat4_array[13],
		mat4_array[2], mat4_array[6], mat4_array[10], mat4_array[14],
		mat4_array[3], mat4_array[7], mat4_array[11], mat4_array[15]);
}

//-----------------------------------------------------------------------------------------------------------------
struct VectorFloatHash {
	size_t operator()(std::vector<float> const& v) const noexcept {
		// 以向量长度作为初始种子
		size_t seed = v.size();
		for (float f : v) {
			// 标准库的 float 哈希
			size_t h = std::hash<float>()(f);
			// 经典的 hash_combine
			seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};

struct Mat4Hash {
	size_t operator()(glm::mat4 const& m) const noexcept {
		// 把 16 个 float 串起来哈希
		size_t seed = 16;
		for (int col = 0; col < 4; ++col) {
			for (int row = 0; row < 4; ++row) {
				size_t h = std::hash<float>()(m[col][row]);
				seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
			}
		}
		return seed;
	}
};

struct Mat4Equal {
	bool operator()(glm::mat4 const& a, glm::mat4 const& b) const noexcept {
		// 如果你只想严格逐元素相等，可直接用 ==  
		// 但为了保险，也可以手动比：
		for (int col = 0; col < 4; ++col)
			for (int row = 0; row < 4; ++row)
				if (a[col][row] != b[col][row])
					return false;
		return true;
	}
};

struct FzbVertexFormatLess {
	bool operator()(FzbVertexFormat const& a,
		FzbVertexFormat const& b) const noexcept {
		return a.getVertexSize() < b.getVertexSize();
	}
};

struct FzbScene {

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;

	VkDescriptorPool sceneDescriptorPool;	//包括所有material和mesh的资源

	std::string scenePath;	//存储当前文件夹的地址
	std::vector<FzbLight> sceneLights;
	std::vector<FzbMesh> sceneMeshSet;	//不同顶点格式的mesh
	std::map<FzbVertexFormat, std::vector<uint32_t>, FzbVertexFormatLess> differentVertexFormatMeshIndexs;

	std::vector<float> sceneVertices;	//压缩后的顶点数据
	std::vector<uint32_t> sceneIndices;
	std::map<std::string, FzbMaterial> sceneMaterials;
	std::map<std::string, std::string> sceneShaderPaths;	//第一个string是materialType，第二个string是shader父目录
	std::map<std::string, FzbImage> sceneImages;	//key是texture Path

	FzbBuffer vertexBuffer;
	
	std::vector<FzbImage> images;

	FzbAABBBox AABB;

	FzbScene() {};

	FzbScene(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue) {
		this->physicalDevice = physicalDevice;
		this->logicalDevice = logicalDevice;
		this->commandPool = commandPool;
		this->graphicsQueue = graphicsQueue;

		FzbMaterial defaultMaterial = FzbMaterial(logicalDevice, "defaultMaterial");
		std::string shaderPath = getRootPath() + "/shaders/forward/diffuse";
		defaultMaterial.addShader(shaderPath);
		this->sceneMaterials.insert({ "defaultMaterial", defaultMaterial });
		// 初始化 glslang
		glslang::InitializeProcess();	//初始化GLSLang库，全局初始化，在程序启动时调用一次
	}

	void clean() {
		
		for (auto& materialPair : sceneMaterials) {
			materialPair.second.clean();
		}
		for (auto& imagePair : sceneImages) {
			imagePair.second.clean();
		}
		for (int i = 0; i < this->sceneMeshSet.size(); i++) {
			sceneMeshSet[i].clean();
		}

		vkDestroyDescriptorPool(logicalDevice, sceneDescriptorPool, nullptr);
		
		vertexBuffer.clean();

		glslang::FinalizeProcess();
	}

	void clear() {
		clean();
		sceneMeshSet.clear();
		differentVertexFormatMeshIndexs.clear();
		sceneVertices.clear();
		sceneIndices.clear();
		sceneMaterials.clear();
	}

//--------------------------------------------------------------获取场景-----------------------------------------------------------------

	//std::vector<FzbMesh> getMeshesFromFolder(std::string path, FzbVertexFormat meshVertexFormat = FzbVertexFormat()) {
	//	std::vector<FzbMesh> results;
	//	std::vector<std::string> meshPaths = get_all_files(path);
	//	for (int i = 0; i < meshPaths.size(); i++) {
	//		std::vector<FzbMesh> meshes = fzbGetMeshFromOBJ(meshPaths[i], meshVertexFormat);
	//		results.insert(results.end(), meshes.begin(), meshes.end());
	//	}
	//	return results;
	//}

	void addMeshToScene(FzbMesh mesh, bool reAdd = false) {
		mesh.indexArrayOffset = this->sceneIndices.size();
		mesh.logicalDevice = logicalDevice;

		//bool skip = true;
		//for (int i = 0; i < this->sceneTransformMatrixs.size(); i++) {
		//	if (mesh.transforms == this->sceneTransformMatrixs[i]) {
		//		mesh.materialUniformObject.transformIndex = i;
		//		skip = false;
		//		break;
		//	}
		//}
		//if (skip) {
		//	mesh.materialUniformObject.transformIndex = this->sceneTransformMatrixs.size();
		//	this->sceneTransformMatrixs.push_back(mesh.transforms);
		//}
		//
		//skip = true;
		//for (int i = 0; i < this->sceneMaterials.size(); i++) {
		//	if (mesh.material == this->sceneMaterials[i]) {
		//		mesh.materialUniformObject.materialIndex = i;
		//		skip = false;
		//		break;
		//	}
		//}
		//if (skip) {
		//	mesh.materialUniformObject.materialIndex = this->sceneMaterials.size();
		//	this->sceneMaterials.push_back(mesh.material);
		//}
		//
		//skip = true;
		//if (mesh.albedoTexture.path != "") {
		//	for (int i = 0; i < this->sceneAlbedoTextures.size(); i++) {
		//		if (mesh.albedoTexture == this->sceneAlbedoTextures[i]) {
		//			mesh.materialUniformObject.albedoTextureIndex = i;
		//			skip = false;
		//			break;
		//		}
		//	}
		//	if (skip) {
		//		mesh.materialUniformObject.albedoTextureIndex = this->sceneAlbedoTextures.size();
		//		this->sceneAlbedoTextures.push_back(mesh.albedoTexture);
		//	}
		//}
		//
		//skip = true;
		//if (mesh.normalTexture.path != "") {
		//	for (int i = 0; i < this->sceneNormalTextures.size(); i++) {
		//		if (mesh.normalTexture == this->sceneNormalTextures[i]) {
		//			mesh.materialUniformObject.normalTextureIndex = i;
		//			skip = false;
		//			break;
		//		}
		//
		//	}
		//	if (skip) {
		//		mesh.materialUniformObject.normalTextureIndex = this->sceneNormalTextures.size();
		//		this->sceneNormalTextures.push_back(mesh.normalTexture);
		//	}
		//}

		this->differentVertexFormatMeshIndexs[mesh.vertexFormat].push_back(this->sceneMeshSet.size());
		if(!reAdd) this->sceneMeshSet.push_back(mesh);
	}

	std::string getRootPath() {
		std::filesystem::path thisFile = __FILE__;
		return thisFile.parent_path().parent_path().parent_path().string();	//得到Renderer文件夹
	}

	void addSceneFromMitsubaXML(std::string path) {
		this->scenePath = path;

		pugi::xml_document doc;
		std::string xmlPath = scenePath + "/scene_onlyDiff.xml";
		auto result = doc.load_file(xmlPath.c_str());
		if (!result) { 
			throw std::runtime_error("pugixml打开文件失败");
		}

		pugi::xml_node scene = doc.document_element();	//获取根节点，即<scene>
		for (pugi::xml_node node : scene.children("bsdf")) {

			FzbMaterial material(logicalDevice);
			material.id = node.attribute("id").value();
			if (sceneMaterials.count(material.id)) {
				std::cout << "重复material读取" << material.id << std::endl;
				continue;
			}

			pugi::xml_node bsdf = node.child("bsdf");
			std::string materialType = bsdf.attribute("type").value();
			material.type = materialType;	// == "diffuse" ? FZB_DIFFUSE : FZB_ROUGH_CONDUCTOR;
			if (materialType == "diffuse") {
				if (pugi::xml_node textureNode = bsdf.select_node(".//texture[@name='reflectance']").node()) {
					std::string texturePath = textureNode.select_node(".//string[@name='filename']").node().attribute("value").value();
					std::string filterType = textureNode.select_node(".//string[@name='filter_type']").node().attribute("value").value();
					VkFilter filter = filterType == "bilinear" ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
					material.properties.textureProperties.insert({"albedoMap", FzbTexture(texturePath, filter) });
				}
				if (pugi::xml_node rgbNode = bsdf.select_node(".//rgb[@name='reflectance']").node()) {
					glm::vec3 albedo = getRGBFromString(rgbNode.attribute("value").value());
					FzbNumberProperty numProperty(glm::vec4(albedo, 0.0f));
					material.properties.numberProperties.insert({ "albedo", numProperty });
				}

				std::string shaderPath = getRootPath() + "/shaders/forward/diffuse";
				material.shader.path = shaderPath;
				//material.addShader(shaderPath);
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
		}

		for (auto& materialPair : sceneMaterials) {
			FzbMaterial& material = materialPair.second;
			std::string materialType = material.type;
			if (sceneShaderPaths.count(materialType) == 0) {
				sceneShaderPaths[materialType] = material.shader.path;
				material.addShader(material.shader.path);
			}
			else {
				material.addShader(sceneShaderPaths[materialType]);
			}
		}
		for (pugi::xml_node node : scene.children("shape")) {
			std::string meshType = node.attribute("type").value();
			if (meshType == "rectangle" || meshType == "emitter") {
				continue;
			}
			std::string meshID = node.attribute("id").value();

			pugi::xml_node material = node.select_node(".//ref").node();
			std::string materialID = material.attribute("id").value();
			if (!sceneMaterials.count(materialID)) {
				materialID = "defaultMaterial";
			}

			//在三叶草的xml中将面法线的设置放在了mesh中，而不是material中，我觉得应该放在material中，由此，这里将faceNormal放入material中
			//算了，先不管面法线的事情
			std::vector<FzbMesh> meshes;
			if (meshType == "obj") {
				if (pugi::xml_node pathNode = node.select_node(".//string[@name='filename']").node()) {
					std::string objPath = path + "/" + std::string(pathNode.attribute("value").value());
					meshes = fzbGetMeshFromOBJ(objPath, sceneMaterials[materialID]);
				}
				else {
					throw std::runtime_error("obj文件没有路径");
				}
			}
			else if (meshType == "rectangle") {
				meshes.resize(1);
				fzbCreateRectangle(meshes[0].vertices, meshes[0].indices, false);
			}

			glm::mat4 modelMatrix;
			if (pugi::xml_node tranform = node.select_node(".//transform[@name='to_world']").node()) {
				modelMatrix = getMat4FromString(tranform.child("matrix").attribute("value").value());
			}

			for (int i = 0; i < meshes.size(); i++) {
				meshes[i].transforms = modelMatrix;
				meshes[i].materialID = materialID;
				meshes[i].id = meshID;
				addMeshToScene(meshes[i]);
			}
		}
	}

//----------------------------------------------------------------初始化场景---------------------------------------------------------
	//创建各种buffer、image和描述符
	void createBufferAndDescriptorOfMaterialAndMesh() {
		uint32_t textureNum = 0;
		uint32_t numberBufferNum = 0;
		for (auto& materialPair : this->sceneMaterials) {
			FzbMaterial& material = materialPair.second;
			if (material.properties.numberProperties.size() > 0) {
				material.createMaterialNumberPropertiesBuffer(physicalDevice);
				numberBufferNum++;
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
		}
		for (int i = 0; i < this->sceneMeshSet.size(); i++) {
			this->sceneMeshSet[i].createBuffer(physicalDevice);
		}

		textureNum = this->sceneImages.size();
		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum = { {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, this->sceneMeshSet.size()},	//mesh特有信息
																	{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textureNum},		//纹理信息
																	{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, numberBufferNum} };	//材质信息
		this->sceneDescriptorPool = fzbCreateDescriptorPool(logicalDevice, bufferTypeAndNum);
		for (auto& materialPair : this->sceneMaterials) {
			materialPair.second.createMaterialDescriptor(sceneDescriptorPool, this->sceneImages);
		}
		for (int i = 0; i < this->sceneMeshSet.size(); i++) {
			sceneMeshSet[i].createDescriptor(sceneDescriptorPool);
		}
	}

	void compressSceneVertics(std::vector<float>& vertices, FzbVertexFormat vertexFormat, std::vector<uint32_t>& indices) {
		std::unordered_map<std::vector<float>, uint32_t, VectorFloatHash, std::equal_to<std::vector<float>>> uniqueVerticesMap{};
		std::vector<std::vector<float>> uniqueVertices;
		std::vector<uint32_t> uniqueIndices;

		uint32_t vertexSize = vertexFormat.getVertexSize() / sizeof(float);
		for (uint32_t i = 0; i < indices.size(); i++) {
			uint32_t vertexIndex = indices[i] * vertexSize;
			std::vector<float> vertex;
			vertex.insert(vertex.end(), vertices.begin() + vertexIndex, vertices.begin() + vertexIndex + vertexSize);

			if (uniqueVerticesMap.count(vertex) == 0) {
				uniqueVerticesMap[vertex] = static_cast<uint32_t>(uniqueVertices.size());
				uniqueVertices.push_back(vertex);
			}
			uniqueIndices.push_back(uniqueVerticesMap[vertex]);
		}

		vertices.clear();
		vertices.reserve(vertexSize * uniqueVertices.size());
		for (int i = 0; i < uniqueVertices.size(); i++) {
			vertices.insert(vertices.end(), uniqueVertices[i].begin(), uniqueVertices[i].end());
		}
		indices = uniqueIndices;
	}

	void getSceneVertics(bool compress = true) {	//目前只处理静态mesh
		uint32_t FzbVertexByteSize = 0;
		FzbVertexFormat vertexFormat;
		for (auto& pair : differentVertexFormatMeshIndexs) {
			vertexFormat = pair.first;
			uint32_t vertexSize = vertexFormat.getVertexSize();	//字节数

			std::vector<uint32_t> meshIndexs = pair.second;
			uint32_t vertexNum = 0;
			uint32_t indexNum = 0;
			for (int i = 0; i < meshIndexs.size(); i++) {
				uint32_t meshIndex = meshIndexs[i];
				vertexNum += sceneMeshSet[meshIndex].vertices.size();
				indexNum += sceneMeshSet[meshIndex].indices.size();
			}

			std::vector<float> compressVertics;
			compressVertics.reserve(vertexNum);
			std::vector<uint32_t> compressIndices;
			compressIndices.reserve(indexNum);
			for (int i = 0; i < meshIndexs.size(); i++) {
				uint32_t meshIndex = meshIndexs[i];

				sceneMeshSet[meshIndex].indexArrayOffset = this->sceneIndices.size() + compressIndices.size();
				sceneMeshSet[meshIndex].indeArraySize = sceneMeshSet[meshIndex].indices.size();	//压缩顶点数据不会改变索引数组的大小和索引的偏移位置
				for (int j = 0; j < sceneMeshSet[meshIndex].indices.size(); j++) {
					sceneMeshSet[meshIndex].indices[j] += compressVertics.size() / (vertexSize / sizeof(float));
				}

				std::vector<float> meshVertices = sceneMeshSet[meshIndex].getVetices();
				compressVertics.insert(compressVertics.end(), meshVertices.begin(), meshVertices.end());
				compressIndices.insert(compressIndices.end(), sceneMeshSet[meshIndex].indices.begin(), sceneMeshSet[meshIndex].indices.end());
				//sceneMeshSet[meshIndex].indices.clear();
			}

			//std::vector<float> testVertices = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f };
			//std::vector<uint32_t> testIndices = { 0, 2, 4, 1, 2, 4, 1, 3, 4 };
			//compressSceneVertics(testVertices, FzbVertexFormat(), testIndices);

			if (compress)
				compressSceneVertics(compressVertics, vertexFormat, compressIndices);
			while (FzbVertexByteSize % vertexSize > 0) {
				this->sceneVertices.push_back(0.0f);
				FzbVertexByteSize += sizeof(float);
			}
			if (FzbVertexByteSize > 0) {
				for (int i = 0; i < compressIndices.size(); i++) {
					compressIndices[i] += FzbVertexByteSize / vertexSize;
				}
			}
			FzbVertexByteSize += compressVertics.size() * sizeof(float);

			this->sceneVertices.insert(sceneVertices.end(), compressVertics.begin(), compressVertics.end());
			this->sceneIndices.insert(sceneIndices.end(), compressIndices.begin(), compressIndices.end());
		}
	}

	void createVertexBuffer() {
		this->vertexBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, sceneVertices.data(), sceneVertices.size() * sizeof(float));
		this->sceneVertices.clear();
	}
	
	void initScene(bool compress = true) {
		getSceneVertics(compress);
		createVertexBuffer();
		createBufferAndDescriptorOfMaterialAndMesh();
	}

	void createAABB() {
		this->AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
		for (int i = 0; i < this->sceneMeshSet.size(); i++) {
			if (sceneMeshSet[i].AABB.isEmpty())
				sceneMeshSet[i].createAABB();
			FzbAABBBox meshAABB = sceneMeshSet[i].AABB;
			AABB.leftX = meshAABB.leftX < AABB.leftX ? meshAABB.leftX : AABB.leftX;
			AABB.rightX = meshAABB.rightX > AABB.rightX ? meshAABB.rightX : AABB.rightX;
			AABB.leftY = meshAABB.leftY < AABB.leftY ? meshAABB.leftY : AABB.leftY;
			AABB.rightY = meshAABB.rightY > AABB.rightY ? meshAABB.rightY : AABB.rightY;
			AABB.leftZ = meshAABB.leftZ < AABB.leftZ ? meshAABB.leftZ : AABB.leftZ;
			AABB.rightZ = meshAABB.rightZ > AABB.rightZ ? meshAABB.rightZ : AABB.rightZ;
		}
		//对于面，我们给个0.2的宽度
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
};

#endif
