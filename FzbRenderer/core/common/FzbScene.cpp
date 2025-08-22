#include "FzbShader.h"
#include "FzbMaterial.h"
#include "./FzbScene.h"

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

size_t VectorFloatHash::operator()(std::vector<float> const& v) const noexcept {
	// 初始 seed（可以用长度，也可以换成常数）
	size_t seed = v.size();

	for (float f : v) {
		// 把 float 按位转换到 uint32_t（C++20 可用 std::bit_cast）
		uint32_t bits;
		std::memcpy(&bits, &f, sizeof(bits));

		// 规范化 ±0.0 -> 0
		if ((bits & 0x7FFFFFFFu) == 0u) {
			bits = 0u;
		}

		// 规范化 NaN（把任意 NaN 映射为一个 canonical NaN 比特形态）
		// IEEE-754 单精度：指数全1 且尾数非0 表示 NaN
		if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
			bits = 0x7FC00000u; // canonical quiet NaN
		}

		// 用 uint32_t 的哈希（或直接用 bits 参与更快的混合）
		size_t h = std::hash<uint32_t>{}(bits);

		// hash_combine 风格混合
		seed ^= h + static_cast<size_t>(0x9e3779b97f4a7c15ULL) + (seed << 6) + (seed >> 2);
	}

	return seed;
}
size_t Mat4Hash::operator()(glm::mat4 const& m) const noexcept {
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
bool Mat4Equal::operator()(glm::mat4 const& a, glm::mat4 const& b) const noexcept {
	// 如果你只想严格逐元素相等，可直接用 ==  
	// 但为了保险，也可以手动比：
	for (int col = 0; col < 4; ++col)
		for (int row = 0; row < 4; ++row)
			if (a[col][row] != b[col][row])
				return false;
	return true;
}
bool FzbVertexFormatLess::operator()(FzbVertexFormat const& a, FzbVertexFormat const& b) const noexcept {
	return a.getVertexSize() < b.getVertexSize();
}

//-----------------------------------------------场景----------------------------------------
FzbScene::FzbScene() {};
FzbScene::FzbScene(std::string scenePath) {
	this->scenePath = scenePath;
	std::string xmlPath = scenePath + "/scene_onlyDiff.xml";
	auto result = this->doc.load_file(xmlPath.c_str());
	if (!result) {
		throw std::runtime_error("pugixml打开文件失败");
	}
	pugi::xml_node scene = this->doc.document_element();
	for (pugi::xml_node node : scene.children("default")) {

	}
	this->width = std::stoi(scene.select_node(".//default[@name='resx']").node().attribute("value").value());
	this->height = std::stoi(scene.select_node(".//default[@name='resy']").node().attribute("value").value());
}
void FzbScene::initScene(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, FzbVertexFormat vertexFormat, bool compress) {
	this->physicalDevice = physicalDevice;
	this->logicalDevice = logicalDevice;
	this->commandPool = commandPool;
	this->graphicsQueue = graphicsQueue;

	createMeshDescriptor(logicalDevice, meshDescriptorSetLayout);
	addSceneFromMitsubaXML(scenePath, vertexFormat);
	getSceneVertics(compress);
	createVertexBuffer();
	createBufferAndDescriptorOfMaterialAndMesh();
	createCameraAndLightBufferAndDescriptor();
}
void FzbScene::clean() {

	for (auto& materialPair : sceneMaterials) {
		materialPair.second.clean();
	}
	for (auto& shaderPair : sceneShaders) {
		shaderPair.second.clean();
	}
	for (auto& imagePair : sceneImages) {
		imagePair.second.clean();
	}
	for (int i = 0; i < this->sceneMeshSet.size(); i++) {
		sceneMeshSet[i].clean();
	}

	if(sceneDescriptorPool) vkDestroyDescriptorPool(logicalDevice, sceneDescriptorPool, nullptr);
	if(meshDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, meshDescriptorSetLayout, nullptr);

	vertexBuffer.clean();
	indexBuffer.clean();

	cameraBuffer.clean();
	lightsBuffer.clean();
	if(cameraAndLightsDescriptorSetLayout) vkDestroyDescriptorSetLayout(logicalDevice, cameraAndLightsDescriptorSetLayout, nullptr);
}
void FzbScene::clear() {
	clean();
	sceneMeshSet.clear();
	differentVertexFormatMeshIndexs.clear();
	sceneVertices.clear();
	sceneIndices.clear();
	sceneMaterials.clear();
}

void FzbScene::addMeshToScene(FzbMesh mesh, bool reAdd) {

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
	if (!reAdd) this->sceneMeshSet.push_back(mesh);
}
/*
std::string FzbScene::getRootPath() {
	std::filesystem::path thisFile = __FILE__;
	return thisFile.parent_path().parent_path().parent_path().string();	//得到Renderer文件夹
}
*/
void FzbScene::createDefaultMaterial(FzbVertexFormat vertexFormat) {
	FzbMaterial defaultMaterial = FzbMaterial(logicalDevice, "defaultMaterial", "diffuse");
	this->sceneMaterials.insert({ "defaultMaterial", defaultMaterial });
	std::string shaderPath = getRootPath() + shaderPaths.at("diffuse");
	this->sceneShaders.insert({ shaderPath, FzbShader(logicalDevice, shaderPath) });
	this->sceneShaders[shaderPath].createShaderVariant(&this->sceneMaterials["defaultMaterial"], vertexFormat);
}
void FzbScene::addSceneFromMitsubaXML(std::string path, FzbVertexFormat vertexFormat) {
	createDefaultMaterial(vertexFormat);

	pugi::xml_node scene = doc.document_element();	//获取根节点，即<scene>
	for (pugi::xml_node node : scene.children("sensor")) {	//这里可能有问题，我现在默认存在这些属性，且相机是透视投影的。有问题之后再说吧
		float fov = glm::radians(std::stof(node.select_node(".//float[@name='fov']").node().attribute("value").value()));
		bool isPerspective = std::string(node.attribute("type").value()) == "perspective" ? true : false;
		pugi::xml_node tranform = node.select_node(".//transform[@name='to_world']").node();
		glm::mat4 inverseViewMatrix = getMat4FromString(tranform.child("matrix").attribute("value").value());
		glm::vec4 position = inverseViewMatrix[3];
		float aspect = (float)this->width / this->height;
		float nearPlane = 0.1f;
		float farPlane = 100.0f;
		fov = 2.0f * atanf(tanf(fov * 0.5f) / aspect);	//三叶草中给的fov是水平方向的，而glm中要的是垂直方向的
		FzbCamera camera = FzbCamera(position, fov, aspect, nearPlane, farPlane);
		camera.setViewMatrix(inverseViewMatrix, true);
		camera.isPerspective = isPerspective;
		this->sceneCameras.push_back(camera);
	}

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
		//std::string shaderPath = getRootPath() + "/shaders/forward/diffuse";
		if (materialType == "diffuse") {
			if (pugi::xml_node textureNode = bsdf.select_node(".//texture[@name='reflectance']").node()) {
				std::string texturePath = textureNode.select_node(".//string[@name='filename']").node().attribute("value").value();
				std::string filterType = textureNode.select_node(".//string[@name='filter_type']").node().attribute("value").value();
				VkFilter filter = filterType == "bilinear" ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
				material.properties.textureProperties.insert({ "albedoMap", FzbTexture(texturePath, filter) });
			}
			if (pugi::xml_node rgbNode = bsdf.select_node(".//rgb[@name='reflectance']").node()) {
				glm::vec3 albedo = getRGBFromString(rgbNode.attribute("value").value());
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
		std::string shaderPath = getRootPath() + shaderPaths.at(material.type);
		if (!this->sceneShaders.count(shaderPath)) {
			this->sceneShaders.insert({ shaderPath, FzbShader(logicalDevice, shaderPath) });
		}
		this->sceneShaders[shaderPath].createShaderVariant(&sceneMaterials[material.id], vertexFormat);
	}

	for (pugi::xml_node node : scene.children("shape")) {
		std::string meshType = node.attribute("type").value();
		if (meshType == "rectangle" || meshType == "emitter") {
			continue;
		}
		std::string meshID = node.attribute("id").value();
		pugi::xml_node tranform = node.select_node(".//transform[@name='to_world']").node();

		if (meshID == "Light") {
			glm::mat4 modelMatrix = getMat4FromString(tranform.child("matrix").attribute("value").value());
			glm::vec3 position = glm::vec3(modelMatrix * glm::vec4(1.0f));
			pugi::xml_node emitter = node.child("emitter");		//不知道会不会有多个发射器，默认1个，以后遇到了再说
			glm::vec3 strength = getRGBFromString(emitter.select_node(".//rgb[@name='radiance']").node().attribute("value").value());
			if (meshType == "point") {
				this->sceneLights.push_back(FzbLight(position, strength));
			}
			continue;
		}

		pugi::xml_node material = node.select_node(".//ref").node();
		std::string materialID = material.attribute("id").value();
		if (!sceneMaterials.count(materialID)) {
			materialID = "defaultMaterial";
		}

		std::vector<FzbMesh> meshes;
		if (meshType == "obj") {
			if (pugi::xml_node pathNode = node.select_node(".//string[@name='filename']").node()) {
				std::string objPath = path + "/" + std::string(pathNode.attribute("value").value());

				meshes = fzbGetMeshFromOBJ(logicalDevice, objPath, sceneMaterials[materialID].vertexFormat);
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

		glm::mat4 modelMatrix(1.0f);
		if (tranform) modelMatrix = getMat4FromString(tranform.child("matrix").attribute("value").value());

		for (int i = 0; i < meshes.size(); i++) {
			meshes[i].transforms = modelMatrix;
			meshes[i].material = &sceneMaterials[materialID];
			meshes[i].id = meshID;
			addMeshToScene(meshes[i]);
		}
	}

	for (auto& pair : sceneShaders) {
		FzbShader& shader = pair.second;
		shader.createMeshBatch(physicalDevice, commandPool, graphicsQueue, sceneMeshSet);
	}

	doc.reset();
}

void FzbScene::createBufferAndDescriptorOfMaterialAndMesh() {
	uint32_t textureNum = 0;
	uint32_t numberBufferNum = 0;
	for (auto& materialPair : this->sceneMaterials) {
		FzbMaterial& material = materialPair.second;
		material.createSource(physicalDevice, commandPool, graphicsQueue, scenePath, numberBufferNum, sceneImages);
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
	numberBufferNum += 2;	// sceneCameras.size() + sceneLights.size();
	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum = { {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, this->sceneMeshSet.size()},	//mesh特有信息
																{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textureNum},		//纹理信息
																{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, numberBufferNum} };	//材质信息
	this->sceneDescriptorPool = fzbCreateDescriptorPool(logicalDevice, bufferTypeAndNum);
	for (int i = 0; i < this->sceneMeshSet.size(); i++) {
		this->sceneMeshSet[i].createBuffer(physicalDevice);
		this->sceneMeshSet[i].createDescriptor(sceneDescriptorPool, meshDescriptorSetLayout);
	}
	for (auto& shader : sceneShaders) {
		shader.second.createDescriptor(sceneDescriptorPool, sceneImages);
	}
}
void FzbScene::compressSceneVertics(std::vector<float>& vertices, FzbVertexFormat vertexFormat, std::vector<uint32_t>& indices) {
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
void FzbScene::getSceneVertics(bool compress) {	//目前只处理静态mesh
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
			this->sceneMeshIndices.push_back(meshIndex);

			sceneMeshSet[meshIndex].indexArrayOffset = this->sceneIndices.size() + compressIndices.size();
			sceneMeshSet[meshIndex].indeArraySize = sceneMeshSet[meshIndex].indices.size();	//压缩顶点数据不会改变索引数组的大小和索引的偏移位置

			//这里用临时数组，因为可能同一个mesh被添加多次，如果每次都直接对mesh的indices加上offset就会出错，所以需要使用临时数据
			//同时这不会影响meshBatch的indexBuffer创建，因此indices的大小和在sceneIndices中的偏移都不变。
			std::vector<uint32_t> meshIndices = sceneMeshSet[meshIndex].indices;
			for (int j = 0; j < meshIndices.size(); j++) {
				meshIndices[j] += compressVertics.size() / (vertexSize / sizeof(float));
			}

			std::vector<float> meshVertices = sceneMeshSet[meshIndex].getVetices();
			compressVertics.insert(compressVertics.end(), meshVertices.begin(), meshVertices.end());
			compressIndices.insert(compressIndices.end(), meshIndices.begin(), meshIndices.end());
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
void FzbScene::createVertexBuffer() {
	this->vertexBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, sceneVertices.data(), sceneVertices.size() * sizeof(float));
	this->sceneVertices.clear();

	this->indexBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, sceneIndices.data(), sceneIndices.size() * sizeof(uint32_t));
	this->sceneIndices.clear();
}
void FzbScene::createAABB() {
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

void FzbScene::createCameraAndLightBufferAndDescriptor() {
	//camera的数据每帧都会变化，所以这里只是创建buffer，但是不填入数据，具体数据由渲染组件填入
	cameraBuffer = fzbCreateUniformBuffers(physicalDevice, logicalDevice, sizeof(FzbCameraUniformBufferObject));
	//FzbCameraUniformBufferObject cameraData{};
	//cameraData.view = sceneCameras[0].GetViewMatrix();
	//cameraData.proj = glm::perspectiveRH_ZO(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 100.0f);
	//cameraData.proj[1][1] *= -1;
	//cameraData.cameraPos = glm::vec4(sceneCameras[0].Position, 0.0f);
	//memcpy(cameraBuffer.mapped, &cameraData, sizeof(FzbCameraUniformBufferObject));

	lightsBuffer = fzbCreateUniformBuffers(physicalDevice, logicalDevice, sizeof(FzbLightsUniformBufferObject));
	FzbLightsUniformBufferObject lightsData(sceneLights.size());
	for (int i = 0; i < sceneLights.size(); i++) {
		FzbLight& light = sceneLights[i];
		lightsData.lightData[i].pos = glm::vec4(light.position, 1.0f);
		lightsData.lightData[i].strength = glm::vec4(light.strength, 1.0f);
	}
	memcpy(lightsBuffer.mapped, &lightsData, sizeof(FzbLightsUniformBufferObject));

	std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
	std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_ALL };
	cameraAndLightsDescriptorSetLayout = fzbCreateDescriptLayout(logicalDevice, 2, descriptorTypes, descriptorShaderFlags);
	cameraAndLightsDescriptorSet = fzbCreateDescriptorSet(logicalDevice, sceneDescriptorPool, cameraAndLightsDescriptorSetLayout);

	std::vector<VkWriteDescriptorSet> uniformDescriptorWrites(sceneCameras.size() + sceneLights.size());
	VkDescriptorBufferInfo cameraUniformBufferInfo{};
	cameraUniformBufferInfo.buffer = cameraBuffer.buffer;
	cameraUniformBufferInfo.offset = 0;
	cameraUniformBufferInfo.range = sizeof(FzbCameraUniformBufferObject);
	uniformDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	uniformDescriptorWrites[0].dstSet = cameraAndLightsDescriptorSet;
	uniformDescriptorWrites[0].dstBinding = 0;
	uniformDescriptorWrites[0].dstArrayElement = 0;
	uniformDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uniformDescriptorWrites[0].descriptorCount = 1;
	uniformDescriptorWrites[0].pBufferInfo = &cameraUniformBufferInfo;

	VkDescriptorBufferInfo lightUniformBufferInfo{};
	lightUniformBufferInfo.buffer = lightsBuffer.buffer;
	lightUniformBufferInfo.offset = 0;
	lightUniformBufferInfo.range = sizeof(FzbLightsUniformBufferObject);
	uniformDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	uniformDescriptorWrites[1].dstSet = cameraAndLightsDescriptorSet;
	uniformDescriptorWrites[1].dstBinding = 1;
	uniformDescriptorWrites[1].dstArrayElement = 0;
	uniformDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	uniformDescriptorWrites[1].descriptorCount = 1;
	uniformDescriptorWrites[1].pBufferInfo = &lightUniformBufferInfo;

	vkUpdateDescriptorSets(logicalDevice, uniformDescriptorWrites.size(), uniformDescriptorWrites.data(), 0, nullptr);
}
void FzbScene::updateCameraBuffer() {
	FzbCameraUniformBufferObject ubo{};
	FzbCamera& camera = this->sceneCameras[0];
	ubo.view = camera.GetViewMatrix();
	ubo.proj = camera.GetProjMatrix();
	ubo.cameraPos = glm::vec4(camera.position, 0.0f);
	memcpy(cameraBuffer.mapped, &ubo, sizeof(ubo));
}