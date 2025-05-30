#pragma once

#include "FzbBuffer.h"
#include "FzbPipeline.h"
#include "FzbDescriptor.h"
#include "FzbShader.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#ifndef FZB_MESH_H
#define FZB_MESH_H
/*
enum FzbVertexType {
	FZB_VERTEX = 0,
	FZB_VERTEX_ONLYPOS = 1,
	FZB_VERTEX_POSNORMAL = 2,
	FZB_VERTEX_POSNORMALTEXCOORD = 3
};

struct FzbVertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 texCoord;
	glm::vec3 tangent;

	FzbVertex() {
		this->pos = glm::vec3(0.0f);
		this->texCoord = glm::vec2(0.0f);
		this->normal = glm::vec3(0.0f);
		this->tangent = glm::vec3(0.0f);
	}

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(FzbVertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {

		//VAO
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
		attributeDescriptions.resize(4);

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(FzbVertex, pos);	//找pos在Vertex中的偏移

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(FzbVertex, normal);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(FzbVertex, texCoord);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(FzbVertex, tangent);

		return attributeDescriptions;
	}

	bool operator==(const FzbVertex& other) const {
		return pos == other.pos && texCoord == other.texCoord && normal == other.normal && tangent == other.tangent;
	}

};

struct FzbVertex_OnlyPos {
	glm::vec3 pos;

	FzbVertex_OnlyPos() {};
	FzbVertex_OnlyPos(FzbVertex vertex) {
		this->pos = vertex.pos;
	}

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(FzbVertex_OnlyPos);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {

		//VAO
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
		attributeDescriptions.resize(1);

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(FzbVertex_OnlyPos, pos);	//找pos在Vertex中的偏移

		return attributeDescriptions;
	}

	bool operator==(const FzbVertex_OnlyPos& other) const {
		return pos == other.pos;
	}

};

struct FzbVertex_PosNormal {
	glm::vec3 pos;
	glm::vec3 normal;

	FzbVertex_PosNormal() {};
	FzbVertex_PosNormal(FzbVertex vertex) {
		this->pos = vertex.pos;
		this->normal = vertex.normal;
	}

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(FzbVertex_PosNormal);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {

		//VAO
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
		attributeDescriptions.resize(2);

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(FzbVertex, pos);	//找pos在Vertex中的偏移

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 2;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(FzbVertex, normal);

		return attributeDescriptions;
	}

	bool operator==(const FzbVertex_PosNormal& other) const {
		return pos == other.pos && normal == other.normal;
	}
};

struct FzbVertex_PosNormalTexCoord {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 texCoord;

	FzbVertex_PosNormalTexCoord() {};
	FzbVertex_PosNormalTexCoord(FzbVertex vertex) {
		this->pos = vertex.pos;
		this->texCoord = vertex.texCoord;
		this->normal = vertex.normal;
	}

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(FzbVertex_PosNormalTexCoord);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {

		//VAO
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
		attributeDescriptions.resize(3);

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(FzbVertex, pos);	//找pos在Vertex中的偏移

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT; 
		attributeDescriptions[1].offset = offsetof(FzbVertex, normal);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(FzbVertex, texCoord);

		return attributeDescriptions;
	}

	bool operator==(const FzbVertex_PosNormalTexCoord& other) const {
		return pos == other.pos && texCoord == other.texCoord && normal == other.normal;
	}
};
*/

struct FzbMaterial {
	glm::vec4 ka;
	glm::vec4 kd;
	glm::vec4 ks;
	glm::vec4 ke;

	FzbMaterial() {
		this->ka = glm::vec4(1.0f);
		this->kd = glm::vec4(1.0f);
		this->ks = glm::vec4(1.0f);
		this->ke = glm::vec4(0.0f);
	}

	bool operator==(const FzbMaterial& other) const {
		return ka == other.ka && kd == other.kd && ks == other.ks && ke == other.ke;
	}
};

struct FzbAABBBox {

	float leftX;
	float rightX;
	float leftY;
	float rightY;
	float leftZ;
	float rightZ;

	glm::vec3 startPos;
	glm::vec3 distanceXYZ;
	glm::vec3 center;

	float getAxis(int k) {
		if (k == 0) {
			return leftX;
		}
		else if (k == 1) {
			return rightX;
		}
		else if (k == 2) {
			return leftY;
		}
		else if (k == 3) {
			return rightY;
		}
		else if (k == 4) {
			return leftZ;
		}
		else if (k == 5) {
			return rightZ;
		}
	}

};

struct FzbTexture {
	//uint32_t id;
	std::string type;
	std::string path = "";

	bool operator==(const FzbTexture& other) const {
		return type == other.type && path == other.path;
	}
};

struct FzbShaderTextureIndexs {
	int albedoTextureIndex = -1;	//反射率纹理索引
	int normalTextureIndex = -1; //发现纹理索引
};

struct FzbMeshMaterialUniformObject {
	int transformIndex = -1;
	int materialIndex = -1;	//当前draw的mesh的材质在材质buffer中的索引
	int albedoTextureIndex = -1;	//反射率纹理索引
	int normalTextureIndex = -1; //发现纹理索引

	FzbMeshMaterialUniformObject() {
		transformIndex = -1;
		materialIndex = -1;
		albedoTextureIndex = -1;
		normalTextureIndex = -1;
	}
};

struct FzbMesh {
public:
	std::vector<float> vertices;	//压缩前的顶点数据，压缩后就会被释放
	std::vector<uint32_t> indices;	//压缩前的顶点索引数据，压缩后就会被释放
	uint32_t indexArrayOffset;
	uint32_t indeArraySize;

	glm::mat4 transforms;	//一个mesh对应一种变换
	FzbMaterial material;
	FzbTexture albedoTexture;
	FzbTexture normalTexture;
	FzbMeshMaterialUniformObject materialUniformObject;

	FzbVertexFormat vertexFormat;	//从obj中获取到的mesh的顶点格式
	FzbShader shader;

	FzbAABBBox AABB;

	FzbMesh() {
		this->transforms = glm::mat4(1.0f);
		this->materialUniformObject = FzbMeshMaterialUniformObject();
		this->vertexFormat = FzbVertexFormat();
	};

	std::vector<float> getVetices() {
		
		if (shader.vertexFormat == vertexFormat) {
			return vertices;
		}
		bool skip = (vertexFormat.useNormal == false && shader.vertexFormat.useNormal == true) || (vertexFormat.useNormal == shader.vertexFormat.useNormal);
		skip = skip && ((vertexFormat.useTangent == false && shader.vertexFormat.useTangent == true) || (vertexFormat.useTangent == shader.vertexFormat.useTangent));
		skip = skip && ((vertexFormat.useTangent == false && shader.vertexFormat.useTangent == true) || (vertexFormat.useTangent == shader.vertexFormat.useTangent));
		if (skip) {
			return vertices;
		}

		uint32_t vertexSize = vertexFormat.getVertexSize() / sizeof(float);
		std::vector<float> vertices_temp;
		vertices_temp.reserve(this->vertices.size());
		for (int i = 0; i < vertices.size(); i += vertexSize) {
			for (int j = 0; j < 3; j++) {	//pos
				vertices_temp.push_back(vertices[i + j]);
			}
			uint32_t oriOffset = 3;
			if (vertexFormat.useNormal) {
				if (shader.vertexFormat.useNormal) {
					for (int j = 3; j < 6; j++) {
						vertices_temp.push_back(vertices[i + j]);
					}
				}
				oriOffset = 6;
			}
			if (vertexFormat.useTexCoord) {
				if (shader.vertexFormat.useTexCoord) {
					for (int j = 0; j < 2; j++) {
						vertices_temp.push_back(vertices[i + oriOffset + j]);
					}
				}
				oriOffset += 2;
			}
			if (vertexFormat.useTangent) {
				if (shader.vertexFormat.useTangent) {
					for (int j = 0; j < 3; j++) {
						vertices_temp.push_back(vertices[i + oriOffset + j]);
					}
				}
			}
		}
		return vertices_temp;
	}

	void clean() {
		
	}

	void render(VkCommandBuffer commandBuffer, std::vector<VkDeviceSize>& offsets) {

		

	}
};

std::vector<FzbTexture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName) {

	std::vector<FzbTexture> textures;
	for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
	{
		aiString str;
		mat->GetTexture(type, i, &str);
		FzbTexture texture;
		//texture.id = TextureFromFile(str.C_Str(), directory);
		texture.type = typeName;
		texture.path = str.C_Str();		//meshPathDirectory + '/' + str.C_Str();
		textures.push_back(texture);
		//sceneTextures->push_back(texture); // 添加到已加载的纹理中
	}

	return textures;

}

FzbMesh processMesh(aiMesh* mesh, const aiScene* scene) {

	FzbMesh fzbMesh;
	for (uint32_t i = 0; i < mesh->mNumVertices; i++) {

		fzbMesh.vertices.push_back(mesh->mVertices[i].x);
		fzbMesh.vertices.push_back(mesh->mVertices[i].y);
		fzbMesh.vertices.push_back(mesh->mVertices[i].z);

		if (mesh->HasNormals()) {
			fzbMesh.vertexFormat.useNormal = true;
			fzbMesh.vertices.push_back(mesh->mNormals[i].x);
			fzbMesh.vertices.push_back(mesh->mNormals[i].y);
			fzbMesh.vertices.push_back(mesh->mNormals[i].z);
		}

		if (mesh->mTextureCoords[0]) // 网格是否有纹理坐标？这里只处理一种纹理uv
		{
			fzbMesh.vertexFormat.useTexCoord = true;
			fzbMesh.vertices.push_back(mesh->mTextureCoords[0][i].x);
			fzbMesh.vertices.push_back(mesh->mTextureCoords[0][i].y);
		}

		if (mesh->HasTangentsAndBitangents()) {
			fzbMesh.vertexFormat.useTangent = true;
			fzbMesh.vertices.push_back(mesh->mTangents[i].x);
			fzbMesh.vertices.push_back(mesh->mTangents[i].y);
			fzbMesh.vertices.push_back(mesh->mTangents[i].z);
		}
	}

	for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (uint32_t j = 0; j < face.mNumIndices; j++) {
			fzbMesh.indices.push_back(face.mIndices[j]);
		}
	}

	FzbMaterial mat;
	if (mesh->mMaterialIndex >= 0) {

		aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
		aiColor3D color;
		material->Get(AI_MATKEY_COLOR_AMBIENT, color);
		mat.ka = glm::vec4(color.r, color.g, color.b, 1.0);
		material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
		mat.kd = glm::vec4(color.r, color.g, color.b, 1.0);
		material->Get(AI_MATKEY_COLOR_SPECULAR, color);
		mat.ks = glm::vec4(color.r, color.g, color.b, 1.0);
		material->Get(AI_MATKEY_COLOR_EMISSIVE, color);
		mat.ke = glm::vec4(color.r, color.g, color.b, 1.0);

		std::vector<FzbTexture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "albedoTexture");
		if (diffuseMaps.size() > 0)
			fzbMesh.albedoTexture = diffuseMaps[0];	//只取一个，多个很少见，有需求了再说

		//std::vector<FzbTexture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
		//textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

		std::vector<FzbTexture> normalMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, "normalTexture");
		if (normalMaps.size() > 0)
			fzbMesh.normalTexture = normalMaps[0];

	}

	return fzbMesh;
}

std::vector<FzbMesh> processNode(aiNode* node, const aiScene* scene) {

	std::vector<FzbMesh> meshes;
	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		meshes.push_back(processMesh(mesh, scene));
	}

	for (uint32_t i = 0; i < node->mNumChildren; i++) {
		std::vector<FzbMesh> results = processNode(node->mChildren[i], scene);
		meshes.insert(meshes.begin(), results.begin(), results.end());
	}

	return meshes;
}

std::vector<FzbMesh> getMeshFromOBJ(std::string path, FzbVertexFormat meshVertexFormat = FzbVertexFormat()) {
	Assimp::Importer import;
	uint32_t needs = aiProcess_Triangulate |
		(meshVertexFormat.useTexCoord ? aiProcess_FlipUVs : aiPostProcessSteps(0u)) |
		(meshVertexFormat.useNormal ? aiProcess_GenSmoothNormals : aiPostProcessSteps(0u)) |
		(meshVertexFormat.useTangent ? aiProcess_CalcTangentSpace : aiPostProcessSteps(0u));
	const aiScene* scene = import.ReadFile(path, needs);

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
		throw std::runtime_error("ERROR::ASSIMP::" + (std::string)import.GetErrorString());
	}

	//std::string  meshPathDirectory = path.substr(0, path.find_last_of('/'));
	return processNode(scene->mRootNode, scene);

}

//批应该分为两种，一种是批内mesh都相同，可以当作一个大的mesh；另一种就是shader相同，但是数据不同
//一个batch中的mesh的shader相同，即顶点格式、所用纹理数量、类型什么的都相同。
struct FzbMeshBatch {
public:
	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;

	FzbShader shader;
	std::vector<FzbMesh*> meshes;

	VkDescriptorSet descriptorSet;

	uint32_t vertexBufferOffset = 0;
	FzbBuffer indexBuffer;
	FzbBuffer materialIndexBuffer;
	uint32_t drawIndexedIndirectCommandSize = 0;
	FzbBuffer drawIndexedIndirectCommandBuffer;
	
	FzbMeshBatch() {};
	FzbMeshBatch(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue) {
		this->physicalDevice = physicalDevice;
		this->logicalDevice = logicalDevice;
		this->commandPool = commandPool;
		this->graphicsQueue = graphicsQueue;
	}

	void clean() {
		indexBuffer.clean();
		materialIndexBuffer.clean();
		drawIndexedIndirectCommandBuffer.clean();

		shader.clean();
	}

	void createMeshBatchIndexBuffer(std::vector<uint32_t>& sceneIndices) {
		std::vector<uint32_t> batchIndices;
		for (int i = 0; i < meshes.size(); i++) {
			uint32_t indicesOffset = meshes[i]->indexArrayOffset;
			batchIndices.insert(batchIndices.end(), sceneIndices.begin() + indicesOffset, sceneIndices.begin() + indicesOffset + meshes[i]->indeArraySize);
		}

		indexBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, batchIndices.data(), batchIndices.size() * sizeof(uint32_t));
	}

	void createMeshBatchMaterialBuffer() {
		std::vector<FzbMeshMaterialUniformObject> batchMaterials;
		for (int i = 0; i < meshes.size(); i++) {
			batchMaterials.push_back(meshes[i]->materialUniformObject);
		}
		materialIndexBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, batchMaterials.data(), batchMaterials.size() * sizeof(FzbMeshMaterialUniformObject));
	}

	void createDrawIndexedIndirectCommandBuffer() {
		std::vector<VkDrawIndexedIndirectCommand> batchDrawIndexedIndirectCommands;
		for (int i = 0; i < meshes.size(); i++) {
			drawIndexedIndirectCommandSize++;
			batchDrawIndexedIndirectCommands.push_back({ meshes[i]->indeArraySize, 1, 0, 0, 0 });
		}
		drawIndexedIndirectCommandBuffer = fzbCreateIndirectCommandBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, batchDrawIndexedIndirectCommands.data(), batchDrawIndexedIndirectCommands.size() * sizeof(VkDrawIndexedIndirectCommand));
	}

	void createDescriptorSet(VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout) {
		descriptorSet = fzbCreateDescriptorSet(logicalDevice, descriptorPool, descriptorSetLayout);
		std::vector<VkWriteDescriptorSet> meshBatchDescriptorWrites(1);
		VkDescriptorBufferInfo materialIndexStorageBufferInfo{};
		materialIndexStorageBufferInfo.buffer = this->materialIndexBuffer.buffer;
		materialIndexStorageBufferInfo.offset = 0;
		materialIndexStorageBufferInfo.range = sizeof(FzbMeshMaterialUniformObject) * meshes.size();
		meshBatchDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		meshBatchDescriptorWrites[0].dstSet = descriptorSet;
		meshBatchDescriptorWrites[0].dstBinding = 0;
		meshBatchDescriptorWrites[0].dstArrayElement = 0;
		meshBatchDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		meshBatchDescriptorWrites[0].descriptorCount = 1;
		meshBatchDescriptorWrites[0].pBufferInfo = &materialIndexStorageBufferInfo;

		vkUpdateDescriptorSets(logicalDevice, meshBatchDescriptorWrites.size(), meshBatchDescriptorWrites.data(), 0, nullptr);
	}

	void render(VkCommandBuffer commandBuffer, VkDescriptorSet componentDescriptorSet, VkDescriptorSet sceneDescriptorSet) {
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shader.pipeline);
		//0：组件的uniform,1: material、texture、transform，2：不同meshBatch的materialIndex
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shader.pipelineLayout, 0, 1, &componentDescriptorSet, 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shader.pipelineLayout, 1, 1, &sceneDescriptorSet, 0, nullptr);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shader.pipelineLayout, 2, 1, &descriptorSet, 0, nullptr);
		vkCmdDrawIndexedIndirect(commandBuffer, drawIndexedIndirectCommandBuffer.buffer, 0, drawIndexedIndirectCommandSize, sizeof(VkDrawIndexedIndirectCommand));
	}


};

#endif