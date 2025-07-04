#pragma once

#include "FzbBuffer.h"
#include "FzbPipeline.h"
#include "FzbDescriptor.h"
#include "FzbShader.h"
#include "FzbMaterial.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>

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

struct FzbAABBBox {

	float leftX = FLT_MAX;
	float rightX = -FLT_MAX;
	float leftY = FLT_MAX;
	float rightY = -FLT_MAX;
	float leftZ = FLT_MAX;
	float rightZ = -FLT_MAX;

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

	bool isEmpty() {
		return leftX == FLT_MAX && rightX == -FLT_MAX && leftY == FLT_MAX && rightY == -FLT_MAX && leftZ == FLT_MAX && rightZ == -FLT_MAX;
	}

};

struct FzbMeshUniformBufferObject {
	glm::mat4 transforms;
};

struct FzbMesh {
public:
	VkDevice logicalDevice;

	std::string id;
	std::vector<float> vertices;	//压缩前的顶点数据，压缩后就会被释放
	std::vector<uint32_t> indices;	//压缩前的顶点索引数据，压缩后就会被释放
	uint32_t indexArrayOffset;
	uint32_t indeArraySize;

	glm::mat4 transforms;	//一个mesh对应一种变换
	std::string materialID;
	FzbVertexFormat vertexFormat;	//从obj中获取到的mesh的顶点格式

	FzbBuffer meshBuffer;
	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	VkDescriptorSet descriptorSet = nullptr;	//所需要的材质和纹理信息

	uint32_t instanceNum = 1;
	FzbAABBBox AABB;

	FzbMesh() {
		this->transforms = glm::mat4(1.0f);
		this->vertexFormat = FzbVertexFormat();
	};

	std::vector<float> getVetices() {
		return this->vertices;
		/*
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
		*/
	}

	void clean() {
		meshBuffer.clean();
		vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
	}

	void createBuffer(VkPhysicalDevice physicalDevice) {
		FzbMeshUniformBufferObject uniformBufferObject;
		uniformBufferObject.transforms = this->transforms;
		this->meshBuffer = fzbCreateUniformBuffers(physicalDevice, logicalDevice, sizeof(FzbMeshUniformBufferObject));
		memcpy(this->meshBuffer.mapped, &uniformBufferObject, sizeof(FzbMeshUniformBufferObject));
	}

	void createDescriptor(VkDescriptorPool descriptorPool) {
		this->descriptorSetLayout = fzbCreateDescriptLayout(logicalDevice, 1, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }, { VK_SHADER_STAGE_ALL });
		this->descriptorSet = fzbCreateDescriptorSet(logicalDevice, descriptorPool, descriptorSetLayout);
		VkDescriptorBufferInfo meshBufferInfo{};
		meshBufferInfo.buffer = this->meshBuffer.buffer;
		meshBufferInfo.offset = 0;
		meshBufferInfo.range = sizeof(FzbMeshUniformBufferObject);
		std::vector<VkWriteDescriptorSet> voxelGridMapDescriptorWrites(1);
		voxelGridMapDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voxelGridMapDescriptorWrites[0].dstSet = descriptorSet;
		voxelGridMapDescriptorWrites[0].dstBinding = 0;
		voxelGridMapDescriptorWrites[0].dstArrayElement = 0;
		voxelGridMapDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		voxelGridMapDescriptorWrites[0].descriptorCount = 1;
		voxelGridMapDescriptorWrites[0].pBufferInfo = &meshBufferInfo;

		vkUpdateDescriptorSets(logicalDevice, voxelGridMapDescriptorWrites.size(), voxelGridMapDescriptorWrites.data(), 0, nullptr);
	}

	void createAABB() {
		uint32_t vertexSize = this->vertexFormat.getVertexSize() / sizeof(float);

		this->AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
		for (int i = 0; i < this->vertices.size(); i += vertexSize) {
			float x = vertices[i];
			float y = vertices[i + 1];
			float z = vertices[i + 2];
			AABB.leftX = x < AABB.leftX ? x : AABB.leftX;
			AABB.rightX = x > AABB.rightX ?x : AABB.rightX;
			AABB.leftY = y < AABB.leftY ? y : AABB.leftY;
			AABB.rightY = y > AABB.rightY ? y : AABB.rightY;
			AABB.leftZ = z < AABB.leftZ ? z : AABB.leftZ;
			AABB.rightZ = z > AABB.rightZ ? z : AABB.rightZ;
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
		//texture.type = typeName;
		texture.path = str.C_Str();		//meshPathDirectory + '/' + str.C_Str();
		textures.push_back(texture);
		//sceneTextures->push_back(texture); // 添加到已加载的纹理中
	}

	return textures;

}

FzbMesh processMesh(aiMesh* mesh, const aiScene* scene, FzbMaterial& material) {

	FzbVertexFormat materialVertexFormat = material.getVertexFormat();

	FzbMesh fzbMesh;
	for (uint32_t i = 0; i < mesh->mNumVertices; i++) {

		fzbMesh.vertices.push_back(mesh->mVertices[i].x);
		fzbMesh.vertices.push_back(mesh->mVertices[i].y);
		fzbMesh.vertices.push_back(mesh->mVertices[i].z);

		if (materialVertexFormat.useNormal) {
			if (mesh->HasNormals()) {
				fzbMesh.vertexFormat.useNormal = true;
				fzbMesh.vertices.push_back(mesh->mNormals[i].x);
				fzbMesh.vertices.push_back(mesh->mNormals[i].y);
				fzbMesh.vertices.push_back(mesh->mNormals[i].z);
			}
			else {
				throw std::runtime_error("该mesh没有法线属性");
			}
		}

		if (materialVertexFormat.useTexCoord) {
			if (mesh->mTextureCoords[0]) // 网格是否有纹理坐标？这里只处理一种纹理uv
			{
				fzbMesh.vertexFormat.useTexCoord = true;
				fzbMesh.vertices.push_back(mesh->mTextureCoords[0][i].x);
				fzbMesh.vertices.push_back(mesh->mTextureCoords[0][i].y);
			}
			else {
				throw std::runtime_error("该mesh没有uv属性");
			}
		}

		if (materialVertexFormat.useTangent) {
			if (mesh->HasTangentsAndBitangents()) {
				fzbMesh.vertexFormat.useTangent = true;
				fzbMesh.vertices.push_back(mesh->mTangents[i].x);
				fzbMesh.vertices.push_back(mesh->mTangents[i].y);
				fzbMesh.vertices.push_back(mesh->mTangents[i].z);
			}
			else {
				throw std::runtime_error("该mesh没有切线属性");
			}
		}

		material.changeVertexFormat(fzbMesh.vertexFormat);	//根据mesh的实际顶点格式修改材质和shader的顶点格式
	}

	for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (uint32_t j = 0; j < face.mNumIndices; j++) {
			fzbMesh.indices.push_back(face.mIndices[j]);
		}
	}
	fzbMesh.indeArraySize = fzbMesh.indices.size();

	/*
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
	*/

	return fzbMesh;
}

std::vector<FzbMesh> processNode(aiNode* node, const aiScene* scene, FzbMaterial& material) {

	std::vector<FzbMesh> meshes;
	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		meshes.push_back(processMesh(mesh, scene, material));
	}

	for (uint32_t i = 0; i < node->mNumChildren; i++) {
		std::vector<FzbMesh> results = processNode(node->mChildren[i], scene, material);
		meshes.insert(meshes.begin(), results.begin(), results.end());
	}

	return meshes;
}

std::vector<FzbMesh> fzbGetMeshFromOBJ(std::string path, FzbMaterial& material) {
	Assimp::Importer import;
	FzbVertexFormat materialVertexFormat = material.getVertexFormat();
	uint32_t needs = aiProcess_Triangulate |
		(materialVertexFormat.useTexCoord ? aiProcess_FlipUVs : aiPostProcessSteps(0u)) |
		(materialVertexFormat.useNormal ? aiProcess_GenSmoothNormals : aiPostProcessSteps(0u)) |
		(materialVertexFormat.useTangent ? aiProcess_CalcTangentSpace : aiPostProcessSteps(0u));
	const aiScene* scene = import.ReadFile(path, needs);	//相对地址从main程序目录开始，也就是在FzbRenderer下，所以相对地址要从那里开始

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
		throw std::runtime_error("ERROR::ASSIMP::" + (std::string)import.GetErrorString());
	}

	//std::string  meshPathDirectory = path.substr(0, path.find_last_of('/'));
	return processNode(scene->mRootNode, scene, material);

}

void fzbCreateCube(std::vector<float>& cubeVertices, std::vector<uint32_t>& cubeIndices) {
	cubeVertices = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f };
	cubeIndices = {
					1, 0, 3, 1, 3, 2,
					4, 5, 6, 4, 6, 7,
					5, 1, 2, 5, 2, 6,
					0, 4, 7, 0, 7, 3,
					7, 6, 2, 7, 2, 3,
					0, 1, 5, 0, 5, 4
	};
}

void fzbCreateCubeWireframe(std::vector<float>& cubeVertices, std::vector<uint32_t>& cubeIndices) {
	cubeVertices = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f };
	cubeIndices = {
		0, 1, 1, 2, 2, 3, 3, 0,
		4, 5, 5, 6, 6, 7, 7, 4,
		0, 4, 1, 5, 2, 6, 3, 7
	};
}

void fzbCreateRectangle(std::vector<float>& cubeVertices, std::vector<uint32_t>& cubeIndices, bool world = true) {
	if (world) {
		cubeVertices = { -1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1.0f, -1.0f, 0.0f, -1.0f };
		cubeIndices = {
			0, 1, 2,
			0, 2, 3
		};
	}
	else {
		cubeVertices = { -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f };
		cubeIndices = {
			0, 1, 2,
			0, 2, 3
		};
	}
	
}
//--------------------------------------------------------------------------------------------------------------------

/*
一个meshBatch中的mesh的shader和宏都相同，即使用的资源数量、类型都相同，只是数据不同而已。
因此共享一个描述符池和描述符集合布局，每个mesh维护一个描述符集合
如果后面加入multidraw，则可能由meshBatch维护所有的资源
*/
struct FzbMeshBatch {
public:
	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;

	std::string materialID;
	std::vector<FzbMesh*> meshes;

	uint32_t vertexBufferOffset = 0;
	FzbBuffer indexBuffer;
	bool useSceneDescriptor = true;
	//FzbBuffer materialIndexBuffer;
	//uint32_t drawIndexedIndirectCommandSize = 0;
	//FzbBuffer drawIndexedIndirectCommandBuffer;
	
	FzbMeshBatch() {};
	FzbMeshBatch(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue) {
		this->physicalDevice = physicalDevice;
		this->logicalDevice = logicalDevice;
		this->commandPool = commandPool;
		this->graphicsQueue = graphicsQueue;
	}

	void clean() {
		indexBuffer.clean();
		//materialIndexBuffer.clean();
		//drawIndexedIndirectCommandBuffer.clean();

	}

	void createMeshBatchIndexBuffer(std::vector<uint32_t>& sceneIndices) {
		std::vector<uint32_t> batchIndices;
		for (int i = 0; i < meshes.size(); i++) {
			uint32_t indicesOffset = meshes[i]->indexArrayOffset;
			batchIndices.insert(batchIndices.end(), sceneIndices.begin() + indicesOffset, sceneIndices.begin() + indicesOffset + meshes[i]->indeArraySize);
		}

		indexBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, batchIndices.data(), batchIndices.size() * sizeof(uint32_t));
	}

	/*
	void createDrawIndexedIndirectCommandBuffer() {
		std::vector<VkDrawIndexedIndirectCommand> batchDrawIndexedIndirectCommands;
		for (int i = 0; i < meshes.size(); i++) {
			drawIndexedIndirectCommandSize++;
			batchDrawIndexedIndirectCommands.push_back({ meshes[i]->indeArraySize, meshes[i]->instanceNum, 0, 0, 0 });
		}
		drawIndexedIndirectCommandBuffer = fzbCreateIndirectCommandBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, batchDrawIndexedIndirectCommands.data(), batchDrawIndexedIndirectCommands.size() * sizeof(VkDrawIndexedIndirectCommand));
	}

	void createMeshBatchMaterialBuffer() {
		std::vector<uint32_t> batchMaterials;
		for (int i = 0; i < meshes.size(); i++) {
			batchMaterials.push_back(meshes[i]->materialIndex);
		}
		materialIndexBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, batchMaterials.data(), batchMaterials.size() * sizeof(FzbMeshMaterialUniformObject));
	}

	void createDescriptorSet(VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout) {
		descriptorSet = fzbCreateDescriptorSet(logicalDevice, descriptorPool, descriptorSetLayout);
		std::vector<VkWriteDescriptorSet> meshBatchDescriptorWrites(1);
		VkDescriptorBufferInfo materialIndexStorageBufferInfo{};
		materialIndexStorageBufferInfo.buffer = this->materialIndexBuffer.buffer;
		materialIndexStorageBufferInfo.offset = 0;
		materialIndexStorageBufferInfo.range = sizeof(uint32_t) * meshes.size();
		meshBatchDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		meshBatchDescriptorWrites[0].dstSet = descriptorSet;
		meshBatchDescriptorWrites[0].dstBinding = 0;
		meshBatchDescriptorWrites[0].dstArrayElement = 0;
		meshBatchDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		meshBatchDescriptorWrites[0].descriptorCount = 1;
		meshBatchDescriptorWrites[0].pBufferInfo = &materialIndexStorageBufferInfo;

		vkUpdateDescriptorSets(logicalDevice, meshBatchDescriptorWrites.size(), meshBatchDescriptorWrites.data(), 0, nullptr);
	}
	*/

	void render(VkCommandBuffer commandBuffer, std::map<std::string, FzbMaterial>& sceneMaterials, std::vector<VkDescriptorSet> componentDescriptorSets) {
		vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
		FzbMaterial material = sceneMaterials[this->materialID];
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, material.shader.pipeline);
		//0：组件的uniform,1: material、texture、transform，2：不同meshBatch的materialIndex
		for (int i = 0; i < componentDescriptorSets.size(); i++) {
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, material.shader.pipelineLayout, i, 1, &componentDescriptorSets[i], 0, nullptr);
		}
		for (int i = 0; i < meshes.size(); i++) {
			uint32_t descriptorSetIndex = componentDescriptorSets.size();
			if (material.descriptorSetLayout) {
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, material.shader.pipelineLayout, descriptorSetIndex, 1, &material.descriptorSet, 0, nullptr);
				descriptorSetIndex++;
			} 
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, material.shader.pipelineLayout, descriptorSetIndex, 1, &meshes[i]->descriptorSet, 0, nullptr);
			vkCmdDrawIndexed(commandBuffer, meshes[i]->indeArraySize, meshes[i]->instanceNum, 0, 0, 0);
		}
		//vkCmdDrawIndexedIndirect(commandBuffer, drawIndexedIndirectCommandBuffer.buffer, 0, drawIndexedIndirectCommandSize, sizeof(VkDrawIndexedIndirectCommand));
	}


};

struct FzbLight {

};

#endif