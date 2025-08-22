#pragma once

#include "StructSet.h"
#include "FzbBuffer.h"
#include "FzbPipeline.h"
#include "FzbDescriptor.h"

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

	float getAxis(int k);

	bool isEmpty();

};

struct FzbMeshUniformBufferObject {
	glm::mat4 transforms;
};

struct FzbMaterial;

struct FzbMesh {
public:
	VkDevice logicalDevice;

	std::string id;
	std::string path;
	std::vector<float> vertices;	//压缩前的顶点数据，压缩后就会被释放
	std::vector<uint32_t> indices;	//压缩前的顶点索引数据，压缩后就会被释放
	uint32_t indexArrayOffset;
	uint32_t indeArraySize;
	//uint32_t indexOffsetInMeshBatchIndexArray;

	glm::mat4 transforms;	//一个mesh对应一种变换
	//std::string materialID;
	FzbMaterial* material;	//sceneXML中指定的material
	FzbVertexFormat vertexFormat;	//从obj中获取到的mesh的顶点格式

	FzbBuffer meshBuffer;
	//VkDescriptorSetLayout descriptorSetLayout = nullptr;
	VkDescriptorSet descriptorSet = nullptr;	//model矩阵

	uint32_t instanceNum = 1;
	FzbAABBBox AABB;

	FzbMesh();
	FzbMesh(VkDevice logicalDevice);

	std::vector<float> getVetices();

	void clean();

	void createBuffer(VkPhysicalDevice physicalDevice);

	void createDescriptor(VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout);

	void createAABB();

	void render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetIndex);
};

void createMeshDescriptor(VkDevice logicalDevice, VkDescriptorSetLayout& descriptorSetLayout);

std::vector<FzbTexture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName);

FzbMesh processMesh(aiMesh* mesh, const aiScene* scene, FzbVertexFormat vertexFormat);

std::vector<FzbMesh> processNode(aiNode* node, const aiScene* scene, FzbVertexFormat vertexFormat);

std::vector<FzbMesh> fzbGetMeshFromOBJ(VkDevice logicalDevice, std::string path, FzbVertexFormat vertexFormat);

void fzbCreateCube(std::vector<float>& cubeVertices, std::vector<uint32_t>& cubeIndices);

void fzbCreateCubeWireframe(std::vector<float>& cubeVertices, std::vector<uint32_t>& cubeIndices);

void fzbCreateRectangle(std::vector<float>& cubeVertices, std::vector<uint32_t>& cubeIndices, bool world = true);
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

	std::vector<FzbMesh*> meshes;
	bool useSameMaterial = false;
	std::vector<FzbMaterial*> materials;

	uint32_t vertexBufferOffset = 0;
	FzbBuffer indexBuffer;
	//FzbBuffer materialIndexBuffer;
	//uint32_t drawIndexedIndirectCommandSize = 0;
	//FzbBuffer drawIndexedIndirectCommandBuffer;
	
	FzbMeshBatch();
	FzbMeshBatch(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue);

	void clean();

	void createMeshBatchIndexBuffer(std::vector<uint32_t>& sceneIndices);

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

	void render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t componentDescriptorSetNum);


};

enum FzbLIghtType {
	FZB_POINT = 0,
	FZB_SPOT = 1,
	FZB_PARALLEL = 2,
	FZB_AREA = 3
};
struct FzbLight {
public:
	FzbLIghtType type = FZB_POINT;
	glm::vec3 position;
	glm::vec3 strength;
	glm::mat4 viewMatrix;
	glm::mat4 projMatrix;

	FzbLight(glm::vec3 position, glm::vec3 strength, glm::mat4 viewMatrix = glm::mat4());
};

#endif