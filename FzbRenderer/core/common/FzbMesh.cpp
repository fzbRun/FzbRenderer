#include "FzbMaterial.h"
#include "./FzbMesh.h"

#include <iostream>

float FzbAABBBox::getAxis(int k) {
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
bool FzbAABBBox::isEmpty() {
	return leftX == FLT_MAX && rightX == -FLT_MAX && leftY == FLT_MAX && rightY == -FLT_MAX && leftZ == FLT_MAX && rightZ == -FLT_MAX;
}

//----------------------------------------------------mesh--------------------------------------------
FzbMesh::FzbMesh() {
	this->transforms = glm::mat4(1.0f);
	this->vertexFormat = FzbVertexFormat();
};
FzbMesh::FzbMesh(VkDevice logicalDevice) {
	this->logicalDevice = logicalDevice;
}
std::vector<float> FzbMesh::getVetices() {
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
void FzbMesh::clean() {
	meshBuffer.clean();
	//vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
}
void FzbMesh::createBuffer(VkPhysicalDevice physicalDevice) {
	FzbMeshUniformBufferObject uniformBufferObject;
	uniformBufferObject.transforms = this->transforms;
	this->meshBuffer = fzbCreateUniformBuffers(physicalDevice, logicalDevice, sizeof(FzbMeshUniformBufferObject));
	memcpy(this->meshBuffer.mapped, &uniformBufferObject, sizeof(FzbMeshUniformBufferObject));
}
void FzbMesh::createDescriptor(VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout) {
	//this->descriptorSetLayout = fzbCreateDescriptLayout(logicalDevice, 1, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }, { VK_SHADER_STAGE_ALL });
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
void FzbMesh::createAABB() {
	uint32_t vertexSize = this->vertexFormat.getVertexSize() / sizeof(float);

	this->AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	for (int i = 0; i < this->vertices.size(); i += vertexSize) {
		float x = vertices[i];
		float y = vertices[i + 1];
		float z = vertices[i + 2];
		AABB.leftX = x < AABB.leftX ? x : AABB.leftX;
		AABB.rightX = x > AABB.rightX ? x : AABB.rightX;
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
void FzbMesh::render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetIndex) {
	//if (this->material->descriptorSet) vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, descriptorSetIndex++, 1, &this->material->descriptorSet, 0, nullptr);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, descriptorSetIndex, 1, &this->descriptorSet, 0, nullptr);
	vkCmdDrawIndexed(commandBuffer, this->indeArraySize, this->instanceNum, indexArrayOffset, 0, 0);
}

void createMeshDescriptor(VkDevice logicalDevice, VkDescriptorSetLayout& descriptorSetLayout) {
	descriptorSetLayout = fzbCreateDescriptLayout(logicalDevice, 1, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }, { VK_SHADER_STAGE_ALL });
}
//---------------------------------------------------加载mesh--------------------------------------------
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
FzbMesh processMesh(aiMesh* mesh, const aiScene* scene, FzbVertexFormat vertexFormat) {

	//FzbVertexFormat materialVertexFormat = material.getVertexFormat();

	FzbMesh fzbMesh;
	for (uint32_t i = 0; i < mesh->mNumVertices; i++) {

		fzbMesh.vertices.push_back(mesh->mVertices[i].x);
		fzbMesh.vertices.push_back(mesh->mVertices[i].y);
		fzbMesh.vertices.push_back(mesh->mVertices[i].z);

		if (vertexFormat.useNormal) {
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

		if (vertexFormat.useTexCoord) {
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

		if (vertexFormat.useTangent) {
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

		//material.changeVertexFormat(fzbMesh.vertexFormat);	//根据mesh的实际顶点格式修改材质和shader的顶点格式
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
std::vector<FzbMesh> processNode(aiNode* node, const aiScene* scene, FzbVertexFormat vertexFormat) {

	std::vector<FzbMesh> meshes;
	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		meshes.push_back(processMesh(mesh, scene, vertexFormat));
	}

	for (uint32_t i = 0; i < node->mNumChildren; i++) {
		std::vector<FzbMesh> results = processNode(node->mChildren[i], scene, vertexFormat);
		meshes.insert(meshes.begin(), results.begin(), results.end());
	}

	return meshes;
}
std::vector<FzbMesh> fzbGetMeshFromOBJ(VkDevice logicalDevice, std::string path, FzbVertexFormat vertexFormat) {
	Assimp::Importer import;
	uint32_t needs = aiProcess_Triangulate |
		(vertexFormat.useTexCoord ? aiProcess_FlipUVs : aiPostProcessSteps(0u)) |
		(vertexFormat.useNormal ? aiProcess_GenSmoothNormals : aiPostProcessSteps(0u)) |
		(vertexFormat.useTangent ? aiProcess_CalcTangentSpace : aiPostProcessSteps(0u));
	const aiScene* scene = import.ReadFile(path, needs);	//相对地址从main程序目录开始，也就是在FzbRenderer下，所以相对地址要从那里开始

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
		throw std::runtime_error("ERROR::ASSIMP::" + (std::string)import.GetErrorString());
	}

	//std::string  meshPathDirectory = path.substr(0, path.find_last_of('/'));
	std::vector<FzbMesh> meshes = processNode(scene->mRootNode, scene, vertexFormat);
	for (int i = 0; i < meshes.size(); i++) meshes[i].logicalDevice = logicalDevice;
	return meshes;

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
void fzbCreateRectangle(std::vector<float>& cubeVertices, std::vector<uint32_t>& cubeIndices, bool world) {
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

//-------------------------------------------------meshBatch-----------------------------------------------
FzbMeshBatch::FzbMeshBatch() {};
FzbMeshBatch::FzbMeshBatch(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue) {
	this->physicalDevice = physicalDevice;
	this->logicalDevice = logicalDevice;
	this->commandPool = commandPool;
	this->graphicsQueue = graphicsQueue;
}
void FzbMeshBatch::clean() {
	indexBuffer.clean();
	//materialIndexBuffer.clean();
	//drawIndexedIndirectCommandBuffer.clean();

}
void FzbMeshBatch::createMeshBatchIndexBuffer(std::vector<uint32_t>& sceneIndices) {
	if (this->meshes.size() == 0) return;
	std::vector<uint32_t> batchIndices;
	for (int i = 0; i < meshes.size(); i++) {
		uint32_t indicesOffset = meshes[i]->indexArrayOffset;
		//meshes[i]->indexOffsetInMeshBatchIndexArray = batchIndices.size();
		batchIndices.insert(batchIndices.end(), sceneIndices.begin() + indicesOffset, sceneIndices.begin() + indicesOffset + meshes[i]->indeArraySize);
	}
	
	indexBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, batchIndices.data(), batchIndices.size() * sizeof(uint32_t));
}
void FzbMeshBatch::render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t componentDescriptorSetNum) {
	//vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
	if (materials[0]->descriptorSet && this->useSameMaterial) vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, componentDescriptorSetNum++, 1, &materials[0]->descriptorSet, 0, nullptr);
	for (int i = 0; i < meshes.size(); i++) {
		uint32_t descriptorSetIndex = componentDescriptorSetNum;
		if (!this->useSameMaterial && materials[i]->descriptorSet) {
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, componentDescriptorSetNum, 1, &materials[i]->descriptorSet, 0, nullptr);
			descriptorSetIndex++;
		}
		meshes[i]->render(commandBuffer, pipelineLayout, descriptorSetIndex);
	}
	//vkCmdDrawIndexedIndirect(commandBuffer, drawIndexedIndirectCommandBuffer.buffer, 0, drawIndexedIndirectCommandSize, sizeof(VkDrawIndexedIndirectCommand));
}

FzbLight::FzbLight(glm::vec3 position, glm::vec3 strength, glm::mat4 viewMatrix) {
	this->position = position;
	this->strength = strength;
	this->viewMatrix = viewMatrix;
}