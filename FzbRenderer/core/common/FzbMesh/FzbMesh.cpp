#include "../FzbMaterial/FzbMaterial.h"
#include "./FzbMesh.h"

#include <iostream>
#include "../FzbRenderer.h"
#include "CUDA/FzbMeshCUDA.cuh"
#include "FzbCreateSimpleMesh.h"

float FzbAABBBox::getAxis(uint32_t k) {
	if (k > 5) throw std::out_of_range("getAxis: AABB没有第" + std::to_string(k) + "轴");
	switch (k) {
		case 0: return leftX;
		case 1: return rightX;
		case 2: return leftY;
		case 3: return rightY;
		case 4: return leftZ;
		case 5: return rightZ;
	}
	return 0.0f;
}
void FzbAABBBox::setAxis(uint32_t k, float value) {
	if (k > 5) throw("setAxis: AABB没有第" + std::to_string(k) + "轴");
	switch (k) {
		case 0: this->leftX = value; break;
		case 1: this->rightX = value; break;
		case 2: this->leftY = value; break;
		case 3: this->rightY = value; break;
		case 4: this->leftZ = value; break;
		case 5: this->rightZ = value; break;
	}
}
void FzbAABBBox::createDistanceAndCenter(bool useCube, float expand) {
	this->distanceX = (this->rightX - this->leftX) * expand;
	this->distanceY = (this->rightY - this->leftY) * expand;
	this->distanceZ = (this->rightZ - this->leftZ) * expand;
	this->distance = glm::max(distanceX, glm::max(distanceY, distanceZ));

	this->centerX = (this->rightX + this->leftX) * 0.5f;
	this->centerY = (this->rightY + this->leftY) * 0.5f;
	this->centerZ = (this->rightZ + this->leftZ) * 0.5f;
}
bool FzbAABBBox::isEmpty() {
	return leftX == FLT_MAX && rightX == -FLT_MAX && leftY == FLT_MAX && rightY == -FLT_MAX && leftZ == FLT_MAX && rightZ == -FLT_MAX;
}

//----------------------------------------------------mesh--------------------------------------------
FzbMesh::FzbMesh(FzbVertexFormat vertexFormat) {
	this->vertexFormat = vertexFormat;
};
std::vector<float> FzbMesh::getVertices(FzbVertexFormat vertexFormat) {
	if(vertexFormat == this->vertexFormat) return this->vertices;
	if (this->vertices.size() >= 4096) {
		return getVertices_CUDA(this->vertices, this->vertexFormat, vertexFormat);
	}
	uint32_t vertexSize = this->vertexFormat.getVertexSize();
	uint32_t vertexNum = this->vertices.size() / vertexSize;	//mesh有几个顶点
	uint32_t newVerticesFloatNum = vertexFormat.getVertexSize() * vertexNum;
	std::vector<float> vertexData;
	vertexData.reserve(newVerticesFloatNum);
	for (int i = 0; i < vertexNum; i++) {
		uint32_t vertexFloatOffset = i * vertexSize;
		vertexData.insert(vertexData.end(), this->vertices.begin() + vertexFloatOffset, this->vertices.begin() + vertexFloatOffset + 3);	//加入pos
		vertexFloatOffset += 3;
		if (vertexFormat.useNormal) {
			if (!this->vertexFormat.useNormal) {
				//throw std::runtime_error("mesh " + id + "没有normal，没法获取");
				vertexData.push_back(0.0f); vertexData.push_back(0.0f); vertexData.push_back(0.0f);
			}
			else {
				vertexData.insert(vertexData.end(), this->vertices.begin() + vertexFloatOffset, this->vertices.begin() + vertexFloatOffset + 3);	//加入normal
				vertexFloatOffset += 3;
			}
		}else if(this->vertexFormat.useNormal) vertexFloatOffset += 3;
		if (vertexFormat.useTexCoord) {
			if (!this->vertexFormat.useTexCoord) {
				//throw std::runtime_error("mesh " + id + "没有texCoords，没法获取");
				vertexData.push_back(0.0f); vertexData.push_back(0.0f);
			}
			else {
				vertexData.insert(vertexData.end(), this->vertices.begin() + vertexFloatOffset, this->vertices.begin() + vertexFloatOffset + 2);	//加入texCoords
				vertexFloatOffset += 2;
			}
		}
		else if (this->vertexFormat.useTexCoord) vertexFloatOffset += 2;
		if (vertexFormat.useTangent) {
			if (!this->vertexFormat.useTangent) {
				//throw std::runtime_error("mesh " + id + "没有tangent，没法获取");
				vertexData.push_back(0.0f); vertexData.push_back(0.0f); vertexData.push_back(0.0f); vertexData.push_back(0.0f);
			}
			else {
				vertexData.insert(vertexData.end(), this->vertices.begin() + vertexFloatOffset, this->vertices.begin() + vertexFloatOffset + 4);	//加入tangent
				vertexFloatOffset += 4;
			}
		}
	}
	return vertexData;
}
void FzbMesh::clean() {}
void FzbMesh::render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetIndex) {
	vkCmdDrawIndexed(commandBuffer, this->indexArraySize, this->instanceNum, indexArrayOffset, 0, 0);
}
FzbAABBBox FzbMesh::getAABB() {
	if (this->AABB.isEmpty()) this->createAABB();
	return this->AABB;
}
void FzbMesh::createAABB() {
	if (this->vertices.size() >= 4096) {
		this->AABB = createAABB_CUDA(this->vertices, this->vertexFormat);
		return;
	}

	uint32_t vertexSize = this->vertexFormat.getVertexSize();
	this->AABB = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
	for (int i = 0; i < this->vertices.size(); i += vertexSize) {
		float x = vertices[i];
		float y = vertices[i + 1];
		float z = vertices[i + 2];
		glm::vec3 pos = glm::vec3(x, y, z);
		x = pos.x; y = pos.y; z = pos.z;
		AABB.leftX = x < AABB.leftX ? x : AABB.leftX;
		AABB.rightX = x > AABB.rightX ? x : AABB.rightX;
		AABB.leftY = y < AABB.leftY ? y : AABB.leftY;
		AABB.rightY = y > AABB.rightY ? y : AABB.rightY;
		AABB.leftZ = z < AABB.leftZ ? z : AABB.leftZ;
		AABB.rightZ = z > AABB.rightZ ? z : AABB.rightZ;
	}
	//对于面，我们给个0.02的宽度
	//if (AABB.leftX == AABB.rightX) {
	//	AABB.leftX = AABB.leftX - 0.01;
	//	AABB.rightX = AABB.rightX + 0.01;
	//}
	//if (AABB.leftY == AABB.rightY) {
	//	AABB.leftY = AABB.leftY - 0.01;
	//	AABB.rightY = AABB.rightY + 0.01;
	//}
	//if (AABB.leftZ == AABB.rightZ) {
	//	AABB.leftZ = AABB.leftZ - 0.01;
	//	AABB.rightZ = AABB.rightZ + 0.01;
	//}
}
//---------------------------------------------------dynamicMesh---------------------------------------
FzbMeshDynamic::FzbMeshDynamic() {
	this->transforms = glm::mat4(1.0f);
	this->vertexFormat = FzbVertexFormat();
};
void FzbMeshDynamic::createBuffer() {
	FzbMeshUniformBufferObject uniformBufferObject;
	uniformBufferObject.transforms = this->transforms;
	this->meshBuffer = fzbCreateUniformBuffer(sizeof(FzbMeshUniformBufferObject));
	memcpy(this->meshBuffer.mapped, &uniformBufferObject, sizeof(FzbMeshUniformBufferObject));
}
void FzbMeshDynamic::createDescriptor() {
	//this->descriptorSetLayout = fzbCreateDescriptLayout(logicalDevice, 1, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }, { VK_SHADER_STAGE_ALL });
	this->descriptorSet = fzbCreateDescriptorSet(FzbRenderer::globalData.descriptorPool, meshDescriptorSetLayout);
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

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, voxelGridMapDescriptorWrites.size(), voxelGridMapDescriptorWrites.data(), 0, nullptr);
}
void FzbMeshDynamic::render(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, uint32_t descriptorSetIndex) {
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, descriptorSetIndex, 1, &this->descriptorSet, 0, nullptr);
	vkCmdDrawIndexed(commandBuffer, this->indexArraySize, this->instanceNum, indexArrayOffset, 0, 0);
}
void FzbMeshDynamic::clean() {
	FzbMesh::clean();
	meshBuffer.clean();
	if (meshDescriptorSetLayout) vkDestroyDescriptorSetLayout(FzbRenderer::globalData.logicalDevice, meshDescriptorSetLayout, nullptr);
}
void fzbCreateMeshDescriptor() {
	FzbMeshDynamic::meshDescriptorSetLayout = fzbCreateDescriptLayout(1, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }, { VK_SHADER_STAGE_ALL });
}
//---------------------------------------------------加载mesh--------------------------------------------
void loadMaterialTextures(FzbMaterial& material, aiMaterial* mat, aiTextureType type, std::string typeName) {
	for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
	{
		aiString path;
		mat->GetTexture(type, i, &path);
		std::string texturePath = path.C_Str();
		//./scenes/sceneName/textures/xxx.jpg，这规定我们需要将texture放到models同级目录的textures下，这是为了与没有mtl的scene匹配
		texturePath = "textures/" + texturePath;	
		FzbTexture texture;
		texture.path = path.C_Str();	
		material.properties.textureProperties.insert({ typeName, texture });
	}
}
/*
如果obj文件带有mtl文件作为材质信息，那么我们就根据mtl文件创建相应的material，存入mainScene的sceneMaterials中
也可以给一个material数组引用，不一定是mainScene的
*/
FzbMaterial getMaterialFromMTL(aiMaterial* materialData) {
	FzbMaterial material;
	loadMaterialTextures(material, materialData, aiTextureType_DIFFUSE, "albedoMap");
	//loadMaterialTextures(material, materialData, aiTextureType_SPECULAR, "specularMap");
	loadMaterialTextures(material, materialData, aiTextureType_HEIGHT, "normalMap");

	material.vertexFormat.useTexCoord = material.properties.textureProperties.size() > 0;

	aiColor3D color;
	//materialData->Get(AI_MATKEY_COLOR_AMBIENT, color);
	materialData->Get(AI_MATKEY_COLOR_DIFFUSE, color);
	material.properties.numberProperties.insert({ "albedo",  { glm::vec4(color.r, color.g, color.b, 1.0) } });
	//materialData->Get(AI_MATKEY_COLOR_SPECULAR, color);
	materialData->Get(AI_MATKEY_COLOR_EMISSIVE, color);
	material.properties.numberProperties.insert({ "emissive", { glm::vec4(color.r, color.g, color.b, 1.0) } });

	int illumModel = 0;
	//materialData->Get(AI_MATKEY_SHADING_MODEL, illumModel);
	materialData->Get("$mat.illum", 0, 0, illumModel);
	if (illumModel == 0) {	//diffuse
		material.type = "diffuse";
	}
	else if (illumModel == 1) {	//roughconductor
		material.type = "roughconductor";
		float alpha = 1.0f;
		materialData->Get(AI_MATKEY_OPACITY, alpha);
		material.properties.numberProperties.insert({ "bsdfPara", { glm::vec4(alpha, 0.0f, 0.0f, 0.0f) } });
	}
	else if (illumModel == 2) {
		material.type = "roughdielectric";
		float alpha = 1.0f;
		materialData->Get(AI_MATKEY_OPACITY, alpha);
		float eta = 1.5f;
		materialData->Get(AI_MATKEY_REFRACTI, eta);
		material.properties.numberProperties.insert({ "bsdfPara", { glm::vec4(alpha, eta, 0.0f, 0.0f) } });
	}
	
	return material;
}
FzbMesh processMesh(aiMesh* meshData, const aiScene* sceneData, FzbVertexFormat vertexFormat, glm::mat4 transformMatrix, std::map<std::string, FzbMaterial>* sceneMaterials) {
	FzbMesh fzbMesh;
	if (vertexFormat.useNormal) {
		if (meshData->HasNormals()) fzbMesh.vertexFormat.useNormal = true;
		else throw std::runtime_error("该mesh没有法线属性");
	}
	if (vertexFormat.useTexCoord) {
		if (meshData->mTextureCoords[0]) fzbMesh.vertexFormat.useTexCoord = true;
		else throw std::runtime_error("该mesh没有uv属性");
	}
	if (vertexFormat.useTangent) {
		if (meshData->HasTangentsAndBitangents()) fzbMesh.vertexFormat.useTangent = true;
		else throw std::runtime_error("该mesh没有切线属性");
	}

	uint32_t vertexNum = meshData->mNumVertices;
	if (vertexNum >= 4096) fzbMesh = createVertices_CUDA(meshData, fzbMesh.vertexFormat);
	else {
		uint32_t verticesFloatNum = fzbMesh.vertexFormat.getVertexSize() * vertexNum;
		fzbMesh.vertices.reserve(verticesFloatNum);
		for (uint32_t i = 0; i < vertexNum; i++) {
			fzbMesh.vertices.emplace_back(meshData->mVertices[i].x);
			fzbMesh.vertices.emplace_back(meshData->mVertices[i].y);
			fzbMesh.vertices.emplace_back(meshData->mVertices[i].z);
			if (vertexFormat.useNormal) {
				fzbMesh.vertices.emplace_back(meshData->mNormals[i].x);
				fzbMesh.vertices.emplace_back(meshData->mNormals[i].y);
				fzbMesh.vertices.emplace_back(meshData->mNormals[i].z);
			}
			if (vertexFormat.useTexCoord) {
				fzbMesh.vertices.emplace_back(meshData->mTextureCoords[0][i].x);
				fzbMesh.vertices.emplace_back(meshData->mTextureCoords[0][i].y);
			}
			if (vertexFormat.useTangent) {
				glm::vec3 T(meshData->mTangents[i].x, meshData->mTangents[i].y, meshData->mTangents[i].z);
				glm::vec3 B(meshData->mBitangents[i].x, meshData->mBitangents[i].y, meshData->mBitangents[i].z);
				glm::vec3 N(meshData->mNormals[i].x, meshData->mNormals[i].y, meshData->mNormals[i].z);

				T = glm::normalize(T);
				N = glm::normalize(N);
				float handed = (glm::dot(glm::cross(N, T), B) < 0.0f) ? -1.0f : 1.0f;

				fzbMesh.vertices.emplace_back(meshData->mTangents[i].x);
				fzbMesh.vertices.emplace_back(meshData->mTangents[i].y);
				fzbMesh.vertices.emplace_back(meshData->mTangents[i].z);
				fzbMesh.vertices.emplace_back(handed);
			}
		}
	}
	verticesTransform_CUDA(fzbMesh.vertices, transformMatrix, fzbMesh.vertexFormat);

	//indices的复制不适合CUDA，因为可能一个aiFace中的面很少，放到GPU里不划算
	uint32_t faceNum = meshData->mNumFaces;
	uint32_t indexNum = 0;
	for (int i = 0; i < faceNum; ++i) indexNum += meshData->mFaces[i].mNumIndices;
	fzbMesh.indices.resize(indexNum);
	uint32_t offset = 0;
	for (uint32_t i = 0; i < faceNum; i++) {
		aiFace& face = meshData->mFaces[i];
		std::memcpy(fzbMesh.indices.data() + offset, face.mIndices, sizeof(uint32_t) * face.mNumIndices);
		offset += face.mNumIndices;
	}
	fzbMesh.indexArraySize = fzbMesh.indices.size();

	if (sceneData->mNumMaterials > 1) {		//有一个默认材质
		aiMaterial* materialData = sceneData->mMaterials[meshData->mMaterialIndex];
		FzbMaterial material = getMaterialFromMTL(materialData);
		material.id = std::string(meshData->mName.C_Str()) + "Material";
		sceneMaterials->insert({ material.id, material });
		fzbMesh.material = &sceneMaterials->at(material.id);
		fzbMesh.mtlMaterial = true;
	}

	return fzbMesh;
}
std::vector<FzbMesh> processNode(aiNode* node, const aiScene* sceneData, FzbVertexFormat vertexFormat, glm::mat4 transformMatrix, std::map<std::string, FzbMaterial>* sceneMaterials) {
	std::vector<FzbMesh> meshes;
	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh* meshData = sceneData->mMeshes[node->mMeshes[i]];
		meshes.push_back(processMesh(meshData, sceneData, vertexFormat, transformMatrix, sceneMaterials));
	}

	for (uint32_t i = 0; i < node->mNumChildren; i++) {
		std::vector<FzbMesh> results = processNode(node->mChildren[i], sceneData, vertexFormat, transformMatrix, sceneMaterials);
		meshes.insert(meshes.begin(), results.begin(), results.end());
	}
	return meshes;
}
std::vector<FzbMesh> fzbGetMeshFromOBJ(std::string path, FzbVertexFormat vertexFormat, glm::mat4 transformMatrix, std::map<std::string, FzbMaterial>* sceneMaterials) {
	Assimp::Importer import;
	uint32_t needs = aiProcess_Triangulate |
		(vertexFormat.useTexCoord ? aiProcess_FlipUVs : aiPostProcessSteps(0u)) |
		(vertexFormat.useNormal ? aiProcess_GenSmoothNormals : aiPostProcessSteps(0u)) |
		(vertexFormat.useTangent ? aiProcess_CalcTangentSpace : aiPostProcessSteps(0u));
	const aiScene* sceneData = import.ReadFile(path, needs);	//相对地址从main程序目录开始，也就是在FzbRenderer下，所以相对地址要从那里开始

	if (!sceneData || sceneData->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !sceneData->mRootNode) {
		std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
		throw std::runtime_error("ERROR::ASSIMP::" + (std::string)import.GetErrorString());
	}

	//std::string  meshPathDirectory = path.substr(0, path.find_last_of('/'));
	if (sceneMaterials == nullptr) sceneMaterials = &FzbRenderer::globalData.mainScene.sceneMaterials;
	std::vector<FzbMesh> meshes = processNode(sceneData->mRootNode, sceneData, vertexFormat, transformMatrix, sceneMaterials);
	return meshes;
}

/*
void fzbCreateCube(FzbMesh& mesh) {
	mesh.vertices = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f };
	mesh.indices = {
					1, 0, 3, 1, 3, 2,
					4, 5, 6, 4, 6, 7,
					5, 1, 2, 5, 2, 6,
					0, 4, 7, 0, 7, 3,
					7, 6, 2, 7, 2, 3,
					0, 1, 5, 0, 5, 4
	};
	mesh.indexArraySize = mesh.indices.size();
	mesh.vertexFormat = FzbVertexFormat();
}
void fzbCreateCubeWireframe(FzbMesh& mesh) {
	mesh.vertices = { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f };
	mesh.indices = {
		0, 1, 1, 2, 2, 3, 3, 0,
		4, 5, 5, 6, 6, 7, 7, 4,
		0, 4, 1, 5, 2, 6, 3, 7
	};
	mesh.indexArraySize = mesh.indices.size();
	mesh.vertexFormat = FzbVertexFormat();
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
*/
//-------------------------------------------------meshBatch-----------------------------------------------
FzbMeshBatch::FzbMeshBatch() {};
void FzbMeshBatch::clean() {
	//indexBuffer.clean();
	//materialIndexBuffer.clean();
	//drawIndexedIndirectCommandBuffer.clean();

}
/*
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
*/
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

FzbLight::FzbLight() {};
FzbLight::FzbLight(glm::vec3 position, glm::mat4 viewMatrix) {
	this->position = position;
	this->viewMatrix = viewMatrix;
}
FzbLight::FzbLight(std::string dataPath, glm::mat4 modelMatrix) {
	this->type = FZB_AREA;
	std::vector<FzbMesh> lights = fzbGetMeshFromOBJ(dataPath, FzbVertexFormat(true), modelMatrix);
	FzbMesh lightMesh = lights[0];
	std::vector<glm::vec3> vertexPos(lightMesh.indices.size());
	for (int i = 0; i < lightMesh.indices.size(); ++i) {
		uint32_t startIndex = lightMesh.indices[i] * 6;
		vertexPos[i] = glm::vec3(lightMesh.vertices[startIndex], lightMesh.vertices[startIndex + 1], lightMesh.vertices[startIndex + 2]);
	}
	this->normal = glm::vec3(lightMesh.vertices[3], lightMesh.vertices[4], lightMesh.vertices[5]);
	this->edge0 = vertexPos[1] - vertexPos[0];
	this->edge1 = vertexPos[vertexPos.size() - 1] - vertexPos[0];
	this->position = vertexPos[0];
	this->area = glm::length(this->edge0) * glm::length(this->edge1);
	this->material = lightMesh.material;
	this->strength = this->material->properties.numberProperties["emissive"].value;
}
FzbLight::FzbLight(glm::mat4 modelMatrix) {
	FzbMesh lightMesh;
	fzbCreateRectangle(lightMesh, FzbVertexFormat(true), modelMatrix);
	this->normal = glm::vec3(lightMesh.vertices[3], lightMesh.vertices[4], lightMesh.vertices[5]);
	this->edge0 = glm::vec3(lightMesh.vertices[6] - lightMesh.vertices[0], lightMesh.vertices[7] - lightMesh.vertices[1], lightMesh.vertices[8] - lightMesh.vertices[2]);
	this->edge1 = glm::vec3(lightMesh.vertices[18] - lightMesh.vertices[0], lightMesh.vertices[19] - lightMesh.vertices[1], lightMesh.vertices[20] - lightMesh.vertices[2]);
	this->position = glm::vec3(lightMesh.vertices[0], lightMesh.vertices[1], lightMesh.vertices[2]);
	this->area = glm::length(this->edge0) * glm::length(this->edge1);
};