#pragma once

#include <glm/ext/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <optional>
#include <iostream>
#include <unordered_map>

#include "FzbMesh.h"
#include "FzbImage.h"
#include "FzbDescriptor.h"

#ifndef FZB_SCENE_H
#define FZB_SCENE_H

/*
namespace std {
	template<> struct hash<FzbVertex> {
		size_t operator()(FzbVertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1) ^ (hash<glm::vec3>()(vertex.tangent) << 1);
		}
	};
	template<> struct hash<FzbVertex_OnlyPos> {
		size_t operator()(FzbVertex_OnlyPos const& vertex) const {
			// ������ pos �Ĺ�ϣֵ
			return hash<glm::vec3>()(vertex.pos);
		}
	};
	template<> struct hash<FzbVertex_PosNormal> {
		size_t operator()(FzbVertex_PosNormal const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1);
		}
	};
	template<> struct hash<FzbVertex_PosNormalTexCoord> {
		size_t operator()(FzbVertex_PosNormalTexCoord const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}
*/
struct VectorFloatHash {
	size_t operator()(std::vector<float> const& v) const noexcept {
		// ������������Ϊ��ʼ����
		size_t seed = v.size();
		for (float f : v) {
			// ��׼��� float ��ϣ
			size_t h = std::hash<float>()(f);
			// ����� hash_combine
			seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
		}
		return seed;
	}
};

struct Mat4Hash {
	size_t operator()(glm::mat4 const& m) const noexcept {
		// �� 16 �� float ��������ϣ
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
		// �����ֻ���ϸ���Ԫ����ȣ���ֱ���� ==  
		// ��Ϊ�˱��գ�Ҳ�����ֶ��ȣ�
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
/*
���ｲһ�����������Ĺ�������
1. ����scene
2. ͨ��addMeshToScene��mesh����scene��Ȼ��õ�mesh
3. ��mesh����shader������shader��Ҫ���޸Ķ����ʽ�Լ���������
4. ���޸ĺ��mesh����differentVertexFormatMeshIndexs
5. ����differentVertexFormatMeshIndexsѹ����ͬ�����ʽ��mesh�Ķ������ݣ�Ȼ�����sceneVertices��sceneIndices
6. ����mesh��shader����meshBatch
7. ÿ��shader����pipeline
8. ��Ⱦ��
*/
struct FzbScene {

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;

	std::vector<FzbMesh> sceneMeshSet;	//��ͬ�����ʽ��mesh
	//std::vector<FzbMeshBatch> sceneMeshBatchs;
	std::map<FzbVertexFormat, std::vector<uint32_t>, FzbVertexFormatLess> differentVertexFormatMeshIndexs;

	std::vector<float> sceneVertices;	//ѹ����Ķ�������
	std::vector<uint32_t> sceneIndices;

	std::vector<glm::mat4> sceneTransformMatrixs;
	std::vector<FzbMaterial> sceneMaterials;
	std::vector<FzbTexture> sceneAlbedoTextures;
	std::vector<FzbTexture> sceneNormalTextures;

	FzbBuffer vertexBuffer;
	FzbBuffer materialBuffer;
	FzbBuffer transformBuffer;
	
	std::vector<FzbImage> albedoImages;
	std::vector<FzbImage> normalImages;

	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout sceneDecriptorSetLayout;
	VkDescriptorSet descriptorSet;

	FzbAABBBox AABB;

	std::string meshPathDirectory;

	FzbScene() {};

	FzbScene(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue) {
		this->physicalDevice = physicalDevice;
		this->logicalDevice = logicalDevice;
		this->commandPool = commandPool;
		this->graphicsQueue = graphicsQueue;
	}

	void clean() {
		vertexBuffer.clean();
		materialBuffer.clean();
		transformBuffer.clean();

		for (int i = 0; i < albedoImages.size(); i++) {
			albedoImages[i].clean();
		}
		for (int i = 0; i < normalImages.size(); i++) {
			normalImages[i].clean();
		}

		vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, sceneDecriptorSetLayout, nullptr);

		for (int i = 0; i < this->sceneMeshSet.size(); i++) {
			sceneMeshSet[i].clean();
		}
	}

	/*
	//һ��node����mesh����node��������Ҫ�ݹ飬�����е�mesh���ó���
	//���е�ʵ�����ݶ���scene�У���node�д洢����scene������
	std::vector<uint32_t> addMeshToScene(std::string path) {

		Assimp::Importer import;
		const aiScene* scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
			std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
			throw std::runtime_error("ERROR::ASSIMP::" + (std::string)import.GetErrorString());
		}

		uint32_t lastMeshSize = this->sceneMeshSet.size();
		meshPathDirectory = path.substr(0, path.find_last_of('/'));
		processNode(scene->mRootNode, scene);
		uint32_t curMeshSize = this->sceneMeshSet.size();

		std::vector<uint32_t> meshIndexs;
		for (int i = lastMeshSize; i < curMeshSize; i++) {
			meshIndexs.push_back(i);
		}
		return meshIndexs;
	}
	void processNode(aiNode* node, const aiScene* scene) {

		for (uint32_t i = 0; i < node->mNumMeshes; i++) {
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			processMesh(mesh, scene);
		}

		for (uint32_t i = 0; i < node->mNumChildren; i++) {
			processNode(node->mChildren[i], scene);
		}

	}
	void processMesh(aiMesh* mesh, const aiScene* scene) {

		this->sceneMeshSet.push_back(FzbMesh());
		FzbMesh& fzbMesh = this->sceneMeshSet[this->sceneMeshSet.size() - 1];
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

			if (mesh->mTextureCoords[0]) // �����Ƿ����������ꣿ����ֻ����һ������uv
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
		this->differentVertexFormatMeshIndexs[fzbMesh.vertexFormat].push_back(this->sceneMeshSet.size() - 1);

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

			bool hasMaterial = false;
			for (int i = 0; i < this->sceneMaterials.size(); i++) {
				if (mat == this->sceneMaterials[i]) {
					fzbMesh.materialUniformObject.materialIndex = i;
					hasMaterial = true;
					break;
				}
			}
			if (!hasMaterial) {
				fzbMesh.materialUniformObject.materialIndex = this->sceneMaterials.size();
				this->sceneMaterials.push_back(mat);
			}

			std::vector<uint32_t> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "albedoTexture");
			if(diffuseMaps.size() > 0)
				fzbMesh.materialUniformObject.albedoTextureIndex = diffuseMaps[0];	//ֻȡһ����������ټ�������������˵

			//std::vector<FzbTexture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular");
			//textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

			std::vector<uint32_t> normalMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, "normalTexture");
			if (normalMaps.size() > 0)
				fzbMesh.materialUniformObject.normalTextureIndex = normalMaps[0];

		}
	}
	std::vector<uint32_t> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName) {

		std::vector<FzbTexture>* sceneTextures = typeName == "albedoTexture" ? &this->sceneAlbedoTextures : &this->sceneNormalTextures;

		std::vector<uint32_t> textureIndexs;
		for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
		{
			aiString str;
			mat->GetTexture(type, i, &str);
			bool skip = false;
			for (unsigned int j = 0; j < sceneTextures->size(); j++)
			{
				if (std::strcmp((*sceneTextures)[j].path.data(), str.C_Str()) == 0)
				{
					textureIndexs.push_back(j);
					skip = true;
					break;
				}
			}
			if (!skip)
			{   // �������û�б����أ��������
				FzbTexture texture;
				//texture.id = TextureFromFile(str.C_Str(), directory);
				texture.type = typeName;
				texture.path = meshPathDirectory + '/' + str.C_Str();
				textureIndexs.push_back(sceneTextures->size());
				sceneTextures->push_back(texture); // ��ӵ��Ѽ��ص�������
			}
		}

		return textureIndexs;

	}
	*/
	void addMeshToScene(FzbMesh mesh) {
		mesh.indeArraySize = mesh.indices.size();
		mesh.indexArrayOffset = this->sceneIndices.size();

		bool skip = true;
		for (int i = 0; i < this->sceneTransformMatrixs.size(); i++) {
			if (mesh.transforms == this->sceneTransformMatrixs[i]) {
				mesh.materialUniformObject.transformIndex = i;
				skip = false;
				break;
			}
		}
		if (skip) {
			mesh.materialUniformObject.transformIndex = this->sceneTransformMatrixs.size();
			this->sceneTransformMatrixs.push_back(mesh.transforms);
		}
		
		skip = true;
		for (int i = 0; i < this->sceneMaterials.size(); i++) {
			if (mesh.material == this->sceneMaterials[i]) {
				mesh.materialUniformObject.materialIndex = i;
				skip = false;
				break;
			}
		}
		if (skip) {
			mesh.materialUniformObject.materialIndex = this->sceneMaterials.size();
			this->sceneMaterials.push_back(mesh.material);
		}

		skip = true;
		if (mesh.albedoTexture.path != "") {
			for (int i = 0; i < this->sceneAlbedoTextures.size(); i++) {
				if (mesh.albedoTexture == this->sceneAlbedoTextures[i]) {
					mesh.materialUniformObject.albedoTextureIndex = i;
					skip = false;
					break;
				}
			}
			if (skip) {
				mesh.materialUniformObject.albedoTextureIndex = this->sceneAlbedoTextures.size();
				this->sceneAlbedoTextures.push_back(mesh.albedoTexture);
			}
		}

		skip = true;
		if (mesh.normalTexture.path != "") {
			for (int i = 0; i < this->sceneNormalTextures.size(); i++) {
				if (mesh.normalTexture == this->sceneNormalTextures[i]) {
					mesh.materialUniformObject.normalTextureIndex = i;
					skip = false;
					break;
				}

			}
			if (skip) {
				mesh.materialUniformObject.normalTextureIndex = this->sceneNormalTextures.size();
				this->sceneNormalTextures.push_back(mesh.normalTexture);
			}
		}

		this->differentVertexFormatMeshIndexs[mesh.shader.vertexFormat].push_back(this->sceneMeshSet.size());
		this->sceneMeshSet.push_back(mesh);
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
	/*
	����˵һ������˼·����������Ϊ���ٻ��󶥵㻺�壬������ǽ����ж��㶼����һ�����㻺���У���ʹ�������ʽ��ͬ������ʹ��float��Ϊ��λ��ÿ��mesh��¼������ƫ��
	��������ȷ��ö��㡣
	���ǣ����ڰ�������ʱֻ������һ��ƫ�ƣ���ˣ����ǲ���Ϊmesh��¼����ƫ�ƣ�������������0��ʼ��������Ҫ����ƫ��Ϊ0����������Ҫ��������ƫ�ƣ����Զ�����������
	�е���������ƫ�ơ�
	�������������ǣ�����ǰ�涥���ʽ��ռ���ֽ������ܱ���ǰ�Ķ����ʽ�ֽ�����������˵���ȫ���������ǵĽ������������padding��ʹ��ǰ����ֽ����ܱ�������Ȼ��
	��ǰ�����Ӧ��������������Ӧ��ƫ�ƣ��Ӷ��õ���ȷ�Ľ��
	һ����˵��paddingռ���˶����ֽڣ�һ������ռ52�ֽڵĻ������Ҳ�Ͷ��51�ֽڶ��ѡ�
	*/
	void getSceneVertics(bool compress = true) {	//Ŀǰֻ����̬mesh
		uint32_t FzbVertexByteSize = 0;
		FzbVertexFormat vertexFormat;
		for (auto& pair : differentVertexFormatMeshIndexs) {
			vertexFormat = pair.first;
			uint32_t vertexSize = vertexFormat.getVertexSize();	//�ֽ���

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
				sceneMeshSet[meshIndex].indeArraySize = sceneMeshSet[meshIndex].indices.size();	//ѹ���������ݲ���ı���������Ĵ�С��������ƫ��λ��

				std::vector<float> meshVertices = sceneMeshSet[meshIndex].getVetices();
				compressVertics.insert(compressVertics.end(), meshVertices.begin(), meshVertices.end());
				compressIndices.insert(compressIndices.end(), sceneMeshSet[meshIndex].indices.begin(), sceneMeshSet[meshIndex].indices.end());
				sceneMeshSet[meshIndex].indices.clear();
			}

			//std::vector<float> testVertices = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f };
			//std::vector<uint32_t> testIndices = { 0, 2, 4, 1, 2, 4, 1, 3, 4 };
			//compressSceneVertics(testVertices, FzbVertexFormat(), testIndices);

			if(compress)
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

	void createBufferAndTexture() {
		this->vertexBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, sceneVertices.data(), sceneVertices.size() * sizeof(float));
		this->sceneVertices.clear();
		this->transformBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, sceneTransformMatrixs.data(), sceneTransformMatrixs.size() * sizeof(glm::mat4));
		this->materialBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, sceneMaterials.data(), sceneMaterials.size() * sizeof(FzbMaterial));
		for (int i = 0; i < this->sceneAlbedoTextures.size(); i++) {
			FzbImage albedoImage;
			albedoImage.texturePath = sceneAlbedoTextures[i].path.c_str();
			albedoImage.fzbCreateImage(physicalDevice, logicalDevice, commandPool, graphicsQueue);
			this->albedoImages.push_back(albedoImage);
		}
		for (int i = 0; i < this->sceneNormalTextures.size(); i++) {
			FzbImage normalImage;
			normalImage.texturePath = sceneNormalTextures[i].path.c_str();
			normalImage.fzbCreateImage(physicalDevice, logicalDevice, commandPool, graphicsQueue);
			this->normalImages.push_back(normalImage);
		}
	}
	void createDescriptor() {
		std::map<VkDescriptorType, uint32_t> bufferTypeAndNum;
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 });	//����
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, this->albedoImages.size() + this->normalImages.size() });	//����
		this->descriptorPool = fzbCreateDescriptorPool(logicalDevice, bufferTypeAndNum);

		std::vector<VkDescriptorType> descriptorTypes = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER };
		std::vector<VkShaderStageFlags> descriptorShaderFlags = { VK_SHADER_STAGE_ALL, VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_FRAGMENT_BIT };
		sceneDecriptorSetLayout = fzbCreateDescriptLayout(logicalDevice, 4, descriptorTypes, descriptorShaderFlags, { 1, 1, uint32_t(this->albedoImages.size()), uint32_t(this->normalImages.size()) }, {false, false, true, true});
		descriptorSet = fzbCreateDescriptorSet(logicalDevice,descriptorPool, sceneDecriptorSetLayout);

		std::vector<VkWriteDescriptorSet> sceneDescriptorWrites;
		VkDescriptorBufferInfo transformStorageBufferInfo{};
		transformStorageBufferInfo.buffer = this->transformBuffer.buffer;
		transformStorageBufferInfo.offset = 0;
		transformStorageBufferInfo.range = sizeof(glm::mat4) * sceneTransformMatrixs.size();
		VkWriteDescriptorSet transformBufferWrite{};
		transformBufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		transformBufferWrite.dstSet = descriptorSet;
		transformBufferWrite.dstBinding = 0;
		transformBufferWrite.dstArrayElement = 0;
		transformBufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		transformBufferWrite.descriptorCount = 1;
		transformBufferWrite.pBufferInfo = &transformStorageBufferInfo;
		sceneDescriptorWrites.push_back(transformBufferWrite);

		VkDescriptorBufferInfo materialStorageBufferInfo{};
		materialStorageBufferInfo.buffer = this->materialBuffer.buffer;
		materialStorageBufferInfo.offset = 0;
		materialStorageBufferInfo.range = sizeof(FzbMaterial) * sceneMaterials.size();
		VkWriteDescriptorSet materialBufferWrite{};
		materialBufferWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		materialBufferWrite.dstSet = descriptorSet;
		materialBufferWrite.dstBinding = 1;
		materialBufferWrite.dstArrayElement = 0;
		materialBufferWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		materialBufferWrite.descriptorCount = 1;
		materialBufferWrite.pBufferInfo = &materialStorageBufferInfo;
		sceneDescriptorWrites.push_back(materialBufferWrite);

		std::vector<VkDescriptorImageInfo> albedoImageInfos{};
		for (int i = 0; i < this->albedoImages.size(); i++) {
			albedoImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			albedoImageInfos[i].imageView = this->albedoImages[i].imageView;
			albedoImageInfos[i].sampler = this->albedoImages[i].textureSampler;
		}
		if (this->albedoImages.size() > 0) {
			VkWriteDescriptorSet albedoImageWrite{};
			albedoImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			albedoImageWrite.dstSet = descriptorSet;
			albedoImageWrite.dstBinding = 2;
			albedoImageWrite.dstArrayElement = 0;
			albedoImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			albedoImageWrite.descriptorCount = this->albedoImages.size();
			albedoImageWrite.pImageInfo = albedoImageInfos.data();
			sceneDescriptorWrites.push_back(albedoImageWrite);
		}

		std::vector<VkDescriptorImageInfo> normalImageInfos{};
		for (int i = 0; i < this->normalImages.size(); i++) {
			normalImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			normalImageInfos[i].imageView = this->normalImages[i].imageView;
			normalImageInfos[i].sampler = this->normalImages[i].textureSampler;
		}
		VkWriteDescriptorSet normalImageWrite{};
		if (this->normalImages.size() > 0) {
			normalImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			normalImageWrite.dstSet = descriptorSet;
			normalImageWrite.dstBinding = 3;
			normalImageWrite.dstArrayElement = 0;
			normalImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			normalImageWrite.descriptorCount = this->normalImages.size();
			normalImageWrite.pImageInfo = normalImageInfos.data();
			sceneDescriptorWrites.push_back(normalImageWrite);
		}

		vkUpdateDescriptorSets(logicalDevice, sceneDescriptorWrites.size(), sceneDescriptorWrites.data(), 0, nullptr);
	}


};

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


#endif
