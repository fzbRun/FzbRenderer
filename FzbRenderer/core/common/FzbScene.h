/*
在这里说一下整个FzbScene的思路
1. 首先，fzbScene维护整个场景的mesh信息，包括mesh的各种资源，如变换、材质和纹理
2. fzbScene中的mesh来源于外部通过addMeshToScene函数传入fzbScene
	2.1 不直接通过fzbScene的函数从OBJ文件中获取mesh，是为了能对mesh先做一些操作再传入fzbScene。实际上这应该是UI界面上该做的事情，但是由于现在还没有
		搞UI，就先放在外部。
	2.2 addMeshToScene会获取mesh的各种信息，然后存入fzbScene的数组中继续维护，mesh中只保留信息在数组中的索引；然后根据mesh的shader的顶点格式要求将mesh
		存入differentVertexFormatMeshIndexs这个map中，方便我们后续去除冗余顶点
3. 将所有的mesh存入fzbScene后，我们可以通过initScene来初始化场景，这包含4步
	3.1 获取mesh的顶点和索引，并得到整个场景的顶点和索引数组。我们选择将所有的顶点信息都存于一个float数组中，这是为了方便我们用一个顶点缓冲存储所有不
		同顶点格式的顶点，从而不需要频繁的换绑顶点缓冲。
		3.1.1 我们通过将differentVertexFormatMeshIndexs的相同顶点格式的mesh进行压缩
		3.1.2 然后，根据前面不同顶点格式顶点所占的字节数偏移后续的顶点索引的大小，从而使得每个mesh的顶点能正确对应正确的顶点
	3.2 根据上一步创建的顶点数组，创建顶点缓冲.索引缓冲由meshBatch独立创建
	3.3 根据之前mesh传入的变换、材质和纹理，创建相应的缓冲和image，这里我们先只搞反射率纹理和法线纹理
	3.4 根据对上一步的缓冲和image创建描述符
4. 后续meshBatch的缓冲创建以及pipeline中描述符集合的使用都需要用到场景信息。每个组件维护一个fzbScene和主组件的fzbScene
*/

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

	std::vector<FzbMesh> sceneMeshSet;	//不同顶点格式的mesh
	//std::vector<FzbMeshBatch> sceneMeshBatchs;
	std::map<FzbVertexFormat, std::vector<uint32_t>, FzbVertexFormatLess> differentVertexFormatMeshIndexs;

	std::vector<float> sceneVertices;	//压缩后的顶点数据
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

	VkDescriptorPool descriptorPool = nullptr;
	VkDescriptorSetLayout sceneDecriptorSetLayout = nullptr;
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
		clearBufferAndDescriptor();
		for (int i = 0; i < this->sceneMeshSet.size(); i++) {
			sceneMeshSet[i].clean();
		}
	}

	void clear() {
		clean();
		sceneMeshSet.clear();
		differentVertexFormatMeshIndexs.clear();
		sceneVertices.clear();
		sceneIndices.clear();
		sceneTransformMatrixs.clear();
		sceneMaterials.clear();
		sceneAlbedoTextures.clear();
		sceneNormalTextures.clear();
	}

	void clearBufferAndDescriptor() {
		vertexBuffer.clean();
		materialBuffer.clean();
		transformBuffer.clean();

		for (int i = 0; i < albedoImages.size(); i++) {
			albedoImages[i].clean();
		}
		for (int i = 0; i < normalImages.size(); i++) {
			normalImages[i].clean();
		}

		if (descriptorPool) {
			vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
			vkDestroyDescriptorSetLayout(logicalDevice, sceneDecriptorSetLayout, nullptr);
		}

	}

	void addMeshToScene(FzbMesh mesh, bool reAdd = false) {
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
		if(!reAdd) this->sceneMeshSet.push_back(mesh);
	}

	void initScene(bool compress = true, bool vertex = true, bool transform = true, bool bufferAndTexture = true, bool descriptor = true) {
		if (vertex) {
			getSceneVertics(compress);
			createVertexBuffer();
		}
		if (transform) createTransformBuffer();
		if (bufferAndTexture) createBufferAndTexture();
		if (descriptor) createDescriptor();
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

	void createVertexBuffer() {
		this->vertexBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, sceneVertices.data(), sceneVertices.size() * sizeof(float));
		this->sceneVertices.clear();
	}

	void createTransformBuffer() {
		this->transformBuffer = fzbCreateStorageBuffer(physicalDevice, logicalDevice, commandPool, graphicsQueue, sceneTransformMatrixs.data(), sceneTransformMatrixs.size() * sizeof(glm::mat4));
	}

	void createBufferAndTexture() {
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
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 });	//材质
		bufferTypeAndNum.insert({ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, this->albedoImages.size() + this->normalImages.size() });	//纹理
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
