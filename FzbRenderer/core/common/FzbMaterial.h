#pragma once

#include "StructSet.h"
#include "FzbShader.h"
#include "FzbDescriptor.h"

#ifndef FZB_MATERIAL_H
#define FZB_MATERIAL_H

enum FzbBsdfDistribution {
	FZB_GGX = 0,
	FZB_BECKMENN = 1
};

enum FzbBsdfType {
	FZB_DIFFUSE = 0,
	FZB_ROUGH_CONDUCTOR,
};

struct FzbCheckboradTexture : public FzbTexture {
	glm::vec3 checkboardColor1;
	glm::vec3 checkboardColor2;

	bool operator==(const FzbCheckboradTexture& other) const {
		return path == other.path && filter == other.filter && checkboardColor1 == other.checkboardColor1 && checkboardColor2 == other.checkboardColor2;
	}
};

struct FzbMaterial {
	VkDevice logicalDevice;

	std::string id = "";
	std::string type;
	FzbShader shader;
	FzbShaderProperty properties;

	FzbBsdfDistribution distribution;
	//FzbTexture spcularReflectanceTexture;

	FzbBuffer numberPropertiesBuffer;
	VkDescriptorSetLayout descriptorSetLayout = nullptr;
	VkDescriptorSet descriptorSet = nullptr;

	FzbMaterial() {};
	FzbMaterial(VkDevice logicalDevice) {
		this->logicalDevice = logicalDevice;
	};
	FzbMaterial(VkDevice logicalDevice, std::string id) {
		this->logicalDevice = logicalDevice;
		this->id = id;
	}

	void clean() {
		this->shader.clean();
		numberPropertiesBuffer.clean();
		vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
	}

	FzbVertexFormat getVertexFormat() {
		//this->vertexFormat = FzbVertexFormat(true, false, false);
		//if (albedoMap.path != "") {
		//	vertexFormat.useTexCoord = true;
		//}
		return this->shader.vertexFormat;
	}

	void changeVertexFormat(FzbVertexFormat newVertexFormat) {
		this->shader.changeVertexFormat(newVertexFormat);
	}

	void changeVertexFormatAndMacros() {
		if (this->properties.textureProperties.size() > 0) {
			shader.vertexFormat.useTexCoord = true;
			shader.macros["useTextureProperty"] = true;
			shader.macros["useVertexTexCoords"] = true;
		}
		else {
			shader.vertexFormat.useTexCoord = false;
			shader.macros["useTextureProperty"] = false;
			shader.macros["useVertexTexCoords"] = false;
		}
		shader.macros["useNumberProperty"] = this->properties.numberProperties.size() == 0 ? false : true;
	}

	//Ӧ����material��shader�Ĳ���������ʵ�������Ǵ���shader��ʱ��Ҫ���Ǹ��࣬����Ӧ����material����shader�ĺ�
	//���shader������һ����Դ������materialû�д��룬��رո���Դ�����shaderû�л�ر���һ����Դ������material�����ˣ��򱨴�
	void addShader(std::string path) {
		FzbShader shader(logicalDevice, path);

		//shader��û�еģ���϶�û�У�material�����򱨴�
		for (auto& shaderPair : this->properties.textureProperties) {
			if(!shader.properties.textureProperties.count(shaderPair.first)) {
				std::string error = "����" + this->id + "����shaderû�е���Դ: " + shaderPair.first;
				throw std::runtime_error(error);
			}
		}
		for (auto& shaderPair : this->properties.numberProperties) {
			if (!shader.properties.numberProperties.count(shaderPair.first)) {
				std::string error = "����" + this->id + "����shaderû�е���Դ: " + shaderPair.first;
				throw std::runtime_error(error);
			}
		}
		//���shader������ĳ����Դ���ͣ�����materialû�д�����Ӧ����Դ����ر�shader����Ӧ����Դ�ࣨ�꣩
		for (auto& shaderPair : shader.properties.textureProperties) {
			std::string textureType = shaderPair.first;
			std::string macro = textureType;
			macro[0] = std::toupper(static_cast<unsigned char>(macro[0]));
			macro = "use" + macro;
			if (this->properties.textureProperties.count(textureType)) shader.macros[macro] = true;
			else shader.macros[macro] = false;
		}
		//number�е㲻ͬ�����materialû�д��룬��shader�����ˣ���ʹ��Ĭ��ֵ
		for (auto& shaderPair : shader.properties.numberProperties) {
			std::string numberType = shaderPair.first;
			std::string macro = numberType;
			macro[0] = std::toupper(static_cast<unsigned char>(macro[0]));
			macro = "use" + macro;
			if (this->properties.numberProperties.count(numberType)) shader.macros[macro] = true;
			else if (shader.macros[macro]) this->properties.numberProperties.insert(shaderPair);
		}
		this->shader = shader;
		changeVertexFormatAndMacros();
	}

	void createMaterialNumberPropertiesBuffer(VkPhysicalDevice physicalDevice) {
		uint32_t numberNum = this->properties.numberProperties.size();
		if (numberNum > 0) {
			uint32_t bufferSize = numberNum * sizeof(glm::vec4);	//ȫ����floa4���洢�����shader�Ĳ��ַ�ʽ�ܷ��˵�
			std::vector<glm::vec4> numberProperties;
			for (auto& property : shader.properties.numberProperties) {
				numberProperties.push_back(property.second.value);
			}
			this->numberPropertiesBuffer = fzbCreateUniformBuffers(physicalDevice, logicalDevice, bufferSize);
			memcpy(numberPropertiesBuffer.mapped, numberProperties.data(), numberProperties.size() * sizeof(glm::vec4));
		}
	}

	void createMaterialDescriptor(VkDescriptorPool sceneDescriptorPool, std::map<std::string, FzbImage> sceneImages) {
		uint32_t textureNum = this->properties.textureProperties.size();
		uint32_t numberNum = this->properties.numberProperties.size() > 0 ? 1 : 0;	//������ֵ������һ��storageBuffer����
		if (textureNum + numberNum == 0) {
			return;
		}
		std::vector<VkDescriptorType> type(textureNum + numberNum);
		std::vector<VkShaderStageFlags> stage(textureNum + numberNum);
		for (int i = 0; i < textureNum; i++) {
			type[i] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			stage[i] = VK_SHADER_STAGE_ALL;	//shaderStage����Ϊall���ص�shader����Ӱ�����ݵĶ�ȡ�ٶȣ�ֻ���ڱ���ʱ���ⷶΧ��ͬ���ѣ�Ӱ������ٶȶ��ѣ���
		}
		if (numberNum) {
			type[textureNum] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			stage[textureNum] = VK_SHADER_STAGE_ALL;
		}
		this->descriptorSetLayout = fzbCreateDescriptLayout(logicalDevice, type.size(), type, stage);

		descriptorSet = fzbCreateDescriptorSet(logicalDevice, sceneDescriptorPool, descriptorSetLayout);
		uint32_t descriptorNum = textureNum + numberNum;
		std::vector<VkWriteDescriptorSet> voxelGridMapDescriptorWrites(descriptorNum);
		uint32_t binding = 0;
		for (auto& pair : this->properties.textureProperties) {
			VkDescriptorImageInfo textureInfo{};
			textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo.imageView = sceneImages[pair.second.path].imageView;
			textureInfo.sampler = sceneImages[pair.second.path].textureSampler;
			voxelGridMapDescriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			voxelGridMapDescriptorWrites[binding].dstSet = descriptorSet;
			voxelGridMapDescriptorWrites[binding].dstBinding = binding;
			voxelGridMapDescriptorWrites[binding].dstArrayElement = 0;
			voxelGridMapDescriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			voxelGridMapDescriptorWrites[binding].descriptorCount = 1;
			voxelGridMapDescriptorWrites[binding].pImageInfo = &textureInfo;
			binding++;
		}
		if (numberNum) {
			VkDescriptorBufferInfo numberBufferInfo{};
			numberBufferInfo.buffer = this->numberPropertiesBuffer.buffer;
			numberBufferInfo.offset = 0;
			numberBufferInfo.range = shader.properties.numberProperties.size() * sizeof(glm::vec4);
			voxelGridMapDescriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			voxelGridMapDescriptorWrites[binding].dstSet = descriptorSet;
			voxelGridMapDescriptorWrites[binding].dstBinding = binding;
			voxelGridMapDescriptorWrites[binding].dstArrayElement = 0;
			voxelGridMapDescriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			voxelGridMapDescriptorWrites[binding].descriptorCount = 1;
			voxelGridMapDescriptorWrites[binding].pBufferInfo = &numberBufferInfo;
		}

		vkUpdateDescriptorSets(logicalDevice, voxelGridMapDescriptorWrites.size(), voxelGridMapDescriptorWrites.data(), 0, nullptr);
	}
};

//struct FzbMaterial_Diffuse : public FzbMaterial {
//	FzbTexture albedoMap;
//	glm::vec3 albedo = glm::vec3(-1.0f);
//
//	FzbMaterial_Diffuse(std::string id, FzbTexture albedoMap, glm::vec3 albedo) {
//		this->id = id;
//		this->albedoMap = albedoMap;
//		this->albedo = albedo;
//	}
//};
//
//struct FzbMaterial_Roughconductor : public FzbMaterial {
//	FzbTexture albedoMap;
//	glm::vec3 albedo = glm::vec3(-1.0f);
//
//	FzbMaterial_Roughconductor(std::string id, FzbTexture albedoMap, glm::vec3 albedo) {
//		this->id = id;
//		this->albedoMap = albedoMap;
//		this->albedo = albedo;
//	}
//};

#endif