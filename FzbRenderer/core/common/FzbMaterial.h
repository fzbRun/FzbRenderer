#pragma once

#include "StructSet.h"
#include "FzbImage.h"
#include "FzbDescriptor.h"
#include "FzbShader.h"

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

/*
struct FzbCheckboradTexture : public FzbTexture {
	glm::vec3 checkboardColor1;
	glm::vec3 checkboardColor2;

	bool operator==(const FzbCheckboradTexture& other) const {
		return path == other.path && filter == other.filter && checkboardColor1 == other.checkboardColor1 && checkboardColor2 == other.checkboardColor2;
	}
};
*/

/*
material���������ݣ�1.���������ݣ�2.�������ݡ��������ݾ���ȡ��ֵ�Եķ�ʽ��key����������Ҳ��shader�еĺ�
*/
struct FzbShaderVariant;
struct FzbMaterial {
public:
	VkDevice logicalDevice;

	std::string id = "";
	std::string type;
	//FzbShader shader;
	//FzbShaderVariant* shader;
	FzbShaderProperty properties;
	FzbVertexFormat vertexFormat;

	FzbBsdfDistribution distribution;
	//FzbTexture spcularReflectanceTexture;

	FzbBuffer numberPropertiesBuffer;
	//VkDescriptorSetLayout descriptorSetLayout = nullptr;
	VkDescriptorSet descriptorSet = nullptr;

	FzbMaterial();
	FzbMaterial(VkDevice logicalDevice);
	FzbMaterial(VkDevice logicalDevice, std::string id);
	FzbMaterial(VkDevice logicalDevice, std::string id, std::string type);

	void clean();

	//FzbVertexFormat getVertexFormat();

	//void changeVertexFormat(FzbVertexFormat newVertexFormat);
	/*
	void changeVertexFormatAndMacros() {
		if (this->properties.textureProperties.size() > 0) {
			shader->vertexFormat.useTexCoord = true;
			shader->macros["useTextureProperty"] = true;
			shader->macros["useVertexTexCoords"] = true;
		}
		else {
			shader->vertexFormat.useTexCoord = false;
			shader->macros["useTextureProperty"] = false;
			shader->macros["useVertexTexCoords"] = false;
		}
		shader->macros["useNumberProperty"] = this->properties.numberProperties.size() == 0 ? false : true;
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
	*/

	void createSource(VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, std::string scenePath, uint32_t& numberBufferNum, std::map<std::string, FzbImage>& sceneImages);
	void createMaterialNumberPropertiesBuffer(VkPhysicalDevice physicalDevice, uint32_t& numberBufferNum);

	void createMaterialDescriptor(VkDescriptorPool sceneDescriptorPool, VkDescriptorSetLayout descriptorSetLayout, std::map<std::string, FzbImage>& sceneImages);
	
	bool operator==(const FzbMaterial& other) const;

};
namespace std {
	template<>
	struct hash<FzbMaterial> {
		size_t operator()(const FzbMaterial& mat) const noexcept {
			using std::hash;
			using std::size_t;

			size_t seed = 0;

			auto combine_hash = [](size_t& seed, size_t h) {
				seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			};

			// ���������� id + type + distribution ������ϣ
			combine_hash(seed, hash<std::string>{}(mat.id));
			combine_hash(seed, hash<std::string>{}(mat.type));
			combine_hash(seed, hash<int>{}(static_cast<int>(mat.distribution)));

			return seed;
		}
	};
}

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