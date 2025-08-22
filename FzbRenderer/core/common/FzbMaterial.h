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
material有两种数据：1.缓冲区数据；2.纹理数据。两种数据均采取键值对的方式，key是数据名，也是shader中的宏
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


	//应该求material和shader的并集，但是实际上我们创建shader的时候要考虑更多，所以应该是material决定shader的宏
	//如果shader开启了一个资源，但是material没有传入，则关闭该资源；如果shader没有或关闭了一个资源，但是material传入了，则报错
	void addShader(std::string path) {
		FzbShader shader(logicalDevice, path);

		//shader中没有的，则肯定没有，material传入则报错
		for (auto& shaderPair : this->properties.textureProperties) {
			if(!shader.properties.textureProperties.count(shaderPair.first)) {
				std::string error = "材质" + this->id + "传入shader没有的资源: " + shaderPair.first;
				throw std::runtime_error(error);
			}
		}
		for (auto& shaderPair : this->properties.numberProperties) {
			if (!shader.properties.numberProperties.count(shaderPair.first)) {
				std::string error = "材质" + this->id + "传入shader没有的资源: " + shaderPair.first;
				throw std::runtime_error(error);
			}
		}
		//如果shader开启了某个资源类型，但是material没有传入相应的资源，则关闭shader的相应的资源类（宏）
		for (auto& shaderPair : shader.properties.textureProperties) {
			std::string textureType = shaderPair.first;
			std::string macro = textureType;
			macro[0] = std::toupper(static_cast<unsigned char>(macro[0]));
			macro = "use" + macro;
			if (this->properties.textureProperties.count(textureType)) shader.macros[macro] = true;
			else shader.macros[macro] = false;
		}
		//number有点不同，如果material没有传入，而shader开启了，则使用默认值
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

			// 假设我们用 id + type + distribution 来做哈希
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