#pragma once

#include "../StructSet.h"
#include "../FzbImage/FzbImage.h"
#include "../FzbDescriptor/FzbDescriptor.h"
#include "../FzbShader/FzbShader.h"

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
material有两种数据：1.缓冲区数据；2.纹理数据。两种数据均采取键值对的方式，key是数据名，也是shader中的宏
*/
struct FzbShaderVariant;
struct FzbMaterial {
public:
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
	FzbMaterial(std::string id);
	FzbMaterial(std::string id, std::string type);

	void clean();

	void createSource(std::string scenePath, uint32_t& numberBufferNum, std::map<std::string, FzbImage>& sceneImages);
	void createMaterialNumberPropertiesBuffer(uint32_t& numberBufferNum);
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

#endif