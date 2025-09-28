#pragma once

#include "../FzbCommon.h"
#include "../FzbImage/FzbImage.h"
#include "../FzbDescriptor/FzbDescriptor.h"
#include "../FzbShader/FzbShader.h"
#include <pugixml/src/pugixml.hpp>

#ifndef FZB_MATERIAL_H
#define FZB_MATERIAL_H

enum MaterialType {
	diffuse = 0,
	roughconductor = 1
};
struct FzbShaderVariant;
struct FzbMaterial {
public:
	std::string id = "";
	//sceneMaterial会根据type读取materialXML和sceneXML；componentMaterial会根据type读取shader；
	//material被scene处理就是sceneMaterial；material被component处理就是componentMaterial
	std::string type;	
	std::string path = "";
	std::string componentName = "";	//被使用的组件名字，不同组件的相同material不同，因为根据material的数据创建的资源可能不同

	//material有两种数据：1.缓冲区数据；2.纹理数据。两种数据均采取键值对的方式，key是数据名，也是shader中的宏
	FzbShaderProperty properties;
	FzbVertexFormat vertexFormat;

	bool hasCreateSource = false;	//当前material是否已经创造过资源了，如buffer和image
	std::vector<FzbImage*> textureProperties;
	FzbBuffer numberPropertiesBuffer;
	VkDescriptorSet descriptorSet = nullptr;

	FzbMaterial();
	FzbMaterial(std::string id);
	FzbMaterial(std::string id, std::string type);

	void getMaterialXMLInfo();
	void getSceneXMLInfo(pugi::xml_node& materialNode);

	void clean();

	void getDescriptorNum(uint32_t& textureNum, uint32_t numberPropertyNum);
	int getMaterialAttributeIndex(std::string attribute);
	int getMaterialTextureNum();
	void createSource(std::string scenePath, std::map<std::string, FzbImage>& sceneImages);
	void createMaterialNumberPropertiesBuffer();
	void createMaterialDescriptor(VkDescriptorPool sceneDescriptorPool, VkDescriptorSetLayout descriptorSetLayout);
	bool operator==(const FzbMaterial& other) const;
};
namespace std {
	template<>
	struct hash<FzbMaterial> {
		std::size_t operator()(FzbMaterial const& m) const noexcept {
			std::size_t seed = 0;
			//hash_combine(seed, std::hash<std::string>{}(m.id));
			hash_combine(seed, std::hash<std::string>{}(m.type));
			hash_combine(seed, std::hash<std::string>{}(m.path));
			hash_combine(seed, std::hash <std::string> {}(m.componentName));
			hash_combine(seed, std::hash<FzbVertexFormat>{}(m.vertexFormat));
			hash_combine(seed, std::hash<FzbShaderProperty>{}(m.properties));
			return seed;
		}
	};
}

#endif