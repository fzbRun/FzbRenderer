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
	//sceneMaterial�����type��ȡmaterialXML��sceneXML��componentMaterial�����type��ȡshader��
	//material��scene�������sceneMaterial��material��component�������componentMaterial
	std::string type;	
	std::string path = "";
	std::string componentName = "";	//��ʹ�õ�������֣���ͬ�������ͬmaterial��ͬ����Ϊ����material�����ݴ�������Դ���ܲ�ͬ

	//material���������ݣ�1.���������ݣ�2.�������ݡ��������ݾ���ȡ��ֵ�Եķ�ʽ��key����������Ҳ��shader�еĺ�
	FzbShaderProperty properties;
	FzbVertexFormat vertexFormat;

	bool hasCreateSource = false;	//��ǰmaterial�Ƿ��Ѿ��������Դ�ˣ���buffer��image
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