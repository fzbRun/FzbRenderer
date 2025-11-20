#pragma once

#include "../../common/FzbCommon.h"
#include "../../common/FzbMaterial/FzbMaterial.h"
#include <pugixml/src/pugixml.hpp>

#ifndef FZB_DIELECTRIC_MATERIAL_H
#define FZB_DIELECTRIC_MATERIAL_H

/*
电解质的折射率只有实部eta
xml中可能给出int_ior和ext_ior，也可能直接给出eta
我们计算出eta后放入bsdfPara中的第一个通道（没有粗糙度）
*/
struct FzbDielectricMaterial {
	uint32_t normalMapIndex = 0;
	uint32_t albedoMapIndex = 1;
	uint32_t albedoIndex = 0;
	uint32_t bsdfPara = 1;	//粗糙度，折射率之比
	inline static uint32_t textureNum = 2;

	static void getSceneXMLInfo(FzbMaterial* material, pugi::xml_node& dielectricNode) {
		if (pugi::xml_node textureNode = dielectricNode.select_node(".//texture[@name='normal']").node()) {
			std::string texturePath = textureNode.select_node(".//string[@name='filename']").node().attribute("value").value();
			std::string filterType = textureNode.select_node(".//string[@name='filter_type']").node().attribute("value").value();
			VkFilter filter = filterType == "bilinear" ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
			material->properties.textureProperties["normalMap"] = FzbTexture(texturePath, filter);
		}
		if (pugi::xml_node textureNode = dielectricNode.select_node(".//texture[@name='reflectance']").node()) {
			std::string texturePath = textureNode.select_node(".//string[@name='filename']").node().attribute("value").value();
			std::string filterType = textureNode.select_node(".//string[@name='filter_type']").node().attribute("value").value();
			VkFilter filter = filterType == "bilinear" ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
			material->properties.textureProperties["albedoMap"] = FzbTexture(texturePath, filter);
		}
		glm::vec4 bsdfPara = glm::vec4(1.5f, 0.0f, 0.0f, 0.0f);
		if (pugi::xml_node etaNode = dielectricNode.select_node(".//float[@name='eta']").node()) {
			bsdfPara.x = std::stof(etaNode.attribute("value").value());
		}else{
			pugi::xml_node int_iorNode = dielectricNode.select_node(".//float[@name='int_ior']").node();
			pugi::xml_node ext_iorNode = dielectricNode.select_node(".//float[@name='ext_ior']").node();
			if (int_iorNode && ext_iorNode) {
				float int_ior = std::stof(int_iorNode.attribute("value").value());
				float ext_ior = std::stof(ext_iorNode.attribute("value").value());

				if (int_ior == ext_ior) throw std::runtime_error("不允许两边折射率相同");
				if (int_ior == 0 || ext_ior == 0) throw std::runtime_error("不允许折射率为0");

				bsdfPara.x = ext_ior / int_ior;
			}
		}
		material->properties.numberProperties["bsdfPara"] = FzbNumberProperty(bsdfPara);
		float eta = bsdfPara.x;
		glm::vec3 albedo = glm::vec3((eta - 1.0f) * (eta - 1.0f)) / ((eta + 1.0f) * (eta + 1.0f));
		FzbNumberProperty numProperty(glm::vec4(albedo, 0.0f));
		material->properties.numberProperties["albedo"] = numProperty;

		if (pugi::xml_node emissiveNode = dielectricNode.child("emissive")) {
			glm::vec3 emissive = fzbGetRGBFromString(emissiveNode.child("rgb").attribute("value").value());
			material->properties.numberProperties.insert({ "emissive", FzbNumberProperty(glm::vec4(emissive, 1.0f)) });
		}
		//删去没有的texture，数值参数会使用默认值
		std::vector<std::string> nullTexture; nullTexture.reserve(material->properties.textureProperties.size());
		for (auto& texturePair : material->properties.textureProperties) {
			if (texturePair.second == FzbTexture()) nullTexture.push_back(texturePair.first);
		}
		for(int i = 0; i < nullTexture.size(); ++i) material->properties.textureProperties.erase(nullTexture[i]);
		
		if (material->properties.textureProperties.size() == 0) material->vertexFormat.useTexCoord = false;
	}
	static int getAttributeIndex(std::string attribute) {
		if (attribute == "normalMap") return 0;
		else if (attribute == "albedoMap") return 1;
		else if (attribute == "albedo") return 0;
		else if (attribute == "bsdfPara") return 1;
		else throw std::runtime_error("电解质没有这种属性：" + attribute);
	}
};


#endif