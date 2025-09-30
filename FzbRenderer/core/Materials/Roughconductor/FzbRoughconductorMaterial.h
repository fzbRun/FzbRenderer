#pragma once

#include "../../common/FzbCommon.h"
#include "../../common/FzbMaterial/FzbMaterial.h"
#include <pugixml/src/pugixml.hpp>

#ifndef FZB_ROUGH_CONDUCTOR_MATERIAL_H
#define FZB_ROUGH_CONDUCTOR_MATERIAL_H

/*
* 导体中的折射率拥有实部eta和虚部k
* 我们可以根据实部，通过斯涅尔定律，得到折射方向；根据虚部，通过比尔-朗伯定律得到消光系数
* 实际上导体只需要考虑反射，我们可以根据xml中给出的eta和k计算出albedo
* 但是，三叶草中的折射率会根据rgb或光谱给出rgb或各种波长的光的折射率
* 我们目前的技术没法实现这种多波长（即使是rgb）的不同折射，所以我们这里采取只考虑绿光或明度的比例的折射率来近似
*/
struct FzbRoughconductorMaterial {
	uint32_t normalMapIndex = 0;
	uint32_t albedoMapIndex = 1;
	uint32_t albedoIndex = 0;
	uint32_t bsdfPara = 1;
	inline static uint32_t textureNum = 2;

	static void getSceneXMLInfo(FzbMaterial* material, pugi::xml_node& roughconductorNode) {
		if (pugi::xml_node textureNode = roughconductorNode.select_node(".//texture[@name='normal']").node()) {
			std::string texturePath = textureNode.select_node(".//string[@name='filename']").node().attribute("value").value();
			std::string filterType = textureNode.select_node(".//string[@name='filter_type']").node().attribute("value").value();
			VkFilter filter = filterType == "bilinear" ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
			material->properties.textureProperties["normalMap"] = FzbTexture(texturePath, filter);
		}
		if (pugi::xml_node textureNode = roughconductorNode.select_node(".//texture[@name='reflectance']").node()) {
			std::string texturePath = textureNode.select_node(".//string[@name='filename']").node().attribute("value").value();
			std::string filterType = textureNode.select_node(".//string[@name='filter_type']").node().attribute("value").value();
			VkFilter filter = filterType == "bilinear" ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
			material->properties.textureProperties["albedoMap"] = FzbTexture(texturePath, filter);
		}
		pugi::xml_node etaNode = roughconductorNode.select_node(".//rgb[@name='eta']").node();
		pugi::xml_node kNode = roughconductorNode.select_node(".//rgb[@name='k']").node();
		if (etaNode && kNode) {
			glm::vec3 eta = fzbGetRGBFromString(etaNode.attribute("value").value());
			glm::vec3 k = fzbGetRGBFromString(etaNode.attribute("value").value());
			float roughness = std::stof(kNode.attribute("value").value());
			glm::vec3 albedo = ((eta - 1.0f) * (eta - 1.0f) + k * k) / ((eta + 1.0f) * (eta + 1.0f) + k * k);
			FzbNumberProperty numProperty(glm::vec4(albedo, 0.0f));
			material->properties.numberProperties["albedo"] = numProperty;
		}
		if (pugi::xml_node roughnessNode = roughconductorNode.select_node(".//float[@name='alpha']").node()) {
			float roughness = std::stof(roughnessNode.attribute("value").value());
			FzbNumberProperty numProperty(glm::vec4(roughness, 0.0f, 0.0f, 0.0f));
			material->properties.numberProperties["bsdfPara"] = numProperty;
		}
		if (pugi::xml_node emissiveNode = roughconductorNode.child("emissive")) {
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
		else throw std::runtime_error("粗糙导体没有这种属性：" + attribute);
	}
};


#endif