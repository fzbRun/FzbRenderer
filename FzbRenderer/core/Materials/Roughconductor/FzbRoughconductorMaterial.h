#pragma once

#include "../../common/FzbCommon.h"
#include "../../common/FzbMaterial/FzbMaterial.h"
#include <pugixml/src/pugixml.hpp>

#ifndef FZB_ROUGH_CONDUCTOR_MATERIAL_H
#define FZB_ROUGH_CONDUCTOR_MATERIAL_H

/*
* �����е�������ӵ��ʵ��eta���鲿k
* ���ǿ��Ը���ʵ����ͨ��˹�������ɣ��õ����䷽�򣻸����鲿��ͨ���ȶ�-�ʲ����ɵõ�����ϵ��
* ʵ���ϵ���ֻ��Ҫ���Ƿ��䣬���ǿ��Ը���xml�и�����eta��k�����albedo
* ���ǣ���Ҷ���е������ʻ����rgb����׸���rgb����ֲ����Ĺ��������
* ����Ŀǰ�ļ���û��ʵ�����ֶನ������ʹ��rgb���Ĳ�ͬ���䣬�������������ȡֻ�����̹�����ȵı�����������������
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
		//ɾȥû�е�texture����ֵ������ʹ��Ĭ��ֵ
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
		else throw std::runtime_error("�ֲڵ���û���������ԣ�" + attribute);
	}
};


#endif