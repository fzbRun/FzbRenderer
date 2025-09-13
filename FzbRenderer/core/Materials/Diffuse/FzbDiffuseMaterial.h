#pragma once

#include "../../common/FzbCommon.h"
#include "../../common/FzbMaterial/FzbMaterial.h"
#include <pugixml/src/pugixml.hpp>

#ifndef FZB_DIFFUSE_MATERIAL_H
#define FZB_DIFFUSE_MATERIAL_H

struct FzbDiffuseMaterial {
	static void getSceneXMLInfo(FzbMaterial* material, pugi::xml_node& diffuseNode) {
		if (pugi::xml_node textureNode = diffuseNode.select_node(".//texture[@name='reflectance']").node()) {
			std::string texturePath = textureNode.select_node(".//string[@name='filename']").node().attribute("value").value();
			std::string filterType = textureNode.select_node(".//string[@name='filter_type']").node().attribute("value").value();
			VkFilter filter = filterType == "bilinear" ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
			material->properties.textureProperties["albedoMap"] = FzbTexture(texturePath, filter);
		}
		if (pugi::xml_node rgbNode = diffuseNode.select_node(".//rgb[@name='reflectance']").node()) {
			glm::vec3 albedo = fzbGetRGBFromString(rgbNode.attribute("value").value());
			FzbNumberProperty numProperty(glm::vec4(albedo, 0.0f));
			material->properties.numberProperties["albedo"] = numProperty;
		}

		//ɾȥû�е�texture����ֵ������ʹ��Ĭ��ֵ
		std::vector<std::string> nullTexture; nullTexture.reserve(material->properties.textureProperties.size());
		for (auto& texturePair : material->properties.textureProperties) {
			if (texturePair.second == FzbTexture()) nullTexture.push_back(texturePair.first);
		}
		for(int i = 0; i < nullTexture.size(); ++i) material->properties.textureProperties.erase(nullTexture[i]);
		
		if (material->properties.textureProperties.size() == 0) material->vertexFormat.useTexCoord = false;
	}
};


#endif