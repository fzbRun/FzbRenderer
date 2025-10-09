#pragma once

#include "../../common/FzbCommon.h"
#include "../../common/FzbMaterial/FzbMaterial.h"

#ifndef FZB_PATH_TRACING_MATERIAL_H
#define FZB_PATH_TRACING_MATERIAL_H

MaterialType fzbGetRayTracingMaterialType(std::string materialTypeString);

struct FzbRayTracingMaterialUniformObject {
	uint32_t materialType;	//enum默认使用int来表示
	int textureIndex[3];
	glm::vec4 numberAttribute[8];
	glm::vec4 emissive;	//第一位为bool，判断是否自发光
};
FzbRayTracingMaterialUniformObject createInitialMaterialUniformObject();

#endif