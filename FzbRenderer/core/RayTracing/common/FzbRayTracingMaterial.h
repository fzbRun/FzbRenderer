#pragma once

#include "../../common/FzbCommon.h"
#include "../../common/FzbMaterial/FzbMaterial.h"

#ifndef FZB_PATH_TRACING_MATERIAL_H
#define FZB_PATH_TRACING_MATERIAL_H

MaterialType fzbGetRayTracingMaterialType(std::string materialTypeString);

struct FzbRayTracingMaterialUniformObject {
	uint32_t materialType;	//enumĬ��ʹ��int����ʾ
	int textureIndex[3];
	glm::vec4 numberAttribute[8];
	glm::vec4 emissive;	//��һλΪbool���ж��Ƿ��Է���
};
FzbRayTracingMaterialUniformObject createInitialMaterialUniformObject();

#endif