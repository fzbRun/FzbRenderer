#pragma once

#include "../../../common/FzbCommon.h"

#ifndef FZB_PATH_TRACING_MATERIAL_H
#define FZB_PATH_TRACING_MATERIAL_H

enum MaterialType {
	diffuse = 0,
	roughconductor = 1
};
MaterialType getMaterialType(std::string materialTypeString);

struct FzbPathTracingMaterialUniformObject {
	uint32_t materialType;	//enum默认使用int来表示
	int textureIndex[3];
	glm::vec4 numberAttribute[8];
	glm::vec4 emissive;	//第一位为bool，判断是否自发光
};
FzbPathTracingMaterialUniformObject createInitialMaterialUniformObject();

//指明diffuseMaterial所需的数据在FzbPathTracingMaterialUniformObject中的索引
struct FzbPathTracingDiffuseMaterialInfo {
	uint32_t normalMapIndex = 0;
	uint32_t albedoMapIndex = 1;
	uint32_t albedoIndex = 0;
};

/*
* 三叶草Mitsuba中对于折射率采取实部和虚部的方案
* 我们可以根据实部，通过斯涅尔定律，得到折射方向；根据虚部，通过比尔-朗伯定律得到消光系数
* 但是，三叶草中的折射率会根据rgb或光谱给出rgb或各种波长的光的折射率
* 我们目前的技术没法实现这种多波长（即使是rgb）的不同折射，所以我们这里采取只考虑绿光的折射率来近似
*/
struct FzbPathTracingRoughconductorMaterialInfo : public FzbPathTracingDiffuseMaterialInfo {
	uint32_t roughness = 1;
	uint32_t specular_reflectance = 2;
	uint32_t eta = 3;	//外部进入内部的折射率之比
	uint32_t k = 4;		//消光系数
};

#endif