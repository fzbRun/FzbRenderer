#pragma once

#include "../../../common/FzbCommon.h"

#ifndef FZB_PATH_TRACING_MATERIAL_H
#define FZB_PATH_TRACING_MATERIAL_H

enum MaterialType {
	diffuse = 0,
	roughconductor = 1
};
MaterialType getMaterialType(std::string materialTypeString) {
	if (materialTypeString == "diffuse") return diffuse;
	else if (materialTypeString == "roughconductor") return roughconductor;
}

struct FzbPathTracingMaterialUniformObject {
	MaterialType materialType = diffuse;	//enumĬ��ʹ��int����ʾ
	int textureIndex[3];
	glm::vec4 numberAttribute[8];
	glm::vec4 emitter;	//��һλΪbool���ж��Ƿ��Է���

	FzbPathTracingMaterialUniformObject() {
		for (int i = 0; i < 3; ++i) textureIndex[i] = -1;
		for (int i = 0; i < 8; ++i) numberAttribute[i] = glm::vec4(1.0f);
	}
};

//ָ��diffuseMaterial�����������FzbPathTracingMaterialUniformObject�е�����
struct FzbPathTracingDiffuseMaterialInfo {
	uint32_t normalMapIndex = 0;
	uint32_t albedoMapIndex = 1;
	uint32_t albedoIndex = 0;
};

/*
* ��Ҷ��Mitsuba�ж��������ʲ�ȡʵ�����鲿�ķ���
* ���ǿ��Ը���ʵ����ͨ��˹�������ɣ��õ����䷽�򣻸����鲿��ͨ���ȶ�-�ʲ����ɵõ�����ϵ��
* ���ǣ���Ҷ���е������ʻ����rgb����׸���rgb����ֲ����Ĺ��������
* ����Ŀǰ�ļ���û��ʵ�����ֶನ������ʹ��rgb���Ĳ�ͬ���䣬�������������ȡֻ�����̹��������������
*/
struct FzbPathTracingRoughconductorMaterialInfo : public FzbPathTracingDiffuseMaterialInfo {
	uint32_t roughness = 1;
	uint32_t specular_reflectance = 2;
	uint32_t eta = 3;	//�ⲿ�����ڲ���������֮��
	uint32_t k = 4;		//����ϵ��
};

#endif