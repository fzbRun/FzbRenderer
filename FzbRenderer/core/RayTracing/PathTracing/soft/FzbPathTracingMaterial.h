#pragma once

#include "../../../common/FzbCommon.h"

#ifndef FZB_PATH_TRACING_MATERIAL_H
#define FZB_PATH_TRACING_MATERIAL_H

enum MaterialType {
	diffuse = 0,
	roughconductor = 1
};

struct FzbPathTracingMaterialUniformObject {
	MaterialType materialType = diffuse;	//enumĬ��ʹ��int����ʾ
	int textureIndex[3];
	glm::vec4 numberAttribute[8];
};

//ָ��diffuseMaterial�����������FzbPathTracingMaterialUniformObject�е�����
struct FzbPathTracingDiffuseMaterialInfo {
	uint32_t albedoMapIndex = 0;
	uint32_t normalMapIndex = 0;
	uint32_t albedoIndex = 0;
};

/*
* ��Ҷ��Mitsuba�ж��������ʲ�ȡʵ�����鲿�ķ���
* ���ǿ��Ը���ʵ����ͨ��˹�������ɣ��õ����䷽�򣻸����鲿��ͨ���ȶ�-�ʲ����ɵõ�����ϵ��
* ���ǣ���Ҷ���е������ʻ����rgb����׸���rgb����ֲ����Ĺ��������
* ����Ŀǰ�ļ���û��ʵ�����ֶನ������ʹ��rgb���Ĳ�ͬ���䣬�������������ȡֻ�����̹��������������
*/
struct FzbPathTracingRoughconductorMaterialInfo {
	uint32_t albedoMapIndex = 0;
	uint32_t normalMapIndex = 0;
	uint32_t albedoIndex = 0;
	uint32_t roughness = 1;
	uint32_t specular_reflectance = 2;
	uint32_t eta = 3;	//�ⲿ�����ڲ���������֮��
	uint32_t k = 4;		//����ϵ��
};

#endif