#pragma once

#include "../FzbCommon.h"
#include "../FzbShader/FzbShader.h"
#include "../FzbRenderer.h"

#ifndef FZB_RASTERIZATION_RENDER_H
#define FZB_RASTERIZATION_RENDER_H

struct FzbRasterizationSourceManager {
	FzbScene componentScene;
	std::map<std::string, FzbShader> shaderSet;	//��һ��string��shaderPath

	std::map<FzbMesh*, FzbMaterial*> meshMaterialPairs;	//һ��renderPass����Ҫ��mesh�Լ�����Ӧ��material
	std::vector<FzbShader*> shaders_vector;
	std::map<std::string, FzbImage*> images;

	VkDescriptorPool descriptorPool = nullptr;	//��������material��mesh����Դ

	FzbRasterizationSourceManager();
	void addMeshMaterial(std::vector<FzbMesh>& meshes, FzbMaterial material = FzbMaterial(), bool loopRender = true);
	void addSource(std::map<std::string, FzbShaderInfo> shaderInfos);

	void clean();

private:
	void createShader(std::map<std::string, FzbShaderInfo> shaderInfos);
	void createShaderMeshBatch();
	void createBufferAndDescriptorOfMaterial();
};

#endif