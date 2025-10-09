#pragma once

#include "../FzbCommon.h"
#include "../FzbShader/FzbShader.h"
#include "../FzbRenderer.h"

#ifndef FZB_RASTERIZATION_RENDER_H
#define FZB_RASTERIZATION_RENDER_H

struct FzbRasterizationSourceManager {
	FzbScene componentScene;
	FzbMeshCanvas canvans;
	std::map<std::string, FzbShader> shaderSet;	//第一个string是shaderPath

	std::map<FzbMesh*, FzbMaterial*> meshMaterialPairs;	//一个renderPass所需要的mesh以及它对应的material
	std::vector<FzbShader*> shaders_vector;
	std::map<std::string, FzbImage*> images;

	VkDescriptorPool descriptorPool = nullptr;	//包括所有material和mesh的资源

	FzbRasterizationSourceManager();
	void createCanvas(FzbMaterial material);
	void addMeshMaterial(std::vector<FzbMesh>& meshes, FzbMaterial material = FzbMaterial(), bool loopRender = true);
	void addMeshMaterial(FzbMesh* mesh, FzbMaterial material, bool loopRender = true);
	void addSource(std::map<std::string, FzbShaderInfo> shaderInfos);

	void clean();

private:
	void createShader(std::map<std::string, FzbShaderInfo> shaderInfos);
	void createShaderMeshBatch();
	void createBufferAndDescriptorOfMaterial();
};

#endif