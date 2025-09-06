#pragma once

#include "../StructSet.h"
#include "../FzbBuffer/FzbBuffer.h"
#include "../FzbImage/FzbImage.h"
#include "../FzbDescriptor/FzbDescriptor.h"
#include "../FzbShader/FzbShader.h"
#include "../FzbMaterial/FzbMaterial.h"
#include "../FzbMesh/FzbMesh.h"
#include "../FzbCamera/FzbCamera.h"

#include <pugixml/src/pugixml.hpp>

#ifndef FZB_SCENE_H
#define FZB_SCENE_H

struct FzbVertexFormatLess {
	bool operator()(FzbVertexFormat const& a, FzbVertexFormat const& b) const noexcept;
};

struct FzbScene {
	VkDescriptorPool sceneDescriptorPool = nullptr;	//包括所有material和mesh的资源

	std::vector<FzbMesh> sceneMeshSet;	//不同顶点格式的mesh
	std::map<FzbVertexFormat, std::vector<uint32_t>, FzbVertexFormatLess> differentVertexFormatMeshIndexs;

	std::map<std::string, FzbShader> sceneShaders;	//第一个string是shaderPath
	std::vector<FzbShader*> sceneShaders_vector;
	std::map<std::string, FzbMaterial> sceneMaterials;
	std::map<std::string, FzbImage> sceneImages;	//key是texture Path

	FzbBuffer vertexBuffer;
	FzbBuffer indexBuffer;
	FzbBuffer vertexPosBuffer;	//只包含scene中mesh的pos数据
	FzbBuffer indexPosBuffer;
	FzbBuffer vertexPosNormalBuffer;	//只包含scene中mesh的pos和normal数据
	FzbBuffer indexPosNormalBuffer;
	
	std::vector<FzbImage> images;

	std::vector<FzbCamera> sceneCameras;	//默认只有一个相机，后续代码也是按照一个相机写的，有多个相机再说吧
	std::vector<FzbLight> sceneLights;		//光源最多传递16个到shader，且无论到不到16个都会传16个，因为要用uniformbuffer，需要固定数量，这也符合unitySRP中的默认
	FzbBuffer cameraBuffer;
	FzbBuffer lightsBuffer;
	VkDescriptorSetLayout cameraAndLightsDescriptorSetLayout = nullptr;
	VkDescriptorSet cameraAndLightsDescriptorSet;

	FzbScene();

	void setScenePath(std::string scenePath);

	void initScene(bool compress = true, bool isMainScene = true);

	FzbAABBBox getAABB();

	void clean();

//--------------------------------------------------------------获取场景-----------------------------------------------------------------
	/*
	这个函数会从根据传入的shaderPath，创建相应的shader，根据material创建shaderVariant；根据传入的mesh，将之与material绑定
	*/
	void addMeshToScene(FzbMesh mesh, FzbMaterial material, std::string shaderPath);
	//std::string getRootPath();
	void createDefaultMaterial(FzbVertexFormat vertexFormat = FzbVertexFormat());
	/*
	这个函数会从sceneXML中读取相机信息；material，并创建相应的shader，根据material创建shaderVariant；读取mesh，将之与material绑定
	*/
	void addSceneFromMitsubaXML(std::string path, FzbVertexFormat vertexFormat);
	void createShaderMeshBatch();
	void createVertexBuffer(bool compress = true, bool useExternal = false);
	void createVertexPairDataBuffer(bool usePosData, bool usePosNormalData, bool compress = true, bool usePosDataExternal = false, bool usePosNormalDataExternal = false);
	void createBufferAndDescriptorOfMaterialAndMesh();	//创建各种buffer、image和描述符

	void createCameraAndLightBufferAndDescriptor();
	void updateCameraBuffer();
	void createAABB();

private:
	std::string scenePath = "";	//存储当前文件夹的地址
	FzbAABBBox AABB;
};

#endif
