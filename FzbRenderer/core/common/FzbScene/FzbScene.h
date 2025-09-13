#pragma once

#include "../FzbCommon.h"
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
	std::vector<FzbMesh> sceneMeshSet;	//不同顶点格式的mesh
	std::map<FzbVertexFormat, std::vector<uint32_t>, FzbVertexFormatLess> differentVertexFormatMeshIndexs;
	std::map<std::string, FzbMaterial> sceneMaterials;
	std::map<std::string, FzbImage> sceneImages;	//key是texture Path

	FzbBuffer vertexBuffer;
	FzbBuffer indexBuffer;
	bool useVertexBufferHandle = false;

	FzbScene();
	virtual void createVertexBuffer(bool compress = true, bool useExternal = false);
	void addMeshToScene(FzbMesh mesh);	//这个函数会从根据传入的shaderPath，创建相应的shader，根据material创建shaderVariant；根据传入的mesh，将之与material绑定
	FzbAABBBox getAABB();
	void createAABB();
	void clean();

private:
	FzbAABBBox AABB;
};

struct FzbMainScene : public FzbScene {
	std::string scenePath = "";	//存储当前文件夹的地址

	FzbVertexFormat vertexFormat_allMesh = FzbVertexFormat();
	FzbVertexFormat vertexFormat_allMesh_prepocess = FzbVertexFormat();		//不包括mesh的顶点格式，只是组件所需顶点数据的格式

	bool useVertexBuffer_prepocess = false;
	FzbBuffer vertexBuffer_prepocess;	//给预处理组件使用，mainLoop前删除
	FzbBuffer indexBuffer_prepocess;
	bool useVertexBufferHandle_preprocess = false;

	std::vector<FzbCamera> sceneCameras;	//默认只有一个相机，后续代码也是按照一个相机写的，有多个相机再说吧
	std::vector<FzbLight> sceneLights;		//光源最多传递16个到shader，且无论到不到16个都会传16个，因为要用uniformbuffer，需要固定数量，这也符合unitySRP中的默认
	FzbBuffer cameraBuffer;
	FzbBuffer lightsBuffer;
	VkDescriptorPool cameraAndLigthsDescriptorPool = nullptr;
	VkDescriptorSetLayout cameraAndLightsDescriptorSetLayout = nullptr;
	VkDescriptorSet cameraAndLightsDescriptorSet = nullptr;

	FzbMainScene();
	/*只添加相机、光照，material的ID和type，mesh的ID和material，不会创建具体数据*/
	FzbMainScene(std::string path);

	void initScene(bool compress = true, bool isMainScene = true);
	void createMaterialSource();

	void createCameraAndLightBuffer();
	void createCameraAndLightDescriptor();
	void updateCameraBuffer();

	void addMaterialToScene(FzbMaterial material);

	void clean();

private:
	void createVertexBuffer_prepocess(bool compress = true, bool useExternal = false);
};

#endif
