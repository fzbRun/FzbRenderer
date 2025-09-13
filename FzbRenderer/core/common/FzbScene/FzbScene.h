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
	std::vector<FzbMesh> sceneMeshSet;	//��ͬ�����ʽ��mesh
	std::map<FzbVertexFormat, std::vector<uint32_t>, FzbVertexFormatLess> differentVertexFormatMeshIndexs;
	std::map<std::string, FzbMaterial> sceneMaterials;
	std::map<std::string, FzbImage> sceneImages;	//key��texture Path

	FzbBuffer vertexBuffer;
	FzbBuffer indexBuffer;
	bool useVertexBufferHandle = false;

	FzbScene();
	virtual void createVertexBuffer(bool compress = true, bool useExternal = false);
	void addMeshToScene(FzbMesh mesh);	//���������Ӹ��ݴ����shaderPath��������Ӧ��shader������material����shaderVariant�����ݴ����mesh����֮��material��
	FzbAABBBox getAABB();
	void createAABB();
	void clean();

private:
	FzbAABBBox AABB;
};

struct FzbMainScene : public FzbScene {
	std::string scenePath = "";	//�洢��ǰ�ļ��еĵ�ַ

	FzbVertexFormat vertexFormat_allMesh = FzbVertexFormat();
	FzbVertexFormat vertexFormat_allMesh_prepocess = FzbVertexFormat();		//������mesh�Ķ����ʽ��ֻ��������趥�����ݵĸ�ʽ

	bool useVertexBuffer_prepocess = false;
	FzbBuffer vertexBuffer_prepocess;	//��Ԥ�������ʹ�ã�mainLoopǰɾ��
	FzbBuffer indexBuffer_prepocess;
	bool useVertexBufferHandle_preprocess = false;

	std::vector<FzbCamera> sceneCameras;	//Ĭ��ֻ��һ���������������Ҳ�ǰ���һ�����д�ģ��ж�������˵��
	std::vector<FzbLight> sceneLights;		//��Դ��ഫ��16����shader�������۵�����16�����ᴫ16������ΪҪ��uniformbuffer����Ҫ�̶���������Ҳ����unitySRP�е�Ĭ��
	FzbBuffer cameraBuffer;
	FzbBuffer lightsBuffer;
	VkDescriptorPool cameraAndLigthsDescriptorPool = nullptr;
	VkDescriptorSetLayout cameraAndLightsDescriptorSetLayout = nullptr;
	VkDescriptorSet cameraAndLightsDescriptorSet = nullptr;

	FzbMainScene();
	/*ֻ�����������գ�material��ID��type��mesh��ID��material�����ᴴ����������*/
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
