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
	VkDescriptorPool sceneDescriptorPool = nullptr;	//��������material��mesh����Դ

	std::vector<FzbMesh> sceneMeshSet;	//��ͬ�����ʽ��mesh
	std::map<FzbVertexFormat, std::vector<uint32_t>, FzbVertexFormatLess> differentVertexFormatMeshIndexs;

	std::map<std::string, FzbShader> sceneShaders;	//��һ��string��shaderPath
	std::vector<FzbShader*> sceneShaders_vector;
	std::map<std::string, FzbMaterial> sceneMaterials;
	std::map<std::string, FzbImage> sceneImages;	//key��texture Path

	FzbBuffer vertexBuffer;
	FzbBuffer indexBuffer;
	FzbBuffer vertexPosBuffer;	//ֻ����scene��mesh��pos����
	FzbBuffer indexPosBuffer;
	FzbBuffer vertexPosNormalBuffer;	//ֻ����scene��mesh��pos��normal����
	FzbBuffer indexPosNormalBuffer;
	
	std::vector<FzbImage> images;

	std::vector<FzbCamera> sceneCameras;	//Ĭ��ֻ��һ���������������Ҳ�ǰ���һ�����д�ģ��ж�������˵��
	std::vector<FzbLight> sceneLights;		//��Դ��ഫ��16����shader�������۵�����16�����ᴫ16������ΪҪ��uniformbuffer����Ҫ�̶���������Ҳ����unitySRP�е�Ĭ��
	FzbBuffer cameraBuffer;
	FzbBuffer lightsBuffer;
	VkDescriptorSetLayout cameraAndLightsDescriptorSetLayout = nullptr;
	VkDescriptorSet cameraAndLightsDescriptorSet;

	FzbScene();

	void setScenePath(std::string scenePath);

	void initScene(bool compress = true, bool isMainScene = true);

	FzbAABBBox getAABB();

	void clean();

//--------------------------------------------------------------��ȡ����-----------------------------------------------------------------
	/*
	���������Ӹ��ݴ����shaderPath��������Ӧ��shader������material����shaderVariant�����ݴ����mesh����֮��material��
	*/
	void addMeshToScene(FzbMesh mesh, FzbMaterial material, std::string shaderPath);
	//std::string getRootPath();
	void createDefaultMaterial(FzbVertexFormat vertexFormat = FzbVertexFormat());
	/*
	����������sceneXML�ж�ȡ�����Ϣ��material����������Ӧ��shader������material����shaderVariant����ȡmesh����֮��material��
	*/
	void addSceneFromMitsubaXML(std::string path, FzbVertexFormat vertexFormat);
	void createShaderMeshBatch();
	void createVertexBuffer(bool compress = true, bool useExternal = false);
	void createVertexPairDataBuffer(bool usePosData, bool usePosNormalData, bool compress = true, bool usePosDataExternal = false, bool usePosNormalDataExternal = false);
	void createBufferAndDescriptorOfMaterialAndMesh();	//��������buffer��image��������

	void createCameraAndLightBufferAndDescriptor();
	void updateCameraBuffer();
	void createAABB();

private:
	std::string scenePath = "";	//�洢��ǰ�ļ��еĵ�ַ
	FzbAABBBox AABB;
};

#endif
