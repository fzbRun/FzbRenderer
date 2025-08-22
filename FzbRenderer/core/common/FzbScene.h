#pragma once

#include "FzbImage.h"
#include "FzbDescriptor.h"
#include "FzbShader.h"
#include "FzbMaterial.h"
#include "FzbMesh.h"
#include "FzbCamera.h"

#include <pugixml/src/pugixml.hpp>

#ifndef FZB_SCENE_H
#define FZB_SCENE_H

//------------------------------------------------------��������------------------------------------------------
std::vector<std::string> get_all_files(const std::string& dir_path);

glm::vec3 getRGBFromString(std::string str);

glm::mat4 getMat4FromString(std::string str);

//-----------------------------------------------------------------------------------------------------------------
struct VectorFloatHash {
	size_t operator()(std::vector<float> const& v) const noexcept;
};

struct Mat4Hash {
	size_t operator()(glm::mat4 const& m) const noexcept;
};

struct Mat4Equal {
	bool operator()(glm::mat4 const& a, glm::mat4 const& b) const noexcept;
};

struct FzbVertexFormatLess {
	bool operator()(FzbVertexFormat const& a, FzbVertexFormat const& b) const noexcept;
};

struct FzbScene {
	std::string scenePath;	//�洢��ǰ�ļ��еĵ�ַ
	pugi::xml_document doc;
	uint32_t width;
	uint32_t height;

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;

	VkDescriptorPool sceneDescriptorPool = nullptr;	//��������material��mesh����Դ

	VkDescriptorSetLayout meshDescriptorSetLayout = nullptr;
	//std::map<std::string, std::string> sceneMeshPath;	//�������ͬ·��mesh�Ҳ�����ͬ����ʹ��ʵ����
	std::vector<FzbMesh> sceneMeshSet;	//��ͬ�����ʽ��mesh
	std::vector<uint32_t> sceneMeshIndices;		//����differentVertexFormatMeshIndexs����sceneMeshSet��˳��ѹ�����㣬�����޷�����sceneMeshSet��˳���ȡ������Ϣ�����Լ�¼һ��mesh�����������ʡ�
	std::map<FzbVertexFormat, std::vector<uint32_t>, FzbVertexFormatLess> differentVertexFormatMeshIndexs;

	std::vector<float> sceneVertices;	//ѹ����Ķ�������
	std::vector<uint32_t> sceneIndices;

	std::map<std::string, FzbShader> sceneShaders;	//��һ��string��shaderPath
	std::map<std::string, FzbMaterial> sceneMaterials;
	//std::map<std::string, std::string> sceneShaderPaths;	//��һ��string��materialType���ڶ���string��shader��Ŀ¼
	std::map<std::string, FzbImage> sceneImages;	//key��texture Path

	FzbBuffer vertexBuffer;
	FzbBuffer indexBuffer;
	
	std::vector<FzbImage> images;

	std::vector<FzbCamera> sceneCameras;	//Ĭ��ֻ��һ���������������Ҳ�ǰ���һ�����д�ģ��ж�������˵��
	std::vector<FzbLight> sceneLights;		//��Դ��ഫ��16����shader�������۵�����16�����ᴫ16������ΪҪ��uniformbuffer����Ҫ�̶���������Ҳ����unitySRP�е�Ĭ��
	FzbBuffer cameraBuffer;
	FzbBuffer lightsBuffer;
	VkDescriptorSetLayout cameraAndLightsDescriptorSetLayout = nullptr;
	VkDescriptorSet cameraAndLightsDescriptorSet;

	FzbAABBBox AABB;

	FzbScene();
	FzbScene(std::string scenePath);
	//FzbScene(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue);
	void initScene(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, FzbVertexFormat vertexFormat, bool compress = true);

	void clean();

	void clear();

//--------------------------------------------------------------��ȡ����-----------------------------------------------------------------

	void addMeshToScene(FzbMesh mesh, bool reAdd = false);
	//std::string getRootPath();
	void createDefaultMaterial(FzbVertexFormat vertexFormat = FzbVertexFormat());
	void addSceneFromMitsubaXML(std::string path, FzbVertexFormat vertexFormat);
	void compressSceneVertics(std::vector<float>& vertices, FzbVertexFormat vertexFormat, std::vector<uint32_t>& indices);
	void getSceneVertics(bool compress = true);
	void createVertexBuffer();
	void createBufferAndDescriptorOfMaterialAndMesh();	//��������buffer��image��������
	void createAABB();

	void createCameraAndLightBufferAndDescriptor();
	void updateCameraBuffer();
};

#endif
