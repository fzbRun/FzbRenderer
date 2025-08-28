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

//------------------------------------------------------辅助函数------------------------------------------------
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
	std::string scenePath = "";	//存储当前文件夹的地址
	pugi::xml_document doc;
	uint32_t width;
	uint32_t height;

	VkPhysicalDevice physicalDevice;
	VkDevice logicalDevice;
	VkCommandPool commandPool;
	VkQueue graphicsQueue;

	VkDescriptorPool sceneDescriptorPool = nullptr;	//包括所有material和mesh的资源

	VkDescriptorSetLayout meshDescriptorSetLayout = nullptr;
	std::vector<FzbMesh> sceneMeshSet;	//不同顶点格式的mesh
	std::vector<uint32_t> sceneMeshIndices;		//由于differentVertexFormatMeshIndexs不按sceneMeshSet的顺序压缩顶点，导致无法根据sceneMeshSet的顺序获取顶点信息，所以记录一个mesh索引用来访问。
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

	FzbAABBBox AABB;

	FzbScene();
	FzbScene(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue);
	void getSceneGlobalInfo(std::string scenePath);
	void initScene(VkPhysicalDevice physicalDevice, VkDevice logicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, std::vector<FzbVertexFormat> componentVertexFormats, bool compress = true);
	void initScene(std::vector<FzbVertexFormat> componentVertexFormats, bool compress = true);

	void clean();

	void clear();

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
	void compressSceneVertics(std::vector<float>& vertices, FzbVertexFormat vertexFormat, std::vector<uint32_t>& indices);
	void createVertexBuffer(bool compress = true);
	void createVertexPairDataBuffer(bool usePosData, bool usePosNormalData, bool compress = true);
	void createBufferAndDescriptorOfMaterialAndMesh();	//创建各种buffer、image和描述符

	void createCameraAndLightBufferAndDescriptor();
	void updateCameraBuffer();

	FzbAABBBox getAABB();
	void createAABB();
};

#endif
