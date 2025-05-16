#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

//������vulkan.hǰ����VK_USE_PLATFORM_WIN32_KHR������vulkan.h��ͨ��glfw3.h����ģ���������#include <GLFW/glfw3.h>ǰ����VK_USE_PLATFORM_WIN32_KHRû��
//���ǣ��Ҵ�vulkan.h���֣�����VK_USE_PLATFORM_WIN32_KHRֻ�������������h�ļ������ֱ���ֶ�����Ҳ��һ���ġ�
#define NOMINMAX	//windows.h��max��std��glm��max�г�ͻ
#include <windows.h>
#include <vulkan/vulkan_win32.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FUNC_QUALIFIER //����cuda�˺���
#include<glm/glm.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <unordered_map>
#include <glm/ext/matrix_transform.hpp>

#include <optional>

#include "FzbMesh.h"

#ifndef FZB_SCENE_H
#define FZB_SCENE_H

struct FzbScene {
	std::vector<std::vector<FzbMesh*>> sceneMeshSet;	//��ͬ�����ʽ��mesh
	FzbAABBBox AABB;
	std::vector<FzbVertex> sceneVertices;
	std::vector<uint32_t> sceneIndices;

	~FzbScene() {
		for (int i = 0; i < sceneMeshSet.size(); i++) {
			for (int j = 0; j < sceneMeshSet[i].size(); j++) {
				delete sceneMeshSet[i][j];
			}
		}
	}

	void fzbAddMeshToScene(std::string path, FzbScene& fzbScene) {

		Assimp::Importer import;
		const aiScene* scene = import.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
			std::cout << "ERROR::ASSIMP::" << import.GetErrorString() << std::endl;
			throw std::runtime_error("ERROR::ASSIMP::" + (std::string)import.GetErrorString());
		}

		//myModel.directory = path.substr(0, path.find_last_of('/'));
		processNode(scene->mRootNode, scene, fzbScene.sceneMeshSet);


	}

	//һ��node����mesh����node��������Ҫ�ݹ飬�����е�mesh���ó���
	//���е�ʵ�����ݶ���scene�У���node�д洢����scene������
	void processNode(aiNode* node, const aiScene* scene, std::vector<std::vector<FzbMesh*>>& sceneMeshSet) {

		for (uint32_t i = 0; i < node->mNumMeshes; i++) {
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			processMesh(mesh, scene, sceneMeshSet);
		}

		for (uint32_t i = 0; i < node->mNumChildren; i++) {
			processNode(node->mChildren[i], scene, sceneMeshSet);
		}

	}

	void processMesh(aiMesh* mesh, const aiScene* scene, std::vector<std::vector<FzbMesh*>>& sceneMeshSet) {

		std::vector<FzbVertex> vertices;
		std::vector<uint32_t> indices;
		std::vector<FzbTexture> textures;

		int meshTypeIndex = 0;
		meshTypeIndex = mesh->HasNormals() ? meshTypeIndex & 0x1 : meshTypeIndex;
		meshTypeIndex = mesh->mTextureCoords[0] ? meshTypeIndex & 0x2 : meshTypeIndex;
		meshTypeIndex = mesh->HasTangentsAndBitangents() ? meshTypeIndex & 0x4 : meshTypeIndex;

		for (uint32_t i = 0; i < mesh->mNumVertices; i++) {

			FzbVertex vertex;
			glm::vec3 vector;

			vector.x = mesh->mVertices[i].x;
			vector.y = mesh->mVertices[i].y;
			vector.z = mesh->mVertices[i].z;
			vertex.pos = vector;

			if (mesh->HasNormals()) {

				vector.x = mesh->mNormals[i].x;
				vector.y = mesh->mNormals[i].y;
				vector.z = mesh->mNormals[i].z;
				vertex.normal = vector;

			}

			if (mesh->HasTangentsAndBitangents()) {

				vector.x = mesh->mTangents[i].x;
				vector.y = mesh->mTangents[i].y;
				vector.z = mesh->mTangents[i].z;
				vertex.tangent = vector;

			}

			if (mesh->mTextureCoords[0]) // �����Ƿ����������ꣿ����ֻ����һ������uv
			{
				glm::vec2 vec;
				vec.x = mesh->mTextureCoords[0][i].x;
				vec.y = mesh->mTextureCoords[0][i].y;
				vertex.texCoord = vec;
			}
			else {
				vertex.texCoord = glm::vec2(0.0f, 0.0f);
			}

			vertices.push_back(vertex);

		}

		for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
			aiFace face = mesh->mFaces[i];
			for (uint32_t j = 0; j < face.mNumIndices; j++) {
				indices.push_back(face.mIndices[j]);
			}
		}

		FzbMaterial mat;
		if (mesh->mMaterialIndex >= 0) {

			aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
			aiColor3D color;
			material->Get(AI_MATKEY_COLOR_AMBIENT, color);
			mat.ka = glm::vec4(color.r, color.g, color.b, 1.0);
			material->Get(AI_MATKEY_COLOR_DIFFUSE, color);
			mat.kd = glm::vec4(color.r, color.g, color.b, 1.0);
			material->Get(AI_MATKEY_COLOR_SPECULAR, color);
			mat.ks = glm::vec4(color.r, color.g, color.b, 1.0);
			material->Get(AI_MATKEY_COLOR_EMISSIVE, color);
			mat.ke = glm::vec4(color.r, color.g, color.b, 1.0);

			std::vector<FzbTexture> diffuseMaps = loadMaterialTextures(material, aiTextureType_DIFFUSE, "texture_albedo", myModel);
			textures.insert(textures.end(), diffuseMaps.begin(), diffuseMaps.end());

			std::vector<Texture> specularMaps = loadMaterialTextures(material, aiTextureType_SPECULAR, "texture_specular", myModel);
			textures.insert(textures.end(), specularMaps.begin(), specularMaps.end());

			std::vector<FzbTexture> normalMaps = loadMaterialTextures(material, aiTextureType_HEIGHT, "texture_normal", myModel);
			textures.insert(textures.end(), normalMaps.begin(), normalMaps.end());

		}

		switch (meshTypeIndex) {
		case 0:
			sceneMeshSet[0].push_back(new FzbMeshType<FzbVertex_OnlyPos>(vertices, indices, textures, mat));
			break;
		case 0x1:
			sceneMeshSet[1].push_back(new FzbMeshType<FzbVertex_PosNormal>(vertices, indices, textures, mat));
			break;
		case 0x3:
			sceneMeshSet[2].push_back(new FzbMeshType<FzbVertex_PosNormalTexCoord>(vertices, indices, textures, mat));
			break;
		case 0x7:
			sceneMeshSet[3].push_back(new FzbMeshType<FzbVertex>(vertices, indices, textures, mat));
			break;
		}

	}

	std::vector<FzbTexture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName, FzbModel& myModel) {

		std::vector<FzbTexture> textures;
		for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
		{
			aiString str;
			mat->GetTexture(type, i, &str);
			bool skip = false;
			for (unsigned int j = 0; j < myModel.textures_loaded.size(); j++)
			{
				if (std::strcmp(myModel.textures_loaded[j].path.data(), str.C_Str()) == 0)
				{
					textures.push_back(myModel.textures_loaded[j]);
					skip = true;
					break;
				}
			}
			if (!skip)
			{   // �������û�б����أ��������
				FzbTexture texture;
				//texture.id = TextureFromFile(str.C_Str(), directory);
				texture.type = typeName;
				texture.path = myModel.directory + '/' + str.C_Str();
				textures.push_back(texture);
				myModel.textures_loaded.push_back(texture); // ��ӵ��Ѽ��ص�������
			}
		}

		return textures;

	}
};



#endif
