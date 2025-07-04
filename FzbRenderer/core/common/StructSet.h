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

#include <array>
#include <optional>
#include <string>
#include <vector>
#include <variant>

#ifndef STRUCT_SET
#define STRUCT_SET

//�������Ҫ���ύ��������ִ�У���ÿ���������ڲ�ͬ�Ķ����壬��ͬ�Ķ�����֧�ֲ�ͬ��ָ�������Щ������ֻ������ֵ���㣬��Щ������ֻ�����ڴ������
//���������������豸�ṩ�ģ���һ�������豸�ṩĳЩ�����壬���Կ����ṩ���㡢��Ⱦ�Ĺ��ܣ��������ṩ������ɴ��Ĺ���
struct FzbQueueFamilyIndices {

	//std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;
	std::optional<uint32_t> graphicsAndComputeFamily;

	bool isComplete() {
		return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
	}

};

struct FzbSwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};
/*
struct FzbVertex {
	glm::vec3 pos;
	glm::vec2 texCoord;
	glm::vec3 normal;
	glm::vec3 tangent;

	FzbVertex() {
		this->pos = glm::vec3(0.0f);
		this->texCoord = glm::vec2(0.0f);
		this->normal = glm::vec3(0.0f);
		this->tangent = glm::vec3(0.0f);
	}

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(FzbVertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {

		//VAO
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
		attributeDescriptions.resize(4);

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(FzbVertex, pos);	//��pos��Vertex�е�ƫ��

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(FzbVertex, texCoord);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(FzbVertex, normal);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(FzbVertex, tangent);

		return attributeDescriptions;
	}

	bool operator==(const FzbVertex& other) const {
		return pos == other.pos && texCoord == other.texCoord && normal == other.normal && tangent == other.tangent;
	}

};

struct FzbVertex_OnlyPos {
	glm::vec3 pos;

	FzbVertex_OnlyPos(FzbVertex vertex) {
		this->pos = vertex.pos;
	}

	FzbVertex_OnlyPos() {
		this->pos = glm::vec3(0.0f);
	}

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(FzbVertex_OnlyPos);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {

		//VAO
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
		attributeDescriptions.resize(1);

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(FzbVertex_OnlyPos, pos);	//��pos��Vertex�е�ƫ��

		return attributeDescriptions;
	}

	bool operator==(const FzbVertex_OnlyPos& other) const {
		return pos == other.pos;
	}

};

struct FzbVertex_PosNormal {
	glm::vec3 pos;
	glm::vec3 normal;

	FzbVertex_PosNormal() {
		this->pos = glm::vec3(0.0f);
		this->normal = glm::vec3(0.0f);
	}

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(FzbVertex_PosNormal);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {

		//VAO
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
		attributeDescriptions.resize(2);

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(FzbVertex, pos);	//��pos��Vertex�е�ƫ��

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 2;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(FzbVertex, normal);
 
		return attributeDescriptions;
	}

	bool operator==(const FzbVertex& other) const {
		return pos == other.pos  && normal == other.normal;
	}
};

struct FzbVertex_PosNormalTexCoord {
	glm::vec3 pos;
	glm::vec2 texCoord;
	glm::vec3 normal;

	FzbVertex_PosNormalTexCoord() {
		this->pos = glm::vec3(0.0f);
		this->texCoord = glm::vec2(0.0f);
		this->normal = glm::vec3(0.0f);
	}

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(FzbVertex_PosNormalTexCoord);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {

		//VAO
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
		attributeDescriptions.resize(3);

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(FzbVertex, pos);	//��pos��Vertex�е�ƫ��

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(FzbVertex, texCoord);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(FzbVertex, normal);

		return attributeDescriptions;
	}

	bool operator==(const FzbVertex& other) const {
		return pos == other.pos && texCoord == other.texCoord && normal == other.normal;
	}
};

struct FzbMaterial {
	glm::vec4 ka;
	glm::vec4 kd;
	glm::vec4 ks;
	glm::vec4 ke;
};

struct FzbAABBBox {

	float leftX;
	float rightX;
	float leftY;
	float rightY;
	float leftZ;
	float rightZ;

	glm::vec3 startPos;
	glm::vec3 distanceXYZ;
	glm::vec3 center;

	float getAxis(int k) {
		if (k == 0) {
			return leftX;
		}
		else if (k == 1) {
			return rightX;
		}
		else if (k == 2) {
			return leftY;
		}
		else if (k == 3) {
			return rightY;
		}
		else if (k == 4) {
			return leftZ;
		}
		else if (k == 5) {
			return rightZ;
		}
	}

};

struct FzbTexture {
	//uint32_t id;
	std::string type;
	std::string path;
};

struct FzbMesh {
	std::vector<FzbVertex> vertices;
	std::vector<uint32_t> indices;
	std::vector<FzbTexture> textures;
	FzbMaterial material;
	FzbAABBBox AABB;

	FzbMesh() {};

	FzbMesh(std::vector<FzbVertex> vertices, std::vector<uint32_t> indices, std::vector<FzbTexture> textures, FzbMaterial material) {
		this->vertices = vertices;
		this->indices = indices;
		this->textures = textures;
		this->material = material;
	}

};

struct FzbModel {
	std::vector<FzbMesh> meshs;
	std::vector<FzbTexture> textures_loaded;
	std::string directory;
	FzbAABBBox AABB;
	std::vector<FzbVertex> modelVertices;	//��ģ�͵Ķ�������Ҫ������ʹ�ã���ô���ǿ��Զ���洢��
	std::vector<uint32_t> modelIndices;
};

struct FzbScene {
	std::vector<FzbModel*> sceneModels;
	FzbAABBBox AABB;
	std::vector<FzbVertex> sceneVertices;
	std::vector<uint32_t> sceneIndices;
};
*/
/*
struct FzbImage {

	const char* texturePath = nullptr;
	bool mipmapEnable = false;

	uint32_t width = 512;
	uint32_t height = 512;
	uint32_t depth = 1;
	uint32_t layerNum = 1;
	uint32_t mipLevels = 1;
	VkImageType type = VK_IMAGE_TYPE_2D;
	VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
	VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT;
	VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
	VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
	VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	VkMemoryPropertyFlagBits properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
	VkFilter filter = VK_FILTER_LINEAR;
	VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	VkBool32 anisotropyEnable = VK_TRUE;

	VkImage image;
	VkImageView imageView;
	VkDeviceMemory imageMemory;
	VkSampler textureSampler;

	HANDLE handle;

};
*/

struct FzbVertexFormat {
	bool useNormal;
	bool useTexCoord;
	bool useTangent;

	FzbVertexFormat() {
		this->useNormal = false;
		this->useTexCoord = false;
		this->useTangent = false;
	}

	FzbVertexFormat(bool useNormal, bool useTexCoord = false, bool useTangent = false) {
		this->useNormal = useNormal;
		this->useTexCoord = useTexCoord;
		this->useTangent = useTangent;
	}

	uint32_t getVertexSize() const {
		uint32_t attributeNum = 3 + useNormal * 3 + useTexCoord * 2 + useTangent * 3;
		return attributeNum * sizeof(float);
	}

	VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = getVertexSize();
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions() {

		//VAO
		std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
		VkVertexInputAttributeDescription posDescriptor{};
		VkVertexInputAttributeDescription normalDescriptor{};
		VkVertexInputAttributeDescription texCoordDescriptor{};
		VkVertexInputAttributeDescription tangentDescriptor{};

		posDescriptor.binding = 0;
		posDescriptor.location = 0;
		posDescriptor.format = VK_FORMAT_R32G32B32_SFLOAT;
		posDescriptor.offset = 0;	//��pos��Vertex�е�ƫ��
		attributeDescriptions.push_back(posDescriptor);

		uint32_t attributeOffset = 0;
		if (useNormal) {
			normalDescriptor.binding = 0;
			normalDescriptor.location = 1;
			normalDescriptor.format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeOffset += 3 * sizeof(float);
			normalDescriptor.offset = attributeOffset;
			attributeDescriptions.push_back(normalDescriptor);
		}
		if (useTexCoord) {
			texCoordDescriptor.binding = 0;
			texCoordDescriptor.location = 2;
			texCoordDescriptor.format = VK_FORMAT_R32G32_SFLOAT;
			attributeOffset += 2 * sizeof(float);
			texCoordDescriptor.offset = attributeOffset;
			attributeDescriptions.push_back(texCoordDescriptor);
		}
		if (useTangent) {
			tangentDescriptor.binding = 0;
			tangentDescriptor.location = 3;
			tangentDescriptor.format = VK_FORMAT_R32G32B32_SFLOAT;
			attributeOffset += 3 * sizeof(float);
			tangentDescriptor.offset = attributeOffset;
			attributeDescriptions.push_back(tangentDescriptor);
		}

		return attributeDescriptions;
	}

	void mergeUpward(FzbVertexFormat vertexFormat) {
		this->useNormal |= vertexFormat.useNormal;
		this->useTexCoord |= vertexFormat.useTexCoord;
		this->useTangent |= vertexFormat.useTangent;
	}

	bool operator==(const FzbVertexFormat& other) const {
		if (!(useNormal == other.useNormal && useTexCoord == other.useTexCoord && useTangent == other.useTangent)) {
			return false;
		}
		return true;
	}

};

struct FzbGlobalUniformBufferObject {
	glm::vec4 swapChainExtent;
};

struct FzbCameraUniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 cameraPos;
};

struct FzbUniformLightBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 lightPos_strength;
};

struct FzbUniformAreaLightBufferObject : public FzbUniformLightBufferObject {
	glm::vec4 normal;
	glm::vec4 size;
};

struct FzbUniformPTAreaLightBufferObject : public FzbUniformAreaLightBufferObject {
	//������β�������
	glm::vec4 ex, ey;
};

struct FzbDescriptorObject {
	VkDescriptorSetLayout discriptorLayout;
	std::vector<VkDescriptorSet> descriptorSets;
};

struct FzbSemaphore {
	VkSemaphore semaphore;
	HANDLE handle;
};

#endif