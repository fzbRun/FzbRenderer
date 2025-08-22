#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

//必须在vulkan.h前定义VK_USE_PLATFORM_WIN32_KHR，但是vulkan.h是通过glfw3.h导入的，但是我在#include <GLFW/glfw3.h>前定义VK_USE_PLATFORM_WIN32_KHR没用
//但是，我打开vulkan.h发现，定义VK_USE_PLATFORM_WIN32_KHR只会包含如下两个h文件，因此直接手动导入也是一样的。
#define NOMINMAX	//windows.h中max与std和glm的max有冲突
#include <windows.h>
#include <vulkan/vulkan_win32.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FUNC_QUALIFIER //用于cuda核函数
#include<glm/glm.hpp>

#include <array>
#include <optional>
#include <string>
#include <vector>
#include <variant>
#include <map>

#ifndef STRUCT_SET
#define STRUCT_SET

//所有命令都要被提交到队列中执行，而每个队列属于不同的队列族，不同的队列族支持不同的指令，比如有些队列族只负责数值计算，有些队列族只负责内存操作等
//而队列族是物理设备提供的，即一个物理设备提供某些队列族，如显卡能提供计算、渲染的功能，而不能提供造宇宙飞船的功能
struct FzbQueueFamilyIndices {

	//std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;
	std::optional<uint32_t> graphicsAndComputeFamily;

	bool isComplete();

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
		attributeDescriptions[0].offset = offsetof(FzbVertex, pos);	//找pos在Vertex中的偏移

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
		attributeDescriptions[0].offset = offsetof(FzbVertex_OnlyPos, pos);	//找pos在Vertex中的偏移

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
		attributeDescriptions[0].offset = offsetof(FzbVertex, pos);	//找pos在Vertex中的偏移

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
		attributeDescriptions[0].offset = offsetof(FzbVertex, pos);	//找pos在Vertex中的偏移

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
	std::vector<FzbVertex> modelVertices;	//当模型的顶点数据要经常被使用，那么我们可以额外存储；
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

	FzbVertexFormat();

	FzbVertexFormat(bool useNormal, bool useTexCoord = false, bool useTangent = false);

	uint32_t getVertexSize() const;

	VkVertexInputBindingDescription getBindingDescription();

	std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();

	void mergeUpward(FzbVertexFormat vertexFormat);
	bool operator==(const FzbVertexFormat& other) const;

};
namespace std {
	template<>
	struct hash<FzbVertexFormat> {
		std::size_t operator()(const FzbVertexFormat& vf) const {
			using std::size_t;
			using std::hash;

			// 计算哈希值的种子
			size_t seed = 0;

			// 辅助函数：组合多个哈希值
			auto combine_hash = [](size_t& seed, size_t hash) {
				seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			};

			// 对 FzbVertexFormat 的成员进行哈希
			combine_hash(seed, hash<bool>{}(vf.useNormal));
			combine_hash(seed, hash<bool>{}(vf.useTexCoord));
			combine_hash(seed, hash<bool>{}(vf.useTangent));

			// 如果 FzbVertexFormat 有其他成员，也需要添加到这里

			return seed;
		}
	};
}

struct FzbGlobalUniformBufferObject {
	glm::vec4 swapChainExtent;
};

struct FzbCameraUniformBufferObject {
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 cameraPos;
};

struct FzbLightDate {
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 pos;
	glm::vec4 strength;
};

struct FzbLightsUniformBufferObject {
	FzbLightDate lightData[16];
	uint32_t lightNum;

	FzbLightsUniformBufferObject();
	FzbLightsUniformBufferObject(uint32_t lightNum);
};


struct FzbAreaLightData : public FzbLightDate {
	glm::vec4 normal;
	glm::vec4 size;
};

struct FzbPTAreaLightData : public FzbAreaLightData {
	//球面矩形采样参数
	glm::vec4 ex, ey;
};

struct FzbDescriptorObject {
	VkDescriptorSetLayout discriptorLayout;
	std::vector<VkDescriptorSet> descriptorSets;
};

struct FzbSemaphore {
	VkSemaphore semaphore = nullptr;
	HANDLE handle = nullptr;

	FzbSemaphore();
	FzbSemaphore(VkDevice logicalDevice, bool UseExternal = false);
	void clean(VkDevice logicalDevice);
};

struct FzbTexture {
	std::string path = "";
	VkFilter filter = VK_FILTER_LINEAR;

	FzbTexture();
	FzbTexture(std::string path, VkFilter filter);

	bool operator==(const FzbTexture& other) const;
};
namespace std {
	template<>
	struct hash<FzbTexture> {
		size_t operator()(const FzbTexture& tex) const noexcept {
			using std::hash;
			using std::size_t;

			size_t seed = 0;

			auto combine_hash = [](size_t& seed, size_t h) {
				seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			};

			// 对 path 做哈希
			combine_hash(seed, hash<std::string>{}(tex.path));

			// 对 filter 做哈希（枚举类型可直接用 hash<int>）
			combine_hash(seed, hash<int>{}(static_cast<int>(tex.filter)));

			return seed;
		}
	};
}

struct FzbNumberProperty {
	glm::vec4 value = glm::vec4(0.0f);

	FzbNumberProperty();
	FzbNumberProperty(glm::vec4 value);

	bool operator==(const FzbNumberProperty& other) const;
};
namespace std {
	template<>
	struct hash<FzbNumberProperty> {
		size_t operator()(const FzbNumberProperty& prop) const noexcept {
			size_t h1 = std::hash<float>{}(prop.value.x);
			size_t h2 = std::hash<float>{}(prop.value.y);
			size_t h3 = std::hash<float>{}(prop.value.z);
			size_t h4 = std::hash<float>{}(prop.value.w);
			// 混合哈希值
			return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1) ^ (h4 << 2);
		}
	};
}

struct FzbShaderProperty {
	std::map<std::string, FzbTexture> textureProperties;
	std::map<std::string, FzbNumberProperty> numberProperties;

	bool keyCompare(FzbShaderProperty& other);
	bool operator==(const FzbShaderProperty& other) const;
};

std::string getRootPath();

#endif