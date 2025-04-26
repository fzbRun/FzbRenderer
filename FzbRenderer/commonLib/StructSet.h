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
#include<glm/glm.hpp>

#include <array>
#include <optional>
#include <string>
#include <vector>

#ifndef STRUCT_SET
#define STRUCT_SET

//所有命令都要被提交到队列中执行，而每个队列属于不同的队列族，不同的队列族支持不同的指令，比如有些队列族只负责数值计算，有些队列族只负责内存操作等
//而队列族是物理设备提供的，即一个物理设备提供某些队列族，如显卡能提供计算、渲染的功能，而不能提供造宇宙飞船的功能
struct QueueFamilyIndices {

	//std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;
	std::optional<uint32_t> graphicsAndComputeFamily;

	bool isComplete() {
		return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
	}

};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
	glm::vec3 pos;
	glm::vec2 texCoord;
	glm::vec3 normal;
	glm::vec3 tangent;

	Vertex() {
		this->pos = glm::vec3(0.0f);
		this->texCoord = glm::vec2(0.0f);
		this->normal = glm::vec3(0.0f);
		this->tangent = glm::vec3(0.0f);
	}

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
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
		attributeDescriptions[0].offset = offsetof(Vertex, pos);	//找pos在Vertex中的偏移

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, normal);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, tangent);

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const {
		return pos == other.pos && texCoord == other.texCoord && normal == other.normal && tangent == other.tangent;
	}

};

struct Vertex_onlyPos {
	glm::vec3 pos;

	Vertex_onlyPos(Vertex vertex) {
		this->pos = vertex.pos;
	}

	Vertex_onlyPos() {
		this->pos = glm::vec3(0.0f);
	}

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex_onlyPos);
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
		attributeDescriptions[0].offset = offsetof(Vertex_onlyPos, pos);	//找pos在Vertex中的偏移

		return attributeDescriptions;
	}

	bool operator==(const Vertex_onlyPos& other) const {
		return pos == other.pos;
	}

};

struct ComputeVertex {

	glm::vec4 pos;
	glm::vec4 normal;

	static VkVertexInputBindingDescription getBindingDescription() {

		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(ComputeVertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;

	}

	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {

		//VAO
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(ComputeVertex, pos);	//找pos在Vertex中的偏移

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(ComputeVertex, normal);

		return attributeDescriptions;
	}

	bool operator==(const ComputeVertex& other) const {
		return pos == other.pos;
	}

};

struct MyImage {

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

struct Material {
	glm::vec4 bxdfPara;	//这里本来是ka的，但是我修改mtl文件，将粗糙度放到了ka.x，金属值放到了ka.y，而折射率放到了ka.z，粗糙度放到
	glm::vec4 kd;
	glm::vec4 ks;
	glm::vec4 ke;
};

struct Texture {
	//uint32_t id;
	std::string type;
	std::string path;
};

struct AABBBox {

	float leftX;
	float rightX;
	float leftY;
	float rightY;
	float leftZ;
	float rightZ;

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

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::vector<Texture> textures;
	Material material;
	AABBBox AABB;

	Mesh(std::vector<Vertex> vertices, std::vector<uint32_t> indices, std::vector<Texture> textures, Material material) {
		this->vertices = vertices;
		this->indices = indices;
		this->textures = textures;
		this->material = material;
	}

};

struct MyModel {
	std::vector<Mesh> meshs;
	std::vector<Texture> textures_loaded;
	std::string directory;
	AABBBox AABB;
};

struct UniformBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 cameraPos;
	glm::vec4 randomNumber;
	glm::vec4 swapChainExtent;
};

struct UniformLightBufferObject {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec4 lightPos_strength;
	glm::vec4 normal;
	glm::vec4 size;

	//球面矩形采样参数
	glm::vec4 ex, ey;

};

struct DescriptorObject {

	VkDescriptorSetLayout discriptorLayout;
	std::vector<VkDescriptorSet> descriptorSets;
};

struct BvhTreeNode {

	BvhTreeNode* leftNode;
	BvhTreeNode* rightNode;
	AABBBox AABB;
	std::vector<uint32_t> meshIndex;

};

struct BvhArrayNode {

	int32_t leftNodeIndex;
	int32_t rightNodeIndex;
	AABBBox AABB;
	//我们只关注叶子节点的mesh，因为我们光线先与scene碰撞，直到bvhTree的叶子节点，再与节点中的mesh碰撞，所以我们只需要记录叶子节点的meshIndex即可
	//而叶子节点最多只有2个mesh
	//我真是天才
	int32_t meshIndex;
	//int32_t meshIndex2;

};

struct ComputeInputMesh {

	Material material;
	glm::ivec2 indexInIndicesArray;
	AABBBox AABB;
	//char padding[4];	//因为material中是vec4，是16字节，所以需要当前ComputeInputMesh是16字节的倍数，这样下一个ComputeInputMesh才能得到正确的值

};

struct Scene {
	std::vector<Mesh>* sceneMeshs;
	BvhTreeNode* bvhTree;
	std::vector<BvhArrayNode> bvhArray;
};

struct UniformBufferObjectVoxel {
	glm::mat4 model;
	glm::mat4 VP[3];
	glm::vec4 voxelSize_Num;
	glm::vec4 voxelStartPos;
};

#endif