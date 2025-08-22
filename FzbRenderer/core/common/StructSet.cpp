#include "StructSet.h"
#include <set>
#include <filesystem>

bool FzbQueueFamilyIndices::isComplete() {
	return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
}

FzbVertexFormat::FzbVertexFormat() {
	this->useNormal = false;
	this->useTexCoord = false;
	this->useTangent = false;
}
FzbVertexFormat::FzbVertexFormat(bool useNormal, bool useTexCoord, bool useTangent) {
	this->useNormal = useNormal;
	this->useTexCoord = useTexCoord;
	this->useTangent = useTangent;
}
uint32_t FzbVertexFormat::getVertexSize() const {
	uint32_t attributeNum = 3 + useNormal * 3 + useTexCoord * 2 + useTangent * 3;
	return attributeNum * sizeof(float);
}
VkVertexInputBindingDescription FzbVertexFormat::getBindingDescription() {

	VkVertexInputBindingDescription bindingDescription{};
	bindingDescription.binding = 0;
	bindingDescription.stride = getVertexSize();
	bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
	return bindingDescription;

}
std::vector<VkVertexInputAttributeDescription> FzbVertexFormat::getAttributeDescriptions() {

	//VAO
	std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
	VkVertexInputAttributeDescription posDescriptor{};
	VkVertexInputAttributeDescription normalDescriptor{};
	VkVertexInputAttributeDescription texCoordDescriptor{};
	VkVertexInputAttributeDescription tangentDescriptor{};

	posDescriptor.binding = 0;
	posDescriptor.location = 0;
	posDescriptor.format = VK_FORMAT_R32G32B32_SFLOAT;
	posDescriptor.offset = 0;	//找pos在Vertex中的偏移
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
		attributeOffset += 3 * sizeof(float);
		texCoordDescriptor.offset = attributeOffset;
		attributeDescriptions.push_back(texCoordDescriptor);
	}
	if (useTangent) {
		tangentDescriptor.binding = 0;
		tangentDescriptor.location = 3;
		tangentDescriptor.format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeOffset += 2 * sizeof(float);
		tangentDescriptor.offset = attributeOffset;
		attributeDescriptions.push_back(tangentDescriptor);
	}

	return attributeDescriptions;
}
void FzbVertexFormat::mergeUpward(FzbVertexFormat vertexFormat) {
	this->useNormal |= vertexFormat.useNormal;
	this->useTexCoord |= vertexFormat.useTexCoord;
	this->useTangent |= vertexFormat.useTangent;
}
bool FzbVertexFormat::operator==(const FzbVertexFormat& other) const {
	if (!(useNormal == other.useNormal && useTexCoord == other.useTexCoord && useTangent == other.useTangent)) {
		return false;
	}
	return true;
}

FzbLightsUniformBufferObject::FzbLightsUniformBufferObject() {}
FzbLightsUniformBufferObject::FzbLightsUniformBufferObject(uint32_t lightNum) {
	this->lightNum = lightNum;
}

FzbSemaphore::FzbSemaphore() {};
void GetSemaphoreWin32HandleKHR(VkDevice device, VkSemaphoreGetWin32HandleInfoKHR* handleInfo, HANDLE* handle) {
	auto func = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
	if (func != nullptr) {
		func(device, handleInfo, handle);
	}
}
FzbSemaphore::FzbSemaphore(VkDevice logicalDevice, bool UseExternal) {
	VkSemaphoreCreateInfo semaphoreInfo{};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkExportSemaphoreCreateInfoKHR exportInfo = {};
	if (UseExternal) {
		exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
		exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
		semaphoreInfo.pNext = &exportInfo;
	}

	VkSemaphore semaphore;
	if (vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
		throw std::runtime_error("failed to create semaphores!");
	}
	this->semaphore = semaphore;

	if (UseExternal) {
		HANDLE handle;
		VkSemaphoreGetWin32HandleInfoKHR handleInfo = {};
		handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
		handleInfo.semaphore = semaphore;
		handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
		GetSemaphoreWin32HandleKHR(logicalDevice, &handleInfo, &handle);
		this->handle = handle;
	}
}
void FzbSemaphore::clean(VkDevice logicalDevice) {
	if (semaphore != VK_NULL_HANDLE) {
		vkDestroySemaphore(logicalDevice, semaphore, nullptr);
		semaphore = VK_NULL_HANDLE;
	}

	if (handle != nullptr) { // HANDLE 在 Win32 下通常是 void* / HANDLE
		CloseHandle(handle);
		handle = nullptr;
	}
}

FzbTexture::FzbTexture() {};
FzbTexture::FzbTexture(std::string path, VkFilter filter) {
	this->path = path;
	this->filter = filter;
}
bool FzbTexture::operator==(const FzbTexture& other) const {	//只需要这两个就行
	return path == other.path && filter == other.filter;
}

FzbNumberProperty::FzbNumberProperty() {};
FzbNumberProperty::FzbNumberProperty(glm::vec4 value) {
	this->value = value;
}
bool FzbNumberProperty::operator==(const FzbNumberProperty& other) const {	//只需要这两个就行
	return value == other.value;
}
bool FzbShaderProperty::keyCompare(FzbShaderProperty& other) {
	auto getKeys = [](const auto& map) {
		std::set<std::string> keys;
		for (const auto& pair : map) keys.insert(pair.first);
		return keys;
	};

	// 比较纹理属性键
	auto thisTexKeys = getKeys(this->textureProperties);
	auto otherTexKeys = getKeys(other.textureProperties);
	if (thisTexKeys != otherTexKeys) return false;

	// 比较数值属性键
	auto thisNumKeys = getKeys(this->numberProperties);
	auto otherNumKeys = getKeys(other.numberProperties);

	return thisNumKeys == otherNumKeys;
};
bool FzbShaderProperty::operator==(const FzbShaderProperty& other) const {
	return textureProperties == other.textureProperties && numberProperties == other.numberProperties;
}

std::string getRootPath() {
	std::filesystem::path thisFile = __FILE__;
	return thisFile.parent_path().parent_path().parent_path().string();	//得到Renderer文件夹
}