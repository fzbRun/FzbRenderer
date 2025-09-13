#include "./FzbMaterial.h"
#include "../FzbRenderer.h"
#include "../../Materials/Diffuse/FzbDiffuseMaterial.h"

std::map<std::string, std::string> materialPaths{
	{ "diffuse", "/core/Materials/Diffuse" }
};

FzbMaterial::FzbMaterial() {};
FzbMaterial::FzbMaterial(std::string id) {
	this->id = id;
}
FzbMaterial::FzbMaterial(std::string id, std::string type) {
	this->id = id;
	this->type = type;
}

void FzbMaterial::getMaterialXMLInfo() {
	if (!materialPaths.count(this->type)) throw std::runtime_error("不存在材质：" + this->type);
	std::string materialXMLPath = fzbGetRootPath() + materialPaths[this->type];
	
	pugi::xml_document doc;
	if (!doc.load_file((materialXMLPath + "/materialInfo.xml").c_str())) {
		throw std::runtime_error("pugixml打开文件失败");
	}

	pugi::xml_node materialInfos = doc.document_element();
	if (pugi::xml_node vertexFormatNode = materialInfos.child("vertexFormat")) {
		this->vertexFormat.useNormal = std::string(vertexFormatNode.child("normal").attribute("available").value()) == "true";
		this->vertexFormat.useTexCoord = std::string(vertexFormatNode.child("texCoords").attribute("available").value()) == "true";
		this->vertexFormat.useTangent = std::string(vertexFormatNode.child("tangent").attribute("available").value()) == "true";
	}
	else this->vertexFormat.available = false;

	if (pugi::xml_node attributesNode = materialInfos.child("attributes")) {
		for (pugi::xml_node attributeNode : attributesNode.children("attribute")) {
			if (std::string(attributeNode.attribute("type").value()) == "texture") {
				this->properties.textureProperties.insert({ std::string(attributeNode.attribute("name").value()), FzbTexture() });
			}
			else {
				glm::vec4 numberAttribute = fzbGetRGBAFromString(attributeNode.attribute("value").value());
				FzbNumberProperty numProperty(numberAttribute);
				this->properties.numberProperties.insert({ std::string(attributeNode.attribute("name").value()), numProperty });
			}
		}
	}
}
void FzbMaterial::getSceneXMLInfo(pugi::xml_node& materialNode) {
	if (this->type == "diffuse") {
		FzbDiffuseMaterial::getSceneXMLInfo(this, materialNode);
	}
}

void FzbMaterial::clean() {
	numberPropertiesBuffer.clean();
}
void FzbMaterial::getDescriptorNum(uint32_t& textureNum, uint32_t numberPropertyNum) {
	if (this->descriptorSet) return;
	textureNum += this->properties.textureProperties.size();
	numberPropertyNum += this->properties.numberProperties.size();
	return;
}
void FzbMaterial::createSource(std::string scenePath, std::map<std::string, FzbImage>& sceneImages) {
	if (this->hasCreateSource) return;
	if (this->properties.numberProperties.size() > 0) this->createMaterialNumberPropertiesBuffer();
	for (auto& texturePair : this->properties.textureProperties) {
		std::string texturePath = texturePair.second.path;
		if (sceneImages.count(texturePath)) {
			texturePair.second.image = &sceneImages[texturePath];
			continue;
		}
		FzbImage image;
		std::string texturePathFromModel = (this->path == "" ? scenePath : this->path) + "/" + texturePath;	//./models/xxx/textures/textureName.jpg
		image.texturePath = texturePathFromModel.c_str();
		image.filter = texturePair.second.filter;
		image.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		image.initImage();
		sceneImages.insert({ texturePath, image });
		texturePair.second.image = &sceneImages[texturePath];
	}
	this->hasCreateSource = true;
}
void FzbMaterial::createMaterialNumberPropertiesBuffer() {
	uint32_t numberNum = this->properties.numberProperties.size();
	if (numberNum > 0) {
		uint32_t bufferSize = numberNum * sizeof(glm::vec4);	//全部按floa4来存储，这个shader的布局方式很烦人的
		std::vector<glm::vec4> numberProperties;
		for (auto& property : this->properties.numberProperties) {
			numberProperties.push_back(property.second.value);
		}
		this->numberPropertiesBuffer = fzbCreateUniformBuffers(bufferSize);
		memcpy(numberPropertiesBuffer.mapped, numberProperties.data(), numberProperties.size() * sizeof(glm::vec4));
	}
}
void FzbMaterial::createMaterialDescriptor(VkDescriptorPool sceneDescriptorPool, VkDescriptorSetLayout descriptorSetLayout, std::map<std::string, FzbImage*>& sceneImages) {
	if (this->descriptorSet) return;	//如果已经创建过了，则直接返回
	uint32_t textureNum = this->properties.textureProperties.size();
	uint32_t numberNum = this->properties.numberProperties.size() > 0 ? 1 : 0;	//所有数值属性用一个storageBuffer即可
	if (textureNum + numberNum == 0) {
		return;
	}

	descriptorSet = fzbCreateDescriptorSet(sceneDescriptorPool, descriptorSetLayout);
	uint32_t descriptorNum = textureNum + numberNum;
	std::vector<VkWriteDescriptorSet> voxelGridMapDescriptorWrites(descriptorNum);
	uint32_t binding = 0;
	for (auto& pair : this->properties.textureProperties) {
		VkDescriptorImageInfo textureInfo{};
		textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo.imageView = sceneImages[pair.second.path]->imageView;
		textureInfo.sampler = sceneImages[pair.second.path]->textureSampler;
		voxelGridMapDescriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voxelGridMapDescriptorWrites[binding].dstSet = descriptorSet;
		voxelGridMapDescriptorWrites[binding].dstBinding = binding;
		voxelGridMapDescriptorWrites[binding].dstArrayElement = 0;
		voxelGridMapDescriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		voxelGridMapDescriptorWrites[binding].descriptorCount = 1;
		voxelGridMapDescriptorWrites[binding].pImageInfo = &textureInfo;
		binding++;
	}
	if (numberNum) {
		VkDescriptorBufferInfo numberBufferInfo{};
		numberBufferInfo.buffer = this->numberPropertiesBuffer.buffer;
		numberBufferInfo.offset = 0;
		numberBufferInfo.range = this->properties.numberProperties.size() * sizeof(glm::vec4);
		voxelGridMapDescriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		voxelGridMapDescriptorWrites[binding].dstSet = descriptorSet;
		voxelGridMapDescriptorWrites[binding].dstBinding = binding;
		voxelGridMapDescriptorWrites[binding].dstArrayElement = 0;
		voxelGridMapDescriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		voxelGridMapDescriptorWrites[binding].descriptorCount = 1;
		voxelGridMapDescriptorWrites[binding].pBufferInfo = &numberBufferInfo;
	}

	vkUpdateDescriptorSets(FzbRenderer::globalData.logicalDevice, voxelGridMapDescriptorWrites.size(), voxelGridMapDescriptorWrites.data(), 0, nullptr);
}
bool FzbMaterial::operator==(const FzbMaterial& other) const {
	return type == other.type && properties == other.properties && vertexFormat == vertexFormat && componentName == other.componentName;
};