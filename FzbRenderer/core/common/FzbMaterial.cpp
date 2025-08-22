#include "./FzbMaterial.h"

FzbMaterial::FzbMaterial() {};
FzbMaterial::FzbMaterial(VkDevice logicalDevice) {
	this->logicalDevice = logicalDevice;
};
FzbMaterial::FzbMaterial(VkDevice logicalDevice, std::string id) {
	this->logicalDevice = logicalDevice;
	this->id = id;
}
FzbMaterial::FzbMaterial(VkDevice logicalDevice, std::string id, std::string type) {
	this->logicalDevice = logicalDevice;
	this->id = id;
	this->type = type;
}
void FzbMaterial::clean() {
	numberPropertiesBuffer.clean();
}
void FzbMaterial::createSource(VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue, std::string scenePath, uint32_t& numberBufferNum, std::map<std::string, FzbImage>& sceneImages) {
	if (this->properties.numberProperties.size() > 0) {
		this->createMaterialNumberPropertiesBuffer(physicalDevice, numberBufferNum);
	}
	for (auto& texturePair : this->properties.textureProperties) {
		std::string texturePath = texturePair.second.path;
		if (sceneImages.count(texturePath)) {
			continue;
		}
		FzbImage image;
		std::string texturePathFromModel = scenePath + "/" + texturePath;	//./models/xxx/textures/textureName.jpg
		image.texturePath = texturePathFromModel.c_str();
		image.filter = texturePair.second.filter;
		image.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		image.fzbCreateImage(physicalDevice, logicalDevice, commandPool, graphicsQueue);
		sceneImages.insert({ texturePath, image });
	}
}
void FzbMaterial::createMaterialNumberPropertiesBuffer(VkPhysicalDevice physicalDevice, uint32_t& numberBufferNum) {
	uint32_t numberNum = this->properties.numberProperties.size();
	if (numberNum > 0) {
		numberBufferNum++;
		uint32_t bufferSize = numberNum * sizeof(glm::vec4);	//ȫ����floa4���洢�����shader�Ĳ��ַ�ʽ�ܷ��˵�
		std::vector<glm::vec4> numberProperties;
		for (auto& property : this->properties.numberProperties) {
			numberProperties.push_back(property.second.value);
		}
		this->numberPropertiesBuffer = fzbCreateUniformBuffers(physicalDevice, logicalDevice, bufferSize);
		memcpy(numberPropertiesBuffer.mapped, numberProperties.data(), numberProperties.size() * sizeof(glm::vec4));
	}
}
void FzbMaterial::createMaterialDescriptor(VkDescriptorPool sceneDescriptorPool, VkDescriptorSetLayout descriptorSetLayout, std::map<std::string, FzbImage>& sceneImages) {

	uint32_t textureNum = this->properties.textureProperties.size();
	uint32_t numberNum = this->properties.numberProperties.size() > 0 ? 1 : 0;	//������ֵ������һ��storageBuffer����
	if (textureNum + numberNum == 0) {
		return;
	}
	/*
	std::vector<VkDescriptorType> type(textureNum + numberNum);
	std::vector<VkShaderStageFlags> stage(textureNum + numberNum);
	for (int i = 0; i < textureNum; i++) {
		type[i] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		stage[i] = VK_SHADER_STAGE_ALL;	//shaderStage����Ϊall���ص�shader����Ӱ�����ݵĶ�ȡ�ٶȣ�ֻ���ڱ���ʱ���ⷶΧ��ͬ���ѣ�Ӱ������ٶȶ��ѣ���
	}
	if (numberNum) {
		type[textureNum] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		stage[textureNum] = VK_SHADER_STAGE_ALL;
	}
	this->descriptorSetLayout = fzbCreateDescriptLayout(logicalDevice, type.size(), type, stage);
	*/

	descriptorSet = fzbCreateDescriptorSet(logicalDevice, sceneDescriptorPool, descriptorSetLayout);
	uint32_t descriptorNum = textureNum + numberNum;
	std::vector<VkWriteDescriptorSet> voxelGridMapDescriptorWrites(descriptorNum);
	uint32_t binding = 0;
	for (auto& pair : this->properties.textureProperties) {
		VkDescriptorImageInfo textureInfo{};
		textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		textureInfo.imageView = sceneImages[pair.second.path].imageView;
		textureInfo.sampler = sceneImages[pair.second.path].textureSampler;
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

	vkUpdateDescriptorSets(logicalDevice, voxelGridMapDescriptorWrites.size(), voxelGridMapDescriptorWrites.data(), 0, nullptr);
}
bool FzbMaterial::operator==(const FzbMaterial& other) const {
	return id == other.id;
};