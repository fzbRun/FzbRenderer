#include "./FzbPathTracingMaterial.h"

MaterialType fzbGetPathTracingMaterialType(std::string materialTypeString) {
	if (materialTypeString == "diffuse") return diffuse;
	else if (materialTypeString == "roughconductor") return roughconductor;
	else if (materialTypeString == "roughdielectric") return roughdielectric;
}
void fzbSetPathTracingNumberAttributeIndex(uint32_t materialType, glm::vec4* numberAttributeIndices, glm::vec4 numberAttribute, std::string numberAttributeType) {
	uint32_t numberAttributeIndexInArray = 0;
	if (materialType == diffuse) numberAttributeIndexInArray = 0;
	else if (materialType == roughconductor) {
		if (numberAttributeType == "albedo") numberAttributeIndexInArray = 0;
		else if (numberAttributeType == "bsdfPara") numberAttributeIndexInArray = 1;
		else {
			printf("fzbSetPathTracingNumberAttribute需要更新\n");
		}
	}
	numberAttributeIndices[numberAttributeIndexInArray] = numberAttribute;
}
void fzbSetPathTracingTextureIndex(uint32_t materialType, int* textureIndices, int textureIndex, std::string textureType) {
	uint32_t textureIndexInArray = 0;
	if (textureType == "normalMap") textureIndexInArray = 0;
	else if (textureType == "albedoMap") textureIndexInArray = 1;
	else {
		if (materialType == roughconductor) printf("fzbSetPathTracingTextureIndex需要更新\n");
	}
	textureIndices[textureIndexInArray] = textureIndex;
}

FzbPathTracingMaterialUniformObject createInitialMaterialUniformObject() {
	FzbPathTracingMaterialUniformObject material;
	material.materialType = 0;
	for (int i = 0; i < 3; ++i) material.textureIndex[i] = -1;
	for (int i = 0; i < 8; ++i) material.numberAttribute[i] = glm::vec4(1.0f);
	material.emissive = glm::vec4(0.0f);
	return material;
}