#include "./FzbPathTracingMaterial.h"

MaterialType getMaterialType(std::string materialTypeString) {
	if (materialTypeString == "diffuse") return diffuse;
	else if (materialTypeString == "roughconductor") return roughconductor;
}

FzbPathTracingMaterialUniformObject createInitialMaterialUniformObject() {
	FzbPathTracingMaterialUniformObject material;
	material.materialType = 0;
	for (int i = 0; i < 3; ++i) material.textureIndex[i] = -1;
	for (int i = 0; i < 8; ++i) material.numberAttribute[i] = glm::vec4(1.0f);
	material.emissive = glm::vec4(0.0f);
	return material;
}