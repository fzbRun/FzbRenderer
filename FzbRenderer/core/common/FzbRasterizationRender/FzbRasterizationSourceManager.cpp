#include "./FzbRasterizationSourceManager.h"
#include <unordered_map>

FzbRasterizationSourceManager::FzbRasterizationSourceManager() {};

void FzbRasterizationSourceManager::createCanvas(FzbMaterial material) {
	if (!this->componentScene.sceneMaterials.count(material.id)) this->componentScene.sceneMaterials.insert({ material.id, material });
	this->meshMaterialPairs.insert({ &this->canvans, &this->componentScene.sceneMaterials[material.id] });
}
void FzbRasterizationSourceManager::addMeshMaterial(std::vector<FzbMesh>& meshes, FzbMaterial material, bool loopRender) {
	if (material == FzbMaterial()) {
		for (int i = 0; i < meshes.size(); ++i) {
			FzbMesh& mesh = meshes[i];
			this->meshMaterialPairs.insert({ &mesh, mesh.material });
			for (auto& texturePair : mesh.material->properties.textureProperties) {
				if (!this->images.count(texturePair.second.path)) this->images.insert({ texturePair.second.path, texturePair.second.image});
			}
		}
		return;
	}
	std::unordered_map<FzbMaterial, FzbMaterial*> uniqueMaterials;
	for (int i = 0; i < meshes.size(); ++i) {
		FzbMesh& mesh = meshes[i];

		FzbMaterial meshMaterial = material;
		meshMaterial.id = material.id + std::to_string(i);
		meshMaterial.vertexFormat = loopRender ? mesh.vertexFormat : FzbRenderer::globalData.mainScene.vertexFormat_allMesh_prepocess;

		FzbMaterial* meshMaterial_ptr;
		if (!uniqueMaterials.count(meshMaterial)) {
			this->componentScene.sceneMaterials.insert({ meshMaterial.id, meshMaterial });
			meshMaterial_ptr = &componentScene.sceneMaterials[meshMaterial.id];
			uniqueMaterials.insert({ meshMaterial, meshMaterial_ptr });

			for (auto& texturePair : meshMaterial.properties.textureProperties) {
				if (!this->images.count(texturePair.second.path)) this->images.insert({ texturePair.second.path, texturePair.second.image });
			}
		}
		else meshMaterial_ptr = uniqueMaterials[meshMaterial];
		this->meshMaterialPairs.insert({ &mesh, meshMaterial_ptr });
	}
}
void FzbRasterizationSourceManager::addMeshMaterial(FzbMesh* mesh, FzbMaterial material, bool loopRender) {
	material.vertexFormat = loopRender ? mesh->vertexFormat : FzbRenderer::globalData.mainScene.vertexFormat_allMesh_prepocess;
	if (!this->componentScene.sceneMaterials.count(material.id)) {
		this->componentScene.sceneMaterials.insert({ material.id, material });
		for (auto& texturePair : material.properties.textureProperties) {
			if (!this->images.count(texturePair.second.path)) this->images.insert({ texturePair.second.path, texturePair.second.image });
		}
	}
	FzbMaterial* meshMaterial_ptr = &componentScene.sceneMaterials[material.id];
	this->meshMaterialPairs.insert({ mesh, meshMaterial_ptr});
}

void FzbRasterizationSourceManager::addSource(std::map<std::string, FzbShaderInfo> shaderInfos) {
	createShader(shaderInfos);
	createShaderMeshBatch();
	createBufferAndDescriptorOfMaterial();
};

void FzbRasterizationSourceManager::clean() {
	VkDevice logicalDevice = FzbRenderer::globalData.logicalDevice;
	if (descriptorPool) vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
	for (auto& shaderPair : shaderSet) shaderPair.second.clean();
	componentScene.clean();
}

void FzbRasterizationSourceManager::createShader(std::map<std::string, FzbShaderInfo> shaderInfos) {
	std::unordered_set<FzbMaterial*> uniqueMaterial;
	for (auto& meshMaterialPair : this->meshMaterialPairs) {
		FzbMaterial* material = meshMaterialPair.second;
		if (uniqueMaterial.count(material)) continue;
		else uniqueMaterial.insert(material);
		FzbShaderInfo shaderInfo;
		if (material->id == "defaultMaterial") shaderInfo = shaderInfos.at("diffuse"); //创造默认shader，匹配默认material
		else if (shaderInfos.count(material->type)) shaderInfo = shaderInfos[material->type];
		else continue;
			
		if (!this->shaderSet.count(shaderInfo.shaderPath)) this->shaderSet.insert({ shaderInfo.shaderPath, FzbShader(fzbGetRootPath() + shaderInfo.shaderPath, shaderInfo.extensions, shaderInfo.staticCompile) });
		this->shaderSet[shaderInfo.shaderPath].createShaderVariant(material);
	}
}
void FzbRasterizationSourceManager::createShaderMeshBatch() {
	for (auto& pair : shaderSet) {
		FzbShader& shader = pair.second;
		shader.createMeshBatch(this->meshMaterialPairs);
		this->shaders_vector.push_back(&shader);
	}
}
/*
创建我所需要渲染的mesh的material资源，如buffer和imgae，如果它没有被创建过的话（createSource函数内部判断）
然后创建渲染material所需的描述符，如果它没被创建过的化（createDescriptor函数内部判断）
所以只需要创建该组件独有的material的资源和描述符即可，复用部分其他组件或mainScene已经创建过了。
*/
void FzbRasterizationSourceManager::createBufferAndDescriptorOfMaterial() {
	//为组件独有的material创建资源
	for (auto& material : this->componentScene.sceneMaterials) {
		material.second.createSource("", this->componentScene.sceneImages);	//mainScene的mateiral归mainScene管
	}
	uint32_t textureNum = 0;
	uint32_t numberBufferNum = 0;
	for (auto& meshMaterialPair : this->componentScene.sceneMaterials) {
		FzbMaterial& material = meshMaterialPair.second;
		material.getDescriptorNum(textureNum, numberBufferNum);
		for (auto& texturePair : material.properties.textureProperties) {
			this->images.insert({ texturePair.second.path, texturePair.second.image });
		}
	}
	std::map<VkDescriptorType, uint32_t> bufferTypeAndNum = { {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, textureNum},		//纹理信息
															{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, numberBufferNum} };	//材质信息
	this->descriptorPool = fzbCreateDescriptorPool(bufferTypeAndNum);

	//创建shader的描述符集合布局和material的描述符集合，不会创造非独有的material的描述符集合
	for (auto& shader : shaderSet) {
		shader.second.createDescriptor(descriptorPool);
	}
}

