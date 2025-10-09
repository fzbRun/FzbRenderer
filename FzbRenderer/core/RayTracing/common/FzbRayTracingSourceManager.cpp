#include "./FzbRayTracingSourceManager.h"
#include "../../common/FzbRenderer.h"
#include "./FzbRayTracingMaterial.h"
#include <unordered_map>

FzbRayTracingSourceManager::FzbRayTracingSourceManager() {
	this->sourceManagerCuda = std::make_shared<FzbRayTracingSourceManager_Cuda>();
};
void FzbRayTracingSourceManager::createSource() {
	std::unordered_map<std::string, int> sceneImagePaths;
	for (auto& materialPair : FzbRenderer::globalData.mainScene.sceneMaterials) {
		FzbMaterial& material = materialPair.second;
		FzbRayTracingMaterialUniformObject materialUniformObject = createInitialMaterialUniformObject();
		materialUniformObject.materialType = fzbGetRayTracingMaterialType(material.type);

		for (auto& texturePair : material.properties.textureProperties) {
			FzbTexture& texture = texturePair.second;
			if (!sceneImagePaths.count(texture.path)) {
				FzbImage image;
				std::string texturePathFromModel = FzbRenderer::globalData.mainScene.scenePath + "/" + texture.path;
				image.texturePath = texturePathFromModel.c_str();
				image.filter = texturePair.second.filter;
				image.layout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR;
				image.UseExternal = true;
				image.initImage();
				this->sceneTextures.push_back(image);
				sceneImagePaths.insert({ texture.path,this->sceneTextures.size() - 1 });
			}
			texture.image = &this->sceneTextures[sceneImagePaths[texture.path]];
			materialUniformObject.textureIndex[material.getMaterialAttributeIndex(texturePair.first)] = sceneImagePaths[texture.path];
		}

		for (auto& numberPropertyPair : material.properties.numberProperties) {
			FzbNumberProperty& numberProperty = numberPropertyPair.second;
			if (numberPropertyPair.first == "emissive") materialUniformObject.emissive = numberProperty.value;
			else materialUniformObject.numberAttribute[material.getMaterialAttributeIndex(numberPropertyPair.first)] = numberProperty.value;
		}
		this->sceneMaterialInfoArray.push_back(materialUniformObject);
	}

	VkExtent2D resolution = FzbRenderer::globalData.getResolution();
	this->rayTracingResultBuffer = fzbCreateStorageBuffer(resolution.width * resolution.height * sizeof(float4), true);
	this->rayTracingFinishedSemphore = FzbSemaphore(true);

	FzbRayTracingCudaSourceSet sourceSet;
	sourceSet.rayTracingResultBuffer = this->rayTracingResultBuffer;
	sourceSet.rayTracingFinishedSemphore = this->rayTracingFinishedSemphore;
	sourceSet.sceneVertices = FzbRenderer::globalData.mainScene.vertexBuffer;
	sourceSet.sceneTextures = this->sceneTextures;
	sourceSet.sceneMaterialInfoArray = this->sceneMaterialInfoArray;
	sourceSet.bvhNodeCount = this->bvh->bvhCuda->triangleNum * 2 - 1;
	sourceSet.bvhNodeArray = this->bvh->bvhCuda->bvhNodeArray;
	sourceSet.bvhTriangleInfoArray = this->bvh->bvhCuda->bvhTriangleInfoArray;

	for (int i = 0; i < FzbRenderer::globalData.mainScene.sceneLights.size(); ++i) {
		FzbLight& light = FzbRenderer::globalData.mainScene.sceneLights[i];
		if (light.type == FZB_POINT) {
			++sourceSet.pointLightCount;
			sourceSet.pointLightInfoArray.push_back({ glm::vec4(light.position, 0.0f), glm::vec4(light.strength, 0.0f) });
		}
		else if (light.type == FZB_AREA) {
			++sourceSet.areaLightCount;
			FzbRayTracingAreaLight areaLight;
			areaLight.worldPos = glm::vec4(light.position, 0.0f);
			areaLight.normal = glm::vec4(light.normal, 0.0f);
			areaLight.radiance = glm::vec4(light.strength, 0.0f);
			areaLight.edge0 = glm::vec4(light.edge0, 0.0f);
			areaLight.edge1 = glm::vec4(light.edge1, 0.0f);
			areaLight.area = light.area;
			sourceSet.areaLightInfoArray.push_back(areaLight);
		}
	}

	this->sourceManagerCuda->initRayTracingSource(sourceSet);
}
void FzbRayTracingSourceManager::clean() {
	sourceManagerCuda->clean();
	rayTracingResultBuffer.clean();
	rayTracingFinishedSemphore.clean();
	for (int i = 0; i < this->sceneTextures.size(); ++i) this->sceneTextures[i].clean();
}