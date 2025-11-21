#include "./FzbRayGenerate.cuh"
#include "./FzbGetIllumination.cuh"
#include "FzbGetIllumination.cuh"

__device__ FzbRay generateFirstRay(FzbPathTracingCameraInfo* cameraInfo, glm::vec2 screenTexel, uint32_t spp, uint32_t sppIndex) {
	glm::vec2 randomNumber = Hammersley(sppIndex, spp);
	glm::vec2 screenPos = (screenTexel + glm::vec2(randomNumber.x, randomNumber.y)) / glm::vec2(cameraInfo->screenWidth, cameraInfo->screenHeight);
	glm::vec4 ndcPos = glm::vec4(screenPos * 2.0f - 1.0f, 0.0f, 1.0f);	//vulkan中近平面ndcDepth在[0,1]
	glm::vec4 sppWorldPos = cameraInfo->inversePVMatrix * ndcPos;
	sppWorldPos /= sppWorldPos.w;
	FzbRay ray;
	ray.startPos = cameraInfo->cameraWorldPos;
	ray.direction = glm::normalize(glm::vec3(sppWorldPos) - ray.startPos);
	ray.depth = FLT_MAX;
	ray.ext = true;

	//float2 randomNumber = Hammersley(sppIndex, spp);
	//screenTexel/*screenPos*/ = (screenTexel + glm::vec2(randomNumber.x, randomNumber.y)) / glm::vec2(cameraInfo->screenWidth, cameraInfo->screenHeight);
	//glm::vec4 texelPos/*ndcPos*/ = glm::vec4(screenTexel * 2.0f - 1.0f, 0.0f, 1.0f);	//vulkan中近平面ndcDepth在[0,1]
	//texelPos = cameraInfo->inversePVMatrix * texelPos;
	//texelPos /= texelPos.w;

	return ray;
}
__device__ void generateRay(const FzbTriangleAttribute& triangleAttribute, float& pdf, FzbRay& ray, uint32_t& randomNumberSeed) {
	float randomNumber1 = rand(randomNumberSeed);
	float randomNumber2 = rand(randomNumberSeed);
	float phi = randomNumber2 * 2 * PI;
	glm::vec3 bitangent = glm::cross(triangleAttribute.normal, triangleAttribute.tangent) * triangleAttribute.handed;
	glm::mat3 TBN = glm::mat3(triangleAttribute.tangent, bitangent, triangleAttribute.normal);

	if (triangleAttribute.materialType == 0) {
		//余弦重要性采样
		float cosTheta = sqrt(randomNumber1);
		float sinTheta = glm::sqrt(1 - cosTheta * cosTheta);
		float x = sinTheta * glm::cos(phi);
		float y = sinTheta * glm::sin(phi);
		float z = cosTheta;
		ray.direction = glm::normalize(TBN * glm::vec3(x, y, z));
		pdf *= z * PI_countdown;
	}
	else if(triangleAttribute.materialType <= 2){
		float cosTheta = glm::sqrt((1.0f - randomNumber1) / ((triangleAttribute.roughness * triangleAttribute.roughness - 1.0f) * randomNumber1 + 1.0f));
		float sinTheta = glm::sqrt(1.0f - cosTheta * cosTheta);
		float x = sinTheta * glm::cos(phi);
		float y = sinTheta * glm::sin(phi);
		float z = cosTheta;
		glm::vec3 h = glm::vec3(x, y, z);
		h = glm::normalize(TBN * h);

		float eta = triangleAttribute.eta;
		if (!ray.ext) {
			h = -h;
			eta = 1.0f / eta;
		}
		float cosTheta_OH = glm::dot(-ray.direction, h);	//h与o需要是同侧，因此余弦必然大于0

		if (triangleAttribute.materialType == 1) {		//粗糙导体
			pdf *= glm::max(DistributionGGX(triangleAttribute.normal, h, triangleAttribute.roughness) * cosTheta / (4.0f * glm::max(glm::dot(-ray.direction, h), 0.01f)), 0.001f);
			ray.direction = glm::normalize(2.0f * cosTheta_OH * h + ray.direction);	//这里得到的只是h，还需要从h转为i
		}
		else if (triangleAttribute.materialType == 2) {		//粗电解质
			float F_oneChanel;
			float randomNumber3 = rand(randomNumberSeed);
			if (eta * eta * (1.0f - cosTheta_OH * cosTheta_OH) >= 1.0f) F_oneChanel = 1.0f;	//全反射
			else {
				glm::vec3 F = fresnelSchlick(cosTheta_OH, triangleAttribute.albedo);
				F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;
			}

			if (randomNumber3 < F_oneChanel) {	//反射
				ray.refraction = false;
				pdf *= F_oneChanel;
				pdf *= glm::max(DistributionGGX(triangleAttribute.normal, h, triangleAttribute.roughness) * cosTheta / (4.0f * glm::max(cosTheta_OH, 0.01f)), 0.001f);
				ray.direction = glm::normalize(2.0f * cosTheta_OH * h + ray.direction);
			}
			else{	//折射
				ray.ext = !ray.ext;

				ray.refraction = true;
				pdf *= 1.0f - F_oneChanel;
				
				ray.direction = glm::normalize
				(
					(eta * cosTheta_OH -
						(glm::sqrt(1.0f + eta * eta * (cosTheta_OH * cosTheta_OH - 1.0f)))
					) * h +
					eta * ray.direction
				);

				float weight = glm::dot(ray.direction, h) + eta * cosTheta_OH;
				weight *= weight;
				weight = (eta * eta * cosTheta_OH) / weight;
				pdf *= glm::max(DistributionGGX(triangleAttribute.normal, h, triangleAttribute.roughness) * weight * cosTheta, 0.001f);
			}
		}
	}
	else if (triangleAttribute.materialType == 3) {
		float eta = triangleAttribute.eta;
		glm::vec3 h = triangleAttribute.normal;
		if (!ray.ext) {
			h = -h;
			eta = 1.0f / eta;
		}
		float cosTheta_OH = glm::dot(-ray.direction, h);

		float F_oneChanel;
		float randomNumber3 = rand(randomNumberSeed);
		if (eta * eta * (1.0f - cosTheta_OH * cosTheta_OH) >= 1.0f) F_oneChanel = 1.0f;	//全反射
		else {
			glm::vec3 F = fresnelSchlick(cosTheta_OH, triangleAttribute.albedo);
			F_oneChanel = glm::max(0.299 * F.x + 0.587 * F.y + 0.114 * F.z, 0.1);
		}

		if (randomNumber3 < F_oneChanel) {	//反射
			ray.refraction = false;
			pdf *= F_oneChanel;
			ray.direction = 2.0f * cosTheta_OH * h + ray.direction;
		}
		else {
			ray.ext = !ray.ext;

			ray.refraction = true;
			pdf *= 1.0f - F_oneChanel;

			ray.direction = glm::normalize
			(
				(
					eta * cosTheta_OH -
					(glm::sqrt(1.0f + eta * eta * (cosTheta_OH * cosTheta_OH - 1.0f)))
					) * h +
				eta * ray.direction
			);
		}
	}
	ray.startPos = ray.direction * 0.001f + ray.hitPos;
	ray.depth = FLT_MAX;
}