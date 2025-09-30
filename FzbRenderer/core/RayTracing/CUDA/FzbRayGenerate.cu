#include "./FzbRayGenerate.cuh"
#include "./FzbGetIllumination.cuh"
#include "FzbGetIllumination.cuh"

__device__ FzbRay generateFirstRay(FzbPathTracingCameraInfo* cameraInfo, glm::vec2 screenTexel, uint32_t spp, uint32_t sppIndex) {
	glm::vec2 randomNumber = Hammersley(sppIndex, spp);
	glm::vec2 screenPos = (screenTexel + glm::vec2(randomNumber.x, randomNumber.y)) / glm::vec2(cameraInfo->screenWidth, cameraInfo->screenHeight);
	glm::vec4 ndcPos = glm::vec4(screenPos * 2.0f - 1.0f, 0.0f, 1.0f);	//vulkan�н�ƽ��ndcDepth��[0,1]
	glm::vec4 sppWorldPos = cameraInfo->inversePVMatrix * ndcPos;
	sppWorldPos /= sppWorldPos.w;
	FzbRay ray;
	ray.startPos = cameraInfo->cameraWorldPos;
	ray.direction = glm::normalize(glm::vec3(sppWorldPos) - ray.startPos);
	ray.depth = FLT_MAX;
	ray.ext = true;

	//float2 randomNumber = Hammersley(sppIndex, spp);
	//screenTexel/*screenPos*/ = (screenTexel + glm::vec2(randomNumber.x, randomNumber.y)) / glm::vec2(cameraInfo->screenWidth, cameraInfo->screenHeight);
	//glm::vec4 texelPos/*ndcPos*/ = glm::vec4(screenTexel * 2.0f - 1.0f, 0.0f, 1.0f);	//vulkan�н�ƽ��ndcDepth��[0,1]
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

	if (triangleAttribute.materialType == 0) {	//������Ҫ�Բ���
		float sinTheta = glm::sqrt(randomNumber1);
		float cosTheta = glm::sqrt(1 - sinTheta * sinTheta);
		float x = sinTheta * glm::cos(phi);
		float y = sinTheta * glm::sin(phi);
		float z = cosTheta;
		ray.direction = glm::normalize(TBN * glm::vec3(x, y, z));
		pdf *= sinTheta * cosTheta / PI;
	}
	else {
		float cosTheta = glm::sqrt((1.0f - randomNumber1) / ((triangleAttribute.roughness * triangleAttribute.roughness - 1.0f) * randomNumber1 + 1.0f));
		float sinTheta = glm::sqrt(1.0f - cosTheta * cosTheta);
		float x = sinTheta * glm::cos(phi);
		float y = sinTheta * glm::sin(phi);
		float z = cosTheta;
		glm::vec3 h = glm::vec3(x, y, z);
		h = glm::normalize(TBN * h);

		if (triangleAttribute.materialType == 1) {		//�ֲڵ���
			ray.direction = 2.0f * (glm::dot(-ray.direction, h) * h + ray.direction);	//����õ���ֻ��h������Ҫ��hתΪi
			pdf *= DistributionGGX(triangleAttribute.normal, h, triangleAttribute.roughness) * cosTheta * sinTheta / (4.0f * glm::max(glm::dot(-ray.direction, h), 0.01f));
		}
		else if (triangleAttribute.materialType == 2) {		//�ֲڵ����
			/*
			������Ҫע�⼸����
			1. ����material�м�¼��eta��mesh������������ı�ֵ�����Ե�ray��mesh�ڲ���ȥʱ��etaΪ����
			2. ����h����mesh����İ����еģ�����ʽҪ��h��o��ͬ��ģ�����������Ҫ����
			*/
			float eta = triangleAttribute.eta;
			if (!ray.ext) {	//���ڲ�(eta��¼���Ǵ��⵽�ڵ�������֮��)
				h = -h;
				eta = 1.0f / eta;
				ray.ext = true;	//�������ص��ⲿ
			}
			else ray.ext = false;
			float cosTheta_OH = glm::dot(-ray.direction, h);	//h��o��Ҫ��ͬ�࣬������ұ�Ȼ����0
			glm::vec3 F = fresnelSchlick(cosTheta_OH, triangleAttribute.albedo);
			float F_oneChanel = glm::max(0.299 * F.x + 0.587 * F.y + 0.114 * F.z, 0.1);
			float randomNumber3 = rand(randomNumberSeed);
			if (randomNumber3 < F_oneChanel) {	//����
				ray.refraction = false;
				pdf *= F_oneChanel;
				ray.direction = 2.0f * glm::dot(-ray.direction, h) * h + ray.direction;
				pdf *= DistributionGGX(triangleAttribute.normal, h, triangleAttribute.roughness) * cosTheta * sinTheta / (4.0f * glm::max(glm::dot(-ray.direction, h), 0.01f));
			}
			else {	//����
				ray.refraction = true;
				pdf *= 1.0f - F_oneChanel;
				//glm::vec3 o = -ray.direction;
				ray.direction = glm::normalize((eta * cosTheta_OH - (glm::sqrt(1.0f + eta * eta * (cosTheta_OH * cosTheta_OH - 1.0f)))) * h + eta * ray.direction);
				float weight = (eta * eta) / ((1.0f + eta * eta + 2.0f * eta) * cosTheta_OH);
				//uint32_t threadIndx = threadIdx.x + blockDim.x * blockIdx.x;
				//printf("rayGenerate:%d  %f %f %f\n", threadIdx.x + blockDim.x * blockIdx.x, h.x, h.y, h.z);
				//printf("%f\n", glm::length(glm::abs(h) - glm::abs(glm::normalize(ray.direction + eta * o))));
				//printf("%f\n", glm::dot(glm::normalize(ray.direction + eta * o), triangleAttribute.normal));
				//printf("rayGenerate:%d %f %f %f %f\n", threadIdx.x + blockDim.x * blockIdx.x, h.x, ray.direction.x, ray.direction.y, ray.direction.z);
				pdf *= DistributionGGX(triangleAttribute.normal, h, triangleAttribute.roughness) * weight;
			}
		}
	}
	ray.startPos = ray.direction * 0.001f + ray.hitPos;
	ray.depth = FLT_MAX;
}