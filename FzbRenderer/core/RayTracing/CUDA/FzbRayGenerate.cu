#include "./FzbRayGenerate.cuh"
#include "./FzbGetIllumination.cuh"

__device__ FzbRay generateFirstRay(FzbPathTracingCameraInfo* cameraInfo, glm::vec2 screenTexel, uint32_t spp, uint32_t sppIndex) {
	float2 randomNumber = Hammersley(sppIndex, spp);
	glm::vec2 screenPos = (screenTexel + glm::vec2(randomNumber.x, randomNumber.y)) / glm::vec2(cameraInfo->screenWidth, cameraInfo->screenHeight);
	glm::vec4 ndcPos = glm::vec4(screenPos * 2.0f - 1.0f, 0.0f, 1.0f);	//vulkan�н�ƽ��ndcDepth��[0,1]
	glm::vec4 sppWorldPos = cameraInfo->inversePVMatrix * ndcPos;
	sppWorldPos /= sppWorldPos.w;
	FzbRay ray;
	ray.startPos = cameraInfo->cameraWorldPos;
	ray.direction = glm::normalize(glm::vec3(sppWorldPos) - ray.startPos);
	ray.depth = FLT_MAX;
	return ray;
}
__device__ void generateRay(uint32_t distributionType, float& pdf, FzbRay& ray, uint32_t& randomNumberSeed) {
	//������Ҫ�Բ���
	float randomNumber1 = getRandomNumber(randomNumberSeed);
	float randomNumber2 = getRandomNumber(randomNumberSeed);
	/*
	* theta = glm::asin(glm::sqrt(randomNumber1))����ôsinTheta = glm::sqrt(randomNumber1)
	* ������f(x)=1�����ȷֲ���y = g(x) = sqrt(x),  ��ôx = g^-1(y) = y^2�������ǵ����ӳ�䣬����Ҫ����һ���ſɱ�����ʽ
	* f(y) = f(g^-1(y)) (dg^-1(y)/dy) = f(y^2) * 2y
	* �������ǿ���ֱ��ȡsinTheta^2 = randomNumber1����pdf = 1 * 2 * randomNumber1
	*/
	float sinTheta = glm::sqrt(randomNumber1);
	float cosTheta = glm::sqrt(1 - sinTheta * sinTheta);
	float phi = randomNumber2 * 2 * PI;
	float x = sinTheta * glm::cos(phi);
	float y = sinTheta * glm::sin(phi);
	float z = cosTheta;
	ray.direction = glm::vec3(x, y, z);
	ray.startPos = ray.direction * 0.001f + ray.hitPos;
	ray.depth = FLT_MAX;

	pdf *= sinTheta * cosTheta / PI;
}