#include "./FzbGetIllumination.cuh"
#include "FzbCollisionDetection.cuh"

__device__ float DistributionGGX(const glm::vec3& N, const glm::vec3& H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = glm::abs(glm::dot(N, H));
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = glm::max(PI * denom * denom, 0.000001f);

	return nom / denom;
}
__device__ float GeometrySchlickGGX(float cosine, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float nom = cosine;
	float denom = glm::max(cosine * (1.0f - k) + k, 0.0001f);

	return nom / denom;
}
__device__ float GeometrySmith(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, float roughness, bool refrction = false)
{
	float NdotV = refrction ? abs(glm::dot(N, V)) : max(glm::dot(N, V), 0.0f);
	float NdotL = refrction ? abs(glm::dot(N, L)) : max(glm::dot(N, L), 0.0f);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}
__device__ glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& F0)
{
	return F0 + (1.0f - F0) * pow(glm::clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}
__device__ glm::vec3 getBSDF(const FzbTriangleAttribute& triangleAttribute, const glm::vec3& incidence, const glm::vec3& outgoing, const FzbRay& ray) {
	if(triangleAttribute.materialType == 0) return glm::vec3(PI_countdown) * triangleAttribute.albedo;
	else if (triangleAttribute.materialType == 3) {
		float eta = ray.ext ? 1.0f / triangleAttribute.eta : triangleAttribute.eta;
		float cosTheta_OH = abs(glm::dot(triangleAttribute.normal, outgoing));

		glm::vec3 F = fresnelSchlick(cosTheta_OH, triangleAttribute.albedo);
		if (ray.refraction) {
			float cosTheta_IH = abs(glm::dot(incidence, triangleAttribute.normal));
			float weight = cosTheta_OH - (1.0f / eta) * cosTheta_IH;
			weight = weight * weight;
			return (1.0f - F) / weight;
		}
		else return F / (4.0f * cosTheta_OH * cosTheta_OH);
	}
	else {
		if (ray.refraction && triangleAttribute.materialType == 2) {
			float eta = ray.ext ? 1.0f / triangleAttribute.eta : triangleAttribute.eta;
			glm::vec3 h = glm::normalize(incidence + outgoing * eta);
			float cosTheta_OH = abs(glm::dot(outgoing, h));
			float cosTheta_IH = abs(glm::dot(incidence, h));

			float NDF = DistributionGGX(triangleAttribute.normal, h, triangleAttribute.roughness);
			float G = GeometrySmith(triangleAttribute.normal, outgoing, incidence, triangleAttribute.roughness, true);
			glm::vec3 F = fresnelSchlick(glm::abs(glm::dot(h, outgoing)), triangleAttribute.albedo);

			float weight = cosTheta_OH - (1.0f / eta) * cosTheta_IH;
			weight = weight * weight;
			weight = (cosTheta_IH * cosTheta_OH) / (glm::dot(incidence, triangleAttribute.normal) * glm::dot(outgoing, triangleAttribute.normal)) / weight;
			weight = glm::abs(weight);
			glm::vec3 ft = NDF * G * (1.0f - F) * weight;
			return ft;
		}
		else {
			glm::vec3 h = glm::normalize(incidence + outgoing);
			float NDF = DistributionGGX(triangleAttribute.normal, h, triangleAttribute.roughness);
			float G = GeometrySmith(triangleAttribute.normal, outgoing, incidence, triangleAttribute.roughness);
			glm::vec3 F = fresnelSchlick(glm::max(glm::dot(h, outgoing), 0.0f), triangleAttribute.albedo);
			glm::vec3 fr = NDF * G * F;
			float weight = 4.0f * glm::max(glm::dot(triangleAttribute.normal, outgoing), 0.0f) * glm::max(glm::dot(triangleAttribute.normal, incidence), 0.0f) + 0.01f;

			//if (!isfinite(fr.x / weight) || !isfinite(fr.y / weight) || !isfinite(fr.z / weight)) printf("fr:%f %f %f weight:%f\n", fr.x, fr.y, fr.z, weight);
			//if (!isfinite(fr.x / weight) || !isfinite(fr.y / weight) || !isfinite(fr.z / weight)) printf("NDF:%f G:%f F:%f %f %f\n", NDF, G, F.x, F.y, F.z);
			//if (!isfinite(fr.x / weight) || !isfinite(fr.y / weight) || !isfinite(fr.z / weight)) printf("normal:%f %f %f\n", triangleAttribute.normal.x, triangleAttribute.normal.y, triangleAttribute.normal.z);
			//if (!isfinite(fr.x / weight) || !isfinite(fr.y / weight) || !isfinite(fr.z / weight)) printf("h:%f %f %f\n", h.x, h.y, h.z);
			//if (!isfinite(fr.x / weight) || !isfinite(fr.y / weight) || !isfinite(fr.z / weight)) printf("roughness:%f\n", triangleAttribute.roughness);
			//if (!isfinite(fr.x / weight) || !isfinite(fr.y / weight) || !isfinite(fr.z / weight)) printf("h:%f %f %f\n", h.x, h.y, h.z);
			//if (!isfinite(fr.x / weight) || !isfinite(fr.y / weight) || !isfinite(fr.z / weight)) printf("incidence:%f %f %f\n", incidence.x, incidence.y, incidence.z);
			//if (!isfinite(fr.x / weight) || !isfinite(fr.y / weight) || !isfinite(fr.z / weight)) printf("outgoing:%f %f %f\n", outgoing.x, outgoing.y, outgoing.z);
			
			return fr /= weight;
		}
	}
	return glm::vec3(0.0f);
}

__device__ FzbQuadrilateral_resue initQuadrilateral(glm::vec3& hitPos, glm::vec3& quadRangleStartPos, glm::vec3& ex, glm::vec3& ey) {
	FzbQuadrilateral_resue quadrangle;
	quadrangle.o = hitPos;
	float exl = glm::length(ex);
	float eyl = glm::length(ey);
	quadrangle.x = ex / exl;
	quadrangle.y = ey / eyl;
	quadrangle.z = glm::cross(quadrangle.x, quadrangle.y);

	glm::vec3 d = quadRangleStartPos - quadrangle.o;
	quadrangle.z0 = glm::dot(d, quadrangle.z);
	if (quadrangle.z0 > 0) {
		quadrangle.z *= -1.0f;
		quadrangle.z0 *= -1.0f;
	}
	quadrangle.z0sq = quadrangle.z0 * quadrangle.z0;

	quadrangle.x0 = glm::dot(d, quadrangle.x);
	quadrangle.y0 = glm::dot(d, quadrangle.y);
	quadrangle.x1 = quadrangle.x0 + exl;
	quadrangle.y1 = quadrangle.y0 + eyl;
	quadrangle.y0sq = quadrangle.y0 * quadrangle.y0;
	quadrangle.y1sq = quadrangle.y1 * quadrangle.y1;

	glm::vec3 v00 = glm::vec3(quadrangle.x0, quadrangle.y0, quadrangle.z0);
	glm::vec3 v01 = glm::vec3(quadrangle.x0, quadrangle.y1, quadrangle.z0);
	glm::vec3 v10 = glm::vec3(quadrangle.x1, quadrangle.y0, quadrangle.z0);
	glm::vec3 v11 = glm::vec3(quadrangle.x1, quadrangle.y1, quadrangle.z0);

	glm::vec3 n0 = glm::vec3(0.0f, quadrangle.z0, -quadrangle.y0);//glm::normalize(cross(v00, v10));
	n0.z /= glm::sqrt(quadrangle.z0sq + quadrangle.y0sq);	//后续只有z需要被用到，x，y会乘以0，不用管
	glm::vec3 n1 = glm::vec3(-quadrangle.z0, 0.0f, quadrangle.x1);//glm::normalize(cross(v10, v11));
	n1.z /= glm::sqrt(quadrangle.z0sq + quadrangle.x1 * quadrangle.x1);
	glm::vec3 n2 = glm::vec3(0.0f, -quadrangle.z0, quadrangle.y1);//glm::normalize(cross(v11, v01));
	n2.z /= glm::sqrt(quadrangle.z0sq + quadrangle.y1sq);
	glm::vec3 n3 = glm::vec3(quadrangle.z0, 0.0f, -quadrangle.x0);//glm::normalize(cross(v01, v00));
	n3.z /= glm::sqrt(quadrangle.z0sq + quadrangle.x0 * quadrangle.x0);

	float g0 = glm::acos(-n0.z * n1.z);	//glm::acos(-glm::dot(n0, n1));
	float g1 = glm::acos(-n1.z * n2.z);	//glm::acos(-glm::dot(n1, n2));
	float g2 = glm::acos(-n2.z * n3.z);	//glm::acos(-glm::dot(n2, n3));
	float g3 = glm::acos(-n3.z * n0.z);	//glm::acos(-glm::dot(n3, n0));

	quadrangle.b0 = n0.z;
	quadrangle.b1 = n2.z;
	quadrangle.b0sq = quadrangle.b0 * quadrangle.b0;
	quadrangle.k = 2 * PI - g2 - g3;
	quadrangle.S = g0 + g1 - quadrangle.k;

	return quadrangle;
}
__device__ glm::vec3 sphericalRectangleSample(const FzbQuadrilateral_resue& quadrangle, float u, float v, float& pdf) {
	pdf *= 1.0f / quadrangle.S;

	float au = u * quadrangle.S + quadrangle.k;
	float fu = (glm::cos(au) * quadrangle.b0 - quadrangle.b1) / glm::sin(au);
	float cu = 1.0f / glm::sqrt(fu * fu + quadrangle.b0sq) * (fu > 0.0f ? 1.0f : -1.0f);
	cu = glm::clamp(cu, -1.0f, 1.0f);
	float xu = -(cu * quadrangle.z0) / glm::sqrt(1.0f - cu * cu);
	xu = glm::clamp(xu, quadrangle.x0, quadrangle.x1);

	float dl = sqrt(xu * xu + quadrangle.z0sq);
	float h0 = quadrangle.y0 / glm::sqrt(dl * dl + quadrangle.y0sq);
	float h1 = quadrangle.y1 / glm::sqrt(dl * dl + quadrangle.y1sq);
	float hv = h0 + v * (h1 - h0);
	float hv2 = hv * hv;
	float yv = hv2 < 1 - 1e-6 ? (hv * dl) / glm::sqrt(1.0f - hv2) : quadrangle.y1;

	return xu * quadrangle.x + yv * quadrangle.y + quadrangle.z0 * quadrangle.z;	//未归一化
}
__device__ glm::vec3 sphericalRectangleSample(const FzbQuadrilateral& quadrangle, glm::vec3& hitPos, float u, float v, float& pdf) {
	float exl = glm::length(quadrangle.edge0);
	float eyl = glm::length(quadrangle.edge1);
	glm::vec3 axisX = quadrangle.edge0 / exl;
	glm::vec3 axisY = quadrangle.edge1 / eyl;
	glm::vec3 axisZ = quadrangle.normal;

	glm::vec3 d = quadrangle.startPos - hitPos;
	float z0 = glm::dot(d, axisZ);
	if (z0 > 0) {
		axisZ *= -1.0f;
		z0 *= -1.0f;
	}
	float z0sq = z0 * z0;

	float x0 = glm::dot(d, axisX);
	float y0 = glm::dot(d, axisY);
	float x1 = x0 + exl;
	float y1 = y0 + eyl;
	float y0sq = y0 * y0;
	float y1sq = y1 * y1;

	glm::vec3 v00 = glm::vec3(x0, y0, z0);
	glm::vec3 v01 = glm::vec3(x0, y1, z0);
	glm::vec3 v10 = glm::vec3(x1, y0, z0);
	glm::vec3 v11 = glm::vec3(x1, y1, z0);

	glm::vec3 n0 = glm::vec3(0.0f, z0, -y0);//glm::normalize(cross(v00, v10));
	n0.z /= glm::sqrt(z0sq + y0sq);	//后续只有z需要被用到，x，y会乘以0，不用管
	glm::vec3 n1 = glm::vec3(-z0, 0.0f, x1);//glm::normalize(cross(v10, v11));
	n1.z /= glm::sqrt(z0sq + x1 * x1);
	glm::vec3 n2 = glm::vec3(0.0f, -z0, y1);//glm::normalize(cross(v11, v01));
	n2.z /= glm::sqrt(z0sq + y1sq);
	glm::vec3 n3 = glm::vec3(z0, 0.0f, -x0);//glm::normalize(cross(v01, v00));
	n3.z /= glm::sqrt(z0sq + x0 * x0);

	float g0 = glm::acos(-n0.z * n1.z);	//glm::acos(-glm::dot(n0, n1));
	float g1 = glm::acos(-n1.z * n2.z);	//glm::acos(-glm::dot(n1, n2));
	float g2 = glm::acos(-n2.z * n3.z);	//glm::acos(-glm::dot(n2, n3));
	float g3 = glm::acos(-n3.z * n0.z);	//glm::acos(-glm::dot(n3, n0));

	float b0 = n0.z;
	float b1 = n2.z;
	float b0sq = b0 * b0;
	float k = 2 * PI - g2 - g3;
	float S = g0 + g1 - k;

	pdf = 1.0f / S;

	float au = u * S + k;
	float fu = (glm::cos(au) * b0 - b1) / glm::sin(au);
	float cu = 1.0f / glm::sqrt(fu * fu + b0sq) * (fu > 0.0f ? 1.0f : -1.0f);
	cu = glm::clamp(cu, -1.0f, 1.0f);
	float xu = -(cu * z0) / glm::sqrt(1.0f - cu * cu);
	xu = glm::clamp(xu, x0, x1);

	float dl = sqrt(xu * xu + z0sq);
	float h0 = y0 / glm::sqrt(dl * dl + y0sq);
	float h1 = y1 / glm::sqrt(dl * dl + y1sq);
	float hv = h0 + v * (h1 - h0);
	float hv2 = hv * hv;
	float yv = hv2 < 1 - 1e-6 ? (hv * dl) / glm::sqrt(1.0f - hv2) : y1;

	return xu * axisX + yv * axisY + z0 * axisZ;	//未归一化
}

/*
返回从光源到上一个撞击点的radiance
这里应该有优化，即不对所有光源进行采样，而是随机的方，但是目前先放着
对于NEE的碰撞检测，如果中间有电解质物体，是否应该进一步折射，这是一个问题，我觉得出于性能考虑，不要做，太慢了
*/
__device__ glm::vec3 NEE(FzbTriangleAttribute& triangleAttribute, FzbRay& ray, const FzbRayTracingLightSet* lightSet,
	const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, uint32_t& randomNumberSeed,
	bool useSphericalRectangleSample) {
	glm::vec3 radiance = glm::vec3(0.0f);
	FzbRay tempRay;
	FzbTriangleAttribute hitTriangleAttribute;
	for (int i = 0; i < lightSet->pointLightCount; ++i) {
		const FzbRayTracingPointLight& light = lightSet->pointLightInfoArray[i];
		glm::vec3 direction = light.worldPos - ray.hitPos;
		if (glm::dot(direction, triangleAttribute.normal) <= 0) continue;
		tempRay.depth = FLT_MAX;
		tempRay.direction = glm::normalize(direction);
		bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, tempRay, hitTriangleAttribute, false);
		if (!hit) continue;
		else if (abs(tempRay.depth - glm::length(direction)) > 0.1f) continue;
		float r2 = glm::length(direction); r2 *= r2;
		float cosTheta = glm::clamp(glm::dot(triangleAttribute.normal, tempRay.direction), 0.0f, 1.0f);
		radiance += cosTheta * light.radiantIntensity / r2 * getBSDF(triangleAttribute, tempRay.direction, -ray.direction, ray);
	}
	for (int i = 0; i < lightSet->areaLightCount; ++i) {
		const FzbRayTracingAreaLight& light = lightSet->areaLightInfoArray[i];
		float randomNumberX = rand(randomNumberSeed);
		float randomNumberY = rand(randomNumberSeed);
		glm::vec3 direction;
		float pdf = 1.0f / light.area;
		if (useSphericalRectangleSample) {
			FzbQuadrilateral quadrangle;
			quadrangle.startPos = light.worldPos;
			quadrangle.normal = light.normal;
			quadrangle.edge0 = light.edge0;
			quadrangle.edge1 = light.edge1;
			direction = sphericalRectangleSample(quadrangle, ray.hitPos, randomNumberX, randomNumberY, pdf);
		}
		else {
			glm::vec3 lightPos = glm::vec3(light.worldPos + randomNumberX * light.edge0 + randomNumberY * light.edge1);
			direction = lightPos - ray.hitPos;
		}
		if ((triangleAttribute.materialType != 2 && triangleAttribute.materialType != 3 && glm::dot(direction, triangleAttribute.normal) <= 0) ||
			glm::dot(-light.normal, direction) <= 0.0f) continue;
		tempRay.direction = glm::normalize(direction);
		tempRay.startPos = ray.hitPos + tempRay.direction * 0.001f;
		tempRay.depth = FLT_MAX;

		float r = glm::length(direction);
		bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, materialTextures, tempRay, hitTriangleAttribute, false);
		if (!hit) continue;
		else if (abs(tempRay.depth - r) > 0.001f) continue;
		glm::vec3 lightRadiance_cosTheta = light.radiance;
		if (!useSphericalRectangleSample) {
			lightRadiance_cosTheta *= glm::dot(-light.normal, tempRay.direction);	//微分单位从dw换为dA
			r = glm::max(r, 0.001f);
			lightRadiance_cosTheta /= r * r;
		}
		lightRadiance_cosTheta *= getBSDF(triangleAttribute, tempRay.direction, -ray.direction, tempRay);
		lightRadiance_cosTheta *= glm::clamp(glm::dot(triangleAttribute.normal, tempRay.direction), 0.0f, 1.0f);
		lightRadiance_cosTheta /= pdf;
		radiance += lightRadiance_cosTheta;
	}
	return radiance;
}

__device__ glm::vec3 getRadiance(FzbTriangleAttribute& triangleAttribute, FzbRay& ray, const FzbRayTracingLightSet* lightSet,
	const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, uint32_t& randomNumberSeed,
	bool useNEE, bool useSphericalRectangleSample) {
	glm::vec3 radiance = glm::vec3(0.0f);
	if(useNEE) radiance += NEE(triangleAttribute, ray, lightSet, vertices, materialTextures, bvhNodeArray, bvhTriangleInfoArray, randomNumberSeed, useSphericalRectangleSample);
	float cosine_ON = 1.0f;
	if (triangleAttribute.materialType != 2 && triangleAttribute.materialType != 3)
		cosine_ON = glm::max(glm::dot(triangleAttribute.normal, -ray.direction), 0.0f);
	radiance += triangleAttribute.emissive * cosine_ON;
	return radiance;
}

