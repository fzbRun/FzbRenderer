#include "./FzbGetTriangleAttribute.cuh"
#include "../../SceneDivision/BVH/CUDA/createBVH.cuh"

//-------------------------------------------------------------------------------------------------
__device__ glm::mat3 getTBN(const glm::vec3& edge0, const glm::vec3& edge1, const glm::vec2& uv_diff0, const glm::vec2& uv_diff1, const glm::vec3& normal, float& handed) {
	glm::vec3 tangent;
	glm::vec3 bitangent;
	float determinant = uv_diff0.x * uv_diff1.y - uv_diff0.y * uv_diff1.x;
	if (determinant < 1e-8f) {	//说明三点在一条直线上，那么任取一条直线，作为tangent
		/*
		glm::vec3 tmp = (fabs(normal.x) > 0.999f) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
		tangent = glm::normalize(glm::cross(tmp, normal));
		bitangent = glm::normalize(glm::cross(normal, tangent));
		*/
		tangent = glm::normalize(edge0);
		tangent = glm::normalize(tangent - glm::dot(tangent, normal) * normal);
		bitangent = glm::normalize(glm::cross(normal, tangent));
		handed = 1.0f;
	}
	else {
		float r = 1.0f / determinant;
		tangent = r * (uv_diff1.y * edge0 - uv_diff0.y * edge1);
		bitangent = r * (-uv_diff1.x * edge0 + uv_diff0.x * edge1);

		tangent = glm::normalize(tangent - normal * glm::dot(normal, tangent));
		handed = (glm::dot(glm::cross(normal, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;
		bitangent = glm::cross(normal, tangent) * handed;
	}
	return glm::mat3(tangent, bitangent, normal);
}
__device__ void getTriangleAttribute(const float* __restrict__ vertices, FzbBvhNodeTriangleInfo triangle, FzbTrianglePos& trianglePos, uint32_t& materialType) {
	materialType = materialInfoArray[triangle.materialIndex].materialType;

	int vertexStride = 3; // 位置总是有3个分量
	if (triangle.vertexFormat & 1) vertexStride += 3; // 法线
	if (triangle.vertexFormat & 2) vertexStride += 2; // 纹理坐标
	if (triangle.vertexFormat & 4) vertexStride += 4; // 切线
	uint32_t attributeStartIndex0 = triangle.indices0 * vertexStride;
	uint32_t attributeStartIndex1 = triangle.indices1 * vertexStride;
	uint32_t attributeStartIndex2 = triangle.indices2 * vertexStride;

	//获取顶点
	trianglePos.pos0 = glm::vec3(vertices[attributeStartIndex0], vertices[attributeStartIndex0 + 1], vertices[attributeStartIndex0 + 2]);
	trianglePos.pos1 = glm::vec3(vertices[attributeStartIndex1], vertices[attributeStartIndex1 + 1], vertices[attributeStartIndex1 + 2]);
	trianglePos.pos2 = glm::vec3(vertices[attributeStartIndex2], vertices[attributeStartIndex2 + 1], vertices[attributeStartIndex2 + 2]);
	//attributeStartIndex0 += 3;
	//attributeStartIndex1 += 3;
	//attributeStartIndex2 += 3;
	//glm::vec3 edge0 = triangleAttribute.pos1 - triangleAttribute.pos0;
	//glm::vec3 edge1 = triangleAttribute.pos2 - triangleAttribute.pos0;
	//triangleAttribute.normal = glm::normalize(glm::cross(edge0, edge1)); //直接使用面法线进行判断，快一点
	/*
	//获取法线
	if (triangle.vertexFormat & 1) {	//三角形顶点属性有法线
		glm::vec3 normal0 = glm::vec3(vertices[attributeStartIndex0], vertices[attributeStartIndex0 + 1], vertices[attributeStartIndex0 + 2]);
		glm::vec3 normal1 = glm::vec3(vertices[attributeStartIndex1], vertices[attributeStartIndex1 + 1], vertices[attributeStartIndex1 + 2]);
		glm::vec3 normal2 = glm::vec3(vertices[attributeStartIndex2], vertices[attributeStartIndex2 + 1], vertices[attributeStartIndex2 + 2]);
		attributeStartIndex0 += 3;
		attributeStartIndex1 += 3;
		attributeStartIndex2 += 3;
		triangleAttribute.normal = glm::normalize((normal0 + normal1 + normal2) / 3.0f);
	}
	else triangleAttribute.normal = glm::normalize(glm::cross(edge0, edge1)); //没有法线，那么使用面法线
	//获取texCoords，根据三个顶点的线性插值
	glm::vec2 texCoords0;
	glm::vec2 texCoords1;
	glm::vec2 texCoords2;
	if (triangle.vertexFormat & 2) {
		texCoords0 = glm::vec2(vertices[attributeStartIndex0], vertices[attributeStartIndex0 + 1]);
		texCoords1 = glm::vec2(vertices[attributeStartIndex1], vertices[attributeStartIndex1 + 1]);
		texCoords2 = glm::vec2(vertices[attributeStartIndex2], vertices[attributeStartIndex2 + 1]);
		triangleAttribute.texCoords = (texCoords0 + texCoords1 + texCoords2) / 3.0f;
		attributeStartIndex0 += 2;
		attributeStartIndex1 += 2;
		attributeStartIndex2 += 2;
	}
	//获取切线
	glm::vec3 tangent;
	float handed;
	if (triangle.vertexFormat & 4) {
		glm::vec3 tangent0 = glm::vec3(vertices[attributeStartIndex0], vertices[attributeStartIndex0 + 1], vertices[attributeStartIndex0 + 2]);
		glm::vec3 tangent1 = glm::vec3(vertices[attributeStartIndex1], vertices[attributeStartIndex1 + 1], vertices[attributeStartIndex0 + 2]);
		glm::vec3 tangent2 = glm::vec3(vertices[attributeStartIndex2], vertices[attributeStartIndex2 + 1], vertices[attributeStartIndex0 + 2]);
		float handed = vertices[attributeStartIndex0 + 3];
		tangent = glm::normalize((tangent0 + tangent1 + tangent2) / 3.0f);
	}

	if (material.textureIndex[0] > -1) {	//	//如果存在normalMap，则采样获取normal
		cudaTextureObject_t noramlTexture = materialTextures[material.textureIndex[1]];
		float4 textureNormal = tex2D<float4>(noramlTexture, triangleAttribute.texCoords.x, triangleAttribute.texCoords.y);

		//创建TBN，将textureNormal变换到worldSpace中
		glm::mat3 TBN;
		if (triangle.vertexFormat & 4) {
			tangent = normalize(tangent - triangleAttribute.normal * glm::dot(triangleAttribute.normal, tangent));
			glm::vec3 bitangent = glm::cross(triangleAttribute.normal, tangent) * handed;
			TBN = glm::mat3(tangent, bitangent, triangleAttribute.normal);
		}
		else {
			glm::vec2 uv_diff0 = texCoords1 - texCoords0;
			glm::vec2 uv_diff1 = texCoords2 - texCoords0;
			TBN = getTBN(edge0, edge1, uv_diff0, uv_diff1, triangleAttribute.normal);
		}
		triangleAttribute.normal = glm::normalize(TBN * (glm::vec3(textureNormal.x, textureNormal.y, textureNormal.z) * 2.0f - 1.0f));
	}
	*/
}
__device__ void getTriangleAttribute(const float* __restrict__ vertices,
	const cudaTextureObject_t* __restrict__ materialTextures,
	const FzbBvhNodeTriangleInfo& triangle, FzbTriangleAttribute& triangleAttribute, const FzbTrianglePos& trianglePos, const glm::vec3& hitPos) {
	FzbRayTracingMaterialUniformObject material = materialInfoArray[triangle.materialIndex];
	triangleAttribute.materialType = material.materialType;
	
	int vertexStride = 3; // 位置总是有3个分量
	if (triangle.vertexFormat & 1) vertexStride += 3; // 法线
	if (triangle.vertexFormat & 2) vertexStride += 2; // 纹理坐标
	if (triangle.vertexFormat & 4) vertexStride += 4; // 切线
	uint32_t attributeStartIndex0 = triangle.indices0 * vertexStride + 3;
	uint32_t attributeStartIndex1 = triangle.indices1 * vertexStride + 3;
	uint32_t attributeStartIndex2 = triangle.indices2 * vertexStride + 3;

	glm::vec3 edge0 = trianglePos.pos1 - trianglePos.pos0;
	glm::vec3 edge1 = trianglePos.pos2 - trianglePos.pos0;
	//获取法线
	if (triangle.vertexFormat & 1) {	//三角形顶点属性有法线
		glm::vec3 normal0 = glm::vec3(vertices[attributeStartIndex0], vertices[attributeStartIndex0 + 1], vertices[attributeStartIndex0 + 2]);
		glm::vec3 normal1 = glm::vec3(vertices[attributeStartIndex1], vertices[attributeStartIndex1 + 1], vertices[attributeStartIndex1 + 2]);
		glm::vec3 normal2 = glm::vec3(vertices[attributeStartIndex2], vertices[attributeStartIndex2 + 1], vertices[attributeStartIndex2 + 2]);
		attributeStartIndex0 += 3;
		attributeStartIndex1 += 3;
		attributeStartIndex2 += 3;
		triangleAttribute.normal = glm::normalize((normal0 + normal1 + normal2) / 3.0f);
	}
	else triangleAttribute.normal = glm::normalize(glm::cross(edge0, edge1)); //没有法线，那么使用面法线
	//获取texCoords，根据三个顶点的线性插值；不对，不应该这样，应该根据hitPos的位置来决定texCoord
	glm::vec2 texCoords0;
	glm::vec2 texCoords1;
	glm::vec2 texCoords2;
	glm::vec2 texCoords;
	if (triangle.vertexFormat & 2) {
		float weight0 = glm::length(hitPos - trianglePos.pos0);
		float weight1 = glm::length(hitPos - trianglePos.pos1);
		float weight2 = glm::length(hitPos - trianglePos.pos2);
		weight0 = 1.0f / (weight0 + 1e-5f);
		weight1 = 1.0f / (weight1 + 1e-5f);
		weight2 = 1.0f / (weight2 + 1e-5f);

		texCoords0 = glm::vec2(vertices[attributeStartIndex0], vertices[attributeStartIndex0 + 1]);
		texCoords1 = glm::vec2(vertices[attributeStartIndex1], vertices[attributeStartIndex1 + 1]);
		texCoords2 = glm::vec2(vertices[attributeStartIndex2], vertices[attributeStartIndex2 + 1]);
		texCoords = weight0 * texCoords0 + weight1 * texCoords1 + weight2 * texCoords2;//(texCoords0 + texCoords1 + texCoords2) / 3.0f;
		texCoords /= (weight0 + weight1 + weight2);
		//printf("%f %f\n", triangleAttribute.texCoords.x, triangleAttribute.texCoords.y);
		attributeStartIndex0 += 2;
		attributeStartIndex1 += 2;
		attributeStartIndex2 += 2;
	}
	//获取切线
	glm::vec3 tangent = glm::normalize(edge0);
	tangent = glm::normalize(tangent - glm::dot(tangent, triangleAttribute.normal) * triangleAttribute.normal);
	float handed = 1.0f;
	if (triangle.vertexFormat & 4) {
		glm::vec3 tangent0 = glm::vec3(vertices[attributeStartIndex0], vertices[attributeStartIndex0 + 1], vertices[attributeStartIndex0 + 2]);
		glm::vec3 tangent1 = glm::vec3(vertices[attributeStartIndex1], vertices[attributeStartIndex1 + 1], vertices[attributeStartIndex0 + 2]);
		glm::vec3 tangent2 = glm::vec3(vertices[attributeStartIndex2], vertices[attributeStartIndex2 + 1], vertices[attributeStartIndex0 + 2]);
		handed = vertices[attributeStartIndex0 + 3];
		tangent = glm::normalize((tangent0 + tangent1 + tangent2) / 3.0f);
	}

	if (material.textureIndex[0] > -1) {	//	//如果存在normalMap，则采样获取normal
		cudaTextureObject_t normalTexture = materialTextures[material.textureIndex[0]];
		float4 textureNormal = tex2D<float4>(normalTexture,texCoords.x, texCoords.y);

		//创建TBN，将textureNormal变换到worldSpace中
		glm::mat3 TBN;
		if (triangle.vertexFormat & 4) {
			tangent = normalize(tangent - triangleAttribute.normal * glm::dot(triangleAttribute.normal, tangent));
			glm::vec3 bitangent = glm::cross(triangleAttribute.normal, tangent) * handed;
			TBN = glm::mat3(tangent, bitangent, triangleAttribute.normal);
		}
		else {
			glm::vec2 uv_diff0 = texCoords1 - texCoords0;
			glm::vec2 uv_diff1 = texCoords2 - texCoords0;
			TBN = getTBN(edge0, edge1, uv_diff0, uv_diff1, triangleAttribute.normal, handed);
			tangent = TBN[0];
		}
		triangleAttribute.normal = glm::normalize(TBN * (glm::vec3(textureNormal.x, textureNormal.y, textureNormal.z) * 2.0f - 1.0f));
	}
	triangleAttribute.tangent = tangent;
	triangleAttribute.handed = handed;
	
	triangleAttribute.albedo = material.numberAttribute[0];	//获取albedo
	if (material.textureIndex[1] > -1) {	//有albedoMap
		cudaTextureObject_t albedoTexture = materialTextures[material.textureIndex[1]];
		float4 textureAlbedo = tex2D<float4>(albedoTexture, texCoords.x, texCoords.y);
		triangleAttribute.albedo *= glm::vec3(textureAlbedo.x, textureAlbedo.y, textureAlbedo.z);
	}
	if (material.materialType == 1) {
		triangleAttribute.albedo = material.numberAttribute[0];
		triangleAttribute.roughness = glm::clamp(material.numberAttribute[1].x, 0.1f, 1.0f);
	}
	else if (material.materialType == 2) {
		triangleAttribute.albedo = material.numberAttribute[0];
		triangleAttribute.roughness = glm::clamp(material.numberAttribute[1].x, 0.1f, 1.0f);
		triangleAttribute.eta = material.numberAttribute[1].y;
	}
	triangleAttribute.emissive = material.emissive;
}
