#include "./FzbMeshCUDA.cuh"
#include "../../../CUDA/commonCudaFunction.cuh"

__global__ void createVertices(float* vertices, uint32_t vertexFormat, uint32_t vertexSize, uint32_t vertexNum,
	float* pos, float* normal, float* texCoords, float* tangent, float* biTangent_device) {
	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= vertexNum) return;
	bool useNormal = vertexFormat & 1u;
	bool useTexCoords = vertexFormat & 2u;
	bool useTangent = vertexFormat & 4u;

	uint32_t vertexIndex = threadIndex * vertexSize;
	uint32_t posIndex = threadIndex * 3;
	vertices[vertexIndex++] = pos[posIndex];
	vertices[vertexIndex++] = pos[posIndex + 1];
	vertices[vertexIndex++] = pos[posIndex + 2];

	glm::vec3 N;
	if (useNormal) {
		N = glm::vec3(normal[posIndex], normal[posIndex + 1], normal[posIndex + 2]);
		vertices[vertexIndex++] = N.x;
		vertices[vertexIndex++] = N.y;
		vertices[vertexIndex++] = N.z;
		
	}
	if (useTexCoords) {
		vertices[vertexIndex++] = texCoords[posIndex];
		vertices[vertexIndex++] = texCoords[posIndex + 1];
	}
	if (useTangent) {
		vertices[vertexIndex++] = tangent[posIndex];
		vertices[vertexIndex++] = tangent[posIndex + 1];
		vertices[vertexIndex++] = tangent[posIndex + 2];

		glm::vec3 T(tangent[posIndex], tangent[posIndex + 1], tangent[posIndex + 2]);
		glm::vec3 B(biTangent_device[posIndex], biTangent_device[posIndex + 1], biTangent_device[posIndex + 2]);
		float handed = (glm::dot(glm::cross(N, T), B) < 0.0f) ? -1.0f : 1.0f;
		vertices[vertexIndex++] = handed;
	}
}
FzbMesh createVertices_CUDA(aiMesh* meshData, FzbVertexFormat vertexFormat) {
	uint32_t vertexNum = meshData->mNumVertices;
	FzbMesh mesh = FzbMesh(vertexFormat);
	uint32_t vertexFormatU = mesh.vertexFormat.useNormal | (mesh.vertexFormat.useTexCoord << 1) | (mesh.vertexFormat.useTangent << 2);

	float* pos_device;
	CHECK(cudaMalloc((void**)&pos_device, sizeof(float) * 3 * vertexNum));
	CHECK(cudaMemcpy(pos_device, meshData->mVertices, sizeof(float) * 3 * vertexNum, cudaMemcpyHostToDevice));

	float* normal_device = nullptr;
	if (mesh.vertexFormat.useNormal) {
		CHECK(cudaMalloc((void**)&normal_device, sizeof(float) * 3 * vertexNum));
		CHECK(cudaMemcpy(normal_device, meshData->mNormals, sizeof(float) * 3 * vertexNum, cudaMemcpyHostToDevice));
	}

	float* texCoords_device = nullptr;
	if (mesh.vertexFormat.useTexCoord) {
		CHECK(cudaMalloc((void**)&texCoords_device, sizeof(float) * 3 * vertexNum));
		CHECK(cudaMemcpy(texCoords_device, meshData->mTextureCoords[0], sizeof(float) * 3 * vertexNum, cudaMemcpyHostToDevice));
	}

	float* tangent_device = nullptr;
	float* biTangent_device = nullptr;
	if (mesh.vertexFormat.useTangent) {
		CHECK(cudaMalloc((void**)&tangent_device, sizeof(float) * 3 * vertexNum));
		CHECK(cudaMemcpy(tangent_device, meshData->mTangents, sizeof(float) * 3 * vertexNum, cudaMemcpyHostToDevice));

		CHECK(cudaMalloc((void**)&biTangent_device, sizeof(float) * 3 * vertexNum));
		CHECK(cudaMemcpy(biTangent_device, meshData->mBitangents, sizeof(float) * 3 * vertexNum, cudaMemcpyHostToDevice));
	}

	uint32_t verticesFloatNum = mesh.vertexFormat.getVertexSize() * vertexNum;
	mesh.vertices.resize(verticesFloatNum);
	float* vertices_device;
	CHECK(cudaMalloc((void**)&vertices_device, sizeof(float) * verticesFloatNum));

	uint32_t gridSize = (vertexNum + 511) / 512;
	uint32_t blockSize = vertexNum > 512 ? 512 : vertexNum;
	createVertices<<<gridSize, blockSize>>>(vertices_device, vertexFormatU, vertexFormat.getVertexSize(), vertexNum, 
											pos_device, normal_device, texCoords_device, tangent_device, biTangent_device);
	CHECK(cudaMemcpy(mesh.vertices.data(), vertices_device, sizeof(float) * verticesFloatNum, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(vertices_device));
	CHECK(cudaFree(pos_device));
	if(normal_device) CHECK(cudaFree(normal_device));
	if(texCoords_device) CHECK(cudaFree(texCoords_device));
	if(tangent_device) CHECK(cudaFree(tangent_device));

	return mesh;
}

__global__ void verticesTransform_device(float* vertices, glm::mat4 transformMatrix, glm::mat3 normalTransformMatrix, uint32_t vertexFormat, uint32_t vertexSize, uint32_t vertexNum) {
	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= vertexNum) return;
	uint32_t startIndex = threadIndex * vertexSize;
	glm::vec4 pos = glm::vec4(vertices[startIndex], vertices[startIndex + 1], vertices[startIndex + 2], 1.0f);
	pos = transformMatrix * pos;
	vertices[startIndex] = pos.x;
	vertices[startIndex + 1] = pos.y;
	vertices[startIndex + 2] = pos.z;
	startIndex += 3;

	if (vertexFormat & 1) {
		glm::vec3 normal = glm::vec3(vertices[startIndex], vertices[startIndex + 1], vertices[startIndex + 2]);
		normal = normalTransformMatrix * normal;
		normal = glm::normalize(normal);
		vertices[startIndex] = normal.x;
		vertices[startIndex + 1] = normal.y;
		vertices[startIndex + 2] = normal.z;
		startIndex += 3;
	}
	if (vertexFormat & 2) startIndex += 2;
	if (vertexFormat & 4) {
		glm::vec3 tangent = glm::vec3(vertices[startIndex], vertices[startIndex + 1], vertices[startIndex + 2]);
		tangent = glm::normalize(glm::mat3(transformMatrix) * tangent);
		vertices[startIndex] = tangent.x;
		vertices[startIndex + 1] = tangent.y;
		vertices[startIndex + 2] = tangent.z;
	}
}
void verticesTransform_CUDA(std::vector<float>& vertices, glm::mat4 transformMatrix, FzbVertexFormat vertexFormat) {
	if (transformMatrix == glm::mat4(1.0f)) return;

	uint32_t vertexFormatU = vertexFormat.useNormal | (vertexFormat.useTexCoord << 1) | (vertexFormat.useTangent << 2);
	uint32_t vertexFloatNum = vertices.size();
	uint32_t vertexSize = vertexFormat.getVertexSize();
	uint32_t vertexNum = vertexFloatNum / vertexSize;

	glm::mat3 normalTransformMatrix = glm::inverse(glm::transpose(glm::mat3(transformMatrix)));

	if (vertexSize < 512) {
		for (int i = 0; i < vertexNum; ++i) {
			uint32_t vertexIndex = i * vertexSize;
			glm::vec4 pos = glm::vec4(vertices[vertexIndex], vertices[vertexIndex + 1], vertices[vertexIndex + 2], 1.0f);
			pos = transformMatrix * pos;
			vertices[vertexIndex++] = pos.x;
			vertices[vertexIndex++] = pos.y;
			vertices[vertexIndex++] = pos.z;

			if (vertexFormat.useNormal) {
				glm::vec3 normal = glm::vec3(vertices[vertexIndex], vertices[vertexIndex + 1], vertices[vertexIndex + 2]);
				normal = glm::normalize(normalTransformMatrix * normal);
				vertices[vertexIndex++] = normal.x;
				vertices[vertexIndex++] = normal.y;
				vertices[vertexIndex++] = normal.z;
			}
			if (vertexFormat.useTexCoord) vertexIndex += 2;
			if (vertexFormat.useTangent) {
				glm::vec3 tangent = glm::vec3(vertices[vertexIndex], vertices[vertexIndex + 1], vertices[vertexIndex + 2]);
				tangent = glm::normalize(glm::mat3(transformMatrix) * tangent);
				vertices[vertexIndex++] = tangent.x;
				vertices[vertexIndex++] = tangent.y;
				vertices[vertexIndex++] = tangent.z;
			}
		}
		return;
	}

	float* vertices_device;
	CHECK(cudaMalloc((void**)&vertices_device, sizeof(float) * vertexFloatNum));
	CHECK(cudaMemcpy(vertices_device, vertices.data(), sizeof(float) * vertexFloatNum, cudaMemcpyHostToDevice));

	uint32_t gridSize = (vertexNum + 511) / 512;
	uint32_t blockSize = vertexNum > 512 ? 512 : vertexNum;
	verticesTransform_device << <gridSize, blockSize >> > (vertices_device, transformMatrix, normalTransformMatrix, vertexFormatU, vertexSize, vertexNum);
	CHECK(cudaMemcpy(vertices.data(), vertices_device, sizeof(float) * vertexFloatNum, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(vertices_device));
}

__global__ void getVertices_device(float* vertices, uint32_t usability, uint32_t vertexSize, uint32_t usability_new, uint32_t vertexSize_new, float* result, uint32_t vertexNum) {
	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (threadIndex >= vertexNum) return;
	uint32_t floatIndex = vertexSize * threadIndex;;
	uint32_t floatIndex_new = vertexSize_new * threadIndex;;
	result[floatIndex_new++] = vertices[floatIndex++];
	result[floatIndex_new++] = vertices[floatIndex++];
	result[floatIndex_new++] = vertices[floatIndex++];

	if (usability_new & 1) {
		if (usability & 1) {
			result[floatIndex_new++] = vertices[floatIndex++];
			result[floatIndex_new++] = vertices[floatIndex++];
			result[floatIndex_new++] = vertices[floatIndex++];
		}
		else {
			result[floatIndex_new++] = 0.0f;
			result[floatIndex_new++] = 0.0f;
			result[floatIndex_new++] = 0.0f;
		}
	}
	else if (usability & 1) floatIndex += 3;
	if (usability_new & 2) {
		if (usability & 2) {
			result[floatIndex_new++] = vertices[floatIndex++];
			result[floatIndex_new++] = vertices[floatIndex++];
		}
		else {
			result[floatIndex_new++] = 0.0f;
			result[floatIndex_new++] = 0.0f;
		}
	}
	else if (usability & 2) floatIndex += 2;
	if (usability_new & 4) {
		if (usability & 4) {
			result[floatIndex_new++] = vertices[floatIndex++];
			result[floatIndex_new++] = vertices[floatIndex++];
			result[floatIndex_new++] = vertices[floatIndex++];
			result[floatIndex_new++] = vertices[floatIndex++];
		}
		else {
			result[floatIndex_new++] = 0.0f;
			result[floatIndex_new++] = 0.0f;
			result[floatIndex_new++] = 0.0f;
			result[floatIndex_new++] = 0.0f;
		}
	}
}
std::vector<float> getVertices_CUDA(std::vector<float>& vertices, FzbVertexFormat vertexFormat, FzbVertexFormat vertexFormat_new) {
	uint32_t vertexFormatU = vertexFormat.useNormal | (vertexFormat.useTexCoord << 1) | (vertexFormat.useTangent << 2);
	uint32_t vertexSize = vertexFormat.getVertexSize();

	uint32_t vertexFormatU_new = vertexFormat_new.useNormal | (vertexFormat_new.useTexCoord << 1) | (vertexFormat_new.useTangent << 2);
	uint32_t vertexSize_new = vertexFormat_new.getVertexSize();

	uint32_t vertexFloatNum = vertices.size();
	uint32_t vertexNum = vertexFloatNum / vertexSize;
	uint32_t vertexFloatNum_new = vertexNum * vertexSize_new;

	float* vertices_device;
	CHECK(cudaMalloc((void**)&vertices_device, sizeof(float) * vertexFloatNum));
	CHECK(cudaMemcpy(vertices_device, vertices.data(), sizeof(float) * vertexFloatNum, cudaMemcpyHostToDevice));

	float* result_device;
	CHECK(cudaMalloc((void**)&result_device, sizeof(float) * vertexFloatNum_new));

	uint32_t gridSize = (vertexNum + 511) / 512;
	uint32_t blockSize = vertexNum > 512 ? 512 : vertexNum;
	getVertices_device<<<gridSize, blockSize>>>(vertices_device, vertexFormatU, vertexSize, vertexFormatU_new, vertexSize_new, result_device, vertexNum);
	
	std::vector<float> result_host(vertexFloatNum_new);
	CHECK(cudaMemcpy(result_host.data(), result_device, sizeof(float) * vertexFloatNum_new, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(vertices_device));
	CHECK(cudaFree(result_device));

	return result_host;
}

struct AABB_CUDA {
	float leftX = FLT_MAX;
	float rightX = -FLT_MAX;
	float leftY = FLT_MAX;
	float rightY = -FLT_MAX;
	float leftZ = FLT_MAX;
	float rightZ = -FLT_MAX;
};
__global__ void createAABB_device(float* vertices, uint32_t vertexSize, AABB_CUDA* AABB, uint32_t vertexNum) {
	__shared__ AABB_CUDA groupAABB;

	uint32_t threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t activeMask = __ballot_sync(0xffffffff, threadIndex < vertexNum);
	if (threadIndex >= vertexNum) return;

	uint32_t warpLane = threadIndex & 31;
	
	uint32_t vertexIndex = threadIndex * vertexSize;
	glm::vec3 pos = glm::vec3(vertices[vertexIndex], vertices[vertexIndex + 1], vertices[vertexIndex + 2]);
	if (threadIdx.x == 0) {
		groupAABB.leftX = FLT_MAX;
		groupAABB.rightX = -FLT_MAX;
		groupAABB.leftY = FLT_MAX;
		groupAABB.rightY = -FLT_MAX;
		groupAABB.leftZ = FLT_MAX;
		groupAABB.rightZ = -FLT_MAX;
	}
	__syncthreads();

	if (activeMask == 0xffffffff) {
		float leftX = warpMin(pos.x);
		float rightX = warpMax(pos.x);
		float leftY = warpMin(pos.y);
		float rightY = warpMax(pos.y);
		float leftZ = warpMin(pos.z);
		float rightZ = warpMax(pos.z);

		if (warpLane == 0) {
			atomicMinFloat(&groupAABB.leftX, leftX);
			atomicMaxFloat(&groupAABB.rightX, rightX);
			atomicMinFloat(&groupAABB.leftY, leftY);
			atomicMaxFloat(&groupAABB.rightY, rightY);
			atomicMinFloat(&groupAABB.leftZ, leftZ);
			atomicMaxFloat(&groupAABB.rightZ, rightZ);
		}
	}
	else {
		atomicMinFloat(&groupAABB.leftX, pos.x);
		atomicMaxFloat(&groupAABB.rightX, pos.x);
		atomicMinFloat(&groupAABB.leftY, pos.y);
		atomicMaxFloat(&groupAABB.rightY, pos.y);
		atomicMinFloat(&groupAABB.leftZ, pos.z);
		atomicMaxFloat(&groupAABB.rightZ, pos.z);
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		atomicMinFloat(&AABB->leftX, groupAABB.leftX);
		atomicMaxFloat(&AABB->rightX, groupAABB.rightX);
		atomicMinFloat(&AABB->leftY, groupAABB.leftY);
		atomicMaxFloat(&AABB->rightY, groupAABB.rightY);
		atomicMinFloat(&AABB->leftZ, groupAABB.leftZ);
		atomicMaxFloat(&AABB->rightZ, groupAABB.rightZ);
	}
}
FzbAABBBox createAABB_CUDA(std::vector<float>& vertices, FzbVertexFormat vertexFormat) {
	uint32_t vertexFloatNum = vertices.size();
	uint32_t vertexSize = vertexFormat.getVertexSize();
	uint32_t vertexNum = vertexFloatNum / vertexSize;

	float* vertices_device;
	CHECK(cudaMalloc((void**)&vertices_device, sizeof(float) * vertexFloatNum));
	CHECK(cudaMemcpy(vertices_device, vertices.data(), sizeof(float) * vertexFloatNum, cudaMemcpyHostToDevice));

	AABB_CUDA* AABB_device;
	CHECK(cudaMalloc((void**)&AABB_device, sizeof(AABB_CUDA)));
	AABB_CUDA AABB_host;
	CHECK(cudaMemcpy(AABB_device, &AABB_host, sizeof(AABB_CUDA), cudaMemcpyHostToDevice));

	uint32_t gridSize = (vertexNum + 511) / 512;
	uint32_t blockSize = vertexNum > 512 ? 512 : vertexNum;
	createAABB_device<<<gridSize, blockSize>>>(vertices_device, vertexSize, AABB_device, vertexNum);
	CHECK(cudaMemcpy(&AABB_host, AABB_device, sizeof(AABB_CUDA), cudaMemcpyDeviceToHost));

	FzbAABBBox resultAABB;
	resultAABB.leftX = AABB_host.leftX;
	resultAABB.rightX = AABB_host.rightX;
	resultAABB.leftY = AABB_host.leftY;
	resultAABB.rightY = AABB_host.rightY;
	resultAABB.leftZ = AABB_host.leftZ;
	resultAABB.rightZ = AABB_host.rightZ;

	CHECK(cudaFree(vertices_device));
	CHECK(cudaFree(AABB_device));
	return resultAABB;
}