#include "./FzbSceneCUDA.cuh"
#include "../../../CUDA/commonCudaFunction.cuh"

void addData_uint(uint32_t* data, uint32_t date, uint32_t dataNum) {
	addDateCUDA_uint(data, date, dataNum);
}
void addPadding(std::vector<float>& sceneVertices, std::vector<float>& compressVertices, std::vector<uint32_t>& compressIndices, uint32_t& FzbVertexByteSize, FzbVertexFormat& vertexFormat) {
	uint32_t vertexSize = vertexFormat.getVertexSize();
	uint32_t verteBytexSize = vertexSize * sizeof(float);

	std::vector<float> padding;
	padding.reserve(vertexSize - 1);
	while (FzbVertexByteSize % verteBytexSize > 0) {
		padding.push_back(0.0f);
		FzbVertexByteSize += sizeof(float);
	}
	sceneVertices.insert(sceneVertices.end(), padding.begin(), padding.end());

	uint32_t indexNum = compressIndices.size();
	uint32_t offset = FzbVertexByteSize / verteBytexSize;
	if (FzbVertexByteSize > 0) {
		if (indexNum < 1024) {
			for (int i = 0; i < compressIndices.size(); i++) {
				compressIndices[i] += offset;
			}
		}
		else addDateCUDA_uint(compressIndices.data(), offset, indexNum);
	}

	FzbVertexByteSize += compressVertices.size() * sizeof(float);
}