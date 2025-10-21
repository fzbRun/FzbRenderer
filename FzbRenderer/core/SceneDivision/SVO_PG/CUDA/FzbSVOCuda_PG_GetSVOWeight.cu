#include "FzbSVOCuda_PG.cuh"
#include "../../../RayTracing/CUDA/FzbCollisionDetection.cuh"

__global__ void getSVONodesWeight_device(
	float** SVONodeWeightsArray,
	FzbSVOIndivisibleNodeInfo* indivisibleNodeInfos,
	uint32_t SVONodeTotalCount, uint32_t SVOInDivisibleNodeTotalCount,
	FzbSVONodeData_PG** SVONodes, 
	FzbSVOLayerInfo* layerInfos, uint32_t maxSVOLayer,
	float* totalWeightArray,
	const float* __restrict__ vertices, const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, const uint32_t rayCount
) {
	__shared__ FzbSVOLayerInfo groupSVOLayerInfos[8];		//128^3ȥ���8��
	__shared__ FzbSVONodeData_PG* groupSVONodesArray[8];
	__shared__ float groupTotalWeight;
	__shared__ FzbSVONodeData_PG nodeData;

	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadIndex >= SVONodeTotalCount * SVOInDivisibleNodeTotalCount) return;

	uint32_t indivisibleNodeIndex = threadIndex / SVONodeTotalCount;
	int targetNodeIndex = threadIndex % SVONodeTotalCount;
	if (threadIdx.x < maxSVOLayer) {
		groupSVOLayerInfos[threadIdx.x] = layerInfos[threadIdx.x];
		groupSVONodesArray[threadIdx.x] = SVONodes[threadIdx.x];
	}
	if (threadIdx.x == 0) {
		FzbSVOIndivisibleNodeInfo nodeInfo = indivisibleNodeInfos[indivisibleNodeIndex];
		nodeData = groupSVONodesArray[nodeInfo.nodeLayer][nodeInfo.nodeIndex];
	}
	__syncthreads();

	//���targetNode��AABB������ǰnode��AABB�����Ǹ���node
	int targetNodeLayer = 1;
	if (targetNodeIndex >= 8) {
		targetNodeIndex -= 8;
		while (targetNodeIndex >= 0) {
			++targetNodeLayer;
			targetNodeIndex -= groupSVOLayerInfos[targetNodeLayer - 1].divisibleNodeCount * 8;
		}
		targetNodeIndex += groupSVOLayerInfos[targetNodeLayer - 1].divisibleNodeCount * 8;
	}

	FzbSVONodeData_PG targetNodeData = groupSVONodesArray[targetNodeLayer][targetNodeIndex];
	bool isFather =
		targetNodeData.AABB.leftX <= nodeData.AABB.leftX &&
		targetNodeData.AABB.leftY <= nodeData.AABB.leftY &&
		targetNodeData.AABB.leftZ <= nodeData.AABB.leftZ &&
		targetNodeData.AABB.rightX >= nodeData.AABB.rightX &&
		targetNodeData.AABB.rightY >= nodeData.AABB.rightY &&
		targetNodeData.AABB.rightZ >= nodeData.AABB.rightZ;
	if (isFather) {

	}
	else {
		for (int i = 0; i < rayCount; ++i) {
			
		}
	}

}

void FzbSVOCuda_PG::getSVONodesWeight() {

}

/*
Ϊÿ�����ɷ�node����weight����
weight�����СΪ��ֵnode�������������ɷ�node
*/
void FzbSVOCuda_PG::initGetSVONodesWeightSource() {
	//ÿ��node�����ָ��
	CHECK(cudaMalloc((void**)&this->SVONodes_multiLayer_Array, (this->SVONodes_maxDepth - 1) * sizeof(FzbSVONodeData_PG*)));
	CHECK(cudaMemcpy(this->SVONodes_multiLayer_Array, SVONodes_multiLayer.data() + 1, sizeof(FzbSVONodeData_PG*), cudaMemcpyHostToDevice));
	
	//���յ�weiht��ÿ�����ɷ�node��Ӧÿ��SVONode(������ֵ)֮���weight
	this->SVONodeWeights.resize(this->SVOInDivisibleNodeTotalCount_host);
	for (int i = 0; i < this->SVOInDivisibleNodeTotalCount_host; ++i) {
		CHECK(cudaMalloc((void**)&this->SVONodeWeights[i], this->SVONodeTotalCount_host * sizeof(float)));
	}

	CHECK(cudaMalloc((void**)&this->SVONodeWeightsArray, this->SVOInDivisibleNodeTotalCount_host * sizeof(float*)));
	CHECK(cudaMemcpy(this->SVONodeWeightsArray, this->SVONodeWeights.data(), this->SVOInDivisibleNodeTotalCount_host * sizeof(float*), cudaMemcpyHostToDevice));
		
	CHECK(cudaMalloc((void**)&this->SVONodeTotalWeightArray, this->SVOInDivisibleNodeTotalCount_host * sizeof(float)));
}