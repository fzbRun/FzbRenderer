#include "FzbSVOCuda_PG.cuh"
#include "../../../RayTracing/CUDA/FzbCollisionDetection.cuh"

__global__ void getSVONodesWeight_device(
	float* SVONodeWeights,
	const FzbSVOIndivisibleNodeInfo* __restrict__ indivisibleNodeInfos,
	uint32_t SVONodeCountInLayer, uint32_t SVONodeTotalCount, uint32_t SVOInDivisibleNodeTotalCount,
	FzbSVONodeData_PG** SVONodes,
	float* SVODivisibleNodeBlockWeight, uint32_t divisibleNodeOffset, uint32_t fatherDivisibleNodeOffset,
	uint32_t maxSVOLayer, uint32_t targetNodeLayer, uint32_t layerNodeOffset,
	const float* __restrict__ vertices, const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray,
	const uint32_t testCount, const uint32_t threadBlockCountForOneNode
) {
	__shared__ FzbSVONodeData_PG* groupSVONodesArray[8];
	__shared__ uint32_t groupRandomNumberSeed;

	__shared__ FzbSVOIndivisibleNodeInfo nodeInfo;
	__shared__ FzbSVONodeData_PG nodeData;

	uint32_t indivisibleNodeIndex = blockIdx.x / threadBlockCountForOneNode;
	int targetNodeIndex = (blockIdx.x % threadBlockCountForOneNode) * blockDim.x + threadIdx.x;
	if (targetNodeIndex >= SVONodeCountInLayer) return;	//���ﲻ�᷵����SVONodes�е��̣߳����Բ���Ժ���ϴ�����Ӱ��

	if (threadIdx.x < maxSVOLayer) groupSVONodesArray[threadIdx.x] = SVONodes[threadIdx.x]; //�߳���������Ϊ8����Ϊ��һ�����8��
	if (threadIdx.x == 0) {
		groupRandomNumberSeed = systemRandomNumberSeed;

		nodeInfo = indivisibleNodeInfos[indivisibleNodeIndex];
		nodeData = groupSVONodesArray[nodeInfo.nodeLayer][nodeInfo.nodeIndex];
	}
	__syncthreads();

	indivisibleNodeIndex = nodeData.label - 1;	//����SVO�еڼ������ɷ�node

	uint32_t hitCount = 0;
	FzbSVONodeData_PG targetNodeData = groupSVONodesArray[targetNodeLayer][targetNodeIndex];
	bool hasData = targetNodeData.irradiance.x != 0 || targetNodeData.irradiance.y != 0 || targetNodeData.irradiance.z != 0;
	float weight = 0.0f;
	if (hasData) {
		if (targetNodeData.indivisible == 0) {
			uint32_t weightIndex = indivisibleNodeIndex * SVONodeTotalCount + divisibleNodeOffset + targetNodeData.label - 1;
			weight = SVODivisibleNodeBlockWeight[weightIndex];
		}
		else if (!(targetNodeLayer == nodeInfo.nodeLayer && targetNodeIndex == nodeInfo.nodeIndex)) {
			bool isFather =
				targetNodeData.AABB.leftX <= nodeData.AABB.leftX &&
				targetNodeData.AABB.leftY <= nodeData.AABB.leftY &&
				targetNodeData.AABB.leftZ <= nodeData.AABB.leftZ &&
				targetNodeData.AABB.rightX >= nodeData.AABB.rightX &&
				targetNodeData.AABB.rightY >= nodeData.AABB.rightY &&
				targetNodeData.AABB.rightZ >= nodeData.AABB.rightZ;

			uint32_t randomNumberSeed = groupRandomNumberSeed + threadIdx.x + blockDim.x * blockIdx.x;
			float distanceX = nodeData.AABB.rightX - nodeData.AABB.leftX;
			float distanceY = nodeData.AABB.rightY - nodeData.AABB.leftY;
			float distanceZ = nodeData.AABB.rightZ - nodeData.AABB.leftZ;

			float targetDistanceX = targetNodeData.AABB.rightX - targetNodeData.AABB.leftX;
			float targetDistanceY = targetNodeData.AABB.rightY - targetNodeData.AABB.leftY;
			float targetDistanceZ = targetNodeData.AABB.rightZ - targetNodeData.AABB.leftZ;
			FzbRay ray;
			FzbTriangleAttribute triangleAttribute;
			if (isFather) {
				for (int i = 0; i < testCount; ++i) {
					ray.depth = FLT_MAX;

					uint32_t faceIndex = uint32_t(rand(randomNumberSeed) * 6);	//0��ʾleftX��1��ʾrightX����
					float randomU = rand(randomNumberSeed);		//��ǰnode��AABB�ϵ������
					float randomV = rand(randomNumberSeed);
					ray.startPos = glm::vec3(nodeData.AABB.leftX, nodeData.AABB.leftY, nodeData.AABB.leftZ);
					if (faceIndex & 4) {
						ray.startPos.z += (faceIndex & 1) * distanceZ;
						ray.startPos.x += randomU * distanceX;
						ray.startPos.y += randomV * distanceY;
					}
					else if (faceIndex & 2) {
						ray.startPos.y += (faceIndex & 1) * distanceY;
						ray.startPos.x += randomU * distanceX;
						ray.startPos.z += randomV * distanceZ;
					}
					else {
						ray.startPos.x += (faceIndex & 1) * distanceX;
						ray.startPos.z += randomU * distanceZ;
						ray.startPos.y += randomV * distanceY;
					}

					faceIndex = uint32_t(rand(randomNumberSeed) * 6);
					randomU = rand(randomNumberSeed);		//targetnode��AABB�ϵ������
					randomV = rand(randomNumberSeed);
					ray.direction = glm::vec3(targetNodeData.AABB.leftX, targetNodeData.AABB.leftY, targetNodeData.AABB.leftZ);
					if (faceIndex & 4) {
						ray.direction.z += (faceIndex & 1) * targetDistanceZ;
						ray.direction.x += randomU * targetDistanceX;
						ray.direction.y += randomV * targetDistanceY;
					}
					else if (faceIndex & 2) {
						ray.direction.y += (faceIndex & 1) * targetDistanceY;
						ray.direction.x += randomU * targetDistanceX;
						ray.direction.z += randomV * targetDistanceZ;
					}
					else {
						ray.direction.x += (faceIndex & 1) * targetDistanceX;
						ray.direction.z += randomU * targetDistanceZ;
						ray.direction.y += randomV * targetDistanceY;
					}

					ray.direction = ray.direction - ray.startPos;
					float r = glm::length(ray.direction);
					ray.direction = glm::normalize(ray.direction);

					bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, nullptr, ray, triangleAttribute, false);
					if (!hit) continue;
					if (ray.hitPos.x < targetNodeData.AABB.leftX || ray.hitPos.y < targetNodeData.AABB.leftY || ray.hitPos.z < targetNodeData.AABB.leftZ ||
						ray.hitPos.x > targetNodeData.AABB.rightX || ray.hitPos.y > targetNodeData.AABB.rightY || ray.hitPos.z > targetNodeData.AABB.rightZ) continue;
					++hitCount;
				}
			}
			else {
				glm::vec3 nodeCenterPos = glm::vec3(nodeData.AABB.leftX + nodeData.AABB.rightX, nodeData.AABB.leftY + nodeData.AABB.rightY, nodeData.AABB.leftZ + nodeData.AABB.rightZ) * 0.5f;
				glm::vec3 targetNodeCenterPos = glm::vec3(targetNodeData.AABB.leftX + targetNodeData.AABB.rightX, targetNodeData.AABB.leftY + targetNodeData.AABB.rightY, targetNodeData.AABB.leftZ + targetNodeData.AABB.rightZ) * 0.5f;
				glm::vec3 nodeDirection = targetNodeCenterPos - nodeCenterPos;

				for (int i = 0; i < testCount; ++i) {
					ray.depth = FLT_MAX;

					uint32_t faceIndex = uint32_t(rand(randomNumberSeed) * 6);	//0��ʾleftX��1��ʾrightX����
					float randomU = rand(randomNumberSeed);		//��ǰnode��AABB�ϵ������
					float randomV = rand(randomNumberSeed);
					ray.startPos = glm::vec3(nodeData.AABB.leftX, nodeData.AABB.leftY, nodeData.AABB.leftZ);
					if (faceIndex & 4) {
						ray.startPos.z += (faceIndex & 1) * distanceZ;
						ray.startPos.x += randomU * distanceX;
						ray.startPos.y += randomV * distanceY;
					}
					else if (faceIndex & 2) {
						ray.startPos.y += (faceIndex & 1) * distanceY;
						ray.startPos.x += randomU * distanceX;
						ray.startPos.z += randomV * distanceZ;
					}
					else {
						ray.startPos.x += (faceIndex & 1) * distanceX;
						ray.startPos.z += randomU * distanceZ;
						ray.startPos.y += randomV * distanceY;
					}

					faceIndex = uint32_t(rand(randomNumberSeed) * 3);
					randomU = rand(randomNumberSeed);		//targetnode��AABB�ϵ������
					randomV = rand(randomNumberSeed);
					ray.direction = glm::vec3(targetNodeData.AABB.leftX, targetNodeData.AABB.leftY, targetNodeData.AABB.leftZ);

					if (faceIndex & 2) {
						ray.direction.z += nodeDirection.z < 0 ? targetDistanceZ : 0.0f;	//�ں��
						ray.direction.x += randomU * targetDistanceX;
						ray.direction.y += randomV * targetDistanceY;
					}
					else if (faceIndex & 1) {
						ray.direction.y += nodeDirection.y < 0 ? targetDistanceY : 0.0f;	//���±�
						ray.direction.x += randomU * targetDistanceX;
						ray.direction.z += randomV * targetDistanceZ;
					}
					else {
						ray.direction.x += nodeDirection.x < 0 ? targetDistanceX : 0.0f;	//�����
						ray.direction.z += randomU * targetDistanceZ;
						ray.direction.y += randomV * targetDistanceY;
					}

					ray.direction = ray.direction - ray.startPos;
					float r = glm::length(ray.direction);
					ray.direction = glm::normalize(ray.direction);

					bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, nullptr, ray, triangleAttribute, false);
					if (!hit) continue;
					if (ray.hitPos.x < targetNodeData.AABB.leftX || ray.hitPos.y < targetNodeData.AABB.leftY || ray.hitPos.z < targetNodeData.AABB.leftZ ||
						ray.hitPos.x > targetNodeData.AABB.rightX || ray.hitPos.y > targetNodeData.AABB.rightY || ray.hitPos.z > targetNodeData.AABB.rightZ) continue;
					++hitCount;
				}
			}
			float occlusionRatio = (float)hitCount / testCount;
			weight = occlusionRatio * glm::length(targetNodeData.irradiance);
		}
	}

	uint32_t warpLane = threadIdx.x & 31;
	uint32_t firstBlockLane = (warpLane / 8) << 3;
	float blockWeightSum = weight;
	for (int offset = 4; offset > 0; offset /= 2)
		blockWeightSum += __shfl_down_sync(0xFFFFFFFF, blockWeightSum, offset);
	blockWeightSum = __shfl_sync(0xFFFFFFFF, blockWeightSum, firstBlockLane);

	if (weight > 0.0f) {
		uint32_t weightIndex = indivisibleNodeIndex * SVONodeTotalCount + layerNodeOffset + targetNodeIndex;
		SVONodeWeights[weightIndex] = weight / blockWeightSum;
	}
	if (warpLane == firstBlockLane && targetNodeLayer > 1) {
		uint32_t fatherNodeLabel = indivisibleNodeIndex * SVONodeTotalCount + fatherDivisibleNodeOffset + targetNodeIndex / 8;
		SVODivisibleNodeBlockWeight[fatherNodeLabel] = blockWeightSum;
	}
}
/*
__global__ void getSVONodesWeight_device_step2(
	float* SVONodeWeights, float* layerWeights,
	uint32_t SVONodeTotalCount, uint32_t SVOInDivisibleNodeTotalCount,
	FzbSVOLayerInfo* layerInfos, uint32_t maxSVOLayer
) {
	__shared__ FzbSVOLayerInfo groupSVOLayerInfos[8];
	uint32_t threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadIndex >= SVONodeTotalCount * SVOInDivisibleNodeTotalCount) return;

	if (blockIdx.x == gridDim.x - 1) {
		if(threadIdx.x == 0) for (int i = 0; i < maxSVOLayer; ++i) groupSVOLayerInfos[i] = layerInfos[i];
	}else if (threadIdx.x < maxSVOLayer) groupSVOLayerInfos[threadIdx.x] = layerInfos[threadIdx.x];
	uint32_t indivisibleNodeIndex = threadIndex / SVONodeTotalCount;
	int targetNodeIndex = threadIndex % SVONodeTotalCount;
	__syncthreads();

	int targetNodeLayer = 1;
	if (targetNodeIndex >= 8) {
		targetNodeIndex -= 8;
		while (targetNodeIndex >= 0) {
			++targetNodeLayer;
			targetNodeIndex -= groupSVOLayerInfos[targetNodeLayer - 1].divisibleNodeCount * 8;
		}
	}

	float weight = SVONodeWeights[threadIndex];
	uint32_t layerWeightIndex = indivisibleNodeIndex * maxSVOLayer + targetNodeLayer;
	float layerWeight = layerWeights[layerWeightIndex];
	SVONodeWeights[threadIndex] = layerWeight == 0.0f ? 0.0f : weight / layerWeight;
	//if (weight != 0.0f) printf("%f %f %f\n", weight, layerWeight, layerWeight == 0.0f ? 0.0f : weight / layerWeight);
}
*/

void FzbSVOCuda_PG::getSVONodesWeight() {
	uint32_t SVODivisibleNodeAccCount = 0;
	for (int i = 1; i < this->SVONodes_maxDepth - 1; ++i)	//Ҷ��divisibleNodeCount = 0
		SVODivisibleNodeAccCount += this->SVOLayerInfos_host[i].divisibleNodeCount;
	uint32_t layerDivisibleNodeOffset = SVODivisibleNodeAccCount;
	uint32_t layerNodeOffset = SVONodeTotalCount_host;
	for (int i = this->SVONodes_maxDepth - 1; i > 0; --i) {
		uint32_t layerNodeCount = SVOLayerInfos_host[i - 1].divisibleNodeCount * 8;		//��һ���node����
		uint32_t blockSize = layerNodeCount;	//�߳����СΪ�ò�node����
		uint32_t threadBlockCountForOneNode = (blockSize + 511) / 512;
		blockSize /= threadBlockCountForOneNode;
		while (((blockSize / threadBlockCountForOneNode) & 7) != 0) ++blockSize;
		uint32_t gridSize = SVOInDivisibleNodeTotalCount_host * threadBlockCountForOneNode;	//threadBlockCountForOneNode���߳������һ�����ɷ�node

		layerDivisibleNodeOffset -= this->SVOLayerInfos_host[i].divisibleNodeCount;		//�ò�Ŀɷ�node��ʼ����
		uint32_t fatherLayerDivisibleNodeOffset = layerDivisibleNodeOffset - this->SVOLayerInfos_host[i - 1].divisibleNodeCount;	//����ɷ�node��ʼ����
		layerNodeOffset -= layerNodeCount;		//�ò�node����ʼ����
		getSVONodesWeight_device<<<gridSize, blockSize, 0, stream>>>
		(
			SVONodeWeights, SVOIndivisibleNodeInfos,
			layerNodeCount, SVONodeTotalCount_host, SVOInDivisibleNodeTotalCount_host,
			SVONodes_multiLayer_Array,
			SVODivisibleNodeBlockWeight, layerDivisibleNodeOffset, fatherLayerDivisibleNodeOffset,
			SVONodes_maxDepth, i, layerNodeOffset,
			sourceManager->vertices, sourceManager->bvhNodeArray, sourceManager->bvhTriangleInfoArray,
			16, threadBlockCountForOneNode
		);
	}
	//uint32_t blockSize = SVONodeTotalCount_host;
	//uint32_t threadBlockCountForOneNode = (blockSize + 511) / 512;
	//blockSize /= threadBlockCountForOneNode;
	//while(((blockSize / threadBlockCountForOneNode) & 7) != 0) ++blockSize;
	//uint32_t gridSize = SVOInDivisibleNodeTotalCount_host * threadBlockCountForOneNode;
	//getSVONodesWeight_device <<<gridSize, blockSize, 0, stream>>>
	//(
	//	SVONodeWeights,
	//	SVOIndivisibleNodeInfos,
	//	SVONodeTotalCount_host, SVOInDivisibleNodeTotalCount_host,
	//	SVONodes_multiLayer_Array,
	//	SVOLayerInfos, SVONodes_maxDepth,
	//	sourceManager->vertices, sourceManager->bvhNodeArray, sourceManager->bvhTriangleInfoArray,
	//	16, threadBlockCountForOneNode
	//);
	//std::vector<float> layerWeights_host(1);
	//CHECK(cudaMemcpy(layerWeights_host.data(), layerWeights + 57, layerWeights_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
	//std::cout << layerWeights_host[0] << std::endl;
	//getSVONodesWeight_device_step2<<<gridSize, blockSize, 0, stream>>>
	//(
	//	SVONodeWeights, layerWeights,
	//	SVONodeTotalCount_host, SVOInDivisibleNodeTotalCount_host,
	//	SVOLayerInfos, SVONodes_maxDepth
	//);
}

/*
Ϊÿ�����ɷ�node����weight����
weight�����СΪ��ֵnode�������������ɷ�node
*/
void FzbSVOCuda_PG::initGetSVONodesWeightSource() {
	//ÿ��node�����ָ��
	CHECK(cudaMalloc((void**)&this->SVONodes_multiLayer_Array, this->SVONodes_maxDepth * sizeof(FzbSVONodeData_PG*)));
	CHECK(cudaMemcpy(this->SVONodes_multiLayer_Array, SVONodes_multiLayer.data(), this->SVONodes_maxDepth * sizeof(FzbSVONodeData_PG*), cudaMemcpyHostToDevice));

	CHECK(cudaMalloc((void**)&this->SVODivisibleNodeBlockWeight, SVOIndivisibleNodeMaxCount * SVONodeMaxCount * sizeof(float)));
	CHECK(cudaMemset(SVODivisibleNodeBlockWeight, 0, SVOIndivisibleNodeMaxCount * SVONodeMaxCount * sizeof(float)));

	//���յ�weiht��ÿ�����ɷ�node��Ӧÿ��SVONode(������ֵ)֮���weight
	CHECK(cudaMalloc((void**)&this->SVONodeWeights, SVOIndivisibleNodeMaxCount * SVONodeMaxCount * sizeof(float)));
	CHECK(cudaMemset(this->SVONodeWeights, 0, SVOIndivisibleNodeMaxCount * SVONodeMaxCount * sizeof(float)));
}