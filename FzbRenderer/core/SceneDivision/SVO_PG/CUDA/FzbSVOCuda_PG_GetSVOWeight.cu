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
	if (targetNodeIndex >= SVONodeCountInLayer) return;	//这里不会返回在SVONodes中的线程，所以不会对后续洗牌造成影响

	if (threadIdx.x < maxSVOLayer) groupSVONodesArray[threadIdx.x] = SVONodes[threadIdx.x]; //线程数量至少为8，因为第一层就有8个
	if (threadIdx.x == 0) {
		groupRandomNumberSeed = systemRandomNumberSeed;

		nodeInfo = indivisibleNodeInfos[indivisibleNodeIndex];
		nodeData = groupSVONodesArray[nodeInfo.nodeLayer][nodeInfo.nodeIndex];
	}
	__syncthreads();

	indivisibleNodeIndex = nodeData.label - 1;	//整棵SVO中第几个不可分node

	//uint32_t hitCount = 0;
	FzbSVONodeData_PG targetNodeData = groupSVONodesArray[targetNodeLayer][targetNodeIndex];
	bool hasData = targetNodeData.irradiance.x != 0 || targetNodeData.irradiance.y != 0 || targetNodeData.irradiance.z != 0;
	float weight = 0.0f;
	if (hasData) {
		if (targetNodeData.indivisible == 0) {		//fatherNode一定是可分node
			uint32_t weightIndex = indivisibleNodeIndex * SVONodeTotalCount + divisibleNodeOffset + targetNodeData.label - 1;
			weight = SVODivisibleNodeBlockWeight[weightIndex];
		}
		else if (!(targetNodeLayer == nodeInfo.nodeLayer && targetNodeIndex == nodeInfo.nodeIndex)) {
			glm::vec3 nodeCenterPos = glm::vec3(nodeData.AABB.leftX + nodeData.AABB.rightX, nodeData.AABB.leftY + nodeData.AABB.rightY, nodeData.AABB.leftZ + nodeData.AABB.rightZ) * 0.5f;
			glm::vec3 targetNodeCenterPos = glm::vec3(targetNodeData.AABB.leftX + targetNodeData.AABB.rightX, targetNodeData.AABB.leftY + targetNodeData.AABB.rightY, targetNodeData.AABB.leftZ + targetNodeData.AABB.rightZ) * 0.5f;
			glm::vec3 nodeDirection = glm::normalize(targetNodeCenterPos - nodeCenterPos);

			glm::vec3 face0Normal = glm::vec3(nodeDirection.x > 0 ? 1.0f : -1.0f, 0.0f, 0.0f);
			glm::vec3 face1Normal = glm::vec3(0.0f, nodeDirection.y > 0 ? 1.0f : -1.0f, 0.0f);
			glm::vec3 face2Normal = glm::vec3(0.0f, 0.0f, nodeDirection.z > 0 ? 1.0f : -1.0f);

			float cosThetaFace0 = glm::dot(nodeDirection, face0Normal);
			float cosThetaFace1 = glm::dot(nodeDirection, face1Normal);
			float cosThetaFace2 = glm::dot(nodeDirection, face2Normal);

			float distanceX = nodeData.AABB.rightX - nodeData.AABB.leftX;
			float distanceY = nodeData.AABB.rightY - nodeData.AABB.leftY;
			float distanceZ = nodeData.AABB.rightZ - nodeData.AABB.leftZ;

			glm::vec3 faceArea;		//这里应该是投影面，而不是直接的面积
			faceArea.x = distanceY * distanceZ * cosThetaFace0;
			faceArea.y = distanceX * distanceZ * cosThetaFace1;
			faceArea.z = distanceX * distanceY * cosThetaFace2;
			glm::vec3 faceSelectWeight = faceArea / (faceArea.x + faceArea.y + faceArea.z);

			float targetDistanceX = targetNodeData.AABB.rightX - targetNodeData.AABB.leftX;
			float targetDistanceY = targetNodeData.AABB.rightY - targetNodeData.AABB.leftY;
			float targetDistanceZ = targetNodeData.AABB.rightZ - targetNodeData.AABB.leftZ;

			glm::vec3 targetFaceArea;		//这里应该是投影面，而不是直接的面积
			targetFaceArea.x = targetDistanceY * targetDistanceZ * cosThetaFace0;
			targetFaceArea.y = targetDistanceX * targetDistanceZ * cosThetaFace1;
			targetFaceArea.z = targetDistanceX * targetDistanceY * cosThetaFace2;
			glm::vec3 targetFaceSelectWeight = targetFaceArea / (targetFaceArea.x + targetFaceArea.y + targetFaceArea.z);

			uint32_t randomNumberSeed = groupRandomNumberSeed + threadIdx.x + blockDim.x * blockIdx.x;
			FzbTriangleAttribute triangleAttribute;
			FzbRay ray;
			for (int i = 0; i < testCount; ++i) {
				float selectFaceProbability = rand(randomNumberSeed);
				ray.startPos = glm::vec3(nodeData.AABB.leftX, nodeData.AABB.leftY, nodeData.AABB.leftZ);
				glm::vec3 faceNormal = glm::vec3(0.0f);
				glm::vec2 randomUV = Hammersley(i, testCount);
				if (selectFaceProbability <= faceSelectWeight.x * 0.5f) {	//左面
					ray.startPos.z += randomUV.x * distanceZ;
					ray.startPos.y += randomUV.y * distanceY;

					faceNormal.x = -1.0f;
				}
				else if (selectFaceProbability <= faceSelectWeight.x) {	//右面
					ray.startPos.x += distanceX;
					ray.startPos.z += randomUV.x * distanceZ;
					ray.startPos.y += randomUV.y * distanceY;

					faceNormal.x = 1.0f;
				}
				else if (selectFaceProbability <= faceSelectWeight.x + faceSelectWeight.y * 0.5f) {	//下面
					ray.startPos.x += randomUV.x * distanceX;
					ray.startPos.z += randomUV.y * distanceZ;

					faceNormal.y = -1.0f;
				}
				else if(selectFaceProbability <= faceSelectWeight.x + faceSelectWeight.y){	//在上面
					ray.startPos.y += distanceY;	
					ray.startPos.x += randomUV.x * distanceX;
					ray.startPos.z += randomUV.y * distanceZ;

					faceNormal.y = 1.0f;
				}
				else if (selectFaceProbability <= faceSelectWeight.x + faceSelectWeight.y + faceSelectWeight.z * 0.5f) {	//后面
					ray.startPos.x += randomUV.x * distanceX;
					ray.startPos.y += randomUV.y * distanceY;

					faceNormal.z = -1.0f;
				}
				else {
					ray.startPos.z += nodeDirection.z > 0 ? distanceZ : 0.0f;	//前面
					ray.startPos.x += randomUV.x * distanceX;
					ray.startPos.y += randomUV.y * distanceY;

					faceNormal.z = 1.0f;
				}

				selectFaceProbability = rand(randomNumberSeed);
				randomUV = Hammersley(testCount - i - 1, testCount);
				ray.direction = glm::vec3(targetNodeData.AABB.leftX, targetNodeData.AABB.leftY, targetNodeData.AABB.leftZ);
				if (selectFaceProbability <= targetFaceSelectWeight.x) {
					ray.direction.x += nodeDirection.x < 0 ? targetDistanceX : 0.0f;	//在左边
					ray.direction.z += randomUV.x * targetDistanceZ;
					ray.direction.y += randomUV.y * targetDistanceY;
				}
				else if (selectFaceProbability <= 1.0f - targetFaceSelectWeight.z) {
					ray.direction.y += nodeDirection.y < 0 ? targetDistanceY : 0.0f;	//在下边
					ray.direction.x += randomUV.x * targetDistanceX;
					ray.direction.z += randomUV.y * targetDistanceZ;
				}
				else {
					ray.direction.z += nodeDirection.z < 0 ? targetDistanceZ : 0.0f;	//在后边
					ray.direction.x += randomUV.x * targetDistanceX;
					ray.direction.y += randomUV.y * targetDistanceY;
				}

				float r = glm::length(ray.direction - ray.startPos);
				ray.direction = glm::normalize(ray.direction - ray.startPos);
				ray.startPos += ray.direction * 0.001f;
				ray.depth = FLT_MAX;

				bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, nullptr, ray, triangleAttribute, false);
				if (!hit) continue;
				//if (ray.depth >= r) ++hitCount;
				if (ray.hitPos.x >= targetNodeData.AABB.leftX &&
					ray.hitPos.y >= targetNodeData.AABB.leftY &&
					ray.hitPos.z >= targetNodeData.AABB.leftZ &&
					ray.hitPos.x <= targetNodeData.AABB.rightX &&
					ray.hitPos.y <= targetNodeData.AABB.rightY &&
					ray.hitPos.z <= targetNodeData.AABB.rightZ) {
					//++hitCount;
					weight += max(dot(triangleAttribute.normal, -ray.direction), 0.0f) * max(dot(faceNormal, ray.direction), 0.0f);
					//可以进一步考虑折射的事情
				}
			}

			//float occlusionRatio = (float)hitCount / testCount;
			//weight = occlusionRatio * glm::length(targetNodeData.irradiance);
			weight *= glm::length(targetNodeData.irradiance) / testCount;
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

	//if (nodeInfo.nodeLayer == 3 && nodeInfo.nodeIndex == 180) {
	//	if (targetNodeLayer == 3 && targetNodeIndex >= 184 && targetNodeIndex <= 191) {
	//		printf("%f\n", weight / blockWeightSum);
	//	}
	//}
}

void FzbSVOCuda_PG::getSVONodesWeight() {
	uint32_t SVODivisibleNodeAccCount = 0;
	for (int i = 1; i < this->SVONodes_maxDepth - 1; ++i)	//叶层divisibleNodeCount = 0
		SVODivisibleNodeAccCount += this->SVOLayerInfos_host[i].divisibleNodeCount;
	uint32_t layerDivisibleNodeOffset = SVODivisibleNodeAccCount;
	uint32_t layerNodeOffset = SVONodeTotalCount_host;
	for (int i = this->SVONodes_maxDepth - 1; i > 0; --i) {
		if (SVOLayerInfos_host[i - 1].divisibleNodeCount == 0) continue;

		uint32_t layerNodeCount = SVOLayerInfos_host[i - 1].divisibleNodeCount * 8;		//这一层的node总数
		uint32_t blockSize = layerNodeCount;	//线程组大小为该层node总数
		uint32_t threadBlockCountForOneNode = (blockSize + 511) / 512;
		blockSize /= threadBlockCountForOneNode;
		while (((blockSize / threadBlockCountForOneNode) & 7) != 0) ++blockSize;
		uint32_t gridSize = SVOInDivisibleNodeTotalCount_host * threadBlockCountForOneNode;	//threadBlockCountForOneNode个线程组代表一个不可分node

		layerDivisibleNodeOffset -= this->SVOLayerInfos_host[i].divisibleNodeCount;		//该层的可分node起始索引
		uint32_t fatherLayerDivisibleNodeOffset = layerDivisibleNodeOffset - this->SVOLayerInfos_host[i - 1].divisibleNodeCount;	//父层可分node起始索引
		layerNodeOffset -= layerNodeCount;		//该层node的起始索引
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
		checkKernelFunction();
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
为每个不可分node创建weight数组
weight数组大小为有值node数量，即包括可分node
*/
void FzbSVOCuda_PG::initGetSVONodesWeightSource() {
	//每层node数组的指针
	CHECK(cudaMalloc((void**)&this->SVONodes_multiLayer_Array, this->SVONodes_maxDepth * sizeof(FzbSVONodeData_PG*)));
	CHECK(cudaMemcpy(this->SVONodes_multiLayer_Array, SVONodes_multiLayer.data(), this->SVONodes_maxDepth * sizeof(FzbSVONodeData_PG*), cudaMemcpyHostToDevice));

	CHECK(cudaMalloc((void**)&this->SVODivisibleNodeBlockWeight, SVOIndivisibleNodeMaxCount * SVONodeMaxCount * sizeof(float)));
	CHECK(cudaMemset(SVODivisibleNodeBlockWeight, 0, SVOIndivisibleNodeMaxCount * SVONodeMaxCount * sizeof(float)));

	//最终的weiht，每个不可分node对应每个SVONode(包括无值)之间的weight
	CHECK(cudaMalloc((void**)&this->SVONodeWeights, SVOIndivisibleNodeMaxCount * SVONodeMaxCount * sizeof(float)));
	CHECK(cudaMemset(this->SVONodeWeights, 0, SVOIndivisibleNodeMaxCount * SVONodeMaxCount * sizeof(float)));
}