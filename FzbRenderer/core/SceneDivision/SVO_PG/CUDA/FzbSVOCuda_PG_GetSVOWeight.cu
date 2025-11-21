#include "FzbSVOCuda_PG.cuh"
#include "../../../RayTracing/CUDA/FzbCollisionDetection.cuh"

const uint32_t testRayCount = 16;

/*
__global__ void getSVONodesWeight_device(
	float* SVONodeWeights,
	const FzbSVOIndivisibleNodeInfo* __restrict__ indivisibleNodeInfos,
	uint32_t SVONodeCountInLayer, uint32_t SVONodeTotalCount, uint32_t SVOInDivisibleNodeTotalCount,
	FzbSVONodeData_PG** SVONodes,
	float* SVODivisibleNodeBlockWeight, uint32_t divisibleNodeOffset, uint32_t fatherDivisibleNodeOffset,
	uint32_t maxSVOLayer, uint32_t targetNodeLayer, uint32_t layerNodeOffset,
	const float* __restrict__ vertices, const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray,
	const uint32_t threadBlockCountForOneNode
) {
	__shared__ FzbSVONodeData_PG* groupSVONodesArray[8];
	__shared__ uint32_t groupRandomNumberSeed;

	__shared__ FzbSVOIndivisibleNodeInfo nodeInfo;
	__shared__ FzbSVONodeData_PG nodeData;

	__shared__ glm::vec3 normalizeNormal_G;
	__shared__ glm::mat3 TBN;
	__shared__ glm::vec3 groupSamplePoses[testRayCount];

	uint32_t indivisibleNodeIndex = blockIdx.x / threadBlockCountForOneNode;
	int targetNodeIndex = (blockIdx.x % threadBlockCountForOneNode) * blockDim.x + threadIdx.x;
	if (targetNodeIndex >= SVONodeCountInLayer) return;	//这里不会返回在SVONodes中的线程，所以不会对后续洗牌造成影响

	if (threadIdx.x < maxSVOLayer) groupSVONodesArray[threadIdx.x] = SVONodes[threadIdx.x]; //线程数量至少为8，因为第一层就有8个
	if (threadIdx.x == 0) {
		groupRandomNumberSeed = systemRandomNumberSeed;

		nodeInfo = indivisibleNodeInfos[indivisibleNodeIndex];
		nodeData = groupSVONodesArray[nodeInfo.nodeLayer][nodeInfo.nodeIndex];

		normalizeNormal_G = glm::normalize(nodeData.meanNormal_G);
		glm::vec3 tangent = glm::vec3(1.0f, 0.0f, 0.0f);
		glm::vec3 bitangent = glm::cross(normalizeNormal_G, tangent);
		tangent = glm::cross(bitangent, normalizeNormal_G);
		TBN = glm::mat3(tangent, bitangent, normalizeNormal_G);
	}
	__syncthreads();
	//if (nodeData.meanNormal_G == glm::vec3(0.0f)) return;
	glm::vec3 nodeCenterPos = glm::vec3(nodeData.AABB_G.leftX + nodeData.AABB_G.rightX, nodeData.AABB_G.leftY + nodeData.AABB_G.rightY, nodeData.AABB_G.leftZ + nodeData.AABB_G.rightZ) * 0.5f;
	if (threadIdx.x < testRayCount) {
		glm::vec2 randomUV = Hammersley(threadIdx.x, testRayCount);

		float phi = randomUV.y * 2 * PI;
		float cosTheta = randomUV.x;
		float sinTheta = glm::sqrt(1 - cosTheta * cosTheta);
		float x = sinTheta * glm::cos(phi);
		float y = sinTheta * glm::sin(phi);
		float z = cosTheta;
		glm::vec3 sampleDirection = glm::normalize(TBN * glm::vec3(x, y, z));

		float nodeSize = glm::length(
			glm::vec3(
				nodeData.AABB_G.rightX - nodeData.AABB_G.leftX,
				nodeData.AABB_G.rightY - nodeData.AABB_G.leftY,
				nodeData.AABB_G.rightZ - nodeData.AABB_G.leftZ
			)
		);
		groupSamplePoses[threadIdx.x] = nodeCenterPos + sampleDirection * nodeSize;
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
			glm::vec3 targetNodeCenterPos = glm::vec3(targetNodeData.AABB_E.leftX + targetNodeData.AABB_E.rightX, targetNodeData.AABB_E.leftY + targetNodeData.AABB_E.rightY, targetNodeData.AABB_E.leftZ + targetNodeData.AABB_E.rightZ) * 0.5f;
			glm::vec3 nodeDirection = glm::normalize(targetNodeCenterPos - nodeCenterPos);

			glm::vec3 face0Normal = glm::vec3(nodeDirection.x > 0 ? 1.0f : -1.0f, 0.0f, 0.0f);
			glm::vec3 face1Normal = glm::vec3(0.0f, nodeDirection.y > 0 ? 1.0f : -1.0f, 0.0f);
			glm::vec3 face2Normal = glm::vec3(0.0f, 0.0f, nodeDirection.z > 0 ? 1.0f : -1.0f);

			float cosThetaFace0 = glm::dot(nodeDirection, face0Normal);
			float cosThetaFace1 = glm::dot(nodeDirection, face1Normal);
			float cosThetaFace2 = glm::dot(nodeDirection, face2Normal);

			float targetDistanceX = targetNodeData.AABB_E.rightX - targetNodeData.AABB_E.leftX;
			float targetDistanceY = targetNodeData.AABB_E.rightY - targetNodeData.AABB_E.leftY;
			float targetDistanceZ = targetNodeData.AABB_E.rightZ - targetNodeData.AABB_E.leftZ;

			glm::vec3 targetFaceArea;		//这里应该是投影面，而不是直接的面积
			targetFaceArea.x = targetDistanceY * targetDistanceZ * cosThetaFace0;
			targetFaceArea.y = targetDistanceX * targetDistanceZ * cosThetaFace1;
			targetFaceArea.z = targetDistanceX * targetDistanceY * cosThetaFace2;
			glm::vec3 targetFaceSelectWeight = targetFaceArea / (targetFaceArea.x + targetFaceArea.y + targetFaceArea.z);

			uint32_t randomNumberSeed = groupRandomNumberSeed + threadIdx.x + blockDim.x * blockIdx.x;
			FzbTriangleAttribute triangleAttribute;
			FzbRay ray;
			for (int i = 0; i < testRayCount; ++i) {
				ray.startPos = groupSamplePoses[i];

				float selectFaceProbability = rand(randomNumberSeed);
				glm::vec2 randomUV = Hammersley(i, testRayCount);
				ray.direction = glm::vec3(targetNodeData.AABB_E.leftX, targetNodeData.AABB_E.leftY, targetNodeData.AABB_E.leftZ);
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
				ray.startPos -= ray.direction * 0.001f;
				ray.depth = FLT_MAX;

				bool hit = sceneCollisionDetection(bvhNodeArray, bvhTriangleInfoArray, vertices, nullptr, ray, triangleAttribute, false);
				//if (!hit) continue;
				if (ray.depth >= r) {	//没打中三角形也算
					//++hitCount;
					//glm::vec3 normalizeNormal_G = nodeData.meanNormal_G == glm::vec3(0.0f) ? glm::vec3(0.0f) : glm::normalize(nodeData.meanNormal_G);
					//glm::vec3 normalizeNormal_E = targetNodeData.meanNormal_E == glm::vec3(0.0f) ? glm::vec3(0.0f) : glm::normalize(targetNodeData.meanNormal_E);
					float cosine_NE_L = dot(triangleAttribute.normal, -ray.direction);
					if (triangleAttribute.materialType == 2) cosine_NE_L = abs(cosine_NE_L);
					
					float rsq = max(r * r, 0.001f);
					weight += max(cosine_NE_L, 0.0f) / rsq;	// *max(dot(normalizeNormal_G, ray.direction), 0.0f);
				} 
				//if (ray.hitPos.x >= targetNodeData.AABB.leftX &&
				//	ray.hitPos.y >= targetNodeData.AABB.leftY &&
				//	ray.hitPos.z >= targetNodeData.AABB.leftZ &&
				//	ray.hitPos.x <= targetNodeData.AABB.rightX &&
				//	ray.hitPos.y <= targetNodeData.AABB.rightY &&
				//	ray.hitPos.z <= targetNodeData.AABB.rightZ) {
				//	//++hitCount;
				//	weight += max(dot(targetNodeData.normal, -ray.direction), 0.0f);
				//	//可以进一步考虑折射的事情
				//}
			}

			//float occlusionRatio = (float)hitCount / testCount;
			//weight = occlusionRatio * glm::length(targetNodeData.irradiance);
			weight *= glm::length(targetNodeData.irradiance) / testRayCount;
		}
	}

	uint32_t warpLane = threadIdx.x & 31;
	uint32_t firstBlockLane = (warpLane / 8) << 3;
	float blockWeightSum = weight;
	for (int offset = 4; offset > 0; offset /= 2)
		blockWeightSum += __shfl_down_sync(0xFFFFFFFF, blockWeightSum, offset);
	blockWeightSum = __shfl_sync(0xFFFFFFFF, blockWeightSum, firstBlockLane);

	if (blockWeightSum > 0.0f) {
		//weight += blockWeightSum * 0.0125f;
		//blockWeightSum *= 1.1f;
		uint32_t weightIndex = indivisibleNodeIndex * SVONodeTotalCount + layerNodeOffset + targetNodeIndex;
		SVONodeWeights[weightIndex] = weight / blockWeightSum;

		if (warpLane == firstBlockLane && targetNodeLayer > 1) {
			uint32_t fatherNodeLabel = indivisibleNodeIndex * SVONodeTotalCount + fatherDivisibleNodeOffset + targetNodeIndex / 8;
			SVODivisibleNodeBlockWeight[fatherNodeLabel] = blockWeightSum;
		}
	}
}
*/
__global__ void getSVONodesWeight_device(
	float* SVONodeWeights,
	const FzbSVOIndivisibleNodeInfo* __restrict__ indivisibleNodeInfos_G,
	uint32_t SVONodeCountInLayer_E, uint32_t SVONode_E_TotalCount,
	FzbSVONodeData_PG_G** SVONodes_G, FzbSVONodeData_PG_E** SVONodes_E,
	float* SVODivisibleNodeBlockWeight, float* SVOFatherDivisibleNodeBlockWeight,
	uint32_t layerDivisibleNodeCount_E, uint32_t fatherLayerDivisibleNodeCount_E,
	uint32_t maxSVOLayer, uint32_t targetNodeLayer_E, uint32_t layerNodeOffset_E,
	const float* __restrict__ vertices, const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray,
	const uint32_t threadBlockCountForOneNode
) {
	__shared__ FzbSVONodeData_PG_E* groupSVONodes_E[8];
	__shared__ uint32_t groupRandomNumberSeed;
	__shared__ float groupEntropyThreshold;

	__shared__ FzbSVOIndivisibleNodeInfo nodeInfo_G;
	__shared__ FzbSVONodeData_PG_G nodeData_G;

	uint32_t indivisibleNodeIndex = blockIdx.x / threadBlockCountForOneNode;
	int targetNodeIndex_E = (blockIdx.x % threadBlockCountForOneNode) * blockDim.x + threadIdx.x;
	if (targetNodeIndex_E >= SVONodeCountInLayer_E) return;	//这里不会返回在SVONodes中的线程，所以不会对后续洗牌造成影响

	if (threadIdx.x < maxSVOLayer) groupSVONodes_E[threadIdx.x] = SVONodes_E[threadIdx.x];

	if (threadIdx.x == 0) {
		groupRandomNumberSeed = systemRandomNumberSeed;
		groupEntropyThreshold = systemSVOUniformData.entropyThreshold;
		nodeInfo_G = indivisibleNodeInfos_G[indivisibleNodeIndex];
		nodeData_G = SVONodes_G[nodeInfo_G.nodeLayer][nodeInfo_G.nodeIndex];
	}
	__syncthreads();

	bool useG = nodeData_G.entropy < groupEntropyThreshold&& nodeData_G.meanNormal != glm::vec3(0.0f);
	glm::vec3 normalizeNormal;
	//glm::mat3 nodeData_G_TBN;
	if (useG) {
		normalizeNormal = glm::normalize(nodeData_G.meanNormal);
		//glm::vec3 bitangent = glm::cross(normalizeNormal, glm::vec3(1.0f, 0.0f, 0.0f));
		//glm::vec3 tangent = glm::cross(bitangent, normalizeNormal);
		//nodeData_G_TBN = glm::mat3(tangent, bitangent, normalizeNormal);
	}

	indivisibleNodeIndex = nodeData_G.label - 1;	//整棵SVO中第几个不可分node

	FzbSVONodeData_PG_E targetNodeData_E = groupSVONodes_E[targetNodeLayer_E][targetNodeIndex_E];
	bool hasData = targetNodeData_E.irradiance.x != 0 || targetNodeData_E.irradiance.y != 0 || targetNodeData_E.irradiance.z != 0;
	float weight = 0.0f;
	if (hasData) {
		if (targetNodeData_E.indivisible == 0) {		//fatherNode一定是可分node
			uint32_t weightIndex = indivisibleNodeIndex * layerDivisibleNodeCount_E + targetNodeData_E.label - 1;
			weight = SVODivisibleNodeBlockWeight[weightIndex];	//子节点已经将weightSum存入
		}
		else {
			glm::vec3 nodeCenterPos = glm::vec3(nodeData_G.AABB.leftX + nodeData_G.AABB.rightX,
				nodeData_G.AABB.leftY + nodeData_G.AABB.rightY,
				nodeData_G.AABB.leftZ + nodeData_G.AABB.rightZ) * 0.5f;
			glm::vec3 targetNodeCenterPos = glm::vec3(targetNodeData_E.AABB.leftX + targetNodeData_E.AABB.rightX, 
				targetNodeData_E.AABB.leftY + targetNodeData_E.AABB.rightY,
				targetNodeData_E.AABB.leftZ + targetNodeData_E.AABB.rightZ) * 0.5f;
			glm::vec3 nodeDirection = glm::normalize(targetNodeCenterPos - nodeCenterPos);

			glm::vec3 face0Normal = glm::vec3(nodeDirection.x > 0 ? 1.0f : -1.0f, 0.0f, 0.0f);
			glm::vec3 face1Normal = glm::vec3(0.0f, nodeDirection.y > 0 ? 1.0f : -1.0f, 0.0f);
			glm::vec3 face2Normal = glm::vec3(0.0f, 0.0f, nodeDirection.z > 0 ? 1.0f : -1.0f);

			float cosThetaFace0 = glm::dot(nodeDirection, face0Normal);
			float cosThetaFace1 = glm::dot(nodeDirection, face1Normal);
			float cosThetaFace2 = glm::dot(nodeDirection, face2Normal);

			float distanceX = nodeData_G.AABB.rightX - nodeData_G.AABB.leftX;
			float distanceY = nodeData_G.AABB.rightY - nodeData_G.AABB.leftY;
			float distanceZ = nodeData_G.AABB.rightZ - nodeData_G.AABB.leftZ;

			glm::vec3 faceArea;		//这里应该是投影面，而不是直接的面积
			faceArea.x = distanceY * distanceZ * cosThetaFace0;
			faceArea.y = distanceX * distanceZ * cosThetaFace1;
			faceArea.z = distanceX * distanceY * cosThetaFace2;
			glm::vec3 faceSelectWeight = faceArea / (faceArea.x + faceArea.y + faceArea.z);

			float targetDistanceX = targetNodeData_E.AABB.rightX - targetNodeData_E.AABB.leftX;
			float targetDistanceY = targetNodeData_E.AABB.rightY - targetNodeData_E.AABB.leftY;
			float targetDistanceZ = targetNodeData_E.AABB.rightZ - targetNodeData_E.AABB.leftZ;

			glm::vec3 targetFaceArea;		//这里应该是投影面，而不是直接的面积
			targetFaceArea.x = targetDistanceY * targetDistanceZ * cosThetaFace0;
			targetFaceArea.y = targetDistanceX * targetDistanceZ * cosThetaFace1;
			targetFaceArea.z = targetDistanceX * targetDistanceY * cosThetaFace2;
			//if (useG) targetFaceArea *= abs(normalizeNormal);
			glm::vec3 targetFaceSelectWeight = targetFaceArea / (targetFaceArea.x + targetFaceArea.y + targetFaceArea.z);

			uint32_t randomNumberSeed = groupRandomNumberSeed + threadIdx.x + blockDim.x * blockIdx.x;
			FzbTriangleAttribute triangleAttribute;
			FzbRay ray;

			for (int i = 0; i < testRayCount; ++i) {
				float selectFaceProbability = rand(randomNumberSeed);
				ray.startPos = glm::vec3(nodeData_G.AABB.leftX, nodeData_G.AABB.leftY, nodeData_G.AABB.leftZ);
				glm::vec2 randomUV = Hammersley(i, testRayCount);
				if (useG) {
					//float phi = randomUV.y * 2 * PI;
					//float cosTheta = randomUV.x;
					//float sinTheta = glm::sqrt(1 - cosTheta * cosTheta);
					//float x = sinTheta * glm::cos(phi);
					//float y = sinTheta * glm::sin(phi);
					//float z = cosTheta;
					//ray.startPos = nodeCenterPos + glm::normalize(nodeData_G_TBN * glm::vec3(x, y, z));
					if (selectFaceProbability <= faceSelectWeight.x) {
						if (normalizeNormal.x < 0) {
							ray.startPos.z += randomUV.x * distanceZ;
							ray.startPos.y += randomUV.y * distanceY;
						}
						else {
							ray.startPos.x += distanceX;
							ray.startPos.z += randomUV.x * distanceZ;
							ray.startPos.y += randomUV.y * distanceY;
						}
					}
					else if (selectFaceProbability <= faceSelectWeight.x + faceSelectWeight.y) {
						if (normalizeNormal.y < 0) {
							ray.startPos.x += randomUV.x * distanceX;
							ray.startPos.z += randomUV.y * distanceZ;
						}
						else {
							ray.startPos.y += distanceY;
							ray.startPos.x += randomUV.x * distanceX;
							ray.startPos.z += randomUV.y * distanceZ;
						}
					}
					else {
						if (normalizeNormal.z < 0) {
							ray.startPos.x += randomUV.x * distanceX;
							ray.startPos.y += randomUV.y * distanceY;
						}
						else {
							ray.startPos.z += nodeDirection.z > 0 ? distanceZ : 0.0f;	//前面
							ray.startPos.x += randomUV.x * distanceX;
							ray.startPos.y += randomUV.y * distanceY;
						}
					}
				}
				else {
					if (selectFaceProbability <= faceSelectWeight.x * 0.5f) {	//左面
						ray.startPos.z += randomUV.x * distanceZ;
						ray.startPos.y += randomUV.y * distanceY;
					}
					else if (selectFaceProbability <= faceSelectWeight.x) {	//右面
						ray.startPos.x += distanceX;
						ray.startPos.z += randomUV.x * distanceZ;
						ray.startPos.y += randomUV.y * distanceY;
					}
					else if (selectFaceProbability <= faceSelectWeight.x + faceSelectWeight.y * 0.5f) {	//下面
						ray.startPos.x += randomUV.x * distanceX;
						ray.startPos.z += randomUV.y * distanceZ;
					}
					else if (selectFaceProbability <= faceSelectWeight.x + faceSelectWeight.y) {	//在上面
						ray.startPos.y += distanceY;
						ray.startPos.x += randomUV.x * distanceX;
						ray.startPos.z += randomUV.y * distanceZ;
					}
					else if (selectFaceProbability <= faceSelectWeight.x + faceSelectWeight.y + faceSelectWeight.z * 0.5f) {	//后面
						ray.startPos.x += randomUV.x * distanceX;
						ray.startPos.y += randomUV.y * distanceY;
					}
					else {
						ray.startPos.z += nodeDirection.z > 0 ? distanceZ : 0.0f;	//前面
						ray.startPos.x += randomUV.x * distanceX;
						ray.startPos.y += randomUV.y * distanceY;
					}
				}

				selectFaceProbability = rand(randomNumberSeed);
				randomUV = Hammersley(testRayCount - i - 1, testRayCount);
				ray.direction = glm::vec3(targetNodeData_E.AABB.leftX, targetNodeData_E.AABB.leftY, targetNodeData_E.AABB.leftZ);
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
				if (triangleAttribute.materialType == 2 || triangleAttribute.materialType == 3) ++weight;
				else if (
					ray.hitPos.x >= targetNodeData_E.AABB.leftX &&
					ray.hitPos.y >= targetNodeData_E.AABB.leftY &&
					ray.hitPos.z >= targetNodeData_E.AABB.leftZ &&
					ray.hitPos.x <= targetNodeData_E.AABB.rightX &&
					ray.hitPos.y <= targetNodeData_E.AABB.rightY &&
					ray.hitPos.z <= targetNodeData_E.AABB.rightZ) {
					float cosine = 1.0f;
					if (targetNodeData_E.meanNormal != glm::vec3(0.0f)) {
						float doubleSide = min(glm::length(targetNodeData_E.meanNormal), 1.0f);
						cosine = dot(glm::normalize(targetNodeData_E.meanNormal), -ray.direction);
						if (cosine > 0) cosine *= doubleSide;
						else cosine = abs(cosine) * (1.0f - doubleSide);
					}
						
					//if (useG) cosine *= max(dot(normalizeNormal, ray.direction), 0.0f);
					weight += cosine;
				}
				//if (ray.depth >= r - 0.01f) {	//没打中三角形也算
				//	//float cosineNLWeight = 1.0f;
				//	//if (targetNodeData.meanNormal_E != glm::vec3(0.0f)) {
				//	//	float doubleSide = glm::length(targetNodeData.meanNormal_G);
				//	//	if (doubleSide > 0.5f) {
				//	//		glm::vec3 normalizeNormal_E = glm::normalize(targetNodeData.meanNormal_E);
				//	//		float cosine_NE_L = dot(normalizeNormal_E, -ray.direction);
				//	//		if (triangleAttribute.materialType == 2) cosine_NE_L = abs(cosine_NE_L);
				//	//		float rsq = max(r * r, 1.0f);
				//	//		cosineNLWeight = max(cosine_NE_L, 0.0f) / rsq;
				//	//	}
				//	//}
				//	//float cosineNVWeight = 1.0f;
				//	//if (nodeData.meanNormal_G != glm::vec3(0.0f)) {
				//	//	float doubleSide = glm::length(nodeData.meanNormal_G);
				//	//	if (doubleSide > 0.5f) {
				//	//		glm::vec3 normalizeNormal_G = glm::normalize(nodeData.meanNormal_G);
				//	//		cosineNVWeight = max(dot(normalizeNormal_G, ray.direction), 0.0f);
				//	//	}
				//	//}
				//	//weight += cosineNLWeight * cosineNVWeight;
				//	++weight;
				//}
			}
			weight *= glm::length(targetNodeData_E.irradiance) / testRayCount;
		}
	}

	uint32_t warpLane = threadIdx.x & 31;
	uint32_t firstBlockLane = (warpLane / 8) << 3;
	float blockWeightSum = weight;
	for (int offset = 4; offset > 0; offset /= 2)
		blockWeightSum += __shfl_down_sync(0xFFFFFFFF, blockWeightSum, offset);
	blockWeightSum = __shfl_sync(0xFFFFFFFF, blockWeightSum, firstBlockLane);

	__syncthreads();
	if (blockWeightSum > 0.0f) {
		weight += blockWeightSum * 0.0125f;
		blockWeightSum *= 1.1f;
		uint32_t weightIndex = indivisibleNodeIndex * SVONode_E_TotalCount + layerNodeOffset_E + targetNodeIndex_E;
		SVONodeWeights[weightIndex] = weight / blockWeightSum;

		if (warpLane == firstBlockLane && targetNodeLayer_E > 1) {
			uint32_t fatherNodeLabel = indivisibleNodeIndex * fatherLayerDivisibleNodeCount_E + targetNodeIndex_E / 8;
			SVOFatherDivisibleNodeBlockWeight[fatherNodeLabel] = blockWeightSum;
		}
	}
}

void FzbSVOCuda_PG::getSVONodesWeight() {
	uint32_t SVODivisibleNode_E_AccCount = 0;
	for (int i = 1; i < this->SVONodes_maxDepth - 1; ++i)	//叶层divisibleNodeCount = 0
		SVODivisibleNode_E_AccCount += this->SVOLayerInfos_E_host[i].divisibleNodeCount;
	uint32_t layerDivisibleNodeOffset_E = SVODivisibleNode_E_AccCount;
	uint32_t layerNodeOffset_E = SVONode_E_TotalCount_host;
	for (int i = this->SVONodes_maxDepth - 1; i > 0; --i) {
		if (SVOLayerInfos_E_host[i - 1].divisibleNodeCount == 0) continue;

		uint32_t layerNodeCount_E = SVOLayerInfos_E_host[i - 1].divisibleNodeCount * 8;		//这一层的node总数
		uint32_t blockSize = layerNodeCount_E;	//线程组大小为该层node总数
		uint32_t threadBlockCountForOneNode = (blockSize + 255) / 256;
		blockSize /= threadBlockCountForOneNode;
		while (((blockSize / threadBlockCountForOneNode) & 7) != 0) ++blockSize;
		uint32_t gridSize = SVOInDivisibleNode_G_TotalCount_host * threadBlockCountForOneNode;	//threadBlockCountForOneNode个线程组代表一个不可分node

		layerNodeOffset_E -= layerNodeCount_E;		//该层node的起始索引
		uint32_t layerDivisibleNodeCount = this->SVOLayerInfos_E_host[i].divisibleNodeCount;
		uint32_t fatherLayerDivisibleNodeCount = this->SVOLayerInfos_E_host[i - 1].divisibleNodeCount;

		getSVONodesWeight_device<<<gridSize, blockSize, 0, stream>>>
		(
			SVONodeWeights,
			SVOIndivisibleNodeInfos_G,
			layerNodeCount_E, SVONode_E_TotalCount_host,
			SVONodes_G_multiLayer_Array, SVONodes_E_multiLayer_Array,
			SVODivisibleNodeBlockWeight, SVOFatherDivisibleNodeBlockWeight,
			layerDivisibleNodeCount, fatherLayerDivisibleNodeCount,
			SVONodes_maxDepth, i, layerNodeOffset_E,
			sourceManager->vertices, sourceManager->bvhNodeArray, sourceManager->bvhTriangleInfoArray,
			threadBlockCountForOneNode
		);

		float* temp = SVODivisibleNodeBlockWeight;
		SVODivisibleNodeBlockWeight = SVOFatherDivisibleNodeBlockWeight;
		SVOFatherDivisibleNodeBlockWeight = temp;
	}
	checkKernelFunction();
}

/*
为每个不可分node创建weight数组
weight数组大小为有值node数量，即包括可分node
*/
void FzbSVOCuda_PG::initGetSVONodesWeightSource(bool allocate) {
	//每层node数组的指针
	if (allocate) {
		CHECK(cudaMalloc((void**)&this->SVONodes_G_multiLayer_Array, this->SVONodes_maxDepth * sizeof(FzbSVONodeData_PG_G*)));
		CHECK(cudaMemcpy(this->SVONodes_G_multiLayer_Array, SVONodes_multiLayer_G.data(), this->SVONodes_maxDepth * sizeof(FzbSVONodeData_PG_G*), cudaMemcpyHostToDevice));

		CHECK(cudaMalloc((void**)&this->SVONodes_E_multiLayer_Array, this->SVONodes_maxDepth * sizeof(FzbSVONodeData_PG_E*)));
		CHECK(cudaMemcpy(this->SVONodes_E_multiLayer_Array, SVONodes_multiLayer_E.data(), this->SVONodes_maxDepth * sizeof(FzbSVONodeData_PG_E*), cudaMemcpyHostToDevice));
	}

	//最终的weiht，每个不可分node对应每个SVONode(包括无值)之间的weight
	uint32_t maxIndivisibleNodeTotalCount_G = 0;
	uint32_t maxTotalNodeCount_E = 0;
	for (int i = 1; i < this->SVONodes_maxDepth; ++i) {
		maxIndivisibleNodeTotalCount_G += SVONodesMaxCount[i];
		maxTotalNodeCount_E += SVONodesMaxCount[i];
	}

	if (allocate) {
		CHECK(cudaMalloc((void**)&this->SVODivisibleNodeBlockWeight, maxIndivisibleNodeTotalCount_G * SVONodesMaxCount[SVONodesMaxCount.size() - 3] * sizeof(float)));
		CHECK(cudaMemset(SVODivisibleNodeBlockWeight, 0, maxIndivisibleNodeTotalCount_G * SVONodesMaxCount[SVONodesMaxCount.size() - 3] * sizeof(float)));

		CHECK(cudaMalloc((void**)&this->SVOFatherDivisibleNodeBlockWeight, maxIndivisibleNodeTotalCount_G * SVONodesMaxCount[SVONodesMaxCount.size() - 2] * sizeof(float)));
		CHECK(cudaMemset(SVOFatherDivisibleNodeBlockWeight, 0, maxIndivisibleNodeTotalCount_G * SVONodesMaxCount[SVONodesMaxCount.size() - 2] * sizeof(float)));

		CHECK(cudaMalloc((void**)&this->SVONodeWeights, maxIndivisibleNodeTotalCount_G * maxTotalNodeCount_E * sizeof(float)));
	}
	CHECK(cudaMemset(this->SVONodeWeights, 0, maxIndivisibleNodeTotalCount_G * maxTotalNodeCount_E * sizeof(float)));
}