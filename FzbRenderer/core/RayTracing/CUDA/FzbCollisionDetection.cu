#include "./FzbCollisionDetection.cuh"

__device__ bool AABBCollisionDetection(FzbAABB AABB, FzbRay ray) {
	//�жϹ����ǲ����ڳ����ڲ������ģ�����ǻ�����ֱ��������ĳ���
	//��Ȼ�ᵼ��ÿ�ζ�����������AABB�ټ��һ��hitMesh
	if (ray.startPos.x > AABB.leftX && ray.startPos.x < AABB.rightX &&
		ray.startPos.y > AABB.leftY && ray.startPos.y < AABB.rightY &&
		ray.startPos.z > AABB.leftZ && ray.startPos.z < AABB.rightZ) {
		return true;
	}

	float maxInTime = -FLT_MAX;
	float minOutTime = FLT_MAX;

	if (ray.direction.x != 0) {	//ֱ�����涼������
		float leftX = (AABB.leftX - ray.startPos.x) / ray.direction.x;
		float rightX = (AABB.rightX - ray.startPos.x) / ray.direction.x;
		maxInTime = max(min(leftX, rightX), maxInTime);
		minOutTime = min(max(leftX, rightX), minOutTime);
	}
	else if (ray.startPos.x < AABB.leftX || ray.startPos.x > AABB.rightX) return false;

	if (ray.direction.y != 0) {
		float leftY = (AABB.leftY - ray.startPos.y) / ray.direction.y;
		float rightY = (AABB.rightY - ray.startPos.y) / ray.direction.y;
		maxInTime = max(min(leftY, rightY), maxInTime);
		minOutTime = min(max(leftY, rightY), minOutTime);
	}
	else if (ray.startPos.y < AABB.leftY || ray.startPos.y > AABB.rightY) return false;

	if (ray.direction.z != 0) {
		float leftZ = (AABB.leftZ - ray.startPos.z) / ray.direction.z;
		float rightZ = (AABB.rightZ - ray.startPos.z) / ray.direction.z;
		maxInTime = max(min(leftZ, rightZ), maxInTime);
		minOutTime = min(max(leftZ, rightZ), minOutTime);
	}
	else if (ray.startPos.z < AABB.leftZ || ray.startPos.z > AABB.rightZ) return false;

	if (minOutTime < maxInTime) return false;
	if (maxInTime > ray.depth) return false;	//��Ȳ���
	return true;
}
__device__ bool meshCollisionDetection(FzbBvhNodeTriangleInfo& triangle, float* __restrict__ vertices, cudaTextureObject_t* __restrict__ materialTextures,
	FzbRay& ray, FzbTriangleAttribute& triangleAttribute) {
	//�ж��������Ƿ�ײ����
	getTriangleVertexAttribute(vertices, materialTextures, triangle, triangleAttribute);
	//ĿǰĬ�ϲ�����
	if (glm::dot(triangleAttribute.normal, -ray.direction)) return false;	//���˱��棬����ײ��
	glm::vec3 E1 = triangleAttribute.pos1 - triangleAttribute.pos0;
	glm::vec3 E2 = triangleAttribute.pos2 - triangleAttribute.pos0;
	glm::vec3 S = ray.startPos - triangleAttribute.pos0;
	glm::vec3 S1 = glm::cross(ray.direction, E2);
	glm::vec3 S2 = cross(S, E1);
	glm::vec3 tbb = 1 / glm::dot(S1, E1) * glm::vec3(glm::dot(S2, E2), glm::dot(S1, S), glm::dot(S2, ray.direction));
	if (tbb.x > 0 && (1.0f - tbb.y - tbb.z) > 0 && tbb.y > 0 && tbb.z > 0) {	//����
		if (tbb.x > ray.depth) return false;	//��Ȳ���ûͨ��
		ray.depth = tbb.x;
	}
	else return false;
	return true;
}
__device__ bool sceneCollisionDetection(FzbBvhNode* __restrict__ bvhNodeArray, FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray,
	float* __restrict__ vertices, cudaTextureObject_t* __restrict__ materialTextures,
	FzbRay& ray, FzbTriangleAttribute& triangleAttribute, bool notOnlyDetection) {
	uint32_t nodeIndices[BVH_MAX_DEPTH] = { 0 };
	int queueTail = 0;
	FzbBvhNode node;
	FzbBvhNodeTriangleInfo hitTriangle;
	bool result = false;
	while (queueTail > -1) {
		uint32_t nodeIndex = nodeIndices[queueTail];
		node = bvhNodeArray[nodeIndex];
		if (AABBCollisionDetection(node.AABB, ray)) {
			if (node.leftNodeIndex == 0) {	//���ײ��Ҷ�ڵ��ˣ������mesh��ײ���
				FzbBvhNodeTriangleInfo triangle = bvhTriangleInfoArray[node.rightNodeIndex];
				if (meshCollisionDetection(triangle, vertices, materialTextures, ray, triangleAttribute)) {
					hitTriangle = triangle;
					result = true;
				}
				--queueTail;
			}
			else {	//����Ҷ�ڵ���������
				nodeIndices[queueTail] = node.rightNodeIndex;
				nodeIndices[++queueTail] = node.leftNodeIndex;
			}
		}
		else --queueTail;
	}
	if (result && notOnlyDetection) getTriangleMaterialAttribute(materialTextures, hitTriangle.materialIndex, triangleAttribute);
	return result;
}