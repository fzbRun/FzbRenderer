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

	if (fabsf(ray.direction.x) > 1e-8f) {	//ֱ�����涼������
		float leftX = (AABB.leftX - ray.startPos.x) / ray.direction.x;
		float rightX = (AABB.rightX - ray.startPos.x) / ray.direction.x;
		maxInTime = max(min(leftX, rightX), maxInTime);
		minOutTime = min(max(leftX, rightX), minOutTime);
	}
	else if (ray.startPos.x < AABB.leftX || ray.startPos.x > AABB.rightX) return false;

	if (fabsf(ray.direction.y) > 1e-8f) {
		float leftY = (AABB.leftY - ray.startPos.y) / ray.direction.y;
		float rightY = (AABB.rightY - ray.startPos.y) / ray.direction.y;
		maxInTime = max(min(leftY, rightY), maxInTime);
		minOutTime = min(max(leftY, rightY), minOutTime);
	}
	else if (ray.startPos.y < AABB.leftY || ray.startPos.y > AABB.rightY) return false;

	if (fabsf(ray.direction.z) > 1e-8f) {
		float leftZ = (AABB.leftZ - ray.startPos.z) / ray.direction.z;
		float rightZ = (AABB.rightZ - ray.startPos.z) / ray.direction.z;
		maxInTime = max(min(leftZ, rightZ), maxInTime);
		minOutTime = min(max(leftZ, rightZ), minOutTime);
	}
	else if (ray.startPos.z < AABB.leftZ || ray.startPos.z > AABB.rightZ) return false;

	if (minOutTime < maxInTime || minOutTime < 0) return false;
	if (maxInTime > ray.depth) return false;	//��Ȳ���
	return true;
}
__device__ bool meshCollisionDetection(const float* __restrict__ vertices, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray, const cudaTextureObject_t* __restrict__ materialTextures,
	FzbRay& ray, FzbBvhNode node, FzbBvhNodeTriangleInfo& hitTriangle, FzbTriangleAttribute& triangleAttribute) {
	bool hit = false;
	for (int i = 0; i < node.triangleCount; ++i) {	//���ڻ��ֲ�����node��maxDepth��node�����ܰ������������
		FzbBvhNodeTriangleInfo triangle = bvhTriangleInfoArray[node.rightNodeIndex + i];
		getTriangleVertexAttribute(vertices, triangle, triangleAttribute);
		//if (glm::dot(triangleAttribute.normal, -ray.direction) <= 0) continue;	//���˱��棬����ײ��
		glm::vec3 E1 = triangleAttribute.pos1 - triangleAttribute.pos0;
		glm::vec3 E2 = triangleAttribute.pos2 - triangleAttribute.pos0;
		glm::vec3 S = ray.startPos - triangleAttribute.pos0;
		glm::vec3 S1 = glm::cross(ray.direction, E2);
		glm::vec3 S2 = glm::cross(S, E1);
		glm::vec3 tbb = 1 / glm::dot(S1, E1) * glm::vec3(glm::dot(S2, E2), glm::dot(S1, S), glm::dot(S2, ray.direction));
		if (tbb.x > 0 && (1.0f - tbb.y - tbb.z) > 0 && tbb.y > 0 && tbb.z > 0) {	//����
			if (tbb.x > ray.depth) continue;	//��Ȳ���ûͨ��
			ray.depth = tbb.x;
			hit = true;
			hitTriangle = triangle;
		}
	}
	return hit;
}
__device__ bool sceneCollisionDetection(const FzbBvhNode* __restrict__ bvhNodeArray, const FzbBvhNodeTriangleInfo* __restrict__ bvhTriangleInfoArray,
	const float* __restrict__ vertices, const cudaTextureObject_t* __restrict__ materialTextures,
	FzbRay& ray, FzbTriangleAttribute& triangleAttribute, bool notOnlyDetection) {
	volatile uint32_t nodeIndices[BVH_MAX_DEPTH];	//��ģ�����volatile��������Ż�IO�����²�֪��ʲô���󣬻���Ҷ�ڵ���˺�nodeIndex��ʱ���õ�0
	int stackTop = 0;
	FzbBvhNodeTriangleInfo bestHitTriangle;
	bool anyHit = false;
	nodeIndices[stackTop] = 0;
	while (stackTop > -1) {
		uint32_t nodeIndex = nodeIndices[stackTop];
		const FzbBvhNode& node = bvhNodeArray[nodeIndex];
		if (!AABBCollisionDetection(node.AABB, ray)) {
			--stackTop;
			continue;
		}
		if (node.leftNodeIndex == 0) {	//���ײ��Ҷ�ڵ��ˣ������mesh��ײ���
			FzbBvhNodeTriangleInfo hitTriangle;
			FzbTriangleAttribute hitTriangleAttribute;
			if (meshCollisionDetection(vertices, bvhTriangleInfoArray, materialTextures, ray, node, hitTriangle, hitTriangleAttribute)) {
				bestHitTriangle = hitTriangle;
				triangleAttribute = hitTriangleAttribute;
				anyHit = true;
			}
			--stackTop;;
		}
		else {	//����Ҷ�ڵ���������
			nodeIndices[stackTop] = node.rightNodeIndex;
			nodeIndices[++stackTop] = node.leftNodeIndex;
		}
	}
	if (anyHit && notOnlyDetection) getTriangleMaterialAttribute(vertices, materialTextures, bestHitTriangle, triangleAttribute);
	return anyHit;
}